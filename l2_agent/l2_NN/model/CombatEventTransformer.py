import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

#: Logit of a masked (padding) entity in the pointer heads; finite and float16-safe.
MASKED_LOGIT = -1e4

class EntityEncoder(nn.Module):
    """
    Entity Encoder for Combat Event Transformer.
    MLP, Masked Attantion    
    """
    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attn_weights = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor [B, L, K, F]
        mask: [B, L, K] (True for valid entities, False for padding)
        Returns: (context [B, L, embed_dim], per-entity embeddings [B, L, K, embed_dim])
        """
        B, L, K, F_dim = x.shape
        # reshape for MLP
        flat_x = x.reshape(B * L * K, F_dim) # flat_x = x.view(-1, F_dim)
        h = self.mlp(flat_x).view(B, L, K, -1) # [B*L*K, embed_dim]

        scores = self.attn_weights(h).squeeze(-1) # [B, L, K]
        if mask is not None:
            # padding mask
            # finfo.min instead of -1e9: -1e9 overflows float16 under AMP autocast
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [B, L, K, 1]
        context = torch.sum(h * weights, dim=2) # [B, L, embed_dim]
        return context, h


class PointerHead(nn.Module):
    """
    Scores every entity slot of the last timestep against the sequence state (additive attention).

    A slot's score depends on *what is in the slot* (its embedding), not on its index, so the head
    works for any slot order and for entities never seen in training (e.g. skills of a new class).
    """
    def __init__(self, d_model: int, entity_dim: int, hidden: int = 64):
        super().__init__()
        self.query = nn.Linear(d_model, hidden)
        self.key = nn.Linear(entity_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        state: [B, d_model], entities: [B, K, entity_dim], valid: [B, K] (bool)
        Returns: logits [B, K], padding slots set to MASKED_LOGIT
        """
        hidden = torch.tanh(self.key(entities) + self.query(state).unsqueeze(1))  # [B, K, hidden]
        logits = self.score(hidden).squeeze(-1)  # [B, K]
        return logits.masked_fill(~valid, MASKED_LOGIT)


class CombatEventTransformer(nn.Module):
    """
    Combat Event Transformer for processing combat events.
    """
    def __init__(
            self,
            player_dim: int = 8,
            mob_features: int = 4,
            skill_features: int = 17,
            max_mobs: int = 10,
            max_skills: int = 64,
            num_action_types: int = 16,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1
        ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 1. Енкодери вхідних фіч
        self.player_mlp = nn.Sequential(
            nn.Linear(player_dim + 1, 32), # +1 для delta_t
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.mob_encoder = EntityEncoder(in_features=mob_features, embed_dim=64)
        self.skill_encoder = EntityEncoder(in_features=skill_features, embed_dim=64)

        # Fusion layer: зведення 64 + 64 + 64 = 192 -> d_model (128)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 2. Causal Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Multi-Discrete Output Heads
        self.head_action_type = nn.Linear(d_model, num_action_types)
        # Pointer heads: the target monster and the skill are chosen by their features, not by slot index.
        self.head_pointer_idx = PointerHead(d_model, entity_dim=64)
        self.head_skill_slot = PointerHead(d_model, entity_dim=64)
        # State value for RL (V-trace); not part of the ONNX policy the actors run.
        self.head_value = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def _generate_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Генерує причинно-наслідкову (Causal) маску, щоб t не дивився в майбутнє."""
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Витягуємо тензори з батчу
        player_state = batch["player_state"]  # [B, L, 8]
        delta_t = batch["delta_t"].unsqueeze(-1)  # [B, L, 1]
        mobs_matrix = batch["mobs_matrix"]   # [B, L, K, 4]
        skills_matrix = batch["skills_matrix"] # [B, L, N, F_skill]
        seq_mask = batch["seq_mask"]         # [B, L] (bool)

        B, L, _ = player_state.shape

        # --- 1. Embedding & Feature Fusion ---
        player_combined = torch.cat([player_state, delta_t], dim=-1) # [B, L, 9]
        emb_player = self.player_mlp(player_combined)                 # [B, L, 64]

        # Для мобів: 4-й прапорець is_padded (1 = padded, 0 = real) -> маска valid = is_padded == 0
        mob_valid_mask = (mobs_matrix[..., 3] == 0)                   # [B, L, K]
        emb_mobs, mob_entities = self.mob_encoder(mobs_matrix, mask=mob_valid_mask)  # [B, L, 64], [B, L, K, 64]

        # Для скілів: порожній слот - повністю нульовий рядок (is_valid = 0)
        skill_valid_mask = (skills_matrix.sum(dim=-1) > 0)             # [B, L, N]
        emb_skills, skill_entities = self.skill_encoder(skills_matrix, mask=skill_valid_mask)  # [B, L, 64], [B, L, N, 64]

        # Конкатенація всіх ембеддингів у покроковий вектор s_t
        state_repr = torch.cat([emb_player, emb_mobs, emb_skills], dim=-1) # [B, L, 192]
        x = self.fusion(state_repr) # [B, L, d_model]

        # --- 2. Temporal Processing (Transformer) ---
        # Causal + padding mask combined into one per-sample [B * nhead, L, L] mask (True = blocked).
        # With left padding and a causal mask, a padded query only sees padded keys; a fully blocked
        # row turns into NaN in eval mode (fast path), which then leaks into the real timesteps.
        # Letting every position attend to itself keeps each row non-empty; real queries never see
        # padded keys, so the result for the real timesteps is unchanged.
        causal_mask = self._generate_causal_mask(L, device=x.device)
        blocked = causal_mask.unsqueeze(0) | (~seq_mask).unsqueeze(1)  # [B, L, L]
        blocked = blocked & ~torch.eye(L, dtype=torch.bool, device=x.device)
        attn_mask = blocked.repeat_interleave(self.nhead, dim=0)  # [B * nhead, L, L]

        out = self.transformer(src=x, mask=attn_mask) # [B, L, d_model]

        # Оскільки використовується Left-Padding, останній кроковий вектор (t = L - 1 = 15)
        # гарантовано містить інтегрований стан останнього дійсного кадру.
        last_step_repr = out[:, -1, :] # [B, d_model]

        # --- 3. Action Heads ---
        logits_action_type = self.head_action_type(last_step_repr) # [B, num_action_types]
        logits_pointer_idx = self.head_pointer_idx(last_step_repr, mob_entities[:, -1], mob_valid_mask[:, -1])  # [B, K]
        logits_skill_slot = self.head_skill_slot(last_step_repr, skill_entities[:, -1], skill_valid_mask[:, -1])  # [B, N]

        return {
            "logits_action_type": logits_action_type,
            "logits_pointer_idx": logits_pointer_idx,
            "logits_skill_slot": logits_skill_slot,
            "value": self.head_value(last_step_repr).squeeze(-1),  # [B]
        }

    def compute_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Обчислює сумарний Cross-Entropy Loss для Behavioral Cloning."""
        loss_action = F.cross_entropy(logits_dict["logits_action_type"], targets["target_action_type"])
        loss_pointer = F.cross_entropy(logits_dict["logits_pointer_idx"], targets["target_pointer_idx"])
        loss_skill = F.cross_entropy(logits_dict["logits_skill_slot"], targets["target_skill_slot"])

        # Зважена сума (можна налаштовувати вагові коефіцієнти)
        total_loss = loss_action + 0.5 * loss_pointer + 0.5 * loss_skill

        metrics = {
            "loss_total": total_loss.item(),
            "loss_action": loss_action.item(),
            "loss_pointer": loss_pointer.item(),
            "loss_skill": loss_skill.item(),
        }
        return total_loss, metrics

