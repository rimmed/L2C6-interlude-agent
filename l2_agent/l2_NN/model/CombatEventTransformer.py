import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: Tensor [B, L, K, F]
        mask: [B, L, K] (True for valid entities, False for padding)
        Returns: Tensor [B, L, embed_dim]
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
        return context


class CombatEventTransformer(nn.Module):
    """
    Combat Event Transformer for processing combat events.
    """
    def __init__(
            self,
            player_dim: int = 8,
            mob_features: int = 4,
            skill_features: int = 3,
            max_mobs: int = 10,
            max_skills: int = 16,
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
        self.skill_encoder = EntityEncoder(in_features=skill_features, embed_dim=32)

        # Fusion layer: зведення 64 + 64 + 32 = 160 -> d_model (128)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, d_model),
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
        self.head_pointer_idx = nn.Linear(d_model, max_mobs)
        self.head_skill_slot = nn.Linear(d_model, max_skills)

    def _generate_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Генерує причинно-наслідкову (Causal) маску, щоб t не дивився в майбутнє."""
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Витягуємо тензори з батчу
        player_state = batch["player_state"]  # [B, L, 8]
        delta_t = batch["delta_t"].unsqueeze(-1)  # [B, L, 1]
        mobs_matrix = batch["mobs_matrix"]   # [B, L, 10, 4]
        skills_matrix = batch["skills_matrix"] # [B, L, 16, 3]
        seq_mask = batch["seq_mask"]         # [B, L] (bool)

        B, L, _ = player_state.shape

        # --- 1. Embedding & Feature Fusion ---
        player_combined = torch.cat([player_state, delta_t], dim=-1) # [B, L, 9]
        emb_player = self.player_mlp(player_combined)                 # [B, L, 64]

        # Для мобів: 4-й прапорець is_padded (1 = padded, 0 = real) -> маска valid = is_padded == 0
        mob_valid_mask = (mobs_matrix[..., 3] == 0)                   # [B, L, 10]
        emb_mobs = self.mob_encoder(mobs_matrix, mask=mob_valid_mask)  # [B, L, 64]

        # Для скілів: перевіряємо, чи є хоча б одна ненульова фіча
        skill_valid_mask = (skills_matrix.sum(dim=-1) > 0)             # [B, L, 16]
        emb_skills = self.skill_encoder(skills_matrix, mask=skill_valid_mask) # [B, L, 32]

        # Конкатенація всіх ембеддингів у покроковий вектор s_t
        state_repr = torch.cat([emb_player, emb_mobs, emb_skills], dim=-1) # [B, L, 160]
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
        logits_pointer_idx = self.head_pointer_idx(last_step_repr) # [B, max_mobs]
        logits_skill_slot = self.head_skill_slot(last_step_repr)   # [B, max_skills]

        return {
            "logits_action_type": logits_action_type,
            "logits_pointer_idx": logits_pointer_idx,
            "logits_skill_slot": logits_skill_slot
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

