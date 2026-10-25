"""
Distributed RL trainer for :class:`CombatEventTransformer` (V-trace actor-critic, IMPALA style).

Many actors (``python -m l2_actor --redis ... --model redis``) play with the latest published policy,
sampling their actions, and push trajectory segments to Redis (format in ``l2_actor/replay.py``).
This process:

1. pops the segments from Redis in a background thread into a replay buffer of the newest
   ``--buffer-steps`` transitions (segments older than ``--max-staleness`` policy versions are dropped);
2. trains on batches of whole segments with the V-trace loss: the actors' policy lags behind the
   learner, V-trace corrects that with clipped importance weights between the behaviour
   log-probability the actor stored and the learner's own;
3. adds a behaviour-cloning term on the human demonstrations (the parquet dataset of ``train.py``)
   whose weight decays from ``--bc-start`` to ``--bc-end`` over ``--bc-decay`` updates, so the policy
   does not forget how humans fight before the rewards teach it something better;
4. publishes the policy to Redis (ONNX + metadata, a new version) every ``--publish-interval`` seconds,
   exported in a child process so learning does not stop for the export;
5. logs to ``<save-dir>/metrics.jsonl`` and keeps ``latest.pth`` / ``best.pth`` (best = highest reward
   per 100 steps over the latest segments; an episode is one life, so lives vary a lot in length). The checkpoints have the layout of
   ``train.py``'s, so ``export_onnx.py`` exports them as well.

The probability of one decision is a product over the heads (``l2_client.policy.CombatPolicy.act``):
the action type among the executable ones, the pointed monster for actions that use it and the
skill slot for ``used_skill`` - the trainer recomputes exactly that from the stored masks.

The model runs in eval mode (no dropout) for both the actors and the learner, otherwise the
importance weights would compare two different policies.

Usage (from ``l2_agent/``)::

    # start from the behaviour-cloning model; actors started with ``--model redis`` get it first
    uv run l2_NN/train_rl.py --init checkpoints/best_model.pth --redis redis://localhost:6379/0

    # continue an interrupted run
    uv run l2_NN/train_rl.py --resume checkpoints/rl/latest.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

# Every batch has another number of states (segments are 1..64 steps long); without expandable
# segments the CUDA caching allocator fragments until an allocation fails.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_actor import replay  # noqa: E402 - PROJECT_ROOT is added just above
from l2_client.policy import POINTER_ACTIONS  # noqa: E402
from l2_NN.dataloader import dataloder as dl  # noqa: E402
from l2_NN.dataloader.augment import AugmentConfig  # noqa: E402
from l2_NN.dataloader.dataloder import SKILL_ACTION_IDS, Lineage2EventDataset  # noqa: E402
from l2_NN.export_onnx import export_policy, load_policy_weights  # noqa: E402
from l2_NN.model.CombatEventTransformer import CombatEventTransformer  # noqa: E402
from l2_NN.train import DEFAULT_DATA_PATH, INPUT_KEYS, AugmentedWindows, compute_losses, prepare_batch  # noqa: E402

#: Logit of an impossible choice (fp32; the model's own padding mask uses -1e4).
IMPOSSIBLE: Final[float] = -1e9


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V-trace RL trainer fed by actors through Redis.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", type=Path, help="Behaviour-cloning checkpoint to start from (train.py)")
    start.add_argument("--resume", type=Path, help="RL checkpoint to continue (latest.pth of an earlier run)")
    p.add_argument("--redis", default="redis://localhost:6379/0", help="Redis URL shared with the actors")
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints/rl"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=("bf16", "off"), default="off",
                   help="Autocast of the forward pass (cuda); bf16 logits add noise to the importance weights")
    p.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("replay")
    g.add_argument("--buffer-steps", type=int, default=50_000, help="Replay buffer capacity in transitions")
    g.add_argument("--min-buffer-steps", type=int, default=2_000, help="Start learning once this many arrived")
    g.add_argument("--max-staleness", type=int, default=20,
                   help="Drop segments played with a policy more than this many versions old")
    g.add_argument("--max-replay-ratio", type=float, default=8.0,
                   help="Pause learning while trained transitions / received transitions exceeds this")

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch-segments", type=int, default=32, help="Segments per batch (up to 64 steps each)")
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--max-updates", type=int, default=0, help="Stop after N updates (0 = run until Ctrl+C)")
    g.add_argument("--value-warmup", type=int, default=200,
                   help="First N updates train only the (new) value head, the policy stays untouched")

    g = p.add_argument_group("V-trace")
    g.add_argument("--gamma", type=float, default=0.99, help="Discount per decision (decisions are ~0.5 s apart)")
    g.add_argument("--lam", type=float, default=1.0, help="Trace-cutting lambda (1 = plain V-trace)")
    g.add_argument("--rho-bar", type=float, default=1.0, help="Importance-weight clip of the value targets")
    g.add_argument("--c-bar", type=float, default=1.0, help="Importance-weight clip of the traces")
    g.add_argument("--pg-rho-bar", type=float, default=1.0, help="Importance-weight clip of the policy gradient")
    g.add_argument("--value-coef", type=float, default=0.5)
    g.add_argument("--entropy-coef", type=float, default=0.01)

    g = p.add_argument_group("demonstrations (behaviour cloning term)")
    g.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Parquet dataset of train.py")
    g.add_argument("--demo-batch", type=int, default=256,
                   help="Demonstration windows per update (~10%% of a 32x64 batch); 0 disables the term")
    g.add_argument("--bc-start", type=float, default=1.0, help="BC weight at the first update")
    g.add_argument("--bc-end", type=float, default=0.05, help="BC weight after --bc-decay updates")
    g.add_argument("--bc-decay", type=int, default=20_000, help="Updates over which the BC weight decays")
    g.add_argument("--num-workers", type=int, default=4, help="DataLoader workers of the demonstrations")

    g = p.add_argument_group("publishing, logging")
    g.add_argument("--publish-interval", type=float, default=30.0, help="Publish the policy every N seconds")
    g.add_argument("--save-every", type=int, default=500, help="Write latest.pth every N updates")
    g.add_argument("--log-every", type=int, default=20)
    g.add_argument("--return-window", type=int, default=100, help="Lives in the mean return per life (logged)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------


class ReplayBuffer:
    """The newest ``capacity`` transitions, kept as whole segments (V-trace needs the sequences)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.segments: deque[replay.Segment] = deque()
        self.steps = 0

    def add(self, segment: replay.Segment) -> None:
        self.segments.append(segment)
        self.steps += segment.steps
        while self.steps > self.capacity and len(self.segments) > 1:
            self.steps -= self.segments.popleft().steps

    def drop_stale(self, current_version: int, max_staleness: int) -> int:
        """Remove segments played with a policy too old to correct; returns how many."""
        def fresh(segment: replay.Segment) -> bool:
            version = segment.header.get("model_version")
            return version is None or current_version - int(version) <= max_staleness

        kept = [s for s in self.segments if fresh(s)]
        dropped = len(self.segments) - len(kept)
        if dropped:
            self.segments = deque(kept)
            self.steps = sum(s.steps for s in kept)
        return dropped

    def sample(self, count: int, rng: random.Random) -> list[replay.Segment]:
        return rng.choices(self.segments, k=count)


class Ingest(threading.Thread):
    """Pops segments from Redis and decodes them off the training thread."""

    def __init__(self, url: str, out: queue.Queue, stop: threading.Event):
        super().__init__(name="redis-ingest", daemon=True)
        self.client = replay.connect(url)
        self.out = out
        self.stop = stop
        self.bad = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                blobs = replay.pop_segments(self.client, 64)
            except Exception as exc:  # Redis restart, network hiccup: retry
                print(f"[ingest] Redis error: {exc}", flush=True)
                self.stop.wait(2.0)
                continue
            if not blobs:
                self.stop.wait(0.2)
                continue
            for blob in blobs:
                try:
                    self.out.put(replay.decode_segment(blob))
                except Exception as exc:  # one malformed segment must not stop the trainer
                    self.bad += 1
                    print(f"[ingest] dropped a malformed segment: {exc}", flush=True)


# --------------------------------------------------------------------------------------
# Batches and losses
# --------------------------------------------------------------------------------------


def collate_segments(segments: list[replay.Segment], device: torch.device) -> dict[str, torch.Tensor]:
    """
    Flatten the states of ``segments`` (each one is an independent model input) and pad the
    per-step arrays to ``[B, T]``.

    ``step_state`` / ``boot_state`` index the flat states (``boot_state`` is -1 for a terminal segment).
    """
    batch_size, t_max = len(segments), max(s.steps for s in segments)
    states = {key: [] for key in replay.STATE_KEYS}
    step_state = np.full((batch_size, t_max), 0, dtype=np.int64)
    boot_state = np.full(batch_size, -1, dtype=np.int64)
    valid = np.zeros((batch_size, t_max), dtype=bool)
    per_step = {
        "action": np.zeros((batch_size, t_max), np.int64),
        "pointer": np.full((batch_size, t_max), -1, np.int64),
        "skill": np.full((batch_size, t_max), -1, np.int64),
        "logp": np.zeros((batch_size, t_max), np.float32),
        "reward": np.zeros((batch_size, t_max), np.float32),
        "n_mobs": np.zeros((batch_size, t_max), np.int64),
    }
    first = segments[0].arrays
    type_mask = np.zeros((batch_size, t_max, first["type_mask"].shape[1]), dtype=bool)
    type_mask[..., 0] = True  # padded steps: any non-empty mask keeps the softmax finite
    skill_mask = np.zeros((batch_size, t_max, first["skill_mask"].shape[1]), dtype=bool)

    offset = 0
    for i, segment in enumerate(segments):
        a, steps = segment.arrays, segment.steps
        for key in replay.STATE_KEYS:
            states[key].append(a[key])
        step_state[i, :steps] = offset + np.arange(steps)
        if not segment.terminal:
            boot_state[i] = offset + steps
        offset += a["player_state"].shape[0]
        valid[i, :steps] = True
        for key in per_step:
            per_step[key][i, :steps] = a[key]
        type_mask[i, :steps] = a["type_mask"]
        skill_mask[i, :steps] = a["skill_mask"]

    batch = {key: torch.from_numpy(np.concatenate(parts)).to(device, non_blocking=True)
             for key, parts in states.items()}
    for key in ("player_state", "delta_t", "mobs_matrix", "skills_matrix"):
        batch[key] = batch[key].float()
    batch.update({key: torch.from_numpy(value).to(device) for key, value in per_step.items()})
    batch.update(
        step_state=torch.from_numpy(step_state).to(device), boot_state=torch.from_numpy(boot_state).to(device),
        valid=torch.from_numpy(valid).to(device), type_mask=torch.from_numpy(type_mask).to(device),
        skill_mask=torch.from_numpy(skill_mask).to(device),
    )
    return batch


@torch.no_grad()
def vtrace(behaviour_logp: torch.Tensor, target_logp: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor,
           next_values: torch.Tensor, discounts: torch.Tensor, valid: torch.Tensor, *, rho_bar: float = 1.0,
           c_bar: float = 1.0, pg_rho_bar: float = 1.0, lam: float = 1.0) -> dict[str, torch.Tensor]:
    """
    V-trace targets (Espeholt et al., 2018) for ``[B, T]`` segments.

    ``next_values[:, t]`` is ``V(x_{t+1})`` (the bootstrap value at the last step of a segment) and
    ``discounts`` is ``gamma`` except 0 after a terminal step; padded steps (``valid`` False) must
    come after the real ones. Returns the value targets ``vs`` and the policy-gradient advantages.
    """
    rho = torch.exp(target_logp - behaviour_logp)
    clipped_rho = torch.clamp(rho, max=rho_bar) * valid
    cs = lam * torch.clamp(rho, max=c_bar) * valid
    deltas = clipped_rho * (rewards + discounts * next_values - values)

    vs_minus_v = torch.zeros_like(values)
    acc = torch.zeros_like(values[:, 0])
    for t in reversed(range(values.shape[1])):
        acc = deltas[:, t] + discounts[:, t] * cs[:, t] * acc
        vs_minus_v[:, t] = acc
    vs = values + vs_minus_v
    # vs_{t+1}: the next step's target inside a segment, the bootstrap value after its last step
    # (padded entries of vs_minus_v are 0, so the shift is right at the segment end).
    next_vs = next_values + torch.cat([vs_minus_v[:, 1:], torch.zeros_like(vs_minus_v[:, :1])], dim=1)
    pg_adv = torch.clamp(rho, max=pg_rho_bar) * (rewards + discounts * next_vs - values) * valid
    return {"vs": vs, "pg_adv": pg_adv, "rho": rho}


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~mask, IMPOSSIBLE).log_softmax(dim=-1)


def entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).masked_fill(~mask, 0.0).sum(dim=-1)


#: Actions whose stored decision includes a skill slot (the character's skill or its summon's).
SKILL_ACTIONS = torch.tensor(SKILL_ACTION_IDS)


def policy_terms(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], pointer_action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(logp, entropy)`` of the stored decisions under the learner, both ``[B, T]``."""
    idx = batch["step_state"]
    logits_type = out["logits_action_type"][idx]  # [B, T, A]
    logits_ptr = out["logits_pointer_idx"][idx]  # [B, T, K]
    logits_skill = out["logits_skill_slot"][idx]  # [B, T, N]

    action, pointer, skill = batch["action"], batch["pointer"], batch["skill"]
    logp_type = masked_log_softmax(logits_type, batch["type_mask"])
    slots = torch.arange(logits_ptr.shape[-1], device=logits_ptr.device)
    ptr_mask = slots < batch["n_mobs"].unsqueeze(-1)
    logp_ptr = masked_log_softmax(logits_ptr, ptr_mask)
    logp_skill = masked_log_softmax(logits_skill, batch["skill_mask"])

    use_ptr = pointer_action[action] & (pointer >= 0)
    use_skill = torch.isin(action, SKILL_ACTIONS.to(action.device)) & (skill >= 0)
    logp = logp_type.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    logp = logp + torch.where(use_ptr, logp_ptr.gather(-1, pointer.clamp(min=0).unsqueeze(-1)).squeeze(-1), 0.0)
    logp = logp + torch.where(use_skill, logp_skill.gather(-1, skill.clamp(min=0).unsqueeze(-1)).squeeze(-1), 0.0)

    ent = entropy(logp_type, batch["type_mask"])
    ent = ent + torch.where(ptr_mask.any(-1), entropy(logp_ptr, ptr_mask), 0.0)
    ent = ent + torch.where(batch["skill_mask"].any(-1), entropy(logp_skill, batch["skill_mask"]), 0.0)
    return logp, ent


def rl_loss(model: CombatEventTransformer, batch: dict[str, torch.Tensor], pointer_action: torch.Tensor,
            args: argparse.Namespace, amp_dtype: torch.dtype | None) -> tuple[torch.Tensor, torch.Tensor,
                                                                              dict[str, float]]:
    """``(policy_loss, value_loss, stats)``; the policy loss includes the entropy bonus."""
    inputs = {key: batch[key] for key in INPUT_KEYS}
    device_type = batch["valid"].device.type
    with torch.autocast(device_type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        out = model(inputs)
    flat_values = out["value"].float()

    valid = batch["valid"]
    values = flat_values[batch["step_state"]]  # [B, T]
    boot = torch.where(batch["boot_state"] >= 0, flat_values[batch["boot_state"].clamp(min=0)], 0.0)
    steps = valid.sum(dim=1)
    last = torch.zeros_like(valid)
    last[torch.arange(valid.shape[0], device=valid.device), steps - 1] = True
    terminal = batch["boot_state"] < 0
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1).detach()
    next_values = torch.where(last, boot.detach().unsqueeze(1), next_values)
    discounts = torch.full_like(values, args.gamma)
    discounts = torch.where(last & terminal.unsqueeze(1), 0.0, discounts)

    logp, ent = policy_terms(out, batch, pointer_action)
    targets = vtrace(batch["logp"], logp.detach(), batch["reward"], values.detach(), next_values, discounts, valid,
                     rho_bar=args.rho_bar, c_bar=args.c_bar, pg_rho_bar=args.pg_rho_bar, lam=args.lam)
    count = valid.sum().clamp(min=1)
    pg_loss = -(targets["pg_adv"] * logp * valid).sum() / count
    ent_mean = (ent * valid).sum() / count
    value_loss = 0.5 * (((targets["vs"] - values) ** 2) * valid).sum() / count

    rho = targets["rho"][valid]
    stats = {
        "pg_loss": pg_loss.detach().item(), "value_loss": value_loss.detach().item(),
        "entropy": ent_mean.detach().item(),
        # 1% margin: the learner's fp32 logits differ from the actor's ONNX ones by float noise
        # rho_clipped: share of steps whose value-target weight is actually clipped; rho_far: share whose
        # policy probability moved by more than ~20 % since the actor played it (the real lag signal)
        "rho_mean": float(rho.mean()), "rho_clipped": float((rho > args.rho_bar).float().mean()),
        "rho_far": float((rho.log().abs() > 0.2).float().mean()),
        "kl_approx": float((batch["logp"] - logp.detach())[valid].mean()),
        "value_mean": float(values.detach()[valid].mean()), "vs_mean": float(targets["vs"][valid].mean()),
        "steps": int(count),
    }
    return pg_loss - args.entropy_coef * ent_mean, value_loss, stats


def bc_weight(update: int, args: argparse.Namespace) -> float:
    progress = min(1.0, update / max(1, args.bc_decay))
    return args.bc_start + (args.bc_end - args.bc_start) * progress


def demo_batches(args: argparse.Namespace, init_args: dict, action_to_id: dict) -> Iterator[dict] | None:
    """Endless augmented windows of the human demonstrations (``None`` when ``--demo-batch 0``)."""
    if args.demo_batch <= 0:
        return None
    print(f"Loading demonstrations from {args.data_path}")
    dataset = Lineage2EventDataset(pd.read_parquet(args.data_path), seq_len=init_args.get("seq_len", 16),
                                   max_mobs=init_args.get("max_mobs", 10), max_skills=init_args.get("max_skills", 64))
    if dataset.action_to_id != action_to_id:
        raise SystemExit("the demonstrations use another action label space than the model")
    windows = AugmentedWindows(dataset, list(range(len(dataset))), AugmentConfig(), args.seed)
    loader = DataLoader(windows, batch_size=args.demo_batch, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=args.device.startswith("cuda"))
    print(f"Demonstrations: {len(dataset)} windows")

    def endless() -> Iterator[dict]:
        while True:
            yield from loader

    return endless()


# --------------------------------------------------------------------------------------
# Training run
# --------------------------------------------------------------------------------------


class Trainer:
    """One learner: model, optimiser, replay buffer, Redis ingest and publishing."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.amp_dtype = torch.bfloat16 if args.amp == "bf16" and self.device.type == "cuda" else None
        self.rng = random.Random(args.seed)
        torch.manual_seed(args.seed)

        source = args.resume or args.init
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        self.model_config: dict = checkpoint["model_config"]
        self.action_to_id: dict = checkpoint["action_to_id"]
        #: the BC run's arguments (seq_len, keep_last_target_flag): the export metadata needs them
        self.init_args: dict = checkpoint["args"]
        self.model = CombatEventTransformer(**self.model_config).to(self.device)
        load_policy_weights(self.model, checkpoint["model_state_dict"])
        self.model.eval()  # no dropout: the learner must score actions like the actors' ONNX policy
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.value_params = {id(p) for p in self.model.head_value.parameters()}

        self.update = 0
        self.best_return = -math.inf
        self.returns: deque[float] = deque(maxlen=args.return_window)
        self.totals = {"segments": 0, "steps": 0, "episodes": 0, "kills": 0, "deaths": 0, "trained_steps": 0,
                       "stale_dropped": 0}
        if args.resume:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            rl = checkpoint["rl_state"]
            self.update, self.best_return = rl["update"], rl["best_return"]
            self.returns.extend(rl["returns"])
            self.totals.update(rl["totals"])
            print(f"Resumed {source} at update {self.update}")
        else:
            print(f"Initialised from {source} (BC epoch {checkpoint.get('epoch')})")

        # A model trained before summons were supported has no pet_attack class.
        pointer_ids = [self.action_to_id[name] for name in POINTER_ACTIONS if name in self.action_to_id]
        self.pointer_action = torch.zeros(len(self.action_to_id), dtype=torch.bool, device=self.device)
        self.pointer_action[pointer_ids] = True

        args.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = args.save_dir / "metrics.jsonl"
        self.redis = replay.connect(args.redis)
        self.version = int(self.redis.get(replay.MODEL_VERSION_KEY) or 0)
        self.buffer = ReplayBuffer(args.buffer_steps)
        self.incoming: queue.Queue = queue.Queue(maxsize=4096)
        self.stop = threading.Event()
        self.ingest = Ingest(args.redis, self.incoming, self.stop)
        self._episode_returns: dict[tuple[str, int], float] = {}
        #: (reward sum, steps) of the latest segments: reward per step does not depend on how long lives are
        self._reward_window: deque[tuple[float, int]] = deque(maxlen=2000)
        self._publisher: subprocess.Popen | None = None
        self.demos = demo_batches(args, self.init_args, self.action_to_id)
        # Demonstrations come in today's feature layouts; an older model may know fewer features.
        saved = checkpoint.get("features") or {}
        self.features = {
            "player": list(saved.get("player") or dl.player_feature_names(self.model_config["player_dim"])),
            "mob": list(saved.get("mob") or dl.mob_feature_names(self.model_config["mob_features"])),
            "skill": list(saved.get("skill") or dl.skill_feature_names(self.model_config["skill_features"])),
            # the class panels the model was trained on (None: the per-character panels of older models)
            "skill_panels": saved.get("skill_panels"),
        }
        self.demo_columns = {
            key: torch.tensor(dl.feature_columns(self.features[name], current))
            for key, name, current in (("player_state", "player", dl.PLAYER_FEATURE_NAMES),
                                       ("mobs_matrix", "mob", dl.MOB_FEATURE_NAMES),
                                       ("skills_matrix", "skill", dl.SKILL_FEATURE_NAMES))
        }

    # -- data ----------------------------------------------------------------------------

    def drain(self) -> None:
        while True:
            try:
                segment = self.incoming.get_nowait()
            except queue.Empty:
                return
            self.buffer.add(segment)
            self.totals["segments"] += 1
            self.totals["steps"] += segment.steps
            self._track_episode(segment)

    def _track_episode(self, segment: replay.Segment) -> None:
        """An episode is one life of a character (kills are rewards inside it, a death ends it)."""
        h = segment.header
        rewards = segment.arrays["reward"]
        self._reward_window.append((float(rewards.sum()), int(rewards.shape[0])))
        self.totals["kills"] += int(h.get("kills", 0))
        key = (str(h.get("actor_id")), int(h.get("episode_id", -1)))
        total = self._episode_returns.pop(key, 0.0) + float(rewards.sum())
        reason = h.get("done_reason")
        if reason is None:
            self._episode_returns[key] = total
            return
        self.totals["episodes"] += 1
        if reason == "death":
            self.totals["deaths"] += 1
        self.returns.append(total)

    def reward_per_100_steps(self) -> float:
        reward = sum(r for r, _ in self._reward_window)
        steps = sum(n for _, n in self._reward_window)
        return 100.0 * reward / steps if steps else math.nan

    # -- learning ------------------------------------------------------------------------

    def step(self) -> dict[str, float]:
        args = self.args
        segments = self.buffer.sample(args.batch_segments, self.rng)
        batch = collate_segments(segments, self.device)
        policy_loss, value_loss, stats = rl_loss(self.model, batch, self.pointer_action, args, self.amp_dtype)

        warmup = self.update < args.value_warmup
        loss = args.value_coef * value_loss
        if not warmup:
            loss = loss + policy_loss
            if self.demos is not None:
                raw = next(self.demos)
                for key, columns in self.demo_columns.items():
                    raw[key] = raw[key].index_select(-1, columns)
                demo, masks = prepare_batch(raw, self.device, not self.init_args.get("keep_last_target_flag", False))
                with torch.autocast(self.device.type, dtype=self.amp_dtype or torch.float32,
                                    enabled=self.amp_dtype is not None):
                    logits = self.model({key: demo[key] for key in INPUT_KEYS})
                bc, _ = compute_losses(logits, demo, masks, {"action": 1.0, "pointer": 1.0, "skill": 1.0})
                weight = bc_weight(self.update - args.value_warmup, args)
                loss = loss + weight * bc
                stats.update(bc_loss=bc.detach().item(), bc_weight=weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {float(loss)} at update {self.update}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if warmup:  # only the value head learns until its estimates make sense
            for p in self.model.parameters():
                if id(p) not in self.value_params:
                    p.grad = None
        stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip))
        self.optimizer.step()
        self.update += 1
        self.totals["trained_steps"] += stats["steps"]
        stats["warmup"] = warmup
        return stats

    # -- publishing and checkpoints ------------------------------------------------------

    def _checkpoint_dict(self) -> dict[str, Any]:
        return {
            "epoch": f"rl-{self.update}",
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "action_to_id": self.action_to_id,
            "features": self.features,
            "args": self.init_args,
            "val_metrics": None,
            "split": None,
            "rl_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
            "rl_state": {"update": self.update, "best_return": self.best_return, "returns": list(self.returns),
                         "totals": dict(self.totals), "model_version": self.version},
        }

    def publish(self, background: bool = False) -> bool:
        """
        Export the current weights to ONNX and publish them.

        ``background`` exports in a child process (``--publish-worker``): the ONNX exporter (torch
        dynamo) uses global torch state, so exporting in a thread of the training process breaks
        training. Returns ``False`` when an earlier export is still running (nothing is started).
        """
        if self._publisher is not None and self._publisher.poll() is None:
            return False
        folder = self.args.save_dir / "publish"
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = {**self._checkpoint_dict(), "optimizer_state_dict": None,
                    "model_state_dict": {k: v.detach().float().cpu() for k, v in self.model.state_dict().items()}}
        if not background:
            publish_snapshot(snapshot, self.args.redis, folder / "combat_policy.onnx")
            self.version = int(self.redis.get(replay.MODEL_VERSION_KEY) or 0)
            return True
        path = folder / "pending.pth"
        torch.save(snapshot, path)
        self._publisher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--publish-worker", str(path),
                                            self.args.redis, str(folder / "combat_policy.onnx")])
        return True

    def poll_publisher(self) -> None:
        """Pick up the version of a finished background publish."""
        if self._publisher is not None and self._publisher.poll() is not None:
            if self._publisher.returncode != 0:
                print(f"Publishing failed (exit code {self._publisher.returncode})", flush=True)
            self._publisher = None
            self.version = int(self.redis.get(replay.MODEL_VERSION_KEY) or 0)

    def save(self, name: str) -> None:
        path = self.args.save_dir / name
        torch.save(self._checkpoint_dict(), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else math.nan

    # -- main loop -----------------------------------------------------------------------

    def run(self) -> None:
        args = self.args
        self.publish()  # actors started with --model redis wait for this one
        self.ingest.start()
        print(f"Waiting for {args.min_buffer_steps} transitions from the actors...", flush=True)
        window: list[dict[str, float]] = []
        window_started = time.perf_counter()
        last_publish = time.time()
        try:
            while not args.max_updates or self.update < args.max_updates:
                self.drain()
                self.poll_publisher()
                self.totals["stale_dropped"] += self.buffer.drop_stale(self.version, args.max_staleness)
                ratio = self.totals["trained_steps"] / max(1, self.totals["steps"])
                if self.buffer.steps < args.min_buffer_steps or ratio > args.max_replay_ratio:
                    time.sleep(0.1)
                    continue
                window.append(self.step())

                if self.update % args.log_every == 0:
                    self.log(window, time.perf_counter() - window_started)
                    window, window_started = [], time.perf_counter()
                if time.time() - last_publish >= args.publish_interval and self.publish(background=True):
                    last_publish = time.time()
                if self.update % args.save_every == 0:
                    self.save("latest.pth")
                    score = self.reward_per_100_steps()
                    if len(self._reward_window) >= 200 and score > self.best_return:
                        self.best_return = score
                        self.save("best.pth")
                        print(f"New best reward per 100 steps {score:+.3f} -> best.pth", flush=True)
        except KeyboardInterrupt:
            print("Interrupted", flush=True)
        finally:
            self.stop.set()
            if self.update:
                self.save("latest.pth")
                print(f"Saved {args.save_dir / 'latest.pth'} at update {self.update}", flush=True)

    def log(self, window: list[dict[str, float]], seconds: float) -> None:
        keys = [k for k in window[-1] if isinstance(window[-1][k], float)]
        avg = {k: float(np.mean([s[k] for s in window if k in s])) for k in keys}
        record = {
            "time": time.time(), "update": self.update, "version": self.version, "warmup": window[-1]["warmup"],
            "updates_per_s": len(window) / max(seconds, 1e-9), "buffer_steps": self.buffer.steps,
            "queue": self.incoming.qsize(), "replay_ratio": self.totals["trained_steps"] / max(1, self.totals["steps"]),
            "mean_return": self.mean_return(), "reward_per_100_steps": self.reward_per_100_steps(),
            "lr": self.optimizer.param_groups[0]["lr"],
            **self.totals, **avg,
        }
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        bc = f" bc {avg['bc_loss']:.3f}x{avg['bc_weight']:.2f}" if "bc_loss" in avg else ""
        print(
            f"upd {self.update:6d} v{self.version} {'[value warm-up] ' if record['warmup'] else ''}"
            f"pg {avg['pg_loss']:+.4f} V {avg['value_loss']:.4f} H {avg['entropy']:.3f}{bc} "
            f"rho {avg['rho_mean']:.2f} ({avg['rho_far']:.0%} far) | "
            f"buf {self.buffer.steps} recv {self.totals['steps']} ratio {record['replay_ratio']:.1f} | "
            f"lives {self.totals['episodes']} K/D {self.totals['kills']}/{self.totals['deaths']} "
            f"R/life {record['mean_return']:+.3f} R/100 {record['reward_per_100_steps']:+.3f} | "
            f"{record['updates_per_s']:.1f} upd/s",
            flush=True,
        )


def publish_snapshot(snapshot: dict, redis_url: str, onnx_path: Path) -> int:
    """Export a checkpoint dict to ONNX and publish it to Redis; returns the new version."""
    started = time.perf_counter()
    model = CombatEventTransformer(**snapshot["model_config"])
    load_policy_weights(model, snapshot["model_state_dict"])
    meta = export_policy(model.eval(), snapshot, onnx_path, atol=1e-3)
    meta["rl_update"] = snapshot["rl_state"]["update"]
    version = replay.publish_model(replay.connect(redis_url), onnx_path.read_bytes(), meta)
    print(f"Published model version {version} (update {meta['rl_update']}, {time.perf_counter() - started:.1f}s)",
          flush=True)
    return version


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--publish-worker"]:  # child process of Trainer.publish(background=True)
        snapshot_path, redis_url, onnx_path = argv[1:4]
        publish_snapshot(torch.load(snapshot_path, map_location="cpu", weights_only=True), redis_url, Path(onnx_path))
        return 0
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    def on_sigterm(signum, frame):  # docker stop / kill: save latest.pth like on Ctrl+C
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    Trainer(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
