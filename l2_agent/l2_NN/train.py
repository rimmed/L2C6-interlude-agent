"""
Behavioral-cloning training loop for the Lineage 2 :class:`CombatEventTransformer`.

The script reads the cleaned player-action Parquet file, builds one
:class:`~l2_NN.dataloader.dataloder.Lineage2EventDataset` over the whole frame (so the per-character
skill panels keep the same slot semantics in train and validation), splits its windows into
train / validation **by group** (``session_id`` by default, ``character_id`` on request) so no
session leaks across the split, and trains the three Multi-Discrete heads with AdamW, cosine LR,
gradient clipping and mixed precision.

Loss and metrics
----------------
The three heads are not equally meaningful for every sample, so their losses are masked:

==============  =====================================  ==============================================
head            supervised on                          why
==============  =====================================  ==============================================
action type     every sample                           the label always exists
pointer idx     samples whose target mob is in Top-K   otherwise the label is the fallback slot 0
skill slot      ``used_skill`` samples                 every other action labels slot 0 by default
==============  =====================================  ==============================================

``total = w_action * L_action + w_pointer * L_pointer + w_skill * L_skill`` (the defaults match
``CombatEventTransformer.compute_loss``).  Accuracies use the same masks.

Label leak at the decision step
-------------------------------
The window ends at the event being predicted, and ``mobs_matrix[t, :, is_current_target]`` is
computed from that very event's ``entity_npc_id`` - i.e. it *is* the pointer label.  By default the
flag is zeroed at the last timestep (history keeps it); ``--keep-last-target-flag`` disables this.

Usage::

    uv run l2_NN/train.py                                  # defaults: ../l2jc6_playeractions_cleaned.parquet
    uv run l2_NN/train.py --epochs 40 --batch-size 128 --lr 3e-4
    uv run l2_NN/train.py --split-by character --device cpu --amp off

    # train while the validation loss keeps falling; stop after 5 epochs without improvement
    uv run l2_NN/train.py --until-plateau
    uv run l2_NN/train.py --until-plateau --patience 10 --min-delta 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, get_worker_info
from tqdm import tqdm

# ``uv run l2_NN/train.py`` puts *this file's* directory on ``sys.path``; the project root (the
# folder holding ``l2_NN/``) has to be added before the package can be imported.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_NN.dataloader.augment import AugmentConfig  # noqa: E402
from l2_NN.dataloader.dataloder import (  # noqa: E402 - PROJECT_ROOT is added just above
    MOB_FEATURE_DIM,
    MOB_FEATURE_NAMES,
    PLAYER_FEATURE_DIM,
    PLAYER_FEATURE_NAMES,
    SKILL_FEATURE_DIM,
    SKILL_FEATURE_NAMES,
    SKILL_ACTION_IDS,
    Lineage2EventDataset,
)
from l2_NN.model.CombatEventTransformer import CombatEventTransformer  # noqa: E402
from l2_client.tools.data import class_panels  # noqa: E402

#: The cleaned dataset lives in the repository root, one level above the ``l2_agent`` project.
DEFAULT_DATA_PATH: Final[Path] = PROJECT_ROOT.parent / "l2jc6_playeractions_cleaned.parquet"

MOB_IS_TARGET_AT: Final[int] = MOB_FEATURE_NAMES.index("is_current_target")
#: Actions whose skill slot the skill head learns (the character's skills and its summon's).
SKILL_ACTION_TENSOR: Final[torch.Tensor] = torch.tensor(SKILL_ACTION_IDS)

INPUT_KEYS: Final[tuple[str, ...]] = ("player_state", "delta_t", "mobs_matrix", "skills_matrix", "seq_mask")
TARGET_KEYS: Final[tuple[str, ...]] = ("target_action_type", "target_pointer_idx", "target_skill_slot")
HEADS: Final[tuple[str, ...]] = ("action", "pointer", "skill")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the L2 CombatEventTransformer by behavioral cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Cleaned player-action Parquet file")
    data.add_argument("--seq-len", type=int, default=16, help="Context window length L")
    data.add_argument("--max-mobs", type=int, default=10, help="Top-K monster candidates")
    data.add_argument("--max-skills", type=int, default=64, help="Skill panel slots N")
    data.add_argument("--val-fraction", type=float, default=0.2, help="Share of windows held out for validation")
    data.add_argument(
        "--split-by",
        choices=("session", "character"),
        default="session",
        help="Group key of the train/val split; no group is ever on both sides",
    )
    data.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    data.add_argument("--no-augment", action="store_true", help="Train on plain windows (see dataloader/augment.py)")
    data.add_argument(
        "--balance-actions",
        type=float,
        default=0.0,
        help="Oversample rare target actions: sample weight = count(action)^-ALPHA (0 = off, 0.5 = moderate)",
    )
    data.add_argument(
        "--keep-last-target-flag",
        action="store_true",
        help="Do not zero is_current_target at the decision step (leaks the pointer label)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--nhead", type=int, default=4)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--dim-feedforward", type=int, default=256)
    model.add_argument("--dropout", type=float, default=0.1)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--epochs", type=int, default=20)
    optim.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    optim.add_argument("--min-lr", type=float, default=1e-6, help="Cosine schedule floor")
    optim.add_argument("--warmup-epochs", type=float, default=1.0, help="Linear warm-up length (may be fractional)")
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient L2 norm")
    optim.add_argument("--w-action", type=float, default=1.0, help="Loss weight of the action-type head")
    optim.add_argument("--w-pointer", type=float, default=0.5, help="Loss weight of the pointer head")
    optim.add_argument("--w-skill", type=float, default=0.5, help="Loss weight of the skill-slot head")
    optim.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop after N epochs without val-loss improvement (0 = off; default 5 with --until-plateau, else 0)",
    )
    optim.add_argument(
        "--min-delta", type=float, default=0.0, help="Val-loss decrease that still counts as an improvement"
    )
    optim.add_argument(
        "--until-plateau",
        action="store_true",
        help="Ignore --epochs: train until the val loss stops improving for --patience epochs "
        "(LR: warm-up + ReduceLROnPlateau instead of cosine)",
    )
    optim.add_argument("--max-epochs", type=int, default=1000, help="Safety cap on epochs with --until-plateau")

    run = parser.add_argument_group("runtime")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda / cuda:1 / cpu")
    run.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default="auto",
        help="Mixed precision; auto = bf16 if supported, else fp16 on CUDA, off on CPU",
    )
    run.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.patience is None:
        args.patience = 5 if args.until_plateau else 0
    if args.until_plateau and args.patience <= 0:
        parser.error("--until-plateau needs --patience >= 1")
    # From here on ``args.epochs`` is the epoch budget of both modes.
    if args.until_plateau:
        args.epochs = args.max_epochs
    return args


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def group_split(
    dataset: Lineage2EventDataset, split_by: str, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Split window indices into train / val so that every group lands on exactly one side.

    The group of a window is the ``session_id`` / ``character_id`` of its target event.  Groups are
    shuffled and assigned to validation until it holds ``val_fraction`` of the windows.
    """
    column = "session_id" if split_by == "session" else "character_id"
    end_rows = np.fromiter((end for _, end in dataset.samples), dtype=np.int64, count=len(dataset.samples))
    window_groups = dataset.data[column].to_numpy()[end_rows]

    groups, counts = np.unique(window_groups, return_counts=True)
    if groups.size < 2:
        raise ValueError(f"cannot split by {split_by}: only {groups.size} distinct value(s)")

    order = np.random.default_rng(seed).permutation(groups.size)
    target = val_fraction * len(end_rows)
    val_groups, taken = [], 0
    for position in order[:-1]:  # never move every group to validation
        if taken >= target:
            break
        val_groups.append(groups[position])
        taken += int(counts[position])

    is_val = np.isin(window_groups, np.asarray(val_groups))
    return np.flatnonzero(~is_val).tolist(), np.flatnonzero(is_val).tolist()


class AugmentedWindows(Dataset):
    """The training split with window augmentations; every DataLoader worker draws from its own RNG."""

    def __init__(self, dataset: Lineage2EventDataset, indices: list[int], cfg: AugmentConfig, seed: int):
        self.dataset = dataset
        self.indices = indices
        self.cfg = cfg
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if self._rng is None:
            worker = get_worker_info()
            self._rng = np.random.default_rng([self.seed, worker.id if worker is not None else 0])
        return self.dataset.window(self.indices[i], self.cfg, self._rng)


def build_loaders(
    args: argparse.Namespace,
) -> tuple[Lineage2EventDataset, DataLoader, DataLoader, dict[str, int]]:
    print(f"Loading {args.data_path}")
    frame = pd.read_parquet(args.data_path)
    started = time.perf_counter()
    dataset = Lineage2EventDataset(
        frame, seq_len=args.seq_len, max_mobs=args.max_mobs, max_skills=args.max_skills, verbose=True
    )
    print(f"Dataset built in {time.perf_counter() - started:.1f}s: {len(dataset)} windows")

    train_idx, val_idx = group_split(dataset, args.split_by, args.val_fraction, args.seed)
    column = "session_id" if args.split_by == "session" else "character_id"
    end_rows = [dataset.samples[i][1] for i in val_idx]
    split_info = {
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "val_groups": int(dataset.data[column].iloc[end_rows].nunique()),
    }
    print(
        f"Split by {args.split_by}: {split_info['train_windows']} train / {split_info['val_windows']} val windows "
        f"({split_info['val_groups']} {args.split_by}(s) held out)"
    )

    pin = args.device.startswith("cuda")
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_set = Subset(dataset, train_idx) if args.no_augment else AugmentedWindows(dataset, train_idx, AugmentConfig(), args.seed)
    sampler = None
    if args.balance_actions > 0:
        actions = dataset.features.action_type[[dataset.samples[i][1] for i in train_idx]]
        counts = np.bincount(actions, minlength=dataset.num_actions).astype(np.float64)
        weights = counts[actions] ** -args.balance_actions
        sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(train_idx), replacement=True,
                                        generator=generator)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, drop_last=True,
                              generator=generator if sampler is None else None, **common)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **common)
    return dataset, train_loader, val_loader, split_info


def prepare_batch(
    batch: dict[str, torch.Tensor], device: torch.device, hide_last_target: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Move a batch to ``device``, derive the head masks and hide the decision-step target flag."""
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    # The pointer label is meaningful only when the targeted mob is among the Top-K candidates,
    # which is exactly when the decision step carries an is_current_target flag.
    last_target_flags = batch["mobs_matrix"][:, -1, :, MOB_IS_TARGET_AT]
    masks = {
        "pointer": last_target_flags.gt(0.5).any(dim=-1),
        "skill": torch.isin(batch["target_action_type"], SKILL_ACTION_TENSOR.to(batch["target_action_type"].device))
        & batch["target_skill_valid"],
    }
    if hide_last_target:
        mobs = batch["mobs_matrix"].clone()
        mobs[:, -1, :, MOB_IS_TARGET_AT] = 0.0
        batch["mobs_matrix"] = mobs
    return batch, masks


# --------------------------------------------------------------------------------------
# Loss & metrics
# --------------------------------------------------------------------------------------


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Cross-entropy over the masked samples; an all-false mask yields a graph-connected zero."""
    if mask is None:
        return F.cross_entropy(logits.float(), target)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask].float(), target[mask])


def compute_losses(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_head = {
        "action": masked_ce(logits["logits_action_type"], batch["target_action_type"], None),
        "pointer": masked_ce(logits["logits_pointer_idx"], batch["target_pointer_idx"], masks["pointer"]),
        "skill": masked_ce(logits["logits_skill_slot"], batch["target_skill_slot"], masks["skill"]),
    }
    total = sum(weights[head] * per_head[head] for head in HEADS)
    return total, per_head


@dataclass
class MetricTracker:
    """
    Sample-weighted running averages of the losses and accuracies of one epoch.

    Each head is averaged over the samples it is supervised on, so a batch with few
    ``used_skill`` samples does not weigh as much as a full one for the skill metrics.
    """

    loss_sum: dict[str, float] = field(default_factory=lambda: dict.fromkeys(("total", *HEADS), 0.0))
    loss_count: dict[str, int] = field(default_factory=lambda: dict.fromkeys(("total", *HEADS), 0))
    correct: dict[str, int] = field(default_factory=lambda: dict.fromkeys(HEADS, 0))
    seen: dict[str, int] = field(default_factory=lambda: dict.fromkeys(HEADS, 0))

    @torch.no_grad()
    def update(
        self,
        total: torch.Tensor,
        per_head: dict[str, torch.Tensor],
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> None:
        batch_size = batch["target_action_type"].shape[0]
        head_masks = {"action": torch.ones(batch_size, dtype=torch.bool, device=total.device), **masks}
        pairs = {
            "action": ("logits_action_type", "target_action_type"),
            "pointer": ("logits_pointer_idx", "target_pointer_idx"),
            "skill": ("logits_skill_slot", "target_skill_slot"),
        }
        self.loss_sum["total"] += float(total) * batch_size
        self.loss_count["total"] += batch_size
        for head, (logit_key, target_key) in pairs.items():
            mask = head_masks[head]
            count = int(mask.sum())
            if count == 0:
                continue
            hits = logits[logit_key][mask].argmax(dim=-1).eq(batch[target_key][mask])
            self.correct[head] += int(hits.sum())
            self.seen[head] += count
            self.loss_sum[head] += float(per_head[head]) * count
            self.loss_count[head] += count

    def summary(self) -> dict[str, float]:
        result = {}
        for key in ("total", *HEADS):
            result[f"loss_{key}"] = self.loss_sum[key] / self.loss_count[key] if self.loss_count[key] else math.nan
        for head in HEADS:
            result[f"acc_{head}"] = self.correct[head] / self.seen[head] if self.seen[head] else math.nan
        return result


# --------------------------------------------------------------------------------------
# Train / validate
# --------------------------------------------------------------------------------------


def resolve_amp(mode: str, device: torch.device) -> torch.dtype | None:
    """Pick the autocast dtype (``None`` = full fp32)."""
    if device.type != "cuda" or mode == "off":
        if mode in ("bf16", "fp16") and device.type != "cuda":
            print(f"Note: --amp {mode} ignored on {device.type}, training in fp32")
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def run_epoch(
    model: CombatEventTransformer,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    amp_dtype: torch.dtype | None,
    hide_last_target: bool,
    description: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One pass over ``loader``; trains when ``optimizer`` is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    tracker = MetricTracker()
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)

    with torch.set_grad_enabled(training):
        for raw_batch in progress:
            batch, masks = prepare_batch(raw_batch, device, hide_last_target)
            inputs = {key: batch[key] for key in INPUT_KEYS}

            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(inputs)
            # Loss in fp32 for numerical stability (masked_ce upcasts the logits).
            total, per_head = compute_losses(logits, batch, masks, weights)

            if training:
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite training loss: {float(total)}")
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total = total.detach()
            tracker.update(total, {k: v.detach() for k, v in per_head.items()}, logits, batch, masks)
            postfix = {"loss": f"{total.item():.4f}"}
            if training:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(postfix)

    return tracker.summary()


def build_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, steps_per_epoch: int
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, torch.optim.lr_scheduler.ReduceLROnPlateau | None]:
    """
    Return ``(per_step, per_epoch)`` schedulers.

    Fixed ``--epochs``: per-step linear warm-up followed by cosine annealing down to ``--min-lr``.
    ``--until-plateau``: the run length is unknown, so cosine cannot be planned; the per-step
    warm-up is followed by :class:`ReduceLROnPlateau` on the val loss, which halves the LR after
    half of ``--patience`` stale epochs - so the LR gets a chance to drop before training stops.
    """
    if args.until_plateau:
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        # After ``total_iters`` LinearLR leaves the LR alone, so it never undoes a plateau reduction.
        warmup = (
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
            if warmup_steps > 0
            else None
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, args.patience // 2),
            threshold=args.min_delta,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )
        return warmup, plateau

    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(int(args.warmup_epochs * steps_per_epoch), total_steps - 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr
    )
    if warmup_steps <= 0:
        return cosine, None
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps]), None


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"loss {metrics['loss_total']:.4f} "
        f"(act {metrics['loss_action']:.4f} | ptr {metrics['loss_pointer']:.4f} | skl {metrics['loss_skill']:.4f})  "
        f"acc act {metrics['acc_action']:6.2%} | ptr {metrics['acc_pointer']:6.2%} | skl {metrics['acc_skill']:6.2%}"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    dataset, train_loader, val_loader, split_info = build_loaders(args)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise SystemExit("train or validation split is empty - lower --batch-size or adjust --val-fraction")

    # Dimensions come from the dataset, not the model defaults, so the two can never drift apart.
    model_config = {
        "player_dim": PLAYER_FEATURE_DIM,
        "mob_features": MOB_FEATURE_DIM,
        "skill_features": SKILL_FEATURE_DIM,
        "max_mobs": args.max_mobs,
        "max_skills": args.max_skills,
        "num_action_types": dataset.num_actions,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model = CombatEventTransformer(**model_config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, plateau_scheduler = build_scheduler(optimizer, args, len(train_loader))
    amp_dtype = resolve_amp(args.amp, device)
    # GradScaler is only needed for fp16; bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler(device.type) if amp_dtype is torch.float16 else None
    if args.until_plateau:
        print(f"Mode: until plateau (patience {args.patience}, min delta {args.min_delta}, cap {args.epochs} epochs)")
    print(
        f"Device: {device}  |  AMP: {amp_dtype or 'off'}  |  hide decision-step target: {not args.keep_last_target_flag}"
    )

    weights = {"action": args.w_action, "pointer": args.w_pointer, "skill": args.w_skill}
    hide_last_target = not args.keep_last_target_flag
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best_model.pth"
    last_path = args.save_dir / "last_model.pth"
    history: list[dict[str, float]] = []
    best_val, best_epoch, stale_epochs = math.inf, 0, 0

    def checkpoint(epoch: int, train_metrics: dict, val_metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "plateau_scheduler_state_dict": plateau_scheduler.state_dict() if plateau_scheduler is not None else None,
            "model_config": model_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "action_to_id": dataset.action_to_id,
            # Feature layouts by name: export_onnx / train_rl must not assume today's layout for old checkpoints.
            "features": {"player": list(PLAYER_FEATURE_NAMES), "mob": list(MOB_FEATURE_NAMES),
                         "skill": list(SKILL_FEATURE_NAMES),
                         # the class panels (l2_client/tools/data/class_panels.json) the skill slots come from
                         "skill_panels": class_panels.panels_hash()},
            "split": {"by": args.split_by, **split_info},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, device, weights, amp_dtype, hide_last_target,
            description=f"Epoch {epoch}/{args.epochs} [train]",
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, grad_clip=args.grad_clip,
        )  # fmt: skip
        val_metrics = run_epoch(
            model, val_loader, device, weights, amp_dtype, hide_last_target,
            description=f"Epoch {epoch}/{args.epochs} [val]  ",
        )  # fmt: skip

        improved = val_metrics["loss_total"] < best_val - args.min_delta
        lr = optimizer.param_groups[0]["lr"]  # the LR this epoch was trained with
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["loss_total"])
        print(f"\nEpoch {epoch:3d}/{args.epochs}  lr {lr:.2e}  {time.perf_counter() - started:.1f}s")
        print(f"  train  {format_metrics(train_metrics)}")
        print(f"  val    {format_metrics(val_metrics)}" + ("  * best" if improved else ""))

        state = checkpoint(epoch, train_metrics, val_metrics)
        torch.save(state, last_path)
        if improved:
            best_val, best_epoch, stale_epochs = val_metrics["loss_total"], epoch, 0
            torch.save(state, best_path)
        else:
            stale_epochs += 1

        history.append({"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})  # fmt: skip
        (args.save_dir / "history.json").write_text(json.dumps(history, indent=2))

        if args.patience and stale_epochs >= args.patience:
            print(f"\nEarly stopping: no val improvement for {args.patience} epoch(s)")
            break
    else:
        if args.until_plateau:
            print(f"\nReached the --max-epochs cap ({args.max_epochs}) while the val loss was still improving")

    print(f"\nBest val loss {best_val:.4f} at epoch {best_epoch} -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
