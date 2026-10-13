"""
Smoke test for the Lineage 2 event ``Dataset`` / ``DataLoader``.

The dataset itself lives in ``l2_NN/dataloader/dataloder.py``; this script is the only place that
reads a Parquet file, builds the dataset, wraps it into a :class:`torch.utils.data.DataLoader`
and verifies the collated batches - shapes, dtypes, finite values, label ranges and the
``skills_matrix`` slot contract (``is_available`` / ``is_off_cooldown`` / ``cooldown_norm`` /
``has_enough_mp`` consistency, plus "a ``used_skill`` target always hits a real slot").  Run it
by hand after touching the dataloader, the log converter or the Parquet tables.

Usage::

    uv run l2_NN/tools/check_dataloader.py -i ../l2jc6_playeractions_cleaned.parquet -w 16

    # verify every batch of the shuffled loader instead of only the first one
    uv run l2_NN/tools/check_dataloader.py -i ../l2jc6_playeractions_cleaned.parquet --all-batches

The script is standalone: it derives the project root from its own location, so it can be run
from any working directory.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd
import torch
from torch.utils.data import DataLoader

# ``uv run l2_NN/tools/check_dataloader.py`` puts *this file's* directory on ``sys.path``, so the
# project root (the folder holding ``l2_NN/``) has to be added before the dataset can be imported.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_NN.dataloader.dataloder import (  # noqa: E402 - PROJECT_ROOT is added just above
    ACTION_TYPE_IDS,
    MOB_FEATURE_DIM,
    PLAYER_FEATURE_DIM,
    SKILL_FEATURE_DIM,
    SKILL_FEATURE_NAMES,
    USED_SKILL_ACTION_ID,
    Lineage2EventDataset,
)

#: Positions inside a skill slot row, resolved from the dataloader's own layout declaration so
#: that the checks keep working when a slot feature is inserted or reordered.
SKILL_AVAILABLE_AT: Final[int] = SKILL_FEATURE_NAMES.index("is_valid")
SKILL_OFF_COOLDOWN_AT: Final[int] = SKILL_FEATURE_NAMES.index("is_off_cooldown")
SKILL_COOLDOWN_NORM_AT: Final[int] = SKILL_FEATURE_NAMES.index("cooldown_norm")
SKILL_ENOUGH_MP_AT: Final[int] = SKILL_FEATURE_NAMES.index("has_enough_mp")


def _check_skill_panel(batch: dict[str, torch.Tensor]) -> None:
    """
    Validate the ``skills_matrix`` contract of one collated batch.

    Every slot row has to stay inside the declared ranges, an empty slot must be all zeros, a
    ready skill must have ``cooldown_norm == 0`` while a cooling-down skill must have
    ``cooldown_norm > 0``, and the ``target_skill_slot`` of a resolved ``used_skill`` target must
    point at a real slot.
    """
    skills = batch["skills_matrix"]  # (B, L, N, F_skill)
    available = skills[..., SKILL_AVAILABLE_AT]
    off_cooldown = skills[..., SKILL_OFF_COOLDOWN_AT]
    cooldown_norm = skills[..., SKILL_COOLDOWN_NORM_AT]
    enough_mp = skills[..., SKILL_ENOUGH_MP_AT]

    for name, flag in (("is_valid", available), ("is_off_cooldown", off_cooldown), ("has_enough_mp", enough_mp)):
        assert bool(((flag == 0.0) | (flag == 1.0)).all()), f"skill slot {name} must be 0.0 or 1.0"
    assert float(cooldown_norm.min()) >= 0.0, "cooldown_norm below 0"
    assert float(cooldown_norm.max()) <= 1.0, "cooldown_norm above 1"

    assert float(skills.min()) >= 0.0 and float(skills.max()) <= 1.0, "skill features must lie in [0, 1]"
    empty = available == 0.0
    assert bool((skills[empty] == 0.0).all()), "an empty skill slot must be all zeros"

    ready = (available == 1.0) & (off_cooldown == 1.0)
    cooling = (available == 1.0) & (off_cooldown == 0.0)
    assert bool((cooldown_norm[ready] == 0.0).all()), "a ready skill must have cooldown_norm == 0"
    assert bool((cooldown_norm[cooling] > 0.0).all()), "a cooling-down skill must have cooldown_norm > 0"

    used_skill = (batch["target_action_type"] == USED_SKILL_ACTION_ID) & batch["target_skill_valid"]
    if bool(used_skill.any()):
        # The target timestep of a sample is its own event, so the label can be checked right here.
        slot_rows = skills[used_skill][:, -1]
        slots = batch["target_skill_slot"][used_skill]
        picked = slot_rows[torch.arange(len(slots)), slots]
        assert bool((picked[:, SKILL_AVAILABLE_AT] == 1.0).all()), "used_skill target points at an empty slot"


def _check_batch(batch: dict[str, torch.Tensor], seq_len: int, max_mobs: int, max_skills: int) -> None:
    """Validate the shapes, dtypes and value ranges of one collated batch."""
    batch_size = batch["player_state"].shape[0]
    assert batch["player_state"].shape == (batch_size, seq_len, PLAYER_FEATURE_DIM), "player_state shape"
    assert batch["mobs_matrix"].shape == (batch_size, seq_len, max_mobs, MOB_FEATURE_DIM), "mobs_matrix shape"
    assert batch["skills_matrix"].shape == (batch_size, seq_len, max_skills, SKILL_FEATURE_DIM), "skills_matrix shape"
    assert batch["delta_t"].shape == (batch_size, seq_len), "delta_t shape"
    assert batch["seq_mask"].shape == (batch_size, seq_len), "seq_mask shape"

    assert batch["player_state"].dtype == torch.float32, "player_state dtype"
    assert batch["mobs_matrix"].dtype == torch.float32, "mobs_matrix dtype"
    assert batch["skills_matrix"].dtype == torch.float32, "skills_matrix dtype"
    assert batch["delta_t"].dtype == torch.float32, "delta_t dtype"
    assert batch["seq_mask"].dtype == torch.bool, "seq_mask dtype"

    for key in ("target_action_type", "target_pointer_idx", "target_skill_slot"):
        assert batch[key].shape == (batch_size,), f"{key} shape"
        assert batch[key].dtype == torch.long, f"{key} dtype"

    assert torch.isfinite(batch["player_state"]).all(), "player_state contains NaN/inf"
    assert torch.isfinite(batch["mobs_matrix"]).all(), "mobs_matrix contains NaN/inf"
    assert torch.isfinite(batch["skills_matrix"]).all(), "skills_matrix contains NaN/inf"
    assert torch.isfinite(batch["delta_t"]).all(), "delta_t contains NaN/inf"
    assert batch["delta_t"].min().item() >= 0.0, "delta_t must be log1p of a non-negative dt"
    assert batch["seq_mask"].any(), "every sample must contain at least one real event"

    assert int(batch["target_action_type"].min()) >= 0, "action id below the label space"
    assert int(batch["target_action_type"].max()) < len(ACTION_TYPE_IDS), "action id above the label space"
    assert int(batch["target_pointer_idx"].min()) >= 0, "pointer index below 0"
    assert int(batch["target_pointer_idx"].max()) < max_mobs, "pointer index above K-1"
    assert int(batch["target_skill_slot"].min()) >= 0, "skill slot below 0"
    assert int(batch["target_skill_slot"].max()) < max_skills, "skill slot above N-1"

    _check_skill_panel(batch)


def main(argv: Sequence[str] | None = None) -> int:
    """Read a Parquet file, build the dataset and push one batch through a DataLoader."""
    parser = argparse.ArgumentParser(description="Smoke test the Lineage 2 event DataLoader.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the player-action Parquet file")
    parser.add_argument("-w", type=int, default=16, help="Window length L / seq_len (default: 16)")
    parser.add_argument("--max-mobs", type=int, default=10, help="Top-K monster candidates (default: 10)")
    parser.add_argument("--max-skills", type=int, default=64, help="Skill panel slots N (default: 64)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size to probe (default: 4)")
    parser.add_argument("--all-batches", action="store_true", help="Iterate the whole loader instead of one batch")
    parser.add_argument("--quiet", action="store_true", help="Suppress the dataset statistics")
    args = parser.parse_args(argv)

    print(f"Loading {args.input}")
    frame = pd.read_parquet(args.input)
    print(f"Loaded data frame: {frame.shape[0]} rows x {frame.shape[1]} columns")

    dataset = Lineage2EventDataset(
        frame,
        seq_len=args.w,
        max_mobs=args.max_mobs,
        max_skills=args.max_skills,
        verbose=not args.quiet,
    )
    print(f"Dataset created. Samples (windows): {len(dataset)}")
    if len(dataset) == 0:
        print("Warning: the dataset is empty - nothing to load.")
        return 1

    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)
    checked = 0
    for batch in loader:
        _check_batch(batch, args.w, args.max_mobs, args.max_skills)
        if checked == 0:
            print("\n--- Batch check ---")
            print("player_state   :", tuple(batch["player_state"].shape), batch["player_state"].dtype)
            print("mobs_matrix    :", tuple(batch["mobs_matrix"].shape), batch["mobs_matrix"].dtype)
            print("skills_matrix  :", tuple(batch["skills_matrix"].shape), batch["skills_matrix"].dtype)
            print("skill features :", SKILL_FEATURE_NAMES)
            last_timestep = batch["skills_matrix"][0, -1]
            filled = last_timestep[:, SKILL_AVAILABLE_AT] == 1.0
            cooling = filled & (last_timestep[:, SKILL_OFF_COOLDOWN_AT] == 0.0)
            print(f"skill slots    : {int(filled.sum())} available, {int(cooling.sum())} cooling down (sample 0, last)")
            if bool(cooling.any()):
                print(
                    "cooldown_norm  :", [round(v, 3) for v in last_timestep[cooling, SKILL_COOLDOWN_NORM_AT].tolist()]
                )
            print("delta_t        :", tuple(batch["delta_t"].shape), batch["delta_t"].dtype)
            print("seq_mask       :", tuple(batch["seq_mask"].shape), batch["seq_mask"].dtype)
            print("seq_mask sum   :", batch["seq_mask"].sum(dim=1).tolist()[:8], "(first 8)")
            print("target_action  :", batch["target_action_type"].tolist()[:8], "(first 8)")
            print("target_pointer :", batch["target_pointer_idx"].tolist()[:8], "(first 8)")
            print("target_skill   :", batch["target_skill_slot"].tolist()[:8], "(first 8)")
            print("player_state[0, -1]:", [round(value, 4) for value in batch["player_state"][0, -1].tolist()])
        checked += 1
        if not args.all_batches:
            break

    print(f"\nOK - {checked} batch(es) verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
