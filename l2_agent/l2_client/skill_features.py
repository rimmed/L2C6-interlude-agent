"""
Feature vector of one skill-panel slot - shared by the training dataloader and the live client.

A slot describes *what the skill is* (category, target, power, MP cost, range, cast and reuse time
from the datapack, see :func:`skill_data.get_skill_info`) and *its state right now* (ready,
remaining cooldown, affordable). Because the model sees the skill's nature instead of a slot
number, it can use skills of classes it was never trained on.

Both sides call :func:`encode_skill_slot`, so the features are identical by construction.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from .tools.data.skill_data import CATEGORIES, TARGETS, SkillInfo

#: Feature layout of a slot. ``is_valid`` is 0 for an empty (padding) slot - the whole row is 0 then.
SKILL_SLOT_FEATURES: Final[tuple[str, ...]] = (
    "is_valid",
    *(f"cat_{c}" for c in CATEGORIES),
    *(f"tgt_{t}" for t in TARGETS),
    "is_magic",
    "power_norm",
    "mp_cost_ratio",
    "cast_range_norm",
    "hit_time_norm",
    "reuse_norm",
    "is_off_cooldown",
    "cooldown_norm",
    "has_enough_mp",
)
SKILL_SLOT_DIM: Final[int] = len(SKILL_SLOT_FEATURES)

#: Normalisation scales (values are clipped to [0, 1] after scaling).
POWER_LOG_SCALE: Final[float] = math.log1p(3000.0)  # physical skill power reaches a few thousand
CAST_RANGE_SCALE: Final[float] = 1200.0
HIT_TIME_SCALE_MS: Final[float] = 5000.0
REUSE_LOG_SCALE: Final[float] = math.log1p(600.0)  # seconds
#: Remaining cooldown in game seconds that maps to cooldown_norm = 1 (the logs cap at 10 s).
COOLDOWN_SCALE_S: Final[float] = 10.0


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def encode_skill_slot(info: SkillInfo, *, max_mp: float, cur_mp: float, cooldown_left_s: float) -> np.ndarray:
    """
    Features of one occupied slot.

    :param info: the skill at the character's level of it
    :param max_mp: the character's maximum MP (the MP cost is fed relative to it)
    :param cur_mp: the character's current MP (``has_enough_mp``)
    :param cooldown_left_s: remaining cooldown in *game* seconds (0 = ready)
    """
    row = np.zeros(SKILL_SLOT_DIM, dtype=np.float32)
    row[0] = 1.0
    row[1 + CATEGORIES.index(info.category)] = 1.0
    base = 1 + len(CATEGORIES)
    row[base + TARGETS.index(info.target)] = 1.0
    base += len(TARGETS)
    left = max(0.0, cooldown_left_s)
    row[base:] = (
        1.0 if info.is_magic else 0.0,
        _clip01(math.log1p(max(info.power, 0.0)) / POWER_LOG_SCALE),
        _clip01(info.mp_cost / max(max_mp, 1.0)),
        _clip01(info.cast_range / CAST_RANGE_SCALE),
        _clip01(info.hit_time_ms / HIT_TIME_SCALE_MS),
        _clip01(math.log1p(max(info.reuse_ms, 0.0) / 1000.0) / REUSE_LOG_SCALE),
        1.0 if left <= 0.0 else 0.0,
        _clip01(left / COOLDOWN_SCALE_S),
        1.0 if cur_mp >= info.mp_cost else 0.0,
    )
    return row
