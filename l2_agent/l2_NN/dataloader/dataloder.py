"""
Event-driven ``Dataset`` / ``DataLoader`` for Lineage 2 Interlude (C6) player-action logs.

The source table (``l2jc6_playeractions_cleaned.parquet``) holds one row per action frame
emitted by the game client - movement, target selection, attacks, skill casts, pick-ups,
... - and *not* a fixed-rate snapshot, so the temporal context of every decision has to be
rebuilt from the events themselves.

For every event ``t`` of a ``session_id`` the dataset emits the trailing window
``[t - L + 1 .. t]`` (``L`` is ``seq_len`` / the ``-w`` flag).  Windows never cross a
session boundary; when the window reaches back before the session start it is padded on
the left with zeros and ``seq_mask`` marks the real timesteps.

Returned sample
---------------
======================  =====================  ==========================================================
key                     shape                  meaning
======================  =====================  ==========================================================
``player_state``        ``(L, F_player)``      HP/MP/CP ratios, combat flag, level and speeds
``mobs_matrix``         ``(L, K, F_monster)``  Top-K nearby monsters sorted by distance (padded slots flagged)
``skills_matrix``       ``(L, N, F_skill)``    the character's skill panel
``delta_t``             ``(L,)``               ``log(1 + dt)`` with ``dt`` in seconds since the previous event
``seq_mask``            ``(L,)``               ``True`` for real events, ``False`` for left padding
``target_action_type``  scalar ``long``        action id of the target event ``t``
``target_pointer_idx``  scalar ``long``        slot of the targeted monster inside ``mobs_matrix[t]``
``target_skill_slot``   scalar ``long``        slot of the used skill inside ``skills_matrix[t]``
======================  =====================  ==========================================================

Schema adaptation notes
-----------------------
The column names of the real Parquet file are not the intuitive ones, so every read goes
through :func:`_feature_number`, which accepts a list of candidate paths (flat or
``struct.field``) and falls back to a default for ``None`` / ``NaN`` / missing values:

============================  =========================================================
logged column                 consumed as
============================  =========================================================
``timestamp``                 string ``%Y.%m.%d %H:%M:%S,%f`` -> ``datetime64`` -> ``delta_t``
``char_stats.cast_speed``     player cast speed (there is **no** flat ``cast_speed`` column)
``char_stats.atk_speed``      player attack speed (likewise no flat column)
``nearby_objects[].id``       monster npc id (the key is ``id``, **not** ``npc_id``)
``entity_npc_id``             npc id of the target of the current event
``active_skills[].id`` and    skill panel; there is **no** ``available_skills`` column
``cooldowns[].skill_id`` /    and no per-skill MP cost anywhere in the logs
``cooldowns[].remaining``
``skill_id``                  the ``target_skill_slot`` label, for ``action_type == used_skill``
============================  =========================================================

Skill panel slots
-----------------
``skills_matrix`` holds ``N`` slots of ``F_skill`` values, one per entry of
:data:`SKILL_FEATURE_NAMES`:

=========================  ==========================================================
``is_available``           the slot holds one of the character's skills
``is_off_cooldown``        the skill is ready to cast (nothing left in ``cooldowns``)
``cooldown_norm``          remaining cooldown / :data:`SKILL_COOLDOWN_NORM_SECONDS`, clipped
``has_enough_mp``          the current MP covers the skill cost
=========================  ==========================================================

``cooldowns[].remaining`` is the only timing info the logs give per skill; it is logged in
milliseconds and converted with :data:`MILLISECONDS_PER_SECOND` before the scaling above.

``move_speed`` and the per-skill MP cost are simply absent from the logs: the move speed
falls back to :data:`DEFAULT_MOVE_SPEED` and an unknown MP cost is treated as affordable.

Action label space
------------------
``target_action_type`` indexes :data:`ACTION_TYPE_IDS`.  Ids ``0..5`` are exactly the
canonical set pinned by the dataloader contract, and the table then continues with every
other action type the log parser can emit, so no logged event has to fall through to
``unknown``:

======  =================  ======  ===================
id      action_type        id      action_type
======  =================  ======  ===================
0       moved              10      resurrected
1       selected_target    11      effect_applied
2       attacked           12      skills_updated
3       used_skill         13      used_item
4       picked_up          14      learned_skill
5       unknown            15      autouse_enabled
6       inflicted_damage   16      autouse_disabled
7       took_damage        17      dropped
8       killed             18      buy
9       died               19      teleported
======  =================  ======  ===================

``unknown`` stays reserved for values the tables do not know (and for the parser's
"``performed_action``"/"``other``" buckets).  :meth:`Lineage2EventDataset.unmapped_action_types`
lists any such value together with its event count, so a missing entry is visible instead of
silently training the fallback class.

Usage::

    from l2_NN.dataloader.dataloder import Lineage2EventDataset

    dataset = Lineage2EventDataset(pd.read_parquet("../l2jc6_playeractions_cleaned.parquet"), seq_len=16)

The shape/dtype smoke test that used to live in this module's ``__main__`` block now lives in
``l2_NN/tools/check_dataloader.py``::

    uv run l2_NN/tools/check_dataloader.py -i ../l2jc6_playeractions_cleaned.parquet -w 16
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["ACTION_TYPE_IDS", "Lineage2EventDataset"]
# --------------------------------------------------------------------------------------
# Feature layout and normalisation constants
# --------------------------------------------------------------------------------------

#: ``action_type`` string -> classification id (the label space of the Action Type head).
#: Ids ``0..5`` are the canonical decision actions pinned by the dataloader contract; the
#: remaining ids cover every other action type the log parser can emit
#: (see ``tools/README.md`` -> "Action types"), so no logged event has to be swallowed by
#: ``unknown`` any more.  ``unknown`` stays reserved for values this table does not know,
#: which keeps the head robust against new client builds.
ACTION_TYPE_IDS: Final[dict[str, int]] = {
    # --- the decisions the agent has to model -------------------------------------------
    "moved": 0,
    "selected_target": 1,
    "attacked": 2,
    "used_skill": 3,
    "picked_up": 4,
    # --- fallback for a value that is not in this table ---------------------------------
    "unknown": 5,
    # --- combat outcomes (the environment answering the player) -------------------------
    "inflicted_damage": 6,
    "took_damage": 7,
    "killed": 8,
    "died": 9,
    "resurrected": 10,
    "effect_applied": 11,
    # --- skills, items and inventory ----------------------------------------------------
    "skills_updated": 12,
    "used_item": 13,
    "learned_skill": 14,
    "autouse_enabled": 15,
    "autouse_disabled": 16,
    "dropped": 17,
    # --- non-combat / locomotion (dropped by the cleaning stage, kept for raw logs) ------
    "buy": 18,
    "teleported": 19,
}

#: Raw spellings that occur in the logs / are produced by the parser, folded onto the
#: canonical names above.  A canonical name never needs an entry here: an unmapped value
#: falls through to itself and is looked up in :data:`ACTION_TYPE_IDS` directly.
ACTION_TYPE_ALIASES: Final[dict[str, str]] = {
    # movement
    "move": "moved",
    "moved_to": "moved",
    "moveto": "moved",
    # targeting
    "select_target": "selected_target",
    "selectedtarget": "selected_target",
    "target_selected": "selected_target",
    # attacks / skills
    "attack": "attacked",
    "use_skill": "used_skill",
    "cast": "used_skill",
    "usedskill": "used_skill",
    # pick-ups (the log says "pickup", the contract says "picked_up")
    "pickup": "picked_up",
    "pick_up": "picked_up",
    "loot": "picked_up",
    "pickedup": "picked_up",
    # combat outcomes
    "damage_inflicted": "inflicted_damage",
    "inflicteddamage": "inflicted_damage",
    "damage_taken": "took_damage",
    "received_damage": "took_damage",
    "tookdamage": "took_damage",
    "kill": "killed",
    "die": "died",
    "death": "died",
    "resurrect": "resurrected",
    "revived": "resurrected",
    "effectapplied": "effect_applied",
    "buff_applied": "effect_applied",
    # skills / items
    "skillsupdated": "skills_updated",
    "skill_updated": "skills_updated",
    "skills_update": "skills_updated",
    "use_item": "used_item",
    "item_used": "used_item",
    "useditem": "used_item",
    "used_potion": "used_item",
    "learn_skill": "learned_skill",
    "learnedskill": "learned_skill",
    "autouse_on": "autouse_enabled",
    "enabledautouseforitem": "autouse_enabled",
    "enable_autouse": "autouse_enabled",
    "autouse_off": "autouse_disabled",
    "disabledautouseforitem": "autouse_disabled",
    "disable_autouse": "autouse_disabled",
    "drop": "dropped",
    "buy_item": "buy",
    "buyitem": "buy",
    "purchased": "buy",
    "teleportedfrom": "teleported",
    "teleport": "teleported",
    # parser buckets that describe "no recognised action", i.e. genuinely unknown events
    "performed_action": "unknown",
    "performedaction": "unknown",
    "other": "unknown",
    "none": "unknown",
}

UNKNOWN_ACTION_ID: Final[int] = ACTION_TYPE_IDS["unknown"]
USED_SKILL_ACTION_ID: Final[int] = ACTION_TYPE_IDS["used_skill"]

PLAYER_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "hp_ratio",
    "mp_ratio",
    "cp_ratio",
    "in_combat",
    "lvl_norm",
    "cast_speed",
    "atk_speed",
    "move_speed",
)
PLAYER_FEATURE_DIM: Final[int] = len(PLAYER_FEATURE_NAMES)

MOB_FEATURE_NAMES: Final[tuple[str, ...]] = ("dist_norm", "lvl_diff", "is_current_target", "is_padded")
MOB_FEATURE_DIM: Final[int] = len(MOB_FEATURE_NAMES)
MOB_PADDED_INDEX: Final[int] = MOB_FEATURE_NAMES.index("is_padded")

SKILL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "is_available",
    "is_off_cooldown",
    "cooldown_norm",
    "has_enough_mp",
)
SKILL_FEATURE_DIM: Final[int] = len(SKILL_FEATURE_NAMES)

#: Timestamp layout used by the client log (and therefore by the Parquet file).
TIMESTAMP_FORMAT: Final[str] = "%Y.%m.%d %H:%M:%S,%f"

MAX_LEVEL: Final[float] = 80.0
MAX_DISTANCE: Final[float] = 2000.0
MAX_CAST_SPEED: Final[float] = 2000.0
MAX_ATK_SPEED: Final[float] = 1500.0
MAX_MOVE_SPEED: Final[float] = 300.0

#: The client logs cooldown timers in milliseconds (``[CD:1177:2466]``), the dataset works in seconds.
MILLISECONDS_PER_SECOND: Final[float] = 1000.0

#: Scale of a "long" skill cooldown: ``cooldowns[].remaining`` (seconds) is divided by this
#: value and clipped to ``[0, 1]`` for the ``cooldown_norm`` slot feature.
SKILL_COOLDOWN_NORM_SECONDS: Final[float] = 10.0

#: Fallbacks used when the logs carry no value at all.
DEFAULT_CAST_SPEED: Final[float] = 333.0
DEFAULT_ATK_SPEED: Final[float] = 300.0
DEFAULT_MOVE_SPEED: Final[float] = 120.0
DEFAULT_DISTANCE: Final[float] = MAX_DISTANCE

#: Candidate column paths, flat columns first, ``char_stats`` children second.
CAST_SPEED_PATHS: Final[tuple[str, ...]] = ("cast_speed", "char_stats.cast_speed")
ATK_SPEED_PATHS: Final[tuple[str, ...]] = ("atk_speed", "char_stats.atk_speed")
MOVE_SPEED_PATHS: Final[tuple[str, ...]] = ("move_speed", "char_stats.move_speed")

#: Columns the dataset cannot work without.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("session_id", "timestamp", "action_type")


# --------------------------------------------------------------------------------------
# NaN / None tolerant accessors
# --------------------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """
    Return ``True`` for ``None``, ``NaN``, ``NaT`` and ``pd.NA``.

    List-like cells (``nearby_objects``, ``cooldowns``, ...) make :func:`pandas.isna`
    return an array, so the array-to-bool conversion is swallowed and reported as "present".
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, default: float) -> float:
    """Cast ``value`` to a finite float, returning ``default`` for ``None``/``NaN``/garbage."""
    if _is_missing(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value: Any) -> bool:
    """
    Cast a possibly missing or stringified Parquet cell to ``bool``.

    ``NaN`` is truthy in plain Python, so the missing-value check has to happen first.
    """
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _lookup(row: Any, key: str, default: Any = None) -> Any:
    """``row.get(key, default)`` that never raises and treats ``NaN``/``None`` as absent."""
    getter = getattr(row, "get", None)
    if getter is None:
        return default
    try:
        value = getter(key, default)
    except (TypeError, ValueError):
        return default
    return default if _is_missing(value) else value


def _feature_number(row: Any, paths: Sequence[str], default: float) -> float:
    """
    Return the first present, finite value found along ``paths``.

    A path may address a ``struct`` child (``"char_stats.cast_speed"``); every candidate is
    tried in order and ``default`` is returned when none of them holds a usable number.
    """
    for path in paths:
        value: Any = row
        for part in path.split("."):
            value = _lookup(value, part, None)
            if value is None:
                break
        else:
            number = _as_float(value, math.nan)
            if math.isfinite(number):
                return number
    return default


def _optional_number(row: Any, paths: Sequence[str]) -> Optional[float]:
    """Like :func:`_feature_number`, but ``None`` when no usable value exists."""
    number = _feature_number(row, paths, math.nan)
    return None if math.isnan(number) else number


def _dict_list(value: Any) -> list[Mapping[str, Any]]:
    """Normalise a ``list<struct>`` Parquet cell into a plain list of dict-like entries."""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _ratio(numerator: float, denominator: float) -> float:
    """Safe ``numerator / denominator`` clamped to ``[0, 1]``."""
    if denominator <= 0.0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


# --------------------------------------------------------------------------------------
# Per-event feature cache
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _FeatureArrays:
    """
    Features of every event, computed once.

    A sample is nothing but a row slice of these arrays plus the trailing targets, so the
    per-row Python work happens exactly once per dataset instead of once per epoch.
    """

    player_state: np.ndarray
    mobs_matrix: np.ndarray
    skills_matrix: np.ndarray
    log_delta_t: np.ndarray
    pointer_idx: np.ndarray
    skill_slot: np.ndarray
    action_type: np.ndarray
    #: ``True`` when the event's ``entity_npc_id`` really was among the Top-K candidates,
    #: i.e. when ``pointer_idx`` is a meaningful supervision signal rather than the fallback 0.
    pointer_resolved: np.ndarray
    #: ``True`` when the event's used skill really is part of the character's panel, i.e. when
    #: ``skill_slot`` is a meaningful supervision signal rather than the fallback 0.
    skill_resolved: np.ndarray

    @property
    def num_events(self) -> int:
        """Number of cached events (== number of rows of the source table)."""
        return int(self.player_state.shape[0])


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class Lineage2EventDataset(Dataset):
    """Event-driven sliding-window dataset for Lineage 2 Interlude (C6) action logs."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 16,
        max_mobs: int = 10,
        max_skills: int = 16,
        *,
        verbose: bool = False,
    ) -> None:
        """
        :param df: frame read from the cleaned player-action Parquet file
        :param seq_len: length ``L`` of the context window fed to the Transformer encoder
        :param max_mobs: number of monster candidates ``K`` in the Top-K target matrix
        :param max_skills: number ``N`` of skill-panel slots
        :param verbose: print construction statistics (sessions, skill panels, timing)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if max_mobs < 1:
            raise ValueError(f"max_mobs must be >= 1, got {max_mobs}")
        if max_skills < 1:
            raise ValueError(f"max_skills must be >= 1, got {max_skills}")

        self.seq_len = int(seq_len)
        self.max_mobs = int(max_mobs)
        self.max_skills = int(max_skills)
        self.verbose = verbose

        #: ``action_type`` string -> classification id.
        self.action_to_id: dict[str, int] = dict(ACTION_TYPE_IDS)
        #: classification id -> canonical ``action_type`` string.
        self.id_to_action: dict[int, str] = {value: key for key, value in self.action_to_id.items()}

        self.data = self._preprocess_dataframe(df)
        #: ``character_id`` -> ordered list of skill ids (the slot -> skill id mapping).
        self.skill_panel, self.skill_mp_cost = self._build_skill_panels(self.data)
        #: per-event features, sliced by :meth:`__getitem__`.
        self.features = self._precompute_features(self.data)
        #: ``(window_start, window_end)`` inclusive positional pairs, one per event.
        self.samples: list[tuple[int, int]] = self._create_samples(self.data)

        # Keep the label space honest: anything the tables above do not know is reported
        # instead of quietly teaching the model an ``unknown`` class.
        unmapped = self.unmapped_action_types()
        if unmapped:
            total = sum(unmapped.values())
            print(
                f"Warning: {total} event(s) use an action_type that is not in ACTION_TYPE_IDS/"
                f"ACTION_TYPE_ALIASES and will be labelled `unknown`: {unmapped}"
            )

        if self.verbose:
            self._report_construction()

    def __len__(self) -> int:
        return len(self.samples)

    # ----------------------------------------------------------------------------------
    # Introspection helpers
    # ----------------------------------------------------------------------------------

    @property
    def feature_dims(self) -> dict[str, int]:
        """Feature-vector sizes the model heads have to expect."""
        return {"player": PLAYER_FEATURE_DIM, "monster": MOB_FEATURE_DIM, "skill": SKILL_FEATURE_DIM}

    @property
    def num_actions(self) -> int:
        """Size of the ``target_action_type`` label space."""
        return len(self.action_to_id)

    def unmapped_action_types(self) -> dict[str, int]:
        """
        Raw ``action_type`` values that fall back to :data:`UNKNOWN_ACTION_ID`.

        A non-empty result means the Parquet file holds an action type that is missing from
        :data:`ACTION_TYPE_IDS` / :data:`ACTION_TYPE_ALIASES`; add it to one of those tables
        instead of letting those events train the ``unknown`` class.
        """
        unresolved: dict[str, int] = {}
        for raw, count in self.data["action_type"].value_counts().items():
            key = str(raw).strip().lower()
            canonical = ACTION_TYPE_ALIASES.get(key, key)
            if canonical not in self.action_to_id:
                unresolved[str(raw)] = int(count)
        return unresolved

    def summary(self) -> dict[str, Any]:
        """Cheap label/coverage statistics - handy as a smoke test and for logging."""
        target_actions = self.features.action_type
        pointer_idx = self.features.pointer_idx
        skill_slot = self.features.skill_slot
        skill_events = target_actions == USED_SKILL_ACTION_ID
        action_counts = {
            self.id_to_action[action_id]: int(np.count_nonzero(target_actions == action_id))
            for action_id in sorted(self.id_to_action)
        }
        return {
            "events": self.features.num_events,
            "windows": len(self.samples),
            "sessions": int(self.data["session_id"].nunique()),
            "characters": int(self.data["character_id"].nunique()) if "character_id" in self.data else 0,
            "mean_log_delta_t": float(np.mean(self.features.log_delta_t)),
            "max_log_delta_t": float(np.max(self.features.log_delta_t)) if self.features.num_events else 0.0,
            "action_counts": action_counts,
            "action_classes_present": sum(1 for count in action_counts.values() if count),
            "unmapped_actions": self.unmapped_action_types(),
            # A slot of 0 is the documented fallback, so coverage is tracked separately:
            # `pointer_resolved` / `skill_resolved` say whether the label points at a real entry.
            "pointer_hits": int(np.count_nonzero(pointer_idx)),
            "pointer_resolved": int(np.count_nonzero(self.features.pointer_resolved)),
            "skill_events": int(np.count_nonzero(skill_events)),
            "skill_slot_hits": int(np.count_nonzero(skill_slot[skill_events])),
            "skill_resolved": int(np.count_nonzero(self.features.skill_resolved)),
            "skill_slots_per_character": {int(key): len(value) for key, value in self.skill_panel.items()},
        }

    def _report_construction(self) -> None:
        stats = self.summary()
        print("\n--- Dataset construction ---")
        print(f"  events / windows        : {stats['events']} / {stats['windows']}")
        print(f"  sessions / characters   : {stats['sessions']} / {stats['characters']}")
        print(f"  feature dims (P/M/S)    : {PLAYER_FEATURE_DIM} / {MOB_FEATURE_DIM} / {SKILL_FEATURE_DIM}")
        print(f"  mean log(delta_t)       : {stats['mean_log_delta_t']:.4f}")
        present = {name: count for name, count in stats["action_counts"].items() if count}
        print(f"  action classes present  : {stats['action_classes_present']} of {self.num_actions}")
        print(f"  action distribution     : {present}")
        print(f"  unmapped action types   : {stats['unmapped_actions'] or 'none'}")
        print(
            f"  target labels resolved  : pointer {stats['pointer_resolved']}, "
            f"skill {stats['skill_resolved']} of {stats['skill_events']} used_skill events"
        )
        print(f"  nonzero pointer slots   : {stats['pointer_hits']}")
        print(f"  skill slots per char    : {stats['skill_slots_per_character']}")

    # ----------------------------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------------------------

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps, order every session chronologically and derive ``delta_t``."""
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"input frame is missing required column(s): {', '.join(missing)}")
        if df.empty:
            raise ValueError("input frame is empty")

        frame = df.copy()

        timestamps = frame["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            # The client writes "2026.07.04 16:39:17,848"; inference is only used when the
            # whole column fails to parse with the known layout.
            parsed = pd.to_datetime(timestamps, format=TIMESTAMP_FORMAT, errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(timestamps, errors="coerce")
            timestamps = parsed
        frame["timestamp"] = timestamps

        unusable = frame["timestamp"].isna() | frame["session_id"].isna()
        if unusable.any():
            print(f"Warning: dropping {int(unusable.sum())} row(s) without a usable timestamp/session_id")
            frame = frame.loc[~unusable]

        # Sessions are only contiguous after this sort.
        frame = frame.sort_values(["session_id", "timestamp"], kind="stable").reset_index(drop=True)

        # delta_t is computed inside each session: shift(1) never bleeds across sessions and
        # the first event of every session gets dt = 0.
        previous = frame.groupby("session_id", sort=False)["timestamp"].shift(1)
        delta_t = (frame["timestamp"] - previous).dt.total_seconds().fillna(0.0).clip(lower=0.0)
        delta_t = delta_t.astype("float64")

        frame["delta_t"] = delta_t
        frame["log_delta_t"] = np.log1p(delta_t.to_numpy())
        frame["action_type"] = frame["action_type"].astype("string").fillna("unknown").astype(str)
        return frame

    def _build_skill_panels(self, frame: pd.DataFrame) -> tuple[dict[Any, list[int]], dict[int, float]]:
        """
        Build the per-character skill panel and the skill -> MP-cost table.

        The logs carry **no** ``available_skills`` list and no per-skill MP cost, so the panel
        is derived from everything the character is ever seen holding:

        * ``active_skills[].id`` - the actives reported by ``skills_updated`` frames,
        * ``cooldowns[].skill_id`` - every skill the client tracks a cooldown for,
        * ``skill_id`` of ``used_skill`` frames - the label space itself.

        Slots are ranked by usage count (descending, then by first appearance) so the
        ``max_skills`` most frequently cast skills survive truncation, which keeps
        ``target_skill_slot`` resolvable.  The ordering is global per character, which gives
        the Skill head stable slot semantics across time.

        Should a future log expose ``available_skills`` (list of dicts with ``skill_id``,
        ``mp_cost``, ``cooldown_left``) it is honoured first and its MP costs are kept.
        """
        usage: defaultdict[Any, Counter] = defaultdict(Counter)
        first_seen: defaultdict[Any, dict[int, int]] = defaultdict(dict)
        mp_cost: dict[int, float] = {}
        appearances = 0

        def register(character_id: Any, skill_id: Any, *, weight: int = 1) -> None:
            """Count one observation of ``skill_id`` for ``character_id``."""
            nonlocal appearances
            if _is_missing(character_id) or _is_missing(skill_id):
                return
            try:
                skill_key = int(skill_id)
            except (TypeError, ValueError):
                return
            usage[character_id][skill_key] += weight
            if skill_key not in first_seen[character_id]:
                first_seen[character_id][skill_key] = appearances
                appearances += 1

        has_available_skills = "available_skills" in frame.columns and frame["available_skills"].notna().any()
        used_skill_label = self.id_to_action[USED_SKILL_ACTION_ID]

        # Positional column lookup: `itertuples` is much faster than `iterrows` here and does
        # not depend on column names being valid Python identifiers.
        positions = {name: position for position, name in enumerate(frame.columns)}
        if "character_id" not in positions:
            print("Warning: the frame has no character_id column - the skill panel stays empty")
            return {}, mp_cost

        character_at = positions["character_id"]
        action_at = positions.get("action_type")
        skill_at = positions.get("skill_id")
        active_at = positions.get("active_skills")
        cooldown_at = positions.get("cooldowns")
        available_at = positions.get("available_skills")

        for row in frame.itertuples(index=False, name=None):
            character_id = row[character_at]
            if not _is_missing(character_id):
                character_id = int(character_id)
            if has_available_skills and available_at is not None:
                for entry in _dict_list(row[available_at]):
                    skill_id = _optional_number(entry, ("skill_id", "id"))
                    register(character_id, skill_id, weight=2)
                    cost = _optional_number(entry, ("mp_cost", "cost"))
                    if skill_id is not None and cost is not None:
                        mp_cost[int(skill_id)] = cost
            if active_at is not None:
                for entry in _dict_list(row[active_at]):
                    register(character_id, _optional_number(entry, ("id", "skill_id")))
            if cooldown_at is not None:
                for entry in _dict_list(row[cooldown_at]):
                    register(character_id, _optional_number(entry, ("skill_id", "id")))
            if action_at is not None and str(row[action_at]) == used_skill_label and skill_at is not None:
                register(character_id, row[skill_at], weight=max(self.max_skills, 4))

        panel: dict[Any, list[int]] = {}
        for character_id, position in first_seen.items():
            ranked = sorted(position, key=lambda skill_id: (-usage[character_id][skill_id], position[skill_id]))
            panel[character_id] = ranked[: self.max_skills]
        return panel, mp_cost

    def _create_samples(self, frame: pd.DataFrame) -> list[tuple[int, int]]:
        """
        Enumerate the ``(start, end)`` windows, one per event, never across sessions.

        Because the frame is sorted by ``session_id``, the rows of a session form a
        contiguous block, so a window is simply ``[end - L + 1 .. end]`` clamped to the block
        start.  Events close to the session start therefore yield short (later left-padded)
        windows instead of borrowing rows from the previous session.
        """
        sessions = frame["session_id"].to_numpy()
        block_starts = np.flatnonzero(np.concatenate(([True], sessions[1:] != sessions[:-1])))
        block_ends = np.concatenate((block_starts[1:], [sessions.size]))

        if block_starts.size != np.unique(sessions).size:
            raise ValueError("session_id values are not contiguous after sorting - cannot build windows")

        samples: list[tuple[int, int]] = []
        for block_start, block_end in zip(block_starts, block_ends):
            for end_idx in range(int(block_start), int(block_end)):
                start_idx = max(int(block_start), end_idx - self.seq_len + 1)
                if end_idx - start_idx + 1 > self.seq_len:  # pragma: no cover - defensive
                    raise AssertionError("window longer than seq_len")
                samples.append((start_idx, end_idx))
        return samples

    # ----------------------------------------------------------------------------------
    # Per-event encoders
    # ----------------------------------------------------------------------------------

    def _encode_action_type(self, row: Any) -> int:
        """Map an ``action_type`` cell onto its classification id."""
        raw = _lookup(row, "action_type", None)
        if raw is None:
            return UNKNOWN_ACTION_ID
        key = str(raw).strip().lower()
        canonical = ACTION_TYPE_ALIASES.get(key, key)
        return self.action_to_id.get(canonical, UNKNOWN_ACTION_ID)

    def _encode_player_state(self, row: Any) -> np.ndarray:
        """
        Encode the player state vector ``V_player`` (``F_player`` values).

        Health/mana/CP are ratios, the combat flag is binary, the level and the three
        speeds are min-max style normalisations.  Every value is NaN/None safe.
        """
        hp = _feature_number(row, ("hp", "char_stats.hp"), 0.0)
        max_hp = _feature_number(row, ("max_hp", "char_stats.max_hp"), 1.0)
        mp = _feature_number(row, ("mp", "char_stats.mp"), 0.0)
        max_mp = _feature_number(row, ("max_mp", "char_stats.max_mp"), 1.0)
        cp = _feature_number(row, ("cp", "char_stats.cp"), 0.0)
        max_cp = _feature_number(row, ("max_cp", "char_stats.max_cp"), 1.0)

        level = _feature_number(row, ("lvl", "level"), 1.0)
        cast_speed = _feature_number(row, CAST_SPEED_PATHS, DEFAULT_CAST_SPEED)
        atk_speed = _feature_number(row, ATK_SPEED_PATHS, DEFAULT_ATK_SPEED)
        move_speed = _feature_number(row, MOVE_SPEED_PATHS, DEFAULT_MOVE_SPEED)

        return np.array(
            (
                _ratio(hp, max_hp),
                _ratio(mp, max_mp),
                _ratio(cp, max_cp),
                1.0 if _as_bool(_lookup(row, "in_combat", None)) else 0.0,
                _ratio(level, MAX_LEVEL),
                _ratio(cast_speed, MAX_CAST_SPEED),
                _ratio(atk_speed, MAX_ATK_SPEED),
                _ratio(move_speed, MAX_MOVE_SPEED),
            ),
            dtype=np.float32,
        )

    def _sorted_mob_candidates(self, row: Any) -> list[Mapping[str, Any]]:
        """Return ``nearby_objects`` sorted by distance ascending (closest first)."""
        nearby = _dict_list(_lookup(row, "nearby_objects", None))
        # ``sorted`` is stable, so entries with equal/absent distance keep their log order.
        return sorted(nearby, key=lambda mob: _as_float(mob.get("distance"), DEFAULT_DISTANCE))

    def _encode_mobs_matrix(self, row: Any) -> tuple[np.ndarray, int, bool]:
        """
        Encode the Top-K target-candidate matrix ``M`` (``(K, F_monster)``) and its label.

        Rows are the ``K`` closest monsters; every unused slot is flagged with
        ``is_padded = 1.0``.  ``is_current_target`` compares the monster's ``id`` (the real
        key in the Parquet list) with the event's ``entity_npc_id``.

        :return: ``(matrix, target_pointer_idx, resolved)`` where ``target_pointer_idx`` is the
            slot of the targeted monster and ``resolved`` tells whether the target really was
            among the Top-K candidates.  When nothing is targeted (or the target is farther
            away than the K closest monsters) the index falls back to ``0``.
        """
        matrix = np.zeros((self.max_mobs, MOB_FEATURE_DIM), dtype=np.float32)
        matrix[:, MOB_PADDED_INDEX] = 1.0

        player_level = _feature_number(row, ("lvl", "level"), 1.0)
        target_npc_id = _optional_number(row, ("entity_npc_id", "entity_npc.id"))

        pointer_idx = 0
        matched = False
        for slot_idx, mob in enumerate(self._sorted_mob_candidates(row)[: self.max_mobs]):
            distance = _as_float(mob.get("distance"), DEFAULT_DISTANCE)
            dist_norm = min(max(distance, 0.0) / MAX_DISTANCE, 1.0)

            mob_level = _optional_number(mob, ("lvl", "level"))
            lvl_diff = 0.0 if mob_level is None else (mob_level - player_level) / MAX_LEVEL

            mob_npc_id = _optional_number(mob, ("id", "npc_id"))
            is_target = 0.0
            if target_npc_id is not None and mob_npc_id is not None and int(mob_npc_id) == int(target_npc_id):
                is_target = 1.0
                if not matched:
                    # Keep the *closest* matching candidate as the label.
                    pointer_idx = slot_idx
                    matched = True

            matrix[slot_idx] = (dist_norm, lvl_diff, is_target, 0.0)

        return matrix, pointer_idx, matched

    def _cooldown_left_by_skill(self, row: Any) -> dict[int, float]:
        """Map ``skill_id -> remaining cooldown`` **in seconds** for the current event."""
        remaining: dict[int, float] = {}
        for entry in _dict_list(_lookup(row, "cooldowns", None)):
            skill_id = _optional_number(entry, ("skill_id", "id"))
            if skill_id is None:
                continue
            # The log stores milliseconds (``remaining`` maxes out at 10000 == the 10 s client
            # cap), so the value is converted here and every consumer works in seconds.
            milliseconds = _as_float(entry.get("remaining"), 0.0)
            remaining[int(skill_id)] = milliseconds / MILLISECONDS_PER_SECOND
        return remaining

    def _current_mp_cost(self, skill_id: int) -> float:
        """MP cost of ``skill_id``; the logs never store one, so this is usually ``0.0``."""
        if skill_id in self.skill_mp_cost:
            return self.skill_mp_cost[skill_id]
        # ``skill_cost`` in the logs is the SP price of a ``learned_skill`` frame, not an MP
        # cost, so it must never be used here; an unknown cost counts as affordable.
        return 0.0

    def _encode_skills_matrix(self, row: Any, action_id: int) -> tuple[np.ndarray, int, bool]:
        """
        Encode the skill panel S ((N, SKILL_FEATURE_DIM)) and the target_skill_slot label.

        Features per slot (``F_skill = SKILL_FEATURE_DIM``, see :data:`SKILL_FEATURE_NAMES`):
        1. is_available     - 1.0 if the slot holds a skill, 0.0 for the padding slots.
        2. is_off_cooldown  - 1.0 if ready to cast, 0.0 while the skill is cooling down.
        3. cooldown_norm    - remaining cooldown / :data:`SKILL_COOLDOWN_NORM_SECONDS`, clipped
                              to ``[0, 1]`` (0.0 means "ready right now").
        4. has_enough_mp    - 1.0 if the current MP covers the skill cost, 0.0 if it does not.
        """
        matrix = np.zeros((self.max_skills, SKILL_FEATURE_DIM), dtype=np.float32)

        character_id = _optional_number(row, ("character_id",))
        panel = self.skill_panel.get(int(character_id), []) if character_id is not None else []

        current_mp = _feature_number(row, ("mp", "char_stats.mp"), 0.0)
        remaining = self._cooldown_left_by_skill(row)

        used_skill_id = _optional_number(row, ("skill_id",)) if action_id == USED_SKILL_ACTION_ID else None
        used_skill_key = None if used_skill_id is None else int(used_skill_id)

        slot_label = 0
        resolved = False

        for slot_idx, skill_id in enumerate(panel[: self.max_skills]):
            cooldown_left = max(0.0, float(remaining.get(skill_id, 0.0)))
            is_off_cooldown = 1.0 if cooldown_left <= 0.0 else 0.0

            # Cooldown progress: 0.0 is "ready", 1.0 a full (or longer than the scale) cooldown.
            cooldown_norm = min(cooldown_left / SKILL_COOLDOWN_NORM_SECONDS, 1.0)

            cost = self._current_mp_cost(skill_id)
            # An unknown cost (the logs carry none) counts as affordable, so the flag only turns
            # 0.0 once a per-skill MP cost table exists and the character cannot pay it.
            has_enough_mp = 1.0 if (cost == 0.0 or current_mp >= cost) else 0.0

            matrix[slot_idx] = (
                1.0,  # [0] is_available
                is_off_cooldown,  # [1] is_off_cooldown (1.0 = ready, 0.0 = cooling down)
                cooldown_norm,  # [2] cooldown_norm (0.0 = ready, >0.0 = seconds to wait)
                has_enough_mp,  # [3] has_enough_mp
            )

            if used_skill_key is not None and skill_id == used_skill_key:
                slot_label = slot_idx
                resolved = True

        return matrix, slot_label, resolved

    # ----------------------------------------------------------------------------------
    # Feature cache and sampling
    # ----------------------------------------------------------------------------------

    def _precompute_features(self, frame: pd.DataFrame) -> _FeatureArrays:
        """Encode every event exactly once so that windows become pure array slices."""
        count = len(frame)
        player_state = np.zeros((count, PLAYER_FEATURE_DIM), dtype=np.float32)
        mobs_matrix = np.zeros((count, self.max_mobs, MOB_FEATURE_DIM), dtype=np.float32)
        mobs_matrix[:, :, MOB_PADDED_INDEX] = 1.0
        skills_matrix = np.zeros((count, self.max_skills, SKILL_FEATURE_DIM), dtype=np.float32)
        log_delta_t = frame["log_delta_t"].to_numpy(dtype=np.float32, copy=True)
        pointer_idx = np.zeros(count, dtype=np.int64)
        skill_slot = np.zeros(count, dtype=np.int64)
        action_type = np.zeros(count, dtype=np.int64)
        pointer_resolved = np.zeros(count, dtype=bool)
        skill_resolved = np.zeros(count, dtype=bool)

        for position, (_, row) in enumerate(frame.iterrows()):
            action_id = self._encode_action_type(row)
            action_type[position] = action_id
            player_state[position] = self._encode_player_state(row)
            mobs_matrix[position], pointer_idx[position], pointer_resolved[position] = self._encode_mobs_matrix(row)
            skills_matrix[position], skill_slot[position], skill_resolved[position] = self._encode_skills_matrix(
                row, action_id
            )

        return _FeatureArrays(
            player_state=player_state,
            mobs_matrix=mobs_matrix,
            skills_matrix=skills_matrix,
            log_delta_t=log_delta_t,
            pointer_idx=pointer_idx,
            skill_slot=skill_slot,
            action_type=action_type,
            pointer_resolved=pointer_resolved,
            skill_resolved=skill_resolved,
        )

    @staticmethod
    def _pad_timesteps(block: np.ndarray, pad_len: int, pad_value: float = 0.0) -> np.ndarray:
        """Left-pad a ``(t, ...)`` block with ``pad_len`` constant timesteps."""
        if pad_len <= 0:
            return block
        padding = np.full((pad_len, *block.shape[1:]), pad_value, dtype=block.dtype)
        return np.concatenate((padding, block), axis=0)

    def _pad_mob_timesteps(self, block: np.ndarray, pad_len: int) -> np.ndarray:
        """Left-pad mob windows, flagging every slot of the padded timesteps as padded."""
        if pad_len <= 0:
            return block
        padding = np.zeros((pad_len, *block.shape[1:]), dtype=block.dtype)
        padding[..., MOB_PADDED_INDEX] = 1.0
        return np.concatenate((padding, block), axis=0)

    def _pad_skill_timesteps(self, block: np.ndarray, pad_len: int) -> np.ndarray:
        """Left-pad skill windows; zeroed slots mean "no skill" (``is_available = 0``)."""
        return self._pad_timesteps(block, pad_len, 0.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start_idx, end_idx = self.samples[index]
        length = end_idx - start_idx + 1
        pad_len = self.seq_len - length
        features = self.features

        player_state = self._pad_timesteps(features.player_state[start_idx : end_idx + 1], pad_len)
        mobs_matrix = self._pad_mob_timesteps(features.mobs_matrix[start_idx : end_idx + 1], pad_len)
        skills_matrix = self._pad_skill_timesteps(features.skills_matrix[start_idx : end_idx + 1], pad_len)
        delta_t = self._pad_timesteps(features.log_delta_t[start_idx : end_idx + 1], pad_len)

        seq_mask = np.zeros(self.seq_len, dtype=bool)
        seq_mask[pad_len:] = True

        return {
            "player_state": torch.from_numpy(np.ascontiguousarray(player_state)),
            "mobs_matrix": torch.from_numpy(np.ascontiguousarray(mobs_matrix)),
            "skills_matrix": torch.from_numpy(np.ascontiguousarray(skills_matrix)),
            "delta_t": torch.from_numpy(np.ascontiguousarray(delta_t)),
            "seq_mask": torch.from_numpy(seq_mask),
            "target_action_type": torch.tensor(int(features.action_type[end_idx]), dtype=torch.long),
            "target_pointer_idx": torch.tensor(int(features.pointer_idx[end_idx]), dtype=torch.long),
            "target_skill_slot": torch.tensor(int(features.skill_slot[end_idx]), dtype=torch.long),
        }
