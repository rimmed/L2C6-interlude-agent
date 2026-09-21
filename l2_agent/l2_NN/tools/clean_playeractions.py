"""
Clean the L2J player-action dataset (``l2jc6_playeractions.parquet``).

The project runs a hierarchical architecture: macro-navigation, pathfinding and
roaming are executed by a deterministic client script, so the learned policy only
covers **combat execution** — from target initiation to the monster's death and the
first loot pickup. The pipeline therefore slices the logs into tightly bounded
combat sessions and discards every record outside an engagement.

Five stages run in this order:

0. **Excluded characters removed** — GM / admin / test accounts listed in
   ``exclude_characters`` are dropped completely before anything else. Their
   names are resolved to ``character_id``s against ``l2jc6_characters.parquet``,
   because the action table only stores ids. Excluded rows never take part in
   session numbering.

1. **Combat sessionization** — boundaries are detected on the *unfiltered*
   timeline so a later filtering step can never fake (or hide) a break. Sessions
   exist only between a target selection and the end of that engagement:

   * **start** — a ``selected_target`` row aimed at a hostile NPC resolved through
     ``l2jc6_npcs.parquet``. Item selections (``[selected target]: Adena``) and
     town-NPC selections (``Merchant`` / ``Folk`` / ``Guard`` / ``Teleporter`` …)
     are not combat and start nothing. Re-selecting the target of the session that
     is already open is a client-side refresh, not a new engagement, and keeps the
     session running (Case C speaks of a *new* target);
   * **end, case A (kill + loot)** — a ``killed`` row followed by a ``pickup``
     before the next engagement or interruption: the session ends on that first
     pickup, which is exactly the "kill → loot" reward horizon;
   * **end, case B (kill only)** — a ``killed`` row with no pickup following: the
     session ends on the last ``killed`` frame of the multi-drop kill run;
   * **end, case C (aborted)** — the engagement is broken before a kill: a raw gap
     above ``gap_seconds``, a death / teleport / respawn / login, or a *different*
     NPC being targeted. The session then ends on the last combat frame before the
     interruption.

   Everything outside ``[start, end]`` — idle roaming, city walking, trading, long
   travel paths — is dropped.

2. **Idle collapsing** — a contiguous run of ``IDLE`` / ``NONE`` records keeps
   only its very first frame, whatever its duration. This removes duplicate noise
   and the multi-second idle clusters without leaving a temporal jump.

3. **Environment array cleaning** (DuckDB) — ``nearby_objects`` is unnested,
   filtered and re-aggregated. The environment vector keeps hostile NPCs
   (``Monster``, ``RaidBoss``, ``GrandBoss``, …), loot / dropped items
   (``Item``, ``L2Item``, …) and other players (``Player``, ``L2PcInstance``),
   drops entries farther than ``max_monster_distance``, sorts by distance and
   truncates to the ``top_n_objects`` closest.

4. **Micro-event aggregation, session validation and dense renumbering**:

   * a rapid run of server/system frames (``killed``/``dropped``/``effect_applied``
     / …) inside a ``burst_ms`` (50 ms) window is a multi-item drop resolving: only
     the **last** frame of the run survives, because it carries the fully updated
     environment state (e.g. ``ground_items`` already shows every drop);
   * explicit player decisions (``moved``, ``used_skill``, ``attacked``,
     ``selected_target``, ``pickup``, ``used_item``) are preserved unless they are
     true sub-millisecond duplicate spam (identical ``action_type`` *and*
     ``action_raw`` within ``debounce_ms``);
   * sessions that contain no combat interaction at all — no damage dealt or
     taken, no skill used, no kill and no pickup — are dropped entirely, so a bare
     target selection never reaches the dataset;
   * the surviving sessions are re-indexed to a dense ``0..K-1`` range ordered by
     session start time.

Routing: DuckDB performs the set-based array cleaning (it can unnest, join and
re-aggregate a million list elements without ever materialising Python objects) and
resolves the hostile NPC ids of the datapack, pandas / NumPy perform the row-level
time-series logic and the combat state machine, and PyArrow performs the
schema-enforced IO.

No numeric normalisation or feature scaling is applied — HP, MP, coordinates,
timestamps and every other value stay exactly as logged. Only one column is
appended to the schema: the ``session_id`` metadata column.

Usage::

    uv run python tools/clean_playeractions.py
    uv run python tools/clean_playeractions.py --input ../l2jc6_playeractions.parquet --output ../l2jc6_playeractions_cleaned.parquet --exclude-characters admin


Created with deepseek-v4-flash    
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

# Repository root, so the default tables are found from any CWD.
_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_INPUT: Final = _REPO_ROOT / "l2jc6_playeractions.parquet"
DEFAULT_NPCS: Final = _REPO_ROOT / "l2jc6_npcs.parquet"
DEFAULT_CHARACTERS: Final = _REPO_ROOT / "l2jc6_characters.parquet"
DEFAULT_OUTPUT: Final = _REPO_ROOT / "l2jc6_playeractions_cleaned.parquet"

TIMESTAMP_FORMAT: Final = "%Y.%m.%d %H:%M:%S,%f"
SESSION_COLUMN: Final = "session_id"
ENV_COLUMN: Final = "nearby_objects"

# --- Action classification -------------------------------------------------
# Dropped in stage 1b: trade, dialogs, crafting, private stores and social
# actions are never part of a combat engagement.
NON_COMBAT_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "buy",
        "sell",
        "trade",
        "talk",
        "dialog",
        "npc_dialog",
        "craft",
        "crafting",
        "enchant",
        "private_store",
        "private_store_buy",
        "private_store_sell",
        "social",
        "social_action",
        "emote",
    }
)

# Collapsed in stage 2: only the first frame of a contiguous run survives.
IDLE_ACTIONS: Final[frozenset[str]] = frozenset({"idle", "none"})

# --- Combat sessionization (stage 1) ---------------------------------------
# The three actions the combat state machine keys on.
TARGET_ACTION: Final = "selected_target"
KILL_ACTION: Final = "killed"
PICKUP_ACTION: Final = "pickup"

# A session only survives stage 4b when it holds at least one *engagement* row.
# A bare target selection is not enough: ``inflicted_damage`` / ``took_damage``
# count, because the client logs the auto-attack damage without emitting a
# separate ``attacked`` line for every swing.
ENGAGEMENT_ACTIONS: Final[frozenset[str]] = frozenset(
    {"attacked", "used_skill", "pickup", "killed", "inflicted_damage", "took_damage"}
)

# Player-facing combat actions (also the "combat share" statistic).
COMBAT_ACTIONS: Final[frozenset[str]] = frozenset({"attacked", "used_skill", "pickup", "selected_target"})

# Damage resolution frames: the RL reward is attributed to them.
DAMAGE_ACTIONS: Final[frozenset[str]] = frozenset({"inflicted_damage", "took_damage"})

# Breaks an open engagement before a kill (case C). The interrupting row itself is
# never kept: the session is truncated at the last combat frame *before* it, so a
# death / teleport is not part of the episode.
INTERRUPT_ACTIONS: Final[frozenset[str]] = frozenset({"died", "teleported", "resurrected", "logged_in"})

# Server-side / system frames collapsed in stage 4a. A rapid run of these is the
# multi-item-drop pattern: one kill resolves into several ``killed`` lines (each
# carrying the next dropped item) 1-50 ms apart, so the state frames are nearly
# identical and only ``ground_items`` grows. Only the LAST frame of such a run is
# kept, because it holds the fully updated environment state (all drops counted).
#
# Explicit player decisions (``moved``, ``used_skill``, ``attacked``,
# ``selected_target``, ``pickup``, ``used_item``) are deliberately absent, so the
# burst rule can never remove them. ``learned_skill`` / ``autouse_*`` are absent
# too: they are discrete progression events, not part of a kill resolution, and
# swallowing the drop frame that precedes them would lose the reward signal.
#
# ``inflicted_damage`` / ``took_damage`` / ``died`` are excluded as well. They are
# not bookkeeping around a kill, they are the damage and survival outcomes the RL
# reward is attributed to, and their ``action_raw`` carries the damage value that
# the next frame's HP alone does not reveal. A damage frame therefore breaks a
# burst instead of being swallowed by it, so ``inflicted_damage`` immediately
# followed by ``killed`` keeps both frames (only the multi-item ``killed`` run
# collapses).
SYSTEM_EVENT_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "killed",
        "drop",
        "dropped",
        "item_drop",
        "loot",
        "looted",
        "spoil",
        "spoiled",
        "effect_applied",
        "effect_removed",
        "skills_updated",
    }
)

# Frames that may legitimately *end* a session in the aborted case C. A trailing
# stretch of pure ``moved`` navigation is deliberately excluded, so the tail of a
# travel path is never swallowed by the engagement it follows.
SESSION_BODY_ACTIONS: Final[frozenset[str]] = frozenset(
    COMBAT_ACTIONS | DAMAGE_ACTIONS | SYSTEM_EVENT_ACTIONS | {KILL_ACTION}
)

# --- Environment object classification (stage 3) ---------------------------
# Explicit hostile NPC types: an object whose ``type`` is one of these is kept
# without a datapack lookup. The bare ``Monster`` spelling is included here, but
# the *lower-case* ``monster`` the client prints is intentionally relegated to
# ``GENERIC_NPC_OBJECT_TYPES`` so it is still validated against the NPC table
# (that spelling also covers town NPCs the client reports as ``Monster:``).
HOSTILE_NPC_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Monster",
        "RaidBoss",
        "GrandBoss",
        "FestivalMonster",
        "EventMonster",
        "FeedableBeast",
        "RiftInvader",
        "L2Monster",
        "L2RaidBoss",
        "L2GrandBoss",
        "L2FestivalMonster",
        "L2EventMonster",
        "raidboss",
        "grandboss",
        "festivalmonster",
        "eventmonster",
        "feedablebeast",
        "riftinvader",
        "l2monster",
        "l2raidboss",
        "l2grandboss",
    }
)

# Loot / dropped items are always relevant for the reward signal.
LOOT_OBJECT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Item",
        "L2Item",
        "Loot",
        "DroppedItem",
        "GroundItem",
        "item",
        "l2item",
        "loot",
        "dropped_item",
        "ground_item",
    }
)

# Other players provide the PvP / co-op context.
PLAYER_OBJECT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Player",
        "L2PcInstance",
        "Pc",
        "L2Playable",
        "player",
        "l2pcinstance",
        "pc",
        "playable",
    }
)

# Ambiguous / generic client spellings that must be resolved against the NPC
# datapack before they can be treated as hostile.
GENERIC_NPC_OBJECT_TYPES: Final[frozenset[str]] = frozenset({"", "monster", "npc", "l2npc", "creature"})

# The Arrow type of one ``nearby_objects`` element, used to rebuild the list.
ENV_STRUCT_SQL: Final = "STRUCT(type VARCHAR, name VARCHAR, id BIGINT, lvl BIGINT, distance DOUBLE)"

# ``selected_target`` lines carry the target name up to the character-status block
# (or, for a monster, the ``| HP:`` block that follows it). Everything after that —
# notably the ``[Env: …]`` trailer — must never be searched for ``NPC_ID``.
TARGET_HEADER_REGEX: Final = r"^\[selected target\]: (.*?)(?:\[CharStatus|\| HP:|\[Env:)"
TARGET_NPC_ID_REGEX: Final = r"\(NPC_ID:(\d+)"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Tunables of every cleaning stage (defaults follow the task specification)."""

    gap_seconds: float = 15.0
    # Kept per row from the environment vector after the distance filter.
    top_n_objects: int = 10
    max_monster_distance: float = 3000.0
    # Identical spam events closer than this (milliseconds) collapse into one.
    debounce_ms: float = 1.0
    # A run of system frames whose consecutive gaps are within this window
    # (milliseconds) is one multi-item-drop burst: only its last frame survives.
    burst_ms: float = 50.0
    # Characters (users) to drop completely, e.g. GM / admin / test accounts.
    exclude_characters: frozenset[str] = field(default_factory=frozenset)
    # The same characters as ``character_id``s, resolved from the character table
    # (the action table stores ids, not names).
    exclude_character_ids: frozenset[int] = field(default_factory=frozenset)
    non_combat_actions: frozenset[str] = field(default_factory=lambda: NON_COMBAT_ACTIONS)
    idle_actions: frozenset[str] = field(default_factory=lambda: IDLE_ACTIONS)
    combat_actions: frozenset[str] = field(default_factory=lambda: COMBAT_ACTIONS)
    damage_actions: frozenset[str] = field(default_factory=lambda: DAMAGE_ACTIONS)
    engagement_actions: frozenset[str] = field(default_factory=lambda: ENGAGEMENT_ACTIONS)
    body_actions: frozenset[str] = field(default_factory=lambda: SESSION_BODY_ACTIONS)
    interrupt_actions: frozenset[str] = field(default_factory=lambda: INTERRUPT_ACTIONS)
    system_actions: frozenset[str] = field(default_factory=lambda: SYSTEM_EVENT_ACTIONS)
    target_action: str = TARGET_ACTION
    kill_action: str = KILL_ACTION
    pickup_action: str = PICKUP_ACTION
    # Ids of the hostile NPCs of the datapack, resolved by :func:`load_hostile_npc_ids`
    # and shared with the DuckDB environment filter, so "hostile target" means the
    # same thing in stage 1 and stage 3.
    hostile_npc_ids: frozenset[int] = field(default_factory=frozenset)
    hostile_npc_types: frozenset[str] = field(default_factory=lambda: HOSTILE_NPC_TYPES)
    loot_object_types: frozenset[str] = field(default_factory=lambda: LOOT_OBJECT_TYPES)
    player_object_types: frozenset[str] = field(default_factory=lambda: PLAYER_OBJECT_TYPES)
    generic_object_types: frozenset[str] = field(default_factory=lambda: GENERIC_NPC_OBJECT_TYPES)


@dataclass(slots=True)
class CombatSession:
    """One extracted engagement: the row span ``[start, end]`` and its outcome."""

    raw_id: int  # id before the dense renumbering
    character_id: int
    start: int  # inclusive position in the sorted, per-character timeline
    end: int  # inclusive position in the sorted, per-character timeline
    start_ts: np.datetime64
    has_kill: bool
    has_pickup: bool
    interrupted: bool  # case C: the break was an interruption, not a kill

    @property
    def frames(self) -> int:
        """Number of raw timeline frames covered by the engagement."""
        return self.end - self.start + 1


@dataclass(slots=True)
class RowDecisions:
    """Per-row outcome of the filtering stages, aligned with the input order."""

    keep: NDArray[np.bool_]  # bool[rows] — survives every stage
    session_id: NDArray[np.int64]  # int64[rows] — dense 0..K-1, -1 for dropped rows
    raw_session_id: NDArray[np.int64]  # int64[rows] — ids before renumbering, for reporting
    sessions: tuple[CombatSession, ...] = ()  # engagement metadata, ordered by start time


@dataclass(slots=True)
class CleaningReport:
    """Counters collected while cleaning, for the final summary."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_excluded: int = 0
    characters_excluded: tuple[int, ...] = ()
    # Stage 1: combat sessionization.
    targets_total: int = 0
    targets_item_ignored: int = 0
    targets_non_hostile_ignored: int = 0
    targets_refresh: int = 0
    interrupts_total: int = 0
    sessions_extracted: int = 0
    sessions_with_kill: int = 0
    sessions_with_pickup: int = 0
    sessions_aborted: int = 0
    dropped_out_of_session: int = 0
    # Stages 1b / 2 / 4a.
    dropped_non_combat: int = 0
    dropped_idle: int = 0
    dropped_burst: int = 0
    dropped_debounced: int = 0
    # Stage 4b / 4c.
    dropped_empty_sessions: int = 0
    sessions_total: int = 0
    sessions_dropped: int = 0
    sessions_out: int = 0
    sessions_without_damage: int = 0
    session_frames_min: int = 0
    session_frames_max: int = 0
    session_frames_mean: float = 0.0
    env_entries_before: int = 0
    env_entries_after: int = 0
    env_rows_emptied: int = 0
    env_rows_null: int = 0


# --- Stage 3: environment array cleaning (DuckDB) --------------------------


def _sql_string_list(values: frozenset[str]) -> str:
    """Render a set of strings as a SQL ``IN``-list body (single quotes escaped)."""
    return ", ".join("'" + value.replace("'", "''") + "'" for value in sorted(values))


def load_hostile_npc_ids(npcs: Path, cfg: CleaningConfig) -> frozenset[int]:
    """
    Resolve the hostile NPC ids of the datapack (``Monster``, ``RaidBoss``, …).

    This is the single source of truth for "hostile": stage 1 uses it to decide
    whether a ``selected_target`` row starts an engagement, and stage 3 passes the
    very same set to DuckDB to filter the environment array. Resolving it once in
    Python keeps the two stages from drifting apart.
    """
    table = pq.read_table(npcs, columns=["id", "type"])
    ids = table.column("id").to_pylist()
    types = table.column("type").to_pylist()
    return frozenset(
        int(npc_id) for npc_id, npc_type in zip(ids, types) if npc_id is not None and npc_type in cfg.hostile_npc_types
    )


def clean_environment(con: duckdb.DuckDBPyConnection, actions: Path, cfg: CleaningConfig) -> pa.Table:
    """
    Return the action table with ``nearby_objects`` cleaned, in the input row order.

    The nested array is unnested and each element is classified as hostile NPC,
    loot / dropped item, other player or irrelevant. Hostile NPCs whose client
    spelling is ambiguous (the lower-case ``monster``) are checked against
    ``cfg.hostile_npc_ids`` — the same set stage 1 uses to recognise a combat
    target — so town NPCs reported as ``Monster:`` are still removed. The kept
    objects are filtered by ``cfg.max_monster_distance``, ranked by distance and
    re-aggregated into at most ``cfg.top_n_objects`` elements per row. Rows whose
    array was originally null stay null; rows whose entries are all filtered out
    become an empty list.
    """
    columns = [field.name for field in pq.read_schema(actions)]
    hostile_types = _sql_string_list(cfg.hostile_npc_types)
    loot_types = _sql_string_list(cfg.loot_object_types)
    player_types = _sql_string_list(cfg.player_object_types)
    generic_types = _sql_string_list(cfg.generic_object_types)

    con.execute(
        """
        CREATE OR REPLACE TABLE hostile_npc_ids AS
        SELECT unnest($ids) AS id
        """,
        {"ids": np.fromiter(sorted(cfg.hostile_npc_ids), dtype=np.int64)},
    )

    # ``file_row_number`` is the position inside the Parquet file, so it is the
    # row order that has to be preserved in the output.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE rows AS
        SELECT file_row_number AS rid, * FROM read_parquet('{actions.as_posix()}', file_row_number = true) AS c
        """
    )
    row_columns = ["rid", *columns]

    con.execute(
        f"""
        CREATE OR REPLACE TABLE cleaned_env AS
        WITH exploded AS (
            SELECT r.rid, e.type, e.name, e.id, e.lvl, e.distance
            FROM rows r, UNNEST(r.{ENV_COLUMN}) AS t(e)
            WHERE e.distance <= {cfg.max_monster_distance}
        ),
        relevant AS (
            SELECT *
            FROM exploded
            WHERE type IN ({loot_types})
               OR type IN ({player_types})
               OR type IN ({hostile_types})
               OR (type IN ({generic_types}) AND id IN (SELECT id FROM hostile_npc_ids))
        ),
        ranked AS (
            SELECT rid,
                   struct_pack(type := type, name := name, id := id, lvl := lvl, distance := distance) AS obj,
                   row_number() OVER (PARTITION BY rid ORDER BY distance ASC, id ASC) AS rn
            FROM relevant
        )
        SELECT rid, list(obj ORDER BY obj.distance ASC, obj.id ASC) AS objs
        FROM ranked
        WHERE rn <= {cfg.top_n_objects}
        GROUP BY rid
        """
    )

    # Rebuild the table in its original column order, replacing only the env array.
    output_columns = ", ".join(
        f'r."{name}"'
        if name != ENV_COLUMN
        else (
            f'CASE WHEN r."{ENV_COLUMN}" IS NULL THEN NULL '
            f'ELSE COALESCE(t.objs, CAST([] AS {ENV_STRUCT_SQL}[])) END AS "{name}"'
        )
        for name in row_columns
    )
    table = con.execute(
        f"""
        SELECT {output_columns}
        FROM rows r
        LEFT JOIN cleaned_env t ON t.rid = r.rid
        ORDER BY r.rid
        """
    ).to_arrow_table()

    return table.select(columns)


# --- Stages 1, 2 and 4: row-level decisions (pandas / NumPy) ---------------


def load_light(actions: Path) -> pd.DataFrame:
    """Load only the columns needed for the row-level stages (plus a parsed timestamp)."""
    light = pd.read_parquet(actions, columns=["character_id", "action_type", "action_raw", "timestamp"])
    light["ts"] = pd.to_datetime(light["timestamp"], format=TIMESTAMP_FORMAT)
    return light


def extract_target_npc_ids(light: pd.DataFrame, cfg: CleaningConfig) -> NDArray[np.int64]:
    """
    Return the NPC id of every ``selected_target`` row, ``-1`` for anything else.

    Only the *target header* of the line is inspected::

        [selected target]: Venomous Spider (NPC_ID:20038, Lvl:18) | HP:0.0/476.0 …  -> 20038
        [selected target]: Superior Herb of Life [CharStatus: HP:607.0/607 …]       -> -1

    The ``[Env: …]`` trailer of the same line also lists ``(NPC_ID:…)`` for every
    nearby monster, so a naive search would turn each item pickup into a monster
    target — the header has to be cut off first. Item targets (``Adena``,
    ``Herb of Life``, ``Recipe: …``) return ``-1``: selecting loot is not combat.
    """
    npc_id = np.full(len(light), -1, dtype=np.int64)
    target_rows = np.flatnonzero(light["action_type"].to_numpy() == cfg.target_action)
    if target_rows.size == 0:
        return npc_id

    raw = pd.Series(light["action_raw"].to_numpy()[target_rows], dtype="string")
    header = raw.str.extract(TARGET_HEADER_REGEX, expand=False).fillna(raw)
    found = pd.to_numeric(header.str.extract(TARGET_NPC_ID_REGEX, expand=False), errors="coerce").to_numpy()
    npc_id[target_rows] = np.where(np.isnan(found), -1, found).astype(np.int64)
    return npc_id


def _combat_group_spans(
    action: NDArray[np.str_],
    ts: NDArray[np.datetime64],
    target_npc: NDArray[np.int64],
    hostile_ids: NDArray[np.int64],
    cfg: CleaningConfig,
    offset: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_], int, int]:
    """
    Extract the combat spans of one character's timeline (fully vectorised).

    ``action`` / ``ts`` / ``target_npc`` hold one character's rows in chronological
    order and ``offset`` is the absolute position of that slice in the globally
    sorted array, so the returned spans map straight back onto input rows. Returns
    ``(starts, ends, has_kill, has_pickup, refreshes, interruptions)`` with
    inclusive bounds.
    """
    no_spans = np.zeros(0, dtype=np.int64)
    no_flags = np.zeros(0, dtype=bool)
    size = action.size

    is_target = (action == cfg.target_action) & np.isin(target_npc, hostile_ids)
    target_idx = np.flatnonzero(is_target)
    if target_idx.size == 0:
        return no_spans, no_spans, no_flags, no_flags, 0, 0

    is_kill = action == cfg.kill_action
    is_pickup = action == cfg.pickup_action
    is_body = np.isin(action, tuple(cfg.body_actions))

    delta = np.zeros(size, dtype=np.float64)
    delta[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    is_interrupt = (delta > cfg.gap_seconds) | np.isin(action, tuple(cfg.interrupt_actions))

    # --- engagement starts --------------------------------------------------
    # A repeated selection of the target the session is already fighting, with no
    # kill and no interruption in between, is a client-side refresh: it keeps the
    # engagement open instead of starting a new one. Case C speaks of a *new*
    # target, and treating refreshes as restarts would shatter one fight into
    # hundreds of one-frame sessions.
    previous, current = target_idx[:-1], target_idx[1:]
    kill_before = np.concatenate(([0], np.cumsum(is_kill)))
    interrupt_before = np.concatenate(([0], np.cumsum(is_interrupt)))
    refresh = (
        (target_npc[previous] == target_npc[current])
        & (kill_before[current] - kill_before[previous + 1] == 0)
        & (interrupt_before[current] - interrupt_before[previous + 1] == 0)
    )
    starts = target_idx[np.concatenate(([True], ~refresh))]

    # --- what closes each engagement ---------------------------------------
    # The next engagement (case B/C) or the first interruption, whichever comes
    # first. ``searchsorted(..., side="right")`` skips any interruption sitting on
    # the starting row itself, and an empty interrupt list simply means "the
    # timeline, not an interruption, closes the engagement".
    next_start = np.concatenate((starts[1:], [size]))
    interrupt_idx = np.flatnonzero(is_interrupt)
    if interrupt_idx.size:
        position = np.searchsorted(interrupt_idx, starts, side="right")
        next_interrupt = np.where(
            position < interrupt_idx.size,
            interrupt_idx[np.clip(position, 0, interrupt_idx.size - 1)],
            size,
        )
    else:
        next_interrupt = np.full(starts.size, size, dtype=np.int64)
    stop = np.minimum(next_start, next_interrupt)

    # Case A / B: a kill closes the engagement — on the first pickup that follows
    # it, or on the last frame of the multi-drop kill run when no pickup comes.
    kill_idx = np.flatnonzero(is_kill)
    has_kill = np.zeros(starts.size, dtype=bool)
    kill_end = np.zeros(starts.size, dtype=np.int64)
    if kill_idx.size:
        kill_lo = np.searchsorted(kill_idx, starts)
        kill_hi = np.searchsorted(kill_idx, stop)
        has_kill = kill_lo < kill_hi
        first_kill = kill_idx[np.clip(kill_lo, 0, kill_idx.size - 1)]
        last_kill = kill_idx[np.clip(kill_hi - 1, 0, kill_idx.size - 1)]
    else:
        first_kill = last_kill = np.zeros(starts.size, dtype=np.int64)

    pickup_idx = np.flatnonzero(is_pickup)
    has_pickup = np.zeros(starts.size, dtype=bool)
    pickup_end = np.zeros(starts.size, dtype=np.int64)
    if pickup_idx.size:
        pickup_lo = np.searchsorted(pickup_idx, first_kill, side="right")
        pickup_hi = np.searchsorted(pickup_idx, stop)
        has_pickup = has_kill & (pickup_lo < pickup_hi)
        pickup_end = pickup_idx[np.clip(pickup_lo, 0, pickup_idx.size - 1)]
    if kill_idx.size:
        kill_end = np.where(has_pickup, pickup_end, last_kill)

    # Case C: aborted engagement, truncated at the last combat frame before the
    # interruption — never at a trailing stretch of pure navigation, so the tail of
    # a travel path is not swallowed by the engagement it follows.
    body_idx = np.flatnonzero(is_body)
    if body_idx.size:
        body_hi = np.searchsorted(body_idx, stop)
        abort_end = body_idx[np.clip(body_hi - 1, 0, body_idx.size - 1)]
    else:  # unreachable with the default ``body_actions`` (the start row is a body frame)
        abort_end = starts.copy()

    ends = np.where(has_kill, kill_end, abort_end)
    return (
        starts + offset,
        ends + offset,
        has_kill,
        has_pickup,
        int(refresh.sum()),
        int(is_interrupt.sum()),
    )


def assign_combat_sessions(
    light: pd.DataFrame,
    eligible: NDArray[np.bool_],
    cfg: CleaningConfig,
    report: CleaningReport,
) -> tuple[NDArray[np.int64], tuple[CombatSession, ...]]:
    """
    Slice every character's timeline into ``selected_target -> end`` engagements.

    Boundaries are detected on the raw *unfiltered* timeline, so a later filtering
    step can neither fake nor hide the end of an engagement. Each row receives the
    id of the engagement covering it, or ``-1`` when it falls outside every
    engagement — idle roaming, city walking, trading, long travel paths — and is
    therefore discarded. Ids are numbered globally in chronological order, so they
    grow with the recording timeline and are unique across characters.
    """
    session = np.full(len(light), -1, dtype=np.int64)
    rows = np.flatnonzero(eligible).astype(np.int64)
    if rows.size == 0:
        return session, ()

    raw_action = light["action_type"].to_numpy()
    target_npc = extract_target_npc_ids(light, cfg)
    is_target_row = raw_action == cfg.target_action
    hostile_ids = np.fromiter(sorted(cfg.hostile_npc_ids), dtype=np.int64)

    report.targets_total = int(is_target_row.sum())
    report.targets_item_ignored = int((is_target_row & (target_npc < 0)).sum())
    report.targets_non_hostile_ignored = int(
        (is_target_row & (target_npc >= 0) & ~np.isin(target_npc, hostile_ids)).sum()
    )

    character = light["character_id"].to_numpy()[rows]
    action = raw_action[rows]
    timestamp = light["ts"].to_numpy()[rows]
    npc = target_npc[rows]

    order = np.lexsort((rows, timestamp, character))
    rows, character, action, timestamp, npc = (
        rows[order],
        character[order],
        action[order],
        timestamp[order],
        npc[order],
    )

    # ``lexsort`` groups a character's rows contiguously, so the group edges are
    # exactly where the character changes.
    edges = np.concatenate(([0], np.flatnonzero(character[1:] != character[:-1]) + 1, [rows.size]))

    starts: list[NDArray[np.int64]] = []
    ends: list[NDArray[np.int64]] = []
    kills: list[NDArray[np.bool_]] = []
    pickups: list[NDArray[np.bool_]] = []
    refreshes = 0
    interruptions = 0
    for begin, stop_at in zip(edges[:-1], edges[1:]):
        group = _combat_group_spans(
            action[begin:stop_at],
            timestamp[begin:stop_at],
            npc[begin:stop_at],
            hostile_ids,
            cfg,
            int(begin),
        )
        group_starts, group_ends, group_kills, group_pickups, refresh, interrupt = group
        refreshes += refresh
        interruptions += interrupt
        if group_starts.size:
            starts.append(group_starts)
            ends.append(group_ends)
            kills.append(group_kills)
            pickups.append(group_pickups)

    report.targets_refresh = refreshes
    report.interrupts_total = interruptions
    if not starts:
        return session, ()

    start_index = np.concatenate(starts)
    end_index = np.concatenate(ends)
    has_kill = np.concatenate(kills)
    has_pickup = np.concatenate(pickups)

    # Chronological ids across characters.
    rank = np.empty(start_index.size, dtype=np.int64)
    rank[np.argsort(timestamp[start_index], kind="stable")] = np.arange(start_index.size, dtype=np.int64)

    sessions: list[CombatSession] = []
    for index in range(start_index.size):
        start, end = int(start_index[index]), int(end_index[index])
        session[rows[start : end + 1]] = rank[index]
        sessions.append(
            CombatSession(
                raw_id=int(rank[index]),
                character_id=int(character[start]),
                start=start,
                end=end,
                start_ts=timestamp[start],
                has_kill=bool(has_kill[index]),
                has_pickup=bool(has_pickup[index]),
                interrupted=not bool(has_kill[index]),
            )
        )

    sessions.sort(key=lambda engagement: engagement.raw_id)
    report.sessions_extracted = len(sessions)
    report.sessions_with_kill = sum(1 for engagement in sessions if engagement.has_kill)
    report.sessions_with_pickup = sum(1 for engagement in sessions if engagement.has_pickup)
    report.sessions_aborted = sum(1 for engagement in sessions if engagement.interrupted)
    report.sessions_total = len(sessions)
    return session, tuple(sessions)


def collapse_idle_streaks(light: pd.DataFrame, keep: NDArray[np.bool_], cfg: CleaningConfig) -> NDArray[np.bool_]:
    """
    Keep only the first frame of every contiguous idle streak of a character.

    A streak is a maximal run of consecutive ``IDLE`` / ``NONE`` records of one
    character on the (already filtered) timeline. Every record of the run except
    the first is dropped, whatever the run's duration — no duration threshold is
    involved, so no multi-second idle cluster is left behind. The first frame
    keeps its original timestamp, which preserves the timing information.
    """
    rows = np.flatnonzero(keep).astype(np.int64)
    if rows.size == 0:
        return keep

    action = light["action_type"].to_numpy()[rows]
    idle = np.isin(action, tuple(cfg.idle_actions))
    if not idle.any():
        return keep

    character_ids = light["character_id"].to_numpy()[rows]
    order = np.lexsort((rows, light["ts"].to_numpy()[rows], character_ids))
    rows, action, character = rows[order], action[order], character_ids[order]
    idle = idle[order]

    # A row belongs to a streak continuation when the previous row of the same
    # character is idle as well.
    continuation = np.zeros(rows.size, dtype=bool)
    continuation[1:] = idle[:-1] & (character[1:] == character[:-1])
    dropped = idle & continuation
    if not dropped.any():
        return keep

    result = keep.copy()
    result[rows[dropped]] = False
    return result


def collapse_system_bursts(
    light: pd.DataFrame,
    keep: NDArray[np.bool_],
    sessions: NDArray[np.int64],
    cfg: CleaningConfig,
) -> NDArray[np.bool_]:
    """
    Keep only the last frame of every rapid server/system burst (multi-item drops).

    A burst is a maximal run of consecutive surviving records of the **same
    character and session** whose action types are all in ``cfg.system_actions``
    and where every consecutive gap is at most ``cfg.burst_ms``. That is exactly
    the shape a kill with several drops produces: one ``killed`` line per dropped
    item, 1-50 ms apart, each with an almost identical state and a growing
    ``ground_items`` counter. Only the final frame of the run carries the fully
    resolved environment, so all earlier frames of the run are dropped.

    Player-initiated records (``moved``, ``used_skill``, ``attacked``,
    ``selected_target``, ``pickup``, ``used_item``, …) are not in
    ``cfg.system_actions``; they break the run and are never removed here. A
    non-positive ``cfg.burst_ms`` disables the aggregation.
    """
    if cfg.burst_ms <= 0.0:
        return keep

    rows = np.flatnonzero(keep).astype(np.int64)
    if rows.size == 0:
        return keep

    character = light["character_id"].to_numpy()[rows]
    action = light["action_type"].to_numpy()[rows]
    timestamp = light["ts"].to_numpy()[rows]
    session = sessions[rows]

    order = np.lexsort((rows, timestamp, character))
    rows, character, action, timestamp, session = (
        rows[order],
        character[order],
        action[order],
        timestamp[order],
        session[order],
    )

    system = np.isin(action, tuple(cfg.system_actions))
    delta = np.zeros(rows.size, dtype=np.float64)
    delta[1:] = (timestamp[1:] - timestamp[:-1]) / np.timedelta64(1, "s")
    window = cfg.burst_ms / 1000.0

    # A frame continues the current burst only when it and its predecessor are
    # system frames of the same character/session within the window.
    continues = np.zeros(rows.size, dtype=bool)
    continues[1:] = (
        system[1:]
        & system[:-1]
        & (character[1:] == character[:-1])
        & (session[1:] == session[:-1])
        & (delta[1:] >= 0.0)
        & (delta[1:] <= window)
    )

    # Keeping the last frame is equivalent to dropping every frame whose successor
    # continues the same burst; the final frame of a run has no successor.
    drop = np.zeros(rows.size, dtype=bool)
    drop[:-1] = continues[1:]
    if not drop.any():
        return keep

    result = keep.copy()
    result[rows[drop]] = False
    return result


def debounce_micro_events(
    light: pd.DataFrame,
    keep: NDArray[np.bool_],
    sessions: NDArray[np.int64],
    cfg: CleaningConfig,
) -> NDArray[np.bool_]:
    """
    Collapse true sub-millisecond duplicate spam into a single frame.

    This is the narrow companion of :func:`collapse_system_bursts` and now mostly
    protects the explicit player records the burst rule refuses to touch. Two
    consecutive records of one character/session are duplicates when they share
    the same ``action_type`` *and* the same ``action_raw`` and are closer than
    ``cfg.debounce_ms`` (sub-millisecond by default). Only the first frame of such
    a run is kept. A non-positive ``cfg.debounce_ms`` disables the debounce.
    """
    if cfg.debounce_ms <= 0.0:
        return keep

    rows = np.flatnonzero(keep).astype(np.int64)
    if rows.size == 0:
        return keep

    character = light["character_id"].to_numpy()[rows]
    action = light["action_type"].to_numpy()[rows]
    raw = light["action_raw"].fillna("").to_numpy()[rows]
    timestamp = light["ts"].to_numpy()[rows]
    session = sessions[rows]

    order = np.lexsort((rows, timestamp, character))
    rows, character, action, raw, timestamp, session = (
        rows[order],
        character[order],
        action[order],
        raw[order],
        timestamp[order],
        session[order],
    )

    window = cfg.debounce_ms / 1000.0
    delta = np.zeros(rows.size, dtype=np.float64)
    delta[1:] = (timestamp[1:] - timestamp[:-1]) / np.timedelta64(1, "s")

    duplicate = np.zeros(rows.size, dtype=bool)
    duplicate[1:] = (
        (character[1:] == character[:-1])
        & (session[1:] == session[:-1])
        & (action[1:] == action[:-1])
        & (raw[1:] == raw[:-1])
        & (delta[1:] >= 0.0)
        & (delta[1:] < window)
    )
    if not duplicate.any():
        return keep

    result = keep.copy()
    result[rows[duplicate]] = False
    return result


def combat_by_session(
    light: pd.DataFrame,
    sessions: NDArray[np.int64],
    keep: NDArray[np.bool_],
    cfg: CleaningConfig,
) -> NDArray[np.bool_]:
    """
    Return a per-session flag telling whether the session holds any combat interaction.

    A session only qualifies when an *engagement* row survived every filter: damage
    dealt or taken, a skill, a kill or a loot pickup. A bare ``selected_target`` —
    the player initiated a fight that never happened — is not combat interaction and
    keeps no session alive.
    """
    rows = np.flatnonzero(sessions >= 0)
    if rows.size == 0:
        return np.zeros(0, dtype=bool)

    has_combat = np.zeros(int(sessions.max()) + 1, dtype=bool)
    is_combat = np.isin(light["action_type"].to_numpy()[rows], tuple(cfg.engagement_actions)) & keep[rows]
    np.logical_or.at(has_combat, sessions[rows], is_combat)
    return has_combat


def keep_characters(light: pd.DataFrame, cfg: CleaningConfig) -> tuple[NDArray[np.bool_], tuple[int, ...]]:
    """
    Return the mask of rows whose character is not excluded, plus the ids matched.

    The names given on the command line are resolved to ``character_id``s against
    ``l2jc6_characters.parquet`` by :func:`resolve_excluded_characters`, because the
    action table stores only ids. The resolution is case-insensitive, so a GM / test
    account is caught whatever the case used on the command line (``admin`` vs
    ``Admin``).
    """
    if not cfg.exclude_character_ids:
        return np.ones(len(light), dtype=bool), ()

    excluded = light["character_id"].isin(cfg.exclude_character_ids).to_numpy()
    matched = light["character_id"].to_numpy()[excluded]
    return ~excluded, tuple(sorted({int(value) for value in matched if value == value}))


def load_character_names(characters: Path) -> dict[int, str]:
    """Return the ``character_id`` -> name map from the character dimension table."""
    table = pq.read_table(characters, columns=["character_id", "name"])
    return {int(row["character_id"]): row["name"] for row in table.to_pylist()}


def resolve_excluded_characters(cfg: CleaningConfig, names_by_id: dict[int, str]) -> CleaningConfig:
    """Resolve ``--exclude-characters`` names to ``character_id``s, case-insensitively."""
    wanted = {name.strip().lower() for name in cfg.exclude_characters if name.strip()}
    if not wanted:
        return cfg

    unknown = wanted - {(name or "").lower() for name in names_by_id.values()}
    if unknown:
        print(f"[Warning] excluded character name(s) not found in the dataset: {sorted(unknown)}")

    ids = frozenset(character_id for character_id, name in names_by_id.items() if (name or "").lower() in wanted)
    return replace(cfg, exclude_character_ids=ids)


def decide_rows(light: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport) -> RowDecisions:
    """
    Run the row-level stages and collect the counters they produce.

    Ordering matters: stage 0 excludes whole characters, stage 1 computes the combat
    session ids on the still complete timeline and only then discards every row that
    falls outside an engagement, stage 2 collapses idle streaks, stage 4 aggregates
    rapid system bursts and micro-event spam, drops sessions without combat
    interaction and renumbers the survivors densely.
    """
    report.rows_in = len(light)

    # Stage 0: drop excluded characters (GM / admin / test accounts) entirely, so
    # their records never take part in session numbering either.
    keep, report.characters_excluded = keep_characters(light, cfg)
    report.dropped_excluded = int((~keep).sum())

    # Stage 1: slice the FULL timeline into combat engagements before any row is
    # dropped, so a run of non-combat records can neither fake nor hide the end of
    # an engagement.
    raw_session, sessions = assign_combat_sessions(light, keep, cfg, report)

    # Stage 1b: everything outside an engagement — idle roaming, city walking,
    # trading, long travel paths — is discarded.
    in_session = keep & (raw_session >= 0)
    report.dropped_out_of_session = int(keep.sum() - in_session.sum())
    keep = in_session

    # Stage 1c: drop the non-combat actions that fall inside an engagement.
    non_combat = light["action_type"].isin(cfg.non_combat_actions).to_numpy() & keep
    report.dropped_non_combat = int(non_combat.sum())
    keep &= ~non_combat

    # Stage 2: collapse idle streaks (keep only the first frame of each run).
    surviving = int(keep.sum())
    keep = collapse_idle_streaks(light, keep, cfg)
    report.dropped_idle = surviving - int(keep.sum())

    # Stage 4a: aggregate rapid system bursts (multi-item drops keep the last
    # frame), then remove any remaining true sub-millisecond duplicate spam.
    surviving = int(keep.sum())
    keep = collapse_system_bursts(light, keep, raw_session, cfg)
    report.dropped_burst = surviving - int(keep.sum())

    surviving = int(keep.sum())
    keep = debounce_micro_events(light, keep, raw_session, cfg)
    report.dropped_debounced = surviving - int(keep.sum())

    # Stage 4b: a session survives only if it holds combat interaction. Every
    # engagement starts on a target selection, so a session whose only surviving
    # frame is that selection — the fight never actually happened — is dropped here.
    has_combat = combat_by_session(light, raw_session, keep, cfg)
    report.sessions_dropped = int(has_combat.size - int(has_combat.sum()))

    # ``has_combat`` is empty when the file holds no engagement at all; indexing it
    # with the clipped raw ids would then raise.
    session_ok = (
        np.where(raw_session >= 0, has_combat[np.clip(raw_session, 0, None)], False)
        if has_combat.size
        else np.zeros(len(light), dtype=bool)
    )
    surviving = int(keep.sum())
    keep &= session_ok
    report.dropped_empty_sessions = surviving - int(keep.sum())

    # Stage 4c: re-index the surviving sessions to a dense 0..K-1 range. The raw
    # ids already follow the session start order, so their ascending rank is the
    # chronological dense numbering.
    surviving_ids = np.unique(raw_session[keep])
    dense = np.full(len(sessions), -1, dtype=np.int64)
    dense[surviving_ids] = np.arange(len(surviving_ids), dtype=np.int64)
    session_id = (
        np.where(raw_session >= 0, dense[np.clip(raw_session, 0, None)], -1).astype(np.int64)
        if dense.size
        else np.full(len(light), -1, dtype=np.int64)
    )
    report.sessions_out = int(len(surviving_ids))

    # Row counts of the engagements that survived every filter.
    kept_per_session = np.bincount(session_id[session_id >= 0], minlength=len(surviving_ids))
    if kept_per_session.size:
        report.session_frames_min = int(kept_per_session.min())
        report.session_frames_max = int(kept_per_session.max())
        report.session_frames_mean = float(kept_per_session.mean())

    # Engagement strength: how many surviving sessions actually produced damage.
    # A session without one is loot collection or a fight whose damage frame never
    # made it into the log — useful to know, but not a combat *execution* episode.
    impact = frozenset({cfg.kill_action}) | cfg.damage_actions
    landed = np.isin(light["action_type"].to_numpy()[keep], tuple(impact))
    seen = np.zeros(report.sessions_out, dtype=bool)
    if seen.size:
        np.logical_or.at(seen, session_id[keep], landed)
    report.sessions_without_damage = int((~seen).sum())

    return RowDecisions(keep=keep, session_id=session_id, raw_session_id=raw_session, sessions=sessions)


# --- Reporting --------------------------------------------------------------


def session_stats(
    light: pd.DataFrame,
    decisions: RowDecisions,
    cfg: CleaningConfig,
    names_by_id: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Per-character statistics of the combat sessions that survived the cleaning."""
    columns = [
        "character",
        "character_id",
        "sessions",
        "frames",
        "frames_per_session",
        "kills",
        "aborted",
        "combat_frames",
        "combat_share",
        "sessions_dropped",
    ]
    tagged = pd.DataFrame(
        {
            "character_id": light["character_id"].to_numpy(),
            "action_type": light["action_type"].to_numpy(),
            "session_id": decisions.session_id,
            "raw_session_id": decisions.raw_session_id,
            "keep": decisions.keep,
        }
    )
    kept = tagged[tagged["keep"]]
    if kept.empty:
        return pd.DataFrame(columns=columns)

    stats = (
        kept.groupby("character_id", observed=True)
        .agg(
            frames=("session_id", "size"),
            sessions=("session_id", "nunique"),
            combat_frames=("action_type", lambda column: int(column.isin(cfg.combat_actions).sum())),
        )
        .reset_index()
    )
    stats["sessions"] = stats["sessions"].astype(int)
    stats["frames_per_session"] = (stats["frames"] / stats["sessions"]).round(1).astype(float)
    stats["combat_share"] = (stats["combat_frames"] / stats["frames"]).round(3).astype(float)

    # Engagement outcomes come from the metadata collected during sessionization.
    # ``survived`` separates the sessions that kept at least one row from the ones
    # stage 4b discarded as combat-free, so both numbers are exact.
    survived_raw = {int(value) for value in np.unique(decisions.raw_session_id[decisions.keep])}
    engagements = pd.DataFrame(
        [
            {
                "character_id": engagement.character_id,
                "survived": engagement.raw_id in survived_raw,
                "has_kill": int(engagement.has_kill),
                "aborted": int(engagement.interrupted),
            }
            for engagement in decisions.sessions
        ]
    )
    if engagements.empty:
        stats["kills"] = 0
        stats["aborted"] = 0
        stats["sessions_dropped"] = 0
    else:
        outcome = (
            engagements[engagements["survived"]]
            .groupby("character_id", observed=True)
            .agg(kills=("has_kill", "sum"), aborted=("aborted", "sum"))
            .reset_index()
        )
        dropped = (
            engagements[~engagements["survived"]]
            .groupby("character_id", observed=True)
            .size()
            .rename("sessions_dropped")
            .reset_index()
        )
        stats = stats.merge(outcome, on="character_id", how="left").merge(dropped, on="character_id", how="left")
        for column in ("kills", "aborted", "sessions_dropped"):
            stats[column] = stats[column].fillna(0).astype(int)

    # The action table stores ids only; show the names next to them for readability.
    names = names_by_id or {}
    stats.insert(0, "character", stats["character_id"].map(names).fillna("-"))
    return stats.sort_values("frames", ascending=False, ignore_index=True)[columns]


# --- Stage 3 counters + output ---------------------------------------------


def count_env_entries(column: pa.ChunkedArray) -> tuple[int, int, int]:
    """Return (total entries, rows with an empty list, rows with a null list)."""
    lengths = pa.compute.list_value_length(column).cast(pa.int64())
    entries = int(pa.compute.sum(pa.compute.fill_null(lengths, 0)).as_py() or 0)
    empties = int(pa.compute.sum(pa.compute.equal(pa.compute.fill_null(lengths, -1), 0).cast(pa.int64())).as_py() or 0)
    nulls = int(pa.compute.sum(pa.compute.is_null(lengths).cast(pa.int64())).as_py() or 0)
    return entries, empties, nulls


def conform_schema(table: pa.Table, expected: pa.Schema) -> pa.Table:
    """Cast every column to the expected type so nested types survive the round-trip."""
    if table.schema.equals(expected):
        return table

    arrays = [table.column(field.name).cast(field.type) for field in expected]
    return pa.Table.from_arrays(arrays, schema=expected)


def write_cleaned(table: pa.Table, decisions: RowDecisions, original: pa.Schema, output: Path) -> pa.Table:
    """
    Write the surviving records with the ``session_id`` column appended.

    The input row order is preserved; the schema is the input schema plus one
    int64 column at the end.
    """
    cleaned = table.filter(pa.array(decisions.keep))
    expected = original.append(pa.field(SESSION_COLUMN, pa.int64()))
    cleaned = cleaned.append_column(SESSION_COLUMN, pa.array(decisions.session_id[decisions.keep], type=pa.int64()))
    cleaned = conform_schema(cleaned, expected)

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(cleaned, output, compression="snappy")
    return cleaned


# --- Orchestration ---------------------------------------------------------


def print_summary(
    report: CleaningReport,
    cfg: CleaningConfig,
    names_by_id: dict[int, str],
    light: pd.DataFrame,
    decisions: RowDecisions,
) -> None:
    """Print the concise stage-by-stage cleaning summary and the session table."""
    print(f"\nRows: {report.rows_in} in -> {report.rows_out} out")
    if report.characters_excluded:
        excluded_names = ", ".join(
            f"{names_by_id.get(character_id, '-')} (id {character_id})" for character_id in report.characters_excluded
        )
        print(f"  stage 0  excluded characters         : {excluded_names} -> {report.dropped_excluded} rows")

    print("  stage 1  combat sessionization        :")
    print(
        f"           target selections            : {report.targets_total} "
        f"({report.targets_item_ignored} item picks + {report.targets_non_hostile_ignored} non-hostile NPCs ignored, "
        f"{report.targets_refresh} same-target refreshes folded in)"
    )
    print(
        f"           interruptions (case C)       : {report.interrupts_total} (gap > {cfg.gap_seconds:g}s, death, teleport, new target)"
    )
    print(
        f"           combat sessions extracted    : {report.sessions_extracted} "
        f"({report.sessions_with_kill} with a kill, {report.sessions_with_pickup} ended on a pickup, "
        f"{report.sessions_aborted} aborted)"
    )
    print(
        f"           rows outside engagements     : {report.dropped_out_of_session} dropped (roaming / travel / city)"
    )
    print(
        f"  stage 1  non-combat actions dropped   : {report.dropped_non_combat} (buy / sell / talk / craft / trade …)"
    )
    print(f"  stage 2  idle streak frames dropped   : {report.dropped_idle} (only the first frame of a run is kept)")
    print(
        f"  stage 3  env entries {report.env_entries_before} -> {report.env_entries_after} "
        f"({report.env_rows_emptied} emptied, {report.env_rows_null} null; before row filtering)"
    )
    print(
        f"  stage 4a system bursts collapsed     : {report.dropped_burst} (multi-item drops; "
        f"run gap <= {cfg.burst_ms:g} ms, last frame kept)"
    )
    print(
        f"  stage 4a sub-ms duplicate spam       : {report.dropped_debounced} "
        f"(< {cfg.debounce_ms:g} ms, identical type + raw)"
    )
    print(
        f"  stage 4b sessions without combat    : {report.sessions_dropped} of {report.sessions_total} "
        f"-> {report.dropped_empty_sessions} rows removed"
    )
    print(
        f"  stage 4c combat sessions in output  : {report.sessions_out} (dense 0..{max(report.sessions_out - 1, 0)}), "
        f"rows/session min {report.session_frames_min} / mean {report.session_frames_mean:.1f} / "
        f"max {report.session_frames_max}"
    )
    print(
        f"  stage 4c sessions with damage       : {report.sessions_out - report.sessions_without_damage} "
        f"({report.sessions_without_damage} loot-only / no damage logged)"
    )

    print("\nSession statistics (sessions that survived):")
    stats = session_stats(light, decisions, cfg, names_by_id)
    print(stats.to_string(index=False) if not stats.empty else "(no surviving sessions)")


def process(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    npcs_path: str | Path = DEFAULT_NPCS,
    characters_path: str | Path = DEFAULT_CHARACTERS,
    cfg: CleaningConfig | None = None,
) -> CleaningReport:
    """Clean the action dataset and write it to ``output_path``."""
    cfg = cfg or CleaningConfig()
    actions, npcs, characters = Path(input_path), Path(npcs_path), Path(characters_path)
    for label, path in (("action", actions), ("NPC", npcs), ("character", characters)):
        if not path.exists():
            raise FileNotFoundError(f"{label} table not found: {path}")

    names_by_id = load_character_names(characters)
    cfg = resolve_excluded_characters(cfg, names_by_id)
    # Resolve "hostile" once and share it with both the state machine and DuckDB.
    cfg = replace(cfg, hostile_npc_ids=load_hostile_npc_ids(npcs, cfg))

    original = pq.read_schema(actions)
    report = CleaningReport()

    # Count the input array before it is cleaned; only one column is read.
    report.env_entries_before, _, _ = count_env_entries(pq.read_table(actions, columns=[ENV_COLUMN]).column(ENV_COLUMN))

    with duckdb.connect() as con:
        print(
            f"Cleaning environment array (hostile NPCs + loot + players, distance <= "
            f"{cfg.max_monster_distance:g}, top-{cfg.top_n_objects}) ..."
        )
        table = clean_environment(con, actions, cfg)

    entries_after, empties_after, nulls_after = count_env_entries(table.column(ENV_COLUMN))
    report.env_entries_after = entries_after
    report.env_rows_emptied = empties_after
    report.env_rows_null = nulls_after

    light = load_light(actions)
    decisions = decide_rows(light, cfg, report)

    cleaned = write_cleaned(table, decisions, original, Path(output_path))
    report.rows_out = cleaned.num_rows

    print_summary(report, cfg, names_by_id, light, decisions)

    print(f"\nSaved {cleaned.num_rows} rows x {cleaned.num_columns} columns to {output_path}")
    print(f"Schema: input schema + '{SESSION_COLUMN}' (int64)")
    return report


def main() -> None:
    """Parse the command line and run the cleaning pipeline."""
    description = (
        "Slice the L2J player-action Parquet dataset into combat sessions "
        "(selected_target -> kill / first pickup) for RL micro-control training."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i", "--input", type=str, default=str(DEFAULT_INPUT), help=f"Input Parquet (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(DEFAULT_OUTPUT), help=f"Output Parquet (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--npcs-parquet",
        type=str,
        default=str(DEFAULT_NPCS),
        help=f"NPC table used to validate targets (default: {DEFAULT_NPCS})",
    )
    parser.add_argument(
        "--characters-parquet",
        type=str,
        default=str(DEFAULT_CHARACTERS),
        help=f"Character table used to resolve excluded names (default: {DEFAULT_CHARACTERS})",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=15.0,
        help="Gap between consecutive raw records of a character that aborts an engagement (default: 15)",
    )
    parser.add_argument(
        "--top-n-objects",
        "--top-n-monsters",
        dest="top_n_objects",
        type=int,
        default=10,
        help="Max environment objects kept per row (default: 10; --top-n-monsters is the legacy alias)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3000.0,
        help="Drop environment objects farther away than this (default: 3000)",
    )
    parser.add_argument(
        "--debounce-ms",
        type=float,
        default=1.0,
        help="Collapse identical events closer than this many milliseconds; 0 disables (default: 1)",
    )
    parser.add_argument(
        "--burst-ms",
        type=float,
        default=50.0,
        help=(
            "Micro-window for a rapid system-event burst (multi-item drops); only the last frame is kept. "
            "0 disables (default: 50)"
        ),
    )
    parser.add_argument(
        "--idle-max-seconds",
        type=float,
        default=3.0,
        help="Deprecated and ignored: an idle streak now keeps only its first frame, whatever its duration.",
    )
    parser.add_argument(
        "--exclude-characters",
        type=str,
        default="",
        help="Comma-separated character/user names to drop completely, e.g. 'admin' or 'admin,TestChar' (case-insensitive)",
    )
    args = parser.parse_args()

    cfg = CleaningConfig(
        gap_seconds=args.gap_seconds,
        top_n_objects=args.top_n_objects,
        max_monster_distance=args.max_distance,
        debounce_ms=args.debounce_ms,
        burst_ms=args.burst_ms,
        exclude_characters=frozenset(part.strip() for part in args.exclude_characters.split(",") if part.strip()),
    )
    process(args.input, args.output, args.npcs_parquet, args.characters_parquet, cfg)


if __name__ == "__main__":
    main()
