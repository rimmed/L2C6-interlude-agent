"""
Lineage2 C6 Interlude -- neural combat policy for the headless client.

Runs the ONNX export of ``CombatEventTransformer`` (see ``l2_NN/export_onnx.py``) on the live game
state.  Only ``numpy`` and ``onnxruntime`` are needed: every normalisation constant, the window
length, the action label space and the per-character skill panels come from the ``.json`` file
that the exporter writes next to the ``.onnx`` file.

How the live state is fed to the network
----------------------------------------
The network was trained on *event* windows (``l2_NN/dataloader/dataloder.py``): one row per logged
event, ``delta_t = log(1 + seconds since the previous event)``, windows never crossing a combat
session (``selected_target`` ... ``killed``/``pickup``) and left-padded to ``seq_len``.  The policy
mirrors that:

* :meth:`CombatPolicy.record_event` appends a snapshot row when the game reports something
  (damage dealt / taken, a kill) - the environment events of the training data;
* :meth:`CombatPolicy.decide` appends the decision row, runs the network on the last ``seq_len``
  rows and returns a :class:`Decision`;
* the history is cleared once a fight is over (no target, no loot left), so a new session starts
  with the "pick a target" decision exactly like the training sessions do.

Features the client cannot observe are filled the way the training data filled them: the logs had
no move speed (constant default) and no MP cost (``has_enough_mp`` is always 1), every panel slot
is ``is_available``.  Skill cooldowns come from the server's SkillCoolTime (0xC1) answers, which
are already scaled by the server's speed; until one arrives after a cast, the datapack reuse delay
divided by ``server_speed`` is used as an estimate.

Head outputs that are not executable right now (attack without any monster, pickup without loot,
a skill on cooldown, the game-response classes such as ``inflicted_damage``) are masked before the
argmax.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tools.data import skill_data

#: Action classes the bot can execute.  The rest of the label space (inflicted_damage, killed, ...)
#: is the game answering the player and is never a valid command.
EXECUTABLE_ACTIONS = ("moved", "selected_target", "attacked", "used_skill", "picked_up", "used_item")

#: The logged InCombat flag is the server's attack stance: it switches on with the first hit (not
#: with target selection) and lapses this long after the last combat action.
IN_COMBAT_TIMEOUT_S = 15.0

#: A skill cast this recently may still be queued behind an auto-attack swing, so a SkillCoolTime
#: answer that does not list it yet must not mark it ready.
CAST_GRACE_S = 1.5

#: The metadata layout this module understands (``METADATA_VERSION`` in ``export_onnx.py``).
SUPPORTED_METADATA_VERSION = 1

# StatusUpdate (0x0E) attribute ids, see ``packets.STATUS_ATTRIBUTE_NAMES``.
_SU_LEVEL, _SU_CUR_HP, _SU_MAX_HP, _SU_CUR_MP, _SU_MAX_MP, _SU_CUR_CP, _SU_MAX_CP = (
    0x01, 0x09, 0x0A, 0x0B, 0x0C, 0x21, 0x22,
)


@dataclass(frozen=True)
class Decision:
    """One step of the policy, already resolved to game objects."""

    action: str
    #: radar ``object_id`` picked by the pointer head (``None`` when no monster is in range)
    mob_object_id: int | None
    #: skill id picked by the skill head (``None`` when no skill is usable)
    skill_id: int | None
    #: softmax probability of ``action`` among the executable actions
    confidence: float

    def describe(self) -> str:
        parts = [f"{self.action} ({self.confidence:.0%})"]
        if self.mob_object_id is not None:
            parts.append(f"mob={self.mob_object_id}")
        if self.skill_id is not None:
            parts.append(f"skill={skill_data.get_skill_name(self.skill_id) or '?'} ({self.skill_id})")
        return " ".join(parts)


@dataclass(frozen=True)
class _Frame:
    """One event row: the model inputs at time ``t`` plus the radar ids behind the mob slots."""

    t: float
    log_dt: float  # log(1 + seconds since the previous row of the session), 0 for the first row
    player: np.ndarray  # (F_player,)
    mobs: np.ndarray  # (K, F_mob)
    skills: np.ndarray  # (N, F_skill)
    mob_ids: tuple[int, ...]  # object ids of the real mob slots, closest first


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class CombatPolicy:
    """Stateful wrapper around the ONNX combat model."""

    def __init__(
        self,
        model_path: str | Path,
        character_id: int | None = None,
        providers: list[str] | None = None,
        server_speed: float = 1.0,
    ):
        """
        :param model_path: ``.onnx`` file; its ``.json`` metadata must sit next to it
        :param character_id: training ``character_id`` whose skill panel to use; ``None`` picks the
            panel that overlaps most with the character's skills (see :meth:`bind_character`)
        :param providers: onnxruntime execution providers (default: CPU)
        """
        import onnxruntime as ort  # imported lazily: the scripted bot must run without it

        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {meta_path} (re-run l2_NN/export_onnx.py)")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        if self.meta.get("metadata_version") != SUPPORTED_METADATA_VERSION:
            raise ValueError(
                f"{meta_path}: metadata_version {self.meta.get('metadata_version')} is not supported "
                f"(expected {SUPPORTED_METADATA_VERSION})"
            )

        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.seq_len: int = self.meta["seq_len"]
        self.max_mobs: int = self.meta["max_mobs"]
        self.max_skills: int = self.meta["max_skills"]
        self.hide_last_target: bool = self.meta["hide_last_target"]
        self.constants: dict[str, float] = self.meta["constants"]
        self.action_to_id: dict[str, int] = self.meta["action_to_id"]

        features = self.meta["features"]
        self._player_names: list[str] = features["player"]
        self._mob_names: list[str] = features["mob"]
        self._skill_names: list[str] = features["skill"]
        self._mob_at = {name: i for i, name in enumerate(self._mob_names)}
        self._skill_at = {name: i for i, name in enumerate(self._skill_names)}

        self._executable = [
            (name, self.action_to_id[name]) for name in EXECUTABLE_ACTIONS if name in self.action_to_id
        ]
        self._requested_character_id = character_id
        self.panel: list[int] = []
        self._known_skills: set[int] = set()

        self._frames: deque[_Frame] = deque(maxlen=self.seq_len)
        self._session_engaged = False
        self._cooldown_until: dict[int, float] = {}
        #: The server runs its timers this many times faster than the datapack values.
        self.server_speed = max(server_speed, 1e-6)
        self._cast_at: dict[int, float] = {}
        self._last_combat_at = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def bind_character(self, char) -> None:
        """
        Pick the skill panel (slot -> skill id) the skill head was trained with.

        Slot semantics are per training character, so the panel must come from the metadata, not
        from the live ``SkillList``.  Without an explicit ``character_id`` the panel sharing the most
        skills with the character is used; with no overlap at all the character's own active skills
        are used instead (the skill head is then unreliable, which is reported).
        """
        self._known_skills = {s["id"] for s in char.skills if not s.get("passive")}
        panels = {int(k): v for k, v in self.meta["skill_panels"].items()}

        if self._requested_character_id is not None:
            if self._requested_character_id not in panels:
                raise ValueError(
                    f"character_id {self._requested_character_id} is not in the model (known: {sorted(panels)})"
                )
            chosen = self._requested_character_id
        else:
            chosen = max(panels, key=lambda cid: len(self._known_skills & set(panels[cid])), default=None)
            if chosen is not None and not self._known_skills & set(panels[chosen]):
                chosen = None

        if chosen is None:
            self.panel = sorted(self._known_skills)[: self.max_skills]
            print(f"[MODEL] !! No training skill panel matches {char.name}'s skills - using its own {self.panel}")
        else:
            self.panel = list(panels[chosen])[: self.max_skills]
            missing = [sid for sid in self.panel if sid not in self._known_skills]
            print(f"[MODEL] Skill panel of training character {chosen}: {self.panel}"
                  + (f" (not learned, masked: {missing})" if missing else ""))

    # ------------------------------------------------------------------
    # Live-state hooks
    # ------------------------------------------------------------------

    @staticmethod
    def update_self_status(char, su: dict) -> bool:
        """
        Apply a StatusUpdate about the character itself to ``char`` (HP/MP/CP/level).

        The scripted game loop only feeds StatusUpdates to the radar, so without this the vitals
        would freeze at their EnterWorld values.  Returns ``True`` when ``su`` was about ``char``.
        """
        if not su or su.get("object_id") != char.object_id:
            return False
        attrs = su.get("attributes", {})
        for attr, field_name in (
            (_SU_CUR_HP, "cur_hp"), (_SU_MAX_HP, "max_hp"),
            (_SU_CUR_MP, "cur_mp"), (_SU_MAX_MP, "max_mp"),
            (_SU_CUR_CP, "cur_cp"), (_SU_MAX_CP, "max_cp"),
            (_SU_LEVEL, "level"),
        ):  # fmt: skip
            if attr in attrs:
                setattr(char, field_name, attrs[attr])
        return True

    def note_combat(self, now: float | None = None) -> None:
        """Mark a combat action - damage dealt/taken, an attack or a skill (drives ``in_combat``)."""
        self._last_combat_at = time.time() if now is None else now

    def note_skill_used(self, skill_id: int, now: float | None = None) -> None:
        """
        Estimate the cooldown of a skill that was just cast.

        The datapack reuse delay scaled by :attr:`server_speed` is only a placeholder until the
        server's SkillCoolTime answer arrives (:meth:`apply_skill_cool_time`).
        """
        now = time.time() if now is None else now
        reuse_s = skill_data.get_skill_reuse_delay(skill_id) / 1000.0 / self.server_speed
        self._cooldown_until[skill_id] = now + reuse_s
        self._cast_at[skill_id] = now

    def apply_skill_cool_time(self, parsed: dict, now: float | None = None) -> None:
        """
        Replace the cooldown table with a parsed SkillCoolTime (0xC1).

        The packet lists every skill that is still cooling down, so a skill missing from it is ready.
        """
        now = time.time() if now is None else now
        fresh = {cd["skill_id"]: now + cd["remaining"] for cd in parsed.get("cooldowns", [])}
        for skill_id, cast_at in self._cast_at.items():
            if skill_id not in fresh and now - cast_at < CAST_GRACE_S and skill_id in self._cooldown_until:
                fresh[skill_id] = self._cooldown_until[skill_id]  # the cast has not started yet
        self._cooldown_until = fresh

    def wants_cooldown_sync(self, now: float | None = None) -> bool:
        """``True`` while a skill is (believed to be) cooling down or was cast very recently."""
        now = time.time() if now is None else now
        return any(until > now for until in self._cooldown_until.values()) or any(
            now - cast_at < CAST_GRACE_S for cast_at in self._cast_at.values()
        )

    def reset_session(self) -> None:
        """Forget the event history: the next row starts a new combat session."""
        self._frames.clear()
        self._session_engaged = False

    def record_event(self, char, current_target_id: int, pos: tuple[int, int], now: float | None = None) -> None:
        """Append a snapshot row for an environment event (damage, kill, ...)."""
        self._snapshot(char, current_target_id, pos, time.time() if now is None else now)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        char,
        current_target_id: int,
        pos: tuple[int, int],
        *,
        has_loot: bool,
        has_potion: bool,
        now: float | None = None,
    ) -> Decision:
        """Append the decision row, run the network and return the best executable action."""
        now = time.time() if now is None else now

        # A training session runs from selected_target to killed/pickup; once the fight is over
        # (no target, nothing to loot) the next decision opens a fresh session.
        engaged = bool(current_target_id) or has_loot
        if not engaged and self._session_engaged:
            self.reset_session()
        self._session_engaged = self._session_engaged or engaged

        frame = self._snapshot(char, current_target_id, pos, now)
        logits_action, logits_pointer, logits_skill = self._run()

        has_mobs = bool(frame.mob_ids)
        target_alive = bool(current_target_id) and current_target_id in char.radar.entries
        usable_slots = [i for i, sid in enumerate(self.panel) if self._skill_usable(sid, now)]
        valid = {
            "moved": True,
            "selected_target": has_mobs,
            "attacked": target_alive or has_mobs,
            "used_skill": bool(usable_slots) and (target_alive or has_mobs),
            "picked_up": has_loot,
            "used_item": has_potion,
        }

        candidates = [(name, idx) for name, idx in self._executable if valid[name]]
        scores = np.array([logits_action[idx] for _, idx in candidates], dtype=np.float64)
        probs = _softmax(scores)
        best = int(np.argmax(probs))
        action = candidates[best][0]

        mob_id = None
        if has_mobs:
            slots = len(frame.mob_ids)
            mob_id = frame.mob_ids[int(np.argmax(logits_pointer[:slots]))]

        skill_id = None
        if usable_slots:
            slot = max(usable_slots, key=lambda i: logits_skill[i])
            skill_id = self.panel[slot]

        return Decision(action=action, mob_object_id=mob_id, skill_id=skill_id, confidence=float(probs[best]))

    # ------------------------------------------------------------------
    # Feature encoding (mirrors l2_NN/dataloader/dataloder.py)
    # ------------------------------------------------------------------

    def _skill_usable(self, skill_id: int, now: float) -> bool:
        return skill_id in self._known_skills and self._cooldown_until.get(skill_id, 0.0) <= now

    def _snapshot(self, char, current_target_id: int, pos: tuple[int, int], now: float) -> _Frame:
        """Encode the current state and append it to the session history."""
        mobs, mob_ids = self._encode_mobs(char, current_target_id, pos)
        # dt is fixed when the row is appended, so it stays the real gap to the previous event
        # even after that event has scrolled out of the window (as in the training windows).
        log_dt = math.log1p(max(0.0, now - self._frames[-1].t)) if self._frames else 0.0
        frame = _Frame(
            t=now,
            log_dt=log_dt,
            player=self._encode_player(char, current_target_id, now),
            mobs=mobs,
            skills=self._encode_skills(now),
            mob_ids=mob_ids,
        )
        self._frames.append(frame)
        return frame

    def _encode_player(self, char, current_target_id: int, now: float) -> np.ndarray:
        c = self.constants

        def ratio(numerator: float, denominator: float) -> float:
            return 0.0 if denominator <= 0 else min(max(numerator / denominator, 0.0), 1.0)

        in_combat = (now - self._last_combat_at) < IN_COMBAT_TIMEOUT_S
        values = {
            "hp_ratio": ratio(char.cur_hp, char.max_hp),
            "mp_ratio": ratio(char.cur_mp, char.max_mp),
            "cp_ratio": ratio(char.cur_cp, char.max_cp),
            "in_combat": 1.0 if in_combat else 0.0,
            "lvl_norm": ratio(char.level, c["max_level"]),
            # The logs' Cast.Speed / Atk.Speed are the UserInfo m_atk_spd / p_atk_spd.
            "cast_speed": ratio(char.m_atk_spd or c["default_cast_speed"], c["max_cast_speed"]),
            "atk_speed": ratio(char.p_atk_spd or c["default_atk_speed"], c["max_atk_speed"]),
            # Training never saw a real move speed - feed the same constant it was trained on.
            "move_speed": ratio(c["default_move_speed"], c["max_move_speed"]),
        }
        return np.array([values[name] for name in self._player_names], dtype=np.float32)

    def _encode_mobs(self, char, current_target_id: int, pos: tuple[int, int]) -> tuple[np.ndarray, tuple[int, ...]]:
        c = self.constants
        matrix = np.zeros((self.max_mobs, len(self._mob_names)), dtype=np.float32)
        matrix[:, self._mob_at["is_padded"]] = 1.0

        # The logs only identify a monster by its template id, so training flagged *every* nearby
        # monster of the target's template as is_current_target - reproduce that here.
        target = char.radar.entries.get(current_target_id) if current_target_id else None
        target_npc_id = target.get("npc_id", 0) if target else 0

        x, y = pos
        ranked = sorted(
            ((math.hypot(x - mob["x"], y - mob["y"]), oid, mob) for oid, mob in char.radar.entries.items()),
            key=lambda item: item[0],
        )[: self.max_mobs]
        for slot, (dist, oid, mob) in enumerate(ranked):
            row = {
                "dist_norm": min(dist / c["max_distance"], 1.0),
                "lvl_diff": (mob["level"] - char.level) / c["max_level"],
                "is_current_target": 1.0 if oid == current_target_id or (
                    target_npc_id and mob.get("npc_id") == target_npc_id) else 0.0,
                "is_padded": 0.0,
            }
            matrix[slot] = [row[name] for name in self._mob_names]
        return matrix, tuple(oid for _, oid, _ in ranked)

    def _encode_skills(self, now: float) -> np.ndarray:
        scale = self.constants["skill_cooldown_norm_seconds"]
        matrix = np.zeros((self.max_skills, len(self._skill_names)), dtype=np.float32)
        for slot, skill_id in enumerate(self.panel):
            left = max(0.0, self._cooldown_until.get(skill_id, 0.0) - now)
            row = {
                "is_available": 1.0,
                "is_off_cooldown": 1.0 if left <= 0.0 else 0.0,
                "cooldown_norm": min(left / scale, 1.0),
                "has_enough_mp": 1.0,  # the training logs carried no MP cost
            }
            matrix[slot] = [row[name] for name in self._skill_names]
        return matrix

    def _run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = list(self._frames)
        real, pad = len(frames), self.seq_len - len(frames)

        player = np.zeros((self.seq_len, len(self._player_names)), dtype=np.float32)
        delta_t = np.zeros(self.seq_len, dtype=np.float32)
        mobs = np.zeros((self.seq_len, self.max_mobs, len(self._mob_names)), dtype=np.float32)
        mobs[..., self._mob_at["is_padded"]] = 1.0
        skills = np.zeros((self.seq_len, self.max_skills, len(self._skill_names)), dtype=np.float32)
        seq_mask = np.zeros(self.seq_len, dtype=bool)
        seq_mask[pad:] = True

        for i, frame in enumerate(frames):
            player[pad + i] = frame.player
            mobs[pad + i] = frame.mobs
            skills[pad + i] = frame.skills
            delta_t[pad + i] = frame.log_dt

        if self.hide_last_target and real:
            mobs[-1, :, self._mob_at["is_current_target"]] = 0.0

        feeds = {
            "player_state": player[None],
            "delta_t": delta_t[None],
            "mobs_matrix": mobs[None],
            "skills_matrix": skills[None],
            "seq_mask": seq_mask[None],
        }
        action, pointer, skill = self.session.run(list(self.meta["outputs"]), feeds)
        return action[0], pointer[0], skill[0]
