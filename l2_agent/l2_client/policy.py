"""
Lineage2 C6 Interlude -- neural combat policy for the headless client.

Runs the ONNX export of ``CombatEventTransformer`` (see ``l2_NN/export_onnx.py``) on the live game
state.  Only ``numpy`` and ``onnxruntime`` are needed: every normalisation constant, the window
length and the action label space come from the ``.json`` file that the exporter writes next to
the ``.onnx`` file.  The skill panel is the character's own combat skills (``SkillList`` filtered
by the datapack catalog, ``skill_data.combat_panel``), each slot described by its datapack features
(``l2_client/skill_features.py``, shared with the training dataloader) - so skills of classes the
model never saw are usable.

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
no move speed (constant default).  Skill cooldowns come from the server's SkillCoolTime (0xC1) answers; until
one arrives after a cast, the datapack reuse delay is used as an estimate.

Two clocks are involved when the server runs with ``TimeScale`` > 1 (``server_speed``): the server
counts cooldowns in *game* time, which runs ``server_speed`` times faster than real time.  The
expiry of every cooldown is therefore stored in real (``time.time()``) seconds, which decides when
a skill is usable, while the ``cooldown_norm`` feature is fed in game seconds - the unit of the
logged ``[CD:...]`` values the model was trained on.

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

from . import skill_features
from .tools.data import skill_data

#: Action classes the bot can execute.  The rest of the label space (inflicted_damage, killed, ...)
#: is the game answering the player and is never a valid command.
EXECUTABLE_ACTIONS = ("moved", "selected_target", "attacked", "used_skill", "picked_up", "used_item")

#: The logged InCombat flag is the server's attack stance: it switches on with the first hit (not
#: with target selection) and lapses this long after the last combat action.
IN_COMBAT_TIMEOUT_S = 15.0

#: Monsters this far above or below the character are left out of the candidates: they stand on a
#: cliff or in a pit, and walking to them straight runs into the terrain (the client has no pathfinding).
MAX_TARGET_DZ = 300

#: A skill cast this recently may still be queued behind an auto-attack swing, so a SkillCoolTime
#: answer that does not list it yet must not mark it ready.
CAST_GRACE_S = 1.5

#: The metadata layout this module understands (``METADATA_VERSION`` in ``export_onnx.py``).
SUPPORTED_METADATA_VERSION = 2

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
    #: the same choice as indices into the model heads (what a trajectory stores): ``action_id`` in
    #: ``action_to_id``, ``pointer_slot`` into ``mobs_matrix``, ``skill_slot`` into ``skills_matrix``
    action_id: int = -1
    pointer_slot: int | None = None
    skill_slot: int | None = None

    def describe(self) -> str:
        parts = [f"{self.action} ({self.confidence:.0%})"]
        if self.mob_object_id is not None:
            parts.append(f"mob={self.mob_object_id}")
        if self.skill_id is not None:
            parts.append(f"skill={skill_data.get_skill_name(self.skill_id) or '?'} ({self.skill_id})")
        return " ".join(parts)


@dataclass(frozen=True)
class Observation:
    """The model input of one decision step plus what is needed to turn head outputs into actions."""

    #: unbatched model inputs: ``player_state`` (L, F_p), ``delta_t`` (L,), ``mobs_matrix`` (L, K, F_m),
    #: ``skills_matrix`` (L, N, F_s), ``seq_mask`` (L,)
    feeds: dict[str, np.ndarray]
    #: radar object ids behind the real ``mobs_matrix`` slots of the last row, closest first
    mob_ids: tuple[int, ...]
    #: executable action name -> whether it can be carried out right now
    valid: dict[str, bool]
    #: ``skills_matrix`` slots whose skill can be cast right now
    usable_slots: tuple[int, ...]
    #: the skill panel (slot -> skill id) the slots refer to
    panel: tuple[int, ...]
    #: ``time.time()`` of the observation
    t: float


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
        model_path: str | Path | None,
        providers: list[str] | None = None,
        server_speed: float = 1.0,
        device: str = "cpu",
        max_target_dz: float | None = MAX_TARGET_DZ,
        metadata_path: str | Path | None = None,
        excluded_skills: frozenset[int] = frozenset(),
    ):
        """
        :param model_path: ``.onnx`` file; its ``.json`` metadata must sit next to it. ``None`` builds
            an encoder only (:meth:`observe` works, :meth:`act` does not) from ``metadata_path``
        :param providers: onnxruntime execution providers; overrides ``device`` when given
        :param device: ``"cpu"`` or ``"gpu"`` (CUDA); a GPU session that silently falls back to the
            CPU raises instead, so a benchmark can never measure the wrong device
        :param max_target_dz: leave out monsters with a larger height difference (``None`` keeps all)
        :param metadata_path: feature-layout ``.json``; defaults to the one next to ``model_path``
        :param excluded_skills: skill ids never offered for casting (e.g. crafting or Relax, which
            the panel of a training character may contain but which are useless in a fight)
        """
        self.excluded_skills = frozenset(excluded_skills)
        if model_path is None and metadata_path is None:
            raise ValueError("either model_path or metadata_path is required")
        meta_path = Path(metadata_path) if metadata_path is not None else Path(model_path).with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {meta_path} (re-run l2_NN/export_onnx.py)")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        if self.meta.get("metadata_version") != SUPPORTED_METADATA_VERSION:
            raise ValueError(
                f"{meta_path}: metadata_version {self.meta.get('metadata_version')} is not supported "
                f"(expected {SUPPORTED_METADATA_VERSION})"
            )

        self.session = None
        self.device = "none"
        if model_path is not None:
            self._open_session(Path(model_path), providers, device)

        #: Wall-clock milliseconds of every ``session.run`` (pure inference) and of every whole
        #: :meth:`decide` call (feature encoding + inference + masking), in call order.
        self.inference_ms: list[float] = []
        self.decide_ms: list[float] = []
        self._check_skill_encoding(meta_path)
        self._init_layout(server_speed, max_target_dz)

    def _open_session(self, model_path: Path, providers: list[str] | None, device: str) -> None:
        import onnxruntime as ort  # imported lazily: the scripted bot must run without it

        if providers is None:
            if device == "gpu":
                # Load the CUDA/cuDNN libraries shipped as nvidia-* wheels (shared with torch).
                ort.preload_dlls()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        if providers[0] not in self.session.get_providers()[:1]:
            raise RuntimeError(f"requested {providers[0]}, but the session runs on {self.session.get_providers()}")
        self.device = "gpu" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"

    def _check_skill_encoding(self, meta_path: Path) -> None:
        """The model was trained with these skill-feature scales; a different encoder would feed it garbage."""
        expected = self.meta.get("skill_encoding", {})
        actual = {
            "power_log_scale": skill_features.POWER_LOG_SCALE,
            "cast_range_scale": skill_features.CAST_RANGE_SCALE,
            "hit_time_scale_ms": skill_features.HIT_TIME_SCALE_MS,
            "reuse_log_scale": skill_features.REUSE_LOG_SCALE,
            "cooldown_scale_s": skill_features.COOLDOWN_SCALE_S,
        }
        mismatch = {k: (expected.get(k), v) for k, v in actual.items() if not math.isclose(expected.get(k, -1.0), v)}
        if mismatch or list(self.meta["features"]["skill"]) != list(skill_features.SKILL_SLOT_FEATURES):
            raise ValueError(f"{meta_path}: skill encoding differs from l2_client/skill_features.py: {mismatch}")

    def _init_layout(self, server_speed: float, max_target_dz: float | None) -> None:
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
        self.panel: list[int] = []
        self._panel_info: list[skill_data.SkillInfo] = []
        self._known_skills: set[int] = set()

        self._frames: deque[_Frame] = deque(maxlen=self.seq_len)
        self._session_engaged = False
        self._cooldown_until: dict[int, float] = {}
        #: Server ``TimeScale``: game time runs this many times faster than real time.
        self.server_speed = max(server_speed, 1e-6)
        self.max_target_dz = max_target_dz
        self._cast_at: dict[int, float] = {}
        self._last_combat_at = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def bind_character(self, char) -> None:
        """
        Build the skill panel from the character's own skills (``char.skills`` from SkillList).

        Only learnable combat skills (attack, debuff, heal, buff) get a slot; item skills, crafting
        and resting never do. Call again whenever a new SkillList arrives.
        """
        self._known_skills = {s["id"] for s in char.skills if not s.get("passive")}
        known = [(s["id"], s.get("level", 1)) for s in char.skills if not s.get("passive")]
        self._panel_info = skill_data.combat_panel(known, self.max_skills)
        self.panel = [info.skill_id for info in self._panel_info]
        print(f"[MODEL] Skill panel of {char.name}: " + ", ".join(
            f"{info.name} ({info.category})" for info in self._panel_info) if self._panel_info
            else f"[MODEL] {char.name} has no combat skill yet")

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

        The datapack reuse delay (game time, converted to real time) is only a placeholder until
        the server's SkillCoolTime answer arrives (:meth:`apply_skill_cool_time`).
        """
        now = time.time() if now is None else now
        reuse_s = skill_data.get_skill_reuse_delay(skill_id) / 1000.0 / self.server_speed
        self._cooldown_until[skill_id] = now + reuse_s
        self._cast_at[skill_id] = now

    def apply_skill_cool_time(self, parsed: dict, now: float | None = None) -> None:
        """
        Replace the cooldown table with a parsed SkillCoolTime (0xC1).

        The packet lists every skill that is still cooling down, so a skill missing from it is ready.
        Its ``remaining`` is in game seconds and is converted to a real-time expiry here.
        """
        now = time.time() if now is None else now
        fresh = {cd["skill_id"]: now + cd["remaining"] / self.server_speed for cd in parsed.get("cooldowns", [])}
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

    @property
    def executable_actions(self) -> list[tuple[str, int]]:
        """``(action name, action id)`` of every action the bot can execute."""
        return list(self._executable)

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
        started = time.perf_counter()
        decision = self.act(self.observe(char, current_target_id, pos, has_loot=has_loot, has_potion=has_potion, now=now))
        self.decide_ms.append((time.perf_counter() - started) * 1000.0)
        return decision

    def observe(
        self,
        char,
        current_target_id: int,
        pos: tuple[int, int],
        *,
        has_loot: bool,
        has_potion: bool,
        now: float | None = None,
    ) -> Observation:
        """Append the decision row and return the model input plus the executable-action masks."""
        now = time.time() if now is None else now

        # A training session runs from selected_target to killed/pickup; once the fight is over
        # (no target, nothing to loot) the next decision opens a fresh session.
        engaged = bool(current_target_id) or has_loot
        if not engaged and self._session_engaged:
            self.reset_session()
        self._session_engaged = self._session_engaged or engaged

        frame = self._snapshot(char, current_target_id, pos, now)
        has_mobs = bool(frame.mob_ids)
        target_alive = bool(current_target_id) and current_target_id in char.radar.entries
        usable_slots = tuple(
            i for i, info in enumerate(self._panel_info)
            if self._skill_usable(info.skill_id, now) and char.cur_mp >= info.mp_cost
        )
        valid = {
            "moved": True,
            "selected_target": has_mobs,
            "attacked": target_alive or has_mobs,
            "used_skill": bool(usable_slots) and (target_alive or has_mobs),
            "picked_up": has_loot,
            "used_item": has_potion,
        }
        return Observation(feeds=self._window(), mob_ids=frame.mob_ids, valid=valid, usable_slots=usable_slots,
                           panel=tuple(self.panel), t=now)

    def act(self, obs: Observation) -> Decision:
        """Run the network on ``obs`` and return the best executable action."""
        if self.session is None:
            raise RuntimeError("this CombatPolicy has no model (model_path=None); use it as an encoder only")
        logits_action, logits_pointer, logits_skill = self._infer(obs.feeds)

        candidates = [(name, idx) for name, idx in self._executable if obs.valid[name]]
        scores = np.array([logits_action[idx] for _, idx in candidates], dtype=np.float64)
        probs = _softmax(scores)
        best = int(np.argmax(probs))
        action, action_id = candidates[best]

        mob_id = pointer_slot = None
        if obs.mob_ids:
            pointer_slot = int(np.argmax(logits_pointer[: len(obs.mob_ids)]))
            mob_id = obs.mob_ids[pointer_slot]

        skill_id = skill_slot = None
        if obs.usable_slots:
            skill_slot = max(obs.usable_slots, key=lambda i: logits_skill[i])
            skill_id = obs.panel[skill_slot]

        return Decision(action=action, mob_object_id=mob_id, skill_id=skill_id, confidence=float(probs[best]),
                        action_id=action_id, pointer_slot=pointer_slot, skill_slot=skill_slot)

    def timing_summary(self, warmup: int = 10) -> dict[str, Any]:
        """
        Latency statistics in milliseconds.

        The first ``warmup`` calls are left out: the first runs pay one-off costs (memory arena
        growth, CUDA kernel selection and context set-up on the GPU) that say nothing about the
        steady-state latency.
        """

        def stats(samples: list[float]) -> dict[str, float]:
            kept = np.asarray(samples[warmup:], dtype=np.float64)
            if kept.size == 0:
                return {"count": 0}
            return {
                "count": int(kept.size),
                "mean": float(kept.mean()),
                "std": float(kept.std()),
                "median": float(np.median(kept)),
                "p95": float(np.percentile(kept, 95)),
                "p99": float(np.percentile(kept, 99)),
                "min": float(kept.min()),
                "max": float(kept.max()),
            }

        return {
            "device": self.device,
            "providers": self.session.get_providers() if self.session is not None else [],
            "warmup_calls_skipped": min(warmup, len(self.inference_ms)),
            "first_call_ms": self.inference_ms[0] if self.inference_ms else None,
            "inference_ms": stats(self.inference_ms),
            "decide_ms": stats(self.decide_ms),
        }

    # ------------------------------------------------------------------
    # Feature encoding (mirrors l2_NN/dataloader/dataloder.py)
    # ------------------------------------------------------------------

    def _skill_usable(self, skill_id: int, now: float) -> bool:
        return (skill_id in self._known_skills and skill_id not in self.excluded_skills
                and self._cooldown_until.get(skill_id, 0.0) <= now)

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
            skills=self._encode_skills(char, now),
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
        char_z = getattr(char, "z", None)

        def reachable(oid: int, mob: dict) -> bool:
            # The current target always stays visible: radar positions are spawn positions, so a monster
            # that came down to fight would otherwise vanish mid-fight.
            if self.max_target_dz is None or char_z is None or oid == current_target_id:
                return True
            return abs(mob.get("z", char_z) - char_z) <= self.max_target_dz

        ranked = sorted(
            (
                (math.hypot(x - mob["x"], y - mob["y"]), oid, mob)
                for oid, mob in char.radar.entries.items()
                if reachable(oid, mob)
            ),
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

    def _encode_skills(self, char, now: float) -> np.ndarray:
        matrix = np.zeros((self.max_skills, len(self._skill_names)), dtype=np.float32)
        for slot, info in enumerate(self._panel_info):
            # Expiries are real time, the feature is game time like the logged cooldowns.
            left = max(0.0, self._cooldown_until.get(info.skill_id, 0.0) - now) * self.server_speed
            matrix[slot] = skill_features.encode_skill_slot(
                info, max_mp=char.max_mp, cur_mp=char.cur_mp, cooldown_left_s=left
            )
        return matrix

    def _window(self) -> dict[str, np.ndarray]:
        """The last ``seq_len`` rows as unbatched, left-padded model inputs."""
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

        return {
            "player_state": player,
            "delta_t": delta_t,
            "mobs_matrix": mobs,
            "skills_matrix": skills,
            "seq_mask": seq_mask,
        }

    def _infer(self, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batched = {name: value[None] for name, value in feeds.items()}
        started = time.perf_counter()
        action, pointer, skill = self.session.run(list(self.meta["outputs"]), batched)
        self.inference_ms.append((time.perf_counter() - started) * 1000.0)
        return action[0], pointer[0], skill[0]
