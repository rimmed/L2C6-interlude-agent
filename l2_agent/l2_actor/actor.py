"""
L2 autonomous combat actor: plays one agent character and records its combat trajectories.

The actor logs in (a fresh account gets a random agent character from the RL-environment server:
random class and level, matching gear, farm-zone teleport), then repeats combat episodes - select a
monster, fight it, loot - and stores every decision step as a transition (state, action, reward,
next state, done) in a local buffer. When the character dies the server revives it in its farm zone;
the actor ends the episode and simply waits for that.

Run (from ``l2_agent/``)::

    # trained model on the CPU, a new random agent character, stop after 100 episodes
    uv run python -m l2_actor --model checkpoints/combat_policy.onnx --max-episodes 100

    # keep playing the same character
    uv run python -m l2_actor --account rlactor7k2p9q --password 'actor!2026' --model checkpoints/combat_policy.onnx

    # RL: send the trajectories to the trainer (l2_NN/train_rl.py) and play its latest policy
    uv run python -m l2_actor --redis redis://192.168.1.10:6379/0 --model redis --server-speed 5

Output (``--output-dir``, default ``trajectories/``):

* ``transitions_<actor>.jsonl`` - one transition per line, layout in ``l2_actor/buffer.py``
  (with ``--redis`` the transitions go to Redis instead, layout in ``l2_actor/replay.py``);
* ``episodes_<actor>.jsonl``    - one summary per finished episode (return, length, reason, kills).

With ``--redis`` the actor samples its actions from the policy (exploration; ``--greedy`` turns
that off) and checks for a newer published policy every ``--model-poll`` seconds; the new weights
are swapped in without leaving the game.

The server's ``TimeScale`` must be passed as ``--server-speed`` (default 1 = no acceleration).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import string
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    # ``uv run l2_actor/actor.py``: make the sibling packages importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l2_actor import replay
from l2_actor.buffer import JsonlBuffer, Transition, TransitionBuffer
from l2_actor.executor import ActionExecutor, ExecutorConfig, find_healing_potion
from l2_actor.reward import RewardConfig
from l2_actor.world import WorldState
from l2_client import packets
from l2_client.connection import GameConnection, login
from l2_client.policy import CombatPolicy, Decision, Observation
from l2_client.tools.data import item_data, npc_data, skill_data

log = logging.getLogger("l2_actor")

AGENT_ROOT = Path(__file__).resolve().parents[1]
DATAPACK = AGENT_ROOT.parent / "L2J_Mobius" / "L2J_Mobius_CT_0_Interlude" / "dist" / "game" / "data" / "stats"

#: Human, Elven and Dark Mystic (levels 1-19), the weak bracket of K/D: with MP to spare they hit with the staff in
#: 30% of their actions and cast in 12% (K/D 1.1 at levels 10-19 against 16.8 of the fighters; 2.0 with these rules).
#: Heal, Cure Poison, Group Heal, Might, Curse: Weakness, Curse: Poison and Wind Shackle are left off the panel, and
#: a base mystic does not hit with its weapon nor switch from a live target while it can cast an attack spell.
BASE_MYSTICS = frozenset((10, 25, 38))
_BASE_MYSTIC_SUPPORT = frozenset((1011, 1012, 1027, 1068, 1164, 1168, 1206))
BASE_MYSTIC_EXCLUDED_SKILLS = dict.fromkeys(BASE_MYSTICS, _BASE_MYSTIC_SUPPORT)


@dataclass
class ActorConfig:
    """Settings of one actor process (see ``parse_args`` for the meaning of each field)."""

    server_ip: str = "192.168.1.200"
    login_port: int = 2106
    account: str | None = None
    password: str = "actor!2026"
    char_slot: int = 0
    model: str = "checkpoints/combat_policy.onnx"
    redis: str | None = None
    #: sample actions from the policy; ``None`` = sample with ``redis`` (RL), argmax without
    sample: bool | None = None
    segment_len: int = 64
    max_queue: int = 20_000
    model_poll: float = 10.0
    model_wait: float = 600.0
    device: str = "cpu"
    server_speed: float = 1.0
    decision_interval: float = 0.5
    output_dir: str = "trajectories"
    flush_every: int = 256
    store_next_state: bool = True
    max_episodes: int = 0
    max_steps: int = 0
    max_minutes: float = 0.0
    setup_timeout: float = 20.0
    settle_time: float = 6.0
    respawn_timeout: float = 60.0
    seed: int | None = None
    #: drink shop buff potions (run / attack / casting speed) when no such buff is on the character
    buff_potions: bool = True
    #: sit down to regenerate when low on MP / HP out of combat (ExecutorConfig.rest_*)
    rest: bool = True
    #: base mystics cast no heals / cures / debuffs (BASE_MYSTIC_EXCLUDED_SKILLS) and nuke instead of meleeing
    nuke_only_base_mystics: bool = True
    #: onnxruntime threads of the model (``None`` = all cores; ``l2_actor.fleet`` uses 1)
    threads: int | None = None


class Actor:
    """One character, one connection, one trajectory buffer."""

    def __init__(self, cfg: ActorConfig, buffer: TransitionBuffer | None = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.account = cfg.account or "rlactor" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=6))
        self.actor_id = self.account
        out = Path(cfg.output_dir)
        self.redis = replay.connect(cfg.redis) if cfg.redis else None
        if cfg.model == "redis" and self.redis is None:
            raise ValueError("--model redis needs --redis")
        # The Redis buffer needs the model's label space, so it is created once the model is loaded.
        self.buffer = buffer
        if buffer is None and self.redis is None:
            self.buffer = JsonlBuffer(out / f"transitions_{self.actor_id}.jsonl", flush_every=cfg.flush_every)
        self.episodes_path = out / f"episodes_{self.actor_id}.jsonl"
        self.sample = cfg.sample if cfg.sample is not None else self.redis is not None
        self.np_rng = np.random.default_rng(cfg.seed)
        self.model_version: int | None = None
        self._last_model_poll = 0.0
        self.reward = RewardConfig()

        self.conn: GameConnection | None = None
        self.encoder: CombatPolicy | None = None
        self.world: WorldState | None = None
        self.executor: ActionExecutor | None = None
        self.model_name = Path(cfg.model).name

        self._pending: Transition | None = None
        self._last_decision = 0.0
        self._restart_requested = False
        #: when to ask for a fresh ItemList: the server tops up shots, arrows and potions after login / revive
        self._item_list_at = 0.0
        self._stop = threading.Event()
        self.started = time.time()
        #: fights (kill / death) - the episode summaries; the history window restarts with each fight
        self.episode_id = 0
        self.episode_step = 0
        #: lives - the RL episodes: a kill is a reward inside a life, only a death (or stopping) ends it
        self.life_id = 0
        self.life_step = 0
        self.episode_return = 0.0
        self.episode_terms: dict[str, float] = {}
        #: decisions of the current fight per action type, and per skill (``"<id>"``, ``"pet:<id>"`` for the summon's)
        self.episode_actions: dict[str, int] = {}
        self.episode_skills: dict[str, int] = {}
        #: target choices of the current fight by distance rank (``"0"`` = the nearest monster on the radar)
        self.episode_target_ranks: dict[str, int] = {}
        #: decisions of the current fight with a living summon
        self.episode_pet_steps = 0
        self.episode_started = 0.0
        self.total_steps = 0
        self.totals = {"episodes": 0, "kills": 0, "deaths": 0, "transitions": 0, "return": 0.0}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Log in, enter the world and wait until the server finished setting the agent up."""
        cfg = self.cfg
        npc_data.load_npc_names(str(DATAPACK / "npcs"))
        item_data.load_item_data(str(DATAPACK / "items"))
        skill_data.load_skill_data(str(DATAPACK / "skills"))

        # The model decides; its .json (feature layout, skill panels) must sit next to the .onnx.
        model_path = self._download_model() if cfg.model == "redis" else cfg.model
        self.encoder = CombatPolicy(model_path, server_speed=cfg.server_speed, device=cfg.device, threads=cfg.threads,
                                    class_excluded_skills=BASE_MYSTIC_EXCLUDED_SKILLS if cfg.nuke_only_base_mystics else None,
                                    caster_classes=BASE_MYSTICS if cfg.nuke_only_base_mystics else frozenset())
        log.info("Model: %s (version %s) on %s, %s actions", model_path, self.model_version, self.encoder.device,
                 "sampled" if self.sample else "greedy")
        if self.buffer is None:
            self.buffer = replay.RedisBuffer(
                self.redis, actor_id=self.actor_id, n_actions=len(self.encoder.meta["action_to_id"]),
                max_skills=self.encoder.meta["max_skills"], segment_len=cfg.segment_len, max_queue=cfg.max_queue,
            )

        ticket = login(cfg.server_ip, self.account, cfg.password, port=cfg.login_port)
        self.conn = GameConnection(ticket, char_slot=cfg.char_slot)
        self.conn.enter_world()
        self.world = WorldState(self.encoder, self.conn.send, cfg.server_speed)
        self.executor = ActionExecutor(self.conn.send, self.world, self.encoder,
                                       ExecutorConfig(), rng=self.rng)

        # 1) the character data; 2) the RL-environment setup (level boost, class transfer, gear,
        # farm-zone teleport) runs right after login - give it time to land before acting.
        deadline = time.time() + cfg.setup_timeout
        while not self.world.ready:
            if time.time() > deadline:
                raise TimeoutError("no UserInfo / SkillList from the server")
            self._pump(0.2)
        settle_until = time.time() + cfg.settle_time
        while time.time() < settle_until:
            teleports = self.world.teleports
            self._pump(0.2)
            if self.world.teleports > teleports:  # in the farm zone; let the nearby objects arrive
                settle_until = min(settle_until, time.time() + 1.5)
        if self.world.home is None:  # no teleport (standard server): roam around the login position
            self.world.home = self.world.pos
        self.encoder.bind_character(self.world.char)
        char = self.world.char
        log.info("Playing '%s' (account %s): level %d, class %d, at %s", self.conn.character_name, self.account,
                 char.level, char.class_id, self.world.pos)
        self._update_header()
        self._item_list_at = time.time() + 2.0

    def _download_model(self) -> Path:
        """Wait for the trainer's policy in Redis and store it under ``output_dir/models/``."""
        deadline = time.time() + self.cfg.model_wait
        while (published := replay.fetch_model(self.redis)) is None:
            if time.time() > deadline:
                raise TimeoutError(f"no model published in Redis within {self.cfg.model_wait:.0f}s")
            log.info("Waiting for the trainer to publish a model...")
            time.sleep(5.0)
        version, onnx_bytes, meta = published
        # One file per model version, shared by every actor of the fleet (a copy per actor was 1.3 GB
        # per fleet start); written under a temporary name and renamed, so no actor reads half a file.
        folder = Path(self.cfg.output_dir) / "models"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"combat_policy_v{version}.onnx"
        if not path.exists() or not path.with_suffix(".json").exists():
            tmp = folder / f".{self.actor_id}.tmp"
            tmp.with_suffix(".onnx").write_bytes(onnx_bytes)
            tmp.with_suffix(".json").write_text(json.dumps(meta))
            tmp.with_suffix(".json").replace(path.with_suffix(".json"))
            tmp.with_suffix(".onnx").replace(path)
        self.model_version = version
        self.model_name = f"redis:v{version}"
        return path

    def _poll_model(self, now: float) -> None:
        """Swap in a newer published policy (keeps the game state and the event history)."""
        if self.redis is None or self.cfg.model != "redis" or now - self._last_model_poll < self.cfg.model_poll:
            return
        self._last_model_poll = now
        try:
            published = replay.fetch_model(self.redis, self.model_version)
        except Exception as exc:  # a Redis hiccup must not stop the game
            log.warning("Model poll failed: %s", exc)
            return
        if published is None:
            return
        version, onnx_bytes, meta = published
        self.encoder.reload(onnx_bytes, meta)
        log.info("Loaded model version %d (was %s)", version, self.model_version)
        self.model_version = version
        self.model_name = f"redis:v{version}"
        self._update_header()

    def _update_header(self) -> None:
        if isinstance(self.buffer, replay.RedisBuffer):
            char = self.world.char
            self.buffer.header.update(model_version=self.model_version, character=self.conn.character_name,
                                      level=char.level, class_id=char.class_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Play until a limit is reached or the process is interrupted; return the run summary."""
        try:
            self.connect()
            self.episode_started = time.time()
            while not self._limit_reached():
                self._pump(0.05)
                self._tick(time.time())
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not lose buffered data
            self._shutdown()
        return self.summary()

    def _pump(self, timeout: float) -> None:
        now = time.time()
        for body in self.conn.poll(timeout):
            for event in self.world.handle(body, now):
                self._on_event(event, now)

    def _tick(self, now: float) -> None:
        world = self.world
        world.char.radar.prune(*world.xy)
        if world.dead:
            self._respawn_fallback(now)
            return
        if self._item_list_at and now >= self._item_list_at:
            self.conn.send(packets.build_request_item_list())
            self._item_list_at = 0.0
        self.executor.sync_cooldowns(now)
        if self.cfg.buff_potions:
            self.executor.drink_buff_potions(now)
        self.executor.drop_stuck_target(now)
        if self.cfg.rest and self.executor.rest(now):
            return
        if now - self._last_decision >= self.cfg.decision_interval and not self.executor.busy(now):
            self._poll_model(now)
            self._step(now)

    def _step(self, now: float) -> None:
        world = self.world
        obs = self.encoder.observe(world.char, world.target_id, world.xy, has_loot=bool(world.pending_loot),
                                   has_potion=find_healing_potion(world.char) is not None, now=now)
        if self._pending is not None:
            self._finish(self._pending, next_state=obs.feeds)

        decision = self.encoder.act(obs, sample=self.sample, rng=self.np_rng)
        executed = self.executor.execute(decision, now)
        self.episode_actions[decision.action] = self.episode_actions.get(decision.action, 0) + 1
        if self.encoder._live_pet(world.char) is not None:
            self.episode_pet_steps += 1
        if decision.action in ("used_skill", "pet_skill") and decision.skill_id is not None:
            key = f"pet:{decision.skill_id}" if decision.action == "pet_skill" else str(decision.skill_id)
            self.episode_skills[key] = self.episode_skills.get(key, 0) + 1
        if decision.action == "selected_target" and decision.pointer_slot is not None:
            rank = str(decision.pointer_slot)  # the radar slots are sorted by distance
            self.episode_target_ranks[rank] = self.episode_target_ranks.get(rank, 0) + 1
        log.debug("[ep %d step %d] %s -> %s", self.episode_id, self.episode_step, decision.describe(), executed)

        self._pending = Transition(
            actor_id=self.actor_id, episode_id=self.life_id, step=self.life_step, t=now,
            state=obs.feeds, action=self._action_record(decision, executed), action_mask=self._mask_record(obs, decision),
            info=self._info_record(),
        )
        self.episode_step += 1
        self.life_step += 1
        self.total_steps += 1
        self._last_decision = now

    def _on_event(self, event: str, now: float) -> None:
        if event == "kill":
            self.totals["kills"] += 1
            # The kill reward goes to the step that killed; the life (RL episode) goes on, so the step
            # is not terminal - its value is bootstrapped from the next observation.
            if self._pending is not None:
                self._finish(self._pending)
                self._pending = None
            self._end_fight("kill")
        elif event == "death":
            self.totals["deaths"] += 1
            self._end_episode("death")
        elif event == "skills_changed" and self.world.ready and self.encoder.panel:
            self.encoder.bind_character(self.world.char)
        elif event == "teleport":
            self.encoder.reset_session()
        elif event == "revive":
            self._restart_requested = False
            self._item_list_at = now + 2.0
            log.info("Revived by the server")

    def _respawn_fallback(self, now: float) -> None:
        """The RL-environment server revives agents in their zone; only a standard server needs this."""
        if not self._restart_requested and now - self.world.dead_since > self.cfg.respawn_timeout:
            log.warning("Not revived after %.0fs -- is RLRespawnOnDeath enabled? Requesting the town respawn.",
                        self.cfg.respawn_timeout)
            self.conn.send(packets.build_request_restart_point(0))
            self._restart_requested = True

    # ------------------------------------------------------------------
    # Transitions and episodes
    # ------------------------------------------------------------------

    def _finish(self, transition: Transition, *, next_state: dict | None = None, done: bool = False,
                truncated: bool = False, reason: str | None = None) -> None:
        """Attach reward / next state / done to a transition and hand it to the buffer."""
        events = self.world.events.take()
        transition.reward, transition.reward_terms = self.reward.compute(events, self.world.char.max_hp)
        transition.next_state = next_state if (self.cfg.store_next_state and not done and not truncated) else None
        transition.done, transition.truncated, transition.done_reason = done, truncated, reason
        self.buffer.add(transition)
        self.episode_return += transition.reward
        for name, value in transition.reward_terms.items():
            self.episode_terms[name] = self.episode_terms.get(name, 0.0) + value
        self.totals["transitions"] += 1
        self.totals["return"] += transition.reward

    def _end_episode(self, reason: str, *, truncated: bool = False) -> None:
        """End of a life: a death (terminal) or the actor stopping (truncated)."""
        if self._pending is not None:
            self._finish(self._pending, done=not truncated, truncated=truncated, reason=reason)
            self._pending = None
        self._end_fight(reason, truncated=truncated)
        self.buffer.end_episode()
        self.life_id += 1
        self.life_step = 0
        self._update_header()

    def _end_fight(self, reason: str, *, truncated: bool = False) -> None:
        """Write the summary of one fight and restart the history window, like a logged combat session."""
        if self.episode_step:
            char = self.world.char
            summary = {
                "actor_id": self.actor_id, "episode_id": self.episode_id, "life_id": self.life_id, "reason": reason,
                "truncated": truncated,
                "steps": self.episode_step, "return": round(self.episode_return, 4),
                "reward_terms": {name: round(value, 4) for name, value in self.episode_terms.items()},
                "actions": self.episode_actions, "skills": self.episode_skills,
                "target_ranks": self.episode_target_ranks, "panel": self.encoder.panel_summary(),
                "pet_share": round(self.episode_pet_steps / self.episode_step, 3),
                "duration_s": round(time.time() - self.episode_started, 2),
                "rest_s": round(self.executor.rest_s, 1),
                "mp_end": round(char.cur_mp / char.max_mp, 3) if char.max_mp else None,
                "character": self.conn.character_name, "level": char.level, "class_id": char.class_id,
                "t": round(time.time(), 1),
            }
            self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
            with self.episodes_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")
            self.totals["episodes"] += 1
            log.info("Episode %d: %s after %d steps, return %.3f (%.1fs)", self.episode_id, reason,
                     self.episode_step, self.episode_return, summary["duration_s"])
            self.episode_id += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_terms = {}
        self.episode_actions = {}
        self.episode_skills = {}
        self.episode_target_ranks = {}
        self.episode_pet_steps = 0
        self.episode_started = time.time()
        self.executor.rest_s = 0.0
        self.encoder.reset_session()

    def stop(self) -> None:
        """Ask a running actor (e.g. in another thread) to finish its episode bookkeeping and log out."""
        self._stop.set()

    def _limit_reached(self) -> bool:
        cfg = self.cfg
        return bool(
            self._stop.is_set()
            or (cfg.max_episodes and self.totals["episodes"] >= cfg.max_episodes)
            or (cfg.max_steps and self.total_steps >= cfg.max_steps)
            or (cfg.max_minutes and time.time() - self.started >= cfg.max_minutes * 60)
        )

    def _shutdown(self) -> None:
        if self.world is not None and self._pending is not None:
            self._end_episode("stopped", truncated=True)
        if self.buffer is not None:
            self.buffer.close()
        if self.conn is not None:
            self.conn.close()

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    def _action_record(self, decision: Decision, executed: str) -> dict[str, Any]:
        return {
            "type": decision.action, "type_id": decision.action_id,
            "pointer_idx": decision.pointer_slot, "skill_slot": decision.skill_slot,
            "mob_object_id": decision.mob_object_id, "skill_id": decision.skill_id,
            "confidence": round(decision.confidence, 4), "logp": decision.logp, "model": self.model_name,
            "model_version": self.model_version, "executed": executed,
        }

    def _mask_record(self, obs: Observation, decision: Decision) -> dict[str, Any]:
        return {
            "types": [action_id for name, action_id in self.encoder.executable_actions if obs.valid[name]],
            "n_mobs": len(obs.mob_ids),
            # the mob slots the pointer head chose from (the monsters hitting the character, or all)
            "pointer_slots": list(obs.pointer_choices()),
            # the slots the skill head chose from: the summon's skills for pet_skill, the character's otherwise
            "skill_slots": list(obs.skill_slots(decision.action)),
        }

    def _info_record(self) -> dict[str, Any]:
        char, world = self.world.char, self.world
        return {
            "character": self.conn.character_name, "level": char.level, "class_id": char.class_id,
            "hp_ratio": round(char.cur_hp / char.max_hp, 4) if char.max_hp else None,
            "mp_ratio": round(char.cur_mp / char.max_mp, 4) if char.max_mp else None,
            "pos": list(world.pos), "target_id": world.target_id, "loot_on_ground": len(world.pending_loot),
        }

    def summary(self) -> dict[str, Any]:
        minutes = (time.time() - self.started) / 60
        result = {"actor_id": self.actor_id, "minutes": round(minutes, 2), "steps": self.total_steps, **self.totals}
        result["return"] = round(result["return"], 4)
        if isinstance(self.buffer, replay.RedisBuffer):
            result.update(sent_segments=self.buffer.sent_segments, sent_steps=self.buffer.sent_steps,
                          sent_mb=round(self.buffer.sent_bytes / 2**20, 2), model_version=self.model_version)
        if self.encoder is not None and self.encoder.session is not None:
            inference = self.encoder.timing_summary()["inference_ms"]
            if inference.get("count"):
                result["inference_ms_mean"] = round(inference["mean"], 3)
        return result


def parse_args(argv: list[str] | None = None) -> ActorConfig:
    d = ActorConfig()
    p = argparse.ArgumentParser(description="L2 autonomous combat actor (trajectory collection).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--server-ip", default=d.server_ip, help="Login server address")
    p.add_argument("--login-port", type=int, default=d.login_port)
    p.add_argument("--account", default=None,
                   help="Account to play; default: a new random 'rlactor…' account, i.e. a new random agent character")
    p.add_argument("--password", default=d.password)
    p.add_argument("--char-slot", type=int, default=d.char_slot)
    p.add_argument("--model", required=True,
                   help="ONNX combat model (l2_NN/export_onnx.py; its .json must sit next to it), "
                        "or 'redis' for the trainer's latest published policy")
    p.add_argument("--redis", default=None,
                   help="Redis URL (redis://host:6379/0): send the trajectories to the RL trainer instead of .jsonl")
    p.add_argument("--greedy", dest="sample", action="store_false", default=None,
                   help="Take the most likely action instead of sampling (default: sample with --redis)")
    p.add_argument("--sample", dest="sample", action="store_true", help="Sample actions even without --redis")
    p.add_argument("--segment-len", type=int, default=d.segment_len, help="Steps per segment sent to Redis")
    p.add_argument("--max-queue", type=int, default=d.max_queue,
                   help="Trim the Redis replay list to this many segments (protects Redis when no trainer runs)")
    p.add_argument("--model-poll", type=float, default=d.model_poll,
                   help="Seconds between checks for a newer published model (--model redis)")
    p.add_argument("--model-wait", type=float, default=d.model_wait,
                   help="How long to wait for the first published model (--model redis)")
    p.add_argument("--device", choices=("cpu", "gpu"), default=d.device, help="Where the model runs")
    p.add_argument("--server-speed", type=float, default=d.server_speed, help="Server TimeScale (General.ini)")
    p.add_argument("--decision-interval", type=float, default=d.decision_interval, help="Seconds between decisions")
    p.add_argument("--output-dir", default=d.output_dir, help="Where the .jsonl files go")
    p.add_argument("--flush-every", type=int, default=d.flush_every, help="Flush the buffer every N transitions")
    p.add_argument("--no-next-state", dest="store_next_state", action="store_false",
                   help="Do not store next_state (it equals the state of the next record of the episode)")
    p.add_argument("--max-episodes", type=int, default=0, help="Stop after N finished episodes (0 = no limit)")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N decisions (0 = no limit)")
    p.add_argument("--max-minutes", type=float, default=0.0, help="Stop after N minutes (0 = no limit)")
    p.add_argument("--respawn-timeout", type=float, default=d.respawn_timeout,
                   help="Request a town respawn if the server has not revived the character after this many seconds")
    p.add_argument("--seed", type=int, default=None, help="Seed of the actor's own randomness (account name, roaming)")
    p.add_argument("-v", "--verbose", action="store_true", help="Log every decision")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return ActorConfig(
        server_ip=args.server_ip, login_port=args.login_port, account=args.account, password=args.password,
        char_slot=args.char_slot, model=args.model, device=args.device, redis=args.redis, sample=args.sample,
        segment_len=args.segment_len, max_queue=args.max_queue, model_poll=args.model_poll,
        model_wait=args.model_wait,
        server_speed=args.server_speed, decision_interval=args.decision_interval, output_dir=args.output_dir,
        flush_every=args.flush_every, store_next_state=args.store_next_state, max_episodes=args.max_episodes,
        max_steps=args.max_steps, max_minutes=args.max_minutes, respawn_timeout=args.respawn_timeout, seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    actor = Actor(parse_args(argv))
    summary = actor.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
