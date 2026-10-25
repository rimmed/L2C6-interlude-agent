"""
Redis link between the actors and the RL trainer (``l2_NN/train_rl.py``).

Actors push trajectory *segments* to a Redis list; the trainer pops them, learns and publishes every
new policy back to Redis, where the actors pick it up between two segments.

Keys
----
=====================  ======================================================================
key                    content
=====================  ======================================================================
``l2:replay``          list of segments (``RPUSH`` by the actors, ``LPOP`` by the trainer); the
                       actors trim it to the newest ``max_queue`` items, so a stopped trainer
                       never fills the Redis memory
``l2:model:version``   integer, bumped by the trainer with every publish
``l2:model:onnx``      the ONNX policy (bytes)
``l2:model:meta``      its ``.json`` metadata (``l2_NN/export_onnx.py``)
=====================  ======================================================================

The three model keys are written in one ``MULTI`` and read with one ``MGET``, so an actor never
gets the weights of one version with the metadata of another.

Segment format
--------------
A segment is up to ``segment_len`` consecutive steps of one episode, packed with
:func:`numpy.savez_compressed` (``allow_pickle`` stays off: the trainer loads data from the network).
For ``T`` steps:

==================  ===============  ==================================================================
array               shape / dtype    content
==================  ===============  ==================================================================
``player_state``    S,L,Fp  f16      model inputs of every step (``S = T + 1`` when the segment ends
``delta_t``         S,L     f16      with a *bootstrap* state - the observation after the last step,
``mobs_matrix``     S,L,K,Fm f16     whose value the trainer uses when the episode goes on - and
``skills_matrix``   S,L,N,Fs f16     ``S = T`` when it ended with a kill / death)
``seq_mask``        S,L     bool
``action``          T int16          action-type id
``pointer``         T int16          mob slot of the pointer head, -1 = no monster in range
``skill``           T int16          skill slot of the skill head, -1 = no castable skill
``logp``            T f32            behaviour log-probability (``Decision.logp``)
``reward``          T f32            step reward (``reward.py``)
``type_mask``       T,A bool         executable action types at that step
``n_mobs``          T int16          valid pointer slots ``0..n_mobs-1``
``skill_mask``      T,N bool         castable skill slots
``pointer_mask``    T,K bool         mob slots the pointer head chose from (the attackers, or all ``n_mobs``)
``header``          uint8            UTF-8 JSON: ``actor_id``, ``episode_id`` (a life), ``segment``,
                                     ``model_version``, ``terminal``, ``done_reason``, ``kills`` in
                                     the segment, character info
==================  ===============  ==================================================================
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .buffer import Transition, TransitionBuffer

log = logging.getLogger(__name__)

REPLAY_KEY = "l2:replay"
MODEL_VERSION_KEY = "l2:model:version"
MODEL_ONNX_KEY = "l2:model:onnx"
MODEL_META_KEY = "l2:model:meta"

STATE_KEYS = ("player_state", "delta_t", "mobs_matrix", "skills_matrix", "seq_mask")


def connect(url: str):
    """A Redis client for ``url`` (``redis://host:6379/0``); fails fast when the server is unreachable."""
    import redis  # imported lazily: the local (JSONL) actor runs without the package

    client = redis.Redis.from_url(url, socket_timeout=30, socket_connect_timeout=10)
    client.ping()
    return client


# ----------------------------------------------------------------------
# Segment encoding
# ----------------------------------------------------------------------


@dataclass
class Segment:
    """Decoded segment (see the module docstring for the arrays)."""

    arrays: dict[str, np.ndarray]
    header: dict[str, Any]

    @property
    def steps(self) -> int:
        return int(self.arrays["action"].shape[0])

    @property
    def terminal(self) -> bool:
        return bool(self.header["terminal"])


def encode_segment(transitions: list[Transition], bootstrap: dict[str, np.ndarray] | None, header: dict[str, Any],
                   *, n_actions: int, max_skills: int) -> bytes:
    """
    Pack the steps of one segment.

    :param bootstrap: model input after the last step when the episode continues, ``None`` when the
        last step ended it (kill / death)
    """
    steps = len(transitions)
    states = [t.state for t in transitions] + ([bootstrap] if bootstrap is not None else [])
    arrays: dict[str, np.ndarray] = {}
    for key in STATE_KEYS:
        stacked = np.stack([s[key] for s in states])
        arrays[key] = stacked if stacked.dtype == bool else stacked.astype(np.float16)

    action = np.full(steps, -1, dtype=np.int16)
    pointer = np.full(steps, -1, dtype=np.int16)
    skill = np.full(steps, -1, dtype=np.int16)
    type_mask = np.zeros((steps, n_actions), dtype=bool)
    skill_mask = np.zeros((steps, max_skills), dtype=bool)
    n_mobs = np.zeros(steps, dtype=np.int16)
    pointer_mask = np.zeros((steps, states[0]["mobs_matrix"].shape[-2]), dtype=bool)
    for i, t in enumerate(transitions):
        action[i] = t.action["type_id"]
        pointer[i] = -1 if t.action["pointer_idx"] is None else t.action["pointer_idx"]
        skill[i] = -1 if t.action["skill_slot"] is None else t.action["skill_slot"]
        type_mask[i, t.action_mask["types"]] = True
        skill_mask[i, t.action_mask["skill_slots"]] = True
        n_mobs[i] = t.action_mask["n_mobs"]
        pointer_mask[i, t.action_mask.get("pointer_slots", list(range(n_mobs[i])))] = True
    arrays.update(
        action=action, pointer=pointer, skill=skill, type_mask=type_mask, skill_mask=skill_mask, n_mobs=n_mobs,
        pointer_mask=pointer_mask,
        logp=np.array([t.action["logp"] for t in transitions], dtype=np.float32),
        reward=np.array([t.reward for t in transitions], dtype=np.float32),
        header=np.frombuffer(json.dumps({**header, "terminal": bootstrap is None}).encode(), dtype=np.uint8),
    )
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def decode_segment(blob: bytes) -> Segment:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    header = json.loads(arrays.pop("header").tobytes().decode())
    steps = arrays["action"].shape[0]
    states = arrays["player_state"].shape[0]
    if steps == 0 or states != steps + (0 if header["terminal"] else 1):
        raise ValueError(f"malformed segment: {steps} steps, {states} states, terminal={header['terminal']}")
    return Segment(arrays, header)


# ----------------------------------------------------------------------
# Actor side
# ----------------------------------------------------------------------


@dataclass
class _Episode:
    transitions: list[Transition] = field(default_factory=list)
    segment: int = 0


class RedisBuffer(TransitionBuffer):
    """
    Sends an actor's transitions to the trainer as segments of ``segment_len`` steps.

    A segment is sent as soon as it is full (its last transition's ``next_state`` is the bootstrap
    state; without one the first step of the next segment provides it), or when the episode ends.
    An episode cut off without a next observation (actor stopped) loses its last step: there is
    nothing to bootstrap it from.
    """

    def __init__(self, client, *, actor_id: str, n_actions: int, max_skills: int, segment_len: int = 64,
                 max_queue: int = 20_000, header: dict[str, Any] | None = None):
        self.client = client
        self.actor_id = actor_id
        self.n_actions = n_actions
        self.max_skills = max_skills
        self.segment_len = max(1, segment_len)
        self.max_queue = max_queue
        #: extra header fields (character, class, level, model version) - the actor keeps it current
        self.header: dict[str, Any] = dict(header or {})
        self._episode = _Episode()
        self.sent_segments = 0
        self.sent_steps = 0
        self.sent_bytes = 0

    def add(self, transition: Transition) -> None:
        episode = self._episode
        if transition.episode_id != self.header.get("episode_id", transition.episode_id):
            episode = self._episode = _Episode()
        self.header["episode_id"] = transition.episode_id
        if transition.action.get("logp") is None:
            raise ValueError("RedisBuffer needs the behaviour log-probability in action['logp']")
        episode.transitions.append(transition)
        if transition.done:
            self._send(episode.transitions, None, transition.done_reason)
            episode.transitions = []
        elif len(episode.transitions) >= self.segment_len and transition.next_state is not None:
            self._send(episode.transitions, transition.next_state, None)
            episode.transitions = []
        elif len(episode.transitions) > self.segment_len:
            # The newest transition's state is the bootstrap of the full segment before it.
            full, rest = episode.transitions[: self.segment_len], episode.transitions[self.segment_len:]
            self._send(full, rest[0].state, None)
            episode.transitions = rest

    def flush(self) -> None:
        """End of episode: send what is left (cut off -> the last step becomes the bootstrap state)."""
        episode = self._episode
        if episode.transitions:
            last = episode.transitions[-1]
            if last.next_state is not None:
                self._send(episode.transitions, last.next_state, last.done_reason)
            elif len(episode.transitions) > 1:
                self._send(episode.transitions[:-1], last.state, last.done_reason)
        self._episode = _Episode()

    def _send(self, transitions: list[Transition], bootstrap: dict | None, reason: str | None) -> None:
        episode = self._episode
        header = {**self.header, "actor_id": self.actor_id, "segment": episode.segment, "done_reason": reason,
                  "kills": sum(1 for t in transitions if t.reward_terms.get("kill", 0) > 0), "sent_at": time.time()}
        blob = encode_segment(transitions, bootstrap, header, n_actions=self.n_actions, max_skills=self.max_skills)
        pipe = self.client.pipeline(transaction=False)
        pipe.rpush(REPLAY_KEY, blob)
        pipe.ltrim(REPLAY_KEY, -self.max_queue, -1)
        pipe.execute()
        episode.segment += 1
        self.sent_segments += 1
        self.sent_steps += len(transitions)
        self.sent_bytes += len(blob)
        log.debug("Sent segment %d of episode %s: %d steps, %d bytes", episode.segment - 1,
                  self.header.get("episode_id"), len(transitions), len(blob))


def fetch_model(client, known_version: int | None = None) -> tuple[int, bytes, dict] | None:
    """
    The published policy as ``(version, onnx_bytes, metadata)``; ``None`` when nothing is published
    or the published version is ``known_version``.
    """
    version = client.get(MODEL_VERSION_KEY)
    if version is None or (known_version is not None and int(version) == known_version):
        return None
    version, onnx_bytes, meta = client.mget(MODEL_VERSION_KEY, MODEL_ONNX_KEY, MODEL_META_KEY)
    if onnx_bytes is None or meta is None:
        return None
    return int(version), onnx_bytes, json.loads(meta)


# ----------------------------------------------------------------------
# Trainer side
# ----------------------------------------------------------------------


def publish_model(client, onnx_bytes: bytes, meta: dict) -> int:
    """Publish a new policy; returns its version."""
    version = int(client.get(MODEL_VERSION_KEY) or 0) + 1
    meta = {**meta, "rl_version": version}
    pipe = client.pipeline(transaction=True)
    pipe.set(MODEL_ONNX_KEY, onnx_bytes)
    pipe.set(MODEL_META_KEY, json.dumps(meta))
    pipe.set(MODEL_VERSION_KEY, version)
    pipe.execute()
    return version


def pop_segments(client, max_items: int = 64) -> list[bytes]:
    """Up to ``max_items`` queued segments, oldest first (non-blocking)."""
    return client.lpop(REPLAY_KEY, max_items) or []
