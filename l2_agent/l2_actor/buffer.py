"""
Transitions and the buffers that store them.

A :class:`Transition` is one decision step of the actor. The actor only talks to the abstract
:class:`TransitionBuffer` (``add`` / ``end_episode`` / ``close``), so the local
:class:`JsonlBuffer` can later be swapped for a Redis-backed buffer without touching the actor.

Record layout (one JSON object per line)
----------------------------------------
=================  ==========================================================================
field              content
=================  ==========================================================================
``actor_id``       id of the actor process (account name)
``episode_id``     life counter of this actor: the RL episode is one life, a kill is a reward
                   inside it and only a death ends it (``episodes_*.jsonl`` summarises the fights)
``step``           step index inside the life
``t``              unix time of the observation
``state``          model input of the step: ``player_state`` (L x F_p), ``delta_t`` (L),
                   ``mobs_matrix`` (L x K x F_m), ``skills_matrix`` (L x N x F_s), ``seq_mask`` (L);
                   the same tensors the ONNX model consumes, left-padded windows of L = seq_len
``action``         ``type`` / ``type_id`` (action head), ``pointer_idx`` (mob slot, pointer head),
                   ``skill_slot`` (skill head), the resolved ``mob_object_id`` / ``skill_id``,
                   ``confidence``, the ``model`` file that chose it and what was ``executed``
``action_mask``    what was allowed: ``types`` (valid action ids), ``n_mobs`` (valid pointer
                   slots ``0..n_mobs-1``), ``skill_slots`` (castable skill slots)
``reward``         scalar reward of the step, ``reward_terms`` its components (see ``reward.py``)
``next_state``     ``state`` of the next step (``null`` for terminal and truncated steps and for
                   the killing step - its successor is the next record of the life)
``done``           the life ended with this step; ``done_reason`` is ``death``
``truncated``      the episode was cut off (actor stopped, target lost) - not a terminal state
``info``           diagnostics: character, level, class, HP/MP ratios, position, target
=================  ==========================================================================
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _encode_tensors(tensors: dict[str, np.ndarray] | None, decimals: int) -> dict[str, Any] | None:
    if tensors is None:
        return None
    encoded = {}
    for name, value in tensors.items():
        if value.dtype == bool:
            encoded[name] = value.tolist()
        else:
            encoded[name] = np.round(value.astype(np.float64), decimals).tolist()
    return encoded


@dataclass
class Transition:
    """One decision step: what the actor saw, did and got."""

    actor_id: str
    episode_id: int
    step: int
    t: float
    state: dict[str, np.ndarray]
    action: dict[str, Any]
    action_mask: dict[str, Any]
    reward: float = 0.0
    reward_terms: dict[str, float] = field(default_factory=dict)
    next_state: dict[str, np.ndarray] | None = None
    done: bool = False
    truncated: bool = False
    done_reason: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_record(self, decimals: int = 5) -> dict[str, Any]:
        """JSON-ready dict; features are rounded to ``decimals`` places (they are ratios / log-times)."""
        return {
            "actor_id": self.actor_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "t": round(self.t, 3),
            "state": _encode_tensors(self.state, decimals),
            "action": self.action,
            "action_mask": self.action_mask,
            "reward": round(self.reward, 6),
            "reward_terms": {name: round(value, 6) for name, value in self.reward_terms.items()},
            "next_state": _encode_tensors(self.next_state, decimals),
            "done": self.done,
            "truncated": self.truncated,
            "done_reason": self.done_reason,
            "info": self.info,
        }


class TransitionBuffer(ABC):
    """Where the actor puts its transitions."""

    @abstractmethod
    def add(self, transition: Transition) -> None:
        """Store one transition (may be buffered)."""

    def end_episode(self) -> None:
        """Called after the last transition of an episode; buffers flush here by default."""
        self.flush()

    @abstractmethod
    def flush(self) -> None:
        """Persist everything buffered so far."""

    def close(self) -> None:
        self.flush()


class JsonlBuffer(TransitionBuffer):
    """
    Local buffer: keeps transitions in memory and appends them to a ``.jsonl`` file every
    ``flush_every`` transitions and at the end of every episode.
    """

    def __init__(self, path: str | Path, *, flush_every: int = 256, decimals: int = 5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, flush_every)
        self.decimals = decimals
        self._pending: list[Transition] = []
        self.written = 0

    def add(self, transition: Transition) -> None:
        self._pending.append(transition)
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for transition in self._pending:
                fh.write(json.dumps(transition.to_record(self.decimals), separators=(",", ":")) + "\n")
        self.written += len(self._pending)
        log.debug("Flushed %d transition(s) to %s (%d total)", len(self._pending), self.path, self.written)
        self._pending.clear()
