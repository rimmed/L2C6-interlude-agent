"""
Load-test worker (NOT for commit): runs N model-driven actors as threads in one process and
measures the server's response latency.

Latency probes (per actor):
* ``cool``  RequestSkillCoolTime (0x9D) -> SkillCoolTime (0xC1); a probe is also sent every
  ``--probe-every`` seconds, so every actor yields samples even when it does not cast;
* ``move``  MoveToLocation (0x01) -> CharMoveToLocation (0x01) for the actor's own object.
Client health:
* ``loop``  how late the actor's 50 ms loop iteration ran (client-side saturation, not server lag).

Every ``--report`` seconds the worker appends one JSON line with the raw samples of that window
to ``--out``; the ramp controller merges the workers.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402

# One intra-op thread per session: hundreds of sessions must not each spawn a pool of 20.
_Session = ort.InferenceSession


def _single_thread_session(model, sess_options=None, providers=None, **kw):
    opts = sess_options or ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return _Session(model, sess_options=opts, providers=providers, **kw)


ort.InferenceSession = _single_thread_session

from l2_actor import actor as actor_mod  # noqa: E402
from l2_actor.buffer import TransitionBuffer  # noqa: E402
from l2_client import packets  # noqa: E402
from l2_client.packets import GameServerPacket  # noqa: E402
from l2_client.tools.data import item_data, npc_data, skill_data  # noqa: E402


class _NoSignal:  # Actor.run touches signal handlers, which only the main thread may do
    SIGINT = 2
    SIG_IGN = None

    @staticmethod
    def signal(*_):
        return None


actor_mod.signal = _NoSignal

COOL_REQ = packets.GameClientPacket.REQUEST_SKILL_COOL_TIME
MOVE_REQ = packets.GameClientPacket.MOVE_TO_LOCATION


class Probe:
    """Shared sample sink of all actors of the worker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = {"cool": [], "move": [], "loop": []}
        self.events = {"connected": 0, "errors": 0, "disconnects": 0, "episodes": 0, "kills": 0, "deaths": 0,
                       "decisions": 0}
        self.error_texts: list[str] = []

    def add(self, kind: str, ms: float) -> None:
        with self.lock:
            self.samples[kind].append(round(ms, 2))

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.events[key] += n

    def take(self) -> tuple[dict, dict, list]:
        with self.lock:
            samples, self.samples = self.samples, {k: [] for k in self.samples}
            errors, self.error_texts = self.error_texts, []
            return samples, dict(self.events), errors


class NullBuffer(TransitionBuffer):
    """Load test: transitions are not kept (hundreds of actors would fill the memory)."""

    def add(self, transition) -> None:
        pass

    def flush(self) -> None:
        pass


class MeasuredActor(actor_mod.Actor):
    """Actor whose connection reports request -> answer latencies."""

    def __init__(self, cfg, probe: Probe, probe_every: float):
        super().__init__(cfg, buffer=NullBuffer())
        self.probe = probe
        self.probe_every = probe_every
        self._cool_sent: deque[float] = deque()
        self._move_sent: deque[float] = deque()
        self._last_probe = 0.0
        self._last_loop = 0.0

    def connect(self) -> None:
        super().connect()
        send = self.conn.send

        def measured_send(data: bytes) -> None:
            pid = data[2]
            now = time.time()
            if pid == COOL_REQ:
                self._cool_sent.append(now)
            elif pid == MOVE_REQ:
                self._move_sent.append(now)
            send(data)

        self.conn.send = measured_send
        # World / executor captured the bound method at construction: point them to the wrapper.
        self.world._send = measured_send
        self.executor._send = measured_send
        self.probe.count("connected")

    def _pump(self, timeout: float) -> None:
        now = time.time()
        for body in self.conn.poll(timeout):
            t = time.time()
            pid = body[0]
            if pid == GameServerPacket.SKILL_COOL_TIME and self._cool_sent:
                self.probe.add("cool", (t - self._cool_sent.popleft()) * 1000)
            elif pid == GameServerPacket.CHAR_MOVE_TO_LOCATION and self._move_sent and len(body) >= 5:
                # The server also moves the character on its own (approach, pickup): pair the answer
                # with the newest MoveToLocation only, and only if that was sent recently.
                if struct.unpack_from("<i", body, 1)[0] == self.world.char.object_id and t - self._move_sent[-1] < 3.0:
                    self.probe.add("move", (t - self._move_sent[-1]) * 1000)
                    self._move_sent.clear()
            for event in self.world.handle(body, now):
                self._on_event(event, now)

    def _tick(self, now: float) -> None:
        if self._last_loop:
            self.probe.add("loop", max(0.0, (now - self._last_loop - 0.05) * 1000))
        self._last_loop = now
        # Unanswered requests (e.g. a move the server rejected) must not shift the FIFO forever.
        for sent in (self._cool_sent, self._move_sent):
            while sent and now - sent[0] > 10.0:
                sent.popleft()
        if now - self._last_probe >= self.probe_every and not self._cool_sent:
            self.conn.send(packets.build_request_skill_cool_time())
            self._last_probe = now
        steps = self.total_steps
        super()._tick(now)
        if self.total_steps != steps:
            self.probe.count("decisions")

    def _end_fight(self, reason: str, *, truncated: bool = False) -> None:
        # a kill ends a fight, a death ends the fight and the life (_end_episode -> _end_fight)
        if not truncated and self.episode_step:
            self.probe.count("episodes")
            if reason in ("kill", "death"):
                self.probe.count("kills" if reason == "kill" else "deaths")
        super()._end_fight(reason, truncated=truncated)


def run_actor(cfg, probe: Probe, probe_every: float, stop: threading.Event) -> None:
    try:
        actor = MeasuredActor(cfg, probe, probe_every)
        actor._limit_reached = lambda: stop.is_set()  # noqa: SLF001 - the worker decides when to stop
        actor.run()
        if not stop.is_set():
            probe.count("disconnects")
    except Exception as exc:  # noqa: BLE001
        probe.count("errors")
        with probe.lock:
            probe.error_texts.append(f"{cfg.account}: {type(exc).__name__}: {exc}"[:300])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True, help="First account index")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--prefix", default="loadt")
    p.add_argument("--model", required=True)
    p.add_argument("--server-ip", default="192.168.1.200")
    p.add_argument("--server-speed", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", type=float, default=10.0)
    p.add_argument("--probe-every", type=float, default=2.0)
    p.add_argument("--stagger", type=float, default=0.5, help="Seconds between two logins")
    p.add_argument("--output-dir", default="/tmp/loadtest_traj")
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)

    # The datapack is loaded once for all threads; the actors' own load calls become no-ops.
    root = actor_mod.DATAPACK
    npc_data.load_npc_names(str(root / "npcs"))
    item_data.load_item_data(str(root / "items"))
    skill_data.load_skill_data(str(root / "skills"))
    for module, name in ((npc_data, "load_npc_names"), (item_data, "load_item_data"), (skill_data, "load_skill_data")):
        setattr(module, name, lambda *a, **k: None)

    probe, stop = Probe(), threading.Event()
    threads = []
    for i in range(args.start, args.start + args.count):
        cfg = actor_mod.ActorConfig(server_ip=args.server_ip, account=f"{args.prefix}{i:04d}", model=args.model,
                                    server_speed=args.server_speed, output_dir=args.output_dir, seed=i,
                                    store_next_state=False, flush_every=100_000)
        t = threading.Thread(target=run_actor, args=(cfg, probe, args.probe_every, stop), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(args.stagger)

    out = Path(args.out)
    try:
        while True:
            time.sleep(args.report)
            samples, events, errors = probe.take()
            alive = sum(t.is_alive() for t in threads)
            with out.open("a") as fh:
                fh.write(json.dumps({"t": time.time(), "start": args.start, "alive": alive, "events": events,
                                     "samples": samples, "errors": errors}) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
