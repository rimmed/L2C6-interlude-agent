"""
Run many actors on one machine: N actors in worker processes of ``--per-proc`` actor threads each.

One Python process per actor wastes memory, one process for hundreds of actors starves them on
the GIL (late socket reads, server disconnects). Load tests on a 20-core machine
(``loadtest/``): 10 actors per process ran 800 actors at ~45-55 % CPU with p95 server latency
< 7 ms, 25 per process already lost connections at 600.

Every worker process loads the datapack once, gives each model session one onnxruntime thread
and restarts an actor that crashed or lost its connection (same account, after
``--restart-delay`` seconds). Logins are spaced ``--stagger`` seconds apart over the whole fleet.

Run (from ``l2_agent/``)::

    # 800 RL actors against the trainer's Redis, server TimeScale 5
    uv run python -m l2_actor.fleet --actors 800 --redis redis://192.168.1.144:6379/0 --model redis --server-speed 5

Accounts are ``<prefix><index>`` (``rlfleet0000`` ...), so a restarted fleet plays the same
characters. Every ``--report`` seconds the fleet prints one summary line (actors online, episodes,
kills / deaths, transitions sent, model versions in use). Ctrl+C / SIGTERM stops all actors cleanly.
Other options (``--decision-interval``, ``--segment-len`` ...) are the ones of ``python -m l2_actor``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l2_actor import actor as actor_mod
from l2_actor.actor import Actor, ActorConfig
from l2_actor.buffer import NullBuffer
from l2_client.tools.data import item_data, npc_data, skill_data

log = logging.getLogger("l2_actor.fleet")


# ----------------------------------------------------------------------
# Worker process: --per-proc actor threads
# ----------------------------------------------------------------------


class Worker:
    """The actors of one process; restarts the ones that stop on their own."""

    def __init__(self, base: ActorConfig, accounts: list[str], stagger: float, restart_delay: float,
                 episodes_only: bool = False):
        self.base = base
        self.episodes_only = episodes_only
        self.accounts = accounts
        self.stagger = stagger
        self.restart_delay = restart_delay
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.actors: dict[str, Actor] = {}
        self.totals: Counter = Counter()

    def run_account(self, account: str, index: int) -> None:
        while not self.stop.is_set():
            cfg = replace(self.base, account=account, seed=index)
            actor = Actor(cfg, buffer=NullBuffer() if self.episodes_only else None)
            with self.lock:
                self.actors[account] = actor
            try:
                summary = actor.run()
                if self.stop.is_set():
                    break
                log.warning("%s stopped on its own (%s); restarting", account, summary.get("steps"))
                self.count("restarts")
            except Exception as exc:  # noqa: BLE001 - any failure of one actor must not stop the others
                log.warning("%s failed: %s: %s; restarting in %.0fs", account, type(exc).__name__, exc, self.restart_delay,
                            exc_info=True)
                self.count("restarts")
            finally:
                self.collect(actor)
            self.stop.wait(self.restart_delay)

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.totals[key] += n

    def collect(self, actor: Actor) -> None:
        """Fold the counters of a finished actor run into the worker totals."""
        with self.lock:
            for key in ("episodes", "kills", "deaths", "transitions"):
                self.totals[key] += actor.totals[key]
            self.totals["sent_segments"] += getattr(actor.buffer, "sent_segments", 0)
            if self.actors.get(actor.account) is actor:
                del self.actors[actor.account]

    def status(self) -> dict:
        with self.lock:
            actors = list(self.actors.values())
            totals = Counter(self.totals)
        online = [a for a in actors if a.conn is not None and a.world is not None and not a._stop.is_set()]
        for a in actors:  # live counters of the running actors
            for key in ("episodes", "kills", "deaths", "transitions"):
                totals[key] += a.totals[key]
            totals["sent_segments"] += getattr(a.buffer, "sent_segments", 0)
        versions = Counter(str(a.model_version) for a in online)
        return {"online": len(online), "versions": dict(versions), **totals}

    def run(self, report: float) -> None:
        threads = []
        for i, account in enumerate(self.accounts):
            if self.stop.is_set():
                break
            t = threading.Thread(target=self.run_account, args=(account, i), name=account, daemon=True)
            t.start()
            threads.append(t)
            self.stop.wait(self.stagger)
        while not self.stop.wait(report):
            print(json.dumps(self.status()), flush=True)
        with self.lock:
            actors = list(self.actors.values())
        for actor in actors:
            actor.stop()
        for t in threads:
            t.join(timeout=30)
        print(json.dumps(self.status()), flush=True)


def worker_main(args: argparse.Namespace, base: ActorConfig) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    root = actor_mod.DATAPACK
    npc_data.load_npc_names(str(root / "npcs"))  # once per process, before the threads
    item_data.load_item_data(str(root / "items"))
    skill_data.load_skill_data(str(root / "skills"))
    accounts = [f"{args.prefix}{i:04d}" for i in range(args.start, args.start + args.count)]
    worker = Worker(base, accounts, args.stagger, args.restart_delay, args.episodes_only)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stop.set())
    worker.run(args.report)
    return 0


# ----------------------------------------------------------------------
# Fleet: worker processes and the summary line
# ----------------------------------------------------------------------


def fleet_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_dir = Path(args.output_dir) / "fleet_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())

    procs: list[tuple[subprocess.Popen, int]] = []
    latest: dict[int, dict] = {}
    lock = threading.Lock()

    def read(proc: subprocess.Popen, start: int) -> None:
        for line in proc.stdout:
            try:
                status = json.loads(line)
            except json.JSONDecodeError:
                continue
            with lock:
                latest[start] = status

    def summary() -> str:
        with lock:
            rows = list(latest.values())
        total = Counter()
        versions = Counter()
        for row in rows:
            for key, value in row.items():
                if key == "versions":
                    versions.update(value)
                elif isinstance(value, int):
                    total[key] += value
        dead = sum(proc.poll() is not None for proc, _ in procs)
        return (f"online {total['online']}/{args.actors} | episodes {total['episodes']} K/D {total['kills']}/{total['deaths']} "
                f"| transitions {total['transitions']} segments {total['sent_segments']} | restarts {total['restarts']} "
                f"| model versions {dict(versions)}" + (f" | {dead} worker process(es) exited" if dead else ""))

    last_report = time.time()
    for start in range(args.first_index, args.first_index + args.actors, args.per_proc):
        if stopping.is_set():
            break
        count = min(args.per_proc, args.first_index + args.actors - start)
        cmd = [sys.executable, "-m", "l2_actor.fleet", "--worker", "--start", str(start), "--count", str(count),
               *[a for a in sys.argv[1:] if a != "--worker"]]
        err = open(log_dir / f"worker_{start:04d}.log", "w")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, cwd=Path(__file__).resolve().parents[1],
                                start_new_session=True)  # Ctrl+C reaches the fleet only; it stops the workers itself
        procs.append((proc, start))
        threading.Thread(target=read, args=(proc, start), daemon=True).start()
        stopping.wait(count * args.stagger)  # keep one login every --stagger seconds over the whole fleet
        if time.time() - last_report >= args.report:
            log.info(summary())
            last_report = time.time()
    log.info("Started %d worker process(es) for %d actor(s), %d per process; logs in %s",
             len(procs), args.actors, args.per_proc, log_dir)

    started = time.time()
    while not stopping.wait(args.report):
        log.info(summary())
        if args.max_minutes and time.time() - started >= args.max_minutes * 60:
            break
        if all(proc.poll() is not None for proc, _ in procs):
            log.error("Every worker process exited; see %s", log_dir)
            break

    log.info("Stopping %d worker process(es)...", len(procs))
    for proc, _ in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.time() + 60
    for proc, _ in procs:
        try:
            proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
    log.info("Final: %s", summary())
    return 0


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, ActorConfig]:
    d = ActorConfig()
    p = argparse.ArgumentParser(description="Run many L2 actors on one machine (threads in worker processes).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--actors", type=int, default=10, help="Number of actors")
    p.add_argument("--per-proc", type=int, default=10, help="Actors per worker process")
    p.add_argument("--prefix", default="rlfleet", help="Account name prefix (accounts <prefix>0000, <prefix>0001, ...)")
    p.add_argument("--first-index", type=int, default=0, help="Index of the first account (several fleets: disjoint ranges)")
    p.add_argument("--stagger", type=float, default=0.5, help="Seconds between two logins over the whole fleet")
    p.add_argument("--restart-delay", type=float, default=10.0, help="Seconds before a stopped actor is restarted")
    p.add_argument("--report", type=float, default=30.0, help="Seconds between two summary lines")
    p.add_argument("--max-minutes", type=float, default=0.0, help="Stop the fleet after N minutes (0 = no limit)")
    p.add_argument("--episodes-only", action="store_true",
                   help="Keep only the episode summaries (no transitions): evaluation runs with a fixed model")
    # Actor options (see python -m l2_actor --help).
    p.add_argument("--server-ip", default=d.server_ip)
    p.add_argument("--login-port", type=int, default=d.login_port)
    p.add_argument("--password", default=d.password)
    p.add_argument("--model", required=True, help="ONNX model, or 'redis' for the trainer's latest published policy")
    p.add_argument("--redis", default=None, help="Redis URL of the RL trainer")
    p.add_argument("--greedy", dest="sample", action="store_false", default=None, help="Argmax instead of sampling")
    p.add_argument("--sample", dest="sample", action="store_true", help="Sample actions even without --redis")
    p.add_argument("--server-speed", type=float, default=d.server_speed, help="Server TimeScale (General.ini)")
    p.add_argument("--decision-interval", type=float, default=d.decision_interval)
    p.add_argument("--segment-len", type=int, default=d.segment_len)
    p.add_argument("--max-queue", type=int, default=d.max_queue)
    p.add_argument("--model-poll", type=float, default=d.model_poll)
    p.add_argument("--model-wait", type=float, default=d.model_wait)
    p.add_argument("--output-dir", default=d.output_dir, help="Episode summaries, downloaded models, fleet logs")
    # Internal: a worker process of the fleet.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--count", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    base = ActorConfig(
        server_ip=args.server_ip, login_port=args.login_port, password=args.password, model=args.model,
        redis=args.redis, sample=args.sample, server_speed=args.server_speed, decision_interval=args.decision_interval,
        segment_len=args.segment_len, max_queue=args.max_queue, model_poll=args.model_poll, model_wait=args.model_wait,
        output_dir=args.output_dir, threads=1, store_next_state=args.redis is not None,
    )
    return args, base


def main(argv: list[str] | None = None) -> int:
    args, base = parse_args(argv)
    if args.worker:
        return worker_main(args, base)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return fleet_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
