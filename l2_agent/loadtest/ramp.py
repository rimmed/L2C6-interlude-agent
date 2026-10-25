"""
Load-test controller (NOT for commit): adds a worker of ``--step`` actors every ``--step-seconds``
and reports, per step, the server's answer latency, the server's CPU / memory and the client's load.

Stops when the p95 SkillCoolTime latency of a step exceeds ``max(--lag-ms, --lag-factor x baseline)``,
when actors fail to connect, when the client machine saturates or at ``--max-actors``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SSH = os.environ.get("RSSH", "")

SERVER_CMD = (
    # fed to "bash -s" on stdin: no shell quoting around it, and no command line that contains the pattern
    "P=$(pgrep -f '^java .*GameServer.jar' | head -1); "
    "read c1 < <(awk '{print $14+$15}' /proc/$P/stat); t1=$(awk '/^cpu /{s=0;for(i=2;i<=NF;i++)s+=$i;print s}' /proc/stat); "
    "sleep 2; "
    "read c2 < <(awk '{print $14+$15}' /proc/$P/stat); t2=$(awk '/^cpu /{s=0;for(i=2;i<=NF;i++)s+=$i;print s}' /proc/stat); "
    "rss=$(awk '/VmRSS/{print $2}' /proc/$P/status); thr=$(awk '/Threads/{print $2}' /proc/$P/status); "
    "echo $(( (c2-c1) )) $(( (t2-t1) )) $rss $thr $(nproc) $(cut -d' ' -f1 /proc/loadavg) "
    "$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
)


def server_stats() -> dict | None:
    try:
        out = subprocess.run([SSH, "ssh", "l2server@192.168.1.200", "bash -s"], input=SERVER_CMD, capture_output=True,
                             text=True, timeout=20).stdout.split()
        proc_ticks, total_ticks, rss, threads, ncpu, load, avail = (float(x) for x in out[-7:])
        # total_ticks is summed over all CPUs: x ncpu gives % of one core (top's convention)
        return {"java_cpu_pct": round(100 * proc_ticks / max(total_ticks, 1) * ncpu, 1), "java_rss_mb": round(rss / 1024),
                "java_threads": int(threads), "load1": load, "mem_avail_mb": round(avail / 1024)}
    except Exception as exc:  # noqa: BLE001
        print(f"server stats failed: {exc}", flush=True)
        return None


_last_cpu: list[int] = []


def client_stats() -> dict:
    """Load average, real CPU busy share since the previous call (/proc/stat) and free memory."""
    with open("/proc/meminfo") as fh:
        mem = {line.split(":")[0]: int(line.split()[1]) for line in fh}
    with open("/proc/stat") as fh:
        ticks = [int(x) for x in fh.readline().split()[1:]]
    busy = None
    if _last_cpu:
        delta = [a - b for a, b in zip(ticks, _last_cpu)]
        idle = delta[3] + delta[4]
        busy = round(100 * (1 - idle / max(sum(delta), 1)), 1)
    _last_cpu[:] = ticks
    return {"client_load1": os.getloadavg()[0], "client_cpu_busy": busy, "client_mem_avail_mb": mem["MemAvailable"] // 1024}


def workers_rss_mb(workers: list) -> int:
    total = 0
    for proc, _ in workers:
        try:
            with open(f"/proc/{proc.pid}/status") as fh:
                total += next(int(line.split()[1]) for line in fh if line.startswith("VmRSS"))
        except (OSError, StopIteration):
            pass
    return total // 1024


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="loadt")
    p.add_argument("--first", type=int, default=10, help="Actors of the baseline step")
    p.add_argument("--step", type=int, default=25)
    p.add_argument("--step-seconds", type=float, default=120)
    p.add_argument("--measure-seconds", type=float, default=60, help="Tail of a step used for its statistics")
    p.add_argument("--max-actors", type=int, default=400)
    p.add_argument("--lag-ms", type=float, default=250)
    p.add_argument("--lag-factor", type=float, default=3.0)
    p.add_argument("--warm", type=int, default=0, help="Start this many actors first (in --step workers) without measuring")
    p.add_argument("--warm-seconds", type=float, default=20, help="Pause between two warm-up workers")
    p.add_argument("--per-proc", type=int, default=0, help="Actors per worker process (0 = one process per step)")
    p.add_argument("--stagger", type=float, default=0.5, help="Seconds between two logins")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workers: list[tuple[subprocess.Popen, Path]] = []
    offsets: dict[Path, int] = {}
    results, baseline, total = [], None, 0
    warm = [args.step] * (args.warm // args.step)
    first = [] if warm else [args.first]
    sizes = warm + first + [args.step] * ((args.max_actors - args.warm - sum(first)) // args.step)

    def read_new() -> list[dict]:
        rows = []
        for _, path in workers:
            if not path.exists():
                continue
            with path.open() as fh:
                fh.seek(offsets.get(path, 0))
                rows += [json.loads(line) for line in fh if line.strip()]
                offsets[path] = fh.tell()
        return rows

    try:
        for chunk, size in enumerate(sizes):
            per_proc = args.per_proc or size
            for start in range(total, total + size, per_proc):
                count = min(per_proc, total + size - start)
                path = out / f"worker_{start:04d}.jsonl"
                cmd = [sys.executable, str(HERE / "worker.py"), "--start", str(start), "--count", str(count), "--prefix",
                       args.prefix, "--model", args.model, "--server-speed", str(args.speed), "--out", str(path),
                       "--stagger", str(args.stagger)]
                workers.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                 stderr=open(out / f"worker_{start:04d}.err", "w")), path))
                time.sleep(count * args.stagger)  # one login every --stagger seconds over all workers
            total += size
            if chunk <= len(warm) - 1:
                print(f"[{time.strftime('%H:%M:%S')}] warm-up: {total} actors", flush=True)
                time.sleep(args.warm_seconds)
                continue
            step_start = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] step: {total} actors", flush=True)
            read_new()  # discard the previous step's tail
            window: list[dict] = []
            srv, cli = [], []
            while time.time() - step_start < args.step_seconds:
                time.sleep(8)
                rows = read_new()
                if time.time() - step_start >= args.step_seconds - args.measure_seconds:
                    window += rows
                    if (s := server_stats()) is not None:
                        srv.append(s)
                    cli.append(client_stats())
            cool = [x for r in window for x in r["samples"]["cool"]]
            move = [x for r in window for x in r["samples"]["move"]]
            loop = [x for r in window for x in r["samples"]["loop"]]
            last, first = {}, {}
            for r in window:
                last[r["start"]] = r  # newest line of every worker
                first.setdefault(r["start"], r)
            span = max((r["t"] for r in window), default=0) - min((r["t"] for r in window), default=0)

            def rate(key: str) -> float | None:
                delta = sum(last[k]["events"][key] - first[k]["events"][key] for k in last)
                return round(60 * delta / span / max(total, 1), 2) if span > 0 else None
            connected = sum(r["events"]["connected"] for r in last.values())
            alive = sum(r["alive"] for r in last.values())
            errors = sum(r["events"]["errors"] + r["events"]["disconnects"] for r in last.values())
            decisions = sum(r["events"]["decisions"] for r in last.values())
            kills = sum(r["events"]["kills"] for r in last.values())
            texts = [e for r in window for e in r["errors"]]
            row = {
                "speed": args.speed, "actors": total, "connected": connected, "alive": alive, "errors": errors,
                "cool_p50": pct(cool, 50), "cool_p95": pct(cool, 95), "cool_p99": pct(cool, 99), "cool_n": len(cool),
                "move_p50": pct(move, 50), "move_p95": pct(move, 95), "move_n": len(move),
                "client_loop_p95": pct(loop, 95),
                "java_cpu_pct": round(float(np.mean([s["java_cpu_pct"] for s in srv])), 1) if srv else None,
                "java_rss_mb": srv[-1]["java_rss_mb"] if srv else None,
                "server_load1": srv[-1]["load1"] if srv else None,
                "client_load1": round(float(np.mean([c["client_load1"] for c in cli])), 1) if cli else None,
                "client_cpu_busy": round(float(np.mean([c["client_cpu_busy"] for c in cli if c["client_cpu_busy"] is not None])), 1)
                if any(c["client_cpu_busy"] is not None for c in cli) else None,
                "client_mem_avail_mb": cli[-1]["client_mem_avail_mb"] if cli else None,
                "decisions_total": decisions, "kills_total": kills,
                "processes": len(workers), "workers_rss_mb": workers_rss_mb(workers),
                "decisions_per_actor_min": rate("decisions"), "kills_per_actor_min": rate("kills"),
            }
            results.append(row)
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
            for t in texts[:5]:
                print("   error:", t, flush=True)

            if baseline is None and row["cool_p95"] is not None:
                baseline = row["cool_p95"]
            limit = max(args.lag_ms, args.lag_factor * (baseline or 0))
            if row["cool_p95"] is not None and row["cool_p95"] > limit:
                print(f"STOP: p95 {row['cool_p95']} ms > {limit:.0f} ms", flush=True)
                break
            if row["errors"] > 0.02 * total:
                print(f"STOP: {row['errors']} disconnects/errors > 2% of {total}", flush=True)
                break
            if connected < 0.9 * total:
                print(f"STOP: only {connected}/{total} actors connected", flush=True)
                break
            if (row["client_cpu_busy"] or 0) > 85 or (row["client_loop_p95"] or 0) > 50:
                print("STOP: client machine saturated", flush=True)
                break
    finally:
        for proc, _ in workers:
            proc.send_signal(2)
        for proc, _ in workers:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
