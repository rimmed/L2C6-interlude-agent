"""
Training dashboard: a local web page with the live state of an RL run.

Reads what the run already writes - nothing is sent to the trainer or the actors:

* ``<run>/metrics.jsonl`` - the trainer's metrics (``l2_NN/train_rl.py``);
* ``<run>/fleet*.log`` - the fleet summaries (agents online, kills / deaths, restarts);
* ``<run>/fleet*/episodes_*.jsonl`` - the fight summaries of the actors (K/D per class and level,
  actions and skills), read incrementally;
* Redis - the published model version and the segment queue;
* ``client_textures/icons/skill<id>.png`` - skill icons from the game client, when they were extracted
  (``l2_client/tools/extract_icons.py``).

A run is a directory with a ``metrics.jsonl`` under ``l2_agent/checkpoints`` (current runs) or under
``archive/`` in the project root (finished runs, listed as ``archive/<name>``).

Usage (from ``l2_agent/``)::

    uv run python -m l2_actor.dashboard                    # http://<this host>:8050
    uv run python -m l2_actor.dashboard --port 8050 --redis redis://localhost:6379/1 --run rl_bc_v4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import bc_page, buff_sets, skill_panels, training_panels
from .report import ACTION_COLUMNS, CLASS_LIST, DATAPACK_SKILLS, class_names

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
ARCHIVE = Path(__file__).resolve().parents[2] / "archive"
#: Skill icons extracted from the client's Icon.utx (``skill0003.png`` ...), served as ``/icons/<id>.png``.
ICONS = Path(__file__).resolve().parents[2] / "client_textures" / "icons"
FLEET_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*?online (?P<online>\d+)/(?P<total>\d+) \| episodes (?P<episodes>\d+) "
    r"K/D (?P<kills>\d+)/(?P<deaths>\d+) \| transitions (?P<steps>\d+) segments \d+ \| restarts (?P<restarts>\d+)")
#: How long after its last write a trainer / fleet still counts as running.
TRAINER_STALE_S = 180
#: A pause in the trainer's metrics longer than this starts a new training session (the elapsed-time tile).
SESSION_GAP_S = 15 * 60
#: A fleet run's intervals this long after all its actors came online still count as its warm-up (left out).
FLEET_WARMUP_S = 5 * 60
FLEET_STALE_S = 300
#: K/D of the recent fights: the last this many seconds of the fight records.
RECENT_S = 900
LEVEL_STEP = 10


def class_tiers(path: Path = CLASS_LIST) -> dict[int, int]:
    """Profession tier of every class: 0 for a starting class, 1-3 for the 1st-3rd profession."""
    if not path.exists():
        return {}
    import xml.etree.ElementTree as ET

    parent = {int(c.get("classId")): int(c.get("parentClassId")) if c.get("parentClassId") else None
              for c in ET.parse(path).getroot().iter("class")}
    # Interlude: ids 88+ are the 3rd professions. The parent chain gives the others, capped at 2 because
    # the datapack has a few wrong parents (e.g. Bladedancer under Shillien Knight).
    tiers = {}
    for class_id in parent:
        tier, current = 0, parent[class_id]
        while current is not None and tier < 3:
            tier, current = tier + 1, parent.get(current)
        tiers[class_id] = 3 if class_id >= 88 else min(tier, 2)
    return tiers


def level_range(level: int) -> str:
    low = (level // LEVEL_STEP) * LEVEL_STEP
    return f"{low:02d}-{low + LEVEL_STEP - 1:02d}"


class Episodes:
    """The fight summaries of one fleet directory, read incrementally (new lines only on every call)."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.offsets: dict[Path, int] = {}
        #: (t, class_id, level, kill) of every fight with a record time; untimed fights count in the totals only
        self.timed: list[tuple[float, int, int, bool]] = []
        self.by_class: dict[int, list] = defaultdict(lambda: [0, 0, set()])
        self.by_level: dict[str, list] = defaultdict(lambda: [0, 0, set()])
        self.actions: dict[int, list] = defaultdict(lambda: [0, defaultdict(int)])
        self.skills: dict[int, list] = defaultdict(lambda: [0, defaultdict(int)])
        #: character -> [kills, deaths, class id, level] (class and level of its latest fight)
        self.by_char: dict[str, list] = defaultdict(lambda: [0, 0, -1, 0])
        #: level range -> character -> [kills, deaths, class id, level] of its fights in that range
        self.by_level_char: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0, -1, 0]))

    def update(self) -> None:
        for file in sorted(self.directory.glob("episodes_*.jsonl")):
            offset = self.offsets.get(file, 0)
            if file.stat().st_size <= offset:
                continue
            with open(file, "rb") as fh:
                fh.seek(offset)
                data = fh.read()
            complete = data.rfind(b"\n") + 1  # a line still being written waits for the next call
            self.offsets[file] = offset + complete
            for line in data[:complete].splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("reason") in ("kill", "death") and not record.get("truncated"):
                    self.add(record)

    def add(self, record: dict) -> None:
        kill = record["reason"] == "kill"
        class_id, level = int(record.get("class_id", -1)), int(record.get("level", 0))
        character = record.get("character", "?")
        for table, key in ((self.by_class, class_id), (self.by_level, level_range(level))):
            row = table[key]
            row[0 if kill else 1] += 1
            row[2].add(character)
        for char in (self.by_char[character], self.by_level_char[level_range(level)][character]):
            char[0 if kill else 1] += 1
            char[2], char[3] = class_id, level
        if "t" in record:
            self.timed.append((float(record["t"]), class_id, level, kill))
        if record.get("actions") is not None:
            entry = self.actions[class_id]
            entry[0] += 1
            for action, n in record["actions"].items():
                entry[1][action] += n
            entry = self.skills[class_id]
            entry[0] += 1
            for skill, n in (record.get("skills") or {}).items():
                entry[1][skill] += n


class Dashboard:
    """Collects the state of a run for the web page (thread-safe, cached per fleet directory)."""

    def __init__(self, redis_url: str, default_run: str | None):
        self.redis_url = redis_url
        self.default_run = default_run
        self.names = class_names()
        self.tiers = class_tiers()
        self.episodes: dict[Path, Episodes] = {}
        self.lock = threading.Lock()
        self._skill_names: dict[int, str] | None = None
        self._panels: skill_panels.SkillPanels | None = None
        #: skill ids that have an icon
        self.icons = {int(p.stem[5:]) for p in ICONS.glob("skill*.png") if p.stem[5:].isdigit()} if ICONS.exists() else set()

    # -- sources ----------------------------------------------------------------------------

    @staticmethod
    def run_dirs() -> dict[str, Path]:
        """Run name -> directory, the most recently updated first (current runs before archived ones)."""
        found: list[tuple[str, Path]] = []
        for root, prefix in ((CHECKPOINTS, ""), (ARCHIVE, "archive/")):
            if root.exists():
                runs = [p for p in root.iterdir() if (p / "metrics.jsonl").exists()]
                runs.sort(key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)
                found += [(prefix + p.name, p) for p in runs]
        return dict(found)

    def runs(self) -> list[str]:
        return list(self.run_dirs())

    def redis_state(self) -> dict:
        try:
            import redis

            client = redis.Redis.from_url(self.redis_url, socket_timeout=2, socket_connect_timeout=2)
            version = client.get("l2:model:version")
            return {"ok": True, "model_version": int(version) if version else None, "queue": client.llen("l2:replay")}
        except Exception as exc:  # noqa: BLE001 - the page shows the error instead of failing
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def trainer(run: Path) -> dict:
        rows = []
        with open(run / "metrics.jsonl", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # NaN (an empty metric window, e.g. during a fleet warm-up) is no JSON for the browser
                rows.append({k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()})
        keys = ("update", "version", "reward_per_100_steps", "mean_return", "value_loss", "pg_loss", "bc_loss",
                "bc_weight", "entropy", "rho_far", "kl_approx", "updates_per_s", "replay_ratio", "lr")
        # A fleet's warm-up after a (re)start (``fleet_warmup``, see train_rl.py) is left out of the charts: its
        # return and K/D dip while the actors log in and the short lives end first, though the policy did not change.
        shown = [r for r in rows if not r.get("fleet_warmup")] or rows
        step = max(1, len(shown) // 400)  # at most ~400 points per chart
        series = [{"t": r["time"], **{k: r.get(k) for k in keys}} for r in shown[::step]]
        if shown and series[-1]["t"] != shown[-1]["time"]:
            series.append({"t": shown[-1]["time"], **{k: shown[-1].get(k) for k in keys}})
        latest = rows[-1] if rows else {}
        # the current training session: from the first metrics after the last pause (a run continued
        # later - an overnight stop, a resume - starts a new session; the hourly server restarts do not)
        session = rows[0]["time"] if rows else None
        for prev, row in zip(rows, rows[1:]):
            if row["time"] - prev["time"] > SESSION_GAP_S:
                session = row["time"]
        return {"series": series, "latest": latest,
                "running": bool(latest) and time.time() - latest.get("time", 0) < TRAINER_STALE_S,
                "started": rows[0]["time"] if rows else None, "session_started": session}

    @staticmethod
    def fleet(run: Path) -> dict:
        """Per-interval kills / deaths from the cumulative counters of every fleet log of the run."""
        points: dict[float, dict] = {}
        logs = sorted(run.glob("fleet*.log"), key=lambda p: p.stat().st_mtime)
        for log in logs:
            previous = None
            full_at = None  # when this fleet run first had all its actors online
            for line in open(log, encoding="utf-8", errors="replace"):
                m = FLEET_LINE.match(line)
                if not m:
                    continue
                t = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").timestamp()
                cur = {k: int(m.group(k)) for k in ("online", "total", "kills", "deaths", "steps", "restarts")}
                if full_at is None and cur["total"] and cur["online"] >= cur["total"]:
                    full_at = t
                # the ramp-up (logging in, spreading over the areas) and FLEET_WARMUP_S after it stay off the charts
                warm = full_at is None or t - full_at < FLEET_WARMUP_S
                if previous and t > previous[0] and not warm:
                    dt = t - previous[0]
                    kills, deaths = cur["kills"] - previous[1]["kills"], cur["deaths"] - previous[1]["deaths"]
                    if kills >= 0 and deaths >= 0:  # a new fleet run starts its counters at 0
                        points[t] = {"t": t, "online": cur["online"], "total": cur["total"], "kills": kills,
                                     "deaths": deaths, "kd": kills / deaths if deaths else None,
                                     "steps_per_s": (cur["steps"] - previous[1]["steps"]) / dt,
                                     "restarts": cur["restarts"]}
                previous = (t, cur)
        series = [points[t] for t in sorted(points)]
        last_write = max((p.stat().st_mtime for p in logs), default=0)
        return {"series": series, "running": time.time() - last_write < FLEET_STALE_S and bool(series)
                and series[-1]["online"] > 0, "logs": [p.name for p in logs]}

    def fights(self, run: Path, fleet_dir: str | None) -> dict:
        dirs = sorted((p for p in run.iterdir() if p.is_dir() and any(p.glob("episodes_*.jsonl"))),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            return {"dirs": [], "selected": None}
        chosen = next((d for d in dirs if d.name == fleet_dir), dirs[0])
        with self.lock:
            episodes = self.episodes.setdefault(chosen, Episodes(chosen))
            episodes.update()
            return {"dirs": [d.name for d in dirs], "selected": chosen.name, **self.tables(episodes)}

    def skill_name(self, key: str) -> str:
        if self._skill_names is None:
            from l2_client.tools.data import skill_data

            if DATAPACK_SKILLS.exists():
                skill_data.load_skill_data(str(DATAPACK_SKILLS))
            self._skill_names = {}
            self._lookup = skill_data.get_skill_name
        pet, _, sid = key.rpartition(":")
        name = self._skill_names.setdefault(int(sid), self._lookup(int(sid)) or sid)
        return f"{name} (pet)" if pet else name

    def skill_panels(self) -> "skill_panels.SkillPanels":
        """The class skill trees, loaded on the first request of the skill panel page."""
        with self.lock:
            if self._panels is None:
                self._panels = skill_panels.SkillPanels(self.icons)
            return self._panels

    def icon(self, key: str) -> int | None:
        """The skill id of ``key`` (``"1177"``, ``"pet:4137"``) when an icon of it was extracted."""
        sid = int(key.rpartition(":")[2])
        return sid if sid in self.icons else None

    def tables(self, e: Episodes) -> dict:
        now = max((t for t, *_ in e.timed), default=None)
        recent_class: dict[int, list] = defaultdict(lambda: [0, 0])
        recent_level: dict[str, list] = defaultdict(lambda: [0, 0])
        if now is not None:
            for t, class_id, level, kill in reversed(e.timed):
                if now - t > RECENT_S:
                    break
                recent_class[class_id][0 if kill else 1] += 1
                recent_level[level_range(level)][0 if kill else 1] += 1

        def kd(kills: int, deaths: int) -> float:
            return kills / max(deaths, 1)  # without a death: the kills

        def members(names: set, stats: dict[str, list] = e.by_char) -> list[dict]:
            out = []
            for name in names:
                k, d, c, lv = stats[name]
                out.append({"name": name, "class": self.names.get(c, f"class {c}"), "level": lv, "kills": k, "deaths": d,
                            "kd": kd(k, d)})
            return sorted(out, key=lambda m: -m["kd"])

        classes = [{"tier": self.tiers.get(c), "name": self.names.get(c, f"class {c}"), "chars": len(r[2]), "kills": r[0], "deaths": r[1],
                    "kd": kd(r[0], r[1]), "recent_kd": kd(*recent_class[c]) if c in recent_class else None,
                    "members": members(r[2])}
                   for c, r in e.by_class.items()]
        levels = [{"name": lv, "chars": len(r[2]), "kills": r[0], "deaths": r[1], "kd": kd(r[0], r[1]),
                   "recent_kd": kd(*recent_level[lv]) if lv in recent_level else None, "members": members(r[2], e.by_level_char[lv])}
                  for lv, r in sorted(e.by_level.items())]
        actions = []
        for c, (fights, counts) in e.actions.items():
            total = sum(counts.values()) or 1
            actions.append({"tier": self.tiers.get(c), "name": self.names.get(c, f"class {c}"), "fights": fights,
                            "per_fight": total / fights,
                            **{title: 100 * counts.get(a, 0) / total for a, title in ACTION_COLUMNS}})
        skills = []
        for c, (fights, counts) in e.skills.items():
            total = sum(counts.values())
            best = sorted(counts.items(), key=lambda item: -item[1])[:6]
            skills.append({"tier": self.tiers.get(c), "name": self.names.get(c, f"class {c}"), "fights": fights,
                           "per_fight": total / fights,
                           "top": [[self.skill_name(k), 100 * n / total, self.icon(k)] for k, n in best] if total else []})
        kills = sum(r[0] for r in e.by_class.values())
        deaths = sum(r[1] for r in e.by_class.values())
        return {"classes": classes, "levels": levels, "actions": actions, "skills": skills,
                "action_columns": [title for _, title in ACTION_COLUMNS], "recent_s": RECENT_S,
                "total": {"fights": kills + deaths, "kills": kills, "deaths": deaths, "kd": kd(kills, deaths)}}

    def status(self, run_name: str | None, fleet_dir: str | None) -> dict:
        dirs = self.run_dirs()
        runs = list(dirs)
        if not runs:
            return {"runs": [], "error": f"no run with a metrics.jsonl under {CHECKPOINTS} or {ARCHIVE}"}
        run_name = run_name if run_name in runs else (self.default_run if self.default_run in runs else runs[0])
        run = dirs[run_name]
        return {"runs": runs, "run": run_name, "now": time.time(), "redis": self.redis_state(),
                "trainer": self.trainer(run), "fleet": self.fleet(run), "fights": self.fights(run, fleet_dir)}


def handler_for(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            elif url.path == "/skills":
                self.reply(200, "text/html; charset=utf-8", skill_panels.PAGE.encode("utf-8"))
            elif url.path == "/api/skillpanels":
                query = parse_qs(url.query)
                try:
                    panels = dashboard.skill_panels()
                    classes = panels.classes()
                    wanted = query.get("class", [""])[0]
                    class_id = int(wanted) if wanted.isdigit() and int(wanted) in panels.trees else classes[0]["id"]
                    body = {"classes": classes, "panel": panels.panels(class_id)}
                    self.reply(200, "application/json", json.dumps(body).encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode("utf-8"))
            elif url.path == "/buffs":
                self.reply(200, "text/html; charset=utf-8", buff_sets.PAGE.encode("utf-8"))
            elif url.path == "/api/buffs":
                query = parse_qs(url.query)
                try:
                    body = buff_sets.payload(query.get("class", [None])[0], query.get("level", [None])[0], dashboard.icons)
                    self.reply(200, "application/json", json.dumps(body).encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode("utf-8"))
            elif url.path == "/bc":
                self.reply(200, "text/html; charset=utf-8", bc_page.PAGE.encode("utf-8"))
            elif url.path == "/api/bc":
                try:
                    body = bc_page.payload(parse_qs(url.query).get("run", [None])[0])
                    self.reply(200, "application/json", json.dumps(body).encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode("utf-8"))
            elif url.path == "/panels":
                self.reply(200, "text/html; charset=utf-8", training_panels.PAGE.encode("utf-8"))
            elif url.path == "/api/panels":
                query = parse_qs(url.query)
                try:
                    body = training_panels.payload(query.get("class", [None])[0], dashboard.icons)
                    self.reply(200, "application/json", json.dumps(body).encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode("utf-8"))
            elif url.path.startswith("/icons/") and url.path[7:-4].isdigit() and url.path.endswith(".png"):
                icon = ICONS / f"skill{int(url.path[7:-4]):04d}.png"
                if icon.exists():
                    self.reply(200, "image/png", icon.read_bytes(), cache=True)
                else:
                    self.reply(404, "text/plain", b"no icon")
            elif url.path == "/api/status":
                query = parse_qs(url.query)
                try:
                    body = dashboard.status(query.get("run", [None])[0], query.get("fleet", [None])[0])
                    self.reply(200, "application/json", json.dumps(body, default=str).encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode("utf-8"))
            else:
                self.reply(404, "text/plain", b"not found")

        def reply(self, code: int, content_type: str, body: bytes, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # quiet: one request every few seconds
            pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L2 RL training</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#6b7385; --line:#e3e6ec; --good:#1a7f4b; --bad:#b42318;
          --c1:#2f6fdb; --c2:#d9534f; --c3:#1a7f4b; --c4:#9a6dd7; --c5:#e08a1e; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029; --ink:#e7eaf0; --muted:#98a0b3; --line:#2b3240;
          --good:#3fbf7f; --bad:#ff6b5e; --c1:#6ea2ff; --c2:#ff7b72; --c3:#3fbf7f; --c4:#b892ff; --c5:#ffb44d; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:14px 20px; border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:1; }
  header h1 { font-size:17px; margin:0 12px 0 0; }
  select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  .badge { padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid var(--line); }
  .on { color:var(--good); border-color:var(--good); } .off { color:var(--muted); }
  main { padding:16px 20px 40px; max-width:1500px; margin:0 auto; }
  .tiles { display:flex; gap:8px; margin-bottom:16px; }
  .tile { flex:1 1 0; min-width:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 10px; }
  .tile .v { font-size:clamp(15px, 1.5vw, 22px); font-weight:650; font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tile .k { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  @media (max-width:900px) { .tiles { flex-wrap:wrap; } .tile { flex:1 1 130px; } }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(440px, 1fr)); gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; min-width:0; }
  .card h2 { font-size:14px; margin:0 0 6px; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted); margin-bottom:4px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:10px; height:3px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  svg { width:100%; height:190px; display:block; }
  svg text { fill:var(--muted); font-size:10px; }
  .wide { grid-column:1 / -1; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { padding:4px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
  th { color:var(--muted); font-weight:600; cursor:pointer; user-select:none; position:sticky; top:0; background:var(--card); }
  th:first-child, td:first-child { text-align:left; }
  #t-skills td:nth-child(2), #t-skills th:nth-child(2), #t-actions td:nth-child(2), #t-actions th:nth-child(2) { text-align:left; }
  .scroll { max-height:520px; overflow:auto; }
  .good { color:var(--good); } .bad { color:var(--bad); }
  .muted { color:var(--muted); }
  .skill { display:inline-flex; align-items:center; gap:5px; margin-right:12px; }
  .ico { width:22px; height:22px; border-radius:4px; flex:none; }
  .noico { display:inline-block; width:22px; height:22px; border-radius:4px; background:var(--line); flex:none; }
  td.skills { white-space:normal; text-align:left; line-height:2; }
  .err { color:var(--bad); }
  .expand { cursor:pointer; white-space:nowrap; } .expand:hover { color:var(--c1); }
  tr.detail > td { background:var(--bg); padding:6px 10px 10px 28px; }
  table.members { width:auto; min-width:60%; font-size:12px; }
  table.members th, table.members td { padding:2px 10px; }
  @media (max-width:520px) { .grid { grid-template-columns:1fr; } header, main { padding-left:16px; padding-right:16px; } }
</style>
</head>
<body>
<header>
  <h1>L2 RL training</h1>
  <label>run <select id="run"></select></label>
  <label>fights <select id="fleetdir"></select></label>
  <span id="trainerBadge" class="badge off">trainer</span>
  <span id="fleetBadge" class="badge off">fleet</span>
  <span id="updated" class="muted"></span>
  <a href="/bc" class="muted">BC training →</a>
  <a href="/buffs" class="muted">agent buffs →</a>
  <a href="/panels" class="muted">training panels →</a>
  <a href="/skills" class="muted">class skill trees →</a>
</header>
<main>
  <div class="tiles" id="tiles"></div>
  <div class="grid">
    <div class="card"><h2>K/D per fleet interval</h2><div class="legend" id="lg-kd"></div><svg id="ch-kd"></svg></div>
    <div class="card"><h2>Kills and deaths per interval</h2><div class="legend" id="lg-kills"></div><svg id="ch-kills"></svg></div>
    <div class="card"><h2>Reward per 100 steps (trainer)</h2><div class="legend" id="lg-reward"></div><svg id="ch-reward"></svg></div>
    <div class="card"><h2>Losses</h2><div class="legend" id="lg-loss"></div><svg id="ch-loss"></svg></div>
    <div class="card"><h2>Policy lag and entropy</h2><div class="legend" id="lg-rho"></div><svg id="ch-rho"></svg></div>
    <div class="card"><h2>Throughput</h2><div class="legend" id="lg-speed"></div><svg id="ch-speed"></svg></div>
    <div class="card wide"><h2>K/D by class <span class="muted" id="recentNote"></span></h2><div class="scroll"><table id="t-class"></table></div></div>
    <div class="card"><h2>K/D by level</h2><div class="scroll"><table id="t-level"></table></div></div>
    <div class="card"><h2>Fights in this fleet directory</h2><div id="totals"></div></div>
    <div class="card wide"><h2>Actions per class (share of decisions, %)</h2><div class="scroll"><table id="t-actions"></table></div></div>
    <div class="card wide"><h2>Skills per class (share of casts)</h2><div class="scroll"><table id="t-skills"></table></div></div>
  </div>
  <p id="error" class="err"></p>
</main>
<script>
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 2) => v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d);
const int = v => v === null || v === undefined ? "–" : Math.round(v).toLocaleString("uk-UA");
const hhmm = t => new Date(t * 1000).toLocaleTimeString("uk-UA", {hour: "2-digit", minute: "2-digit"});
let state = {run: null, fleet: null, sort: {}, open: {}};

function chart(id, legendId, rows, lines) {
  const svg = $(id), W = svg.clientWidth || 440, H = 190, L = 44, R = 10, T = 8, B = 22;
  const pts = rows.filter(r => r.t).sort((a, b) => a.t - b.t);
  $(legendId).innerHTML = lines.map(l => `<span><i style="background:${css(l.color)}"></i>${l.label}</span>`).join("");
  if (pts.length < 2) { svg.innerHTML = `<text x="${L}" y="${H / 2}">no data yet</text>`; return; }
  // Pauses (no training for a while) are cut out of the time axis: each one takes a narrow band,
  // so the runs with data keep their width instead of being squeezed beside a long flat line.
  const dts = pts.slice(1).map((r, i) => r.t - pts[i].t).sort((a, b) => a - b);
  const gapAfter = Math.max(300, 5 * dts[Math.floor(dts.length / 2)]);
  const active = pts.slice(1).reduce((sum, r, i) => sum + (r.t - pts[i].t > gapAfter ? 0 : r.t - pts[i].t), 0) || 1;
  const nGaps = pts.slice(1).filter((r, i) => r.t - pts[i].t > gapAfter).length;
  const gapU = nGaps ? active * 0.03 : 0;
  const u = [0], gaps = [];
  for (let i = 1; i < pts.length; i++) {
    const dt = pts[i].t - pts[i - 1].t, isGap = dt > gapAfter;
    u.push(u[i - 1] + (isGap ? gapU : dt));
    if (isGap) gaps.push({u0: u[i - 1], u1: u[i], dt, i});
  }
  const uMax = u[u.length - 1] || 1;
  const vals = lines.flatMap(l => pts.map(r => r[l.key])).filter(v => v !== null && v !== undefined && isFinite(v));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lines.every(l => l.zero)) lo = Math.min(0, lo);
  if (hi === lo) { hi += 1; lo -= 1; }
  const x = uu => L + (W - L - R) * uu / uMax, y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, yy = y(v);
    g += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="${css('--line')}"/>`;
    g += `<text x="${L - 6}" y="${yy + 3}" text-anchor="end">${Math.abs(hi - lo) < 5 ? v.toFixed(2) : Math.round(v)}</text>`;
  }
  const dur = s => s >= 3600 ? `${Math.floor(s / 3600)} h ${String(Math.round(s % 3600 / 60)).padStart(2, "0")} min` : `${Math.round(s / 60)} min`;
  for (const gp of gaps) {
    const x0 = x(gp.u0), x1 = x(gp.u1);
    g += `<rect x="${x0}" y="${T}" width="${Math.max(x1 - x0, 2)}" height="${H - T - B}" fill="${css('--line')}" opacity="0.6"><title>no data for ${dur(gp.dt)}</title></rect>`;
    g += `<text x="${(x0 + x1) / 2}" y="${T + 10}" text-anchor="middle">${dur(gp.dt)}</text>`;
  }
  // time labels only on stretches with data: the real time at evenly spaced positions of the axis
  const timeAt = uu => {
    for (let i = 1; i < pts.length; i++) {
      if (uu <= u[i]) {
        if (pts[i].t - pts[i - 1].t > gapAfter) return null;
        return pts[i - 1].t + (pts[i].t - pts[i - 1].t) * (uu - u[i - 1]) / Math.max(u[i] - u[i - 1], 1e-9);
      }
    }
    return pts[pts.length - 1].t;
  };
  for (let i = 0; i <= 4; i++) { const uu = uMax * i / 4, t = timeAt(uu); if (t !== null) g += `<text x="${x(uu)}" y="${H - 6}" text-anchor="middle">${hhmm(t)}</text>`; }
  for (const l of lines) {
    let d = "", pen = false;
    pts.forEach((r, i) => {
      const v = r[l.key];
      if (i > 0 && r.t - pts[i - 1].t > gapAfter) pen = false;
      if (v === null || v === undefined || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(u[i]).toFixed(1)},${y(v).toFixed(1)}`; pen = true;
    });
    g += `<path d="${d}" fill="none" stroke="${css(l.color)}" stroke-width="1.8"/>`;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = g;
}

function table(id, cols, rows, defaultSort, detail) {
  const s = state.sort[id] || defaultSort;
  const open = state.open[id] = state.open[id] || new Set();
  const sorted = [...rows].sort((a, b) => {
    const va = a[s.key], vb = b[s.key];
    if (typeof va === "string") return s.dir * va.localeCompare(vb);
    return s.dir * ((va ?? -Infinity) - (vb ?? -Infinity));
  });
  const head = "<tr>" + cols.map(c => `<th data-k="${c.key}">${c.label}${s.key === c.key ? (s.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr>";
  const body = sorted.map(r => {
    const row = "<tr>" + cols.map(c => {
      let v = c.f ? c.f(r[c.key], r) : r[c.key];
      if (detail && c.key === "name") v = `<span class="expand" data-name="${String(r.name).replace(/"/g, "&quot;")}">${open.has(r.name) ? "▾" : "▸"} ${v}</span>`;
      return `<td class="${c.cls ? c.cls(r) : ""}">${v}</td>`;
    }).join("") + "</tr>";
    return detail && open.has(r.name) ? row + `<tr class="detail"><td colspan="${cols.length}">${detail(r)}</td></tr>` : row;
  }).join("");
  const el = $(id);
  el.innerHTML = head + body;
  el.querySelectorAll(".expand").forEach(sp => sp.onclick = () => {
    const n = sp.dataset.name;
    open.has(n) ? open.delete(n) : open.add(n);
    table(id, cols, rows, defaultSort, detail);
  });
  el.querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k, cur = state.sort[id] || defaultSort;
    state.sort[id] = {key: k, dir: cur.key === k ? -cur.dir : (k === "name" ? 1 : -1)};
    table(id, cols, rows, defaultSort, detail);
  });
}

const kdCls = r => r.recent_kd === null || r.kd === null ? "" : r.recent_kd > r.kd * 1.1 ? "good" : r.recent_kd < r.kd * 0.9 ? "bad" : "";

function tile(k, v) { return `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`; }

function render(s) {
  $("error").textContent = s.error || "";
  const runSel = $("run");
  runSel.innerHTML = (s.runs || []).map(r => `<option ${r === s.run ? "selected" : ""}>${r}</option>`).join("");
  if (!s.run) return;
  state.run = s.run;
  const tr = s.trainer, fl = s.fleet, fi = s.fights, lat = tr.latest || {}, fs = fl.series, last = fs[fs.length - 1] || {};
  $("trainerBadge").className = "badge " + (tr.running ? "on" : "off");
  $("trainerBadge").textContent = tr.running ? "trainer running" : "trainer stopped";
  $("fleetBadge").className = "badge " + (fl.running ? "on" : "off");
  $("fleetBadge").textContent = fl.running ? "fleet running" : "fleet stopped";
  $("updated").textContent = "updated " + new Date().toLocaleTimeString("uk-UA");
  const fleetSel = $("fleetdir");
  fleetSel.innerHTML = (fi.dirs || []).map(d => `<option ${d === fi.selected ? "selected" : ""}>${d}</option>`).join("");
  state.fleet = fi.selected;

  // whole minutes of the current training session (to the last metrics when the trainer has stopped)
  const sessionEnd = tr.running ? s.now : (tr.latest.time || s.now);
  const elapsed = tr.session_started ? Math.max(0, Math.floor((sessionEnd - tr.session_started) / 60)) : null;
  $("tiles").innerHTML = [
    tile("agents online", last.online !== undefined ? int(last.online) : "–"),
    tile("K/D last interval", fmt(last.kd)),
    tile("trainer update", int(lat.update)),
    tile("model version (Redis)", s.redis.ok ? int(s.redis.model_version) : `<span class="err">Redis?</span>`),
    tile("reward / 100 steps", fmt(lat.reward_per_100_steps)),
    tile("updates / s", fmt(lat.updates_per_s, 1)),
    tile("steps / s (fleet)", int(last.steps_per_s)),
    tile("agent restarts", int(last.restarts)),
    tile("trained steps", int(lat.trained_steps)),
    tile(tr.running ? "current training" : "last training (stopped)", elapsed === null ? "–" : `${Math.floor(elapsed / 60)} h ${String(elapsed % 60).padStart(2, "0")} min`),
  ].join("");

  chart("ch-kd", "lg-kd", fs, [{key: "kd", label: "K/D", color: "--c1", zero: true}]);
  chart("ch-kills", "lg-kills", fs, [{key: "kills", label: "kills", color: "--c3", zero: true}, {key: "deaths", label: "deaths", color: "--c2", zero: true}]);
  chart("ch-reward", "lg-reward", tr.series, [{key: "reward_per_100_steps", label: "reward / 100 steps", color: "--c1"}]);
  chart("ch-loss", "lg-loss", tr.series, [{key: "value_loss", label: "value", color: "--c1"}, {key: "bc_loss", label: "BC", color: "--c4"}, {key: "pg_loss", label: "policy", color: "--c5"}]);
  chart("ch-rho", "lg-rho", tr.series, [{key: "rho_far", label: "rho far (share)", color: "--c2", zero: true}, {key: "entropy", label: "entropy", color: "--c3", zero: true}]);
  chart("ch-speed", "lg-speed", tr.series, [{key: "updates_per_s", label: "updates / s", color: "--c1", zero: true}, {key: "replay_ratio", label: "replay ratio", color: "--c5", zero: true}]);

  if (!fi.selected) { ["t-class", "t-level", "t-actions", "t-skills"].forEach(id => $(id).innerHTML = ""); $("totals").textContent = "no fight records in this run"; return; }
  $("recentNote").textContent = `· “recent” = last ${Math.round(fi.recent_s / 60)} min of fights; green / red = better / worse than the whole run`;
  const kdCols = [
    {key: "name", label: "class"}, {key: "chars", label: "chars", f: int}, {key: "kills", label: "kills", f: int},
    {key: "deaths", label: "deaths", f: int}, {key: "kd", label: "K/D", f: v => fmt(v)},
    {key: "recent_kd", label: "K/D recent", f: v => fmt(v), cls: kdCls},
  ];
  table("t-class", [{key: "tier", label: "prof", f: v => v ?? "–"}, ...kdCols], fi.classes, {key: "name", dir: 1}, r =>
    `<table class="members"><tr><th>character</th><th>level</th><th>kills</th><th>deaths</th><th>K/D</th></tr>`
    + r.members.map(m => `<tr><td>${m.name}</td><td>${m.level}</td><td>${int(m.kills)}</td><td>${int(m.deaths)}</td><td>${fmt(m.kd)}</td></tr>`).join("") + "</table>");
  table("t-level", [{...kdCols[0], label: "levels"}, ...kdCols.slice(1)], fi.levels, {key: "name", dir: 1}, r =>
    `<table class="members"><tr><th>character</th><th>class</th><th>level</th><th>kills</th><th>deaths</th><th>K/D</th></tr>`
    + r.members.map(m => `<tr><td>${m.name}</td><td>${m.class}</td><td>${m.level}</td><td>${int(m.kills)}</td><td>${int(m.deaths)}</td><td>${fmt(m.kd)}</td></tr>`).join("") + "</table>");
  $("totals").innerHTML = `<table><tr><td>fights</td><td>${int(fi.total.fights)}</td></tr><tr><td>kills</td><td>${int(fi.total.kills)}</td></tr>`
    + `<tr><td>deaths</td><td>${int(fi.total.deaths)}</td></tr><tr><td>K/D</td><td>${fmt(fi.total.kd)}</td></tr></table>`;
  if (!fi.actions.length) {
    $("t-actions").innerHTML = `<tr><td class="muted">no action records here (actors before the action counters)</td></tr>`;
    $("t-skills").innerHTML = "";
  } else {
    table("t-actions", [{key: "tier", label: "prof", f: v => v ?? "–"}, {key: "name", label: "class"}, {key: "fights", label: "fights", f: int}, {key: "per_fight", label: "decisions / fight", f: v => fmt(v, 1)},
      ...fi.action_columns.map(c => ({key: c, label: c, f: v => fmt(v, 1)}))],
      [...fi.actions].sort((a, b) => a.name.localeCompare(b.name)), {key: "tier", dir: 1});
    table("t-skills", [{key: "tier", label: "prof", f: v => v ?? "–"}, {key: "name", label: "class"}, {key: "fights", label: "fights", f: int}, {key: "per_fight", label: "casts / fight", f: v => fmt(v)},
      {key: "top", label: "most used skills", cls: () => "skills", f: v => v.map(([n, p, id]) =>
        `<span class="skill">${id !== null ? `<img class="ico" src="/icons/${id}.png" alt="">` : `<span class="noico"></span>`}${n} <span class="muted">${p.toFixed(0)}%</span></span>`).join("") || "–"}],
      [...fi.skills].sort((a, b) => a.name.localeCompare(b.name)), {key: "tier", dir: 1});
  }
}

async function refresh() {
  try {
    const q = new URLSearchParams();
    if (state.run) q.set("run", state.run);
    if (state.fleet) q.set("fleet", state.fleet);
    const r = await fetch("/api/status?" + q);
    render(await r.json());
  } catch (e) { $("error").textContent = "cannot reach the dashboard server: " + e; }
}
$("run").onchange = e => { state.run = e.target.value; state.fleet = null; refresh(); };
$("fleetdir").onchange = e => { state.fleet = e.target.value; refresh(); };
window.addEventListener("resize", () => refresh());
refresh();
setInterval(refresh, 20000);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local web page with the state of the RL training.")
    p.add_argument("--host", default="0.0.0.0", help="Interface to listen on (default: every interface)")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--redis", default="redis://localhost:6379/1", help="Redis of the run (model version, queue)")
    p.add_argument("--run", default=None, help="Run shown first (default: the most recently updated)")
    args = p.parse_args(argv)

    dashboard = Dashboard(args.redis, args.run)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(dashboard))
    print(f"Dashboard on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
