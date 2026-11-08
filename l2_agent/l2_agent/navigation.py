"""
Walking to a point. The server finds the path around walls (``PathFinding = 2``) for each move; the agent sends
the target in legs of at most ``leg`` units (the server refuses far moves), checks that it gets closer and, when it
does not, tries a detour to the side before giving up.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Callable
from dataclasses import dataclass

from l2_client import packets

from .geomap import GeoMap
from .knowledge import Point
from .session import Session

log = logging.getLogger(__name__)

#: the trail keeps a spot every this many units, the last ``TRAIL_LENGTH`` of them
TRAIL_STEP = 250.0
TRAIL_LENGTH = 40
#: walks shorter than this go straight; longer ones follow a geodata route (a short one too: into a shop)
ROUTE_MIN = 150.0
#: a turn point of a geodata route counts as reached this close
ROUTE_ARRIVE = 60.0


@dataclass
class NavConfig:
    """Leg length and stuck detection of a walk."""

    leg: float = 2500.0
    #: a leg is re-sent when the character stands still this long short of its end
    stall_s: float = 2.5
    #: progress (units) that counts as moving
    min_progress: float = 40.0
    detours: int = 6
    detour_offset: float = 600.0


class Navigator:
    """Walks the character to points, with detours when it is stuck."""

    def __init__(self, session: Session, cfg: NavConfig | None = None, rng: random.Random | None = None,
                 geomap: GeoMap | None = None):
        self.session = session
        #: the agent's own path finding; without it the server's (which misses some ways) is all there is
        self.geomap = geomap
        self.cfg = cfg or NavConfig()
        self.rng = rng or random.Random()
        #: places where the agent got stuck (for the exploration report)
        self.stuck_spots: list[tuple[int, int, int]] = []
        #: spots the character walked through, newest last: known ways back out of a dead end
        self.trail: list[Point] = []
        #: asked every tick of a walk: deals with what stops a player on the way (a monster attacking, death) and
        #: returns ``True`` when it did something, then the walk plans again from where the character is
        self.guard: Callable[[], bool] | None = None
        self._teleports = session.world.teleports if session.world is not None else 0

    def move_to(self, target: Point, arrive: float = 120.0, timeout: float = 600.0,
                interrupt=None) -> bool:
        """
        Walk until within ``arrive`` of ``target``, along the geodata path when there is one. ``interrupt()`` is asked
        every tick; when it returns ``True`` the walk stops (e.g. a monster attacks) and ``False`` is returned, as on
        timeout or when stuck.
        """
        s = self.session
        deadline = time.time() + timeout
        for _ in range(3):  # a stall on the route replans from where the character is
            if s.distance(target.x, target.y) <= arrive:
                return True
            route = self._route(target) if s.distance(target.x, target.y) > ROUTE_MIN else None
            if not route:
                return self._walk(target, arrive, deadline - time.time(), interrupt, self.cfg.detours)
            ok = True
            for point in route[:-1]:
                if not self._walk(point, ROUTE_ARRIVE, min(deadline - time.time(), 60.0), interrupt, 2):
                    ok = False
                    break
            if ok:
                return self._walk(target, arrive, deadline - time.time(), interrupt, self.cfg.detours)
            if interrupt is not None and interrupt() or s.world.dead or time.time() >= deadline:
                return False
        if s.distance(target.x, target.y) > arrive and not s.world.dead and time.time() < deadline:
            route = self._route(target)  # the last try after a fight or a respawn
            if route and all(self._walk(p, ROUTE_ARRIVE, min(deadline - time.time(), 60.0), interrupt, 2)
                             for p in route[:-1]):
                return self._walk(target, arrive, deadline - time.time(), interrupt, self.cfg.detours)
        return False

    def _route(self, target: Point) -> list[Point] | None:
        if self.geomap is None:
            return None
        x, y, z = self.session.position
        started = time.time()
        route = self.geomap.find_path(Point(x, y, z), target)
        if route:
            log.info("Route to %s: %d turn(s), %.1fs", target, len(route), time.time() - started)
        return route

    def _walk(self, target: Point, arrive: float, timeout: float, interrupt, max_detours: int) -> bool:
        """A straight walk to ``target`` (the server may still bend it), with detours when it stalls."""
        s, cfg = self.session, self.cfg
        deadline = time.time() + max(timeout, 1.0)
        detours = 0
        best = s.distance(target.x, target.y)
        last_progress = time.time()
        waypoint: Point | None = None
        self._send_leg(target)
        while time.time() < deadline:
            s.pump(0.1)
            if interrupt is not None and interrupt():
                return False
            if self.guard is not None and self.guard():
                return False
            if s.world.dead:
                return False
            self._note_trail()
            dist = s.distance(target.x, target.y)
            if dist <= arrive:
                return True
            if waypoint is not None and s.distance(waypoint.x, waypoint.y) <= arrive:
                waypoint = None
                self._send_leg(target)
            if dist < best - cfg.min_progress:
                best, last_progress = dist, time.time()
            moving = s.motion is not None and not s.motion.arrived(time.time())
            if not moving and time.time() - last_progress > 0.6:
                self._send_leg(waypoint or target)
            if time.time() - last_progress > cfg.stall_s * (1 + detours):
                if detours >= max_detours:
                    self.stuck_spots.append(s.position)
                    log.warning("Stuck at %s on the way to %s (%.0f left)", s.position, target, dist)
                    return False
                detours += 1
                # Odd tries go back along the trail (the server finds no path out of some spots its geodata gets
                # wrong, but the way the character came in works), even ones to the side.
                back = self._trail_point(detours)
                waypoint = back if back is not None and detours % 2 else self._detour(target)
                log.info("No progress at %s: detour %d via %s", s.position, detours, waypoint)
                self._send_leg(waypoint)
                last_progress = time.time()
        log.warning("Timeout on the way to %s (%.0f left)", target, s.distance(target.x, target.y))
        return False

    def _note_trail(self) -> None:
        x, y, z = self.session.position
        if not self.trail or self.trail[-1].distance(x, y) >= TRAIL_STEP:
            self.trail.append(Point(x, y, z))
            del self.trail[:-TRAIL_LENGTH]

    def _trail_point(self, detours: int) -> Point | None:
        """A trail spot farther back with each try, at least ``TRAIL_STEP`` away from where the character is."""
        x, y, _ = self.session.position
        behind = [p for p in reversed(self.trail) if p.distance(x, y) >= TRAIL_STEP]
        return behind[min((detours - 1) // 2, len(behind) - 1)] if behind else None

    def _send_leg(self, target: Point) -> None:
        s = self.session
        if s.world.sitting:  # a sitting character does not walk: stand up first
            s.send(packets.build_request_action_use(0))
            s.wait_until(lambda: not s.world.sitting, 3.0)
        if s.world.teleports != self._teleports:  # respawned or teleported: the old trail leads nowhere from here
            self._teleports = s.world.teleports
            self.trail.clear()
        x, y, z = s.position
        dx, dy = target.x - x, target.y - y
        dist = math.hypot(dx, dy)
        if dist > self.cfg.leg:
            share = self.cfg.leg / dist
            tx, ty, tz = int(x + dx * share), int(y + dy * share), int(z + (target.z - z) * share)
        else:
            tx, ty, tz = target.x, target.y, target.z
        s.send(packets.build_move_to_location(tx, ty, tz, x, y, z))

    def _detour(self, target: Point) -> Point:
        """A point to the side of the straight way, alternating left and right."""
        x, y, z = self.session.position
        angle = math.atan2(target.y - y, target.x - x) + self.rng.choice((-1, 1)) * math.pi / 2
        offset = self.cfg.detour_offset * self.rng.uniform(0.6, 1.4)
        return Point(int(x + math.cos(angle) * offset), int(y + math.sin(angle) * offset), z)
