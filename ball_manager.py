"""
ball_manager.py

Owns the (up to two) filter orbs and performs the cross-orb work a single
BallSystem can't:

- Spawning: a free fist claims the next idle orb. With two fists you get two
  separate orbs; each hand can hold at most one.
- Assignment: each frame, attached orbs are bound to *distinct* hands, so
  two orbs can't pile onto the same hand and MediaPipe hand-order flips
  can't drag one orb onto another.
- Merging: when two attached orbs' hands come within MERGE_DIST, they fuse
  into a single bigger "twin" orb (a fusion burst + rainbow ring).
- Absorbing: a flying orb that lands on a hand which already holds an orb is
  eaten by that orb instead of colliding.

Events (clap / wave) are routed here too so they can affect every orb.
"""

import random

import numpy as np

from gestures import hand_centroid, hand_scale, is_fist
from ball import (
    BallSystem,
    STATE_GONE,
    STATE_SPAWN,
    STATE_HELD,
    STATE_PINCHED,
    STATE_PUMP,
    STATE_TWIN,
    STATE_CATCH,
    STATE_STRETCH,
)

MERGE_DIST = 2.4


class BallManager:
    def __init__(self, ss=2):
        self.orbs = [BallSystem(ss=ss), BallSystem(ss=ss)]
        for orb in self.orbs:
            orb.manual_spawn = True
            orb._land_cb = self._on_land
        self._mesh = {}
        self._last_hands = []
        self._last_centroids = []
        self._last_scales = []

    # ------------------------------------------------------------- queries

    def any_active(self):
        return any(o.is_active() for o in self.orbs)

    def states(self):
        return [o.state for o in self.orbs]

    def theme_name(self):
        for o in self.orbs:
            if o.is_active():
                return o.theme_name
        return self.orbs[0].theme_name

    def cycle_theme(self):
        for o in self.orbs:
            if o.is_active():
                o.cycle_theme()
                return
        self.orbs[0].cycle_theme()

    def set_theme(self, idx):
        for o in self.orbs:
            o.set_theme(idx)

    def theme_idx(self):
        for o in self.orbs:
            if o.is_active():
                return o.theme_idx()
        return self.orbs[0].theme_idx()

    def summary(self):
        if not self.any_active():
            return "none"
        return "+".join(o.state for o in self.orbs if o.is_active())

    # -------------------------------------------------------------- events

    def on_clap(self):
        any_active = False
        for o in self.orbs:
            if o.is_active():
                any_active = True
                o._celebrate()
        if not any_active:
            col = self.orbs[0]._rainbow(0)
            for _ in range(5):
                px = random.uniform(60, 620)
                self.orbs[0]._burst(np.array([px, random.uniform(40, 180)], dtype=np.float32),
                                    26, col, 240)

    def on_wave(self, src):
        src = np.array(src, dtype=np.float32)
        targets = [o for o in self.orbs if o.is_active()]
        if not targets:
            dst = np.array([320.0, 240.0], dtype=np.float32)
            self.orbs[0]._wind(src, dst, self.orbs[0]._rainbow(0))
            return
        for o in targets:
            o._wind(src, o.pos)

    # ------------------------------------------------------------- ops

    def update(self, hands, dt):
        self._last_hands = hands
        self._last_centroids = [hand_centroid(h["points"]) for h in hands]
        self._last_scales = [hand_scale(h["points"]) for h in hands]
        centroids = self._last_centroids
        scales = self._last_scales
        self._mesh = {}

        # 1. attach attached orbs to distinct hands (greedy, nearest first)
        need = [
            (oi, orb) for oi, orb in enumerate(self.orbs)
            if orb.state in (STATE_SPAWN, STATE_HELD, STATE_PINCHED, STATE_PUMP,
                             STATE_TWIN, STATE_CATCH)
        ]
        occupied = set()
        for oi, orb in need:
            best, bd = -1, float("inf")
            for hi, scl in enumerate(scales):
                if hi in occupied:
                    continue
                center = orb.pos if orb.anchor is None else orb.anchor
                d = float(np.hypot(*(np.array(centroids[hi], dtype=np.float32) - center)))
                limit = max(2.5 * scl, orb.radius + 40.0)
                if d < limit and d < bd:
                    bd, best = d, hi
            if best >= 0:
                occupied.add(best)
                orb.forced_idx = best
                self._mesh[oi] = best
            else:
                orb.forced_idx = -1

        # 2. spawn: a free fist claims the next idle orb
        for hi, hand in enumerate(hands):
            if hi in occupied or not is_fist(hand["points"]):
                continue
            if self._is_stretch_partner(hi, centroids):
                continue
            if self._spawn_blocked_near_twin(hi, centroids, scales):
                continue
            for orb in self.orbs:
                if not orb.is_active():
                    orb.spawn(centroids[hi], scales[hi])
                    occupied.add(hi)
                    break

        # 3. merge: two attached orbs whose hands are close fuse into a twin
        merged = (STATE_HELD, STATE_PINCHED, STATE_PUMP, STATE_TWIN)
        for oi, oa in enumerate(self.orbs):
            if oa.state not in merged or oi not in self._mesh:
                continue
            hi = self._mesh[oi]
            for oj in range(oi + 1, len(self.orbs)):
                ob = self.orbs[oj]
                if ob.state not in merged or oj not in self._mesh:
                    continue
                hj = self._mesh[oj]
                if hj == hi:
                    continue
                gap = float(np.hypot(
                    *(np.array(centroids[hi], dtype=np.float32) - np.array(centroids[hj], dtype=np.float32))
                ))
                if gap < MERGE_DIST * min(scales[hi], scales[hj]):
                    self._merge(oa, ob, oi, oj)
                    break

        # 4. let each orb think
        for oi, orb in enumerate(self.orbs):
            orb.update(hands, dt, forced_idx=self._mesh.get(oi, -1))

    def render(self, frame, filtered):
        for orb in self.orbs:
            orb.render(frame, filtered)

    # ------------------------------------------------------------- internals

    def _is_stretch_partner(self, hi, centroids):
        for orb in self.orbs:
            if orb.state != STATE_STRETCH:
                continue
            for hp in (orb._hand_a, orb._hand_b):
                if hp is None:
                    continue
                if float(np.hypot(*(np.array(hp) - np.array(centroids[hi])))) < 60.0:
                    return True
        return False

    def _spawn_blocked_near_twin(self, hi, centroids, scales):
        twin = None
        for orb in self.orbs:
            if orb.state == STATE_TWIN:
                twin = orb
                break
        if twin is None:
            return False
        if twin.forced_idx == hi:
            return True
        gap = float(np.hypot(*(np.array(twin.pos) - np.array(centroids[hi]))))
        return gap < MERGE_DIST * min(scales[hi], self._last_scales[hi] if hi < len(self._last_scales) else scales[hi])

    def _merge(self, oa, ob, oi, oj):
        mid = (oa.pos + ob.pos) / 2.0
        oa._start_twin(mid)
        ob._start_absorb(mid)

    def _on_land(self, orb):
        """Called when a flying orb finishes its arc. Returns True if the
        orb was absorbed into a hand that already holds another orb."""
        target = orb._target_hand
        for other in self.orbs:
            if other is orb or not other.is_active():
                continue
            if other.forced_idx >= 0 and (target < 0 or other.forced_idx == target):
                if other.state in (STATE_HELD, STATE_PINCHED, STATE_PUMP, STATE_TWIN):
                    other._absorbed(orb.color, orb.pos)
                    orb._start_absorb(other.pos)
                    return True
        return False