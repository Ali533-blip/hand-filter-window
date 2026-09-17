"""
ball.py

The interactive "filter orb": an energy ball made out of the live camera
filter that forms in your fist and can be carried, pinched, thrown, caught,
stretched like taffy between two hands, merged with a second orb (manager),
and recoloured through 5 themes.

State machine
-------------
    gone -> spawn (fist) -> held (rests in fist OR open palm)
    held  <-> pinched (thumb + index)
    held / pinched -> flight (flick or bolt) -> catch -> held
    held / pinched -> stretch (a second hand comes in) -> held
    held -> hop  (point: orb jumps to the holder's fingertip)
    held -> pump (thumbs-up: orb swells with a rainbow band)
    twin (two orbs fused by ball_manager) behaves like a bigger held orb
    absorb (a flying orb eaten by a hand that already holds one)

Visual polish
-------------
- Supersampled rendering (self.ss, default 2): the orb patch is drawn at 2x
  the resolution it will be displayed at, then downscaled - crisp, aliased
  edges even on small orbs.
- Theme colours lerp smoothly to the new hue instead of snapping.
- A critically-damped spring follows the hand, the orb bobs while idle,
  squashes on catch and stretches along its velocity in flight.
- Rings / shockwaves, rainbow bands (twin / pump) and "wind" particles
  respond to events: spawn, merge, pump, victory, clap, wave.
"""

import math
import random

import cv2
import numpy as np

from gestures import (
    WRIST,
    INDEX_TIP,
    hand_centroid,
    hand_scale,
    is_open_hand,
    is_pinch,
    is_middle_thumb_touch,
    is_pointing,
    is_gun,
    is_thumbs_up,
    is_victory,
)

STATE_GONE = "gone"
STATE_SPAWN = "spawn"
STATE_HELD = "held"
STATE_PINCHED = "pinched"
STATE_DESTROY = "destroy"
STATE_FLIGHT = "flight"
STATE_CATCH = "catch"
STATE_STRETCH = "stretch"
STATE_TWIN = "twin"
STATE_HOP = "hop"
STATE_PUMP = "pump"
STATE_ABSORB = "absorb"

THEMES = [
    ("amber", (0, 170, 255)),
    ("cyan", (255, 200, 0)),
    ("rose", (235, 80, 250)),
    ("emerald", (85, 235, 130)),
    ("indigo", (255, 110, 140)),
]

THROW_SPEED_FACTOR = 5.0
STRETCH_ENGAGE_DIST = 1.9
PUMP_TIME = 4.0


def _soften(img, sigma):
    """Blur an image cheaply by blurring at a fraction of the resolution and
    upscaling - the landing zone for the halo and the orb-mask feather.
    Visually identical to a big full-res Gaussian but several times faster
    (final smoothness comes from the upscale, not the pixel budget)."""
    h, w = img.shape[:2]
    div = 4 if w >= 32 else (2 if w >= 8 else 1)
    if div == 1:
        return cv2.GaussianBlur(img, (0, 0), max(1.0, sigma))
    small = cv2.resize(img, (max(1, w // div), max(1, h // div)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma * (1.0 / div)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color


class BallSystem:
    def __init__(self, ss=2):
        self.ss = max(int(ss), 1)
        self._theme_idx = 0
        self.state = STATE_GONE
        self.state_t = 0.0
        self.manual_spawn = False
        self.forced_idx = -1
        self._land_cb = None

        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.anchor = None
        self.radius = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.angle = 0.0
        self.pulse = 0.0

        self._spring_v = np.zeros(2, dtype=np.float32)
        self._color_cur = np.array(THEMES[0][1], dtype=np.float32)

        self.particles = []

        self._anchor_prev = None
        self._anchor_vel = np.zeros(2, dtype=np.float32)
        self._speed_samples = []

        self._tap_cd = 0.0
        self._hop_cd = 0.0
        self._gun_cd = 0.0
        self._pump_cd = 0.0
        self._cel_cd = 0.0

        self._flight_t = 0.0
        self._flight_dur = 1.0
        self._flight_start = np.zeros(2, dtype=np.float32)
        self._flight_end = np.zeros(2, dtype=np.float32)
        self._flight_arc = 0.0
        self._flight_r = 0.0
        self._target_hand = -1
        self._trail_acc = 0.0
        self._bolt = False
        self._hop_t = 0.0

        self._destroy_r0 = 1.0
        self._hand_a = None
        self._hand_b = None

        self._pump_t = 0.0
        self._radius_mult = 1.0
        self._prev_mult = 1.0
        self._absorb_gain = 0.0

        self._absorb_t = 0.0
        self._absorb_target = None

        self._squash_x = 1.0
        self._squash_y = 1.0

        self._ring = None

    # ------------------------------------------------------------ properties

    @property
    def color(self):
        return (int(self._color_cur[0]), int(self._color_cur[1]), int(self._color_cur[2]))

    @property
    def theme_name(self):
        return THEMES[self._theme_idx][0]

    @property
    def target_color(self):
        return THEMES[self._theme_idx][1]

    def is_active(self):
        return self.state != STATE_GONE

    @staticmethod
    def _dim(c, amt=90):
        return tuple(int(max(0, v - amt)) for v in c)

    @staticmethod
    def _bright(c, amt=100):
        return tuple(int(min(255, v + amt)) for v in c)

    def _base_radius(self, scl):
        return float(np.clip(scl * 0.72, 16.0, 100.0) * self._radius_mult)

    def _rainbow(self, offset=0):
        hue = ((self.state_t * 90.0) + offset * 40.0) % 180.0
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def _reset_scale(self):
        self._radius_mult = 1.0
        self._prev_mult = 1.0
        self._absorb_gain = 0.0
        self._squash_x = 1.0
        self._squash_y = 1.0
        self._ring = None

    def _set_state(self, state):
        self.state = state
        self.state_t = 0.0

    def cycle_theme(self, direction=1):
        self._theme_idx = (self._theme_idx + direction) % len(THEMES)
        self._burst(self.pos, 26, self.target_color, 220)
        self._start_ring(self.pos, self.target_color)

    def set_theme(self, idx):
        self._theme_idx = idx % len(THEMES)
        self._color_cur = np.array(THEMES[self._theme_idx][1], dtype=np.float32)

    def theme_idx(self):
        return self._theme_idx

    # ---------------------------------------------------------------- update

    def update(self, hands, dt, forced_idx=-1):
        dt = min(max(dt, 1e-4), 0.1)
        self.forced_idx = forced_idx
        self.state_t += dt
        self.pulse = 0.5 + 0.5 * math.sin(self.state_t * 4.0)

        self._tap_cd = max(0.0, self._tap_cd - dt)
        self._hop_cd = max(0.0, self._hop_cd - dt)
        self._gun_cd = max(0.0, self._gun_cd - dt)
        self._pump_cd = max(0.0, self._pump_cd - dt)
        self._cel_cd = max(0.0, self._cel_cd - dt)

        self._squash_x += (1.0 - self._squash_x) * min(1.0, dt * 7.0)
        self._squash_y += (1.0 - self._squash_y) * min(1.0, dt * 7.0)
        self._absorb_gain = max(0.0, self._absorb_gain - dt * 28.0)
        if self._ring is not None:
            self._ring["t"] += dt
            if self._ring["t"] >= self._ring["dur"]:
                self._ring = None

        target = np.array(THEMES[self._theme_idx][1], dtype=np.float32)
        self._color_cur += (target - self._color_cur) * min(1.0, dt * 6.5)

        centroids = [hand_centroid(h["points"]) for h in hands]
        scales = [hand_scale(h["points"]) for h in hands]

        if self.state == STATE_GONE:
            self._update_gone()
        elif self.state == STATE_SPAWN:
            self._update_spawn(hands, centroids, scales, dt)
        elif self.state == STATE_HELD:
            self._update_held(hands, centroids, scales, dt)
        elif self.state == STATE_PINCHED:
            self._update_pinched(hands, centroids, scales, dt)
        elif self.state == STATE_DESTROY:
            self._update_destroy(dt)
        elif self.state == STATE_FLIGHT:
            self._update_flight(hands, centroids, scales, dt)
        elif self.state == STATE_CATCH:
            self._update_catch(hands, centroids, scales, dt)
        elif self.state == STATE_STRETCH:
            self._update_stretch(hands, centroids, scales, dt)
        elif self.state == STATE_TWIN:
            self._update_pump(hands, centroids, scales, dt, pump_time=None)
        elif self.state == STATE_HOP:
            self._update_hop(dt)
        elif self.state == STATE_PUMP:
            self._update_pump(hands, centroids, scales, dt)
        elif self.state == STATE_ABSORB:
            self._update_absorb(dt)

        self._advance_particles(dt)

    def spawn(self, c, scl):
        if self.state != STATE_GONE:
            return
        self._reset_scale()
        self._start_spawn(c, scl)

    def _update_gone(self):
        # spawning is driven by ball_manager.spawn(); without a manager the
        # orb just waits here (kept for direct-library users/tests)
        return

    # ------------------------------------------------------------- state logic

    def _start_spawn(self, c, scl):
        self._reset_scale()
        self._set_state(STATE_SPAWN)
        c = np.array(c, dtype=np.float32)
        self.pos = c.copy()
        self.anchor = c.copy()
        self.radius = 4.0
        self.rx = self.radius
        self.ry = self.radius
        self.angle = 0.0
        self._spring_v = np.zeros(2, dtype=np.float32)
        self._target_radius = self._base_radius(scl)
        self._burst(self.pos, 18, self.color, 160)
        self._start_ring(self.pos, self.color)

    def _update_spawn(self, hands, centroids, scales, dt):
        idx = self._nearest_hand(centroids, scales)
        if idx < 0:
            self._start_destroy()
            return
        self.anchor = np.array(centroids[idx], dtype=np.float32)
        self.pos = self._follow(self.pos, self.anchor, 0.045, dt)
        target = self._target_radius
        self.radius = min(self.radius + (target - self.radius) * min(1.0, dt * 14), target)
        self.rx = self.ry = self.radius
        if self.radius >= target * 0.96:
            self._set_state(STATE_HELD)
            self.anchor = np.array(centroids[idx], dtype=np.float32)

    def _update_held(self, hands, centroids, scales, dt):
        idx = self._nearest_hand(centroids, scales)
        if idx < 0:
            self._start_destroy()
            return
        points = hands[idx]["points"]
        scl = scales[idx]
        self._track_anchor_velocity(points[WRIST], scl, dt)

        if self._tap_cd <= 0 and is_middle_thumb_touch(points):
            self.cycle_theme()
            self._tap_cd = 0.35
        if is_pinch(points):
            self._set_state(STATE_PINCHED)
            return
        if self._hop_cd <= 0 and is_pointing(points):
            self._start_hop(np.array(points[INDEX_TIP], dtype=np.float32))
            return
        if self._gun_cd <= 0 and is_gun(points):
            self._fire_bolt(hands, centroids, scales)
            return
        if self._pump_cd <= 0 and is_thumbs_up(points):
            self._start_pump()
            return
        if self._cel_cd <= 0 and is_victory(points):
            self._celebrate()
            self._cel_cd = 0.9
        if self._flick_ready(scl):
            self._launch(hands, centroids, scales)
            return
        if self._other_near(idx, centroids, scl, scales):
            self._set_state(STATE_STRETCH)
            return

        self.anchor = np.array(centroids[idx], dtype=np.float32)
        self._spring_track(self.anchor, dt, 30.0)
        target = self._base_radius(scl) + self._absorb_gain
        self.radius += (target - self.radius) * min(1.0, dt * 9.0)
        self.rx = self.ry = self.radius

    def _update_pinched(self, hands, centroids, scales, dt):
        idx = self._nearest_hand(centroids, scales)
        if idx < 0:
            self._start_destroy()
            return
        points = hands[idx]["points"]
        scl = scales[idx]
        self._track_anchor_velocity(points[WRIST], scl, dt)

        if self._tap_cd <= 0 and is_middle_thumb_touch(points):
            self.cycle_theme()
            self._tap_cd = 0.35
        if is_open_hand(points) and not is_pinch(points):
            self._start_destroy()
            return
        if not is_pinch(points):
            self._set_state(STATE_HELD)
            return
        if self._flick_ready(scl):
            self._launch(hands, centroids, scales)
            return
        if self._other_near(idx, centroids, scl, scales):
            self._set_state(STATE_STRETCH)
            return

        self.anchor = np.array(centroids[idx], dtype=np.float32)
        self.pos = self._follow(self.pos, self.anchor, 0.05, dt)
        target = self._base_radius(scl) * 0.55
        self.radius += (target - self.radius) * min(1.0, dt * 12.0)
        self.rx = self.ry = self.radius

    def _update_pump(self, hands, centroids, scales, dt, pump_time=PUMP_TIME):
        idx = self._nearest_hand(centroids, scales)
        if idx < 0:
            self._start_destroy()
            return
        points = hands[idx]["points"]
        scl = scales[idx]
        self._track_anchor_velocity(points[WRIST], scl, dt)

        if self._tap_cd <= 0 and is_middle_thumb_touch(points):
            self.cycle_theme()
            self._tap_cd = 0.35
        if is_pinch(points):
            self._set_state(STATE_PINCHED)
            return
        if self._hop_cd <= 0 and is_pointing(points):
            self._start_hop(np.array(points[INDEX_TIP], dtype=np.float32))
            return
        if self._flick_ready(scl):
            self._launch(hands, centroids, scales)
            return
        # NOTE: no `other_near -> stretch` transition here. That belongs to
        # the plain HELD/PINCHED orbs; a PUMP or TWIN orb is a committed
        # single object and a docked second hand must not tear it open.

        self.anchor = np.array(centroids[idx], dtype=np.float32)
        self._spring_track(self.anchor, dt, 26.0)
        target = self._base_radius(scl) + self._absorb_gain
        self.radius += (target - self.radius) * min(1.0, dt * 8.0)
        self.rx = self.ry = self.radius

        if pump_time is not None:
            self._pump_t += dt
            if self._pump_t > pump_time:
                self._pump_t = 0.0
                self._radius_mult = self._prev_mult
                self._set_state(STATE_HELD)

    def _start_pump(self):
        self._prev_mult = self._radius_mult
        self._radius_mult = 1.5
        self._pump_t = 0.0
        self._set_state(STATE_PUMP)
        self._spring_v = np.zeros(2, dtype=np.float32)
        self._burst(self.pos, 42, self._rainbow(0), 320)
        self._start_ring(self.pos, self._rainbow(1))
        self._pump_cd = 1.2

    def _start_twin(self, mid):
        self._prev_mult = self._radius_mult
        self._radius_mult = 1.45
        self._pump_t = 0.0
        self._set_state(STATE_TWIN)
        self.pos = np.array(mid, dtype=np.float32).copy()
        self.anchor = self.pos.copy()
        self._spring_v = np.zeros(2, dtype=np.float32)
        self._speed_samples.clear()
        self.radius = max(self.radius * 1.2, 18.0)
        self.rx = self.ry = self.radius
        self._burst(self.pos, 55, self._rainbow(0), 300)
        self._start_ring(self.pos, self._rainbow(1))

    def _start_hop(self, target):
        self._set_state(STATE_HOP)
        self._hop_t = 0.0
        start = np.array(self.pos, dtype=np.float32)
        end = np.array(target, dtype=np.float32)
        d = float(np.hypot(*(end - start)))
        self._flight_start = start.copy()
        self._flight_end = end.copy()
        self._flight_arc = max(16.0, d * 0.18)
        self._flight_dur = max(0.2, min(0.45, d / 900.0))
        self._flight_r = max(self.radius * 0.8, 6.0)
        self._speed_samples.clear()
        self._burst(start, 16, self.color, 190)
        self._hop_cd = 0.8

    def _update_hop(self, dt):
        self._hop_t += dt
        t = min(self._hop_t / max(self._flight_dur, 1e-6), 1.0)
        p0, p1 = self._flight_start, self._flight_end
        x = p0 + (p1 - p0) * t
        arc = self._flight_arc * math.sin(math.pi * t)
        newpos = np.array([x[0], x[1] - arc], dtype=np.float32)
        self.vel = (newpos - self.pos) / dt
        self.pos = newpos
        self.radius = max(self._flight_r, self.radius * (1.0 - 0.35 * t))
        self.rx = self.ry = self.radius
        if random.random() < 0.5:
            jx = self.pos[0] + random.uniform(-self.radius, self.radius)
            self._burst(np.array([jx, self.pos[1]], dtype=np.float32), 1, self.color, 60)
        if t >= 1.0:
            self.pos = np.array(self._flight_end, dtype=np.float32)
            self._set_state(STATE_HELD)

    def _start_destroy(self):
        if self.state in (STATE_GONE, STATE_DESTROY, STATE_ABSORB):
            return
        self._set_state(STATE_DESTROY)
        self._destroy_r0 = max(self.radius, 1.0)
        self._burst(self.pos, 42, self.color, 260)
        self._start_ring(self.pos, self.color)

    def _update_destroy(self, dt):
        self.radius = max(self._destroy_r0 * (1.0 - min(self.state_t / 0.35, 1.0)), 0.0)
        self.rx = self.ry = self.radius
        if random.random() < 0.6:
            jx = self.pos[0] + random.uniform(-self.radius, self.radius)
            jy = self.pos[1] + random.uniform(-self.radius, self.radius)
            self._burst(np.array([jx, jy], dtype=np.float32), 3, self.color, 130)
        if self.state_t >= 0.35 or self.radius <= 0.5:
            self._reset_scale()
            self._set_state(STATE_GONE)
            self.anchor = None
            self._anchor_prev = None

    def _celebrate(self):
        self._burst(self.pos, 42, self._rainbow(0), 300)
        self._burst(self.pos, 18, self.color, 120)
        self._start_ring(self.pos, self._rainbow(1))

    def _launch(self, hands, centroids, scales, bolt=False, dirv=None):
        self._set_state(STATE_FLIGHT)
        self._flight_t = 0.0
        self._trail_acc = 0.0
        self._bolt = bolt
        start = np.array(self.pos, dtype=np.float32)
        if dirv is None:
            speed = float(np.hypot(*self._anchor_vel))
            dirv = self._anchor_vel / (speed + 1e-6)
        dirv = np.array(dirv, dtype=np.float32) / (float(np.hypot(*dirv)) + 1e-6)

        if bolt:
            target_i, end = self._pick_bolt_target(centroids, scales, dirv)
        else:
            target_i, end = self._pick_target(centroids, scales, dirv)
        if end is None:
            end = start + dirv * 550.0
        self._flight_start = start.copy()
        self._flight_end = np.array(end, dtype=np.float32)
        d = float(np.hypot(*(self._flight_end - start)))
        self._flight_arc = max(16.0, d * 0.2) * (0.25 if bolt else 1.0)
        self._flight_dur = max(0.26, min(0.8, d / 1500.0)) * (0.6 if bolt else 1.0)
        self._target_hand = -1 if target_i is None else target_i
        self._flight_r = max(self.radius * 0.85, 8.0)
        self._speed_samples.clear()
        self._burst(start, 30 if bolt else 24, self.color, 240)

    def _fire_bolt(self, hands, centroids, scales):
        idx = self.forced_idx
        if idx < 0 or idx >= len(centroids):
            return
        other = self._other_hand(idx, centroids)
        if other is None:
            return
        v = np.array(centroids[other], dtype=np.float32) - self.pos
        n = float(np.hypot(*v)) + 1e-6
        self._launch(hands, centroids, scales, bolt=True, dirv=v / n)
        self._gun_cd = 0.5

    def _pick_bolt_target(self, centroids, scales, dirv):
        start, best, best_score = self.pos, -1, 0.0
        for i, c in enumerate(centroids):
            vec = np.array(c, dtype=np.float32) - start
            d = float(np.hypot(*vec))
            if d < 40.0:
                continue
            align = float(np.dot(vec, dirv)) / (d + 1e-6)
            if align < 0.35:
                continue
            if d > best_score:
                best_score, best = d, i
        if best < 0:
            return -1, None
        c = np.array(centroids[best], dtype=np.float32)
        return best, c.copy()

    def _pick_target(self, centroids, scales, dirv):
        if not centroids:
            return -1, None
        start = self.pos
        holder = min(
            range(len(centroids)),
            key=lambda i: float(np.hypot(*(np.array(centroids[i], dtype=np.float32) - start))),
        )
        best, best_score = -1, 0.0
        for i, c in enumerate(centroids):
            if i == holder:
                continue
            vec = np.array(c, dtype=np.float32) - start
            d = float(np.hypot(*vec))
            if d < 40.0:
                continue
            align = float(np.dot(vec, dirv)) / (d + 1e-6)
            if align < 0.5:
                continue
            score = align * d
            if score > best_score:
                best_score, best = score, i
        if best < 0:
            return -1, None
        c = np.array(centroids[best], dtype=np.float32)
        end = c - dirv * (self.radius * 0.7 + 12.0)
        return best, end

    def _update_flight(self, hands, centroids, scales, dt):
        self._flight_t += dt
        t = min(self._flight_t / max(self._flight_dur, 1e-6), 1.0)
        p0, p1 = self._flight_start, self._flight_end
        x = p0 + (p1 - p0) * t
        arc = self._flight_arc * math.sin(math.pi * t)
        newpos = np.array([x[0], x[1] - arc], dtype=np.float32)
        self.vel = (newpos - self.pos) / dt
        self.pos = newpos
        self.radius += (self._flight_r - self.radius) * min(1.0, dt * 10.0)
        self.rx = self.ry = self.radius

        self._trail_acc += dt
        if self._trail_acc > 0.018:
            self._trail_acc = 0.0
            self._burst(self.pos, 2, self.color, 70)

        if t >= 1.0:
            if self._land_cb is not None and self._land_cb(self):
                return
            end = np.array(self._flight_end, dtype=np.float32)
            tgt = self._target_hand
            if tgt >= 0 and tgt < len(centroids):
                self._start_catch(
                    np.array(centroids[tgt], dtype=np.float32),
                    hands[tgt]["points"][WRIST],
                )
            else:
                best, bd = -1, float("inf")
                for i, c in enumerate(centroids):
                    deg = float(np.hypot(*(np.array(c, dtype=np.float32) - end)))
                    if deg < bd:
                        bd, best = deg, i
                if best >= 0 and bd < 260.0:
                    self._start_catch(
                        np.array(centroids[best], dtype=np.float32),
                        hands[best]["points"][WRIST],
                    )
                else:
                    self._burst(self.pos, 26, self.color, 260)
                    self._reset_scale()
                    self._set_state(STATE_GONE)
                    self.anchor = None
                    self._anchor_prev = None

    def _start_catch(self, c, wrist):
        self._set_state(STATE_CATCH)
        c = np.array(c, dtype=np.float32)
        self.pos = c.copy()
        self.anchor = c.copy()
        self._anchor_prev = np.array(wrist, dtype=np.float32)
        self._anchor_vel = np.zeros(2, dtype=np.float32)
        self._speed_samples.clear()
        self._spring_v = np.zeros(2, dtype=np.float32)
        self.radius = max(self.radius * 0.6, 6.0)
        self.rx = self.ry = self.radius
        self._squash_x = 1.5
        self._squash_y = 0.6
        n = 40 if self._bolt else 30
        self._burst(c, n, self.color, 240)

    def _update_catch(self, hands, centroids, scales, dt):
        idx = self._nearest_hand(centroids, scales)
        if idx < 0:
            self._start_destroy()
            return
        self.anchor = np.array(centroids[idx], dtype=np.float32)
        self._spring_track(self.anchor, dt, 18.0)
        target = self._base_radius(scales[idx]) + self._absorb_gain
        self.radius += (target - self.radius) * min(1.0, dt * 8.0)
        self.rx = self.ry = self.radius
        if self.state_t > 0.25:
            self._set_state(STATE_HELD)

    def _update_stretch(self, hands, centroids, scales, dt):
        if len(centroids) < 2:
            if len(centroids) == 1:
                self.anchor = np.array(centroids[0], dtype=np.float32)
                self._set_state(STATE_HELD)
                self.pos = self.anchor.copy()
                self.radius = self._base_radius(scales[0])
                self.rx = self.ry = self.radius
            else:
                self._start_destroy()
            return

        a_i, b_i = 0, 1
        sep_best = -1.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                d = float(np.hypot(*(np.array(centroids[i], dtype=np.float32) - np.array(centroids[j], dtype=np.float32))))
                if d > sep_best:
                    sep_best, a_i, b_i = d, i, j

        a = np.array(centroids[a_i], dtype=np.float32)
        b = np.array(centroids[b_i], dtype=np.float32)
        sep = float(np.hypot(*(b - a)))
        avg_scale = (scales[a_i] + scales[b_i]) / 2.0
        base = self._base_radius(avg_scale)

        mid = (a + b) / 2.0
        g_a = float(np.hypot(*(a - mid)))
        g_b = float(np.hypot(*(b - mid)))
        lo, hi = min(g_a, g_b), max(g_a, g_b)
        if lo < 1.2 * base and hi > max(lo * 4.0, 3.0 * base):
            close = a if g_a < g_b else b
            self.anchor = close.copy()
            self._set_state(STATE_HELD)
            self.pos = close.copy()
            self.radius = base
            self.rx = self.ry = base
            self.angle = 0.0
            return

        self._hand_a = a.copy()
        self._hand_b = b.copy()
        self.anchor = mid
        self.pos = self.anchor.copy()
        self.angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        self.rx = min(sep / 2.0, 380.0)
        self.ry = max(base * (1.0 - 0.5 * (sep / (base * 4.0))), base * 0.35)
        self.ry = max(self.ry, 12.0)
        self.radius = self.ry

        for i in (a_i, b_i):
            if self._tap_cd <= 0 and is_middle_thumb_touch(hands[i]["points"]):
                self.cycle_theme()
                self._tap_cd = 0.35

        if random.random() < 0.3:
            t = random.random()
            jx = a[0] + (b[0] - a[0]) * t
            jy = a[1] + (b[1] - a[1]) * t + random.uniform(-self.ry, self.ry)
            self._burst(np.array([jx, jy], dtype=np.float32), 1, self.color, 30)

    # ---------------------------------------------------------------- absorb

    def _start_absorb(self, target):
        self._set_state(STATE_ABSORB)
        self._absorb_t = 0.0
        self._absorb_target = np.array(target, dtype=np.float32)
        self._burst(self.pos, 18, self.color, 220)

    def _absorbed(self, incoming_color, incoming_pos):
        self._absorb_gain = 12.0
        self._radius_mult = min(1.6, self._radius_mult + 0.07)
        self._burst(np.array(incoming_pos, dtype=np.float32), 36, incoming_color, 280)
        self._start_ring(np.array(incoming_pos, dtype=np.float32), incoming_color)
        self._spring_v = np.zeros(2, dtype=np.float32)

    def _update_absorb(self, dt):
        self._absorb_t += dt
        t = min(1.0, dt * 16.0)
        self.pos += (self._absorb_target - self.pos) * t
        self.radius = max(self.radius * (1.0 - min(1.0, dt * 8.0)), 2.0)
        self.rx = self.ry = self.radius
        if random.random() < 0.4:
            jx = self.pos[0] + random.uniform(-8, 8)
            jy = self.pos[1] + random.uniform(-8, 8)
            self._burst(np.array([jx, jy], dtype=np.float32), 1, self.color, 80)
        if self.radius <= 3.0 or self._absorb_t > 0.5:
            self._reset_scale()
            self._set_state(STATE_GONE)
            self.anchor = None
            self._anchor_prev = None

    # ---------------------------------------------------------------- helpers

    def _nearest_hand(self, centroids, scales):
        if not centroids:
            return -1
        if 0 <= self.forced_idx < len(centroids):
            return self.forced_idx
        if self.anchor is None:
            return -1
        best, bd = -1, float("inf")
        for i, c in enumerate(centroids):
            d = float(np.hypot(*(np.array(c, dtype=np.float32) - self.anchor)))
            limit = max(2.5 * scales[i], self.radius + 40.0)
            if d < limit and d < bd:
                bd, best = d, i
        return best

    def _other_near(self, idx, centroids, scl, scales):
        other = self._other_hand(idx, centroids)
        if other is None:
            return False
        gap = float(np.hypot(
            *(np.array(centroids[other], dtype=np.float32) - self.anchor)
        ))
        return gap < STRETCH_ENGAGE_DIST * min(scl, scales[other])

    def _other_hand(self, idx, centroids):
        if len(centroids) < 2:
            return None
        a = np.array(centroids[idx], dtype=np.float32)
        best, bd = -1, float("inf")
        for i, c in enumerate(centroids):
            if i == idx:
                continue
            d = float(np.hypot(*(np.array(c, dtype=np.float32) - a)))
            if d < bd:
                bd, best = d, i
        return best

    def _track_anchor_velocity(self, wrist, scl, dt):
        """Tracks the holder hand's *wrist* position, not its centroid: a
        pinch/tap/etc. changes the centroid from a pose shift, but watching
        the wrist means only real flicking motion can register as a throw.
        If the wrist jumps more than ~1.25 hand-sizes in a single frame the
        ball just re-anchored to a different hand (or MediaPipe flipped the
        hand order) - not a flick, so the tracker resets silently."""
        wrist = np.array(wrist, dtype=np.float32)
        if self._anchor_prev is not None:
            disp = float(np.hypot(*(wrist - self._anchor_prev)))
            if disp > max(1.25 * scl, 120.0):
                self._anchor_vel = np.zeros(2, dtype=np.float32)
                self._speed_samples.clear()
                self._anchor_prev = wrist
                return
            inst = (wrist - self._anchor_prev) / dt
            self._anchor_vel = 0.55 * self._anchor_vel + 0.45 * inst
            self._speed_samples.append(float(np.hypot(*inst)))
            if len(self._speed_samples) > 5:
                del self._speed_samples[0]
        else:
            self._anchor_vel = np.zeros(2, dtype=np.float32)
            self._speed_samples.clear()
        self._anchor_prev = wrist

    def _flick_ready(self, scl):
        if not self._speed_samples:
            return False
        fast = sum(1 for s in self._speed_samples if s > THROW_SPEED_FACTOR * scl)
        return fast >= 2

    @staticmethod
    def _follow(cur, target, tau, dt):
        a = 1.0 - math.exp(-dt / tau)
        return cur + (target - cur) * a

    def _spring_track(self, target, dt, stiff):
        damp = 2.0 * math.sqrt(stiff)
        self.pos += self._spring_v * dt
        self._spring_v += (target - self.pos) * stiff * dt - self._spring_v * damp * dt

    # ---------------------------------------------------------------- particles

    def _burst(self, pos, n, color, speed):
        cx, cy = float(pos[0]), float(pos[1])
        for _ in range(n):
            ang = random.uniform(0.0, 2.0 * math.pi)
            sp = speed * random.uniform(0.4, 1.0)
            self.particles.append(
                Particle(
                    cx, cy,
                    math.cos(ang) * sp, math.sin(ang) * sp,
                    random.uniform(0.3, 0.7), random.uniform(2.0, 5.0),
                    tuple(color),
                )
            )
        if len(self.particles) > 600:
            del self.particles[: len(self.particles) - 600]

    def _wind(self, src, dst, color=None, n=40, speed=280.0):
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        base = dst - src
        nb = float(np.hypot(*base)) + 1e-6
        base = base / nb * speed
        if color is None:
            color = self.color
        for _ in range(n):
            vx = base[0] + random.uniform(-120, 120)
            vy = base[1] + random.uniform(-120, 120)
            self.particles.append(
                Particle(
                    float(src[0]), float(src[1]),
                    vx, vy,
                    random.uniform(0.4, 0.8), random.uniform(2.0, 4.0),
                    color,
                )
            )
        if len(self.particles) > 600:
            del self.particles[: len(self.particles) - 600]

    def _advance_particles(self, dt):
        drop = math.exp(-4.0 * dt)
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vx *= drop
                p.vy *= drop
                alive.append(p)
        self.particles = alive

    def _draw_particles(self, frame):
        h, w = frame.shape[:2]
        for p in self.particles:
            f = max(p.life / p.max_life, 0.0)
            x, y = int(p.x), int(p.y)
            if not (0 <= x < w and 0 <= y < h):
                continue
            r = max(int(p.size * (0.5 + 0.5 * f)), 1)
            col = tuple(int(min(255, c * (0.35 + 0.65 * f))) for c in p.color)
            cv2.circle(frame, (x, y), r, col, -1, cv2.LINE_AA)
            if f > 0.6:
                cv2.circle(frame, (x, y), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    # ---------------------------------------------------------------- rings

    def _start_ring(self, pos, color=None):
        self._ring = {
            "t": 0.0,
            "dur": 0.45,
            "pos": np.array(pos, dtype=np.float32),
            "color": color or self.color,
            "r0": max(self.radius, 12.0),
        }

    def _draw_ring(self, frame):
        ring = self._ring
        if ring is None:
            return
        f = min(ring["t"] / ring["dur"], 1.0)
        r = int(ring["r0"] * (1.0 + 3.0 * f))
        thick = max(2, int(3 * (1.0 - f)))
        col = tuple(int(max(0, c * (1.0 - f * 0.7))) for c in ring["color"])
        center = (int(ring["pos"][0]), int(ring["pos"][1]))
        cv2.circle(frame, center, r, col, thick, cv2.LINE_AA)

    # ---------------------------------------------------------------- rendering

    def render(self, frame, filtered):
        if self.state == STATE_GONE:
            self._draw_particles(frame)
            return

        if self._ring is not None:
            self._draw_ring(frame)

        if self.state == STATE_STRETCH and self._hand_a is not None:
            a = tuple(int(v) for v in self._hand_a)
            b = tuple(int(v) for v in self._hand_b)
            cv2.line(
                frame, a, b,
                self._dim(self.color, 70), max(2, int(self.ry * 0.12)),
                cv2.LINE_AA,
            )

        cx = self.pos[0]
        cy = self.pos[1]
        if self.state in (STATE_HELD, STATE_SPAWN, STATE_CATCH, STATE_PUMP, STATE_TWIN):
            cy = cy + math.sin(self.state_t * 3.2) * 2.5

        if self.state == STATE_STRETCH:
            self._draw_orb(frame, filtered, cx, cy, self.rx, self.ry, self.angle, grip=True)
        elif self.state == STATE_PINCHED:
            self._draw_orb(frame, filtered, cx, cy, self.radius * 0.7, self.radius, 0.0, glow=1.25)
        elif self.state == STATE_CATCH:
            self._draw_orb(
                frame, filtered, cx, cy,
                self.radius * self._squash_x, self.radius * self._squash_y, 0.0,
            )
        elif self.state in (STATE_FLIGHT, STATE_HOP):
            vx, vy = float(self.vel[0]), float(self.vel[1])
            nv = math.hypot(vx, vy) + 1e-6
            ang = math.degrees(math.atan2(vy, vx))
            self._draw_orb(
                frame, filtered, self.pos[0], self.pos[1],
                self.radius * 1.22, self.radius * 0.85, ang,
            )
        else:
            self._draw_orb(frame, filtered, cx, cy, self.radius, self.radius, 0.0)

        if self.state in (STATE_TWIN, STATE_PUMP):
            self._draw_band(frame)

        self._draw_particles(frame)

    def _draw_band(self, frame):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = max(int(self.radius * 1.22), 16)
        for k in range(3):
            col = self._rainbow(k)
            a0 = int((self.state_t * 70 + k * 120) % 360)
            cv2.ellipse(frame, (cx, cy), (r, r), 0, a0, a0 + 55, col, 2, cv2.LINE_AA)

    def _draw_orb(self, frame, filtered, cx, cy, rx, ry, angle, grip=False,
                  squash=(1.0, 1.0), glow=1.0):
        h, w = frame.shape[:2]
        R = max(int(math.ceil(max(rx, ry))), 1)
        feather = max(3.0, min(16.0, R * 0.12))
        glow_m = min(int(math.ceil(R * 0.5 * glow)), 70)
        m = int(math.ceil(3.0 * feather)) + glow_m
        x0 = max(int(cx) - R - m, 0)
        y0 = max(int(cy) - R - m, 0)
        x1 = min(int(cx) + R + m, w)
        y1 = min(int(cy) + R + m, h)
        if x1 - x0 <= 4 or y1 - y0 <= 4:
            return

        ss = self.ss
        region = frame[y0:y1, x0:x1]
        pw, ph = region.shape[1], region.shape[0]
        big = cv2.resize(region, (pw * ss, ph * ss), interpolation=cv2.INTER_LINEAR)
        dx = (int(cx) - x0) * ss
        dy = (int(cy) - y0) * ss
        rxi = max(int(rx * squash[0] * ss), 1)
        ryi = max(int(ry * squash[1] * ss), 1)
        fea = max(2.0, feather * ss)

        color = self.color
        dark = self._dim(color, 90)
        bright = self._bright(color, 110)

        layer = np.zeros(big.shape, dtype=np.uint8)
        cv2.ellipse(layer, (dx, dy), (rxi, ryi), int(angle), 0, 360, dark, -1)
        cv2.ellipse(
            layer, (dx, dy),
            (max(1, int(rxi * 0.8)), max(1, int(ryi * 0.8))),
            int(angle), 0, 360, color, -1,
        )
        cv2.ellipse(
            layer, (dx, dy),
            (max(1, int(rxi * 0.55)), max(1, int(ryi * 0.55))),
            int(angle), 0, 360, bright, -1,
        )
        sigma = max(2.0, min(R * ss * 0.28, 9.0 * ss)) * glow
        blur = _soften(layer, sigma)
        cv2.add(big, blur, dst=big)

        if filtered is not None:
            freg = filtered[y0:y1, x0:x1]
            fbig = cv2.resize(freg, (pw * ss, ph * ss), interpolation=cv2.INTER_LINEAR)
            mask = np.zeros((ph * ss, pw * ss), dtype=np.uint8)
            cv2.ellipse(mask, (dx, dy), (rxi, ryi), int(angle), 0, 360, 255, -1)
            if fea > 1:
                mask = _soften(mask, fea)
            m3 = cv2.merge([mask, mask, mask])
            base = cv2.multiply(big, 255 - m3, scale=1.0 / 255.0, dtype=cv2.CV_8U)
            fill = cv2.multiply(fbig, m3, scale=1.0 / 255.0, dtype=cv2.CV_8U)
            big = cv2.add(base, fill)

        cv2.ellipse(
            big, (dx, dy), (rxi, ryi), int(angle), 0, 360,
            bright, max(1, int(R * ss * 0.05)), cv2.LINE_AA,
        )
        pr_x = max(2, int(rxi * (1 + 0.06 * self.pulse)))
        pr_y = max(2, int(ryi * (1 + 0.06 * self.pulse)))
        cv2.ellipse(
            big, (dx, dy), (pr_x, pr_y), int(angle), 0, 360,
            dark, max(1, int(ss)), cv2.LINE_AA,
        )
        hx = max(dx - int(rxi * 0.30), 0)
        hy = max(dy - int(ryi * 0.32), 0)
        cv2.circle(big, (hx, hy), max(1, int(R * ss * 0.10)), (255, 255, 255), -1, cv2.LINE_AA)

        if grip and self._hand_a is not None:
            for hp in (self._hand_a, self._hand_b):
                gx = (int(hp[0]) - x0) * ss
                gy = (int(hp[1]) - y0) * ss
                if 0 <= gx < pw * ss and 0 <= gy < ph * ss:
                    gg = max(4, int(self.ry * 0.5 * ss))
                    cv2.circle(big, (gx, gy), gg, self._dim(color, 30), -1, cv2.LINE_AA)
                    cv2.circle(big, (gx, gy), max(2, int(gg * 0.4)), bright, 2, cv2.LINE_AA)
                    cv2.circle(big, (gx, gy), max(2, int(gg * 0.18)), (255, 255, 255), -1, cv2.LINE_AA)

        region[:, :] = cv2.resize(big, (pw, ph), interpolation=cv2.INTER_AREA)