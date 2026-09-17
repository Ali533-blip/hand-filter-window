r"""
gestures.py

All gesture logic lives here, built on the 21-point MediaPipe hand landmark
layout:

        8   12  16  20      <- fingertips (index, middle, ring, pinky)
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18      <- PIP joints
        |   |   |   |
        5   9   13  17      <- MCP knuckles
         \  |   |  /
          \ |   | /
       4    \  |  /
        \    \ | /
         3    \|/
          \    0            <- wrist
           2
            \
             1
              \
              (thumb, 1-4)

Three core gestures drive the app window:

1. FRAME  - two hands out, thumb tip + index tip of each hand = 4 points.
            Those 4 points are the corners of the filter window.
2. PRAYER - both palms pressed together, fingers pointing up -> the window
            expands to fill the whole frame.
3. FIST   - one hand, fingers curled in -> a glowing "orbs" forms in the fist.

The orbs (ball.py / ball_manager.py) add more single-hand poses on top:

- PINCH   - thumb + index together, three fingers out    -> squeeze / release
- TAP     - thumb + middle touching, >=2 fingers out     -> cycle colour
- POINT   - only the index finger out                    -> orb hops to fingertip
- GUN     - index + thumb out, three fingers curled      -> orb fires a bolt
- THUMBSUP- all four fingers curled, thumb up            -> orb levels up
- VICTORY - index + middle up, ring + pinky down         -> orb celebrates
- FLAT    - open hand, fingers together                  -> wave (see WaveDetector)
- CLAP    - two palms closing fast                       -> see ClapDetector

Thresholds below are all expressed relative to `hand_scale()` (roughly the
palm size in pixels) rather than fixed pixel distances, so the gestures
keep working whether your hand is close to the camera or far from it.
"""

import numpy as np

WRIST = 0

THUMB_TIP = 4

INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def hand_scale(points):
    """A per-hand size reference (wrist -> middle knuckle), so thresholds
    scale automatically with distance from the camera."""
    return _dist(points[WRIST], points[MIDDLE_MCP]) + 1e-6


def hand_centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def is_fist(points):
    """True when index/middle/ring/pinky tips are curled back toward the
    wrist instead of extended away from it."""
    scale = hand_scale(points)
    wrist = points[WRIST]
    curled = 0
    for tip_i, mcp_i in zip(FINGER_TIPS, FINGER_MCPS):
        tip_dist = _dist(points[tip_i], wrist)
        mcp_dist = _dist(points[mcp_i], wrist)
        # an extended finger's tip sits well past its own knuckle; a curled
        # one folds back close to (or past) it
        if tip_dist < mcp_dist + 0.35 * scale:
            curled += 1
    return curled >= 3


def is_prayer_pose(hands):
    """Two hands, palms pressed together, fingers pointing roughly upward."""
    if len(hands) != 2:
        return False

    a, b = hands[0]["points"], hands[1]["points"]
    scale = (hand_scale(a) + hand_scale(b)) / 2.0

    wrist_gap = _dist(a[WRIST], b[WRIST])
    tip_gap = _dist(a[MIDDLE_TIP], b[MIDDLE_TIP])
    if wrist_gap > 2.2 * scale or tip_gap > 1.6 * scale:
        return False

    a_pointing_up = a[MIDDLE_TIP][1] < a[WRIST][1]
    b_pointing_up = b[MIDDLE_TIP][1] < b[WRIST][1]
    return a_pointing_up and b_pointing_up


def get_frame_corners(hands):
    """Two hands -> thumb tip + index tip of each become the 4 corners of
    the filter window, ordered so cv2.fillConvexPoly draws a clean quad
    instead of a bowtie."""
    if len(hands) != 2:
        return None

    pts = []
    for hand in hands:
        p = hand["points"]
        pts.append(p[THUMB_TIP])
        pts.append(p[INDEX_TIP])

    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)

    def angle(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    ordered = sorted(pts, key=lambda p: angle(p))
    return np.array(ordered, dtype=np.float32)


def _finger_extended(points, tip_i, mcp_i):
    """A finger counts as extended when its tip sits well past its own
    knuckle (measured from the wrist), not when it's curled back."""
    scale = hand_scale(points)
    wrist = points[WRIST]
    return _dist(points[tip_i], wrist) > _dist(points[mcp_i], wrist) + 0.3 * scale


def _open_fingers(points):
    """How many of index/middle/ring/pinky are currently extended."""
    return sum(
        1
        for t, m in zip(FINGER_TIPS, FINGER_MCPS)
        if _finger_extended(points, t, m)
    )


def is_open_hand(points):
    """An open palm: all four fingers extended past their knuckles."""
    return _open_fingers(points) == 4


def is_pinch(points, factor=0.6):
    """Thumb + index tips together (and clearly closer to each other than
    the thumb is to the middle tip) with the other three fingers out. The
    gap-between-gaps rule is what separates a pinch (thumb+index) from a
    'tap' (thumb+middle), and the extended-fingers rule keeps it from
    firing while holding a plain fist."""
    scale = hand_scale(points)
    idx_gap = _dist(points[THUMB_TIP], points[INDEX_TIP])
    mid_gap = _dist(points[THUMB_TIP], points[MIDDLE_TIP])
    if idx_gap >= factor * scale or idx_gap >= 0.85 * mid_gap:
        return False
    return all(
        _finger_extended(points, t, m)
        for t, m in zip((MIDDLE_TIP, RING_TIP, PINKY_TIP), (MIDDLE_MCP, RING_MCP, PINKY_MCP))
    )


def is_middle_thumb_touch(points, factor=0.55):
    """The 'tap' that changes the ball's colour: thumb + middle tips touching.
    Requires the thumb to be clearly closer to the middle tip than to the
    index tip (so a pinch doesn't double-fire it) and at least two fingers
    extended (so a plain fist doesn't)."""
    scale = hand_scale(points)
    idx_gap = _dist(points[THUMB_TIP], points[INDEX_TIP])
    mid_gap = _dist(points[THUMB_TIP], points[MIDDLE_TIP])
    if mid_gap >= factor * scale or mid_gap >= 0.85 * idx_gap:
        return False
    return _open_fingers(points) >= 2


# ---------------------------------------------------------------- new poses

def _extended_flags(points):
    """Per-finger extended booleans, order: index, middle, ring, pinky."""
    return tuple(
        _finger_extended(points, t, m)
        for t, m in zip(FINGER_TIPS, FINGER_MCPS)
    )


def is_pointing(points):
    """Only the index finger is extended (classic 'point'). The thumb must
    stay tucked in toward the palm so a gun (thumb out) reads differently."""
    f = _extended_flags(points)
    if not f[0] or any(f[1:]):
        return False
    scale = hand_scale(points)
    return _dist(points[THUMB_TIP], points[MIDDLE_MCP]) <= 0.9 * scale


def point_direction(points):
    """Unit vector from the wrist toward the index fingertip."""
    a = np.array(points[WRIST], dtype=np.float32)
    b = np.array(points[INDEX_TIP], dtype=np.float32)
    v = b - a
    n = float(np.hypot(*v)) + 1e-6
    return v / n


def is_victory(points):
    """Index + middle up, ring + pinky curled ('peace' / 'victory')."""
    f = _extended_flags(points)
    return f[0] and f[1] and not f[2] and not f[3]


def is_gun(points, factor=0.55):
    """Index + thumb pointed out, other three fingers curled. The thumb must
    be clearly away from the index tip (so a pinch can't double-fire it) and
    clearly away from the palm (so a plain point / fist can't)."""
    scale = hand_scale(points)
    f = _extended_flags(points)
    if not f[0] or any(f[1:]):
        return False
    idx_to_thumb = _dist(points[INDEX_TIP], points[THUMB_TIP])
    thumb_out = _dist(points[THUMB_TIP], points[MIDDLE_MCP]) > 0.9 * scale
    return thumb_out and idx_to_thumb > factor * scale


def is_thumbs_up(points, factor=0.3):
    """All four fingers curled, thumb sticking up past the knuckle row.
    A plain fist keeps the thumb in; a thumbs-up pops it out above the palm."""
    scale = hand_scale(points)
    if _open_fingers(points) != 0:
        return False
    tip = points[THUMB_TIP]
    mid = points[MIDDLE_MCP]
    above = (mid[1] - tip[1]) > factor * scale
    far = _dist(tip, mid) > 0.9 * scale
    return above and far


def is_flat_hand(points, factor=0.45):
    """Open hand but with the fingers together (side by side, not splayed) -
    the classic 'wave / hold it up' hand."""
    if not is_open_hand(points):
        return False
    scale = hand_scale(points)
    adj = [
        _dist(points[INDEX_TIP], points[MIDDLE_TIP]),
        _dist(points[MIDDLE_TIP], points[RING_TIP]),
        _dist(points[RING_TIP], points[PINKY_TIP]),
    ]
    return max(adj) < factor * scale


# ------------------------------------------------------------ live detectors

class WaveDetector:
    """A flat hand rocking side to side (a "wave"). Tracks the wrist x-axis,
    counts direction reversals inside a time window. dt-based so it works in
    the app loop and in synthetic tests identically."""

    def __init__(self, window=1.0, reversals=3, min_amp=16.0, cooldown=1.0):
        self.window = window
        self.reversals = reversals
        self.min_amp = min_amp
        self.cooldown = cooldown
        self._samples = []
        self._t = 0.0
        self._fired_at = -10.0

    def _reset(self):
        self._samples = []

    def update(self, points, dt):
        """points = 21-landmark list for the waving hand (or None if the
        window lost that hand). Returns True once per completed wave."""
        self._t += dt
        if points is None or not is_flat_hand(points):
            self._reset()
            return False
        if self._t - self._fired_at < self.cooldown:
            return False

        self._samples.append((self._t, points[WRIST][0]))
        while self._samples and self._t - self._samples[0][0] > self.window:
            self._samples.pop(0)

        x = [s[1] for s in self._samples]
        if len(x) < 4:
            return False

        reversals = 0
        prev = 0.0
        for i in range(1, len(x)):
            d = x[i] - x[i - 1]
            if abs(d) < self.min_amp * 0.5:
                continue
            s = 1.0 if d > 0 else -1.0
            if prev != 0.0 and s != prev:
                reversals += 1
            prev = s
        if reversals >= self.reversals:
            self._fired_at = self._t
            self._reset()
            return True
        return False


class ClapDetector:
    """Two open palms swing together quickly: the wrist gap is now small and
    dropped fast recently. dt-based like WaveDetector."""

    def __init__(self, cooldown=1.0):
        self.cooldown = cooldown
        self._samples = []
        self._t = 0.0
        self._fired_at = -10.0

    def update(self, hands, dt):
        self._t += dt
        if len(hands) < 2:
            self._samples = []
            return False
        if self._t - self._fired_at < self.cooldown:
            return False

        a, b = hands[0]["points"], hands[1]["points"]
        gap = _dist(a[WRIST], b[WRIST])
        scale = min(hand_scale(a), hand_scale(b))
        self._samples.append((self._t, gap, scale))
        while self._samples and self._t - self._samples[0][0] > 0.4:
            self._samples.pop(0)

        peak = max(g for _, g, _ in self._samples)
        if gap < 0.35 * scale and (peak - gap) > 0.5 * scale:
            self._fired_at = self._t
            self._samples = []
            return True
        return False
