import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gestures import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, INDEX_MCP, MIDDLE_MCP, PINKY_MCP,
    hand_scale, is_fist, is_pinch, is_open_hand, is_middle_thumb_touch,
    is_pointing, is_gun, is_thumbs_up, is_victory, is_flat_hand,
    is_prayer_pose, get_frame_corners, WaveDetector, ClapDetector,
)
from tests.fake_hand import make_hand, hand

scl = 100.0


def run():
    checks = []

    def ok(name):
        checks.append(name)

    # base poses sanity
    h = make_hand(300, 200, scl, "fist")
    assert is_fist(h), "fist"
    assert not is_open_hand(h), "fist not open"
    assert not is_pinch(h), "fist not pinch"
    assert not is_pointing(h), "fist not point"
    assert not is_gun(h), "fist not gun"
    assert not is_thumbs_up(h), "fist not thumbs"
    assert abs(hand_scale(h) - scl) < 1e-3, "scale == scl"
    ok("fist base")

    h = make_hand(300, 200, scl, "open")
    assert is_open_hand(h), "open"
    assert not is_fist(h), "open not fist"
    ok("open base")

    h = make_hand(300, 200, scl, "pinch")
    assert is_pinch(h), "pinch"
    assert is_open_hand(h), "pinch quack-open"
    ok("pinch base")

    h = make_hand(300, 200, scl, "tap")
    assert is_middle_thumb_touch(h), "tap (middle+thumb)"
    ok("tap base")

    # pointing / gun / thumbs / victory / flat
    for pose, fn in (
        ("point", is_pointing), ("gun", is_gun), ("thumbs", is_thumbs_up),
        ("victory", is_victory), ("flat", is_flat_hand),
    ):
        h = make_hand(300, 200, scl, pose)
        assert fn(h), "%s detects" % pose
    ok("5 new poses detect")

    # separation: gun vs point (thumb-in vs thumb-out)
    h = make_hand(300, 200, scl, "gun")
    assert is_gun(h), "gun"
    assert not is_pointing(h), "gun != point"
    h = make_hand(300, 200, scl, "point")
    assert is_pointing(h), "point"
    assert not is_gun(h), "point != gun"

    # prayer frame corners
    a = hand(200, 200, scl, "point")
    b = hand(500, 300, scl, "point")
    got = get_frame_corners([a, b])
    assert got is not None, "corners"
    assert got.shape == (4, 2), "four float corners: %r" % (got.shape,)
    assert len(set(map(tuple, np.round(got, 4)))) == 4, "corners are distinct"
    h = hand(200, 200, scl, "fist")
    assert is_prayer_pose([h, hand(400, 400, scl, "pinch")]) is False, "no false prayer"
    ok("prayer + corners")

    # wave detector: flat hand rocking x
    wave = WaveDetector(reversals=2, min_amp=15.0, cooldown=0.5)
    fired = False
    for i in range(60):
        x = 300 + (30 if (i // 4) % 2 == 0 else -30)
        h = make_hand(x, 200, scl, "flat")
        if wave.update(h, 1 / 60):
            fired = True
            break
    assert fired, "wave fires"
    wave2 = WaveDetector(reversals=3)
    fired2 = False
    for i in range(60):
        h = make_hand(300 + i, 200, scl, "fist")  # no flat hand
        if wave2.update(h, 1 / 60):
            fired2 = True
    assert not fired2, "non-flat no wave"
    ok("wave detector")

    # clap detector: two open hands swing toward each other
    clap = ClapDetector(cooldown=0.5)
    fired = False
    for i in range(60):
        gapx = max(400 - i * 11, 18)  # closing fast down to near-touch
        a = hand(300, 200, scl, "open")
        b = hand(300 + gapx, 200, scl, "open")
        if clap.update([a, b], 1 / 60):
            fired = True
            break
    assert fired, "clap fires"
    clap2 = ClapDetector()
    fired2 = False
    a = hand(300, 200, scl, "open")
    b = hand(800, 200, scl, "open")
    for _ in range(60):
        if clap2.update([a, b], 1 / 60):
            fired2 = True
    assert not fired2, "far hands no clap"
    ok("clap detector")

    print("test_gestures: %d checks passed" % len(checks))


if __name__ == "__main__":
    run()