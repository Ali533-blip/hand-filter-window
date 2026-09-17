import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gestures import hand_centroid
from ball import BallSystem, STATE_HELD, STATE_HOP, STATE_FLIGHT, STATE_PUMP, STATE_TWIN
from tests.fake_hand import hand

DT = 1 / 60
scl = 100.0


def run():
    checks = []

    def ok(name):
        checks.append(name)

    # ---------------- point -> hop
    ball = BallSystem(ss=1)
    h_a = hand(200, 250, scl, "fist")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a], DT)
    assert ball.state == STATE_HELD, "held before point"

    point_hand = hand(200, 250, scl, "point")
    tip = point_hand["points"][8].copy()
    ball.update([point_hand, hand(640, 250, scl, "open")], DT)
    assert ball.state == STATE_HOP, "point -> hop: %r" % ball.state
    hop_min = float("inf")
    for _ in range(60):
        ball.update([point_hand, hand(640, 250, scl, "open")], DT)
        if ball.state in (STATE_HOP,):
            hop_min = min(hop_min, float(np.hypot(*(ball.pos - tip))))
        elif ball.state == STATE_HELD:
            break
    assert ball.state == STATE_HELD, "hop lands -> held: %r" % ball.state
    assert hop_min < 40, "hop arc passes over the fingertip: %.1f" % hop_min
    ok("point -> hop over fingertip")

    # ---------------- gun -> bolt (two hands: gun on A, open hand B far)
    ball = BallSystem(ss=1)
    h_a = hand(200, 250, scl, "fist")
    h_b = hand(540, 260, scl, "open")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a, h_b], DT)
    ball.update([hand(200, 250, scl, "gun"), h_b], DT, forced_idx=0)
    assert ball.state == STATE_FLIGHT, "gun -> bolt: %r" % ball.state
    assert ball._bolt, "bolt is set"
    for _ in range(300):
        global_h = hand(220, 250, scl, "fist")
        ball.update([global_h, h_b], DT)
        if ball.state in (STATE_HELD, "gone"):
            break
    assert ball.state == STATE_HELD, "bolt lands + caught: %r" % ball.state
    ok("gun -> bolt to other hand")

    # ---------------- thumbs-up -> pump
    ball = BallSystem(ss=1)
    h_a = hand(300, 250, scl, "fist")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a], DT)
    before = ball.theme_idx()
    ball.update([hand(300, 250, scl, "thumbs"), hand(700, 250, scl, "open")], DT)
    assert ball.state == STATE_PUMP or ball._pump_cd > 0.5, "thumbs -> pump: %r" % ball.state
    assert ball._radius_mult > 1.0, "pump enlarges ball"
    # settle many seconds -> returns to held
    for _ in range(400):
        ball.update([hand(300, 250, scl, "thumbs"), hand(700, 250, scl, "open")], DT)
        if ball.state == "held":
            break
    assert ball.state == "held", "pump finishes -> held: %r" % ball.state
    assert ball._radius_mult == 1.0, "pump resets scale"
    assert ball.theme_idx() == before, "pump is not a theme change"
    ok("thumbs-up -> pump then settle")

    # ---------------- victory -> celebration (particle burst + ring)
    ball = BallSystem(ss=1)
    h_a = hand(300, 250, scl, "fist")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a], DT)
    n0 = len(ball.particles)
    ball.update([hand(300, 250, scl, "victory"), hand(700, 250, scl, "open")], DT)
    assert len(ball.particles) > n0 + 20, "victory bursts"
    assert ball._ring is not None, "victory ring"
    ok("victory -> celebration burst")

    print("test_new_gestures: %d checks passed" % len(checks))


if __name__ == "__main__":
    run()