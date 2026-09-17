import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gestures import WRIST, hand_centroid, hand_scale
from ball import BallSystem, STATE_GONE, STATE_HELD, STATE_PINCHED, STATE_DESTROY, \
    STATE_FLIGHT, STATE_CATCH, STATE_STRETCH
from tests.fake_hand import hand, make_hand, frame

DT = 1 / 60
scl = 100.0


def run():
    checks = []

    def ok(name):
        checks.append(name)

    ball = BallSystem(ss=1)
    frame_img = frame(640, 480)

    # 1. spawn -> held
    h1 = hand(300, 250, scl, "fist")
    hands = [h1]
    c = hand_centroid(h1["points"])
    ball.spawn(c, scl)
    assert ball.state == "spawn", "spawned"
    for _ in range(80):
        ball.update(hands, DT)
    assert ball.state == "held", "held after spawn: %r" % ball.state
    assert ball.is_active()
    ok("spawn -> held")

    # 2. tap cycles theme
    theme0 = ball.theme_idx()
    h1 = hand(300, 250, scl, "tap")
    for _ in range(2):
        ball.update([h1], DT)
    assert ball.theme_idx() != theme0, "tap cycles theme"
    ok("tap cycles theme")

    # 3. open palm keeps the ball (no destroy)
    h1 = hand(300, 250, scl, "open")
    for _ in range(10):
        ball.update([h1], DT)
    assert ball.state == "held", "open palm keeps ball: %r" % ball.state
    ok("open palm keeps ball")

    # 4. pinch -> pinched, then open hand -> destroy -> gone
    h1 = hand(300, 250, scl, "pinch")
    for _ in range(2):
        ball.update([h1], DT)
    assert ball.state == "pinched", "pinched: %r" % ball.state
    h1 = hand(300, 250, scl, "open")
    for _ in range(2):
        ball.update([h1], DT)
    assert ball.state == "destroy", "destroy: %r" % ball.state
    for _ in range(60):
        ball.update([h1], DT)
    assert ball.state == "gone", "gone after destroy"
    assert not ball.is_active()
    ok("pinch -> destroy -> gone")

    # 5. throw -> flight -> catch by the other hand -> held
    h_a = hand(200, 250, scl, "fist")
    h_b = hand(520, 250, scl, "open")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a], DT)
    assert ball.state == "held", "held pre-throw"

    # warm the wrist tracker, then flick hand A hard along +x
    for _ in range(4):
        h_a = hand(200, 250, scl, "fist")
        ball.update([h_a, h_b], DT)
    for i in range(6):
        h_a = hand(232 + i * 14, 250, scl, "fist")  # 14px/frame on a 100px hand
        ball.update([h_a, h_b], DT)
    assert ball.state == "flight", "throw -> flight: %r" % ball.state

    for _ in range(300):
        h_a = hand(330, 250, scl, "fist")  # hand A settles; orb is airborne
        ball.update([h_a, h_b], DT)
        if ball.state == "held":
            break
    assert ball.state == "held", "catch -> held: %r" % ball.state
    ok("throw -> catch -> held")

    # 6. stretch with two hands (fist B joins), grows, then collapse on leave
    ball = BallSystem(ss=1)
    h_a = hand(200, 250, scl, "fist")
    ball.spawn(hand_centroid(h_a["points"]), scl)
    for _ in range(80):
        ball.update([h_a], DT)
    assert ball.state == STATE_HELD, "held before stretch"
    for _ in range(10):
        ball.update([h_a, hand(286, 250, scl, "fist")], DT)
    assert ball.state == "stretch", "two hands -> stretch: %r" % ball.state
    rx_after = ball.rx
    # spread apart a bit -> persists (stays stretch)
    for _ in range(10):
        ball.update([hand(170, 250, scl, "fist"), hand(360, 250, scl, "fist")], DT)
    assert ball.state == "stretch", "persists while pulled: %r" % ball.state
    assert ball.rx > rx_after, "grows when spread: %r vs %r" % (ball.rx, rx_after)
    # hand leaves -> falls back to held
    for _ in range(10):
        ball.update([hand(170, 250, scl, "fist")], DT)
    assert ball.state == "held", "collapse to held: %r" % ball.state
    ok("stretch grows + persists + collapses")

    print("test_states: %d checks passed" % len(checks))


if __name__ == "__main__":
    run()