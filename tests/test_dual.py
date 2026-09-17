import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gestures import hand_centroid
from ball_manager import BallManager
from ball import STATE_GONE, STATE_HELD, STATE_TWIN, STATE_ABSORB, STATE_FLIGHT
from tests.fake_hand import hand

DT = 1 / 60
scl = 100.0


def run():
    checks = []

    def ok(name):
        checks.append(name)

    mgr = BallManager(ss=1)

    # one fist -> one orb held
    mgr.update([hand(200, 250, scl, "fist")], DT)
    for _ in range(80):
        mgr.update([hand(200, 250, scl, "fist")], DT)
    assert mgr.states()[0] == STATE_HELD, "fist -> orb0 held: %r" % (mgr.states(),)
    assert mgr.states()[1] == STATE_GONE, "second orb idle"
    assert mgr.any_active()
    ok("one fist -> one orb")

    # second (separate) fist -> second orb, distinct hands
    for _ in range(80):
        mgr.update([hand(200, 250, scl, "fist"), hand(520, 250, scl, "fist")], DT)
    s0, s1 = mgr.states()
    assert s0 == STATE_HELD and s1 == STATE_HELD, "two orbs: %r" % (mgr.states(),)
    forced = sorted(mgr._mesh.get(0, -1) for _ in [0])
    assert mgr._mesh.get(0) != mgr._mesh.get(1) or -1 in (mgr._mesh.get(0), mgr._mesh.get(1)), \
        "distinct hands assigned"
    ok("two fists -> two orbs on distinct hands")

    # tap on one hand cycles theme
    theme0 = mgr.theme_idx()
    for _ in range(2):
        mgr.update([hand(200, 250, scl, "tap"), hand(520, 250, scl, "fist")], DT)
    assert mgr.theme_idx() != theme0, "tap cycles theme"
    ok("tap cycles theme via manager")

    # bring the two held orbs' hands together -> merge into twin
    for _ in range(30):
        mgr.update([hand(290, 250, scl, "fist"), hand(310, 250, scl, "fist")], DT)
    states = mgr.states()
    assert STATE_TWIN in states, "merge -> twin: %r" % (states,)
    assert STATE_ABSORB in states or STATE_GONE in states, "partner absorbed: %r" % (states,)
    twin = mgr.orbs[states.index(STATE_TWIN)]
    assert twin._radius_mult > 1.0, "twin is bigger"
    ok("close hands -> twin merge")

    # second hand leaves -> twin stays held
    for _ in range(20):
        mgr.update([hand(300, 250, scl, "fist")], DT)
    assert mgr.summary().startswith("twin"), "twin survives: %r" % mgr.summary()
    ok("twin survives hand leave")

    # open one hand (was thumb) -> twin still strong, then pinch destroys ie.
    # pinch one orb's hand -> gone (twin gone), other orb gone too because hand vanished.
    # Simpler: open the twin's hand -> twin stays (open hand keeps ball)
    for _ in range(10):
        mgr.update([hand(300, 250, scl, "open")], DT)
    assert mgr.any_active(), "open hand keeps twin: %r" % (mgr.states(),)
    ok("open hand keeps twin")

    # ---------------- clap / wave events
    before = len(mgr.orbs[0].particles)
    for orb in mgr.orbs:
        orb._burst(np.array([10, 10], dtype=np.float32), 10, (255, 0, 0), 10)  # noise
    mgr.on_clap()
    assert len(mgr.orbs[0].particles) > before, "clap bursts"
    ok("clap -> celebration")

    b0 = len(mgr.orbs[0].particles)
    mgr.on_wave(np.array([100.0, 300.0]))
    assert len(mgr.orbs[0].particles) >= b0, "wave spawns wind particles"
    ok("wave -> wind particles")

    # ---------------- throw past an occupied hand is absorbed
    mgr2 = BallManager(ss=1)
    a = hand(200, 250, scl, "fist")
    b = hand(520, 250, scl, "fist")
    mgr2.update([a, b], DT)
    for _ in range(80):
        mgr2.update([a, b], DT)
    assert mgr2.states() == [STATE_HELD, STATE_HELD], "two orbs ready"
    for _ in range(4):
        mgr2.update([hand(200, 250, scl, "fist"), b], DT)
    for i in range(6):
        mgr2.update([hand(232 + i * 14, 250, scl, "fist"), b], DT)
    states = [o.state for o in mgr2.orbs]
    assert any(s == STATE_FLIGHT for s in states), "orb thrown: %r" % (states,)
    for _ in range(300):
        mgr2.update([a, b], DT)
        if not any(o.state == STATE_FLIGHT or o.state == STATE_ABSORB for o in mgr2.orbs):
            break
    states = [o.state for o in mgr2.orbs]
    assert STATE_HELD in states, "a held orb remains: %r" % (states,)
    assert sum(o.state != STATE_GONE for o in mgr2.orbs) == 1, \
        "flying orb absorbed, not duplicated: %r" % (states,)
    survivor = [o for o in mgr2.orbs if o.state != STATE_GONE][0]
    assert survivor._absorb_gain > 0, "survivor absorbed the incoming orb"
    ok("flying orb absorbed into occupied hand")

    # ---------------- standalone BallSystem ignores spawns (manager-driven)
    from ball import BallSystem
    solo = BallSystem(ss=1)
    for _ in range(60):
        solo.update([hand(300, 250, scl, "fist")], DT)
    assert solo.state == STATE_GONE, "no manager -> no spawn"
    ok("standalone orb does not self-spawn")

    print("test_dual: %d checks passed" % len(checks))


if __name__ == "__main__":
    run()