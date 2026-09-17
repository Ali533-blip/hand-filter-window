import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_gestures
import test_states
import test_new_gestures
import test_dual


def main():
    for mod, label in (
        (test_gestures, "gestures"),
        (test_states, "orbs/states"),
        (test_new_gestures, "gesture actions"),
        (test_dual, "dual orbs / events"),
    ):
        try:
            print("== %s ==" % label)
            mod.run()
        except AssertionError as e:
            print("FAIL [%s]: %s" % (label, e))
            return 1
        print()
    print("ALL SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())