"""
main.py

Hand-Filter Window
------------------
Hold up both hands, thumb + index finger of each, like you're framing a
shot with your fingers. Those four fingertips become the corners of a
little filter window that stretches and moves with your hands.

On top of that lives the interactive "filter orbs" (ball.py + ball_manager):

- Press your palms together (prayer pose) -> the window blows up to fill
  the whole screen.
- Make a fist -> a glowing orb made of the current filter forms and rests
  in that hand. Make a fist with each hand and you get two orbs.
- Pinch thumb + index -> the orb squeezes smaller. Open the hand after
  pinching -> the orb dissolves in a burst of particles.
- Flick the hand holding an orb -> it's thrown as a ball (aim at the other
  hand to pass it there; if the other hand already holds an orb, the flying
  one is absorbed into it).
- Bring two held orbs' hands together (or two hands up to one orb) -> the
  orbs merge into one bigger twin orb / the orb stretches like taffy
  between your hands.
- Bring a second hand up near a resting orb -> it stretches between your
  hands. Part them -> it falls back into one hand.
- Point, gun (point + thumb out), thumbs-up, victory, wave, and clap all
  trigger extra tricks; taps (middle finger + thumb) cycle the colour themes.

Controls:
    f / d     cycle filter forward / backward
    c         cycle the orb colour theme
    l         toggle hand-landmark overlay (debug)
    h         toggle the on-screen HUD
    s         save a screenshot (hand_filter_shot_<time>.png)
    r         toggle MP4 recording
    q / Esc   quit (settings are saved to config.json)

Run it:
    python main.py
    python main.py --camera 1           # different webcam
    python main.py --width 1920 --height 1080
    python main.py --ss 2 --record      # 2x supersampling + record from launch
    python main.py --no-config          # ignore config.json
"""

import argparse
import os
import time

import cv2

from camera_stream import CameraStream
from hand_tracker import HandTracker
from gestures import is_prayer_pose, get_frame_corners
from gestures import ClapDetector, WaveDetector
from filters import FILTERS, FILTER_NAMES
from compositor import Smoother, composite_quad, composite_full
from ball_manager import BallManager
from config import load as load_config, save as save_config
from hud import draw_hud
from recorder import Recorder

MODE_NONE = "none"
MODE_FRAME = "frame"
MODE_FULL = "full"

WINDOW_NAME = "Hand Filter Window"


def make_filtered(frame, fn, scale=0.5):
    """Run a filter, but at `scale` resolution then upscale back - the
    filtered area is small in quad/orb mode so this is a big FPS win for
    barely any visible difference."""
    h, w = frame.shape[:2]
    if scale < 1.0:
        small = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        small = fn(small)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn(frame)


def _mode_label(manager, mode):
    if manager.any_active():
        return "orbs (%s)" % manager.summary()
    return mode


def screenshot(out):
    name = time.strftime("hand_filter_shot_%Y%m%d_%H%M%S.png")
    cv2.imwrite(name, out)
    print("saved", name)
    return name


def main():
    parser = argparse.ArgumentParser(description="Hand-gesture filter window")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--track-width", type=int, default=None,
                        help="downscale feeding the hand-tracking model (320; lower = faster)")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie-mirror flip")
    parser.add_argument("--ss", type=int, default=None,
                        help="orb supersampling factor (2; 1 = off, higher = prettier)")
    parser.add_argument("--record", action="store_true",
                        help="start recording hand_filter_recording.mp4 from launch")
    parser.add_argument("--record-out", type=str, default=None,
                        help="recording filename (default hand_filter_recording.mp4)")
    parser.add_argument("--no-config", action="store_true", help="ignore config.json")
    args = parser.parse_args()

    cfg = load_config() if not args.no_config else {}
    if args.no_config:
        from config import DEFAULTS
        cfg = dict(DEFAULTS)
        cfg.pop("record_path", None)

    width = args.width or cfg.get("width", 1280)
    height = args.height or cfg.get("height", 720)
    track_width = args.track_width or cfg.get("track_width", 320)
    ss = args.ss or cfg.get("ss", 2)
    mirror = not args.no_mirror and bool(cfg.get("mirror", True))
    record_path = args.record_out or cfg.get("record_path", "hand_filter_recording.mp4")

    stream = CameraStream(src=args.camera if args.camera is not None else 0,
                          width=width, height=height).start()
    if not stream.is_opened():
        print("Could not open the webcam. Try a different --camera value.")
        return

    tracker = HandTracker(max_hands=2, track_width=track_width)

    filter_idx = max(0, min(int(cfg.get("filter", 0)), len(FILTER_NAMES) - 1))
    show_landmarks = False
    show_hud = bool(cfg.get("hud", True))
    onboarded = False

    corner_smoother = Smoother(alpha=0.4)
    manager = BallManager(ss=ss)
    manager.set_theme(int(cfg.get("theme", 0)))

    clap = ClapDetector()
    wave = WaveDetector()

    recording = args.record
    recorder = Recorder()
    if recording:
        started = recorder.start(width, height, record_path)
        if started:
            print("recording -> %s [%s]" % (started, recorder.codec))
        else:
            print("recording could not be started:", record_path)
            recording = False

    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = min(max(now - prev_time, 1e-4), 0.1)
            prev_time = now

            hands = tracker.process(frame)
            filter_fn = FILTERS[FILTER_NAMES[filter_idx]]

            manager.update(hands, dt)
            if manager.any_active():
                onboarded = True

            # ---- gesture events (scale-free, detector cooldowns handle spam)
            if clap.update(hands, dt):
                manager.on_clap()
            flat_points = None
            for hand in hands:
                if hand["points"] is not None and wave.update(hand["points"], dt):
                    flat_points = hand["points"]
                    break
            if flat_points is not None:
                manager.on_wave(flat_points[0])

            mode = MODE_NONE
            corners = None
            if len(hands) == 2 and is_prayer_pose(hands):
                mode = MODE_FULL
            elif not manager.any_active():
                if len(hands) == 2:
                    corners = get_frame_corners(hands)
                    if corners is not None:
                        mode = MODE_FRAME

            needs_filter = mode in (MODE_FULL, MODE_FRAME) or manager.any_active()
            filtered = None
            if needs_filter:
                filtered = make_filtered(
                    frame, filter_fn, scale=1.0 if mode == MODE_FULL else 0.5
                )

            out = frame
            if mode == MODE_FULL and filtered is not None:
                out = composite_full(frame, filtered)
            elif mode == MODE_FRAME and filtered is not None:
                smoothed = corner_smoother.update(corners)
                out = composite_quad(frame, filtered, smoothed)
            if mode != MODE_FRAME:
                corner_smoother.reset()

            manager.render(out, filtered)

            # recording captures the clean visuals (no HUD / debug overlays)
            if recorder.is_recording:
                recorder.write(out)

            if show_landmarks:
                for hand in hands:
                    tracker.draw(out, hand["raw"])

            if dt > 0:
                fps_smooth = (
                    0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else (1.0 / dt)
                )

            draw_hud(
                out, manager, _mode_label(manager, mode),
                FILTER_NAMES[filter_idx], fps_smooth,
                show_hud, not onboarded, recorder.is_recording,
            )

            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                filter_idx = (filter_idx + 1) % len(FILTER_NAMES)
            elif key == ord("d"):
                filter_idx = (filter_idx - 1) % len(FILTER_NAMES)
            elif key == ord("c"):
                manager.cycle_theme()
            elif key == ord("l"):
                show_landmarks = not show_landmarks
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("s"):
                screenshot(out)
            elif key == ord("r"):
                if recorder.is_recording:
                    saved = recorder.stop()
                    print("saved", saved)
                else:
                    started = recorder.start(width, height, record_path)
                    if started:
                        print("recording -> %s [%s]" % (started, recorder.codec))
                    else:
                        print("recording could not be started:", record_path)

    finally:
        if recorder.is_recording:
            saved = recorder.stop()
            print("saved", saved)
        stream.stop()
        tracker.close()
        cv2.destroyAllWindows()
        if not args.no_config:
            try:
                save_config({
                    "filter": filter_idx,
                    "theme": manager.theme_idx(),
                    "mirror": mirror,
                    "ss": ss,
                    "width": width,
                    "height": height,
                    "track_width": track_width,
                    "record_path": record_path,
                    "hud": show_hud,
                })
            except Exception:
                pass


if __name__ == "__main__":
    main()