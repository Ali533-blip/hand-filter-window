<div align="center">

# Hand Filter Window

**Turn your hands into a live, interactive camera filter — pixels included.**

Track your hands with a webcam, shape a filter window with your fingertips,
and summon glowing "filter orbs" that you can carry, throw, stretch, merge,
and even shoot at each other — all without touching a keyboard.

</div>

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![CI](https://github.com/Ali533-blip/hand-filter-window/actions/workflows/ci.yml/badge.svg)](https://github.com/Ali533-blip/hand-filter-window/actions)

---

## What it does

- **Frame window** — hold up both hands and the tips of your thumbs and
  index fingers become the 4 corners of a live filter quad that follows you.
- **Full screen** — press your palms together (prayer pose) and the filter
  fills the whole frame.
- **Filter orbs** — make a fist and a glowing orb made of the current filter
  forms in your hand; make a fist with *each* hand and you get two.
- **Orb physics** — pinch to shrink, flick to throw, catch it with the other
  hand, stretch it like taffy between two hands, or **merge** two orbs into
  one bigger twin orb.
- **Express yourself** — point to make the orb hop to your fingertip, give a
  thumbs-up to pump it up, throw a 🔫 "gun" bolt across the screen, wave for
  sparkles, or clap for a celebration burst.
- **Five colour themes** — amber, cyan, rose, emerald, indigo, cycled live.
- **Record your output** — press `r` to capture the clean visuals to an MP4.

All gesture thresholds are relative to hand size in the frame, so everything
keeps working whether you are close to the camera or further back.

## Built with

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — 21-point hand tracking
- [OpenCV](https://opencv.org/) — filters, compositing, supersampled rendering
- [NumPy](https://numpy.org/) — landmark math
- Plain Python — no framework, just a tight camera loop

---

## Installation

### Prerequisites

- Python 3.9+ and a webcam
- ~150 MB free disk (Python packages + the model file)

### 1. Clone & set up a virtual environment

```bash
git clone https://github.com/Ali533-blip/hand-filter-window.git
cd hand-filter-window

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the hand-tracking model

The `hand_landmarker.task` model is downloaded at set-up (it's a ~8 MB
binary and is gitignored):

```bash
python scripts/fetch_model.py
```

Or download it manually from
[MediaPipe models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
into the `models/` folder.

### 4. Run it

```bash
python main.py
```

---

## Controls

| Key       | Action |
|-----------|--------|
| `f` / `d` | next / previous filter |
| `c`       | cycle the orb's colour theme |
| `s`       | save a PNG screenshot |
| `r`       | start/stop recording to `hand_filter_recording.mp4` (clean visuals, no HUD) |
| `h`       | toggle the on-screen HUD |
| `l`       | toggle the hand-landmark debug overlay |
| `q` / Esc | quit (settings are saved to `config.json`) |

### Command line

```
--camera N          camera index (default 0)
--width W           capture width (default 1280)
--height H          capture height (default 720)
--track-width T     hand-tracking resolution, lower = faster (default 320)
--ss N              orb supersampling factor (default 1; 2 = crisper edges)
--record            start recording from launch
--record-out PATH   custom recording filename
--no-mirror         disable the selfie-mirror flip
--no-config         ignore config.json
```

---

## Gesture guide

| Pose | What happens |
|------|--------------|
| **Two hands, thumbs + index up** | 4 corners → the filter window |
| **Prayer pose** (palms together) | Filter covers the whole frame |
| **Fist** (one hand) | An orb forms in that hand |
| **Fist** (both hands) | One orb per hand |
| **Fists together** | Two orbs merge into a bigger twin orb |
| **Pinch** (thumb + index) | Orb squeezes flat; open the hand to destroy it |
| **Flick** (quick wrist move) | Orb is thrown; the other hand catches it |
| **Near a held orb** (second hand) | The orb stretches between both hands |
| **Middle finger + thumb touch** | Cycles the colour theme |
| **Point** (index up, thumb tucked) | Orb hops to your fingertip |
| **Gun** (index up, thumb out) | Orb bolts across the screen |
| **Thumbs-up** | Orb pulses up to 1.5× size |
| **Victory** (index + middle up) | Orb showers a celebration burst |
| **Wave** (flat hand, rocking) | Sparkle wind streams from your hand |
| **Clap** (two palms together fast) | Global celebration across all orbs |

## Filters included

`thermal`, `cartoon`, `edges`, `negative`, `pop-color`. Adding a new one is
a 5-line job: drop a `frame -> frame` function into `filters.py` and register
it in the `FILTERS` dict.

---

## Testing

The test suite runs **headless** — every test uses the synthetic hand
generator in `tests/fake_hand.py`, so no webcam, display, or model is needed.

```bash
python tests/run_all.py
```

28 checks across 4 suites cover pose detection, the orb state machine, the
new gesture actions, and dual-orb merge/absorb behaviour. CI runs this on
Python 3.9–3.12 for Linux and Windows on every push.

## Project layout

```
main.py             app loop / entry point
camera_stream.py    threaded webcam capture
hand_tracker.py     MediaPipe Hands wrapper
gestures.py         pose + gesture detection (fist, prayer, pinch, point, gun, ...)
filters.py          the visual effects
compositor.py       smoothing + mask compositing
ball.py             the orb: state machine + supersampled rendering
ball_manager.py     dual-orb orchestration (spawn / merge / absorb / events)
config.py           JSON persistence for user settings
hud.py              translucent HUD + onboarding hints
recorder.py         MP4/AVI output recording with codec fallback
tests/              synthetic-hand test suite (no camera required)
```

## Troubleshooting

- **Black window / no camera image** — try `--camera 1` (or 2); index 0 isn't
  always the right webcam.
- **Low FPS on an older machine** — drop `--width`/`--height` (e.g.
  `--width 640 --height 480`) and keep the default `--ss 1`.
- **Gestures feel unreliable** — ensure even lighting; MediaPipe's detector
  struggles in dim or heavily backlit rooms.
- **Recording won't open** — the recorder picks a supported codec
  automatically (`mp4v`/`avc1` MP4, or `MJPG`/`XVID` AVI). If it still won't
  open, try `--record-out test.avi`.

## License

Distributed under the [MIT License](LICENSE). Copyright (c) 2026 **Ali Zain**.

---

<p align="center">
  Made with <a href="https://github.com/Ali533-blip">Ali Zain</a>
</p>