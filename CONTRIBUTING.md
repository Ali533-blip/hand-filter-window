# Contributing to Hand Filter Window

Thanks for wanting to help out! This project is small and approachable — a
few minutes of reading will get you up to speed.

## Project shape

The app is deliberately split into small, single-purpose modules:

| File | What it does |
|------|--------------|
| `main.py` | Entry point, live loop, key handling, config persistence |
| `camera_stream.py` | Threaded webcam capture (the big FPS win) |
| `hand_tracker.py` | MediaPipe Hands wrapper |
| `gestures.py` | All pose detectors + live `WaveDetector`/`ClapDetector` |
| `filters.py` | The visual effects (`frame -> frame` funcs) |
| `compositor.py` | Smoothing + mask compositing for the window quad |
| `ball.py` | The orb: state machine + supersampled rendering |
| `ball_manager.py` | Dual-orb orchestration (spawn / merge / absorb) |
| `config.py`, `hud.py`, `recorder.py` | Settings, HUD, video output |
| `tests/` | Synthetic-hand test suite (no webcam required) |

## Setting up a dev environment

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python scripts/fetch_model.py     # downloads the MediaPipe model
```

## Running the tests

```bash
python tests/run_all.py
```

Every test uses the synthetic hand generator in `tests/fake_hand.py`, so the
suite runs headless — no camera, no display, no model file. It must stay that
way. If a feature needs hardware, test the logic parts headless and describe
the manual check in the PR instead.

## Before you open a PR

1. **Run the full suite** — `python tests/run_all.py` and make sure every
   suite reports pass.
2. **Keep behavior relative to hand size.** Thresholds in `gestures.py` are
   scaled by `hand_scale()`. Please don't introduce fixed pixel distances.
3. **Match the style** — no type annotations beyond the plain builtins, no
   comments unless they explain *why*, 4-space indent.
4. **Performance matters.** The renderer runs per-frame; anything you add
   that uses `cv2.GaussianBlur` or big per-pixel work will be held to
   account (see `ball._soften` for the trick).

## Feature ideas worth picking up

- Audio cues (the codebase is currently visual-only by design).
- Blob/background-segmentation filters in `filters.py`.
- A `--camera` selector in the HUD.
- More orb themes (add to `THEMES` in `ball.py`).

Thanks again — PRs are reviewed as they come in.