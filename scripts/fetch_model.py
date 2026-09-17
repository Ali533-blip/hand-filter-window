#!/usr/bin/env python
"""
Downloads the MediaPipe hand-landmark model that hand_tracker.py needs
(the app will not start without models/hand_landmarker.task).

Usage:
    python scripts/fetch_model.py

The model is gitignored because it is a ~8 MB binary asset downloaded from
Google; fetch it once after cloning.
"""

import os
import sys
import urllib.request

URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "hand_landmarker.task",
)


def main():
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    if os.path.exists(TARGET) and os.path.getsize(TARGET) > 0:
        print("Model already present:", TARGET)
        return 0
    print("Downloading hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(URL, TARGET)
    except Exception as exc:  # pragma: no cover - network failures vary
        print("Download failed:", exc)
        return 1
    print("Saved to", TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())