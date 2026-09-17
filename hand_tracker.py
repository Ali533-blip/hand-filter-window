"""
hand_tracker.py

Thin wrapper around MediaPipe HandLandmarker (Tasks API). Converts the
normalized (0-1) landmark coordinates MediaPipe returns into pixel
coordinates for the current frame, which is what the gesture logic and
compositor actually want to work with.

Things that used to bite (kept for whoever hacks on this next):

- MediaPipe wants RGB, OpenCV gives you BGR. Skipping the cvtColor is the
  classic "it sort of tracks but gestures feel wrong" bug.
- `mp.solutions.hands` was removed in MediaPipe >= 0.10.14 / 1.x. This uses
  `mp.tasks.vision.HandLandmarker` with the bundled `hand_landmarker.task`
  model (downloaded into `models/` at setup time).
- Hand tracking is the single most expensive step in the loop, so the input
  frame is downscaled to `track_width` before inference here. Because
  MediaPipe returns *normalized* coordinates, the landmarks map straight
  back onto the full-resolution frame for free - no accuracy hit worth
  worrying about, ~3-4x faster.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

MODEL_PATH = "models/hand_landmarker.task"
DEFAULT_TRACK_WIDTH = 320


class HandTracker:
    def __init__(
        self,
        max_hands=2,
        detection_conf=0.6,
        tracking_conf=0.6,
        model_path=MODEL_PATH,
        track_width=DEFAULT_TRACK_WIDTH,
    ):
        self.track_width = track_width
        # VIDEO mode needs monotonically increasing timestamps
        self._ts = int(time.monotonic() * 1000)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Returns a list of dicts:
        {"label": "Left"/"Right", "points": [(x, y), ...21], "raw": landmarks}"""
        h, w = frame_bgr.shape[:2]

        if self.track_width and w > self.track_width:
            th = max(1, int(h * self.track_width / w))
            small = cv2.resize(
                frame_bgr, (self.track_width, th), interpolation=cv2.INTER_AREA
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            self._ts += 33
            results = self.landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts
            )
        finally:
            rgb.flags.writeable = True

        hands_out = []
        if results.hand_landmarks:
            for i, lms in enumerate(results.hand_landmarks):
                label = "Unknown"
                if results.handedness and i < len(results.handedness) and results.handedness[i]:
                    label = results.handedness[i][0].category_name
                points = [(int(l.x * w), int(l.y * h)) for l in lms]
                hands_out.append({"label": label, "points": points, "raw": lms})
        return hands_out

    def draw(self, frame, raw_landmarks):
        drawing_utils.draw_landmarks(
            frame,
            raw_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
        )

    def close(self):
        self.landmarker.close()