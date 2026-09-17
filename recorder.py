"""
recorder.py

Video recording for the app's composited output.

Behaviour
---------
- Always writes to the fixed file ``hand_filter_recording.mp4`` in the
  project folder, overwriting the previous take (nothing accumulates,
  you always know where the video is).
- Picks a codec/container the current OpenCV build actually supports:
  ``mp4v`` / ``avc1`` (.mp4) first, falling back to ``MJPG`` (.avi) and
  finally ``LAGS``-less stock ``XVID`` if needed.
- ``stop()`` finalizes the file and returns the path that was written.
  The codec that was actually used is available as ``.codec``.
"""

import os

import cv2

CODE_SHOTS = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("MJPG", ".avi"),
    ("XVID", ".avi"),
)

FALLBACK_PATH = "hand_filter_recording.avi"


class Recorder:
    def __init__(self):
        self._writer = None
        self.path = None
        self.codec = None

    @property
    def is_recording(self):
        return self._writer is not None

    def start(self, width, height, path="hand_filter_recording.mp4"):
        """Open the recorder at the given size. Tries each codec in order and
        uses the first one OpenCV accepts. Returns the actual path written."""
        if self.is_recording:
            return self.path
        for codec, ext in CODE_SHOTS:
            cand = os.path.splitext(path)[0] + ext if ext != ".mp4" else path
            w = cv2.VideoWriter(
                cand, cv2.VideoWriter_fourcc(*codec), 30, (int(width), int(height))
            )
            if w is not None and w.isOpened():
                self._writer = w
                self.path = cand
                self.codec = codec
                return cand
            if w is not None:
                w.release()
        # everything failed: last-ditch plain avi with implicit codec
        w = cv2.VideoWriter(FALLBACK_PATH, 0, 30, (int(width), int(height)))
        if w is not None and w.isOpened():
            self._writer = w
            self.path = FALLBACK_PATH
            self.codec = "default"
            return FALLBACK_PATH
        self._writer = None
        return None

    def write(self, frame):
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self):
        """Finalize the file. Returns the path written (or None)."""
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        path = self.path
        self.path = None
        return path