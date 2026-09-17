"""
camera_stream.py

A tiny threaded webcam reader.

The naive way to grab frames is:

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        ...do hand tracking + filters on `frame`...

The problem: cap.read() and your processing happen on the SAME thread, back
to back. If hand tracking + filtering takes 40ms, you're capped well under
30 FPS even on a fast machine, because the camera sits idle while you crunch
numbers, and you sit idle while the camera driver fills its buffer.

CameraStream puts the capture loop on its own thread. It's always grabbing
the newest frame in the background; the main thread just asks for whatever
is freshest whenever it's ready. This is the single change that took this
project from ~12 FPS to a solid 30.
"""

import threading
import time

import cv2


class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # keep the OS-level buffer small so we don't fall behind on old frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                # camera hiccup - don't busy-loop
                time.sleep(0.005)

    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.stopped = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.cap.release()
