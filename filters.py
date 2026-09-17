"""
filters.py

The actual visual effects that show up inside your hand-window. Cycle
through them at runtime with 'f' (next) / 'd' (previous).

Each function takes a BGR frame and returns a BGR frame of the same shape -
that keeps them interchangeable and easy to add to.
"""

import cv2
import numpy as np


def apply_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def apply_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


def apply_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_negative(frame):
    return 255 - frame


def apply_pop_color(frame):
    """Saturation + contrast boost - makes the windowed area 'pop' against
    a normal background."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.6, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(boosted, alpha=1.15, beta=8)


FILTERS = {
    "thermal": apply_thermal,
    "cartoon": apply_cartoon,
    "edges": apply_edges,
    "negative": apply_negative,
    "pop-color": apply_pop_color,
}
FILTER_NAMES = list(FILTERS.keys())
