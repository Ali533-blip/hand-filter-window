"""
compositor.py

Turns "here's a shape" + "here's a filtered frame" into the final composited
image: original frame outside the shape, filtered frame inside it, with a
soft feathered edge (via Gaussian blur on the mask) so the boundary doesn't
look like a hard cutout.

Smoother is a small exponential-moving-average helper. Raw landmark
positions jitter frame to frame even when your hand is still; without
smoothing, the window edges visibly shake. Lower alpha = smoother but more
laggy; higher alpha = snappier but jitterier. 0.3-0.4 felt like the sweet
spot.

Performance notes:
- Only the mask's bounding box is blended, so a small fist window costs a
  tiny fraction of a full-frame pass.
- The blend uses uint8 cv2.multiply/add ops (which saturate correctly)
  instead of float32 array gymnastics over the whole frame.
"""

import cv2
import numpy as np


class Smoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        new_value = np.array(new_value, dtype=np.float32)
        if self.value is None or self.value.shape != new_value.shape:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def _feathered_patch(shape_hw, draw_fn, feather):
    """Draw the shape into a mask, then return (x0, y0, patch) where patch
    is the feathered mask cropped to the shape's bounding box (+ feather
    margin). Cropping first keeps the Gaussian blur on a tiny array instead
    of a full-frame pass - that alone was a ~20ms tax."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_fn(mask)
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw <= 0 or bh <= 0:
        return None
    m = int(np.ceil(3.0 * feather)) if feather > 0 else 0
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + bw + m, w), min(y + bh + m, h)
    patch = mask[y0:y1, x0:x1].copy()
    if feather > 0:
        patch = cv2.GaussianBlur(patch, (0, 0), feather)
    return x0, y0, patch


def _blend_patch(out, filtered, patch_info):
    """Blend `filtered` into `out` inside the given patch (in place)."""
    x0, y0, patch = patch_info
    bh, bw = patch.shape
    m3 = cv2.merge([patch, patch, patch])
    sub = out[y0:y0 + bh, x0:x0 + bw]
    base = cv2.multiply(sub, 255 - m3, scale=1.0 / 255.0, dtype=cv2.CV_8U)
    fill = cv2.multiply(filtered[y0:y0 + bh, x0:x0 + bw], m3, scale=1.0 / 255.0, dtype=cv2.CV_8U)
    sub[:, :] = cv2.add(base, fill)


def composite_quad(frame, filtered, corners, feather=15):
    h, w = frame.shape[:2]
    corners_i = corners.astype(np.int32)
    patch = _feathered_patch(
        (h, w), lambda m: cv2.fillConvexPoly(m, corners_i, 255), feather
    )
    if patch is not None:
        _blend_patch(frame, filtered, patch)
    cv2.polylines(frame, [corners_i], True, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def composite_circle(frame, filtered, center, radius, feather=12):
    h, w = frame.shape[:2]
    c = (int(center[0]), int(center[1]))
    r = max(int(radius), 1)
    patch = _feathered_patch((h, w), lambda m: cv2.circle(m, c, r, 255, -1), feather)
    if patch is not None:
        _blend_patch(frame, filtered, patch)
    cv2.circle(frame, c, r, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def composite_full(frame, filtered):
    return filtered