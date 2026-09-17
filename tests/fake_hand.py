import numpy as np


def make_hand(cx, cy, scl, pose):
    """Synthetic 21-point hand. Wrist sits one `scl` below the MCP row, so
    hand_scale() == scl. All 21 landmarks are filled (like real MediaPipe
    output) so centroids behave realistically."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (cx, cy + scl)                      # wrist
    pts[1] = (cx, cy + scl * 0.85)               # thumb c-mc
    pts[2] = (cx - scl * 0.45, cy + scl * 0.75)  # thumb m-mc
    pts[3] = (cx - scl * 0.65, cy + scl * 0.35)  # thumb i-mc
    pts[4] = (cx - scl * 0.8, cy + scl * 0.15)   # thumb tip

    fingers = (5, 9, 13, 17)
    for mcp, dx in ((5, -1.0), (9, 0.0), (13, 1.0), (17, 2.0)):
        pts[mcp] = (cx + dx * scl * 0.6, cy)

    if pose == "pinch":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy - scl * 0.25)
            pts[mcp + 2] = (pts[mcp][0], cy - scl * 0.5)
            pts[mcp + 3] = (pts[mcp][0], cy - scl * 0.75)
        pts[4] = (pts[8][0], pts[8][1])          # thumb tip == index tip
        return pts

    if pose == "tap":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy - scl * 0.25)
            pts[mcp + 2] = (pts[mcp][0], cy - scl * 0.5)
            pts[mcp + 3] = (pts[mcp][0], cy - scl * 0.75)
        pts[4] = (pts[12][0], pts[12][1])        # thumb tip == middle tip
        return pts

    if pose == "open":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy - scl * 0.25)
            pts[mcp + 2] = (pts[mcp][0], cy - scl * 0.5)
            pts[mcp + 3] = (pts[mcp][0], cy - scl * 0.75)
        pts[4] = (cx - scl * 0.9, cy)
        return pts

    if pose == "flat":
        for mcp, dx in ((5, -0.6), (9, 0.0), (13, 0.6), (17, 1.2)):
            pts[mcp + 1] = (pts[mcp][0], cy - scl * 0.25)
            pts[mcp + 2] = (pts[mcp][0], cy - scl * 0.5)
            pts[mcp + 3] = (cx + dx * scl * 0.25, cy - scl * 0.95)
        pts[4] = (cx - scl * 0.9, cy)
        return pts

    if pose == "point":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy + scl * 0.05)
            pts[mcp + 2] = (pts[mcp][0], cy + scl * 0.1)
            pts[mcp + 3] = (pts[mcp][0], cy + scl * 0.15)
        pts[6] = (pts[5][0], cy - scl * 0.25)
        pts[7] = (pts[5][0], cy - scl * 0.5)
        pts[8] = (pts[5][0], cy - scl * 0.75)
        pts[4] = (cx - scl * 0.45, cy + scl * 0.35)
        return pts

    if pose == "gun":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy + scl * 0.05)
            pts[mcp + 2] = (pts[mcp][0], cy + scl * 0.1)
            pts[mcp + 3] = (pts[mcp][0], cy + scl * 0.15)
        pts[6] = (pts[5][0], cy - scl * 0.25)
        pts[7] = (pts[5][0], cy - scl * 0.5)
        pts[8] = (pts[5][0], cy - scl * 0.75)
        pts[4] = (cx - scl * 0.95, cy - scl * 0.02)
        return pts

    if pose == "victory":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy + scl * 0.05)
            pts[mcp + 2] = (pts[mcp][0], cy + scl * 0.1)
            pts[mcp + 3] = (pts[mcp][0], cy + scl * 0.15)
        for mcp in (5, 9):
            pts[mcp + 1] = (pts[mcp][0], cy - scl * 0.25)
            pts[mcp + 2] = (pts[mcp][0], cy - scl * 0.5)
            pts[mcp + 3] = (pts[mcp][0], cy - scl * 0.75)
        pts[4] = (cx - scl * 0.45, cy + scl * 0.35)
        return pts

    if pose == "thumbs":
        for mcp in fingers:
            pts[mcp + 1] = (pts[mcp][0], cy + scl * 0.05)
            pts[mcp + 2] = (pts[mcp][0], cy + scl * 0.1)
            pts[mcp + 3] = (pts[mcp][0], cy + scl * 0.15)
        pts[4] = (cx - scl * 0.9, cy - scl * 0.55)
        return pts

    # fist: everything curls back toward the palm / MCP row
    for mcp in fingers:
        pts[mcp + 1] = (pts[mcp][0], cy + scl * 0.05)
        pts[mcp + 2] = (pts[mcp][0], cy + scl * 0.1)
        pts[mcp + 3] = (pts[mcp][0], cy + scl * 0.15)
    return pts


def hand(cx, cy, scl, pose):
    return {"label": "L", "points": make_hand(cx, cy, scl, pose), "raw": None}


def frame(w, h):
    return np.full((h, w, 3), 40, dtype=np.uint8)