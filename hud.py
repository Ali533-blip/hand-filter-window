"""
hud.py

The on-screen chrome: a translucent info bar up top (mode / colour / filter /
FPS / recording dot) and an onboarding hint that fades away once the user
has summoned their first orb. 'h' toggles the whole thing.

Drawn as a small alpha-blended panel so the camera stays visible behind it.
"""

import cv2

PANEL_H = 78
PAD = 10


def _text(out, text, y, scale, color=(220, 220, 230), thick=1):
    cv2.putText(out, text, (PAD, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(out, text, (PAD, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(out, manager, mode, filter_name, fps, show_hud, onboard, recording):
    h, w = out.shape[:2]
    if not show_hud:
        return

    panel = out[:PANEL_H, :w].copy()
    cv2.rectangle(panel, (0, 0), (w, PANEL_H), (16, 18, 28), -1)
    out[:PANEL_H, :w] = cv2.addWeighted(panel, 0.62, out[:PANEL_H, :w], 0.38, 0)

    rec_suffix = "  [REC]" if recording else ""
    bar1 = (f"mode: {mode}   color: {manager.theme_name}   filter: {filter_name}"
            f"   fps: {fps:0.0f}{rec_suffix}")
    bar2 = ("window: 2x(index+thumb) | full: prayer | orbs: fist | throw: flick"
            " | stretch: 2 hands | clap: +wave: wind")
    bar3 = ("pinch: open destroys | point: jump | gun: bolt | thumbs-up: pump"
            " | victory: cheer | color: middle+thumb / c   f filter  s shot  r record  h hud  q quit")

    _text(out, bar1, 26, 0.58)
    _text(out, bar2, 48, 0.42, color=(170, 175, 190))
    _text(out, bar3, 66, 0.40, color=(150, 155, 175))

    if onboard and not any(o.is_active() for o in manager.orbs):
        hint = ("First time? Make a fist to summon an orb. Pinch it, flick it to throw,"
                " tap middle+thumb to change colour.")
        (thw, thh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_h = thh + 24
        box_w = thw + 28
        x0 = max(8, (w - box_w) // 2)
        y0 = h - 34 - box_h
        region = out[y0:y0 + box_h, x0:x0 + box_w].copy()
        cv2.rectangle(region, (0, 0), (box_w, box_h), (16, 18, 28), -1)
        out[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(
            region, 0.72, out[y0:y0 + box_h, x0:x0 + box_w], 0.28, 0
        )
        _text(out, hint, y0 + thh + 6, 0.5, color=(120, 200, 255))
        cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (60, 120, 200), 1)