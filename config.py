"""
config.py

Tiny JSON persistence for user preferences so the app starts up the way
you left it (filter, colour theme, mirroring, supersampling, geometry).
Never crashes if the file is missing or corrupt - falls back to defaults.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "filter": 0,
    "theme": 0,
    "mirror": True,
    "ss": 1,
    "width": 1280,
    "height": 720,
    "track_width": 320,
    "record_path": "hand_filter_recording.mp4",
    "hud": True,
}


def load(path=CONFIG_PATH):
    cfg = dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            for k in DEFAULTS:
                if k in data:
                    cfg[k] = data[k]
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg, path=CONFIG_PATH):
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(cfg, fp, indent=2)
    except OSError:
        pass