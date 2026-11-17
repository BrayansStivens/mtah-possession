"""Path resolution for the match assets.

The bundle is organized by recording, the same way the organizers deliver it:

    <root>/tactical/BAR-ATH_tactical_atd.txt
    <root>/broadcast/BAR-ATH_broadcast_atd.txt

The root is taken from MTAH_DATA_DIR if it exists; otherwise from ./data in the repo root.
Nothing in this repo writes a file name by hand outside this module.
"""

import os

from .constants import GAME

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}

ASSETS = {
    "tactical": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    "broadcast": ("video", "atd_meta", "atd_raw", "homography"),
}


def data_dir():
    return os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO_ROOT, "data")


def cache_dir():
    path = os.path.join(REPO_ROOT, "out", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def asset_paths(source="tactical", game=GAME, root=None):
    """Dict asset -> path. Does not check existence; that is what require() is for."""
    if source not in ASSETS:
        raise KeyError("unknown source {!r}; available: {}".format(source, sorted(ASSETS)))
    folder = os.path.join(root or data_dir(), source)
    stem = "{}_{}".format(game, source)
    return {name: os.path.join(folder, stem + SUFFIXES[name]) for name in ASSETS[source]}


def require(names, source="tactical", game=GAME, root=None):
    """Returns the requested paths, or fails with a message saying what to download."""
    paths = asset_paths(source, game, root)
    missing = [(n, paths[n]) for n in names if not os.path.exists(paths[n])]
    if missing:
        detail = "\n".join("  missing {:<14} -> {}".format(n, p) for n, p in missing)
        raise FileNotFoundError(
            "Match assets not found in {}\n{}\n"
            "Download them from the competition Drive and unzip keeping the\n"
            "structure <root>/tactical/..., or point MTAH_DATA_DIR to where you have them."
            .format(data_dir(), detail))
    return [paths[n] for n in names]


def available(source="tactical", game=GAME, root=None):
    """Dict asset -> bool, to know what can already be worked with."""
    paths = asset_paths(source, game, root)
    return {name: os.path.exists(path) for name, path in paths.items()}
