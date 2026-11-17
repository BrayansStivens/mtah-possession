"""Boundary detector features: what changes between just before and just after.

The emissions look at windows of +-50 frames (and team shape +-200): they know *which*
team dominates a stretch, but not *where* it starts. This looks, for each frame, at the
mean of 13 signals over a short window after the frame and its difference from the one
before, at 0.24 / 0.48 / 1 / 2 s. It is consumed by `models/boundary_detector.py` in the
submission and by `training/boundaries.py` to train and measure; both have to see exactly
the same columns, which is why they live here and nowhere else.
"""

import numpy as np

from ..constants import FPS
from ..models import emissions

# Windows on each side of the frame, in frames: from 0.24 s to 2 s.
WINDOWS_FRAMES = (6, 12, 25, 50)
PRODUCT_WINDOWS_FRAMES = (25, 50)

SIGNAL_NAMES = ("control_diff", "nearest_gap_m", "activity", "ball_speed", "ball_seen",
                "ball_out", "ball_vx", "home_vx", "away_vx", "centroid_gap_x",
                "spread_ratio_x", "home_logit", "dead_logit")


def feature_names():
    """Names of the `feature_matrix` columns, in order. Does not depend on the data."""
    names = ["%s_%s_%d" % (signal, kind, width) for signal in SIGNAL_NAMES
             for width in WINDOWS_FRAMES for kind in ("change", "after")]
    names += ["both_vx_change_product_%d" % width for width in PRODUCT_WINDOWS_FRAMES]
    return names + ["frames_since_ball_seen", "frames_until_ball_seen", "ball_gap_frames"]


def window_means(values, width):
    """(before, after): mean ignoring NaN over [t-width, t) and over [t, t+width).

    NaN where the window has no defined value: a ball gap is missing data, not a zero, and
    the classifier (HistGradientBoosting) treats NaN as such.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(defined, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(defined)))
    n = len(values)
    frame = np.arange(n)
    lo, hi = np.maximum(frame - width, 0), np.minimum(frame + width, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        before = (total[frame] - total[lo]) / (count[frame] - count[lo])
        after = (total[hi] - total[frame]) / (count[hi] - count[frame])
    return before, after


def _frames_since(event):
    """(n,) frames since the last time `event` was True (n if never)."""
    n = len(event)
    last = np.maximum.accumulate(np.where(event, np.arange(n), -1))
    return np.where(last >= 0, np.arange(n) - last, n).astype(np.float64)


def frame_signals(features, config):
    """Per-frame signals, unsmoothed, from which the before/after changes are taken."""
    seen = features["ball_seen"]
    ball_x = np.where(seen, features["ball_x"], np.nan)
    return {
        "control_diff": np.where(seen, features["control_home"] - features["control_away"],
                                 np.nan),
        "nearest_gap_m": np.where(seen, features["nearest_home"] - features["nearest_away"],
                                  np.nan),
        "activity": features["activity"],
        "ball_speed": features["ball_speed"],
        "ball_seen": seen.astype(np.float64),
        "ball_out": features["ball_out"].astype(np.float64),
        "ball_vx": np.diff(ball_x, prepend=np.nan) * FPS,
        "home_vx": np.diff(features["home_centroid_x"], prepend=np.nan) * FPS,
        "away_vx": np.diff(features["away_centroid_x"], prepend=np.nan) * FPS,
        "centroid_gap_x": features["centroid_gap_x"],
        "spread_ratio_x": features["spread_ratio_x"],
        "home_logit": emissions.home_logit(features, config),
        "dead_logit": emissions.dead_logit(features, config),
    }


def feature_matrix(features, config):
    """(matrix (n, k) float32, names). Everything the detector sees at each frame."""
    signals = frame_signals(features, config)
    if tuple(signals) != SIGNAL_NAMES:
        raise ValueError("frame_signals and SIGNAL_NAMES do not match")
    columns = []
    for values in signals.values():
        for width in WINDOWS_FRAMES:
            before, after = window_means(values, width)
            columns += [after - before, after]
    # A turnover reverses the direction of both blocks at once: if the two centroids
    # change speed in opposite directions, the product comes out negative.
    for width in PRODUCT_WINDOWS_FRAMES:
        home = window_means(signals["home_vx"], width)
        away = window_means(signals["away_vx"], width)
        columns.append((home[1] - home[0]) * (away[1] - away[0]))
    seen = features["ball_seen"]
    columns += [_frames_since(seen), _frames_since(seen[::-1])[::-1],
                np.asarray(features["ball_gap"], dtype=np.float64)]
    return np.column_stack(columns).astype(np.float32), feature_names()
