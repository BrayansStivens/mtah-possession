"""Per-frame homography: normalized pitch (0-1) -> image pixel.

One row per video frame, `frame` counting from 0, so for the tactical video the
index matches the submission row. Columns 1-9 are the 3x3 matrix in row order.

What it is really for: projecting the players onto the video frame and checking with your own
eyes what fooled the model. It is a debugging tool, not a strong feature --
measured on this match, `reprojection_error` separates "camera pointing at the pitch" from
"camera on the crowd", but it does NOT separate live play from dead ball (median 0.231 in play vs
0.395 at half-time, p90 in play 0.308).
"""

import numpy as np
import pandas as pd

from ..constants import GAME
from ..paths import require


def load(source="tactical", game=GAME):
    """(H, reprojection_error, valid) with H of shape (n, 3, 3)."""
    path = require(["homography"], source=source, game=game)[0]
    frame = pd.read_csv(path)
    expected = ["frame", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
                "reprojection_error", "valid"]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))
    if not (frame["frame"].values == np.arange(len(frame))).all():
        raise ValueError("{}: the frame column is not 0..n-1 in order".format(path))

    matrices = frame[expected[1:10]].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return (matrices,
            frame["reprojection_error"].to_numpy(dtype=np.float64),
            frame["valid"].to_numpy(dtype=bool))


def project(matrix, x, y):
    """Point(s) of the normalized pitch -> pixel. Accepts scalars or arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    px = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
    py = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    return px, py


def unproject(matrix, px, py):
    """Pixel -> point of the normalized pitch. Inverse of project()."""
    inverse = np.linalg.inv(matrix)
    return project(inverse, px, py)
