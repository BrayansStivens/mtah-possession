"""Ball features derived from the position.

Why this module exists: **the ball speed that comes in the file is NaN in 91.3% of the
frames in which the ball IS seen.** Measured over the full match. That is, the ball-speed
column is unusable and has to be derived from the position.

The same measurement on the player speeds: NaN in 62.8% of the cells. Anything that
depends on the speeds in the file has to tolerate that.
"""

import numpy as np

from ..constants import BALL_MAX_SPEED, PITCH


def speed(tracking):
    """(n,) ball speed in m/s, derived from the position difference.

    NaN where there is no ball at t or at t-1, so it is defined in 56.8% of the frames.
    It is clipped to BALL_MAX_SPEED: the automatic tracking produces impossible jumps when
    it confuses the ball with something else, and without clipping a single jump dominates
    any mean.
    """
    xy = np.asarray(tracking.ball_xy, dtype=np.float32) * np.asarray(PITCH, dtype=np.float32)
    out = np.full(len(xy), np.nan, dtype=np.float32)
    step = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1)) * 25.0
    out[1:] = step
    return np.clip(out, 0.0, BALL_MAX_SPEED)


def displacement(tracking, half_width=25):
    """(n,) distance in meters between the ball position half_width frames before and after.

    Complementary to speed: a ball that jitters in place has high instantaneous speed and
    zero net displacement, which is exactly what happens with the tracking of a ball resting
    on the ground between a player's legs.
    """
    xy = np.asarray(tracking.ball_xy, dtype=np.float32) * np.asarray(PITCH, dtype=np.float32)
    n = len(xy)
    index = np.arange(n)
    before = xy[np.maximum(0, index - half_width)]
    after = xy[np.minimum(n - 1, index + half_width)]
    return np.sqrt(((after - before) ** 2).sum(axis=1))
