"""*Centered* smoothing of features. Never of labels.

The distinction matters and it is the central lesson of the challenge. Smoothing **labels**
with a window is what the official baselines do, and to avoid fragmenting a long
possession they have to erase the short ones. Smoothing **features** before a decoder that
penalises transitions does not have that problem: the signal gets cleaned and the number
of segments is still controlled by the decoder.

Everything here is centered and therefore zero-latency. Measured by degrading the public
ground truth: +1 frame of delay is free (spellF1 1.0000), but +25 frames (1 s) drops to
0.7765 and +50 (2 s) to 0.5882. A causal filter is not an option in this problem.

Measured on this match, smoothing raises the AUC of each signal:

    player activity, dead-ball detection:         0.8336 raw -> 0.8452 with +-50f
    control difference, telling the teams apart:  0.8213 raw -> 0.9091 with +-50f
"""

import numpy as np


def box_mean(values, half_width, fill=None):
    """Mean over a centered window of +-half_width frames. Cost independent of the width.

    NaNs are filled beforehand with `fill` (the median by default) because a cumulative
    sum does not tolerate NaN: a single NaN poisons everything that comes after it.
    """
    values = np.asarray(values, dtype=np.float64)
    if half_width <= 0:
        return values.astype(np.float32)
    if np.isnan(values).any():
        replacement = fill if fill is not None else np.nanmedian(values)
        values = np.nan_to_num(values, nan=float(replacement))

    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return ((cumulative[high] - cumulative[low]) / (high - low)).astype(np.float32)


def box_nanmean(values, half_width):
    """Centered mean over the frames of the window **where the value is defined**.

    Different from box_mean with filling, and the difference matters for the signals that
    depend on the ball: where the ball is not seen, the control is not 0, it is unknown.
    Filling with 0 dilutes the signal towards "nobody controls"; averaging only the defined
    frames carries the neighbours' information across the gap, which is what we want when
    the ball is missing in 43% of frames.

    Returns NaN where the whole window is undefined.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = ~np.isnan(values)
    if half_width <= 0:
        return values.astype(np.float32)

    filled = np.where(defined, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0.0], np.cumsum(defined.astype(np.float64))))

    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    total = cumulative[high] - cumulative[low]
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan).astype(np.float32)
