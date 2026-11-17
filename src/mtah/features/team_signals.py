"""Shape and movement of each team **in its own attacking direction**. They do not look at the ball.

These are the signals of the team classifier (`models/team_classifier.py`), designed for the
41% of scored frames in which the ATD has no ball: there the hand-built emissions only
carry over the last thing they saw, and who has possession has to be read from the players.

**Why oriented.** `features/geometry.py` already had a centroid difference
(`centroid_gap_x`) and it was dropped (AUC 0.53): the teams switch ends at halftime and
the sign cancels out. Here each team is measured on an axis that points at the opponent's
goal, so "the block pushes up" means the same thing in both halves. The attacking direction
comes from the tracking itself, with no labels or metadata (`attack_direction`).

**Why only six.** The Nov 11-12 experiment measured nine
(`analysis/no-ball-1111/SUMMARY.md`). The centroid, the block length and the fraction of
players in the opponent's half are left out: they are the ones that best tell the challenge
match apart from the IDSSE ones (a domain classifier separates them with AUC 0.93; with
these six, 0.72) and dropping them costs almost nothing (team AUC 0.9518 -> 0.9506).

The matrix is built per **perspective**: the home team's and the away team's are the same
function with the roles swapped. Training on both and subtracting the two outputs gives an
antisymmetric logit, which cannot learn "the home team tends to have the ball".
"""

import numpy as np

from ..constants import FPS, PITCH
from . import geometry, smoothing

# Per team, on its attacking axis (x > 0 is the opponent's half):
#   wid   spread across the width (standard deviation of y)
#   dfl   defensive line: second-deepest player (the deepest is the goalkeeper)
#   atl   attacking line: second-most advanced player
#   tide  mean speed toward the opponent's goal
#   spd   mean speed
#   spr   fraction of players above SPRINT_SPEED_MS
# The order is that of the saved model's columns: it cannot be changed without retraining.
SIGNALS = ("wid", "dfl", "atl", "tide", "spd", "spr")

# Signals with their own sign (forward or backward): besides the difference between
# teams, the sum is passed, which says whether the two blocks move toward the same goal.
SUM_SIGNALS = ("tide",)

# Half-widths of the centered smoothing: 1, 3 and 8 s windows.
WINDOWS_FRAMES = (12, 37, 100)

# Velocity by centered difference at +-5 frames (0.4 s). Above 12 m/s it is not a player
# running: it is the tracker swapping the identity of the slot.
VELOCITY_LAG_FRAMES = 5
MAX_SPEED_MS = 12.0
SPRINT_SPEED_MS = 5.0

HALF_LENGTH_M = PITCH[0] / 2.0


# --------------------------------------------------------------------------- attack direction

def centroid_gap(tracking):
    """(n,) away centroid minus home centroid in x, meters. >0: the home team defends x=0."""
    x = tracking.players_xy[:, :, 0].astype(np.float64) * PITCH[0]
    with np.errstate(invalid="ignore"):
        home = np.nanmean(x[:, tracking.team_of_slot == geometry.HOME_CODE], axis=1)
        away = np.nanmean(x[:, tracking.team_of_slot == geometry.AWAY_CODE], axis=1)
    return away - home


def attack_direction(tracking, blocks=None):
    """(n,) +1 where the home team attacks toward +x, -1 toward -x; and a per-half report.

    The two teams are ordered on the pitch (each has its goalkeeper and its defense behind
    it), so the sign of the centroid difference says who defends which goal. `blocks` are
    the halves if they are known (GRADED_BLOCKS in the challenge match); otherwise the
    single sign change is searched for, which is halftime. `agreement` in the report is
    |mean of the sign| in the half: close to 1 if the direction is clear.
    """
    sign = np.sign(np.nan_to_num(centroid_gap(tracking)))
    n = len(sign)
    if blocks is None:
        cumulative = np.cumsum(sign)
        split = int(np.argmax(np.abs(2 * cumulative - cumulative[-1]))) + 1
        blocks = ((0, split), (split, n))
    direction = np.ones(n, dtype=np.float64)
    report = []
    previous = 1.0
    for start, end in blocks:
        mean = float(sign[start:end].mean()) if end > start else 0.0
        value = previous if mean == 0 else (1.0 if mean > 0 else -1.0)
        direction[start:end] = value
        report.append({"block": [int(start), int(end)], "direction": value,
                       "agreement": abs(mean)})
        previous = value
    # Frames outside the halves (pre-match, halftime): the direction of the half that follows.
    if blocks[0][0] > 0:
        direction[:blocks[0][0]] = direction[blocks[0][0]]
    for (_, end), (next_start, _) in zip(blocks[:-1], blocks[1:]):
        if next_start > end:
            direction[end:next_start] = direction[next_start]
    return direction, report


# --------------------------------------------------------------------------- per frame

def _velocity(xy_m):
    """(n, slots, 2) velocity per slot in m/s, NaN where the slot is not the same object."""
    lag = VELOCITY_LAG_FRAMES
    velocity = np.full(xy_m.shape, np.nan, dtype=np.float32)
    velocity[lag:-lag] = (xy_m[2 * lag:] - xy_m[:-2 * lag]) * (FPS / (2.0 * lag))
    with np.errstate(invalid="ignore"):
        jump = np.hypot(velocity[:, :, 0], velocity[:, :, 1]) > MAX_SPEED_MS
    velocity[jump] = np.nan
    return velocity


def _second(ordered, count, from_end):
    """Second-lowest (or second-highest) value of each already-sorted row, with NaNs at the end."""
    n = len(ordered)
    index = np.clip(count - 2, 0, ordered.shape[1] - 1) if from_end else np.ones(n, int)
    return ordered[np.arange(n), index]


def raw(tracking, direction):
    """{team: {signal: (n,) float32}} unsmoothed. NaN if fewer than 3 players are seen."""
    xy = geometry.to_metres(tracking.players_xy)
    velocity = _velocity(xy)
    speed = tracking.players_speed
    out = {}
    for team, own in ((geometry.HOME_CODE, direction), (geometry.AWAY_CODE, -direction)):
        slots = np.flatnonzero(tracking.team_of_slot == team)
        own = own.astype(np.float32)[:, None]
        forward = own * (xy[:, slots, 0] - HALF_LENGTH_M)        # >0: opponent's half
        count = np.sum(~np.isnan(forward), axis=1)
        ordered = np.sort(forward, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            signals = {
                "wid": np.nanstd(xy[:, slots, 1], axis=1),
                "dfl": _second(ordered, count, from_end=False),
                "atl": _second(ordered, count, from_end=True),
                "tide": np.nanmean(own * velocity[:, slots, 0], axis=1),
                "spd": np.nanmean(speed[:, slots], axis=1),
                "spr": (np.nansum(speed[:, slots] > SPRINT_SPEED_MS, axis=1)
                        / np.maximum(count, 1)),
            }
        enough = count >= geometry.MIN_TRACKED_FOR_SHAPE
        out[team] = {name: np.where(enough, signals[name], np.nan).astype(np.float32)
                     for name in SIGNALS}
    return out


def smoothed(tracking, blocks=None):
    """(2, signals, windows, n) float32: each signal of each team, smoothed and centered.

    `blocks` is passed as is to `attack_direction`.
    """
    direction, _ = attack_direction(tracking, blocks)
    signals = raw(tracking, direction)
    return np.stack([[[smoothing.box_nanmean(signals[team][name], window)
                       for window in WINDOWS_FRAMES] for name in SIGNALS]
                     for team in (geometry.HOME_CODE, geometry.AWAY_CODE)]).astype(np.float32)


# --------------------------------------------------------------------------- matrix

def feature_names():
    """Names of the columns of `matrix`, in order."""
    names = []
    for window in WINDOWS_FRAMES:
        names += ["own_%s_%d" % (name, window) for name in SIGNALS]
        names += ["rival_%s_%d" % (name, window) for name in SIGNALS]
        names += ["diff_%s_%d" % (name, window) for name in SIGNALS]
        names += ["sum_%s_%d" % (name, window) for name in SUM_SIGNALS]
    return names


def matrix(stack, perspective):
    """(n, columns) what `perspective` sees (0 home, 1 away): own, rival and difference."""
    own, rival = perspective, 1 - perspective
    sums = [SIGNALS.index(name) for name in SUM_SIGNALS]
    columns = []
    for window in range(stack.shape[2]):
        columns += [stack[own, :, window].T, stack[rival, :, window].T,
                    (stack[own, :, window] - stack[rival, :, window]).T,
                    (stack[own, sums, window] + stack[rival, sums, window]).T]
    return np.concatenate(columns, axis=1)
