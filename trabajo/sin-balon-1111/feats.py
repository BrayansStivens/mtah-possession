"""New signals for the ball-absent stretches. Experiment, not submission code.

Three families, all per frame and then smoothed with centered windows:

- `team`: shape and movement of each team **in its own attack direction**. The
  `centroid_gap_x` of features/geometry.py was discarded (AUC 0.53) because it is not oriented:
  teams switch ends at half-time and the sign cancels out. Here each team is measured on an
  axis pointing at the opponent's goal, so "the block moves up" means the same in both
  halves.
- `dead`: collective kinematics and gap context (where the ball was when it was lost and
  where it reappears) to decide live or dead ball.
- `ballteam`: whose ball it was when it was lost and whose it is when it reappears.

The team matrix is built per **perspective**: the home team's and the away team's are the
same function with the roles swapped. Training on both and subtracting the two outputs
gives an antisymmetric logit: it cannot learn "the home team usually has the ball".
"""

import numpy as np

from mtah.config import DEFAULT
from mtah.features import smoothing
from mtah.models import emissions

LENGTH, WIDTH = 105.3, 68.0
HALF = LENGTH / 2.0
TEAM_WINDOWS = (12, 37, 100)
DEAD_WINDOWS = (12, 50, 125)
VEL_LAG = 5                     # centered difference at +-5 frames (0.4 s)
VEL_MAX = 12.0                  # m/s; above this it is an identity swap of the slot
TEAM_SIGNALS = ("cen", "len", "wid", "dfl", "atl", "tide", "spd", "opp", "spr")
SUM_SIGNALS = ("cen", "tide")
CLIP_FRAMES = 500


# --------------------------------------------------------------------------- attack direction

def centroid_gap(tracking):
    """(n,) away centroid minus home centroid in x, meters. >0: the home team defends x=0."""
    x = tracking.players_xy[:, :, 0].astype(np.float64) * LENGTH
    with np.errstate(invalid="ignore"):
        home = np.nanmean(x[:, tracking.team_of_slot == 0], axis=1)
        away = np.nanmean(x[:, tracking.team_of_slot == 1], axis=1)
    return away - home


def attack_direction(tracking, blocks=None):
    """(n,) +1 where the home team attacks toward +x, -1 where it attacks toward -x; and a report.

    No labels and no metadata: the two teams are arranged on the pitch (each has its
    goalkeeper and its defense behind it), so the sign of the centroid difference says
    which team defends which goal. `blocks` are the halves if known (GRADED_BLOCKS in the
    ATD); otherwise the single sign change (the half-time break) is searched for.
    """
    sign = np.sign(np.nan_to_num(centroid_gap(tracking)))
    n = len(sign)
    if blocks is None:
        cum = np.cumsum(sign)
        split = int(np.argmax(np.abs(2 * cum - cum[-1]))) + 1
        blocks = ((0, split), (split, n))
    direction = np.ones(n, dtype=np.float64)
    report = []
    previous = 1.0
    for a, b in blocks:
        mean = float(sign[a:b].mean()) if b > a else 0.0
        value = previous if mean == 0 else (1.0 if mean > 0 else -1.0)
        direction[a:b] = value
        report.append({"block": [int(a), int(b)], "direction": value, "agreement": abs(mean)})
        previous = value
    # Frames outside the blocks (half-time, pre-match): the direction of the nearest block.
    if blocks[0][0] > 0:
        direction[:blocks[0][0]] = direction[blocks[0][0]]
    for (_, b), (a2, _) in zip(blocks[:-1], blocks[1:]):
        if a2 > b:
            direction[b:a2] = direction[a2]
    return direction, report


# --------------------------------------------------------------------------- per frame

def _velocity(xy_m):
    """(n, slots, 2) velocity per slot in m/s, NaN where the slot is not the same object."""
    v = np.full(xy_m.shape, np.nan, dtype=np.float32)
    v[VEL_LAG:-VEL_LAG] = (xy_m[2 * VEL_LAG:] - xy_m[:-2 * VEL_LAG]) * (25.0 / (2 * VEL_LAG))
    with np.errstate(invalid="ignore"):
        bad = np.hypot(v[:, :, 0], v[:, :, 1]) > VEL_MAX
    v[bad] = np.nan
    return v


def _second(sorted_values, count, from_end):
    """Second-lowest (or second-highest) value of each already sorted row, NaNs at the end."""
    n = len(sorted_values)
    index = np.clip(count - 2, 0, sorted_values.shape[1] - 1) if from_end else np.ones(n, int)
    out = sorted_values[np.arange(n), index]
    return np.where(count >= 3, out, np.nan)


def team_raw(tracking, direction):
    """{team: {signal: (n,)}} with each team on its own attack axis."""
    xy = tracking.players_xy.astype(np.float32) * np.array([LENGTH, WIDTH], dtype=np.float32)
    velocity = _velocity(xy)
    speed = tracking.players_speed
    out = {}
    for team, own in ((0, direction), (1, -direction)):
        slots = np.flatnonzero(tracking.team_of_slot == team)
        own = own.astype(np.float32)[:, None]
        u = own * (xy[:, slots, 0] - HALF)             # >0: opponent's half
        y = xy[:, slots, 1]
        count = np.sum(~np.isnan(u), axis=1)
        ordered = np.sort(u, axis=1)
        enough = count >= 3
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = {
                "cen": np.nanmean(u, axis=1), "len": np.nanstd(u, axis=1),
                "wid": np.nanstd(y, axis=1),
                "dfl": _second(ordered, count, from_end=False),
                "atl": _second(ordered, count, from_end=True),
                "tide": np.nanmean(own * velocity[:, slots, 0], axis=1),
                "spd": np.nanmean(speed[:, slots], axis=1),
                "opp": np.nansum(u > 0, axis=1) / np.maximum(count, 1),
                "spr": np.nansum(speed[:, slots] > 5.0, axis=1) / np.maximum(count, 1),
            }
        out[team] = {k: np.where(enough, v, np.nan).astype(np.float32) for k, v in sig.items()}
    return out


def team_smoothed(tracking, direction):
    """{(team, signal, window): (n,) float32}."""
    raw = team_raw(tracking, direction)
    return {(team, name, w): smoothing.box_nanmean(raw[team][name], w)
            for team in (0, 1) for name in TEAM_SIGNALS for w in TEAM_WINDOWS}


def team_names():
    names = []
    for w in TEAM_WINDOWS:
        names += ["p_%s_%d" % (s, w) for s in TEAM_SIGNALS]
        names += ["q_%s_%d" % (s, w) for s in TEAM_SIGNALS]
        names += ["d_%s_%d" % (s, w) for s in TEAM_SIGNALS]
        names += ["s_%s_%d" % (s, w) for s in SUM_SIGNALS]
    return names


def team_matrix(smoothed, perspective):
    """(n, f) team matrix seen from `perspective` (0 home, 1 away)."""
    p, q = perspective, 1 - perspective
    cols = []
    for w in TEAM_WINDOWS:
        cols += [smoothed[(p, s, w)] for s in TEAM_SIGNALS]
        cols += [smoothed[(q, s, w)] for s in TEAM_SIGNALS]
        cols += [smoothed[(p, s, w)] - smoothed[(q, s, w)] for s in TEAM_SIGNALS]
        cols += [smoothed[(p, s, w)] + smoothed[(q, s, w)] for s in SUM_SIGNALS]
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------- gap context

def _fill(values, seen, forward):
    """Carries the last (or the next) ball-seen value across the gaps."""
    n = len(values)
    index = np.where(seen, np.arange(n), -1 if forward else n)
    if forward:
        index = np.maximum.accumulate(index)
        ok = index >= 0
    else:
        index = np.minimum.accumulate(index[::-1])[::-1]
        ok = index < n
    out = np.full(n, np.nan, dtype=np.float32)
    out[ok] = values[index[ok]]
    distance = np.where(ok, np.abs(np.arange(n) - index), CLIP_FRAMES)
    return out, np.minimum(distance, CLIP_FRAMES).astype(np.float32)


def gap_context(tracking, features):
    """Dict of gap signals: last and next seen ball, and how long ago/until."""
    seen = tracking.ball_seen
    bx = tracking.ball_xy[:, 0].astype(np.float64) * LENGTH
    by = tracking.ball_xy[:, 1].astype(np.float64) * WIDTH
    edge = np.minimum(np.minimum(bx, LENGTH - bx), np.minimum(by, WIDTH - by))
    edge_prev, since = _fill(edge, seen, True)
    edge_next, until = _fill(edge, seen, False)
    x_prev, _ = _fill(bx, seen, True)
    x_next, _ = _fill(bx, seen, False)
    y_prev, _ = _fill(by, seen, True)
    y_next, _ = _fill(by, seen, False)
    control = (features["control_home"] - features["control_away"]).astype(np.float64)
    control = np.where(seen, control, np.nan)
    control_short = smoothing.box_nanmean(control, 6)
    ctl_prev, _ = _fill(control_short, seen & ~np.isnan(control_short), True)
    ctl_next, _ = _fill(control_short, seen & ~np.isnan(control_short), False)
    speed_prev, _ = _fill(np.nan_to_num(tracking.ball_speed, nan=0.0), seen, True)
    return {
        "since": since, "until": until, "gap": np.minimum(since + until, 2 * CLIP_FRAMES),
        "edge_prev": edge_prev, "edge_next": edge_next,
        "jump": np.hypot(x_next - x_prev, y_next - y_prev).astype(np.float32),
        "speed_prev": speed_prev, "ctl_prev": ctl_prev, "ctl_next": ctl_next,
    }


def dead_matrix(tracking, features, config=DEFAULT):
    """(n, f) matrix for live/dead ball and its names."""
    xy = tracking.players_xy.astype(np.float32) * np.array([LENGTH, WIDTH], dtype=np.float32)
    speed = tracking.players_speed
    velocity = _velocity(xy)
    count = np.sum(~np.isnan(speed), axis=1)
    ordered = np.sort(speed, axis=1)
    rows = np.arange(len(count))
    x, y = xy[:, :, 0], xy[:, :, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = {
            "act": np.nanmean(speed, axis=1),
            "spd_p25": np.where(count > 3, ordered[rows, np.clip(count // 4, 0, 45)], np.nan),
            "spd_p75": np.where(count > 3, ordered[rows, np.clip(3 * count // 4, 0, 45)], np.nan),
            "slow": np.nansum(speed < 1.0, axis=1) / np.maximum(count, 1),
            "fast": np.nansum(speed > 4.0, axis=1) / np.maximum(count, 1),
            "all_len": np.nanstd(x, axis=1), "all_wid": np.nanstd(y, axis=1),
            "drift": np.hypot(np.nanmean(velocity[:, :, 0], axis=1),
                              np.nanmean(velocity[:, :, 1], axis=1)),
            "touch": np.nansum((y < 2.0) | (y > WIDTH - 2.0), axis=1).astype(np.float32),
            "box": np.maximum(
                np.nansum((x < 16.5) & (np.abs(y - WIDTH / 2) < 20.16), axis=1),
                np.nansum((x > LENGTH - 16.5) & (np.abs(y - WIDTH / 2) < 20.16), axis=1)
            ).astype(np.float32),
            "n": count.astype(np.float32),
        }
    names, cols = [], []
    for name, values in raw.items():
        for w in DEAD_WINDOWS:
            names.append("%s_%d" % (name, w))
            cols.append(smoothing.box_nanmean(values.astype(np.float32), w))
    seen = tracking.ball_seen
    ball_speed = np.where(seen, tracking.ball_speed, np.nan)
    for w in (6, 25, 75):
        names.append("ball_speed_%d" % w)
        cols.append(smoothing.box_nanmean(ball_speed, w))
        names.append("ball_seen_%d" % w)
        cols.append(smoothing.box_mean(seen.astype(np.float32), w))
    names.append("ball_out_12")
    cols.append(smoothing.box_mean(features["ball_out"].astype(np.float32), 12))
    context = gap_context(tracking, features)
    for name in ("since", "until", "gap", "edge_prev", "edge_next", "jump", "speed_prev"):
        names.append(name)
        cols.append(context[name])
    names.append("hand_dead_logit")
    cols.append(emissions.dead_logit(features, config))
    return np.column_stack(cols).astype(np.float32), names


def ballteam_matrix(features, context, config=DEFAULT):
    """(n, f) ball-dependent team signals, seen by the home team, and names.

    All of them change sign when the perspective changes, except those marked as even.
    """
    control = (features["control_home"] - features["control_away"]).astype(np.float64)
    control = np.where(features["ball_seen"], control, np.nan)
    near = (features["nearest_away"] - features["nearest_home"]).astype(np.float64)
    names, cols, odd = [], [], []
    for w in (6, 25, 50, 125):
        names.append("ctl_%d" % w); cols.append(smoothing.box_nanmean(control, w)); odd.append(1)
        names.append("near_%d" % w); cols.append(smoothing.box_nanmean(near, w)); odd.append(1)
    for name in ("ctl_prev", "ctl_next"):
        names.append(name); cols.append(context[name]); odd.append(1)
    for name in ("since", "until"):
        names.append(name); cols.append(context[name]); odd.append(0)
    names.append("hand_home_logit"); cols.append(emissions.home_logit(features, config)); odd.append(1)
    return np.column_stack(cols).astype(np.float32), names, np.array(odd, dtype=bool)
