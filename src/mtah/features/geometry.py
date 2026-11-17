"""Per-frame geometric features, in meters and vectorized over the whole match.

Everything is computed in meters, not in normalized coordinates: the pitch is 105.3 x 68 m
and comparing distances without scaling the two axes stretches every distance along the
short axis and picks the wrong player near the touchlines.

Two families, on purpose:

- **Ball-dependent** (distance to the nearest player, control, pressure). They are the most
  informative and the least reliable: the ball is missing in 43% of frames, and
  disproportionately in the ones that matter -- crowded boxes, aerial duels, tackles.
- **Ball-independent** (player activity, team shape, centroids). They are the ones that
  have to carry the 41% of scored frames without a ball, and the reason this module
  computes team shape even when there is no ball to see.
"""

import numpy as np

from ..constants import PITCH

HOME_CODE, AWAY_CODE = 0, 1
SCALE = np.asarray(PITCH, dtype=np.float32)

# Minimums below which team shape means nothing. A team with fewer than 3 players seen has
# no "shape", and an x spread below 1 m does not happen in a real match: it is the tracker
# seeing two people together.
MIN_TRACKED_FOR_SHAPE = 3
MIN_SPREAD_M = 1.0


def to_metres(xy):
    """Normalized 0-1 coordinates -> meters, scaling each axis."""
    return np.asarray(xy, dtype=np.float32) * SCALE


def ball_distances(tracking):
    """(n, slots) distance in meters from each slot to the ball. NaN where either is missing."""
    ball = to_metres(tracking.ball_xy)[:, None, :]          # (n, 1, 2)
    players = to_metres(tracking.players_xy)                # (n, slots, 2)
    delta = players - ball
    return np.sqrt(np.einsum("nsc,nsc->ns", delta, delta))


def nearest_per_team(distances, team_of_slot):
    """(n, 2) distance to the nearest player of each team. NaN if the team has nobody."""
    out = np.full((distances.shape[0], 2), np.nan, dtype=np.float32)
    for team in (HOME_CODE, AWAY_CODE):
        slots = np.flatnonzero(team_of_slot == team)
        subset = distances[:, slots]
        empty = np.isnan(subset).all(axis=1)
        with np.errstate(invalid="ignore"):
            nearest = np.nanmin(np.where(np.isnan(subset), np.inf, subset), axis=1)
        nearest[empty] = np.nan
        out[:, team] = nearest
    return out


def control_scores(distances, team_of_slot, tau=3.0):
    """(n, 2) influence of each team on the ball: sum of exp(-d / tau).

    An exponential kernel instead of "the nearest wins" because the nearest is the decision
    of a single player over tracking with gaps and identity switches: if that slot is lost
    for one frame, the label jumps to the other team. Summing over all the players makes
    the signal collective, which is the only thing this dataset supports -- there are no
    player identifiers, only 46 slots that appear and disappear.

    `tau`, in meters, sets how much being close counts. 3 m is a reasonable control radius.
    """
    weights = np.exp(-distances / float(tau))
    weights = np.where(np.isnan(distances), 0.0, weights)
    out = np.zeros((distances.shape[0], 2), dtype=np.float32)
    for team in (HOME_CODE, AWAY_CODE):
        out[:, team] = weights[:, np.flatnonzero(team_of_slot == team)].sum(axis=1)
    return out


def count_within(distances, team_of_slot, radius):
    """(n, 2) players of each team within `radius` meters of the ball."""
    close = np.where(np.isnan(distances), False, distances < float(radius))
    out = np.zeros((distances.shape[0], 2), dtype=np.int16)
    for team in (HOME_CODE, AWAY_CODE):
        out[:, team] = close[:, np.flatnonzero(team_of_slot == team)].sum(axis=1)
    return out


def player_activity(tracking):
    """(n,) mean speed of the visible players, in m/s.

    The best candidate for a dead-ball detector in this data that does not depend on the
    ball: with play stopped people walk or stand still, and with the ball live they run.
    The mean is used rather than the sum so it does not depend on how many slots the
    tracker sees in that frame -- it counts between 20 and 23, and that varies by itself.
    """
    speed = tracking.players_speed
    with np.errstate(invalid="ignore"):
        return np.nanmean(speed, axis=1).astype(np.float32)


def team_shape(tracking):
    """Dict of team-shape features, all independent of the ball.

    Why they matter: in the 41% of scored frames without a ball they are all that is left.
    The defending team compresses and drops back; the attacking one stretches and moves up.
    The asymmetry between the two shapes carries information about who has the ball even
    when the ball is not seen.
    """
    xy = to_metres(tracking.players_xy)
    out = {}
    for team, name in ((HOME_CODE, "home"), (AWAY_CODE, "away")):
        slots = np.flatnonzero(tracking.team_of_slot == team)
        subset = xy[:, slots, :]
        with np.errstate(invalid="ignore"):
            out[name + "_centroid_x"] = np.nanmean(subset[:, :, 0], axis=1).astype(np.float32)
            out[name + "_centroid_y"] = np.nanmean(subset[:, :, 1], axis=1).astype(np.float32)
            out[name + "_spread_x"] = np.nanstd(subset[:, :, 0], axis=1).astype(np.float32)
            out[name + "_spread_y"] = np.nanstd(subset[:, :, 1], axis=1).astype(np.float32)
            out[name + "_n_tracked"] = np.sum(~np.isnan(subset[:, :, 0]), axis=1).astype(np.int16)
    out["centroid_gap_x"] = out["home_centroid_x"] - out["away_centroid_x"]

    # NaN, not a number, when the ratio means nothing. If the tracker sees fewer than two
    # players of a team in that frame, the standard deviation is 0 and the ratio collapses
    # to 0 or to infinity, giving a very strong and unfounded opinion about who has the
    # ball. Marking it undefined makes the shape term neutralise itself instead of lying.
    degenerate = ((out["home_n_tracked"] < MIN_TRACKED_FOR_SHAPE)
                  | (out["away_n_tracked"] < MIN_TRACKED_FOR_SHAPE)
                  | (out["home_spread_x"] < MIN_SPREAD_M)
                  | (out["away_spread_x"] < MIN_SPREAD_M))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = out["home_spread_x"] / out["away_spread_x"]
    out["spread_ratio_x"] = np.where(degenerate, np.nan, ratio).astype(np.float32)
    out["shape_degenerate"] = degenerate
    return out


def ball_out_of_pitch(tracking, margin=0.01):
    """(n,) True where the tracked ball falls outside the pitch -> the ball is dead.

    A hard, cheap signal, but with limited coverage: when the ball goes out, the tracker
    usually loses it instead of following it outside, so this flags fewer frames than it
    should. It does not replace a detector, it anchors it.
    """
    xy = tracking.ball_xy
    outside = (xy[:, 0] < -margin) | (xy[:, 0] > 1 + margin) | \
              (xy[:, 1] < -margin) | (xy[:, 1] > 1 + margin)
    return np.where(np.isnan(xy[:, 0]), False, outside)


def ball_gap_lengths(tracking):
    """(n,) length of the ball-absent run that each frame belongs to.

    A gap of 3 frames is an occlusion and gets interpolated; one of 400 means the ball has
    gone 15 s without being seen and any position inferred there is fiction. Telling them
    apart lets us trust the fill only where we can.
    """
    seen = tracking.ball_seen
    out = np.zeros(len(seen), dtype=np.int32)
    start = None
    for i, visible in enumerate(seen):
        if not visible and start is None:
            start = i
        elif visible and start is not None:
            out[start:i] = i - start
            start = None
    if start is not None:
        out[start:] = len(seen) - start
    return out


def build(tracking, tau=3.0, radii=(2.0, 5.0)):
    """All the features in a dict of arrays of length n_frames."""
    distances = ball_distances(tracking)
    features = {
        "ball_seen": tracking.ball_seen,
        "ball_speed": tracking.ball_speed,
        "ball_x": tracking.ball_xy[:, 0],
        "ball_y": tracking.ball_xy[:, 1],
        "ball_out": ball_out_of_pitch(tracking),
        "ball_gap": ball_gap_lengths(tracking),
        "activity": player_activity(tracking),
    }
    nearest = nearest_per_team(distances, tracking.team_of_slot)
    control = control_scores(distances, tracking.team_of_slot, tau)
    features["nearest_home"], features["nearest_away"] = nearest[:, 0], nearest[:, 1]
    features["control_home"], features["control_away"] = control[:, 0], control[:, 1]
    for radius in radii:
        counts = count_within(distances, tracking.team_of_slot, radius)
        features["n_home_{:g}m".format(radius)] = counts[:, 0]
        features["n_away_{:g}m".format(radius)] = counts[:, 1]
    features.update(team_shape(tracking))
    return features
