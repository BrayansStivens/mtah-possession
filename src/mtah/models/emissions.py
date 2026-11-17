"""From per-frame features to a cost matrix (n, 3) for the decoder.

Model v1: geometric, interpretable, untrained. Each weight comes from the measured AUC of
its signal, not from tuning the metric.

The structure is a product of two decisions, which is how the problem is built:

    P(H) = P(live) * P(home | live)
    P(A) = P(live) * P(away | live)
    P(D) = P(dead)

and the cost is -log P. Splitting it this way matters because the two halves rely on
different signals. **Live/dead** comes from the players' kinematics and does not need the
ball. **Ownership** comes mostly from ball-player geometry, which does need it. With the
ball absent in 41% of scored frames, mixing them in a single classifier throws away the
half that keeps working.

Where there is no ball, ownership is decided only by team shape, which is a weak signal
(AUC 0.8124) and therefore produces a probability close to 0.5. The decoder's transition
cost then carries the previous state forward, which is exactly what we want: riding out an
occlusion silently inside a possession is free in spellF1, whereas guessing can split the
possession in two.
"""

import numpy as np

from ..config import DEFAULT
from ..decode.viterbi import A, D, H, N_STATES


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def dead_logit(features, config=None):
    """(n,) logit of "the ball is dead". Sum of independent pieces of evidence."""
    config = config or DEFAULT

    activity = np.nan_to_num(features["activity_s"], nan=config["activity_center_ms"])
    logit = config["activity_weight"] * (
        (config["activity_center_ms"] - activity) / config["activity_scale_ms"])

    # Undefined where the ball is not seen anywhere in the window: there it contributes
    # nothing and is neutralized by setting it to its own center.
    ball_speed = np.nan_to_num(features["ball_speed_s"], nan=config["ball_speed_center_ms"])
    logit = logit + config["ball_speed_weight"] * (
        (config["ball_speed_center_ms"] - ball_speed) / config["ball_speed_scale_ms"])

    logit = logit + config["ball_unseen_logit"] * (~features["ball_seen"]).astype(np.float32)
    logit = logit + config["ball_out_logit"] * features["ball_out"].astype(np.float32)
    return logit


def dead_probability(features, config=None):
    return _sigmoid(dead_logit(features, config))


def home_logit(features, config=None):
    """(n,) logit of "the home team has the ball", given that it is live.

    Two terms: the ball-player control, which is the strong one but only exists where there
    is a ball, and the team-shape asymmetry, which is weak but always exists.
    """
    config = config or DEFAULT

    # NaN where there is no ball anywhere in the window -> the term is zeroed out.
    control = features["control_diff_s"]
    logit = np.where(np.isnan(control), 0.0,
                     config["control_weight"] * (np.nan_to_num(control) - config["control_center"]))

    shape = np.nan_to_num(features["spread_ratio_s"], nan=config["shape_center"])
    logit = logit + config["shape_weight"] * (shape - config["shape_center"])
    return logit.astype(np.float64)


def home_probability(features, config=None):
    return _sigmoid(home_logit(features, config))


def team_tag_logit(team_tags, config=None):
    """(n,) what Smart Tagging adds to the home logit: weight * (+1 home, -1 away, 0).

    It goes here and not inside `home_logit` on purpose: `home_logit` is an input to the
    boundary detector, which was trained without tagging (the IDSSE does not have it).
    Adding it there would change what the detector sees. See training/team_tags.py for the
    weight.
    """
    config = config or DEFAULT
    team_tags = np.asarray(team_tags)
    if team_tags.ndim != 1 or not np.isin(team_tags, (-1, 0, 1)).all():
        raise ValueError("team_tags must be (n,) with values in {-1, 0, +1}")
    return config["smart_tagging_team_weight"] * team_tags.astype(np.float64)


def costs_from_logits(home, dead, config=None):
    """(n, 3) cost per frame and state from the two logits. Columns H, A, D."""
    config = config or DEFAULT
    floor = float(config["prob_floor"])

    dead = np.clip(_sigmoid(dead), floor, 1.0 - floor)
    home = np.clip(_sigmoid(home), floor, 1.0 - floor)
    alive = 1.0 - dead

    out = np.empty((len(dead), N_STATES), dtype=np.float64)
    out[:, H] = -np.log(alive * home)
    out[:, A] = -np.log(alive * (1.0 - home))
    out[:, D] = -np.log(dead)
    return out


def team_classifier_logit(team_logit, config=None):
    """(n,) what the team classifier adds to the home logit: weight * its logit.

    Kept out of `home_logit` for the same reason as Smart Tagging: the boundary detector was
    trained without it. See training/team_classifier.py for the weight.
    """
    config = config or DEFAULT
    team_logit = np.asarray(team_logit, dtype=np.float64)
    if team_logit.ndim != 1 or not np.isfinite(team_logit).all():
        raise ValueError("team_logit must be (n,) and finite")
    return config["team_classifier_weight"] * team_logit


def costs(features, config=None, team_tags=None, team_logit=None):
    """(n, 3) cost per frame and state, lower is better. Columns in order H, A, D.

    `team_tags=None` means "no Smart Tagging" (the IDSSE); the submission always passes it.
    `team_logit=None` means "no team classifier" (models/team_classifier.py).
    """
    config = config or DEFAULT
    home = home_logit(features, config)
    if team_logit is not None:
        if len(team_logit) != len(home):
            raise ValueError("team_logit has %d frames and the features %d"
                             % (len(team_logit), len(home)))
        home = home + team_classifier_logit(team_logit, config)
    if team_tags is not None:
        if len(team_tags) != len(home):
            raise ValueError("team_tags has %d frames and the features %d"
                             % (len(team_tags), len(home)))
        home = home + team_tag_logit(team_tags, config)
    return costs_from_logits(home, dead_logit(features, config), config)
