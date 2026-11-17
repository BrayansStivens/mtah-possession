"""Feature assembly: the one place that decides what goes into the model.

Two families, kept apart on purpose because they fail at different moments:

- **Ball-dependent** (control, distance to the nearest player, ball speed).
  The most informative and the least available: the ball is missing in 43% of frames.
- **Ball-independent** (player activity, team shape). These are the ones that have to carry
  the 41% of scored frames without a ball.

AUC measured on this match against the public ground truth:

    detect dead ball             smoothed activity +-50f   0.8452
                                 smoothed ball speed +-25f 0.7986
                                 ball not visible          0.7700
                                 n tracked objects         0.5987  <- discarded
    tell the teams apart (live)  smoothed control +-50f    0.9091
                                 spread_ratio_x            0.8124  <- without using the ball
                                 centroid x gap            0.5294  <- discarded
"""

import numpy as np

from . import ball, geometry, smoothing


def build_all(tracking, config):
    """Dict of arrays of length n_frames, ready for models.emissions."""
    features = {
        "ball_seen": tracking.ball_seen,
        "ball_x": tracking.ball_xy[:, 0],
        "ball_y": tracking.ball_xy[:, 1],
        "ball_out": geometry.ball_out_of_pitch(tracking),
        "ball_gap": geometry.ball_gap_lengths(tracking),
    }

    # --- live / dead, without depending on the ball ---
    activity = geometry.player_activity(tracking)
    features["activity"] = activity
    features["activity_s"] = smoothing.box_mean(activity, config["activity_smooth"])

    # The ball speed in the file is NaN in 91.3% of the frames where the ball is seen, so
    # it is derived from the position. box_nanmean, not box_mean: where there is no ball
    # the value is unknown, not zero.
    ball_speed = ball.speed(tracking)
    features["ball_speed"] = ball_speed
    features["ball_speed_s"] = smoothing.box_nanmean(ball_speed, config["ball_speed_smooth"])
    features["ball_displacement"] = ball.displacement(tracking, config["ball_speed_smooth"])

    # --- team ownership ---
    distances = geometry.ball_distances(tracking)
    control = geometry.control_scores(distances, tracking.team_of_slot, config["control_tau_m"])
    nearest = geometry.nearest_per_team(distances, tracking.team_of_slot)
    features["control_home"], features["control_away"] = control[:, 0], control[:, 1]
    features["nearest_home"], features["nearest_away"] = nearest[:, 0], nearest[:, 1]

    # Undefined where there is no ball, so that the smoothing carries over the neighbours
    # instead of diluting towards "nobody controls".
    difference = np.where(tracking.ball_seen, control[:, 0] - control[:, 1], np.nan)
    features["control_diff_s"] = smoothing.box_nanmean(difference, config["control_smooth"])

    shape = geometry.team_shape(tracking)
    features.update(shape)
    # box_nanmean: the ratio is NaN where it means nothing (a team with fewer than 3
    # players seen), and there we have to average the defined neighbours, not fill in.
    features["spread_ratio_s"] = smoothing.box_nanmean(shape["spread_ratio_x"],
                                                       config["shape_smooth"])
    return features
