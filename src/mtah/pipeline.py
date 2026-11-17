"""The full pipeline: tracking -> features -> emissions -> decoding -> submission.

A single place where everything is chained, so that `python -m mtah build` is literally the
command that regenerates the submitted file. The deliverable requires it: running this repo
against the data has to reproduce the CSV, not approximately but the very same file.
"""

import numpy as np

from . import submission
from .config import DEFAULT
from .constants import GRADED_BLOCKS
from .decode import viterbi
from . import features as feature_set
from .features import tagging
from .io import atd, smart_tagging
from .models import boundary_detector, emissions, team_classifier


def labels_from_tracking(tracking, config=None, team_tags=None):
    """Full-length labels from an already loaded Tracking object.

    `team_tags`: the Smart Tagging read with `features.tagging.team_tags`, or None without it.
    """
    config = config or DEFAULT

    features = feature_set.build_all(tracking, config)
    team_logit = None
    if config["team_classifier_weight"] > 0:
        team_logit = team_classifier.home_logit(tracking, blocks=GRADED_BLOCKS)
    cost = emissions.costs(features, config, team_tags=team_tags, team_logit=team_logit)
    transition = viterbi.switch_costs(turnover=config["switch_turnover"],
                                      to_dead=config["switch_to_dead"],
                                      from_dead=config["switch_from_dead"])
    discount = None
    if config["boundary_discount_weight"] > 0:
        transition = config["boundary_switch_scale"] * transition
        discount = boundary_detector.switch_discount(features, config)
    codes = viterbi.decode(cost, transition, switch_discount=discount)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"])
    return submission.build(viterbi.to_labels(codes))


def run(config=None, source="tactical", use_cache=True):
    """Loads the tracking and returns the submission labels."""
    tracking = atd.load(source=source, use_cache=use_cache)
    team_tags = tagging.team_tags(smart_tagging.load(source=source), tracking.n_frames)
    return labels_from_tracking(tracking, config, team_tags)
