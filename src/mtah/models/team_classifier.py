"""The team classifier without the ball, as the submission uses it.

A `HistGradientBoostingClassifier` trained on the 7 degraded IDSSE matches
(`python -m mtah.training.team_classifier export`). It looks only at the players
(`features/team_signals.py`: width, lines, speed toward the opponent's goal) and returns,
per frame, a "home team has possession" logit, which is added to the emissions' team logit
with `team_classifier_weight`. It exists for the frames without the ball: there the
hand-built emissions only carry over the last signal they saw.

The logit is **antisymmetric**: the model is trained on both perspectives (the home team's
and the away team's are the same matrix with the roles swapped) and half the difference
between the two outputs is used. Swapping the team names flips its sign, so it cannot
learn "the home team tends to have the ball".

Training data: IDSSE, Bassek et al., Sci Data 12, 195 (2025),
doi:10.1038/s41597-025-04505-y, CC-BY 4.0 with authorization from the DFL. The model is a
derivative work: citing and crediting the DFL is a condition of the license.

**Why it fails instead of carrying on.** Same as the boundary detector: a scikit-learn
pickle is only reliable with the version that wrote it, and if the signals change order or
definition the model scores garbage without complaining. Both are checked on load. Its
features do not depend on config.py, so there is nothing from config to check.
"""

import functools
import json
import os

import numpy as np

from ..features import team_signals
from .boundary_detector import _as_json, _sha256

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "team_classifier.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "team_classifier.json")


def feature_settings():
    """The constants of features/team_signals.py that determine what the model sees."""
    return _as_json({
        "signals": team_signals.SIGNALS,
        "sum_signals": team_signals.SUM_SIGNALS,
        "windows_frames": team_signals.WINDOWS_FRAMES,
        "velocity_lag_frames": team_signals.VELOCITY_LAG_FRAMES,
        "max_speed_ms": team_signals.MAX_SPEED_MS,
        "sprint_speed_ms": team_signals.SPRINT_SPEED_MS,
    })


@functools.lru_cache(maxsize=1)
def load():
    """(model, metadata), checking integrity and scikit-learn version."""
    import joblib
    import sklearn

    for path in (ARTIFACT_PATH, METADATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "{} is missing. Generate it with `python -m mtah.training.team_classifier export`, "
                "or set team_classifier_weight = 0 in config.py to decode without the "
                "classifier.".format(path))
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if _sha256(ARTIFACT_PATH) != metadata["artifact_sha256"]:
        raise ValueError("{} does not match the sha256 in {}: the model has been modified "
                         "or is not the one it describes".format(ARTIFACT_PATH, METADATA_PATH))
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError("the classifier was saved with scikit-learn {} and {} is installed "
                         "here: a pickle from another version is not reliable. Install "
                         "requirements.txt."
                         .format(metadata["sklearn_version"], sklearn.__version__))
    return joblib.load(ARTIFACT_PATH), metadata


def antisymmetric_logit(model, stack):
    """(n,) float32 home logit: mean of what each perspective says, with its sign."""
    home = model.decision_function(team_signals.matrix(stack, 0))
    away = model.decision_function(team_signals.matrix(stack, 1))
    return (0.5 * (home - away)).astype(np.float32)


def home_logit(tracking, blocks=None):
    """(n,) "home team has possession" logit from the teams' shape, without looking at the ball.

    `blocks` are the halves of the match if they are known (see `team_signals.attack_direction`).
    """
    model, metadata = load()
    if (team_signals.feature_names() != metadata["feature_names"]
            or feature_settings() != metadata["feature_settings"]):
        raise ValueError("the team signals are not the ones used in training; retrain the "
                         "classifier with `python -m mtah.training.team_classifier export`")
    return antisymmetric_logit(model, team_signals.smoothed(tracking, blocks))


def save(model, metadata):
    """Writes the model and its metadata (with the sha256 and the scikit-learn version)."""
    import joblib
    import sklearn

    joblib.dump(model, ARTIFACT_PATH, compress=3)
    metadata = dict(metadata, artifact_sha256=_sha256(ARTIFACT_PATH),
                    sklearn_version=sklearn.__version__)
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    load.cache_clear()
    return metadata
