"""The trained boundary detector, as the submission uses it.

A `HistGradientBoostingClassifier` trained on the 7 degraded IDSSE matches
(`python -m mtah.training.boundaries export`). It returns, per frame, a
`transition_boundary_score` in [0, 1] that the decoder uses as a discount on the cost of
switching state. Measured with nested leave-one-match-out: +1.17 pp of spellF1 over the
model without the detector (STATUS.md, Nov 7).

Training data: IDSSE, Bassek et al., Sci Data 12, 195 (2025),
doi:10.1038/s41597-025-04505-y, CC-BY 4.0 with authorization from the DFL. The model is a
derivative work: citing and crediting the DFL is a condition of the license.

**Why it fails instead of carrying on.** A scikit-learn pickle is only reliable with the
version that wrote it; if the features change order the model scores garbage without
complaining; and if the emission parameters change (two of its signals are the logits) the
detector sees a different distribution from the one it learned. All three are checked on
load.
"""

import functools
import hashlib
import json
import os

from ..features.boundaries import feature_matrix

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "boundary_detector.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "boundary_detector.json")

# Parameters that only affect decoding: changing them does not change what the detector
# sees, so they do not invalidate the model. All the others in config.py do.
DECODER_ONLY_KEYS = frozenset({
    "switch_turnover", "switch_to_dead", "switch_from_dead", "min_duration_frames",
    "prob_floor", "boundary_discount_weight", "boundary_switch_scale",
    "smart_tagging_team_weight", "team_classifier_weight",
})


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _as_json(value):
    """Normalizes tuples to lists to compare with what is saved in JSON."""
    return json.loads(json.dumps(value))


def feature_config(config):
    """The config.py parameters that determine the detector's features."""
    return {key: _as_json(value) for key, value in sorted(config.items())
            if key not in DECODER_ONLY_KEYS}


@functools.lru_cache(maxsize=1)
def load():
    """(model, metadata), checking integrity and scikit-learn version."""
    import joblib
    import sklearn

    for path in (ARTIFACT_PATH, METADATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "{} is missing. Generate it with `python -m mtah.training.boundaries export`, "
                "or set boundary_discount_weight = 0 in config.py to decode without the detector."
                .format(path))
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if _sha256(ARTIFACT_PATH) != metadata["artifact_sha256"]:
        raise ValueError("{} does not match the sha256 in {}: the model has been modified "
                         "or is not the one it describes".format(ARTIFACT_PATH, METADATA_PATH))
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError("the detector was saved with scikit-learn {} and {} is installed here: "
                         "a pickle from another version is not reliable. Install "
                         "requirements.txt.".format(
                             metadata["sklearn_version"], sklearn.__version__))
    return joblib.load(ARTIFACT_PATH), metadata


def score(features, config):
    """(n,) transition_boundary_score per frame."""
    model, metadata = load()
    if feature_config(config) != metadata["feature_config"]:
        changed = sorted(key for key, value in feature_config(config).items()
                         if metadata["feature_config"].get(key) != value)
        raise ValueError("the detector was trained with other feature/emission parameters "
                         "({}); retrain it with `python -m mtah.training.boundaries export`"
                         .format(", ".join(changed) or "different keys"))
    matrix, names = feature_matrix(features, config)
    if names != metadata["feature_names"]:
        raise ValueError("the feature columns are not the ones used in training; "
                         "retrain the detector")
    return model.predict_proba(matrix)[:, 1]


def switch_discount(features, config):
    """(n,) per-frame discount for `viterbi.decode(..., switch_discount=...)`."""
    return config["boundary_discount_weight"] * score(features, config)


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
