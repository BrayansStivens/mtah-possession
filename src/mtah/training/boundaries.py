"""Possession boundary detector, trained on the IDSSE and measured with nested LOMO.

**What it tries to fix.** The model's error concentrates next to the changes: within +-12
frames of a real transition it gets one in every two frames wrong, and possessions shorter
than 5 s are recovered ~8%. The emissions are smoothed to +-50 frames, so they see *which*
team dominates a window but not *where* it starts. This detector looks at short windows on
each side of each frame (what changes between before and after) and returns a
`transition_boundary_score` in [0, 1]. The decoder uses it as a discount on the cost of
switching state (`viterbi.decode(..., switch_discount=...)`): where the detector sees a
boundary, cutting is cheaper.

Origin plan: `plan-fronteras-segmentos.md` (Day 1 and Day 2). The name is "score" and not
"probability" because nobody has validated its calibration, and the decoder does not treat
it as one.

**How it is measured, and why that way.** *Nested* leave-one-match-out over the 7 matches:
for each held-out match, the detector is trained on the other 6, and the discount weight is
also chosen on those 6 with out-of-sample scores (each one scored by a detector that did
not see it). The held-out match influences neither the model nor the weight. The three
degradation seeds of the same match are not independent matches: 1 and 2 are only used to
see whether the result holds up under another occlusion pattern, with the model and the
weight already fixed on seed 0.

No label of the challenge match enters here. The ATD only contributes the distribution of
occlusions (`harness.real_spec`, no labels).

Usage:  python -m mtah.training.boundaries      (needs requirements-train.txt + scikit-learn)
"""

import itertools
import json
import os
import time

import numpy as np

from .. import evaluate as ev
from .. import features as feature_set
from ..config import DEFAULT
from ..decode import viterbi
from ..features import geometry
from ..features.boundaries import WINDOWS_FRAMES, feature_matrix, feature_names
from ..models import boundary_detector, emissions
from ..paths import REPO_ROOT, cache_dir
from . import convention, degrade, harness, idsse

# A frame is positive if it falls within this distance of a label change. 12 frames (0.48 s)
# is the radius within which the error diagnosis measures half of the failures.
BOUNDARY_HALF_WIDTH_FRAMES = 12

# One in every N frames for training. Neighbors are nearly identical: a million correlated
# frames are not a million examples, and this way each fit takes seconds.
TRAIN_STRIDE_FRAMES = 4

# Decoder grid, chosen inside each fold. `switch_scale` multiplies the three switch costs
# of config.py: with the detector making boundaries cheaper, it may pay to make the rest
# more expensive. (1.0, 0.0) is the current model as is.
DISCOUNT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 0.95)
SWITCH_SCALES = (1.0, 1.25, 1.5, 2.0)

# Three arms with the same nested choice, to know where the improvement comes from. Making
# switches more expensive already helps a bit without the detector, and without this
# control that part would be attributed to it.
ARMS = {
    "scale_only": [(0.0, scale) for scale in SWITCH_SCALES],
    "discount_only": [(weight, 1.0) for weight in DISCOUNT_WEIGHTS],
    "both": list(itertools.product(DISCOUNT_WEIGHTS, SWITCH_SCALES)),
}

SEEDS = (0, 1, 2)

PREPARED_KEYS = ("cost", "matrix", "labels", "ball_seen", "home_logit", "dead_logit")

# Where the ball-absent gaps fall when degrading. Training and choosing always use `random`;
# the other two are stress tests for the held-out match (training/evidence.py).
GAP_PLACEMENTS = ("random", "crowded", "transitions")
CROWD_RADIUS_M = 5.0
TRANSITION_RADIUS_FRAMES = 12
GAP_WEIGHT_FLOOR = 0.02


def log(message):
    print(message, flush=True)


# --------------------------------------------------------------------------- target

def boundary_target(labels, half_width=BOUNDARY_HALF_WIDTH_FRAMES):
    """(n,) bool: True within `half_width` frames of a label change."""
    labels = np.asarray(labels)
    change_frames = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    near = np.zeros(len(labels) + 1, dtype=np.int32)
    np.add.at(near, np.clip(change_frames - half_width, 0, len(labels)), 1)
    np.add.at(near, np.clip(change_frames + half_width, 0, len(labels)), -1)
    return np.cumsum(near)[:-1] > 0


# --------------------------------------------------------------------------- data

def gap_weights(placement, clean, labels):
    """Per-frame weight to center a ball-absent gap there. None = at random, the usual one.

    `crowded` does not look at labels: it weights the number of players within
    CROWD_RADIUS_M of the ball (squared), which is how DATA.md describes where the ATD
    loses the ball. Worst case, `transitions`: centers the gaps next to the real possession
    changes.
    """
    if placement == "random":
        return None
    if placement == "crowded":
        with np.errstate(invalid="ignore"):
            crowd = (geometry.ball_distances(clean) < CROWD_RADIUS_M).sum(axis=1)
        return GAP_WEIGHT_FLOOR + crowd.astype(np.float64) ** 2
    if placement == "transitions":
        return np.where(boundary_target(labels, TRANSITION_RADIUS_FRAMES), 1.0,
                        GAP_WEIGHT_FLOOR)
    raise ValueError("unknown placement {!r}; options are {}".format(placement, GAP_PLACEMENTS))


def prepare(match_id, spec, seed, config=None, placement="random"):
    """{cost, matrix, labels, ball seen, emission logits} of a degraded match, cached."""
    config = config or DEFAULT
    suffix = "" if placement == "random" else "_" + placement
    path = os.path.join(cache_dir(), "boundary_{}{}_s{}.npz".format(match_id, suffix, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(PREPARED_KEYS) <= set(blob.files):
            return {key: blob[key] for key in PREPARED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed),
                          gap_weights=gap_weights(placement, clean, labels))
    features = feature_set.build_all(dirty, config)
    matrix, _ = feature_matrix(features, config)
    data = {"cost": emissions.costs(features, config), "matrix": matrix, "labels": labels,
            "ball_seen": dirty.ball_seen,
            "home_logit": emissions.home_logit(features, config),
            "dead_logit": emissions.dead_logit(features, config)}
    np.savez_compressed(path, **data)
    return data


# --------------------------------------------------------------------------- model

def fit(match_data):
    """Boundary classifier trained on a list of prepared matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = [d["matrix"][::TRAIN_STRIDE_FRAMES] for d in match_data]
    targets = [boundary_target(d["labels"])[::TRAIN_STRIDE_FRAMES] for d in match_data]
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    model.fit(np.concatenate(rows), np.concatenate(targets))
    return model


def boundary_score(model, data):
    return model.predict_proba(data["matrix"])[:, 1]


def decode(data, config, discount_weight=0.0, switch_scale=1.0, score=None):
    """Labels of the current pipeline, with or without the detector's guidance."""
    blocks = ((0, len(data["cost"])),)
    transition = switch_scale * viterbi.switch_costs(
        config["switch_turnover"], config["switch_to_dead"], config["switch_from_dead"])
    discount = None if score is None or discount_weight == 0 else discount_weight * score
    codes = viterbi.decode(data["cost"], transition, blocks=blocks, switch_discount=discount)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"], blocks=blocks)
    return viterbi.to_labels(codes)


def score_labels(predicted, data):
    gt_codes = ev.pm.label_codes(data["labels"])
    result = ev.score(predicted, gt_codes=gt_codes)
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"]}


def detector_quality(score, data):
    """AUC and average precision of the score against the target, plus the boundary hit rate."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    target = boundary_target(data["labels"])
    return {"auc": float(roc_auc_score(target, score)),
            "average_precision": float(average_precision_score(target, score)),
            "positive_rate": float(target.mean())}


# --------------------------------------------------------------------------- evaluation

def choose_decoder(data_by_match, score_by_match, config, grid):
    """(weight, scale) from `grid` with the best mean spellF1 over those matches; ties: macroF1."""
    results = []
    for weight, scale in grid:
        scores = [score_labels(decode(data_by_match[m], config, weight, scale,
                                      score_by_match[m]), data_by_match[m])
                  for m in data_by_match]
        results.append((round(float(np.mean([s["spell_f1"] for s in scores])), 6),
                        round(float(np.mean([s["macro_f1"] for s in scores])), 6),
                        weight, scale))
    best = max(results, key=lambda r: (r[0], r[1]))
    return {"discount_weight": best[2], "switch_scale": best[3],
            "inner_spell_f1": best[0], "inner_macro_f1": best[1],
            "grid": [list(r) for r in results]}


def nested_leave_one_match_out(data, config):
    """See the module docstring. `data[seed][match]` prepared."""
    matches = sorted(data[SEEDS[0]])
    models = {}

    def model_for(train_ids):
        key = frozenset(train_ids)
        if key not in models:
            started = time.time()
            models[key] = fit([data[SEEDS[0]][m] for m in sorted(key)])
            log("   detector trained on %s (%.0fs)" % (",".join(sorted(key)),
                                                         time.time() - started))
        return models[key]

    folds = []
    for held_out in matches:
        log("== held out: %s" % held_out)
        inner = [m for m in matches if m != held_out]
        inner_scores = {m: boundary_score(model_for([j for j in inner if j != m]),
                                          data[SEEDS[0]][m]) for m in inner}
        inner_data = {m: data[SEEDS[0]][m] for m in inner}
        choices = {arm: choose_decoder(inner_data, inner_scores, config, grid)
                   for arm, grid in ARMS.items()}
        for arm, choice in choices.items():
            log("   %-13s chosen on the 6 inside: weight %.2f scale %.2f (inside %.4f)"
                % (arm, choice["discount_weight"], choice["switch_scale"],
                   choice["inner_spell_f1"]))

        outer_model = model_for(inner)
        fold = {"held_out": held_out, "choices": choices, "seeds": {}}
        for seed in SEEDS:
            held = data[seed][held_out]
            score = boundary_score(outer_model, held)
            row = {"control": score_labels(decode(held, config), held),
                   "detector": detector_quality(score, held)}
            for arm, choice in choices.items():
                row[arm] = score_labels(decode(held, config, choice["discount_weight"],
                                               choice["switch_scale"], score), held)
            fold["seeds"][seed] = row
            log("   seed %d  control %.4f  " % (seed, row["control"]["spell_f1"])
                + "  ".join("%s %+.2f" % (arm, 100 * (row[arm]["spell_f1"]
                                                      - row["control"]["spell_f1"]))
                            for arm in ARMS)
                + "  AUC %.3f" % row["detector"]["auc"])
        folds.append(fold)
    return folds


def _paired(folds, seed, arm, reference):
    """Difference in spellF1 and macroF1, match by match, between two arms."""
    spell = np.array([f["seeds"][seed][arm]["spell_f1"] - f["seeds"][seed][reference]["spell_f1"]
                      for f in folds])
    macro = np.array([f["seeds"][seed][arm]["macro_f1"] - f["seeds"][seed][reference]["macro_f1"]
                      for f in folds])
    return {"delta_spell_f1_pp": 100 * float(spell.mean()),
            "matches_better": int((spell > 0).sum()), "matches_worse": int((spell < 0).sum()),
            "delta_macro_f1_pp": 100 * float(macro.mean())}


def summarise(folds):
    """Each arm against the control and, what decides, `both` against `scale_only`."""
    out = {}
    comparisons = [(arm, "control") for arm in ARMS] + [("both", "scale_only")]
    for seed in SEEDS:
        out[seed] = {"control_spell_f1": float(np.mean(
            [f["seeds"][seed]["control"]["spell_f1"] for f in folds]))}
        for arm, reference in comparisons:
            key = "%s_vs_%s" % (arm, reference)
            out[seed][key] = _paired(folds, seed, arm, reference)
            log("seed %d  %-24s %+.2f pp spellF1 (%d better / %d worse)  %+.2f pp macroF1"
                % (seed, key, out[seed][key]["delta_spell_f1_pp"],
                   out[seed][key]["matches_better"], out[seed][key]["matches_worse"],
                   out[seed][key]["delta_macro_f1_pp"]))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    data = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for match_id in sorted(idsse.MATCHES):
            data[seed][match_id] = prepare(match_id, spec, seed, config)
        log("seed %d prepared (%.0fs)" % (seed, time.time() - started))

    folds = nested_leave_one_match_out(data, config)
    log("=" * 78)
    summary = summarise(folds)
    path = os.path.join(REPO_ROOT, "out", "boundaries.json")
    with open(path, "w") as handle:
        json.dump({"folds": folds, "summary": summary,
                   "settings": {"half_width": BOUNDARY_HALF_WIDTH_FRAMES,
                                "windows": WINDOWS_FRAMES, "stride": TRAIN_STRIDE_FRAMES,
                                "arms": ARMS}},
                  handle, indent=2, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))


def export(config=None):
    """Trains the submission's detector on the 7 matches and chooses weight and scale.

    The same procedure that was measured, without a held-out match: weight and scale are
    chosen over the 7 with out-of-sample scores (each match scored by a detector trained on
    the other 6), and the final model is trained on the 7. Seed 0, as in the measurement.
    Writes models/boundary_detector.{joblib,json}; the two chosen values are transcribed to
    config.py with their provenance, and `score()` checks on load that the features are
    computed with the same config as here.
    """
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    matches = sorted(idsse.MATCHES)
    data = {m: prepare(m, spec, SEEDS[0], config) for m in matches}
    out_of_fold = {m: boundary_score(fit([data[j] for j in matches if j != m]), data[m])
                   for m in matches}
    choice = choose_decoder(data, out_of_fold, config, ARMS["both"])
    model = fit([data[m] for m in matches])
    names = feature_names()
    if {d["matrix"].shape[1] for d in data.values()} != {len(names)}:
        raise ValueError("the cache out/cache/boundary_*.npz is from another version of the "
                         "features; delete it and prepare again")
    metadata = boundary_detector.save(model, {
        "trained_on": matches, "degradation_seed": SEEDS[0],
        "half_width_frames": BOUNDARY_HALF_WIDTH_FRAMES,
        "train_stride_frames": TRAIN_STRIDE_FRAMES,
        "feature_names": names,
        "feature_config": boundary_detector.feature_config(config),
        "chosen_discount_weight": choice["discount_weight"],
        "chosen_switch_scale": choice["switch_scale"],
        "out_of_fold_spell_f1": choice["inner_spell_f1"],
        "out_of_fold_macro_f1": choice["inner_macro_f1"],
        "grid": choice["grid"],
    })
    log("chosen over the 7 (out of sample): weight %.2f scale %.2f -> spellF1 %.4f"
        % (choice["discount_weight"], choice["switch_scale"], choice["inner_spell_f1"]))
    log("saved %s (sha256 %s..., %.0f min)" % (
        boundary_detector.ARTIFACT_PATH, metadata["artifact_sha256"][:12],
        (time.time() - started) / 60))
    return metadata


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
