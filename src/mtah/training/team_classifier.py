"""Team classifier without the ball, trained on the IDSSE and measured with nested LOMO.

**What it tries to fix.** On the 7 IDSSE matches degraded like the ATD, the model gets
87.5% of the frames with the ball right and 74.5% of those without it (41% of the total),
and of the real possessions that pass almost entirely without the ball it recovers 15%.
Without the ball the hand-built emissions only carry over the last signal seen. This
classifier reads who has possession from the shape and movement of the two teams
(`features/team_signals.py`) and adds its logit to the emissions' team logit.

**How it is measured, and why that way.** Not per frame: in spellF1, inside the
submission's decoder. Hand-built emissions plus `weight x logit`, with the boundary
detector scoring out of sample and the discount and scale from config.py. *Nested*
leave-one-match-out over the 7 matches: for each held-out match, the classifier is trained
on the other 6 and the weight is also chosen on those 6, each one scored by a classifier
that saw neither that match nor the held-out one. The held-out match influences neither
the model nor the weight. Degradation seeds 1 and 2 only check that the result holds up
under another occlusion pattern, with the model and the weight already fixed on seed 0.

The boundary detector that comes with each match was trained on the other 6, so the one
for an inner match did see the held-out match. That is not what is chosen here, and the
held-out match is always evaluated with a detector that did not see it.

No label of the challenge match enters here. The ATD only contributes the distribution of
occlusions (`harness.real_spec`, no labels).

**What this measurement does not see.** The degraded IDSSE loses the ball at random and
keeps the players; the ATD loses it mostly with the players standing still, and then it is
also missing players (19.2 tracked on average without the ball, 21.4 with it). A domain
classifier separates the signals of the challenge match from those of the IDSSE with
AUC 0.72 (0.53 between IDSSE matches), so the gain here is an optimistic bound on the
challenge match's. Details in analysis/no-ball-1111/SUMMARY.md.

Usage:  python -m mtah.training.team_classifier           measurement -> out/team_classifier.json
        python -m mtah.training.team_classifier export    the submission's artifact
        (needs requirements-train.txt + scikit-learn)
"""

import json
import os
import time

import numpy as np

from ..config import DEFAULT
from ..features import team_signals
from ..models import emissions, team_classifier
from ..paths import REPO_ROOT, cache_dir
from . import boundaries, convention, degrade, harness, idsse
from .boundaries import SEEDS, TRAIN_STRIDE_FRAMES, log

# Fixed grid of the weight on the classifier's logit; 0 is the submission without it. It is
# the one from the Nov 11 experiment and it is not extended upward even if the edge wins: the
# weight matters more the less the challenge match resembles the IDSSE, and that cannot be
# measured here.
WEIGHTS = (0.0, 0.5, 1.0, 2.0)

# The same as the boundary detector's; they were not searched.
MODEL_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                "min_samples_leaf": 200, "l2_regularization": 1.0, "early_stopping": False,
                "random_state": 0}

PREPARED_KEYS = ("stack", "labels", "ball_seen")


# --------------------------------------------------------------------------- data

def _settings():
    return json.dumps(team_classifier.feature_settings(), sort_keys=True)


def prepare(match_id, spec, seed):
    """{smoothed signals, labels, ball seen} of a degraded match, cached.

    The same degradation as `boundaries.prepare` (same seed, random gaps): the emissions'
    logits and these signals have to come from the same tracking.
    """
    path = os.path.join(cache_dir(), "team_{}_s{}.npz".format(match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(PREPARED_KEYS) <= set(blob.files) and str(blob["settings"]) == _settings():
            return {key: blob[key] for key in PREPARED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed), gap_weights=None)
    data = {"stack": team_signals.smoothed(dirty),
            "labels": np.asarray(labels, dtype="<U1"), "ball_seen": dirty.ball_seen}
    np.savez_compressed(path, settings=np.array(_settings()), **data)
    return data


def load_matches(spec, seeds, config):
    """(base, team): `boundaries.prepare` and `prepare` for each seed and match."""
    base = {seed: {} for seed in seeds}
    team = {seed: {} for seed in seeds}
    for seed in seeds:
        for match_id in sorted(idsse.MATCHES):
            base[seed][match_id] = boundaries.prepare(match_id, spec, seed, config)
            team[seed][match_id] = prepare(match_id, spec, seed)
            if not np.array_equal(base[seed][match_id]["ball_seen"],
                                  team[seed][match_id]["ball_seen"]):
                raise ValueError("out/cache/boundary_{0}_s{1}.npz and team_{0}_s{1}.npz do not "
                                 "come from the same degradation; delete them and prepare again"
                                 .format(match_id, seed))
    return base, team


# --------------------------------------------------------------------------- model

def fit(match_data):
    """Classifier trained on both perspectives of the live frames of those matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows, targets = [], []
    for data in match_data:
        labels = data["labels"][::TRAIN_STRIDE_FRAMES]
        alive = labels != "D"
        for perspective, mine in ((0, "H"), (1, "A")):
            rows.append(team_signals.matrix(data["stack"], perspective)
                        [::TRAIN_STRIDE_FRAMES][alive])
            targets.append(labels[alive] == mine)
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(np.concatenate(rows), np.concatenate(targets))
    return model


def out_of_fold_detector(base):
    """{seed: {match: boundary score}} with the detector of the other 6 (seed 0)."""
    matches = sorted(base[SEEDS[0]])
    scores = {seed: {} for seed in base}
    for match_id in matches:
        model = boundaries.fit([base[SEEDS[0]][j] for j in matches if j != match_id])
        for seed in base:
            scores[seed][match_id] = boundaries.boundary_score(model, base[seed][match_id])
    return scores


def team_quality(logit_by_match, team_by_match):
    """AUC of the logit against the real team on live frames; separately, on those with no ball."""
    from sklearn.metrics import roc_auc_score

    logit, truth, unseen = [], [], []
    for match_id, data in team_by_match.items():
        alive = data["labels"] != "D"
        logit.append(logit_by_match[match_id][alive])
        truth.append(data["labels"][alive] == "H")
        unseen.append(~data["ball_seen"][alive])
    logit, truth, unseen = (np.concatenate(v) for v in (logit, truth, unseen))
    return {"auc": float(roc_auc_score(truth, logit)),
            "auc_ball_unseen": float(roc_auc_score(truth[unseen], logit[unseen]))}


# --------------------------------------------------------------------------- decoding

def decode_score(base, logit, detector_score, config, weight):
    """spellF1 and macroF1 of a match in the submission's decoder, with that weight.

    The logit is added outside `home_logit`, as in `emissions.costs`: the boundary
    detector does not see it. With weight 0 it is exactly the detector's measurement.
    """
    data = base
    if weight:
        home = base["home_logit"].astype(np.float64) + weight * logit.astype(np.float64)
        data = dict(base, cost=emissions.costs_from_logits(
            home, base["dead_logit"].astype(np.float64), config))
    labels = boundaries.decode(data, config, config["boundary_discount_weight"],
                               config["boundary_switch_scale"], detector_score)
    return boundaries.score_labels(labels, base)


def choose_weight(base_by_match, logit_by_match, detector_by_match, config, weights=WEIGHTS):
    """Weight from `weights` with the best mean spellF1 over those matches. Tie-break: macroF1."""
    results = []
    for weight in weights:
        scores = [decode_score(base_by_match[m], logit_by_match[m], detector_by_match[m],
                               config, weight) for m in base_by_match]
        results.append((round(float(np.mean([s["spell_f1"] for s in scores])), 6),
                        round(float(np.mean([s["macro_f1"] for s in scores])), 6), weight))
    best = max(results, key=lambda r: (r[0], r[1]))
    return {"weight": best[2], "inner_spell_f1": best[0], "inner_macro_f1": best[1],
            "grid": [list(r) for r in results]}


# --------------------------------------------------------------------------- evaluation

def nested_leave_one_match_out(base, team, detector, config):
    """See the module docstring. `base[seed][match]`, `team[seed][match]` prepared."""
    matches = sorted(team[SEEDS[0]])
    models = {}

    def model_for(train_ids):
        key = frozenset(train_ids)
        if key not in models:
            models[key] = fit([team[SEEDS[0]][m] for m in sorted(key)])
        return models[key]

    folds = []
    for held_out in matches:
        started = time.time()
        inner = [m for m in matches if m != held_out]
        inner_logit = {m: team_classifier.antisymmetric_logit(
            model_for([j for j in inner if j != m]), team[SEEDS[0]][m]["stack"]) for m in inner}
        choice = choose_weight({m: base[SEEDS[0]][m] for m in inner}, inner_logit,
                               {m: detector[SEEDS[0]][m] for m in inner}, config)
        outer_model = model_for(inner)
        fold = {"held_out": held_out, "choice": choice, "seeds": {}}
        log("== held out: %s   weight chosen on the 6 inside: %.1f (inside %.4f)   %.0fs"
            % (held_out, choice["weight"], choice["inner_spell_f1"], time.time() - started))
        for seed in SEEDS:
            logit = team_classifier.antisymmetric_logit(outer_model, team[seed][held_out]["stack"])
            by_weight = {"%g" % weight: decode_score(base[seed][held_out], logit,
                                                     detector[seed][held_out], config, weight)
                         for weight in WEIGHTS}
            row = {"control": by_weight["0"], "chosen": by_weight["%g" % choice["weight"]],
                   "by_weight": by_weight,
                   "team": team_quality({held_out: logit}, {held_out: team[seed][held_out]})}
            fold["seeds"][seed] = row
            log("   seed %d  control %.4f  chosen %+.2f  " % (
                seed, row["control"]["spell_f1"],
                100 * (row["chosen"]["spell_f1"] - row["control"]["spell_f1"]))
                + "  ".join("weight %s %+.2f" % (key, 100 * (value["spell_f1"]
                                                           - row["control"]["spell_f1"]))
                            for key, value in by_weight.items() if key != "0")
                + "  AUC %.3f (no ball %.3f)" % (row["team"]["auc"],
                                                   row["team"]["auc_ball_unseen"]))
        folds.append(fold)
    return folds


def _paired(folds, seed, pick):
    """Difference in spellF1 and macroF1 against the control, match by match."""
    spell = np.array([pick(f["seeds"][seed])["spell_f1"] - f["seeds"][seed]["control"]["spell_f1"]
                      for f in folds])
    macro = np.array([pick(f["seeds"][seed])["macro_f1"] - f["seeds"][seed]["control"]["macro_f1"]
                      for f in folds])
    return {"delta_spell_f1_pp": 100 * float(spell.mean()),
            "matches_better": int((spell > 0).sum()), "matches_worse": int((spell < 0).sum()),
            "delta_macro_f1_pp": 100 * float(macro.mean())}


def summarise(folds):
    """The nested-chosen weight against the control and, as a reference, each fixed weight."""
    out = {}
    arms = [("chosen", lambda row: row["chosen"])]
    arms += [("weight_%g" % w, lambda row, key="%g" % w: row["by_weight"][key])
             for w in WEIGHTS if w]
    for seed in SEEDS:
        out[seed] = {"control_spell_f1": float(np.mean(
            [f["seeds"][seed]["control"]["spell_f1"] for f in folds]))}
        for name, pick in arms:
            out[seed][name] = _paired(folds, seed, pick)
            log("seed %d  control %.4f  %-10s %+.2f pp spellF1 (%d better / %d worse)  "
                "%+.2f pp macroF1" % (seed, out[seed]["control_spell_f1"], name,
                                      out[seed][name]["delta_spell_f1_pp"],
                                      out[seed][name]["matches_better"],
                                      out[seed][name]["matches_worse"],
                                      out[seed][name]["delta_macro_f1_pp"]))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    base, team = load_matches(harness.real_spec(), SEEDS, config)
    log("matches prepared (%.0fs)" % (time.time() - started))
    detector = out_of_fold_detector(base)
    log("boundary detector out of sample (%.0fs)" % (time.time() - started))

    folds = nested_leave_one_match_out(base, team, detector, config)
    log("=" * 78)
    summary = summarise(folds)
    path = os.path.join(REPO_ROOT, "out", "team_classifier.json")
    with open(path, "w") as handle:
        json.dump({"folds": folds, "summary": summary,
                   "settings": {"weights": WEIGHTS, "stride": TRAIN_STRIDE_FRAMES,
                                "model": MODEL_PARAMS,
                                "features": team_classifier.feature_settings()}},
                  handle, indent=2, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))


def export(config=None):
    """Trains the submission's classifier on the 7 matches and chooses its weight.

    The same procedure that was measured, without a held-out match: the weight is chosen on
    the 7 with out-of-sample logits (each match scored by a classifier trained on the other
    6), and the final model is trained on the 7. Seed 0, as in the measurement.
    Writes models/team_classifier.{joblib,json}; the chosen weight is copied into config.py
    (`team_classifier_weight`) with its provenance.
    """
    config = config or DEFAULT
    started = time.time()
    seed = SEEDS[0]
    base, team = load_matches(harness.real_spec(), (seed,), config)
    base, team = base[seed], team[seed]
    matches = sorted(team)
    detector = out_of_fold_detector({seed: base})[seed]
    out_of_fold = {m: team_classifier.antisymmetric_logit(
        fit([team[j] for j in matches if j != m]), team[m]["stack"]) for m in matches}
    choice = choose_weight(base, out_of_fold, detector, config)
    control = next(row for row in choice["grid"] if row[2] == 0)
    quality = team_quality(out_of_fold, team)
    model = fit([team[m] for m in matches])
    metadata = team_classifier.save(model, {
        "trained_on": matches, "degradation_seed": seed,
        "train_stride_frames": TRAIN_STRIDE_FRAMES,
        "model_params": MODEL_PARAMS,
        "feature_names": team_signals.feature_names(),
        "feature_settings": team_classifier.feature_settings(),
        "chosen_weight": choice["weight"],
        "out_of_fold_spell_f1": choice["inner_spell_f1"],
        "out_of_fold_macro_f1": choice["inner_macro_f1"],
        "control_spell_f1": control[0], "control_macro_f1": control[1],
        "out_of_fold_auc": quality["auc"],
        "out_of_fold_auc_ball_unseen": quality["auc_ball_unseen"],
        "grid": choice["grid"],
    })
    log("chosen over the 7 (out of sample): weight %.1f -> spellF1 %.4f (without classifier "
        "%.4f), team AUC %.4f" % (choice["weight"], choice["inner_spell_f1"], control[0],
                                  quality["auc"]))
    log("saved %s (sha256 %s..., %.0f min)" % (
        team_classifier.ARTIFACT_PATH, metadata["artifact_sha256"][:12],
        (time.time() - started) / 60))
    return metadata


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
