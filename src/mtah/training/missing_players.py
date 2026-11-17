"""Stress test: the team classifier when players are missing where the ball is missing.

**What it checks.** The +16.1 pp of the team classifier (training/team_classifier.py) was
measured with the degraded IDSSE, which keeps the players when the ball is missing: 20.4
objects on average with and without the ball. In the challenge's ATD that is not so: 21.4
with the ball, 19.2 without it, and 14.3% of the ball-absent frames have fewer than 16
(STATUS.md §6). That is exactly where the classifier has to work. Here the held-out-match
evaluation is repeated with the ATD's object histogram *without the ball* imposed on the
ball-absent frames (`degrade.thin_where_ball_unseen`), and nothing else changes: same ball,
same players in the frames with the ball, same seeds.

Two arms with the same per-frame histogram:

- `thinned`: the players that are removed are drawn in each frame. They flicker, and the
  1 to 8 s windows of the signals average out a good part of that noise: it is the easy
  version.
- `persistent`: they are drawn once per ball-absent run and are missing for the whole run,
  like a lost ATD track. **The decision comes from this arm.**

**What is not touched.** The models are those of the Nov 12 measurement *exactly as they
were submitted*: for each held-out match, the classifier and the boundary detector are
trained on the other 6, degraded the usual way (seed 0). The weight is the one in
config.py (`team_classifier_weight`), fixed; nothing is chosen here. The question is
whether what was submitted holds up against this data, not which configuration would hold
up better.

**Check that the harness is the same.** The same run also scores the usual degradation
(`standard`), which has to reproduce `data/equipo_sin_balon_lomo.json` (the chosen weight
was 2.0 in all 7 folds). The arms' differences are paired against their own control with
the same models, so a small mismatch (another machine retraining the boosting) does not
invalidate them, but it is reported.

**Decision rule, written on Nov 15 before seeing any result.** Over the 21 runs of the
`persistent` arm (7 matches x 3 seeds), difference in spellF1 with the config.py weight
against weight 0:

- mean >= +5 pp and at least 15 of 21 improve: the `a43c52fb...` submission **is kept**;
- mean <= +1 pp, or fewer than 11 of 21 improve: **revert** to `e7938350...`
  (`team_classifier_weight = 0`);
- in between: it is kept, and STATUS.md reports the reduced gain.

If `thinned` (the easy version) already says "revert", that is enough to revert. Nothing is
retuned with this: not the weight, not the switch costs, not the signals.

**What we still don't see.** Which players are missing: here they are chosen at random
without looking at the team; in the ATD it is probably the ones far from the camera or the
bunched-up ones that are missing, and more may be missing from one team than from the other
(the home - away imbalance of the ATD is compared, without labels, with the one here and
reported). In the frames with the ball the degraded IDSSE stays at ~20.4 objects against
21.4 in the ATD (one can only remove, not invent). The ball-absent gaps are still random
(the test with gaps placed by activity is another one,
`analysis/no-ball-1111/realista.py`). It is not a worst-case bound.

Usage:  python -m mtah.training.missing_players      -> out/missing_players.json
        (needs MTAH_DATA_DIR to measure the ATD, requirements-train.txt and scikit-learn)
"""

import json
import os
import time

import numpy as np

from .. import features as feature_set
from ..config import DEFAULT
from ..constants import GRADED_BLOCKS
from ..features import geometry, team_signals
from ..features.boundaries import feature_matrix
from ..io import atd
from ..models import emissions
from ..models import team_classifier as shipped_classifier
from ..paths import REPO_ROOT, cache_dir
from . import boundaries, convention, degrade, harness, idsse
from . import team_classifier as team_training
from .boundaries import SEEDS, log

# Arm -> draw the missing ones once per ball-absent run (True) or in each frame.
ARMS = {"thinned": False, "persistent": True}
DECISION_ARM = "persistent"

THINNED_KEYS = boundaries.PREPARED_KEYS + ("stack", "n_tracked", "n_home")

# Thresholds of the decision rule in the docstring. They are not changed after seeing the result.
KEEP_MIN_PP, KEEP_MIN_BETTER = 5.0, 15
REVERT_MAX_PP, REVERT_MIN_BETTER = 1.0, 11

FEW_PLAYERS = 16

LOMO_TABLE = os.path.join(REPO_ROOT, "analysis", "handoff-plugin-data", "data",
                          "equipo_sin_balon_lomo.json")


# --------------------------------------------------------------------------- data

def _thinning_rng(seed):
    """The thinning's own generator, distinct from that of `degrade.apply` with that seed."""
    return np.random.default_rng([seed, 7])


def _tracked(tracking):
    """(objects per frame, home team's objects per frame)."""
    live = np.isfinite(tracking.players_xy[:, :, 0])
    home = live[:, tracking.team_of_slot == geometry.HOME_CODE]
    return live.sum(axis=1).astype(np.int16), home.sum(axis=1).astype(np.int16)


def prepare_thinned(match_id, spec, seed, config, persistent):
    """What `boundaries.prepare` and `team_classifier.prepare` do, on the thinned tracking."""
    name = "thinned_persistent" if persistent else "thinned"
    path = os.path.join(cache_dir(), "{}_{}_s{}.npz".format(name, match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(THINNED_KEYS) <= set(blob.files):
            return {key: blob[key] for key in THINNED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed), gap_weights=None)
    thinned = degrade.thin_where_ball_unseen(dirty, spec, _thinning_rng(seed),
                                             persistent=persistent)
    features = feature_set.build_all(thinned, config)
    matrix, _ = feature_matrix(features, config)
    n_tracked, n_home = _tracked(thinned)
    data = {"cost": emissions.costs(features, config), "matrix": matrix, "labels": labels,
            "ball_seen": thinned.ball_seen,
            "home_logit": emissions.home_logit(features, config),
            "dead_logit": emissions.dead_logit(features, config),
            "stack": team_signals.smoothed(thinned),
            "n_tracked": n_tracked, "n_home": n_home}
    np.savez_compressed(path, **data)
    return data


def player_counts(n_tracked, n_home, ball_seen):
    """Objects with and without the ball, ball-absent frames with fewer than 16, and the
    imbalance between teams.

    `team_gap_unseen` is the mean of |home - away| in tracked objects, without the ball.
    """
    unseen = ~ball_seen
    n_tracked, n_home = n_tracked.astype(np.int64), n_home.astype(np.int64)
    gap = np.abs(2 * n_home - n_tracked)[unseen]
    return {"mean_ball_seen": float(n_tracked[ball_seen].mean()),
            "mean_ball_unseen": float(n_tracked[unseen].mean()) if unseen.any() else None,
            "unseen_below_16": float((n_tracked[unseen] < FEW_PLAYERS).mean())
            if unseen.any() else None,
            "team_gap_unseen": float(gap.mean()) if unseen.any() else None}


def atd_player_counts():
    """The same measured on the graded frames of the challenge's ATD, without labels."""
    tracking = atd.load()
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, end in GRADED_BLOCKS:
        graded[start:end] = True
    n_tracked, n_home = _tracked(tracking)
    return player_counts(n_tracked[graded], n_home[graded], tracking.ball_seen[graded])


# --------------------------------------------------------------------------- evaluation

def evaluate(base, team, arms, config):
    """Per held-out match and seed: control and config weight on each degradation."""
    weight = config["team_classifier_weight"]
    matches = sorted(team[SEEDS[0]])
    folds = []
    for held_out in matches:
        started = time.time()
        inner = [m for m in matches if m != held_out]
        classifier = team_training.fit([team[SEEDS[0]][m] for m in inner])
        detector = boundaries.fit([base[SEEDS[0]][m] for m in inner])
        fold = {"held_out": held_out, "seeds": {}}
        for seed in SEEDS:
            sources = [("standard", base[seed][held_out], team[seed][held_out]["stack"])]
            sources += [(arm, arms[arm][seed][held_out], arms[arm][seed][held_out]["stack"])
                        for arm in ARMS]
            row = {}
            for name, data, stack in sources:
                logit = shipped_classifier.antisymmetric_logit(classifier, stack)
                score = boundaries.boundary_score(detector, data)
                control = team_training.decode_score(data, logit, score, config, 0.0)
                chosen = team_training.decode_score(data, logit, score, config, weight)
                quality = team_training.team_quality(
                    {held_out: logit}, {held_out: {"labels": data["labels"],
                                                   "ball_seen": data["ball_seen"]}})
                row[name] = {"control": control, "weighted": chosen, "team": quality,
                             "delta_spell_f1_pp": 100 * (chosen["spell_f1"]
                                                         - control["spell_f1"])}
                if name in ARMS:
                    row[name]["players"] = player_counts(data["n_tracked"], data["n_home"],
                                                         data["ball_seen"])
            fold["seeds"][seed] = row
            log("%s s%d  " % (held_out, seed) + "  ".join(
                "%s %+.2f pp (AUC ball-absent %.3f)" % (
                    name, row[name]["delta_spell_f1_pp"], row[name]["team"]["auc_ball_unseen"])
                for name in ("standard",) + tuple(ARMS)))
        log("   %s in %.0fs" % (held_out, time.time() - started))
        folds.append(fold)
    return folds


def check_reproduces_lomo(folds, weight, path=LOMO_TABLE, tolerance=1e-9):
    """List of discrepancies of the usual degradation against the Nov 12 table (empty = ok)."""
    if not os.path.exists(path):
        return ["not found: {}".format(path)]
    with open(path, encoding="utf-8") as handle:
        reference = {f["held_out"]: f for f in json.load(handle)["folds"]}
    problems = []
    for fold in folds:
        ref = reference.get(fold["held_out"])
        if ref is None:
            problems.append("%s is not in the table" % fold["held_out"])
            continue
        for seed, row in fold["seeds"].items():
            ref_row = ref["seeds"][str(seed)]
            pairs = (("control", row["standard"]["control"], ref_row["control"]),
                     ("weight %g" % weight, row["standard"]["weighted"],
                      ref_row["by_weight"]["%g" % weight]))
            for name, mine, theirs in pairs:
                if abs(mine["spell_f1"] - theirs["spell_f1"]) > tolerance:
                    problems.append("%s s%s %s: %.6f here, %.6f in the table" % (
                        fold["held_out"], seed, name, mine["spell_f1"], theirs["spell_f1"]))
    return problems


def decide(deltas_pp):
    """The rule in the docstring over the spellF1 differences (pp) of the 21 runs."""
    deltas_pp = np.asarray(deltas_pp, dtype=np.float64)
    mean, better = float(deltas_pp.mean()), int((deltas_pp > 0).sum())
    if mean >= KEEP_MIN_PP and better >= KEEP_MIN_BETTER:
        verdict = "mantener"
    elif mean <= REVERT_MAX_PP or better < REVERT_MIN_BETTER:
        verdict = "volver a peso 0"
    else:
        verdict = "mantener, con la ganancia reducida documentada"
    return {"mean_pp": mean, "better": better, "runs": int(len(deltas_pp)),
            "verdict": verdict}


def final_verdict(summary):
    """The persistent arm's, unless the easy one already says revert."""
    if summary["thinned"]["verdict"] == "volver a peso 0":
        return "volver a peso 0"
    return summary[DECISION_ARM]["verdict"]


def summarise(folds):
    out = {}
    for name in ("standard",) + tuple(ARMS):
        deltas = [f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds for s in f["seeds"]]
        per_seed = {s: float(np.mean([f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds]))
                    for s in SEEDS}
        out[name] = dict(decide(deltas), per_seed_mean_pp=per_seed,
                         min_pp=float(np.min(deltas)), max_pp=float(np.max(deltas)))
        log("%-10s %+.2f pp on average (%d of %d improve; from %+.2f to %+.2f)  per seed %s" % (
            name, out[name]["mean_pp"], out[name]["better"], out[name]["runs"],
            out[name]["min_pp"], out[name]["max_pp"],
            "  ".join("%+.2f" % v for v in per_seed.values())))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    real_counts = atd_player_counts()
    log("challenge ATD: %s" % json.dumps(spec.summary()))
    log("challenge ATD, objects: %s" % json.dumps(real_counts))
    base, team = team_training.load_matches(spec, SEEDS, config)
    arms = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    for arm, persistent in ARMS.items():
        for seed in SEEDS:
            for match_id in sorted(idsse.MATCHES):
                data = prepare_thinned(match_id, spec, seed, config, persistent)
                if not np.array_equal(data["ball_seen"], base[seed][match_id]["ball_seen"]):
                    raise ValueError("arm {0} of {1} s{2} does not come from the same degradation "
                                     "as boundary_{1}_s{2}.npz; delete the caches and run "
                                     "again".format(arm, match_id, seed))
                arms[arm][seed][match_id] = data
    log("matches prepared (%.0fs)" % (time.time() - started))

    folds = evaluate(base, team, arms, config)
    log("=" * 78)
    problems = check_reproduces_lomo(folds, config["team_classifier_weight"])
    for problem in problems:
        log("DOES NOT REPRODUCE: " + problem)
    summary = summarise(folds)
    simulated = {arm: player_counts(
        np.concatenate([arms[arm][s][m]["n_tracked"] for s in SEEDS for m in arms[arm][s]]),
        np.concatenate([arms[arm][s][m]["n_home"] for s in SEEDS for m in arms[arm][s]]),
        np.concatenate([arms[arm][s][m]["ball_seen"] for s in SEEDS for m in arms[arm][s]]))
        for arm in ARMS}
    for arm, counts in simulated.items():
        log("objects in %-10s %s" % (arm, json.dumps(counts)))
    verdict = final_verdict(summary)
    log("reproduces the Nov 12 table: %s" % ("yes" if not problems else
                                             "NO (%d discrepancies)" % len(problems)))
    log("decision rule (arm %s): %s" % (DECISION_ARM, verdict.upper()))

    path = os.path.join(REPO_ROOT, "out", "missing_players.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"folds": folds, "summary": summary, "verdict": verdict,
                   "reproduces_lomo": not problems, "lomo_problems": problems,
                   "spec": spec.summary(), "atd_players": real_counts,
                   "simulated_players": simulated,
                   "settings": {"weight": config["team_classifier_weight"],
                                "decision_arm": DECISION_ARM,
                                "keep": [KEEP_MIN_PP, KEEP_MIN_BETTER],
                                "revert": [REVERT_MAX_PP, REVERT_MIN_BETTER]}},
                  handle, indent=2, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
