"""Evidence that the boundary detector helps, where its gain comes from, and whether it
survives reality.

The nested LOMO in `boundaries.py` gave +1.17 pp of spellF1. This module answers the three
questions that number leaves open, without touching any label of the challenge match:

1. **Stress** (`stress`). The degradation places the ball-absent gaps at random; the real
   ATD loses them in crowded areas and duels. The held-out match evaluation is repeated
   with the gaps centered where there are people near the ball (`crowded`, no labels) and,
   as the worst case, next to the real possession changes (`transitions`). Detector,
   weight and scale do NOT change: they are those of each LOMO fold, trained and chosen
   with random gaps -- exactly as the submission was made. If the gain disappears here,
   the submission goes back to `boundary_discount_weight = 0`.
2. **Mechanism** (same pass). Besides spellF1: precision and recall of the changes at
   +-0.5 s and +-1 s, recall of short possessions by duration, and error rate on the
   frames next to a change versus the rest. It is what the diagnosis flagged as the weak
   point; if the detector fixes it, it has to show up there.
3. **BAR-ATH without labels** (`tagging`). The Smart Tagging marks restarts (`SET
   PIECES`) and transitions (`ATTACKING / DEFENSIVE TRANSITION`) automatically and
   independently: the pipeline does not use it. We measure whether the submission's cuts
   fall inside those windows more often than by chance, before and after the detector.
   There is no ground truth at any step, so it is a reading, never a criterion: nothing
   is chosen with this.

Usage:  python -m mtah.training.evidence stress     (~10 min, needs the IDSSE in cache)
        python -m mtah.training.evidence tagging    (needs MTAH_DATA_DIR)
"""

import json
import os
import sys
import time

import numpy as np

from .. import evaluate as ev
from .. import pipeline
from ..config import DEFAULT
from ..constants import FPS, GRADED_BLOCKS
from ..paths import REPO_ROOT
from . import boundaries, harness, idsse

pm = ev.pm

LOMO_RESULTS_PATH = os.path.join(REPO_ROOT, "analysis", "handoff-plugin-data", "data",
                                 "fronteras_lomo_anidado.json")
DATA_DIR = os.path.dirname(LOMO_RESULTS_PATH)

EVALUATED_ARMS = ("scale_only", "both")
COMPARISONS = (("both", "control"), ("both", "scale_only"))
TRANSITION_TOLERANCES_FRAMES = (12, 25)             # +-0.5 s and +-1 s
SPELL_DURATION_BINS_S = ((0, 2), (2, 5), (5, 10), (10, None))
SPELL_IOU = pm.SPELL_IOU_HEADLINE
BOOTSTRAP_RESAMPLES = 10_000

# Smart Tagging: margin around each window to count a cut as "inside". The restart windows
# already cover ~5 s around the restart (features/tagging.py).
TAG_MARGIN_FRAMES = 25
TAG_TURNOVER_CODES = ("attacking transition", "defensive transition")


def log(message):
    print(message, flush=True)


# --------------------------------------------------------------------------- metrics

def mechanism(predicted, truth):
    """spellF1 and the weak-point metrics, for a pair of 'H'/'A'/'D' label arrays."""
    pred_codes, gt_codes = pm.label_codes(predicted), pm.label_codes(truth)
    result = ev.score(predicted, gt_codes=gt_codes)
    out = {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
           "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"]}

    runs = [(0, len(gt_codes))]
    gt_transitions = pm.transitions(gt_codes, runs)
    pred_transitions = pm.transitions(pred_codes, runs)
    for tolerance in TRANSITION_TOLERANCES_FRAMES:
        found = pm.transition_scores(gt_transitions, pred_transitions, tolerance)
        for key in ("precision", "recall", "f1"):
            out["transition_%s_%dfr" % (key, tolerance)] = found[key]

    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(pred_codes, runs)
    matched = pm.matched_gt_flags(pm.spell_overlaps(gt_spells, pred_spells),
                                  len(gt_spells), SPELL_IOU)
    duration_s = np.array([(stop - start) / float(FPS) for start, stop, _ in gt_spells])
    for low, high in SPELL_DURATION_BINS_S:
        in_bin = (duration_s >= low) & (duration_s < (np.inf if high is None else high))
        out["spell_recall_%s" % _bin_name(low, high)] = \
            float(matched[in_bin].mean()) if in_bin.any() else None

    near = boundaries.boundary_target(truth)
    wrong = np.asarray(predicted) != np.asarray(truth)
    out["error_near_change"] = float(wrong[near].mean())
    out["error_elsewhere"] = float(wrong[~near].mean())
    return out


def _bin_name(low, high):
    return "%s_%ss" % (low, "inf" if high is None else high)


def paired_statistics(deltas, rng):
    """Mean, bootstrap 95% CI and sign test (one-sided) of per-match differences."""
    deltas = np.asarray(deltas, dtype=np.float64)
    resampled = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas))).mean(axis=1)
    better = int((deltas > 0).sum())
    n = len(deltas)
    from math import comb
    sign_p = sum(comb(n, k) for k in range(better, n + 1)) / 2.0 ** n
    return {"mean_pp": 100 * float(deltas.mean()),
            "ci95_pp": [100 * float(np.percentile(resampled, 2.5)),
                        100 * float(np.percentile(resampled, 97.5))],
            "matches_better": better, "matches_worse": int((deltas < 0).sum()),
            "sign_test_p_one_sided": float(sign_p),
            "per_match_pp": [round(100 * float(d), 2) for d in deltas]}


# --------------------------------------------------------------------------- 1 and 2

def lomo_choices():
    """{held-out match: {arm: (weight, scale)}} as they were chosen in the LOMO."""
    with open(LOMO_RESULTS_PATH) as handle:
        folds = json.load(handle)["folds"]
    return {fold["held_out"]: {arm: (fold["choices"][arm]["discount_weight"],
                                     fold["choices"][arm]["switch_scale"])
                               for arm in EVALUATED_ARMS}
            for fold in folds}, folds


def check_reproduces_lomo(rows, folds):
    """With random gaps, this pass must give bit for bit what the LOMO gave."""
    published = {(f["held_out"], int(seed)): f["seeds"][seed] for f in folds
                 for seed in f["seeds"]}
    for row in rows:
        if row["placement"] != "random":
            continue
        reference = published[(row["match"], row["seed"])]
        for arm in ("control",) + EVALUATED_ARMS:
            if abs(row[arm]["spell_f1"] - reference[arm]["spell_f1"]) > 1e-12:
                raise RuntimeError(
                    "%s seed %d %s: %.6f != %.6f published. The code or the cache changed "
                    "since the LOMO; this check is not valid until that is explained."
                    % (row["match"], row["seed"], arm, row[arm]["spell_f1"],
                       reference[arm]["spell_f1"]))


def stress_rows(config):
    """One row per (held-out match, placement, seed) with the control and the two arms."""
    spec = harness.real_spec()
    matches = sorted(idsse.MATCHES)
    choices, folds = lomo_choices()
    train = {m: boundaries.prepare(m, spec, boundaries.SEEDS[0], config) for m in matches}
    rows = []
    for held_out in matches:
        model = boundaries.fit([train[m] for m in matches if m != held_out])
        for placement in boundaries.GAP_PLACEMENTS:
            for seed in boundaries.SEEDS:
                data = boundaries.prepare(held_out, spec, seed, config, placement)
                score = boundaries.boundary_score(model, data)
                labels = data["labels"]
                change_frames = np.flatnonzero(labels[1:] != labels[:-1]) + 1
                row = {"match": held_out, "placement": placement, "seed": seed,
                       "ball_visible": float(data["ball_seen"].mean()),
                       "changes_without_ball": float((~data["ball_seen"][change_frames]).mean()),
                       "detector": boundaries.detector_quality(score, data),
                       "control": mechanism(boundaries.decode(data, config), labels)}
                for arm in EVALUATED_ARMS:
                    weight, scale = choices[held_out][arm]
                    row[arm] = mechanism(boundaries.decode(data, config, weight, scale, score),
                                         labels)
                rows.append(row)
                log("%s %-11s s%d  visible %.2f  changes without ball %.2f  AUC %.3f   "
                    "control %.4f  both %+.2f pp  vs scale %+.2f pp" % (
                        held_out, placement, seed, row["ball_visible"],
                        row["changes_without_ball"], row["detector"]["auc"],
                        row["control"]["spell_f1"],
                        100 * (row["both"]["spell_f1"] - row["control"]["spell_f1"]),
                        100 * (row["both"]["spell_f1"] - row["scale_only"]["spell_f1"])))
    check_reproduces_lomo(rows, folds)
    return rows


def summarise_stress(rows):
    """Per placement: paired per-match statistic and the mean of each metric per arm."""
    rng = np.random.default_rng(0)
    out = {}
    for placement in boundaries.GAP_PLACEMENTS:
        subset = [r for r in rows if r["placement"] == placement]
        matches = sorted({r["match"] for r in subset})
        summary = {"ball_visible": float(np.mean([r["ball_visible"] for r in subset])),
                   "changes_without_ball": float(np.mean([r["changes_without_ball"]
                                                          for r in subset])),
                   "detector_auc": float(np.mean([r["detector"]["auc"] for r in subset])),
                   "detector_average_precision": float(np.mean(
                       [r["detector"]["average_precision"] for r in subset]))}
        for arm, reference in COMPARISONS:
            key = "%s_vs_%s" % (arm, reference)
            seed0 = [r[arm]["spell_f1"] - r[reference]["spell_f1"]
                     for r in subset if r["seed"] == boundaries.SEEDS[0]]
            seed_mean = [np.mean([r[arm]["spell_f1"] - r[reference]["spell_f1"]
                                  for r in subset if r["match"] == m]) for m in matches]
            summary[key] = {"seed0": paired_statistics(seed0, rng),
                            "mean_of_seeds": paired_statistics(seed_mean, rng)}
        metrics = [k for k in subset[0]["control"] if subset[0]["control"][k] is not None]
        summary["metrics_mean_all_seeds"] = {
            arm: {k: float(np.mean([r[arm][k] for r in subset if r[arm][k] is not None]))
                  for k in metrics}
            for arm in ("control",) + EVALUATED_ARMS}
        out[placement] = summary
    return out


def print_stress(summary):
    log("=" * 96)
    log("%-12s %8s %8s %6s   %-30s %-30s" % ("placement", "visible", "changes", "AUC",
                                            "both vs control (seed 0)",
                                            "both vs scale_only (seed 0)"))
    for placement, s in summary.items():
        cells = []
        for key in ("both_vs_control", "both_vs_scale_only"):
            st = s[key]["seed0"]
            cells.append("%+.2f [%+.2f, %+.2f] %d/7 p=%.3f" % (
                st["mean_pp"], st["ci95_pp"][0], st["ci95_pp"][1], st["matches_better"],
                st["sign_test_p_one_sided"]))
        log("%-12s %8.2f %8.2f %6.3f   %-30s %-30s" % (
            placement, s["ball_visible"], s["changes_without_ball"], s["detector_auc"],
            cells[0], cells[1]))
    log("-" * 96)
    for placement, s in summary.items():
        log("[%s] mean of 7 matches x 3 seeds" % placement)
        metrics = s["metrics_mean_all_seeds"]
        for key in metrics["control"]:
            log("   %-26s control %.4f   scale_only %.4f   both %.4f" % (
                key, metrics["control"][key], metrics["scale_only"][key],
                metrics["both"][key]))


def stress(config=None):
    config = config or DEFAULT
    started = time.time()
    rows = stress_rows(config)
    summary = summarise_stress(rows)
    print_stress(summary)
    result = {"rows": rows, "summary": summary,
              "settings": {"placements": boundaries.GAP_PLACEMENTS,
                           "crowd_radius_m": boundaries.CROWD_RADIUS_M,
                           "transition_radius_frames": boundaries.TRANSITION_RADIUS_FRAMES,
                           "gap_weight_floor": boundaries.GAP_WEIGHT_FLOOR,
                           "transition_tolerances_frames": TRANSITION_TOLERANCES_FRAMES,
                           "spell_iou": SPELL_IOU}}
    path = os.path.join(DATA_DIR, "evidencia_estres_mecanismo.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))
    return result


# --------------------------------------------------------------------------- 3

def tag_windows(phases):
    """{'restart': [(a, b)], 'turnover': [(a, b)]} from the Smart Tagging, without duplicates.

    The ATTACKING / DEFENSIVE TRANSITION tags come mirrored (same interval, one tag per
    team), hence the set.
    """
    restart = sorted({(p.start_frame, p.end_frame) for p in phases if p.is_set_piece})
    turnover = sorted({(p.start_frame, p.end_frame) for p in phases
                       if p.code.strip().lower() in TAG_TURNOVER_CODES})
    return {"restart": restart, "turnover": turnover}


def window_mask(windows, n_frames, margin=TAG_MARGIN_FRAMES):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in windows:
        mask[max(0, start - margin):min(n_frames, stop + margin)] = True
    return mask


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in GRADED_BLOCKS:
        mask[start:stop] = True
    return mask


def tagging_alignment(labels, windows, graded):
    """How many cuts fall in tagging windows and how many windows have some cut."""
    codes = pm.label_codes(labels)
    cuts = pm.transitions(codes, list(GRADED_BLOCKS))
    n = len(labels)
    out = {"n_cuts": len(cuts)}
    kinds = {"restart": lambda frm, to: frm == pm.D and to in pm.POSSESSION_LABELS,
             "turnover": lambda frm, to: frm != to}
    for name, is_kind in kinds.items():
        inside = window_mask(windows[name], n)
        coverage = float(inside[graded].mean())
        frames = np.array([f for f, frm, to in cuts if is_kind(frm, to)])
        hit = inside[frames] if len(frames) else np.zeros(0, dtype=bool)
        cut_mask = np.zeros(n, dtype=bool)
        cut_mask[frames] = True
        covered = [bool(cut_mask[max(0, a - TAG_MARGIN_FRAMES):b + TAG_MARGIN_FRAMES].any())
                   for a, b in windows[name]
                   if any(lo <= a < hi for lo, hi in GRADED_BLOCKS)]
        out[name] = {"n_windows": len(covered), "window_coverage_of_graded": coverage,
                     "n_cuts_of_kind": int(len(frames)),
                     "cuts_inside": float(hit.mean()) if len(hit) else None,
                     "lift_over_chance": float(hit.mean() / coverage) if len(hit) else None,
                     "windows_with_cut": float(np.mean(covered))}
    return out


def changed_cuts(before, after, windows, graded, tolerance=TAG_MARGIN_FRAMES):
    """Cuts the detector added and removed, and what fraction of each falls in a window."""
    def frames(labels):
        return np.array([f for f, _, _ in pm.transitions(pm.label_codes(labels),
                                                          list(GRADED_BLOCKS))])

    old, new = frames(before), frames(after)
    added = np.array([f for f in new if np.abs(old - f).min() > tolerance])
    removed = np.array([f for f in old if np.abs(new - f).min() > tolerance])
    inside = window_mask(windows["restart"] + windows["turnover"], len(before))
    return {"added": int(len(added)), "removed": int(len(removed)),
            "added_inside": float(inside[added].mean()) if len(added) else None,
            "removed_inside": float(inside[removed].mean()) if len(removed) else None,
            "chance_inside": float(inside[graded].mean())}


def tagging(config=None):
    from ..io import atd, smart_tagging

    config = config or DEFAULT
    without = dict(config, boundary_discount_weight=0.0)
    tracking = atd.load()
    before = pipeline.labels_from_tracking(tracking, without)
    after = pipeline.labels_from_tracking(tracking, config)
    windows = tag_windows(smart_tagging.load())
    graded = graded_mask(len(before))
    result = {"without_detector": tagging_alignment(before, windows, graded),
              "with_detector": tagging_alignment(after, windows, graded),
              "changed_cuts": changed_cuts(before, after, windows, graded),
              "settings": {"margin_frames": TAG_MARGIN_FRAMES,
                           "turnover_codes": TAG_TURNOVER_CODES}}
    log(json.dumps(result, indent=1))
    path = os.path.join(DATA_DIR, "evidencia_smart_tagging.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    log("saved %s" % path)
    return result


if __name__ == "__main__":
    commands = {"stress": stress, "tagging": tagging}
    if sys.argv[1:2] == [] or sys.argv[1] not in commands:
        raise SystemExit("usage: python -m mtah.training.evidence {%s}" % "|".join(commands))
    commands[sys.argv[1]]()
