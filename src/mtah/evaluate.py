"""Local scoring with the *official* scorer vendored in vendor/evaluation.

We do not reimplement the metric on purpose: `possession_metrics.py` is the specification and
the file that decides the competition. If our number and theirs disagree, the bug is
ours.

Reminder from EVALUATION.md, worth having in front of you every time you look at a number
from here: the public window is 5 minutes and 17 possessions. **It is no use for choosing
between two versions of your model**, only for checking that the pipeline is correct. One
missed possession moves spellF1 by about 6 points here. Differences below ~0.04 in
spellF1 (and ~0.01 in macroF1) are noise even over the full match.
"""

import os
import sys

import numpy as np

from .paths import REPO_ROOT

VENDOR = os.path.join(REPO_ROOT, "vendor", "evaluation")
PUBLIC_GT = os.path.join(VENDOR, "gt", "BAR-ATH_possession_gt_public.csv")

if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)
import possession_metrics as pm                              # noqa: E402


def public_gt():
    """Codes of the public ground truth (5 real minutes, the rest X)."""
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def score(labels, gt_codes=None, bootstrap=0):
    """Dict with all the official metrics for an array of labels."""
    gt_codes = public_gt() if gt_codes is None else gt_codes
    pred = pm.label_codes(np.asarray(labels, dtype="<U1"))
    return pm.evaluate(gt_codes, pred, bootstrap_n=bootstrap)


def headline(labels, gt_codes=None, bootstrap=0):
    """The numbers that matter, in a small dict."""
    result = score(labels, gt_codes, bootstrap)
    return {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_pred_poss": result["n_pred_spells"],
        "n_gt_poss": result["n_gt_spells"],
        "mean_dur_s": result["mean_spell_duration_s"],
        "gt_mean_dur_s": result["gt_mean_spell_duration_s"],
        "f1_D": result["per_class"]["D"]["f1"],
        "latency_mean": result["transitions"][25]["latency_mean"]
        if "latency_mean" in result["transitions"][25] else None,
    }


def report(labels, name="submission", gt_codes=None, bootstrap=0):
    """Prints a one-line summary plus the per-class breakdown."""
    result = score(labels, gt_codes, bootstrap)
    print("{}  spellF1 {:.4f}  macroF1 {:.4f}   poss {} vs {} (gt)   dur {:.1f}s vs {:.1f}s"
          .format(name, result["spell_f1_mean"], result["macro_f1"],
                  result["n_pred_spells"], result["n_gt_spells"],
                  result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    for label in ("H", "A", "D"):
        cls = result["per_class"][label]
        print("   {}  P {:.4f}  R {:.4f}  F1 {:.4f}".format(
            label, cls["precision"], cls["recall"], cls["f1"]))
    per_iou = result["spell_f1"]
    print("   spellF1 per IoU: " + "  ".join(
        "@{} {:.4f}".format(k, v["f1"]) for k, v in sorted(per_iou.items())))
    return result
