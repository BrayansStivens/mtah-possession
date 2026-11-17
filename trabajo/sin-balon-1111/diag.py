"""Where the submitted model loses when the ball is not seen. Degraded IDSSE, seed 0.

Predictions = the submission pipeline on each match (emissions from config.py + out-of-sample
boundary detector, weight 0.95 and scale 2.0). Read-only: it chooses nothing.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.constants import GRADED_BLOCKS                       # noqa: E402
from mtah.io import atd                                        # noqa: E402
from mtah.training import boundaries as B                      # noqa: E402

GAP_BUCKETS = (("visto", 0, 0), ("hueco 1-12", 1, 12), ("hueco 13-50", 13, 50),
               ("hueco 51-125", 51, 125), ("hueco 126-250", 126, 250),
               ("hueco >250", 251, 10 ** 9))
FRAC_BUCKETS = (("0 %", -1, 0.0), ("0-25 %", 0.0, 0.25), ("25-50 %", 0.25, 0.5),
                ("50-75 %", 0.5, 0.75), ("75-100 %", 0.75, 1.0))


def frame_tables(gt, pred, seen):
    gap = C.gap_lengths(seen)
    blind = C.no_ball_in_window(seen)
    rows = {}
    masks = {name: (gap >= lo) & (gap <= hi) for name, lo, hi in GAP_BUCKETS}
    masks["sin balon (todo)"] = ~seen
    masks["sin balon a +-50 (control NaN)"] = blind
    masks["todo"] = np.ones(len(gt), dtype=bool)
    alive_gt, alive_pred = gt != "D", pred != "D"
    for name, mask in masks.items():
        n = int(mask.sum())
        wrong = mask & (gt != pred)
        rows[name] = {
            "frames": n, "aciertos": int((mask & (gt == pred)).sum()),
            "err_equipo": int((wrong & alive_gt & alive_pred).sum()),
            "err_muerto_no_visto": int((wrong & ~alive_gt).sum()),
            "err_muerto_inventado": int((wrong & alive_gt & ~alive_pred).sum()),
            "gt_D": int((mask & ~alive_gt).sum()),
        }
    return rows


def spell_tables(gt, pred, seen):
    gt_codes, pred_codes = C.pm.label_codes(gt), C.pm.label_codes(pred)
    runs = C.pm.valid_runs(gt_codes)
    gt_sp = C.pm.possession_spells(gt_codes, runs)
    pred_sp = C.pm.possession_spells(pred_codes, runs)
    overlaps = C.pm.spell_overlaps(gt_sp, pred_sp)
    flags = C.pm.matched_gt_flags(overlaps, len(gt_sp), 0.5)
    matched_pred = np.zeros(len(pred_sp), dtype=bool)
    used_gt = set()
    for iou, i, j in overlaps:
        if iou < 0.5:
            break
        if i in used_gt or matched_pred[j]:
            continue
        used_gt.add(i)
        matched_pred[j] = True
    cum = np.concatenate(([0], np.cumsum(~seen)))

    def table(spells, hit):
        frac = np.array([(cum[b] - cum[a]) / float(b - a) for a, b, _ in spells])
        dur = np.array([(b - a) / 25.0 for a, b, _ in spells])
        out = {"por_fraccion_sin_balon": {}, "por_duracion": {}}
        for name, lo, hi in FRAC_BUCKETS:
            m = (frac > lo) & (frac <= hi)
            out["por_fraccion_sin_balon"][name] = [int(m.sum()), int(hit[m].sum())]
        for name, lo, hi in (("<3 s", 0, 3), ("3-6 s", 3, 6), ("6-12 s", 6, 12),
                             ("12-25 s", 12, 25), (">25 s", 25, 1e9)):
            m = (dur >= lo) & (dur < hi)
            out["por_duracion"][name] = [int(m.sum()), int(hit[m].sum())]
        return out

    return {"gt": table(gt_sp, flags), "pred": table(pred_sp, matched_pred)}


def transition_tables(gt, pred, seen, tol=25):
    """Real changes found within +-tol frames with the same target state."""
    blind = C.no_ball_in_window(seen, 12)
    change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    pchange = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    out = {}
    for f in change:
        kind = ("robo" if gt[f - 1] != "D" and gt[f] != "D"
                else "a_muerto" if gt[f] == "D" else "reinicio")
        where = "sin balon a +-12" if blind[f] else "con balon"
        near = pchange[(pchange >= f - tol) & (pchange <= f + tol)]
        hit = bool(len(near)) and bool((pred[near] == gt[f]).any())
        key = "%s | %s" % (kind, where)
        n, h = out.get(key, (0, 0))
        out[key] = (n + 1, h + int(hit))
    return {k: list(v) for k, v in out.items()}


def add(total, part):
    for k, v in part.items():
        if isinstance(v, dict):
            add(total.setdefault(k, {}), v)
        elif isinstance(v, (list, tuple)):
            total[k] = [a + b for a, b in zip(total.get(k, [0] * len(v)), v)]
        else:
            total[k] = total.get(k, 0) + v


def atd_gap_profile():
    """No labels: how the graded frames of the real ATD split by gap type."""
    tracking = atd.load(source="tactical")
    seen = tracking.ball_seen
    gap, blind = C.gap_lengths(seen), C.no_ball_in_window(seen)
    mask = np.zeros(len(seen), dtype=bool)
    for a, b in GRADED_BLOCKS:
        mask[a:b] = True
    out = {name: float(((gap >= lo) & (gap <= hi))[mask].mean()) for name, lo, hi in GAP_BUCKETS}
    out["sin balon a +-50 (control NaN)"] = float(blind[mask].mean())
    return out


def main():
    data = {s: C.base(s) for s in C.SEEDS}
    scores = C.oof_scores(data)
    total = {"frames": {}, "spells": {}, "transitions": {}}
    per_match = {}
    for m in C.MATCHES:
        d = data[0][m]
        gt = np.asarray(d["labels"], dtype="<U1")
        pred = np.asarray(B.decode(d, DEFAULT, C.DISCOUNT, C.SCALE, scores[0][m]), dtype="<U1")
        seen = d["ball_seen"].astype(bool)
        spell, macro, n_pred, n_gt = C.score(pred, gt)
        per_match[m] = {"spell_f1": spell, "macro_f1": macro, "n_pred": n_pred, "n_gt": n_gt,
                        "visible": float(seen.mean())}
        print("%s  spellF1 %.4f  macroF1 %.4f  pos %d vs %d  visible %.3f"
              % (m, spell, macro, n_pred, n_gt, seen.mean()), flush=True)
        add(total["frames"], frame_tables(gt, pred, seen))
        add(total["spells"], spell_tables(gt, pred, seen))
        add(total["transitions"], transition_tables(gt, pred, seen))
    total["per_match"] = per_match
    total["mean_spell_f1"] = float(np.mean([v["spell_f1"] for v in per_match.values()]))
    total["mean_macro_f1"] = float(np.mean([v["macro_f1"] for v in per_match.values()]))
    total["atd_reparto_huecos"] = atd_gap_profile()
    n_all = total["frames"]["todo"]["frames"]
    total["idsse_reparto_huecos"] = {k: v["frames"] / float(n_all)
                                     for k, v in total["frames"].items() if k != "todo"}
    path = os.path.join(os.path.dirname(__file__), "diag.json")
    with open(path, "w") as handle:
        json.dump(total, handle, indent=1)

    print("\nMEAN spellF1 %.4f  macroF1 %.4f" % (total["mean_spell_f1"], total["mean_macro_f1"]))
    print("\n== FRAMES (sum over the 7 matches)")
    print("%-32s %8s %7s %7s | %8s %8s %8s | %6s" % (
        "bucket", "frames", "%tot", "correct", "team", "D missed", "D false", "%gt D"))
    for name, r in total["frames"].items():
        n = max(r["frames"], 1)
        print("%-32s %8d %6.1f%% %6.1f%% | %7.1f%% %7.1f%% %7.1f%% | %5.1f%%" % (
            name, r["frames"], 100.0 * r["frames"] / n_all, 100.0 * r["aciertos"] / n,
            100.0 * r["err_equipo"] / n, 100.0 * r["err_muerto_no_visto"] / n,
            100.0 * r["err_muerto_inventado"] / n, 100.0 * r["gt_D"] / n))
    for side, label in (("gt", "REAL POSSESSIONS: recovered at IoU>=0.5"),
                        ("pred", "PREDICTED POSSESSIONS: correct at IoU>=0.5")):
        print("\n== %s" % label)
        for group, rows in total["spells"][side].items():
            for name, (n, h) in rows.items():
                print("  %-24s %-9s n %5d  ok %5d  %5.1f%%" % (group, name, n, h,
                                                              100.0 * h / max(n, 1)))
    print("\n== REAL CHANGES found within +-1 s with the correct target state")
    for name in sorted(total["transitions"]):
        n, h = total["transitions"][name]
        print("  %-30s n %5d  ok %5d  %5.1f%%" % (name, n, h, 100.0 * h / max(n, 1)))
    print("\n== FRAME DISTRIBUTION BY GAP: real ATD (graded blocks) vs degraded IDSSE")
    for name, v in total["atd_reparto_huecos"].items():
        print("  %-32s ATD %5.1f%%   IDSSE %5.1f%%" % (
            name, 100 * v, 100 * total["idsse_reparto_huecos"].get(name, float("nan"))))


if __name__ == "__main__":
    main()
