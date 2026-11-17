"""Candidates fixed on the IDSSE, applied to the challenge match. Read-only.

The variants and their weights are the ones chosen by the nested selection in exp.py (random
gaps, seed 0); nothing is chosen here. For the real match we print label-free aggregates and
the **reading** of the public window (5 min, 17 possessions), which is not a criterion for
anything: one missed possession moves ~6 points.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
import modelos as M                                            # noqa: E402
from mtah import evaluate as ev                                # noqa: E402
from mtah import features as feature_set                       # noqa: E402
from mtah import submission                                    # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.constants import GRADED_BLOCKS, graded_mask          # noqa: E402
from mtah.decode import viterbi                                # noqa: E402
from mtah.features import tagging                              # noqa: E402
from mtah.io import atd, smart_tagging                         # noqa: E402
from mtah.models import boundary_detector, emissions           # noqa: E402
from mtah.paths import REPO_ROOT, cache_dir                    # noqa: E402

HERE = os.path.dirname(__file__)
CANDIDATES = {
    "entrega": (),
    "equipo (c sustituye, 3.0)": (("c_rep", 3.0),),
    "equipo (a suma, 2.0)": (("a", 2.0),),
    "muerto (b, 1.0, 0.5)": (("b", 1.0, 0.5),),
    "equipo c + muerto b": (("c_rep", 3.0), ("b", 1.0, 0.5)),
    "equipo a + muerto b": (("a", 2.0), ("b", 1.0, 0.5)),
}


def atd_logits(tracking, features):
    """Logits (a, b, c) on the ATD from models trained on the 7 IDSSE matches."""
    path = os.path.join(cache_dir(), "sinbalon_atd_logits.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    direction, _ = F.attack_direction(tracking, blocks=GRADED_BLOCKS)
    smoothed = F.team_smoothed(tracking, direction)
    team = np.stack([[[smoothed[(t, s, w)] for w in F.TEAM_WINDOWS] for s in F.TEAM_SIGNALS]
                     for t in (0, 1)]).astype(np.float32)
    dead, _ = F.dead_matrix(tracking, features, DEFAULT)
    ballteam, _, odd = F.ballteam_matrix(features, F.gap_context(tracking, features), DEFAULT)
    data = {"team": team, "dead": dead, "ballteam": ballteam, "odd": odd}
    train = [M.feat(m) for m in C.MATCHES]
    out = {k: M.predict(k, M.fit(k, train), data) for k in M.KINDS}
    np.savez_compressed(path, **out)
    return out


def spells(labels, graded):
    """(number of H/A possessions, mean duration in s, H/A/D split) within the graded blocks."""
    count, frames = 0, 0
    for a, b in GRADED_BLOCKS:
        w = labels[a:b]
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i, j in zip(bounds[:-1], bounds[1:]):
            if w[i] in ("H", "A"):
                count += 1
                frames += j - i
    share = {s: float((labels[graded] == s).mean()) for s in "HAD"}
    return count, frames / max(count, 1) / 25.0, share


def main():
    tracking = atd.load()
    n = tracking.n_frames
    graded = graded_mask(n)
    phases = smart_tagging.load()
    tags = tagging.team_tags(phases, n)
    weak, conflict = tagging.weak_labels(phases, n)
    usable = graded & ~conflict & (weak != "")
    features = feature_set.build_all(tracking, DEFAULT)
    hand_home = emissions.home_logit(features, DEFAULT)
    hand_dead = emissions.dead_logit(features, DEFAULT)
    tag_logit = emissions.team_tag_logit(tags, DEFAULT)
    logit = {k: v.astype(np.float64) for k, v in atd_logits(tracking, features).items()}
    transition = DEFAULT["boundary_switch_scale"] * viterbi.switch_costs(
        DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"], DEFAULT["switch_from_dead"])
    discount = boundary_detector.switch_discount(features, DEFAULT)
    delivered = np.asarray(ev.pm.read_label_csv(
        os.path.join(REPO_ROOT, "out", "submission.csv"))["label"], dtype="<U1")
    seen = tracking.ball_seen

    out = {}
    print("%-28s | public window (READING): spellF1 macroF1  pos | full match, label-free: pos  dur(s)   %%H   %%A   %%D | same as submission  seen  ball-absent | phase agreement" % "candidate")
    for name, variant in CANDIDATES.items():
        home, dead = hand_home.copy(), hand_dead.copy()
        for part in variant:
            if part[0] == "a":
                home = home + part[1] * logit["a"]
            elif part[0] == "c_rep":
                home = part[1] * logit["c"]
            elif part[0] == "b":
                dead = (1.0 - part[1]) * dead + part[1] * part[2] * logit["b"]
        cost = emissions.costs_from_logits(home + tag_logit, dead, DEFAULT)
        codes = viterbi.decode(cost, transition, switch_discount=discount)
        codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"])
        labels = submission.build(viterbi.to_labels(codes))
        if name == "entrega":
            same = bool((labels == delivered).all())
            print("   check: reproduces out/submission.csv frame by frame: %s" % same)
            assert same
        r = ev.score(labels)
        count, duration, share = spells(labels, graded)
        alive = usable & (labels != "D")
        agree = float((labels[alive] == weak[alive]).mean())
        row = {"spell_f1_public": r["spell_f1_mean"], "macro_f1_public": r["macro_f1"],
               "n_pred_public": r["n_pred_spells"], "n_gt_public": r["n_gt_spells"],
               "n_spells_match": count, "mean_duration_s": duration, "share": share,
               "same_as_delivery": float((labels[graded] == delivered[graded]).mean()),
               "same_seen": float((labels[graded & seen] == delivered[graded & seen]).mean()),
               "same_unseen": float((labels[graded & ~seen] == delivered[graded & ~seen]).mean()),
               "weak_agreement_alive": agree}
        out[name] = row
        print("%-28s |                             %.4f  %.4f  %2d/%d |                         %5d  %5.1f  %4.1f  %4.1f  %4.1f |           %5.1f%%  %5.1f%%  %5.1f%% |  %5.1f%%" % (
            name, row["spell_f1_public"], row["macro_f1_public"], row["n_pred_public"],
            row["n_gt_public"], count, duration, 100 * share["H"], 100 * share["A"],
            100 * share["D"], 100 * row["same_as_delivery"], 100 * row["same_seen"],
            100 * row["same_unseen"], 100 * agree), flush=True)
    with open(os.path.join(HERE, "real.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
