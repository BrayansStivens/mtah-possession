"""Transfer check on the challenge match, without labels.

The classifiers are trained on the 7 degraded IDSSE matches and applied to the ATD. They are
compared with Smart Tagging (Metrica's automatic phases, ~18 s): it is a diagnostic, not a
tuning criterion. Only aggregates are printed.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
import modelos as M                                            # noqa: E402
from mtah import features as feature_set                       # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.constants import GRADED_BLOCKS, graded_mask          # noqa: E402
from mtah.features import tagging                              # noqa: E402
from mtah.io import atd, smart_tagging                         # noqa: E402
from mtah.models import emissions                              # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    tracking = atd.load()
    phases = smart_tagging.load()
    n = tracking.n_frames
    graded = graded_mask(n)
    weak, conflict = tagging.weak_labels(phases, n)
    usable = graded & ~conflict & (weak != "")
    features = feature_set.build_all(tracking, DEFAULT)
    hand_home = emissions.home_logit(features, DEFAULT)
    hand_dead = emissions.dead_logit(features, DEFAULT)

    direction, report = F.attack_direction(tracking, blocks=GRADED_BLOCKS)
    smoothed = F.team_smoothed(tracking, direction)
    team = np.stack([[[smoothed[(t, s, w)] for w in F.TEAM_WINDOWS] for s in F.TEAM_SIGNALS]
                     for t in (0, 1)]).astype(np.float32)
    dead, _ = F.dead_matrix(tracking, features, DEFAULT)
    context = F.gap_context(tracking, features)
    ballteam, _, odd = F.ballteam_matrix(features, context, DEFAULT)
    data = {"team": team, "dead": dead, "ballteam": ballteam, "odd": odd}

    train = [M.feat(m) for m in C.MATCHES]
    logit = {k: M.predict(k, M.fit(k, train), data).astype(np.float64) for k in M.KINDS}

    seen = tracking.ball_seen
    blind = C.no_ball_in_window(seen)
    regions = {"todo": np.ones(n, bool), "visto": seen, "sin balon": ~seen, "sin balon a +-50": blind}
    out = {"direction": report, "team": {}, "dead": {}}
    print("attack direction per half:", [(r["direction"], round(r["agreement"], 3)) for r in report])
    print("== TEAM against weak Smart Tagging labels (graded frames with a phase and no conflict)")
    print("%-18s %8s | AUC %8s %8s %8s %8s | agreement %8s %8s %8s" % (
        "tramo", "frames", "entrega", "(a)", "ent+(a)", "(c)", "entrega", "(a)", "(c)"))
    for name, mask in regions.items():
        sel = usable & mask
        y = weak[sel] == "H"
        cols = [hand_home[sel], logit["a"][sel], hand_home[sel] + logit["a"][sel], logit["c"][sel]]
        aucs = [float(roc_auc_score(y, c)) for c in cols]
        acc = [float(((c > 0) == y).mean()) for c in (cols[0], cols[1], cols[3])]
        out["team"][name] = {"n": int(sel.sum()), "auc": aucs, "agreement": acc,
                             "majority": float(max(y.mean(), 1 - y.mean()))}
        print("%-18s %8d |     %8.4f %8.4f %8.4f %8.4f |           %8.4f %8.4f %8.4f" % (
            name, sel.sum(), *aucs, *acc))

    setpiece = tagging.set_piece_window(phases, n)
    print("== DEAD: Smart Tagging set-piece windows (~5 s around the restart)")
    g = graded
    print("   graded frames in set-piece window: %.1f%%" % (100 * setpiece[g].mean()))
    print("   ball-absent inside set-piece window %.1f%%   outside %.1f%%" % (
        100 * (~seen)[g & setpiece].mean(), 100 * (~seen)[g & ~setpiece].mean()))
    out["dead"]["unseen_in_setpiece"] = float((~seen)[g & setpiece].mean())
    out["dead"]["unseen_outside"] = float((~seen)[g & ~setpiece].mean())
    print("%-18s %8s | AUC set-piece window: %8s %8s | P(dead) average: %8s %8s" % (
        "tramo", "frames", "entrega", "(b)", "entrega", "(b)"))
    for name, mask in regions.items():
        sel = g & mask
        aucs = [float(roc_auc_score(setpiece[sel], hand_dead[sel])),
                float(roc_auc_score(setpiece[sel], logit["b"][sel]))]
        means = [float(sigmoid(hand_dead[sel]).mean()), float(sigmoid(logit["b"][sel]).mean())]
        out["dead"][name] = {"n": int(sel.sum()), "auc_setpiece": aucs, "mean_p_dead": means}
        print("%-18s %8d |                      %8.4f %8.4f |                  %8.4f %8.4f" % (
            name, sel.sum(), *aucs, *means))
    with open(os.path.join(os.path.dirname(__file__), "atd_check.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
