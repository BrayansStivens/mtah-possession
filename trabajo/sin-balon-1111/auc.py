"""AUC fuera de muestra de las senales nuevas frente a las de la entrega, por visibilidad."""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import modelos as M                                            # noqa: E402


def main(seed=0):
    base = C.base(seed)
    pred = np.load(M.pred_path(seed))
    rows = {"team": {}, "dead": {}}
    parts = {k: [] for k in ("labels", "seen", "blind", "hand_home", "hand_dead", "a", "b", "c")}
    for m in C.MATCHES:
        b = base[m]
        parts["labels"].append(b["labels"]); parts["seen"].append(b["ball_seen"])
        parts["blind"].append(C.no_ball_in_window(b["ball_seen"]))
        parts["hand_home"].append(b["home_logit"]); parts["hand_dead"].append(b["dead_logit"])
        for k in M.KINDS:
            parts[k].append(pred["%s_%s" % (k, m)])
    z = {k: np.concatenate(v) for k, v in parts.items()}
    alive = z["labels"] != "D"
    regions = {"todo": np.ones(len(alive), bool), "visto": z["seen"], "sin balon": ~z["seen"],
               "sin balon a +-50": z["blind"]}
    print("== EQUIPO (frames vivos, objetivo: local). AUC")
    print("%-18s %8s %8s %8s %8s %8s" % ("tramo", "frames", "entrega", "(a)", "ent+(a)", "(c)"))
    for name, mask in regions.items():
        sel = mask & alive
        y = z["labels"][sel] == "H"
        vals = [roc_auc_score(y, z["hand_home"][sel]), roc_auc_score(y, z["a"][sel]),
                roc_auc_score(y, z["hand_home"][sel] + z["a"][sel]), roc_auc_score(y, z["c"][sel])]
        rows["team"][name] = {"n": int(sel.sum()), "auc": vals}
        print("%-18s %8d %8.4f %8.4f %8.4f %8.4f" % (name, sel.sum(), *vals))
    print("== VIVO/MUERTO (todos los frames, objetivo: muerto). AUC")
    print("%-18s %8s %8s %8s" % ("tramo", "frames", "entrega", "(b)"))
    for name, mask in regions.items():
        y = z["labels"][mask] == "D"
        vals = [roc_auc_score(y, z["hand_dead"][mask]), roc_auc_score(y, z["b"][mask])]
        rows["dead"][name] = {"n": int(mask.sum()), "auc": vals}
        print("%-18s %8d %8.4f %8.4f" % (name, mask.sum(), *vals))
    print("== por partido, sin balon a +-50: AUC equipo entrega / (a) / (c);  muerto entrega / (b)")
    for m in C.MATCHES:
        b = base[m]
        blind = C.no_ball_in_window(b["ball_seen"]); al = b["labels"] != "D"
        y = b["labels"][blind & al] == "H"; yd = b["labels"][blind] == "D"
        print("  %s  %.4f %.4f %.4f   %.4f %.4f" % (
            m, roc_auc_score(y, b["home_logit"][blind & al]),
            roc_auc_score(y, pred["a_" + m][blind & al]), roc_auc_score(y, pred["c_" + m][blind & al]),
            roc_auc_score(yd, b["dead_logit"][blind]), roc_auc_score(yd, pred["b_" + m][blind])))
    with open(os.path.join(os.path.dirname(__file__), "auc_s%d.json" % seed), "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
