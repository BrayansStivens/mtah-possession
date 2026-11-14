"""Deriva sin etiquetas: se parecen las senales de forma del ATD a las del IDSSE degradado?

El clasificador de equipo (a) se entrena con jugadores degradados quitando jugadores al
azar frame a frame. En el ATD real faltan mas jugadores justo cuando no se ve el balon.
Aqui se compara, por tramo (balon visto / sin balon), la distribucion de las senales de
forma y lo facil que es distinguir el ATD de un partido del IDSSE con un clasificador de
dominio. Referencia: lo facil que es distinguir un partido del IDSSE de los demas.
Del ATD solo se imprimen agregados.
"""

import json
import os
import sys
from multiprocessing import get_context

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
import modelos as M                                            # noqa: E402
from mtah.constants import GRADED_BLOCKS, graded_mask          # noqa: E402
from mtah.io import atd                                        # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402

HERE = os.path.dirname(__file__)
STRIDE = 10
WID, LEN, SPD = (F.TEAM_SIGNALS.index(s) for s in ("wid", "len", "spd"))
W100 = F.TEAM_WINDOWS.index(100)


def counts(tracking):
    """(n, 2) jugadores trackeados por equipo."""
    present = ~np.isnan(tracking.players_xy[:, :, 0])
    return np.stack([present[:, tracking.team_of_slot == t].sum(axis=1) for t in (0, 1)], axis=1)


def idsse_counts(match):
    return counts(C.dirty(match, 0)[1])


def summary(team, count, mask):
    wid = team[:, WID, W100][:, mask]
    return {"frames": int(mask.sum()),
            "jugadores": float(count[mask].sum(axis=1).mean()),
            "menos_de_16": float((count[mask].sum(axis=1) < 16).mean()),
            "equipo_menos_de_7": float((count[mask].min(axis=1) < 7).mean()),
            "anchura_m": float(np.nanmean(wid)),
            "dif_anchura_std_m": float(np.nanstd(wid[0] - wid[1])),
            "longitud_m": float(np.nanmean(team[:, LEN, W100][:, mask])),
            "velocidad": float(np.nanmean(team[:, SPD, W100][:, mask]))}


def matrix(team, mask, half, builder=None):
    """Filas de las dos perspectivas en `mask`, primera (0) o segunda (1) mitad de esos frames."""
    builder = builder or M.team_matrix
    index = np.flatnonzero(mask)
    index = index[: len(index) // 2] if half == 0 else index[len(index) // 2:]
    index = index[::STRIDE]
    return np.concatenate([builder(team, p)[index] for p in (0, 1)])


def domain_auc(target, target_mask, others, other_masks, builder=None):
    """AUC de distinguir `target` de otros partidos. Se entrena con la primera mitad del
    objetivo y la mitad de los otros partidos; se mide con la segunda mitad y el resto."""
    k = len(others) // 2
    def stack(half, group):
        pos = matrix(target, target_mask, half, builder)
        neg = np.concatenate([matrix(t, m, half, builder)
                              for t, m in zip(others[group], other_masks[group])])
        return np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    groups = (slice(0, k), slice(k, None))
    x, y = stack(0, groups[0])
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
                                           min_samples_leaf=100, random_state=0).fit(x, y)
    x, y = stack(1, groups[1])
    return float(roc_auc_score(y, model.decision_function(x)))


def main():
    tracking = atd.load()
    n = tracking.n_frames
    graded = graded_mask(n)
    direction, _ = F.attack_direction(tracking, blocks=GRADED_BLOCKS)
    smoothed = F.team_smoothed(tracking, direction)
    real_team = np.stack([[[smoothed[(t, s, w)] for w in F.TEAM_WINDOWS] for s in F.TEAM_SIGNALS]
                          for t in (0, 1)]).astype(np.float32)
    real_count = counts(tracking)
    real_seen = tracking.ball_seen

    with get_context("fork").Pool(7) as pool:
        proxy_count = dict(zip(C.MATCHES, pool.map(idsse_counts, C.MATCHES)))
    proxy_team = {m: M.feat(m, 0)["team"] for m in C.MATCHES}
    proxy_seen = {m: M.feat(m, 0)["ball_seen"] for m in C.MATCHES}
    realistic_seen = {m: np.load(os.path.join(cache_dir(), "sinbalon_realista_%s_s0.npz" % m))["ball_seen"]
                      for m in C.MATCHES}

    rows = {"ATD visto": summary(real_team, real_count, graded & real_seen),
            "ATD sin balon": summary(real_team, real_count, graded & ~real_seen)}
    for label, seen_by_match, keep in (("IDSSE visto (azar)", proxy_seen, True),
                                       ("IDSSE sin balon (azar)", proxy_seen, False),
                                       ("IDSSE visto (realista)", realistic_seen, True),
                                       ("IDSSE sin balon (realista)", realistic_seen, False)):
        parts = [summary(proxy_team[m], proxy_count[m], seen_by_match[m] == keep) for m in C.MATCHES]
        rows[label] = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
    print("%-28s %8s %10s %9s %10s %9s %12s %9s %9s" % (
        "tramo", "frames", "jugadores", "<16 jug", "equipo <7", "anchura", "std dif anch", "longitud", "veloc"))
    for name, r in rows.items():
        print("%-28s %8d %10.1f %8.1f%% %9.1f%% %8.1fm %11.2fm %8.1fm %9.2f" % (
            name, r["frames"], r["jugadores"], 100 * r["menos_de_16"], 100 * r["equipo_menos_de_7"],
            r["anchura_m"], r["dif_anchura_std_m"], r["longitud_m"], r["velocidad"]))

    out = {"rows": rows, "auc": {}}
    print("== clasificador de dominio sobre las 87 senales de equipo (0,5 = indistinguible)")
    for region, keep in (("visto", True), ("sin balon", False)):
        teams = [proxy_team[m] for m in C.MATCHES]
        masks = [realistic_seen[m] == keep for m in C.MATCHES]
        real = domain_auc(real_team, graded & (real_seen == keep), teams[:6], masks[:6])
        reference = []
        for i, m in enumerate(C.MATCHES):
            rest = [j for j in range(len(C.MATCHES)) if j != i]
            reference.append(domain_auc(teams[i], masks[i], [teams[j] for j in rest],
                                        [masks[j] for j in rest]))
        out["auc"][region] = {"atd_vs_idsse": real, "idsse_match_vs_rest": reference}
        print("  %-10s ATD contra IDSSE %.3f   |  un partido del IDSSE contra el resto: media %.3f  rango %.3f-%.3f" % (
            region, real, np.mean(reference), min(reference), max(reference)))
    with open(os.path.join(HERE, "deriva.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
