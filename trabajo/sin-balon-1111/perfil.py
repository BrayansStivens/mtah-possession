"""Perfil sin etiquetas del ATD: cuando desaparece el balon. Frente al IDSSE degradado.

Del ATD solo se usan tracking y mascara de bloques puntuados; se imprimen agregados.
"""

import json
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
from mtah import features as feature_set                       # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.constants import graded_mask                         # noqa: E402
from mtah.io import atd                                        # noqa: E402

EDGES = np.array([0.0, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.2, 99.0])


def profile(tracking, keep, labels=None):
    features = feature_set.build_all(tracking, DEFAULT)
    activity = np.asarray(features["activity_s"], dtype=np.float64)
    seen = tracking.ball_seen
    count = np.sum(~np.isnan(tracking.players_xy[:, :, 0]), axis=1)
    bins = np.clip(np.digitize(activity, EDGES) - 1, 0, len(EDGES) - 2)
    out = {"n": np.zeros(len(EDGES) - 1), "seen": np.zeros(len(EDGES) - 1),
           "dead": np.zeros(len(EDGES) - 1), "dead_seen": np.zeros(len(EDGES) - 1),
           "dead_unseen": np.zeros(len(EDGES) - 1)}
    for b in range(len(EDGES) - 1):
        sel = keep & (bins == b)
        out["n"][b] = sel.sum()
        out["seen"][b] = (sel & seen).sum()
        if labels is not None:
            dead = labels == "D"
            out["dead"][b] = (sel & dead).sum()
            out["dead_seen"][b] = (sel & dead & seen).sum()
            out["dead_unseen"][b] = (sel & dead & ~seen).sum()
    extra = {"players_seen": float(count[keep & seen].mean()),
             "players_unseen": float(count[keep & ~seen].mean()),
             "few_players_seen": float((count[keep & seen] < 16).mean()),
             "few_players_unseen": float((count[keep & ~seen] < 16).mean()),
             "activity_quartiles": [float(q) for q in np.nanpercentile(activity[keep], (25, 50, 75))]}
    return out, extra


def idsse_one(match):
    clean, dirty, labels = C.dirty(match, 0)
    return profile(dirty, np.ones(dirty.n_frames, bool), labels)


def main():
    tracking = atd.load()
    real, real_extra = profile(tracking, graded_mask(tracking.n_frames))
    with get_context("fork").Pool(7) as pool:
        parts = pool.map(idsse_one, C.MATCHES)
    proxy = {k: sum(p[0][k] for p in parts) for k in real}
    proxy_extra = {k: float(np.mean([p[1][k] for p in parts], axis=0)) if k != "activity_quartiles"
                   else [float(x) for x in np.mean([p[1][k] for p in parts], axis=0)]
                   for k in real_extra}
    p_dead = proxy["dead"] / np.maximum(proxy["n"], 1)
    print("actividad (m/s)   | ATD: %frames  balon visto | IDSSE degradado: %frames  balon visto  P(muerto)")
    for b in range(len(EDGES) - 1):
        print("  %.1f - %-4s      |      %5.1f%%      %5.1f%%   |                  %5.1f%%      %5.1f%%     %5.1f%%" % (
            EDGES[b], "%.1f" % EDGES[b + 1] if EDGES[b + 1] < 99 else "mas",
            100 * real["n"][b] / real["n"].sum(), 100 * real["seen"][b] / max(real["n"][b], 1),
            100 * proxy["n"][b] / proxy["n"].sum(), 100 * proxy["seen"][b] / max(proxy["n"][b], 1),
            100 * p_dead[b]))
    unseen = real["n"] - real["seen"]
    est_unseen = float((unseen * p_dead).sum() / unseen.sum())
    est_seen = float((real["seen"] * p_dead).sum() / real["seen"].sum())
    est_all = float((real["n"] * p_dead).sum() / real["n"].sum())
    print("cuartiles de actividad  ATD %s   IDSSE %s" % (
        ["%.2f" % q for q in real_extra["activity_quartiles"]],
        ["%.2f" % q for q in proxy_extra["activity_quartiles"]]))
    print("jugadores trackeados (media)  ATD visto %.1f / sin balon %.1f   IDSSE visto %.1f / sin balon %.1f" % (
        real_extra["players_seen"], real_extra["players_unseen"],
        proxy_extra["players_seen"], proxy_extra["players_unseen"]))
    print("frames con menos de 16 jugadores  ATD visto %.1f%% / sin balon %.1f%%   IDSSE %.1f%% / %.1f%%" % (
        100 * real_extra["few_players_seen"], 100 * real_extra["few_players_unseen"],
        100 * proxy_extra["few_players_seen"], 100 * proxy_extra["few_players_unseen"]))
    print("P(muerto) en el IDSSE: %.1f%%  (visto %.1f%%, sin balon %.1f%%)" % (
        100 * proxy["dead"].sum() / proxy["n"].sum(),
        100 * proxy["dead_seen"].sum() / proxy["seen"].sum(),
        100 * proxy["dead_unseen"].sum() / (proxy["n"].sum() - proxy["seen"].sum())))
    print("P(muerto) ESTIMADA en el ATD si P(muerto | actividad) transfiere: todo %.1f%%  visto %.1f%%  sin balon %.1f%%" % (
        100 * est_all, 100 * est_seen, 100 * est_unseen))
    with open(os.path.join(os.path.dirname(__file__), "perfil.json"), "w") as handle:
        json.dump({"edges": EDGES.tolist(), "atd": {k: v.tolist() for k, v in real.items()},
                   "idsse": {k: v.tolist() for k, v in proxy.items()},
                   "atd_extra": real_extra, "idsse_extra": proxy_extra,
                   "est_dead": {"all": est_all, "seen": est_seen, "unseen": est_unseen}}, handle, indent=1)


if __name__ == "__main__":
    main()
