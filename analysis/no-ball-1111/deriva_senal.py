"""Which team signals drift between the ATD and the degraded IDSSE. No labels, aggregates.

Per signal (its columns from the three windows, both perspectives): AUC of an ATD-versus-IDSSE
domain classifier, and the reference of one IDSSE match against the rest.
Also saves the ATD team array in the cache so it is not recomputed.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import deriva as D                                             # noqa: E402
import feats as F                                              # noqa: E402
import modelos as M                                            # noqa: E402
from mtah.constants import GRADED_BLOCKS, graded_mask          # noqa: E402
from mtah.io import atd                                        # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402

HERE = os.path.dirname(__file__)


def atd_team():
    path = os.path.join(cache_dir(), "sinbalon_atd_team.npz")
    if os.path.exists(path):
        blob = np.load(path)
        return blob["team"], blob["seen"], blob["graded"]
    tracking = atd.load()
    direction, _ = F.attack_direction(tracking, blocks=GRADED_BLOCKS)
    smoothed = F.team_smoothed(tracking, direction)
    team = np.stack([[[smoothed[(t, s, w)] for w in F.TEAM_WINDOWS] for s in F.TEAM_SIGNALS]
                     for t in (0, 1)]).astype(np.float32)
    graded = graded_mask(tracking.n_frames)
    np.savez_compressed(path, team=team, seen=tracking.ball_seen, graded=graded)
    return team, tracking.ball_seen, graded


def one_signal(team, perspective):
    """(n, 12): the own signal, the opponent's, the difference and the sum, in the three windows."""
    p, q = team[perspective, 0], team[1 - perspective, 0]
    return np.concatenate([p.T, q.T, (p - q).T, (p + q).T], axis=1)


def main():
    real_team, real_seen, graded = atd_team()
    teams = [M.feat(m, 0)["team"] for m in C.MATCHES]
    seens = [np.load(os.path.join(cache_dir(), "sinbalon_realista_%s_s0.npz" % m))["ball_seen"]
             for m in C.MATCHES]
    out = {}
    print("%-6s | seen:  ATD-IDSSE  reference  | no ball:   ATD-IDSSE  reference" % "signal")
    for i, signal in enumerate(F.TEAM_SIGNALS):
        row = {}
        for region, keep in (("visto", True), ("sin balon", False)):
            sub = [t[:, i:i + 1] for t in teams]
            masks = [s == keep for s in seens]
            real = D.domain_auc(real_team[:, i:i + 1], graded & (real_seen == keep), sub[:6], masks[:6],
                                one_signal)
            reference = [D.domain_auc(sub[j], masks[j], [sub[k] for k in range(7) if k != j],
                                      [masks[k] for k in range(7) if k != j], one_signal)
                         for j in (0, 3, 6)]
            row[region] = {"atd": real, "ref": float(np.mean(reference))}
        out[signal] = row
        print("%-6s |        %.3f       %.3f    |            %.3f       %.3f" % (
            signal, row["visto"]["atd"], row["visto"]["ref"],
            row["sin balon"]["atd"], row["sin balon"]["ref"]), flush=True)
    with open(os.path.join(HERE, "deriva_senal.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
