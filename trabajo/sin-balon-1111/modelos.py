"""New classifiers: ball-absent team (a), learned live/dead (b), team with context (c).

Everything is out of sample per match. `predict(held, pair)` returns the logits of one match
from models that saw neither that match nor `pair` (to choose weights in a nested way).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402

STRIDE = 4
KINDS = ("a", "b", "c")


def feat(match, seed=0):
    blob = np.load(os.path.join(cache_dir(), "sinbalon_feat_%s_s%d.npz" % (match, seed)))
    return {k: blob[k] for k in blob.files}


def team_matrix(team, perspective):
    """(n, f) from the array (2, signals, windows, n)."""
    p, q = perspective, 1 - perspective
    sums = [F.TEAM_SIGNALS.index(s) for s in F.SUM_SIGNALS]
    cols = []
    for w in range(team.shape[2]):
        cols += [team[p, :, w].T, team[q, :, w].T, (team[p, :, w] - team[q, :, w]).T,
                 (team[p, sums, w] + team[q, sums, w]).T]
    return np.concatenate(cols, axis=1)


def ballteam(data, perspective):
    m = data["ballteam"]
    if perspective == 0:
        return m
    return np.where(data["odd"][None, :], -m, m)


def team_xy(data, with_ball, rows):
    """Rows of the two perspectives in the live frames of `rows`."""
    labels = data["labels"][rows]
    alive = labels != "D"
    xs, ys = [], []
    for perspective, mine in ((0, "H"), (1, "A")):
        x = team_matrix(data["team"], perspective)[rows][alive]
        if with_ball:
            x = np.concatenate([x, ballteam(data, perspective)[rows][alive]], axis=1)
        xs.append(x)
        ys.append(labels[alive] == mine)
    return np.concatenate(xs), np.concatenate(ys)


def team_x(data, with_ball, perspective):
    x = team_matrix(data["team"], perspective)
    if with_ball:
        x = np.concatenate([x, ballteam(data, perspective)], axis=1)
    return x


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def fit(kind, train):
    """`train`: list of feat() dicts."""
    if kind == "b":
        x = np.concatenate([d["dead"][::STRIDE] for d in train])
        y = np.concatenate([d["labels"][::STRIDE] == "D" for d in train])
    else:
        parts = [team_xy(d, kind == "c", slice(None, None, STRIDE)) for d in train]
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
    return new_model().fit(x, y)


def predict(kind, model, data):
    """(n,) logit: dead for b, home (antisymmetric) for a and c."""
    if kind == "b":
        return model.decision_function(data["dead"]).astype(np.float32)
    home = model.decision_function(team_x(data, kind == "c", 0))
    away = model.decision_function(team_x(data, kind == "c", 1))
    return (0.5 * (home - away)).astype(np.float32)


def pred_path(seed=0):
    return os.path.join(cache_dir(), "sinbalon_pred_s%d.npz" % seed)


def build(seed=0, nested=True):
    """Saves {kind_held: logit} (model without held) and {kind_held_pair: logit}
    (model without held or pair)."""
    data = {m: feat(m, seed) for m in C.MATCHES}
    out = {}
    started = time.time()
    for kind in KINDS:
        for held in C.MATCHES:
            model = fit(kind, [data[m] for m in C.MATCHES if m != held])
            out["%s_%s" % (kind, held)] = predict(kind, model, data[held])
            print("%s held out %s  %.0fs" % (kind, held, time.time() - started), flush=True)
        if not nested:
            continue
        for i, first in enumerate(C.MATCHES):
            for second in C.MATCHES[i + 1:]:
                model = fit(kind, [data[m] for m in C.MATCHES if m not in (first, second)])
                out["%s_%s_%s" % (kind, first, second)] = predict(kind, model, data[first])
                out["%s_%s_%s" % (kind, second, first)] = predict(kind, model, data[second])
            print("%s pairs of %s  %.0fs" % (kind, first, time.time() - started), flush=True)
        np.savez_compressed(pred_path(seed), **out)
    np.savez_compressed(pred_path(seed), **out)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
