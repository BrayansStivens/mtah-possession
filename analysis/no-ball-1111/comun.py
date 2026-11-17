"""Common utilities for the "sin balon" (ball-absent) experiments of Nov 11, 2026.

It is not submission code: nothing here is imported from src/. Everything is measured on
the 7 degraded IDSSE matches; no label from the challenge match goes into anything.
"""

import os

import numpy as np

from mtah import evaluate as ev
from mtah.config import DEFAULT
from mtah.paths import cache_dir
from mtah.training import boundaries as B
from mtah.training import convention, degrade, harness, idsse

MATCHES = sorted(idsse.MATCHES)
SEEDS = (0, 1, 2)
pm = ev.pm

# Submission config for the detector: what pipeline.py uses.
DISCOUNT = DEFAULT["boundary_discount_weight"]
SCALE = DEFAULT["boundary_switch_scale"]


def base(seed=0):
    """{match: prepared data} with the same cache as training/boundaries.py."""
    spec = harness.real_spec()
    return {m: B.prepare(m, spec, seed, DEFAULT) for m in MATCHES}


def oof_scores(data_by_seed):
    """{seed: {match: detector score}} out of sample.

    Same protocol as Simon's LOMO: the detector is trained on seed 0 of the other 6 matches
    and scores the three seeds of the held-out match.
    """
    path = os.path.join(cache_dir(), "sinbalon_oofscore.npz")
    if os.path.exists(path):
        blob = np.load(path)
        return {s: {m: blob["%s_%d" % (m, s)] for m in MATCHES} for s in SEEDS}
    out = {s: {} for s in SEEDS}
    for held in MATCHES:
        model = B.fit([data_by_seed[0][m] for m in MATCHES if m != held])
        for s in SEEDS:
            out[s][held] = B.boundary_score(model, data_by_seed[s][held]).astype(np.float32)
    np.savez_compressed(path, **{"%s_%d" % (m, s): out[s][m] for s in SEEDS for m in MATCHES})
    return out


def dirty(match, seed=0):
    """(clean, degraded, harmonized labels). Deterministic: same as the cache of base()."""
    clean, labels = idsse.load(match)
    labels, _ = convention.harmonise(labels)
    d = degrade.apply(clean, harness.real_spec(), rng=np.random.default_rng(seed))
    return clean, d, np.asarray(labels, dtype="<U1")


def gap_lengths(seen):
    """(n,) length of the ball-absent gap each frame belongs to (0 if the ball is seen)."""
    edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(seen), dtype=np.int32)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def no_ball_in_window(seen, half_width=50):
    """(n,) True where no frame within +-half_width has the ball: there the control is NaN."""
    c = np.concatenate(([0], np.cumsum(seen.astype(np.int64))))
    i = np.arange(len(seen))
    lo, hi = np.maximum(0, i - half_width), np.minimum(len(seen), i + half_width + 1)
    return (c[hi] - c[lo]) == 0


def score(pred_labels, gt_labels):
    r = ev.score(pred_labels, gt_codes=pm.label_codes(np.asarray(gt_labels, dtype="<U1")))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]
