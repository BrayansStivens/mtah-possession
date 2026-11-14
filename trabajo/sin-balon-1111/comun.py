"""Utilidades comunes de los experimentos "sin balon" del 11-11-2026.

No es codigo de entrega: nada de aqui se importa desde src/. Todo se mide en los 7
partidos del IDSSE degradados; ninguna etiqueta del partido del reto entra en nada.
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

# Config de la entrega para el detector: lo que usa pipeline.py.
DISCOUNT = DEFAULT["boundary_discount_weight"]
SCALE = DEFAULT["boundary_switch_scale"]


def base(seed=0):
    """{partido: datos preparados} con la misma cache que training/boundaries.py."""
    spec = harness.real_spec()
    return {m: B.prepare(m, spec, seed, DEFAULT) for m in MATCHES}


def oof_scores(data_by_seed):
    """{semilla: {partido: score del detector}} fuera de muestra.

    Mismo protocolo que el LOMO de Simon: el detector se entrena con la semilla 0 de los
    otros 6 partidos y puntua las tres semillas del partido de fuera.
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
    """(limpio, degradado, etiquetas armonizadas). Determinista: igual que la cache de base()."""
    clean, labels = idsse.load(match)
    labels, _ = convention.harmonise(labels)
    d = degrade.apply(clean, harness.real_spec(), rng=np.random.default_rng(seed))
    return clean, d, np.asarray(labels, dtype="<U1")


def gap_lengths(seen):
    """(n,) longitud del hueco sin balon al que pertenece cada frame (0 si se ve)."""
    edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(seen), dtype=np.int32)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def no_ball_in_window(seen, half_width=50):
    """(n,) True donde no hay ni un frame con balon a +-half_width: ahi el control es NaN."""
    c = np.concatenate(([0], np.cumsum(seen.astype(np.int64))))
    i = np.arange(len(seen))
    lo, hi = np.maximum(0, i - half_width), np.minimum(len(seen), i + half_width + 1)
    return (c[hi] - c[lo]) == 0


def score(pred_labels, gt_labels):
    r = ev.score(pred_labels, gt_codes=pm.label_codes(np.asarray(gt_labels, dtype="<U1")))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]
