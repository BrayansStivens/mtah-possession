"""Segunda etapa: re-puntuar los picos del detector con contexto y enlazarlos en el tiempo.

    python -m mtah.vision.rescore [--model ballnet_r2]

Lo que el detector no ve y aqui si esta:
  - la camara sigue al balon (distancia al centro de la imagen);
  - los jugadores, del ATD, tambien donde el balon falta (distancia, gente a 5 m, centroide);
  - continuidad: si hay un candidato cerca en t-5..t+5;
  - persistencia: en cuantos bloques de 30 s del partido hay detecciones en esa misma
    celda del campo o de la imagen. El punto central del campo explico el 14 % de los
    errores confiados del 9-11: una marca blanca del tamano del balon que esta en todos.

Un clasificador (HistGradientBoosting) aprende "este candidato es el balon (< 2 m del
ATD)" en los bloques apartados, con ajuste cruzado entre sus dos mitades: cada mitad se
puntua con el modelo de la otra, y los frames sin ATD con la media de los dos. Despues,
un Viterbi por tramo contiguo elige un candidato por frame (probabilidad contra salto), y
un umbral sobre la probabilidad media en +-WINDOW frames decide que se rellena. El umbral
se elige en una mitad y se mide en la otra.
"""

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..constants import PITCH, graded_mask
from ..io import atd, homography
from . import dataset, track

CORRECT_M = 2.0
LAGS = (-5, -2, -1, 1, 2, 5)
PERSISTENCE_BLOCK_FRAMES = 750
PERSISTENCE_MIN_SCORE = 0.2
FREE_JUMP_M = 1.0                # salto entre frames sin coste (balon rapido ~1,6 m/frame)
JUMP_COST_PER_M = 3.0
WINDOW = 12
TARGET_PRECISION = 0.9


def holdout_halves(n_frames):
    """(n_frames,) 0/1 para cada mitad de los bloques apartados, -1 fuera."""
    edges = np.linspace(0, n_frames, dataset.N_BLOCKS + 1).astype(int)
    half = np.full(n_frames, -1)
    for i, block in enumerate(range(2, dataset.N_BLOCKS, dataset.HOLDOUT_EVERY)):
        half[edges[block]:edges[block + 1]] = i % 2
    return half


def _persistence(frames, scores, coords, cell):
    n, k = scores.shape
    key = np.floor(coords / cell).astype(np.int64)
    key = key[..., 0] * 100003 + key[..., 1]
    block = np.repeat((frames // PERSISTENCE_BLOCK_FRAMES)[:, None], k, 1)
    strong = scores >= PERSISTENCE_MIN_SCORE
    pairs = np.unique(np.stack([key[strong], block[strong]], 1), axis=0)
    keys, counts = np.unique(pairs[:, 0], return_counts=True)
    lookup = dict(zip(keys.tolist(), counts.tolist()))
    own = set(map(tuple, pairs.tolist()))
    out = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            pair = (int(key[i, j]), int(block[i, j]))
            out[i, j] = lookup.get(pair[0], 0) - (pair in own)
    return out


def features(frames, peaks, field_xy, tracking):
    """(n, K, F) features por candidato."""
    n, k = peaks.shape[:2]
    scores = peaks[..., 2].astype(np.float64)
    players = tracking.players_xy[frames].astype(np.float64) * np.asarray(PITCH)
    distance = np.hypot(field_xy[:, :, None, 0] - players[:, None, :, 0],
                        field_xy[:, :, None, 1] - players[:, None, :, 1])
    with np.errstate(all="ignore"):
        nearest = np.nanmin(distance, axis=2)
        centroid = np.nanmean(players, axis=1)
    columns = [scores, np.broadcast_to(np.arange(k), (n, k)),
               scores / np.maximum(scores[:, :1], 1e-3),
               np.abs(peaks[..., 0] - 960) / 960, peaks[..., 1] / 1080,
               nearest, np.nansum(distance < 5, axis=2),
               np.hypot(*(field_xy - centroid[:, None]).transpose(2, 0, 1)),
               _persistence(frames, scores, field_xy, 1.0),
               _persistence(frames, scores, peaks[..., :2].astype(np.float64), 20.0)]
    row_of = {f: i for i, f in enumerate(frames)}
    for lag in LAGS:
        near, near_score = np.full((n, k), 50.0), np.zeros((n, k))
        for i, f in enumerate(frames):
            j = row_of.get(f + lag)
            if j is None:
                continue
            d = np.hypot(*(field_xy[i][:, None] - field_xy[j][None]).transpose(2, 0, 1))
            nearest_candidate = d.argmin(1)
            near[i] = np.minimum(d[np.arange(k), nearest_candidate], 50.0)
            near_score[i] = scores[j, nearest_candidate]
        columns += [near, near_score]
    return np.nan_to_num(np.stack(columns, -1), nan=99.0)


def contiguous_runs(frames):
    return np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 1) + 1)


def viterbi(probability, field_xy):
    """Candidato elegido por frame en un tramo contiguo."""
    n, k = probability.shape
    emission = -np.log(np.clip(probability, 1e-3, 1.0))
    cost = emission[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        jump = np.hypot(*(field_xy[i][None] - field_xy[i - 1][:, None]).transpose(2, 0, 1))
        total = cost[:, None] + JUMP_COST_PER_M * np.maximum(0.0, jump - FREE_JUMP_M)
        back[i] = total.argmin(0)
        cost = total.min(0) + emission[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = cost.argmin()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def link(frames, probability, field_xy):
    """(posicion elegida (n, 2) en m, confianza (n,) = probabilidad media en +-WINDOW)."""
    chosen = np.full((len(frames), 2), np.nan)
    confidence = np.zeros(len(frames))
    kernel = np.ones(2 * WINDOW + 1) / (2 * WINDOW + 1)
    for rows in contiguous_runs(frames):
        path = viterbi(probability[rows], field_xy[rows])
        chosen[rows] = field_xy[rows, path]
        picked = probability[rows, path]
        confidence[rows] = np.convolve(np.pad(picked, WINDOW, mode="edge"), kernel, "valid")
    return chosen, confidence


def threshold_for(confidence, error_m, target=TARGET_PRECISION):
    """El umbral mas bajo con precision >= target (o None)."""
    for threshold in np.arange(0.05, 1.0, 0.01):
        keep = confidence >= threshold
        if keep.sum() >= 50 and (error_m[keep] < CORRECT_M).mean() >= target:
            return float(threshold)
    return None


def run(name="ballnet_r2"):
    directory = dataset.vision_dir()
    blob = np.load(os.path.join(directory, "peaks_fill_%s.npz" % name))
    order = np.argsort(blob["frame"])
    frames, peaks = blob["frame"][order], blob["peaks"][order].astype(np.float64)
    tracking = atd.load()
    H = homography.load()[0]
    field_xy = np.stack([track.to_field(H, f, p[:, :2]) for f, p in zip(frames, peaks)])
    X = features(frames, peaks, field_xy, tracking)
    n, k, width = X.shape
    truth = tracking.ball_xy[frames].astype(np.float64) * np.asarray(PITCH)
    error = np.hypot(*(field_xy - truth[:, None]).transpose(2, 0, 1))
    label = error < CORRECT_M
    half = holdout_halves(tracking.n_frames)[frames]
    has_truth = tracking.ball_seen[frames] & dataset.evaluation_mask(tracking.n_frames)[frames]

    probability = np.zeros((n, k))
    models = []
    for h in (0, 1):
        fit = (half == 1 - h) & has_truth
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               random_state=0)
        model.fit(X[fit].reshape(-1, width), label[fit].reshape(-1))
        models.append(model)
        rows = half == h
        probability[rows] = model.predict_proba(X[rows].reshape(-1, width))[:, 1].reshape(-1, k)
    rest = half < 0
    probability[rest] = np.mean([m.predict_proba(X[rest].reshape(-1, width))[:, 1]
                                 for m in models], axis=0).reshape(-1, k)

    chosen, confidence = link(frames, probability, field_xy)
    chosen_error = np.hypot(*(chosen - truth).T)
    report = {"model": name, "halves": {}}
    thresholds = []
    for h in (0, 1):
        tune = (half == 1 - h) & has_truth
        measure = (half == h) & has_truth
        threshold = threshold_for(confidence[tune], chosen_error[tune])
        thresholds.append(threshold)
        row = {"threshold": threshold, "frames": int(measure.sum()),
               "path_correct_all": float((chosen_error[measure] < CORRECT_M).mean())}
        if threshold is not None:
            keep = measure & (confidence >= threshold)
            row.update({"coverage": float(keep.sum() / measure.sum()),
                        "precision_1m": float((chosen_error[keep] < 1).mean()),
                        "precision_2m": float((chosen_error[keep] < 2).mean()),
                        "precision_4m": float((chosen_error[keep] < 4).mean())})
        report["halves"][h] = row
        print("mitad %d: camino correcto (<2 m) en %.1f%% de frames; umbral elegido en la otra"
              " mitad: %s" % (h, 100 * row["path_correct_all"], threshold))
        if threshold is not None:
            print("   cobertura %.1f%%, precision 1 m %.1f%%, 2 m %.1f%%, 4 m %.1f%%" % (
                100 * row["coverage"], 100 * row["precision_1m"], 100 * row["precision_2m"],
                100 * row["precision_4m"]))
    missing = dataset.evaluation_mask(tracking.n_frames)[frames] & ~tracking.ball_seen[frames]
    if all(t is not None for t in thresholds):
        threshold = float(np.mean(thresholds))
        filled = missing & (confidence >= threshold)
        report["fill_threshold"] = threshold
        report["missing_graded_frames"] = int(missing.sum())
        report["missing_filled_share"] = float(filled.sum() / missing.sum())
        graded = graded_mask(tracking.n_frames)
        before = tracking.ball_seen[graded].mean()
        report["visibility_graded"] = [float(before),
                                       float(before + filled.sum() / graded.sum())]
        print("frames puntuados sin balon del ATD: %d; rellenados %.1f%% (umbral %.2f); "
              "visibilidad %.1f%% -> %.1f%%" % (
                  missing.sum(), 100 * report["missing_filled_share"], threshold,
                  100 * report["visibility_graded"][0], 100 * report["visibility_graded"][1]))
        ball = np.full((tracking.n_frames, 2), np.nan)
        ball[frames[filled]] = chosen[filled]
        np.save(os.path.join(directory, "filled_ball_%s.npy" % name), ball)
    with open(os.path.join(directory, "rescore_%s.json" % name), "w") as handle:
        json.dump(report, handle, indent=1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ballnet_r2")
    run(parser.parse_args().model)
