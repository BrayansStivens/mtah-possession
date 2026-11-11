"""Acuerdo con el ATD en los bloques apartados (nivel 1 del plan: el mas limpio y el mas
optimista -- son justo los frames que el tracker profesional si resolvio).

    python -m mtah.vision.evaluate [peaks|tracks] [--model ballnet]

Para cada frame apartado con balon del ATD dentro de la imagen: el pico mas alto del
detector, su probabilidad y su distancia a la proyeccion. Con un umbral sobre la
probabilidad, precision = aciertos / detecciones y recall = aciertos / frames. Un acierto
es caer a menos de RADII px (30 px ~ 1 m: lo que necesita la posesion).

El criterio de abandono del plan: 50 % de recall al 90 % de precision.
"""

import argparse
import json
import os

import numpy as np

from ..constants import PITCH, graded_mask
from ..io import atd, homography
from . import dataset

RADII = (15, 30, 60)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def curve(score, distance, radius):
    rows = []
    for threshold in THRESHOLDS:
        detected = score >= threshold
        hits = detected & (distance < radius)
        rows.append({"threshold": threshold,
                     "precision": float(hits.sum() / max(1, detected.sum())),
                     "recall": float(hits.mean())})
    return rows


def recall_at_precision(rows, precision=0.9):
    ok = [r["recall"] for r in rows if r["precision"] >= precision]
    return max(ok) if ok else 0.0


def evaluate(name="ballnet"):
    directory = dataset.vision_dir()
    blob = np.load(os.path.join(directory, "peaks_holdout_%s.npz" % name))
    frame, peaks = blob["frame"], blob["peaks"]
    tracking = atd.load()
    H = homography.load()[0]
    pixel = dataset.projected_ball(tracking, H)[frame]
    has_ball = dataset.inside_image(np.nan_to_num(pixel, nan=-1e9), 0)
    has_ball &= dataset.evaluation_mask(tracking.n_frames)[frame]

    top = peaks[:, 0]
    distance = np.hypot(top[:, 0] - pixel[:, 0], top[:, 1] - pixel[:, 1])
    # el mejor de los K picos, para saber si el balon esta en la lista aunque no gane
    any_distance = np.nanmin(np.hypot(peaks[..., 0] - pixel[:, None, 0],
                                      peaks[..., 1] - pixel[:, None, 1]), axis=1)
    scored = has_ball
    report = {"model": name, "frames": int(len(frame)), "with_atd_ball": int(scored.sum()),
              "top1_distance_px_p50": float(np.median(distance[scored])),
              "in_top_k_within_30px": float((any_distance[scored] < 30).mean()),
              "curves": {}}
    print("frames apartados %d, en juego con balon del ATD %d" % (len(frame), scored.sum()))
    print("distancia del pico 1 a la proyeccion: p50 %.1f px; balon entre los %d picos "
          "(<30 px): %.1f%%" % (report["top1_distance_px_p50"], peaks.shape[1],
                                100 * report["in_top_k_within_30px"]))
    for radius in RADII:
        rows = curve(top[scored, 2], distance[scored], radius)
        report["curves"][radius] = rows
        print("radio %d px: recall a precision 90%% = %.1f%%" % (
            radius, 100 * recall_at_precision(rows)))
        for row in rows:
            print("   umbral %.2f  precision %.3f  recall %.3f" % (
                row["threshold"], row["precision"], row["recall"]))
    missing = ~tracking.ball_seen[frame] & graded_mask(tracking.n_frames)[frame]
    report["atd_missing_frames"] = int(missing.sum())
    report["atd_missing_detected_share"] = {
        str(th): float((top[missing, 2] >= th).mean()) for th in THRESHOLDS}
    print("frames apartados SIN balon del ATD: %d; con deteccion >= umbral:" % missing.sum())
    print("   " + "  ".join("%.1f:%.0f%%" % (th, 100 * (top[missing, 2] >= th).mean())
                            for th in THRESHOLDS))
    with open(os.path.join(directory, "eval_%s.json" % name), "w") as handle:
        json.dump(report, handle, indent=1)
    return report


FIELD_RADII_M = (1.0, 2.0, 4.0)


def evaluate_tracks(name="ballnet"):
    """Trayectorias enlazadas (track.py) contra el ATD en los bloques apartados.

    Se enlaza usando SOLO los picos del detector, como si el ATD no existiera, y se
    compara en metros donde el ATD si tiene balon. Despues, cuanto rellena de los frames
    puntuados en los que el ATD no tiene balon (lo que de verdad se gana).
    """
    from . import track
    directory = dataset.vision_dir()
    blob = np.load(os.path.join(directory, "peaks_fill_%s.npz" % name))
    tracking = atd.load()
    H = homography.load()[0]
    n = tracking.n_frames
    filled = track.fill(blob["frame"], blob["peaks"], H, n)
    has_fill = ~np.isnan(filled[:, 0])

    truth = tracking.ball_xy.astype(np.float64) * np.asarray(PITCH)
    holdout = dataset.holdout_mask(n) & tracking.ball_seen & dataset.evaluation_mask(n)
    distance = np.hypot(*(filled[holdout] - truth[holdout]).T)
    covered = has_fill[holdout]
    report = {"model": name, "holdout_atd_frames": int(holdout.sum()),
              "holdout_coverage": float(covered.mean()), "precision": {}, "recall": {}}
    print("bloques apartados con balon del ATD: %d frames; la trayectoria cubre %.1f%%" % (
        holdout.sum(), 100 * covered.mean()))
    for radius in FIELD_RADII_M:
        hit = covered & (distance < radius)
        report["precision"][radius] = float(hit.sum() / max(1, covered.sum()))
        report["recall"][radius] = float(hit.mean())
        print("   a menos de %.0f m: precision %.1f%%, recall %.1f%%" % (
            radius, 100 * report["precision"][radius], 100 * report["recall"][radius]))
    missing = graded_mask(n) & ~tracking.ball_seen
    report["missing_graded_frames"] = int(missing.sum())
    report["missing_filled_share"] = float(has_fill[missing].mean())
    seen_after = (tracking.ball_seen | has_fill)[graded_mask(n)].mean()
    report["visibility_graded_before"] = float(tracking.ball_seen[graded_mask(n)].mean())
    report["visibility_graded_after"] = float(seen_after)
    print("frames puntuados sin balon del ATD: %d; rellenados %.1f%%" % (
        missing.sum(), 100 * report["missing_filled_share"]))
    print("visibilidad en frames puntuados: %.1f%% -> %.1f%%" % (
        100 * report["visibility_graded_before"], 100 * seen_after))
    with open(os.path.join(directory, "eval_tracks_%s.json" % name), "w") as handle:
        json.dump(report, handle, indent=1)
    return report, filled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", choices=("peaks", "tracks"), default="peaks")
    parser.add_argument("--model", default="ballnet")
    args = parser.parse_args()
    (evaluate if args.what == "peaks" else evaluate_tracks)(args.model)
