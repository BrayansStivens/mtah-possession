"""Negativos duros: los errores con confianza del propio modelo sobre frames enteros.

    python -m mtah.vision.mine [--model ballnet]

Primera evaluacion apartada (9-11): el pico mas alto es el balon en el 49 % de los
frames y el balon esta entre los 5 picos en el 71 %, pero la precision no sube con el
umbral (~0,5 en todos): la red nunca vio un frame entero de senuelos, solo dos recortes
negativos por frame. Aqui se ejecuta sobre frames de ENTRENAMIENTO (nunca los apartados),
se toman los picos que no son el balon y se recortan como nuevos ejemplos.

Si el balon cae dentro del recorte de un senuelo, el recorte lleva el balon como etiqueta:
asi la red aprende a preferir el balon al senuelo de al lado, no a callarse.
"""

import argparse
import os
import time

import numpy as np

from ..io import atd, homography
from . import dataset, detect, frames

FRAME_STRIDE = 3                 # uno de cada 3 frames de entrenamiento
MIN_SCORE = 0.1
NOT_BALL_PX = 30                 # un pico a mas de esto del balon proyectado es un error
PER_FRAME = 2                    # los errores mas fuertes de cada frame
EDGE = 8                         # balon a menos de esto del borde del recorte: ambiguo


def crop(full, x0, y0, size=dataset.PATCH):
    """Recorte de la pila de frame entero, con ceros fuera de la imagen."""
    out = np.zeros((full.shape[0], size, size), dtype=full.dtype)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(full.shape[2], x0 + size), min(full.shape[1], y0 + size)
    if xe > xs and ye > ys:
        out[:, ys - y0:ye - y0, xs - x0:xe - x0] = full[:, ys:ye, xs:xe]
    return out


def mine(name="ballnet", seed=1):
    started = time.time()
    rng = np.random.default_rng(seed)
    model = detect.load_model(name)
    tracking = atd.load()
    H = homography.load()[0]
    pixel = dataset.projected_ball(tracking, H)
    wanted = dataset.training_frames(pixel)[::FRAME_STRIDE]
    capacity = PER_FRAME * len(wanted)
    directory = dataset.vision_dir()
    patches = np.lib.format.open_memmap(
        os.path.join(directory, "hard_patches.npy"), mode="w+", dtype=np.uint8,
        shape=(capacity, frames.CHANNELS, dataset.PATCH, dataset.PATCH))
    target = np.full((capacity, 2), -1.0, dtype=np.float32)
    frame_of = np.zeros(capacity, dtype=np.int32)
    count, with_ball, batch, batch_frames = 0, 0, [], []

    def flush():
        nonlocal count, with_ball
        found = detect.peaks(model, np.stack(batch))
        for full, t, candidates in zip(batch, batch_frames, found):
            ball = pixel[t]
            wrong = [p for p in candidates
                     if p[2] >= MIN_SCORE and np.hypot(*(p[:2] - ball)) > NOT_BALL_PX]
            for x, y, _ in wrong[:PER_FRAME]:
                u, v = rng.integers(EDGE, dataset.PATCH - EDGE, size=2)
                x0, y0 = int(round(x)) - u, int(round(y)) - v
                local = ball - (x0, y0)
                inside = ((local >= -EDGE) & (local < dataset.PATCH + EDGE)).all()
                clear = ((local >= EDGE) & (local < dataset.PATCH - EDGE)).all()
                if inside and not clear:
                    continue                      # balon pegado al borde: ni si ni no
                patches[count] = crop(full, x0, y0)
                if clear:
                    target[count] = local
                    with_ball += 1
                frame_of[count] = t
                count += 1
        batch.clear()
        batch_frames.clear()

    for done, (t, frames_t) in enumerate(frames.neighbourhoods(wanted)):
        batch.append(frames.stack(H, t, frames_t))
        batch_frames.append(t)
        if len(batch) == detect.BATCH:
            flush()
        if done % 2000 == 0:
            print("  %d/%d frames, %d recortes (%.1f min)" % (
                done, len(wanted), count, (time.time() - started) / 60), flush=True)
    if batch:
        flush()
    patches.flush()
    del patches
    np.savez(os.path.join(directory, "hard_meta.npz"), target=target[:count],
             frame=frame_of[:count], count=count)
    print("negativos duros: %d recortes de %d frames (%d con el balon dentro), %.1f min" % (
        count, len(wanted), with_ball, (time.time() - started) / 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ballnet")
    mine(parser.parse_args().model)
