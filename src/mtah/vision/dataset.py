"""Recortes de entrenamiento con etiquetas gratis: el balon del ATD proyectado al pixel.

Por cada frame de entrenamiento salen tres recortes de PATCH x PATCH:
  - positivo: el balon proyectado dentro, en una posicion aleatoria (no siempre al centro);
  - negativo al azar: una region de la imagen lejos del balon;
  - negativo duro: centrado en un jugador lejos del balon (medias blancas, camisetas,
    el Athletic va de blanco).

La etiqueta es ruidosa: medido sobre 2.400 frames, la mancha blanca mas cercana a la
proyeccion esta a 12 px de mediana (p90 40 px), sin desfase temporal (el minimo de
distancia esta en k=0) y con un sesgo vertical de ~5 px (el centro del balon esta por
encima del suelo que mapea la homografia). Para decidir posesion sobra: 1 m son ~30 px.

Validacion por bloques temporales: los frames de los bloques de HOLDOUT no entran nunca
en el entrenamiento, ni sus vecinos a menos de HOLDOUT_MARGIN.
"""

import json
import os
import time

import numpy as np

from ..constants import GRADED_BLOCKS, KICKOFF_2ND, N_FRAMES, graded_mask
from ..io import atd, homography
from . import frames

PATCH = 96
STRIDE = 4                      # un frame de entrenamiento de cada 4 con balon
BALL_JITTER = 16                # el balon cae en [BALL_JITTER, PATCH - BALL_JITTER)
NEGATIVE_CLEARANCE_PX = 110     # ningun negativo a menos de esto del balon proyectado
N_BLOCKS = 40                   # ~200 s cada uno
HOLDOUT_EVERY = 5               # bloque de validacion: 2, 7, 12, ... (20 %)
HOLDOUT_MARGIN = 50
KINDS = {"ball": 0, "random": 1, "player": 2}


def vision_dir():
    """Fuera de OneDrive: son gigas de recortes del partido y no se sincronizan."""
    path = os.environ.get("MTAH_VISION_DIR") or os.path.join("out", "cache", "vision")
    os.makedirs(path, exist_ok=True)
    return path


def holdout_mask(n_frames=N_FRAMES, margin=0):
    edges = np.linspace(0, n_frames, N_BLOCKS + 1).astype(int)
    mask = np.zeros(n_frames, dtype=bool)
    for block in range(2, N_BLOCKS, HOLDOUT_EVERY):
        mask[max(0, edges[block] - margin):edges[block + 1] + margin] = True
    return mask


def evaluation_mask(n_frames=N_FRAMES):
    """Frames en los que se mide el detector: puntuados y en juego.

    Fuera el descanso y los 2.934 frames puntuados antes del saque de la segunda parte
    (balon parado segun el FAQ): con varios balones en el campo durante el calentamiento,
    el ATD sigue uno y el detector ve otro (top-1 correcto 14-19 % ahi, 68 % en juego). Ahi
    la posesion es D de todos modos. Decidido el 9-11 despues de ver ese corte por bloques.
    """
    mask = graded_mask(n_frames)
    mask[GRADED_BLOCKS[1][0]:KICKOFF_2ND] = False
    return mask


def projected_ball(tracking, H):
    """(n, 2) pixel del balon del ATD en cada frame; NaN donde no hay balon."""
    out = np.full((tracking.n_frames, 2), np.nan)
    seen = np.flatnonzero(tracking.ball_seen)
    xy = tracking.ball_xy[seen].astype(np.float64)
    h = H[seen]
    denominator = h[:, 2, 0] * xy[:, 0] + h[:, 2, 1] * xy[:, 1] + h[:, 2, 2]
    out[seen, 0] = (h[:, 0, 0] * xy[:, 0] + h[:, 0, 1] * xy[:, 1] + h[:, 0, 2]) / denominator
    out[seen, 1] = (h[:, 1, 0] * xy[:, 0] + h[:, 1, 1] * xy[:, 1] + h[:, 1, 2]) / denominator
    return out


def inside_image(pixel, margin):
    return ((pixel[:, 0] >= margin) & (pixel[:, 0] < frames.WIDTH - margin)
            & (pixel[:, 1] >= margin) & (pixel[:, 1] < frames.HEIGHT - margin))


def training_frames(pixel):
    """Frames con balon dentro de la imagen, fuera de validacion, uno de cada STRIDE."""
    usable = inside_image(np.nan_to_num(pixel, nan=-1e9), PATCH)
    usable &= ~holdout_mask(len(pixel), HOLDOUT_MARGIN)
    candidates = np.flatnonzero(usable)
    return candidates[::STRIDE]


def _origin_far_from(ball, rng, centre=None):
    """Esquina de un recorte que no contenga el balon. centre: (x, y) a rodear o None."""
    for _ in range(50):
        if centre is None:
            x0 = rng.integers(0, frames.WIDTH - PATCH)
            y0 = rng.integers(0, frames.HEIGHT - PATCH)
        else:
            x0 = int(np.clip(centre[0] - PATCH // 2 + rng.integers(-20, 21), 0,
                             frames.WIDTH - PATCH))
            y0 = int(np.clip(centre[1] - PATCH * 0.7 + rng.integers(-20, 21), 0,
                             frames.HEIGHT - PATCH))
        middle = np.array([x0 + PATCH / 2, y0 + PATCH / 2])
        if np.hypot(*(middle - ball)) > NEGATIVE_CLEARANCE_PX:
            return int(x0), int(y0)
    return None


def build(seed=0):
    """Escribe patches.npy (N, CHANNELS, PATCH, PATCH) uint8 y meta.npz. ~20 min."""
    started = time.time()
    rng = np.random.default_rng(seed)
    tracking = atd.load()
    H = homography.load()[0]
    pixel = projected_ball(tracking, H)
    wanted = training_frames(pixel)
    players = tracking.players_xy

    directory = vision_dir()
    capacity = 3 * len(wanted)
    patches = np.lib.format.open_memmap(
        os.path.join(directory, "patches_tmp.npy"), mode="w+", dtype=np.uint8,
        shape=(capacity, frames.CHANNELS, PATCH, PATCH))
    target = np.full((capacity, 2), -1.0, dtype=np.float32)
    kind = np.zeros(capacity, dtype=np.int8)
    frame_of = np.zeros(capacity, dtype=np.int32)
    count = 0

    def add(t, frames_t, x0, y0, label, kind_name):
        nonlocal count
        patches[count] = frames.stack(H, t, frames_t, x0, y0, PATCH, PATCH)
        if label is not None:
            target[count] = label
        kind[count] = KINDS[kind_name]
        frame_of[count] = t
        count += 1

    for done, (t, frames_t) in enumerate(frames.neighbourhoods(wanted)):
        ball = pixel[t]
        u, v = rng.integers(BALL_JITTER, PATCH - BALL_JITTER, size=2)
        x0, y0 = int(round(ball[0])) - u, int(round(ball[1])) - v
        add(t, frames_t, x0, y0, (ball[0] - x0, ball[1] - y0), "ball")

        origin = _origin_far_from(ball, rng)
        if origin:
            add(t, frames_t, *origin, None, "random")

        visible = np.flatnonzero(~np.isnan(players[t, :, 0]))
        if len(visible):
            slot = rng.choice(visible)
            foot = homography.project(H[t], players[t, slot, 0], players[t, slot, 1])
            if 0 <= foot[0] < frames.WIDTH and 0 <= foot[1] < frames.HEIGHT:
                origin = _origin_far_from(ball, rng, centre=foot)
                if origin:
                    add(t, frames_t, *origin, None, "player")
        if done % 5000 == 0:
            print("  %d/%d frames, %d recortes, %.0f s" % (
                done, len(wanted), count, time.time() - started), flush=True)

    patches.flush()
    del patches
    final = os.path.join(directory, "patches.npy")
    tmp = np.load(os.path.join(directory, "patches_tmp.npy"), mmap_mode="r")
    out = np.lib.format.open_memmap(final, mode="w+", dtype=np.uint8,
                                    shape=(count,) + tmp.shape[1:])
    for start in range(0, count, 4096):
        out[start:start + 4096] = tmp[start:min(count, start + 4096)]
    out.flush()
    del out, tmp
    os.remove(os.path.join(directory, "patches_tmp.npy"))
    np.savez(os.path.join(directory, "meta.npz"), target=target[:count], kind=kind[:count],
             frame=frame_of[:count])
    summary = {"frames": int(len(wanted)), "patches": int(count),
               "by_kind": {k: int((kind[:count] == v).sum()) for k, v in KINDS.items()},
               "minutes": (time.time() - started) / 60}
    with open(os.path.join(directory, "dataset.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(summary)
    return summary


if __name__ == "__main__":
    build()
