"""De picos por frame a trayectorias del balon en el campo.

Un pico suelto del detector puede ser una media blanca; diez picos seguidos que se mueven
como se mueve un balon, casi seguro que no. Esto enlaza los picos frame a frame en
tramos (tracklets), con una puerta fisica: entre dos detecciones separadas d frames el
balon no puede haber recorrido mas de BALL_MAX_SPEED * d / FPS + GATE_SLACK_M metros.
Despues se quedan los tramos con suficiente evidencia y, donde dos se pisan, el mas fuerte.

Las coordenadas de campo salen de deshacer la homografia, que mapea el plano del suelo:
un balon en el aire cae proyectado mas lejos de la camara de lo que esta. Para decidir
posesion en el suelo es la posicion que importa; en el aire, el error es el mismo que
tendria cualquier tracker 2D.
"""

import numpy as np

from ..constants import BALL_MAX_SPEED, FPS, PITCH

SCORE_LOW = 0.15          # un pico por debajo no extiende ningun tramo
SCORE_START = 0.4         # ni lo empieza por debajo de esto
GATE_SLACK_M = 1.5
MAX_MISS = 6              # frames sin deteccion que un tramo aguanta vivo
MIN_DETECTIONS = 8        # detecciones para quedarse un tramo
MIN_MEAN_SCORE = 0.35


def to_field(H, frame, pixel):
    """(K, 2) pixel del frame -> (K, 2) metros en el campo."""
    inverse = np.linalg.inv(H[frame])
    x, y = pixel[:, 0], pixel[:, 1]
    denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
    u = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / denominator
    v = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / denominator
    return np.stack([u * PITCH[0], v * PITCH[1]], axis=1)


class Tracklet:
    __slots__ = ("frames", "xy", "scores")

    def __init__(self, frame, xy, score):
        self.frames, self.xy, self.scores = [frame], [xy], [score]

    @property
    def last(self):
        return self.frames[-1]

    def gate(self, frame):
        return BALL_MAX_SPEED * (frame - self.last) / FPS + GATE_SLACK_M

    def strength(self):
        return float(np.sum(self.scores))


def link(frames, field_xy, scores):
    """Tramos a partir de candidatos por frame.

    frames   (n,)        frames con picos, en orden creciente
    field_xy (n, K, 2)   candidatos en metros
    scores   (n, K)      probabilidad del detector
    """
    active, closed = [], []
    for row, frame in enumerate(frames):
        still = []
        for tracklet in active:
            (still if frame - tracklet.last <= MAX_MISS + 1 else closed).append(tracklet)
        active = still
        usable = np.flatnonzero(scores[row] >= SCORE_LOW)
        taken = set()
        # cada tramo vivo se lleva el candidato mas cercano dentro de su puerta, los
        # tramos mas fuertes eligen primero
        for tracklet in sorted(active, key=lambda t: -t.strength()):
            if tracklet.last == frame:
                continue
            best, best_distance = None, tracklet.gate(frame)
            for k in usable:
                if k in taken:
                    continue
                distance = np.hypot(*(field_xy[row, k] - tracklet.xy[-1]))
                if distance <= best_distance:
                    best, best_distance = k, distance
            if best is not None:
                taken.add(best)
                tracklet.frames.append(frame)
                tracklet.xy.append(field_xy[row, best])
                tracklet.scores.append(float(scores[row, best]))
        for k in usable:
            if k not in taken and scores[row, k] >= SCORE_START:
                active.append(Tracklet(frame, field_xy[row, k], float(scores[row, k])))
    return closed + active


def select(tracklets, n_frames):
    """(n_frames, 2) metros, NaN donde ningun tramo aceptado cubre el frame.

    Tramos de mas fuerte a mas debil; cada uno rellena solo los frames que nadie ha
    rellenado, interpolando linealmente sus huecos internos (<= MAX_MISS frames).
    """
    out = np.full((n_frames, 2), np.nan)
    kept = [t for t in tracklets
            if len(t.frames) >= MIN_DETECTIONS and np.mean(t.scores) >= MIN_MEAN_SCORE]
    for tracklet in sorted(kept, key=lambda t: -t.strength()):
        frames = np.asarray(tracklet.frames)
        xy = np.asarray(tracklet.xy)
        span = np.arange(frames[0], frames[-1] + 1)
        free = np.isnan(out[span, 0])
        if not free.any():
            continue
        filled = np.stack([np.interp(span, frames, xy[:, 0]),
                           np.interp(span, frames, xy[:, 1])], axis=1)
        out[span[free]] = filled[free]
    return out


def fill(peak_frames, peaks, H, n_frames):
    """Trayectoria en metros a partir de la salida de detect.run()."""
    order = np.argsort(peak_frames)
    peak_frames, peaks = peak_frames[order], peaks[order]
    field_xy = np.stack([to_field(H, f, p[:, :2]) for f, p in zip(peak_frames, peaks)])
    return select(link(peak_frames, field_xy, peaks[..., 2]), n_frames)
