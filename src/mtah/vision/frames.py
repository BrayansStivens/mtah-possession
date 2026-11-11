"""Lectura secuencial del video tactico, con vecinos alineados por la homografia.

La camara tactica panea. Para que la red vea el movimiento del balon y no el de la camara,
los frames t-k y t+k se deforman a la vista de t con H_t . H_s^-1: la homografia lleva el
plano del campo al pixel en cada frame, asi que esa composicion es exactamente el
movimiento de camara entre s y t (para todo lo que esta en el suelo, que es casi todo).

Nunca se busca un frame suelto en el video (~1 s por salto): siempre un pase secuencial.
"""

import cv2
import numpy as np

from .. import paths

NEIGHBOUR_OFFSET = 2            # frames entre t y cada vecino (80 ms)
CHANNELS = 5                    # B, G, R de t + gris de t-k + gris de t+k deformados
WIDTH, HEIGHT = 1920, 1080


def camera_motion(H, source, target):
    """Matriz 3x3 que lleva un pixel del frame `source` al pixel del frame `target`."""
    return H[target] @ np.linalg.inv(H[source])


def neighbourhoods(wanted, offset=NEIGHBOUR_OFFSET, video_path=None):
    """Para cada t de `wanted` (en orden), (t, {s: frame_bgr}) con s en {t-k, t, t+k}.

    Un unico pase secuencial; solo se decodifican a imagen los frames que hacen falta.
    Los t sin vecinos dentro del video se saltan.
    """
    wanted = np.unique(np.asarray(wanted, dtype=np.int64))
    if not len(wanted):
        return
    video = cv2.VideoCapture(video_path or paths.require(["video"])[0])
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = wanted[(wanted >= offset) & (wanted < total - offset)]
    needed = np.zeros(total, dtype=bool)
    for shift in (-offset, 0, offset):
        needed[wanted + shift] = True
    first = int(wanted[0]) - offset
    if first > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, first)
    buffer = {}
    cursor = 0
    for index in range(first, int(wanted[-1]) + offset + 1):
        if not video.grab():
            raise RuntimeError("el video se acaba en el frame %d" % index)
        if needed[index]:
            ok, image = video.retrieve()
            if not ok:
                raise RuntimeError("no se pudo decodificar el frame %d" % index)
            buffer[index] = image
        centre = index - offset
        if cursor < len(wanted) and wanted[cursor] == centre:
            yield centre, {s: buffer[s] for s in (centre - offset, centre, centre + offset)}
            cursor += 1
        # lo que ya no puede ser vecino de nadie se suelta
        for stale in [s for s in buffer if s < centre - offset + 1]:
            del buffer[stale]
    video.release()


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def stack(H, t, frames, x0=0, y0=0, width=WIDTH, height=HEIGHT, offset=NEIGHBOUR_OFFSET):
    """(CHANNELS, height, width) uint8 de la region [x0, x0+width) x [y0, y0+height) de t.

    Lo que cae fuera de la imagen (en t o en el vecino deformado) queda a 0.
    """
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    out = np.zeros((CHANNELS, height, width), dtype=np.uint8)
    current = cv2.warpAffine(frames[t], shift[:2], (width, height)) \
        if (x0, y0, width, height) != (0, 0, WIDTH, HEIGHT) else frames[t]
    out[:3] = current.transpose(2, 0, 1)
    for channel, source in ((3, t - offset), (4, t + offset)):
        motion = shift @ camera_motion(H, source, t)
        out[channel] = cv2.warpPerspective(_gray(frames[source]), motion, (width, height),
                                           flags=cv2.INTER_LINEAR)
    return out
