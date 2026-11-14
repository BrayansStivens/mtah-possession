"""Forma y movimiento de cada equipo **en su propio sentido de ataque**. No miran el balon.

Son las senales del clasificador de equipo (`models/team_classifier.py`), pensado para el
41 % de frames puntuados en los que el ATD no tiene balon: ahi las emisiones a mano solo
arrastran lo ultimo que vieron, y quien tiene la posesion hay que leerlo de los jugadores.

**Por que orientadas.** `features/geometry.py` ya tenia una diferencia de centroides
(`centroid_gap_x`) y se descarto (AUC 0,53): los equipos cambian de campo en el descanso y
el signo se cancela. Aqui cada equipo se mide en un eje que apunta a la porteria rival, y
"el bloque sube" significa lo mismo en las dos partes. El sentido de ataque sale del propio
tracking, sin etiquetas ni metadatos (`attack_direction`).

**Por que solo seis.** El experimento del 11-12 de noviembre midio nueve
(`trabajo/sin-balon-1111/RESUMEN.md`). Se quedan fuera el centroide, la longitud del bloque
y la fraccion de jugadores en campo rival: son las que mas distinguen el partido del reto
de los del IDSSE (un clasificador de dominio las separa con AUC 0,93; con estas seis, 0,72)
y quitarlas casi no cuesta (AUC de equipo 0,9518 -> 0,9506).

La matriz se construye por **perspectiva**: la del local y la del visitante son la misma
funcion con los papeles cambiados. Entrenar con las dos y restar las dos salidas da un
logit antisimetrico, que no puede aprender "el local suele tener el balon".
"""

import numpy as np

from ..constants import FPS, PITCH
from . import geometry, smoothing

# Por equipo, en su eje de ataque (x > 0 es campo rival):
#   wid   dispersion a lo ancho (desviacion de y)
#   dfl   linea defensiva: segundo jugador mas retrasado (el primero es el portero)
#   atl   linea de ataque: segundo jugador mas adelantado
#   tide  velocidad media hacia la porteria rival
#   spd   velocidad media
#   spr   fraccion de jugadores por encima de SPRINT_SPEED_MS
# El orden es el de las columnas del modelo guardado: no se puede cambiar sin reentrenar.
SIGNALS = ("wid", "dfl", "atl", "tide", "spd", "spr")

# Senales con signo propio (hacia delante o hacia atras): ademas de la diferencia entre
# equipos se pasa la suma, que dice si los dos bloques se mueven hacia la misma porteria.
SUM_SIGNALS = ("tide",)

# Medias anchuras de suavizado centrado: 1, 3 y 8 s de ventana.
WINDOWS_FRAMES = (12, 37, 100)

# Velocidad por diferencia centrada a +-5 frames (0,4 s). Por encima de 12 m/s no es un
# jugador corriendo: es el tracker cambiando la identidad del slot.
VELOCITY_LAG_FRAMES = 5
MAX_SPEED_MS = 12.0
SPRINT_SPEED_MS = 5.0

HALF_LENGTH_M = PITCH[0] / 2.0


# --------------------------------------------------------------------------- sentido de ataque

def centroid_gap(tracking):
    """(n,) centroide visitante menos centroide local en x, metros. >0: el local defiende x=0."""
    x = tracking.players_xy[:, :, 0].astype(np.float64) * PITCH[0]
    with np.errstate(invalid="ignore"):
        home = np.nanmean(x[:, tracking.team_of_slot == geometry.HOME_CODE], axis=1)
        away = np.nanmean(x[:, tracking.team_of_slot == geometry.AWAY_CODE], axis=1)
    return away - home


def attack_direction(tracking, blocks=None):
    """(n,) +1 donde el local ataca hacia +x, -1 donde ataca hacia -x; y un informe por parte.

    Los dos equipos estan ordenados en el campo (cada uno tiene a su portero y a su defensa
    detras), asi que el signo de la diferencia de centroides dice quien defiende que
    porteria. `blocks` son las partes si se conocen (GRADED_BLOCKS en el partido del reto);
    si no, se busca el unico cambio de signo, que es el descanso. `agreement` en el informe
    es |media del signo| en la parte: cerca de 1 si el sentido esta claro.
    """
    sign = np.sign(np.nan_to_num(centroid_gap(tracking)))
    n = len(sign)
    if blocks is None:
        cumulative = np.cumsum(sign)
        split = int(np.argmax(np.abs(2 * cumulative - cumulative[-1]))) + 1
        blocks = ((0, split), (split, n))
    direction = np.ones(n, dtype=np.float64)
    report = []
    previous = 1.0
    for start, end in blocks:
        mean = float(sign[start:end].mean()) if end > start else 0.0
        value = previous if mean == 0 else (1.0 if mean > 0 else -1.0)
        direction[start:end] = value
        report.append({"block": [int(start), int(end)], "direction": value,
                       "agreement": abs(mean)})
        previous = value
    # Frames fuera de las partes (previa, descanso): el sentido de la parte que viene.
    if blocks[0][0] > 0:
        direction[:blocks[0][0]] = direction[blocks[0][0]]
    for (_, end), (next_start, _) in zip(blocks[:-1], blocks[1:]):
        if next_start > end:
            direction[end:next_start] = direction[next_start]
    return direction, report


# --------------------------------------------------------------------------- por frame

def _velocity(xy_m):
    """(n, slots, 2) velocidad por slot en m/s, NaN donde el slot no es el mismo objeto."""
    lag = VELOCITY_LAG_FRAMES
    velocity = np.full(xy_m.shape, np.nan, dtype=np.float32)
    velocity[lag:-lag] = (xy_m[2 * lag:] - xy_m[:-2 * lag]) * (FPS / (2.0 * lag))
    with np.errstate(invalid="ignore"):
        jump = np.hypot(velocity[:, :, 0], velocity[:, :, 1]) > MAX_SPEED_MS
    velocity[jump] = np.nan
    return velocity


def _second(ordered, count, from_end):
    """Segundo valor mas bajo (o mas alto) de cada fila ya ordenada, con los NaN al final."""
    n = len(ordered)
    index = np.clip(count - 2, 0, ordered.shape[1] - 1) if from_end else np.ones(n, int)
    return ordered[np.arange(n), index]


def raw(tracking, direction):
    """{equipo: {senal: (n,) float32}} sin suavizar. NaN si se ven menos de 3 jugadores."""
    xy = geometry.to_metres(tracking.players_xy)
    velocity = _velocity(xy)
    speed = tracking.players_speed
    out = {}
    for team, own in ((geometry.HOME_CODE, direction), (geometry.AWAY_CODE, -direction)):
        slots = np.flatnonzero(tracking.team_of_slot == team)
        own = own.astype(np.float32)[:, None]
        forward = own * (xy[:, slots, 0] - HALF_LENGTH_M)        # >0: campo rival
        count = np.sum(~np.isnan(forward), axis=1)
        ordered = np.sort(forward, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            signals = {
                "wid": np.nanstd(xy[:, slots, 1], axis=1),
                "dfl": _second(ordered, count, from_end=False),
                "atl": _second(ordered, count, from_end=True),
                "tide": np.nanmean(own * velocity[:, slots, 0], axis=1),
                "spd": np.nanmean(speed[:, slots], axis=1),
                "spr": (np.nansum(speed[:, slots] > SPRINT_SPEED_MS, axis=1)
                        / np.maximum(count, 1)),
            }
        enough = count >= geometry.MIN_TRACKED_FOR_SHAPE
        out[team] = {name: np.where(enough, signals[name], np.nan).astype(np.float32)
                     for name in SIGNALS}
    return out


def smoothed(tracking, blocks=None):
    """(2, senales, ventanas, n) float32: cada senal de cada equipo, suavizada y centrada.

    `blocks` se pasa tal cual a `attack_direction`.
    """
    direction, _ = attack_direction(tracking, blocks)
    signals = raw(tracking, direction)
    return np.stack([[[smoothing.box_nanmean(signals[team][name], window)
                       for window in WINDOWS_FRAMES] for name in SIGNALS]
                     for team in (geometry.HOME_CODE, geometry.AWAY_CODE)]).astype(np.float32)


# --------------------------------------------------------------------------- matriz

def feature_names():
    """Nombres de las columnas de `matrix`, en su orden."""
    names = []
    for window in WINDOWS_FRAMES:
        names += ["own_%s_%d" % (name, window) for name in SIGNALS]
        names += ["rival_%s_%d" % (name, window) for name in SIGNALS]
        names += ["diff_%s_%d" % (name, window) for name in SIGNALS]
        names += ["sum_%s_%d" % (name, window) for name in SUM_SIGNALS]
    return names


def matrix(stack, perspective):
    """(n, columnas) lo que ve `perspective` (0 local, 1 visitante): propio, rival y diferencia."""
    own, rival = perspective, 1 - perspective
    sums = [SIGNALS.index(name) for name in SUM_SIGNALS]
    columns = []
    for window in range(stack.shape[2]):
        columns += [stack[own, :, window].T, stack[rival, :, window].T,
                    (stack[own, :, window] - stack[rival, :, window]).T,
                    (stack[own, sums, window] + stack[rival, sums, window]).T]
    return np.concatenate(columns, axis=1)
