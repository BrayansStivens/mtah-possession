"""Features del balon derivados de la posicion.

Por que existe este modulo: **la velocidad del balon que trae el fichero es NaN en el
91.3% de los frames en los que el balon SI se ve.** Medido sobre el partido completo. Es
decir, la columna de velocidad del balon es inservible y hay que derivarla de la posicion.

La misma medicion sobre las velocidades de jugador: NaN en el 62.8% de las celdas.
Cualquier cosa que dependa de las velocidades del fichero tiene que tolerar eso.
"""

import numpy as np

from ..constants import BALL_MAX_SPEED, PITCH


def speed(tracking):
    """(n,) velocidad del balon en m/s, derivada de la diferencia de posiciones.

    NaN donde no hay balon en t o en t-1, asi que esta definida en el 56.8% de los frames.
    Se recorta a BALL_MAX_SPEED: el tracking automatico produce saltos imposibles cuando
    confunde el balon con otra cosa, y sin recortar un solo salto domina cualquier media.
    """
    xy = np.asarray(tracking.ball_xy, dtype=np.float32) * np.asarray(PITCH, dtype=np.float32)
    out = np.full(len(xy), np.nan, dtype=np.float32)
    step = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1)) * 25.0
    out[1:] = step
    return np.clip(out, 0.0, BALL_MAX_SPEED)


def displacement(tracking, half_width=25):
    """(n,) distancia en metros entre la posicion del balon half_width frames antes y despues.

    Complementaria a la velocidad: un balon que vibra en el sitio tiene velocidad
    instantanea alta y desplazamiento neto cero, que es exactamente lo que pasa con el
    tracking de un balon parado en el suelo entre las piernas de un jugador.
    """
    xy = np.asarray(tracking.ball_xy, dtype=np.float32) * np.asarray(PITCH, dtype=np.float32)
    n = len(xy)
    index = np.arange(n)
    before = xy[np.maximum(0, index - half_width)]
    after = xy[np.minimum(n - 1, index + half_width)]
    return np.sqrt(((after - before) ** 2).sum(axis=1))
