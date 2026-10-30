"""Suavizado *centrado* de features. Nunca de etiquetas.

La distincion importa y es la leccion central del reto. Suavizar **etiquetas** con una
ventana es lo que hacen los baselines oficiales, y para no fragmentar una posesion larga
tienen que borrar las cortas. Suavizar **features** antes de un decodificador que penaliza
transiciones no tiene ese problema: la senal se limpia y el numero de segmentos lo sigue
controlando el decodificador.

Todo aqui es centrado y por tanto de latencia cero. Medido degradando el ground truth
publico: +1 frame de retraso es gratis (spellF1 1.0000), pero +25 frames (1 s) baja a
0.7765 y +50 (2 s) a 0.5882. Un filtro causal no es una opcion en este problema.

Medido sobre este partido, suavizar sube el AUC de cada senal:

    actividad de jugadores, deteccion de balon muerto:  0.8336 cruda -> 0.8452 con +-50f
    diferencia de control, distinguir equipo:           0.8213 cruda -> 0.9091 con +-50f
"""

import numpy as np


def box_mean(values, half_width, fill=None):
    """Media sobre una ventana centrada de +-half_width frames. Coste independiente del ancho.

    Los NaN se rellenan antes con `fill` (por defecto la mediana) porque una suma
    acumulada no tolera NaN: un solo NaN envenena todo lo que viene detras.
    """
    values = np.asarray(values, dtype=np.float64)
    if half_width <= 0:
        return values.astype(np.float32)
    if np.isnan(values).any():
        replacement = fill if fill is not None else np.nanmedian(values)
        values = np.nan_to_num(values, nan=float(replacement))

    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return ((cumulative[high] - cumulative[low]) / (high - low)).astype(np.float32)


def box_nanmean(values, half_width):
    """Media centrada sobre los frames de la ventana **donde el valor esta definido**.

    Distinto de box_mean con relleno, y la diferencia importa para las senales que
    dependen del balon: donde el balon no se ve, el control no vale 0, es que no se sabe.
    Rellenar con 0 diluye la senal hacia "nadie controla"; promediar solo los frames
    definidos arrastra la informacion de los vecinos a traves del hueco, que es lo que se
    quiere cuando el balon falta en el 43% de los frames.

    Devuelve NaN donde la ventana entera esta indefinida.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = ~np.isnan(values)
    if half_width <= 0:
        return values.astype(np.float32)

    filled = np.where(defined, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0.0], np.cumsum(defined.astype(np.float64))))

    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    total = cumulative[high] - cumulative[low]
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan).astype(np.float32)
