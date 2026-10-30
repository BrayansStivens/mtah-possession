"""Construccion, escritura y validacion del fichero de entrega.

Contrato (SUBMISSION.md): cabecera exacta `frame,label`, 199.319 filas de datos, columna
`frame` de 0 a 199.318 en orden, etiqueta en {H, A, D, X}, sin celdas vacias.

Decision de diseno: **nunca escribimos X en un frame puntuado.**

Esta medido con el scorer oficial. El scorer rellena las X hacia delante con tu ultima
etiqueta del mismo tiempo *antes* de emparejar posesiones, asi que en spellF1 una X es
exactamente "repite mi respuesta anterior". Pero macroF1 la cobra como fallo. Abstenerse
en el 20% de los frames puntuados frente a escribir uno mismo ese mismo relleno:

    con 20% de X              spellF1 0.3610   macroF1 0.5438
    mismo pero relleno a mano spellF1 0.3610   macroF1 0.6192

Identico en la metrica de ranking, 0.075 peor en el desempate. La X esta estrictamente
dominada dentro de los bloques puntuados. Fuera de ellos no se puntua nada, y ahi X es la
respuesta honesta.
"""

import os

import numpy as np

from .constants import (ABSTAIN, DEAD, GRADED_BLOCKS, KICKOFF_2ND, LABELS, N_FRAMES)

HEADER = "frame,label"


def build(labels_graded, n_frames=N_FRAMES, blocks=GRADED_BLOCKS,
          outside=ABSTAIN, dead_before_2nd_kickoff=True):
    """Array de etiquetas de longitud completa a partir de las de los bloques puntuados.

    `labels_graded` puede ser:
      - un array de longitud n_frames, del que solo se respetan los bloques puntuados
      - una lista de arrays, uno por bloque, en el orden de `blocks`
    """
    out = np.full(n_frames, outside, dtype="<U1")

    if isinstance(labels_graded, (list, tuple)) and len(labels_graded) == len(blocks) \
            and not isinstance(labels_graded[0], str):
        for (start, stop), chunk in zip(blocks, labels_graded):
            chunk = np.asarray(chunk, dtype="<U1")
            if len(chunk) != stop - start:
                raise ValueError("bloque {}-{}: esperaba {} etiquetas, hay {}"
                                 .format(start, stop, stop - start, len(chunk)))
            out[start:stop] = chunk
    else:
        source = np.asarray(labels_graded, dtype="<U1")
        if len(source) != n_frames:
            raise ValueError("esperaba {} etiquetas, hay {}".format(n_frames, len(source)))
        for start, stop in blocks:
            out[start:stop] = source[start:stop]

    if dead_before_2nd_kickoff:
        # El segundo tiempo se puntua desde 110.808 pero no se saca hasta 113.742.
        # Son 2.934 frames de balon parado, y el dato es publico (FAQ del repo oficial).
        second_start = blocks[1][0]
        out[second_start:KICKOFF_2ND] = DEAD

    # Invariante del diseno: ninguna X dentro de los bloques puntuados.
    for start, stop in blocks:
        abstained = out[start:stop] == ABSTAIN
        if abstained.any():
            raise ValueError(
                "{} frames puntuados quedaron en X (p.ej. {}). La X esta dominada dentro "
                "de los bloques: escribe tu propio relleno."
                .format(int(abstained.sum()), int(start + np.flatnonzero(abstained)[0])))
    return out


def write(path, labels):
    """Escribe el CSV de entrega. Devuelve la ruta."""
    labels = np.asarray(labels, dtype="<U1")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(HEADER + "\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def validate(path, n_expected=N_FRAMES):
    """None si el fichero es una entrega valida, o la razon en una linea.

    Reimplementa las mismas comprobaciones que el validador oficial, leyendo el fichero
    desde disco: lo que se valida es el fichero que se va a subir, no el array en memoria.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(path, dtype={"label": "object"})
    except Exception as error:                              # noqa: BLE001
        return "no se puede leer: {}".format(error)

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "frame" not in frame.columns or "label" not in frame.columns:
        return "esperaba columnas 'frame' y 'label', hay {}".format(list(frame.columns))
    if len(frame) != n_expected:
        return "{} filas != {}".format(len(frame), n_expected)

    frames = frame["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        return "la columna 'frame' no es entera (dtype {})".format(frames.dtype)
    if not np.array_equal(frames, np.arange(n_expected)):
        return "la columna 'frame' no es 0..{} en orden".format(n_expected - 1)

    values = frame["label"].values
    blank = pd.isna(values)
    if blank.any():
        return "etiqueta vacia en {} filas; escribe X para abstenerte".format(int(blank.sum()))

    cleaned = np.array([str(v).strip().upper() for v in values], dtype=object)
    bad = sorted(set(cleaned) - set(LABELS))
    if bad:
        return "etiquetas fuera de {}: {}".format(list(LABELS), bad[:5])
    return None


def counts(labels):
    labels = np.asarray(labels, dtype="<U1")
    return {label: int(np.sum(labels == label)) for label in LABELS}
