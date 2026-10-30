"""Homografia por frame: campo normalizado (0-1) -> pixel de la imagen.

Una fila por frame de video, `frame` contando desde 0, asi que para el video tactico el
indice coincide con la fila de la submission. Columnas 1-9 son la matriz 3x3 en orden de
filas.

Para que sirve de verdad: proyectar los jugadores sobre el frame del video y mirar con tus
ojos que engaño al modelo. Es la herramienta de depuracion, no un feature fuerte --
medido sobre este partido, `reprojection_error` separa "camara apuntando al campo" de
"camara en el publico", pero NO separa jugada de balon parado (mediana 0.231 en juego vs
0.395 en el descanso, p90 en juego 0.308).
"""

import numpy as np
import pandas as pd

from ..constants import GAME
from ..paths import require


def load(source="tactical", game=GAME):
    """(H, reprojection_error, valid) con H de forma (n, 3, 3)."""
    path = require(["homography"], source=source, game=game)[0]
    frame = pd.read_csv(path)
    expected = ["frame", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
                "reprojection_error", "valid"]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError("{}: faltan columnas {}".format(path, missing))
    if not (frame["frame"].values == np.arange(len(frame))).all():
        raise ValueError("{}: la columna frame no es 0..n-1 en orden".format(path))

    matrices = frame[expected[1:10]].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return (matrices,
            frame["reprojection_error"].to_numpy(dtype=np.float64),
            frame["valid"].to_numpy(dtype=bool))


def project(matrix, x, y):
    """Punto(s) del campo normalizado -> pixel. Acepta escalares o arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    px = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
    py = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    return px, py


def unproject(matrix, px, py):
    """Pixel -> punto del campo normalizado. Inversa de project()."""
    inverse = np.linalg.inv(matrix)
    return project(inverse, px, py)
