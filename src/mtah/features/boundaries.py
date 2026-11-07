"""Features del detector de fronteras: que cambia entre justo antes y justo despues.

Las emisiones miran ventanas de +-50 frames (y la forma de equipo +-200): saben *que*
equipo domina un tramo, pero no *donde* empieza. Esto mira, para cada frame, la media de
13 senales en una ventana corta despues del frame y su diferencia con la de antes, a
0,24 / 0,48 / 1 / 2 s. Lo consume `models/boundary_detector.py` en la entrega y
`training/boundaries.py` para entrenar y medir; los dos tienen que ver exactamente las
mismas columnas, por eso viven aqui y en ningun otro sitio.
"""

import numpy as np

from ..constants import FPS
from ..models import emissions

# Ventanas a cada lado del frame, en frames: de 0,24 s a 2 s.
WINDOWS_FRAMES = (6, 12, 25, 50)
PRODUCT_WINDOWS_FRAMES = (25, 50)

SIGNAL_NAMES = ("control_diff", "nearest_gap_m", "activity", "ball_speed", "ball_seen",
                "ball_out", "ball_vx", "home_vx", "away_vx", "centroid_gap_x",
                "spread_ratio_x", "home_logit", "dead_logit")


def feature_names():
    """Nombres de las columnas de `feature_matrix`, en orden. No depende de los datos."""
    names = ["%s_%s_%d" % (signal, kind, width) for signal in SIGNAL_NAMES
             for width in WINDOWS_FRAMES for kind in ("change", "after")]
    names += ["both_vx_change_product_%d" % width for width in PRODUCT_WINDOWS_FRAMES]
    return names + ["frames_since_ball_seen", "frames_until_ball_seen", "ball_gap_frames"]


def window_means(values, width):
    """(antes, despues): media ignorando NaN en [t-width, t) y en [t, t+width).

    NaN donde la ventana no tiene ningun valor definido: un hueco de balon es un dato que
    falta, no un cero, y el clasificador (HistGradientBoosting) trata NaN como tal.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(defined, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(defined)))
    n = len(values)
    frame = np.arange(n)
    lo, hi = np.maximum(frame - width, 0), np.minimum(frame + width, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        before = (total[frame] - total[lo]) / (count[frame] - count[lo])
        after = (total[hi] - total[frame]) / (count[hi] - count[frame])
    return before, after


def _frames_since(event):
    """(n,) frames desde la ultima vez que `event` fue True (n si nunca)."""
    n = len(event)
    last = np.maximum.accumulate(np.where(event, np.arange(n), -1))
    return np.where(last >= 0, np.arange(n) - last, n).astype(np.float64)


def frame_signals(features, config):
    """Senales por frame, sin suavizar, de las que se miran los cambios antes/despues."""
    seen = features["ball_seen"]
    ball_x = np.where(seen, features["ball_x"], np.nan)
    return {
        "control_diff": np.where(seen, features["control_home"] - features["control_away"],
                                 np.nan),
        "nearest_gap_m": np.where(seen, features["nearest_home"] - features["nearest_away"],
                                  np.nan),
        "activity": features["activity"],
        "ball_speed": features["ball_speed"],
        "ball_seen": seen.astype(np.float64),
        "ball_out": features["ball_out"].astype(np.float64),
        "ball_vx": np.diff(ball_x, prepend=np.nan) * FPS,
        "home_vx": np.diff(features["home_centroid_x"], prepend=np.nan) * FPS,
        "away_vx": np.diff(features["away_centroid_x"], prepend=np.nan) * FPS,
        "centroid_gap_x": features["centroid_gap_x"],
        "spread_ratio_x": features["spread_ratio_x"],
        "home_logit": emissions.home_logit(features, config),
        "dead_logit": emissions.dead_logit(features, config),
    }


def feature_matrix(features, config):
    """(matriz (n, k) float32, nombres). Todo lo que ve el detector en cada frame."""
    signals = frame_signals(features, config)
    if tuple(signals) != SIGNAL_NAMES:
        raise ValueError("frame_signals y SIGNAL_NAMES no coinciden")
    columns = []
    for values in signals.values():
        for width in WINDOWS_FRAMES:
            before, after = window_means(values, width)
            columns += [after - before, after]
    # Un robo invierte el sentido de los dos bloques a la vez: si los dos centroides
    # cambian de velocidad en sentidos opuestos, el producto sale negativo.
    for width in PRODUCT_WINDOWS_FRAMES:
        home = window_means(signals["home_vx"], width)
        away = window_means(signals["away_vx"], width)
        columns.append((home[1] - home[0]) * (away[1] - away[0]))
    seen = features["ball_seen"]
    columns += [_frames_since(seen), _frames_since(seen[::-1])[::-1],
                np.asarray(features["ball_gap"], dtype=np.float64)]
    return np.column_stack(columns).astype(np.float32), feature_names()
