"""El clasificador de equipo sin balon, tal y como lo usa la entrega.

Un `HistGradientBoostingClassifier` entrenado en los 7 partidos del IDSSE degradados
(`python -m mtah.training.team_classifier export`). Mira solo a los jugadores
(`features/team_signals.py`: anchura, lineas, velocidad hacia la porteria rival) y devuelve
por frame un logit de "tiene la posesion el local", que se suma al logit de equipo de las
emisiones con `team_classifier_weight`. Existe por los frames sin balon: ahi las emisiones a
mano solo arrastran la ultima senal vista.

El logit es **antisimetrico**: el modelo se entrena con las dos perspectivas (la del local
y la del visitante son la misma matriz con los papeles cambiados) y se usa la mitad de la
diferencia entre las dos salidas. Cambiar los equipos de nombre le cambia el signo, asi que
no puede aprender "el local suele tener el balon".

Datos de entrenamiento: IDSSE, Bassek et al., Sci Data 12, 195 (2025),
doi:10.1038/s41597-025-04505-y, CC-BY 4.0 con autorizacion de la DFL. El modelo es una
obra derivada: la cita y el credito a la DFL son condicion de la licencia.

**Por que falla en vez de seguir.** Igual que el detector de fronteras: un pickle de
scikit-learn solo es fiable con la version que lo escribio, y si las senales cambian de
orden o de definicion el modelo puntua basura sin quejarse. Las dos cosas se comprueban al
cargar. Sus features no dependen de config.py, asi que no hay nada de config que comprobar.
"""

import functools
import json
import os

import numpy as np

from ..features import team_signals
from .boundary_detector import _as_json, _sha256

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "team_classifier.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "team_classifier.json")


def feature_settings():
    """Las constantes de features/team_signals.py que determinan lo que ve el modelo."""
    return _as_json({
        "signals": team_signals.SIGNALS,
        "sum_signals": team_signals.SUM_SIGNALS,
        "windows_frames": team_signals.WINDOWS_FRAMES,
        "velocity_lag_frames": team_signals.VELOCITY_LAG_FRAMES,
        "max_speed_ms": team_signals.MAX_SPEED_MS,
        "sprint_speed_ms": team_signals.SPRINT_SPEED_MS,
    })


@functools.lru_cache(maxsize=1)
def load():
    """(modelo, metadatos), comprobando integridad y version de scikit-learn."""
    import joblib
    import sklearn

    for path in (ARTIFACT_PATH, METADATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "falta {}. Se genera con `python -m mtah.training.team_classifier export`, o "
                "pon team_classifier_weight = 0 en config.py para decodificar sin el "
                "clasificador.".format(path))
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if _sha256(ARTIFACT_PATH) != metadata["artifact_sha256"]:
        raise ValueError("{} no coincide con el sha256 de {}: el modelo se ha modificado "
                         "o no es el que describe".format(ARTIFACT_PATH, METADATA_PATH))
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError("el clasificador se guardo con scikit-learn {} y aqui hay {}: un "
                         "pickle de otra version no es fiable. Instala requirements.txt."
                         .format(metadata["sklearn_version"], sklearn.__version__))
    return joblib.load(ARTIFACT_PATH), metadata


def antisymmetric_logit(model, stack):
    """(n,) float32 logit de local: media de lo que dice cada perspectiva, con su signo."""
    home = model.decision_function(team_signals.matrix(stack, 0))
    away = model.decision_function(team_signals.matrix(stack, 1))
    return (0.5 * (home - away)).astype(np.float32)


def home_logit(tracking, blocks=None):
    """(n,) logit de "posesion del local" segun la forma de los equipos, sin mirar el balon.

    `blocks` son las partes del partido si se conocen (ver `team_signals.attack_direction`).
    """
    model, metadata = load()
    if (team_signals.feature_names() != metadata["feature_names"]
            or feature_settings() != metadata["feature_settings"]):
        raise ValueError("las senales de equipo no son las del entrenamiento; reentrena el "
                         "clasificador con `python -m mtah.training.team_classifier export`")
    return antisymmetric_logit(model, team_signals.smoothed(tracking, blocks))


def save(model, metadata):
    """Escribe el modelo y sus metadatos (con el sha256 y la version de scikit-learn)."""
    import joblib
    import sklearn

    joblib.dump(model, ARTIFACT_PATH, compress=3)
    metadata = dict(metadata, artifact_sha256=_sha256(ARTIFACT_PATH),
                    sklearn_version=sklearn.__version__)
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    load.cache_clear()
    return metadata
