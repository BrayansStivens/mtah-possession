"""El detector de fronteras entrenado, tal y como lo usa la entrega.

Un `HistGradientBoostingClassifier` entrenado en los 7 partidos del IDSSE degradados
(`python -m mtah.training.boundaries export`). Devuelve por frame un
`transition_boundary_score` en [0, 1] que el decodificador usa como descuento sobre el
coste de cambiar de estado. Medido con leave-one-match-out anidado: +1,17 pp de spellF1
sobre el modelo sin detector (ESTADO.md, 7-11).

Datos de entrenamiento: IDSSE, Bassek et al., Sci Data 12, 195 (2025),
doi:10.1038/s41597-025-04505-y, CC-BY 4.0 con autorizacion de la DFL. El modelo es una
obra derivada: la cita y el credito a la DFL son condicion de la licencia.

**Por que falla en vez de seguir.** Un pickle de scikit-learn solo es fiable con la version
que lo escribio; si las features cambian de orden el modelo puntua basura sin quejarse; y
si cambian los parametros de las emisiones (dos de sus senales son los logits) el detector
ve otra distribucion de la que aprendio. Las tres cosas se comprueban al cargar.
"""

import functools
import hashlib
import json
import os

from ..features.boundaries import feature_matrix

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "boundary_detector.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "boundary_detector.json")

# Parametros que solo afectan a la decodificacion: cambiarlos no cambia lo que ve el
# detector, asi que no invalidan el modelo. Todos los demas de config.py si.
DECODER_ONLY_KEYS = frozenset({
    "switch_turnover", "switch_to_dead", "switch_from_dead", "min_duration_frames",
    "prob_floor", "boundary_discount_weight", "boundary_switch_scale",
})


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _as_json(value):
    """Normaliza tuplas a listas para comparar con lo guardado en JSON."""
    return json.loads(json.dumps(value))


def feature_config(config):
    """Los parametros de config.py que determinan las features del detector."""
    return {key: _as_json(value) for key, value in sorted(config.items())
            if key not in DECODER_ONLY_KEYS}


@functools.lru_cache(maxsize=1)
def load():
    """(modelo, metadatos), comprobando integridad y version de scikit-learn."""
    import joblib
    import sklearn

    for path in (ARTIFACT_PATH, METADATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "falta {}. Se genera con `python -m mtah.training.boundaries export`, o pon "
                "boundary_discount_weight = 0 en config.py para decodificar sin detector."
                .format(path))
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if _sha256(ARTIFACT_PATH) != metadata["artifact_sha256"]:
        raise ValueError("{} no coincide con el sha256 de {}: el modelo se ha modificado "
                         "o no es el que describe".format(ARTIFACT_PATH, METADATA_PATH))
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError("el detector se guardo con scikit-learn {} y aqui hay {}: un pickle "
                         "de otra version no es fiable. Instala requirements.txt.".format(
                             metadata["sklearn_version"], sklearn.__version__))
    return joblib.load(ARTIFACT_PATH), metadata


def score(features, config):
    """(n,) transition_boundary_score por frame."""
    model, metadata = load()
    if feature_config(config) != metadata["feature_config"]:
        changed = sorted(key for key, value in feature_config(config).items()
                         if metadata["feature_config"].get(key) != value)
        raise ValueError("el detector se entreno con otros parametros de features/emisiones "
                         "({}); reentrenalo con `python -m mtah.training.boundaries export`"
                         .format(", ".join(changed) or "claves distintas"))
    matrix, names = feature_matrix(features, config)
    if names != metadata["feature_names"]:
        raise ValueError("las columnas de features no son las del entrenamiento; "
                         "reentrena el detector")
    return model.predict_proba(matrix)[:, 1]


def switch_discount(features, config):
    """(n,) descuento por frame para `viterbi.decode(..., switch_discount=...)`."""
    return config["boundary_discount_weight"] * score(features, config)


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
