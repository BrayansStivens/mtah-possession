"""Arnes de validacion: medir el pipeline sobre un partido COMPLETO con etiquetas.

El problema que resuelve. El unico ground truth que tenemos del partido del reto son 5
minutos y 17 posesiones, y EVALUATION.md es explicito en que **no sirve para elegir entre
dos versiones del modelo**: el baseline que gana ahi pierde en el partido completo. Hasta
aqui cada numero de este proyecto sale de esa ventana.

Un partido del IDSSE degradado si da un spellF1 de partido completo contra etiquetas
reales. Siete veces.

**Lo que mide bien y lo que no.** Mide bien las diferencias *relativas*: si un cambio
mejora en los siete partidos, mejora. El valor *absoluto* solo es orientativo, por dos
motivos que no se pueden quitar: el convenio de `BallStatus` del DFL no es identico al `D`
del reto, y la Bundesliga no es este Barcelona-Athletic. Ademas la degradacion reparte las
rachas sin balon al azar mientras que el ATD las concentra en los frames dificiles, asi
que el proxy es **optimista** por construccion.

Antes de creerse ningun numero, `compare_stats()` comprueba que lo degradado se parece al
ATD en los ejes que se midieron. Si no se parece, el numero no vale nada.
"""

import numpy as np

from .. import evaluate as ev
from .. import features as feature_set
from .. import submission
from ..config import DEFAULT
from ..constants import GRADED_BLOCKS
from ..decode import viterbi
from ..io import atd
from ..models import emissions
from . import convention, degrade, idsse


def real_spec(source="tactical"):
    """Parametros de degradacion medidos sobre el ATD real del partido del reto."""
    return degrade.estimate(atd.load(source=source), GRADED_BLOCKS)


def _stats(tracking, blocks=None):
    n = tracking.n_frames
    mask = np.ones(n, dtype=bool)
    if blocks is not None:
        mask[:] = False
        for a, b in blocks:
            mask[a:b] = True
    visible = ~np.isnan(tracking.players_xy[:, :, 0])
    n_tracked = visible[mask].sum(axis=1)
    runs = degrade._runs(~tracking.ball_seen[mask])
    return {
        "balon_visible_pct": 100.0 * tracking.ball_seen[mask].mean(),
        "objetos_media": float(n_tracked.mean()),
        "objetos_p50": float(np.median(n_tracked)),
        "objetos_max": int(n_tracked.max()),
        "rachas_sin_balon_n": len(runs),
        "racha_media_fr": float(runs.mean()) if len(runs) else 0.0,
        "racha_p50_fr": float(np.median(runs)) if len(runs) else 0.0,
        "racha_max_fr": int(runs.max()) if len(runs) else 0,
    }


def compare_stats(degraded, source="tactical"):
    """Imprime lado a lado las estadisticas del ATD real y las del proxy degradado."""
    real = _stats(atd.load(source=source), GRADED_BLOCKS)
    fake = _stats(degraded)
    print("%-22s %>12s %12s" .replace(">", "") % ("estadistica", "ATD real", "degradado"))
    for k in real:
        print("%-22s %12.2f %12.2f" % (k, real[k], fake[k]))
    return real, fake


def predict(tracking, config=None):
    """(etiquetas, features) del pipeline sobre un tracking cualquiera, en un solo bloque.

    Un solo bloque porque el fichero del IDSSE no trae los frames del descanso: no hay
    hueco que separar y el decodificador no arrastra nada a traves de nada. Devuelve
    tambien los features para que el analisis de errores no tenga que recalcularlos.

    Sin el detector de fronteras, a proposito: se entreno con estos mismos 7 partidos y
    aplicarlo aqui seria medirlo sobre su conjunto de entrenamiento. Su medicion honesta
    es la de `training/boundaries.py` (leave-one-match-out anidado).
    """
    config = config or DEFAULT
    blocks = ((0, tracking.n_frames),)
    features = feature_set.build_all(tracking, config)
    cost = emissions.costs(features, config)
    transition = viterbi.switch_costs(config["switch_turnover"], config["switch_to_dead"],
                                      config["switch_from_dead"])
    codes = viterbi.decode(cost, transition, blocks=blocks)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"], blocks=blocks)
    return viterbi.to_labels(codes), features


def run_match(match_id, config=None, seed=0, spec=None, verbose=True, harmonise=True):
    """Degrada un partido del IDSSE, pasa el pipeline y puntua contra sus etiquetas.

    `harmonise=True` lleva antes las etiquetas al convenio de posesion del reto (ver
    convention.py). Sin eso se le pide al modelo acertar 421 posesiones cuando esta
    disenado para producir ~220, y el spellF1 sale hundido por un motivo que no tiene que
    ver con la calidad del modelo.
    """
    config = config or DEFAULT
    spec = spec or real_spec()
    clean, labels = idsse.load(match_id)
    if harmonise:
        labels, _ = convention.harmonise(labels)

    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed))
    pred, _ = predict(dirty, config)

    gt_codes = ev.pm.label_codes(np.asarray(labels, dtype="<U1"))
    result = ev.score(pred, gt_codes=gt_codes)
    if verbose:
        print("%-8s spellF1 %.4f  macroF1 %.4f   poss %d vs %d   dur %.1fs vs %.1fs" % (
            match_id, result["spell_f1_mean"], result["macro_f1"],
            result["n_pred_spells"], result["n_gt_spells"],
            result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    return result


def run_all(config=None, seed=0, matches=None, spec=None, harmonise=True):
    """Los siete partidos. Devuelve (lista de resultados, media de spellF1, media macroF1).

    La media sobre siete partidos es el numero en el que apoyarse para decidir, no el de
    ningun partido suelto ni el de la ventana publica.
    """
    spec = spec or real_spec()
    out = []
    for m in (matches or sorted(idsse.MATCHES)):
        out.append(run_match(m, config=config, seed=seed, spec=spec,
                             harmonise=harmonise))
    spell = float(np.mean([r["spell_f1_mean"] for r in out]))
    macro = float(np.mean([r["macro_f1"] for r in out]))
    print("-" * 64)
    print("MEDIA de %d partidos   spellF1 %.4f   macroF1 %.4f" % (len(out), spell, macro))
    return out, spell, macro
