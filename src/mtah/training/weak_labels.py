"""Diagnostico sobre el partido del reto: "de quien es el balon" contra el Smart Tagging.

**Para que existe.** Todo lo que medimos del modelo sale de los partidos del IDSSE
degradados: otra liga, otro tracker. Esta es la unica medicion de la mitad "local o
visitante" sobre BAR-ATH mismo. El Smart Tagging lo genera Metrica automaticamente, sin
revision humana, asi que usarlo no choca con la regla de "nada producido por humanos sobre
este partido". Aun asi es un **diagnostico, no un conjunto de ajuste**: elegir pesos
maximizando este acuerdo seria ajustar contra fases tacticas de Metrica, que tienen sus
propios errores, y el documento tecnico tendria que justificarlo.

**Como leerlo.** Las etiquetas debiles son fases de ~18 s (ver features/tagging.py para
como se lee cada codigo); un modelo perfecto frame a frame no llegaria a 1.0 contra ellas.
Lo informativo es (a) el acuerdo frente a predecir siempre la clase mayoritaria en los
mismos frames, y (b) que el acuerdo suba con la confianza de la emision. Si sube, la
evidencia es real aunque el promedio no lo muestre.

Portado de DeteccionPosesionBalon/possession_emissions.py::weak_label_agreement.

    python -m mtah.training.weak_labels          # imprime el informe en JSON
"""

import json

import numpy as np

from .. import features as feature_set
from ..config import DEFAULT
from ..constants import graded_mask
from ..decode import viterbi
from ..features import tagging
from ..models import emissions

N_CONFIDENCE_BANDS = 4


def _fraction(values):
    """None, no NaN, para un subconjunto vacio: NaN no es JSON valido."""
    return float(np.mean(values)) if len(values) else None


def agreement(predicts_home, weak_label, comparable, confidence=None):
    """Acuerdo H/A entre una prediccion booleana y las etiquetas debiles.

    predicts_home  (n,) bool, True donde el modelo dice local
    weak_label     (n,) 'H', 'A' o ''
    comparable     (n,) bool, frames que entran en la comparacion
    confidence     (n,) opcional; si se da, se reporta el acuerdo por cuartil de confianza
    """
    comparable = comparable & (weak_label != "")
    if not comparable.any():
        raise ValueError("ningun frame comparable con etiqueta debil")
    label_is_home = weak_label[comparable] == "H"
    correct = predicts_home[comparable] == label_is_home
    report = {
        "frames": int(comparable.sum()),
        "agreement": _fraction(correct),
        # El suelo a superar, medido en los mismos frames.
        "majority_class_agreement": float(max(label_is_home.mean(), 1 - label_is_home.mean())),
        "agreement_on_home_labels": _fraction(correct[label_is_home]),
        "agreement_on_away_labels": _fraction(correct[~label_is_home]),
    }
    if confidence is not None:
        confidence = confidence[comparable]
        edges = np.quantile(confidence, np.linspace(0, 1, N_CONFIDENCE_BANDS + 1))
        band = np.clip(np.searchsorted(edges, confidence, side="right") - 1,
                       0, N_CONFIDENCE_BANDS - 1)
        report["agreement_by_confidence_quartile"] = [
            {"minimum_confidence": round(float(edges[q]), 4),
             "frames": int((band == q).sum()),
             "agreement": _fraction(correct[band == q])}
            for q in range(N_CONFIDENCE_BANDS)]
    return report


def report(tracking, phases, config=None):
    """Acuerdo de la emision por frame y de la salida decodificada con el Smart Tagging."""
    config = config or DEFAULT
    n_frames = tracking.n_frames
    weak_label, conflict = tagging.weak_labels(phases, n_frames)
    graded = graded_mask(n_frames)
    usable = graded & ~conflict

    features = feature_set.build_all(tracking, config)
    home_probability = emissions.home_probability(features, config)
    ball_seen = np.asarray(features["ball_seen"], dtype=bool)
    emission_confidence = np.abs(home_probability - 0.5)

    cost = emissions.costs(features, config)
    transition = viterbi.switch_costs(config["switch_turnover"], config["switch_to_dead"],
                                      config["switch_from_dead"])
    codes = viterbi.enforce_min_duration(viterbi.decode(cost, transition),
                                         config["min_duration_frames"])
    decoded_live = codes != viterbi.D

    return {
        "weak_label_coverage": {
            "graded_frames": int(graded.sum()),
            "labelled_frames": int((graded & (weak_label != "")).sum()),
            "conflicting_frames": int((graded & conflict).sum()),
        },
        # P(local | vivo) de la emision, antes de decodificar. Incluye los frames que el
        # modelo daria por muertos: la pregunta aqui es solo "de quien".
        "emission": agreement(home_probability > 0.5, weak_label, usable,
                              emission_confidence),
        "emission_ball_seen": agreement(home_probability > 0.5, weak_label,
                                        usable & ball_seen, emission_confidence),
        "emission_ball_unseen": agreement(home_probability > 0.5, weak_label,
                                          usable & ~ball_seen, emission_confidence),
        # La entrega: solo frames que el decodificador da por vivos.
        "decoded": agreement(codes == viterbi.H, weak_label, usable & decoded_live),
        "decoded_live_fraction_of_labelled": _fraction(
            decoded_live[usable & (weak_label != "")]),
    }


def main():
    from ..io import atd, smart_tagging
    print(json.dumps(report(atd.load(), smart_tagging.load()), indent=2))


if __name__ == "__main__":
    main()
