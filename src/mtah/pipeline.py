"""El pipeline completo: tracking -> features -> emisiones -> decodificacion -> entrega.

Un solo sitio donde se encadena todo, para que `python -m mtah build` sea literalmente el
comando que regenera el fichero entregado. El entregable lo exige: ejecutar este repo
contra los datos tiene que reproducir el CSV, no aproximadamente sino el mismo fichero.
"""

import numpy as np

from . import submission
from .config import DEFAULT
from .decode import viterbi
from . import features as feature_set
from .features import tagging
from .io import atd, smart_tagging
from .models import boundary_detector, emissions


def labels_from_tracking(tracking, config=None, team_tags=None):
    """Etiquetas de longitud completa a partir de un objeto Tracking ya cargado.

    `team_tags`: el Smart Tagging leido con `features.tagging.team_tags`, o None sin el.
    """
    config = config or DEFAULT

    features = feature_set.build_all(tracking, config)
    cost = emissions.costs(features, config, team_tags=team_tags)
    transition = viterbi.switch_costs(turnover=config["switch_turnover"],
                                      to_dead=config["switch_to_dead"],
                                      from_dead=config["switch_from_dead"])
    discount = None
    if config["boundary_discount_weight"] > 0:
        transition = config["boundary_switch_scale"] * transition
        discount = boundary_detector.switch_discount(features, config)
    codes = viterbi.decode(cost, transition, switch_discount=discount)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"])
    return submission.build(viterbi.to_labels(codes))


def run(config=None, source="tactical", use_cache=True):
    """Carga el tracking y devuelve las etiquetas de la entrega."""
    tracking = atd.load(source=source, use_cache=use_cache)
    team_tags = tagging.team_tags(smart_tagging.load(source=source), tracking.n_frames)
    return labels_from_tracking(tracking, config, team_tags)
