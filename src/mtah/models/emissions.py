"""De features por frame a una matriz de costes (n, 3) para el decodificador.

Modelo v1: geometrico, interpretable, sin entrenamiento. Cada peso sale del AUC medido de
su senal, no de ajustar la metrica.

La estructura es un producto de dos decisiones, que es como esta hecho el problema:

    P(H) = P(vivo) * P(local | vivo)
    P(A) = P(vivo) * P(visitante | vivo)
    P(D) = P(muerto)

y el coste es -log P. Separarlo asi importa porque las dos mitades se apoyan en senales
distintas. **Vivo/muerto** sale de la cinematica de los jugadores y no necesita el balon.
**La propiedad** sale sobre todo de la geometria balon-jugador, que si lo necesita. Con el
balon ausente en el 41% de los frames puntuados, mezclarlas en un solo clasificador tira a
la basura la mitad que sigue funcionando.

Donde no hay balon, la propiedad se decide solo con la forma de equipo, que es una senal
debil (AUC 0.8124) y por eso produce una probabilidad cercana a 0.5. El coste de
transicion del decodificador arrastra entonces el estado anterior, que es exactamente lo
que se quiere: pasar una oclusion en silencio dentro de una posesion es gratis en spellF1,
mientras que adivinar puede partir la posesion en dos.
"""

import numpy as np

from ..config import DEFAULT
from ..decode.viterbi import A, D, H, N_STATES


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def dead_logit(features, config=None):
    """(n,) logit de "el balon esta muerto". Suma de evidencias independientes."""
    config = config or DEFAULT

    activity = np.nan_to_num(features["activity_s"], nan=config["activity_center_ms"])
    logit = config["activity_weight"] * (
        (config["activity_center_ms"] - activity) / config["activity_scale_ms"])

    # Indefinida donde el balon no se ve en toda la ventana: ahi no aporta nada y se
    # neutraliza poniendola en su propio centro.
    ball_speed = np.nan_to_num(features["ball_speed_s"], nan=config["ball_speed_center_ms"])
    logit = logit + config["ball_speed_weight"] * (
        (config["ball_speed_center_ms"] - ball_speed) / config["ball_speed_scale_ms"])

    logit = logit + config["ball_unseen_logit"] * (~features["ball_seen"]).astype(np.float32)
    logit = logit + config["ball_out_logit"] * features["ball_out"].astype(np.float32)
    return logit


def dead_probability(features, config=None):
    return _sigmoid(dead_logit(features, config))


def home_logit(features, config=None):
    """(n,) logit de "el local tiene el balon", dado que esta vivo.

    Dos sumandos: el control balon-jugador, que es el fuerte pero solo existe donde hay
    balon, y la asimetria de forma de equipo, que es debil pero existe siempre.
    """
    config = config or DEFAULT

    # NaN donde no hay balon en ninguna parte de la ventana -> el sumando se anula.
    control = features["control_diff_s"]
    logit = np.where(np.isnan(control), 0.0,
                     config["control_weight"] * (np.nan_to_num(control) - config["control_center"]))

    shape = np.nan_to_num(features["spread_ratio_s"], nan=config["shape_center"])
    logit = logit + config["shape_weight"] * (shape - config["shape_center"])
    return logit.astype(np.float64)


def home_probability(features, config=None):
    return _sigmoid(home_logit(features, config))


def team_tag_logit(team_tags, config=None):
    """(n,) lo que el Smart Tagging suma al logit de local: peso * (+1 local, -1 visitante, 0).

    Va aqui y no dentro de `home_logit` a proposito: `home_logit` es una entrada del
    detector de fronteras, que se entreno sin tagging (el IDSSE no lo tiene). Sumarlo alli
    cambiaria lo que el detector ve. Ver training/team_tags.py para el peso.
    """
    config = config or DEFAULT
    team_tags = np.asarray(team_tags)
    if team_tags.ndim != 1 or not np.isin(team_tags, (-1, 0, 1)).all():
        raise ValueError("team_tags tiene que ser (n,) con valores en {-1, 0, +1}")
    return config["smart_tagging_team_weight"] * team_tags.astype(np.float64)


def costs_from_logits(home, dead, config=None):
    """(n, 3) coste por frame y estado a partir de los dos logits. Columnas H, A, D."""
    config = config or DEFAULT
    floor = float(config["prob_floor"])

    dead = np.clip(_sigmoid(dead), floor, 1.0 - floor)
    home = np.clip(_sigmoid(home), floor, 1.0 - floor)
    alive = 1.0 - dead

    out = np.empty((len(dead), N_STATES), dtype=np.float64)
    out[:, H] = -np.log(alive * home)
    out[:, A] = -np.log(alive * (1.0 - home))
    out[:, D] = -np.log(dead)
    return out


def costs(features, config=None, team_tags=None):
    """(n, 3) coste por frame y estado, menor es mejor. Columnas en orden H, A, D.

    `team_tags=None` es "no hay Smart Tagging" (el IDSSE); la entrega lo pasa siempre.
    """
    config = config or DEFAULT
    home = home_logit(features, config)
    if team_tags is not None:
        if len(team_tags) != len(home):
            raise ValueError("team_tags tiene %d frames y los features %d"
                             % (len(team_tags), len(home)))
        home = home + team_tag_logit(team_tags, config)
    return costs_from_logits(home, dead_logit(features, config), config)
