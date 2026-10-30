"""Ensamblado de features: el unico sitio que decide que entra al modelo.

Dos familias, separadas a proposito porque fallan en momentos distintos:

- **Dependientes del balon** (control, distancia al mas cercano, velocidad del balon).
  Las mas informativas y las menos disponibles: el balon falta en el 43% de los frames.
- **Independientes del balon** (actividad de los jugadores, forma de equipo). Son las que
  tienen que sostener el 41% de frames puntuados sin balon.

AUC medido sobre este partido contra el ground truth publico:

    detectar balon muerto        actividad suav +-50f   0.8452
                                 vel. balon suav +-25f  0.7986
                                 balon no visible       0.7700
                                 n objetos trackeados   0.5987  <- descartado
    distinguir equipo (vivos)    control suav +-50f     0.9091
                                 spread_ratio_x         0.8124  <- sin usar el balon
                                 gap de centroides x    0.5294  <- descartado
"""

import numpy as np

from . import ball, geometry, smoothing


def build_all(tracking, config):
    """Dict de arrays de longitud n_frames, listo para models.emissions."""
    features = {
        "ball_seen": tracking.ball_seen,
        "ball_x": tracking.ball_xy[:, 0],
        "ball_y": tracking.ball_xy[:, 1],
        "ball_out": geometry.ball_out_of_pitch(tracking),
        "ball_gap": geometry.ball_gap_lengths(tracking),
    }

    # --- vivo / muerto, sin depender del balon ---
    activity = geometry.player_activity(tracking)
    features["activity"] = activity
    features["activity_s"] = smoothing.box_mean(activity, config["activity_smooth"])

    # La velocidad del balon del fichero es NaN en el 91.3% de los frames en que se ve,
    # asi que se deriva de la posicion. box_nanmean, no box_mean: donde no hay balon el
    # valor es desconocido, no cero.
    ball_speed = ball.speed(tracking)
    features["ball_speed"] = ball_speed
    features["ball_speed_s"] = smoothing.box_nanmean(ball_speed, config["ball_speed_smooth"])
    features["ball_displacement"] = ball.displacement(tracking, config["ball_speed_smooth"])

    # --- propiedad del equipo ---
    distances = geometry.ball_distances(tracking)
    control = geometry.control_scores(distances, tracking.team_of_slot, config["control_tau_m"])
    nearest = geometry.nearest_per_team(distances, tracking.team_of_slot)
    features["control_home"], features["control_away"] = control[:, 0], control[:, 1]
    features["nearest_home"], features["nearest_away"] = nearest[:, 0], nearest[:, 1]

    # Indefinido donde no hay balon, para que el suavizado arrastre a los vecinos en lugar
    # de diluir hacia "nadie controla".
    difference = np.where(tracking.ball_seen, control[:, 0] - control[:, 1], np.nan)
    features["control_diff_s"] = smoothing.box_nanmean(difference, config["control_smooth"])

    shape = geometry.team_shape(tracking)
    features.update(shape)
    # box_nanmean: el ratio es NaN donde no significa nada (un equipo con menos de 3
    # jugadores vistos), y ahi hay que promediar los vecinos definidos, no rellenar.
    features["spread_ratio_s"] = smoothing.box_nanmean(shape["spread_ratio_x"],
                                                       config["shape_smooth"])
    return features
