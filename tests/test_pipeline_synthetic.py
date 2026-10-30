"""Pipeline extremo a extremo con tracking sintetico.

Existe para que el dia que llegue el `atd.txt` el pipeline ya este probado: aqui se
fabrica un partido falso con geometria conocida y se comprueba que las piezas hacen lo que
dicen y que la salida cumple el contrato de entrega.
"""

import numpy as np
import pytest

from mtah import pipeline, submission
from mtah.config import DEFAULT, merged
from mtah.constants import GRADED_BLOCKS, N_FRAMES, N_TRACK_SLOTS, PITCH
from mtah import features as feature_set
from mtah.features import ball, geometry, smoothing
from mtah.io.atd import Tracking
from mtah.models import emissions


def make_tracking(n_frames, ball_x=0.25, activity=5.0, ball_visible=True,
                  slots=N_TRACK_SLOTS):
    """Partido sintetico: los slots locales pegados al balon, los visitantes lejos."""
    team_of_slot = np.array([0] * (slots // 2) + [1] * (slots - slots // 2), dtype=np.int8)

    players_xy = np.empty((n_frames, slots, 2), dtype=np.float32)
    home = team_of_slot == 0
    players_xy[:, home, 0] = ball_x          # local encima del balon
    players_xy[:, home, 1] = 0.5
    players_xy[:, ~home, 0] = 0.9            # visitante al otro lado
    players_xy[:, ~home, 1] = 0.5

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    if ball_visible:
        ball_xy[:, 0] = ball_x
        ball_xy[:, 1] = 0.5

    return Tracking(
        ball_xy=ball_xy,
        ball_speed=np.full(n_frames, 8.0, dtype=np.float32),
        players_xy=players_xy,
        players_speed=np.full((n_frames, slots), activity, dtype=np.float32),
        team_of_slot=team_of_slot, home_name="Local", away_name="Visitante")


def test_distancias_en_metros_no_en_coordenadas_normalizadas():
    tracking = make_tracking(10, ball_x=0.0)
    tracking.players_xy[:, 0, :] = [1.0, 0.5]        # mismo eje y, extremo opuesto en x
    distances = geometry.ball_distances(tracking)
    assert distances[0, 0] == pytest.approx(PITCH[0], rel=1e-3)


def test_el_equipo_mas_cercano_al_balon_es_el_local():
    tracking = make_tracking(10)
    distances = geometry.ball_distances(tracking)
    nearest = geometry.nearest_per_team(distances, tracking.team_of_slot)
    assert (nearest[:, 0] < nearest[:, 1]).all()


def test_el_control_favorece_al_equipo_cercano():
    tracking = make_tracking(10)
    distances = geometry.ball_distances(tracking)
    control = geometry.control_scores(distances, tracking.team_of_slot,
                                      DEFAULT["control_tau_m"])
    assert (control[:, 0] > control[:, 1]).all()


def test_sin_balon_la_opinion_de_equipo_es_debil():
    """Sin balon no hay control, asi que solo queda la forma de equipo: senal flojita.

    El sintetico pone los dos equipos en una linea vertical, asi que la dispersion en x es
    identica y el sumando de forma se anula: el frame se abstiene del todo y manda el
    decodificador. Eso es lo que se quiere en una oclusion dentro de una posesion.
    """
    tracking = make_tracking(10, ball_visible=False)
    features = feature_set.build_all(tracking, merged())
    assert np.allclose(emissions.home_probability(features), 0.5, atol=0.05)


def test_baja_actividad_implica_balon_parado():
    config = merged()
    quieto = feature_set.build_all(make_tracking(400, activity=0.4), config)
    jugando = feature_set.build_all(make_tracking(400, activity=6.0), config)
    assert emissions.dead_probability(quieto).mean() > 0.8
    assert emissions.dead_probability(jugando).mean() < 0.2


def test_la_velocidad_del_balon_se_deriva_de_la_posicion():
    """La columna de velocidad del fichero es NaN en el 91.3% de los frames con balon."""
    tracking = make_tracking(10)
    tracking.ball_xy[:, 0] = np.linspace(0.0, 0.1, 10)      # avanza 10.53 m en 9 frames
    speed = ball.speed(tracking)
    assert np.isnan(speed[0])                                # no hay t-1
    expected = (0.1 / 9) * PITCH[0] * 25.0
    assert speed[5] == pytest.approx(expected, rel=1e-3)


def test_el_suavizado_es_centrado_y_no_introduce_retraso():
    """Un escalon suavizado tiene que quedar centrado en el escalon, no desplazado."""
    step = np.concatenate([np.zeros(200), np.ones(200)])
    smoothed = smoothing.box_mean(step, 25)
    assert smoothed[200] == pytest.approx(0.5, abs=0.02)
    assert smoothed[174] == pytest.approx(0.0, abs=0.02)
    assert smoothed[226] == pytest.approx(1.0, abs=0.02)


def test_box_nanmean_arrastra_a_los_vecinos_en_un_hueco():
    """Donde el valor es desconocido, promediar los definidos, no rellenar con cero."""
    values = np.full(100, 5.0)
    values[40:60] = np.nan
    smoothed = smoothing.box_nanmean(values, 25)
    assert smoothed[50] == pytest.approx(5.0, abs=1e-5)
    assert not np.isnan(smoothed).any()


def test_huecos_de_balon_se_miden_por_racha():
    tracking = make_tracking(20)
    tracking.ball_xy[5:12, :] = np.nan
    gaps = geometry.ball_gap_lengths(tracking)
    assert gaps[5] == 7 and gaps[11] == 7
    assert gaps[4] == 0 and gaps[12] == 0


def test_balon_fuera_del_campo_se_detecta():
    tracking = make_tracking(10)
    tracking.ball_xy[3:6, 0] = 1.2
    assert geometry.ball_out_of_pitch(tracking)[3:6].all()
    assert not geometry.ball_out_of_pitch(tracking)[0]


def test_extremo_a_extremo_produce_una_entrega_valida(tmp_path):
    """El pipeline completo sobre un partido sintetico de longitud real."""
    tracking = make_tracking(N_FRAMES)
    labels = pipeline.labels_from_tracking(tracking, merged())

    assert len(labels) == N_FRAMES
    path = submission.write(str(tmp_path / "sintetico.csv"), labels)
    assert submission.validate(path) is None

    # El local esta encima del balon todo el partido y los jugadores corren:
    # el pipeline tiene que decir H en los bloques puntuados.
    for start, stop in GRADED_BLOCKS:
        window = labels[start:stop]
        assert (window == "H").mean() > 0.9, (start, np.unique(window, return_counts=True))
