"""End-to-end pipeline with synthetic tracking.

It exists so that the day `atd.txt` arrives the pipeline is already tested: here a fake
match with known geometry is built and we check that the pieces do what they say and that
the output meets the submission contract.
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
    """Synthetic match: the home slots glued to the ball, the away ones far away."""
    team_of_slot = np.array([0] * (slots // 2) + [1] * (slots - slots // 2), dtype=np.int8)

    players_xy = np.empty((n_frames, slots, 2), dtype=np.float32)
    home = team_of_slot == 0
    players_xy[:, home, 0] = ball_x          # home on top of the ball
    players_xy[:, home, 1] = 0.5
    players_xy[:, ~home, 0] = 0.9            # away on the other side
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
    tracking.players_xy[:, 0, :] = [1.0, 0.5]        # same y axis, opposite end in x
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
    """Without the ball there is no control, so only team shape is left: a faint signal.

    The synthetic match puts both teams on a vertical line, so the spread in x is
    identical and the shape term cancels out: the frame abstains entirely and the
    decoder takes over. That is what is wanted in an occlusion inside a possession.
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
    """The file's speed column is NaN in 91.3% of the frames with the ball."""
    tracking = make_tracking(10)
    tracking.ball_xy[:, 0] = np.linspace(0.0, 0.1, 10)      # advances 10.53 m in 9 frames
    speed = ball.speed(tracking)
    assert np.isnan(speed[0])                                # there is no t-1
    expected = (0.1 / 9) * PITCH[0] * 25.0
    assert speed[5] == pytest.approx(expected, rel=1e-3)


def test_el_suavizado_es_centrado_y_no_introduce_retraso():
    """A smoothed step has to stay centered on the step, not shifted."""
    step = np.concatenate([np.zeros(200), np.ones(200)])
    smoothed = smoothing.box_mean(step, 25)
    assert smoothed[200] == pytest.approx(0.5, abs=0.02)
    assert smoothed[174] == pytest.approx(0.0, abs=0.02)
    assert smoothed[226] == pytest.approx(1.0, abs=0.02)


def test_box_nanmean_arrastra_a_los_vecinos_en_un_hueco():
    """Where the value is unknown, average the defined ones, do not fill in with zero."""
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
    """The full pipeline on a synthetic match of real length."""
    tracking = make_tracking(N_FRAMES)
    labels = pipeline.labels_from_tracking(tracking, merged())

    assert len(labels) == N_FRAMES
    path = submission.write(str(tmp_path / "sintetico.csv"), labels)
    assert submission.validate(path) is None

    # The home team is on top of the ball the whole match and the players run:
    # the pipeline has to say H in the graded blocks.
    for start, stop in GRADED_BLOCKS:
        window = labels[start:stop]
        assert (window == "H").mean() > 0.9, (start, np.unique(window, return_counts=True))
