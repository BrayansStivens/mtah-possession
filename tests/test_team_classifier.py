"""Team classifier without the ball: signals, antisymmetry, artifact and hook into the pipeline.

Nothing here needs the IDSSE or the challenge match.
"""

import json

import numpy as np
import pytest

from mtah.config import DEFAULT, merged
from mtah.features import team_signals
from mtah.io.atd import Tracking
from mtah.models import boundary_detector, emissions, team_classifier
from mtah.training import team_classifier as training


def _match(n=600, seed=0, flip_at=None):
    """Two teams of 11 that move smoothly; the home team defends x=0 until `flip_at`."""
    rng = np.random.default_rng(seed)
    team_of_slot = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    base = np.concatenate([rng.uniform([0.05, 0.1], [0.55, 0.9], (11, 2)),
                           rng.uniform([0.45, 0.1], [0.95, 0.9], (11, 2))])
    drift = np.cumsum(rng.normal(0, 0.0015, (n, 22, 2)), axis=0)
    players_xy = np.clip(base[None] + drift, 0.0, 1.0).astype(np.float32)
    if flip_at is not None:
        players_xy[flip_at:, :, 0] = 1.0 - players_xy[flip_at:, :, 0]
    players_xy[rng.random((n, 22)) < 0.03] = np.nan
    return Tracking(
        ball_xy=np.full((n, 2), np.nan, dtype=np.float32),
        ball_speed=np.full(n, np.nan, dtype=np.float32),
        players_xy=players_xy,
        players_speed=rng.uniform(0, 7, (n, 22)).astype(np.float32),
        team_of_slot=team_of_slot, home_name="Local", away_name="Visitante")


def _swapped(tracking):
    return Tracking(ball_xy=tracking.ball_xy, ball_speed=tracking.ball_speed,
                    players_xy=tracking.players_xy, players_speed=tracking.players_speed,
                    team_of_slot=(1 - tracking.team_of_slot).astype(np.int8),
                    home_name=tracking.away_name, away_name=tracking.home_name)


def _features(n=50):
    rng = np.random.default_rng(0)
    return {"activity_s": rng.uniform(0, 6, n), "ball_speed_s": rng.uniform(0, 20, n),
            "ball_seen": rng.random(n) > 0.4, "ball_out": np.zeros(n, dtype=bool),
            "control_diff_s": rng.normal(0, 1, n), "spread_ratio_s": rng.uniform(0.8, 1.2, n)}


# --------------------------------------------------------------------------- signals

def test_los_nombres_de_columna_no_dependen_de_los_datos():
    names = team_signals.feature_names()
    assert len(names) == len(set(names)) == 57
    stack = team_signals.smoothed(_match(200))
    assert team_signals.matrix(stack, 0).shape == (200, 57)


def test_el_sentido_de_ataque_sale_de_los_centroides():
    direction, report = team_signals.attack_direction(_match(400))
    assert (direction == 1.0).all() and report[0]["agreement"] == 1.0


def test_el_sentido_de_ataque_encuentra_el_descanso_sin_que_se_lo_digan():
    tracking = _match(600, flip_at=250)
    direction, report = team_signals.attack_direction(tracking)
    assert (direction[:250] == 1.0).all() and (direction[250:] == -1.0).all()
    assert [row["block"] for row in report] == [[0, 250], [250, 600]]


def test_con_las_partes_dadas_el_descanso_hereda_el_sentido_de_la_parte_que_viene():
    tracking = _match(600, flip_at=250)
    direction, _ = team_signals.attack_direction(tracking, blocks=((20, 240), (300, 600)))
    assert (direction[:240] == 1.0).all() and (direction[240:] == -1.0).all()


def test_las_senales_estan_en_el_eje_de_ataque_de_cada_equipo():
    """After halftime the pitch is flipped and each team's signals do not change."""
    tracking = _match(600, flip_at=300)
    direction, _ = team_signals.attack_direction(tracking)
    raw = team_signals.raw(tracking, direction)
    for team in (0, 1):
        # Second-deepest player: in its own half (<0) in both halves.
        assert np.nanmean(raw[team]["dfl"][:290]) < 0 and np.nanmean(raw[team]["dfl"][310:]) < 0


def test_con_menos_de_tres_jugadores_la_senal_es_nan():
    tracking = _match(100)
    tracking.players_xy[40:60, 2:11] = np.nan           # 2 home players left
    direction, _ = team_signals.attack_direction(tracking)
    raw = team_signals.raw(tracking, direction)
    assert np.isnan(raw[0]["wid"][40:60]).all() and not np.isnan(raw[1]["wid"][40:60]).any()


# --------------------------------------------------------------------------- model

def test_cambiar_los_equipos_cambia_el_signo_del_logit():
    tracking = _match(500, seed=3)
    logit = team_classifier.home_logit(tracking)
    assert logit.shape == (500,) and logit.dtype == np.float32 and np.isfinite(logit).all()
    assert np.array_equal(team_classifier.home_logit(_swapped(tracking)), -logit)


def test_el_modelo_guardado_corresponde_a_estas_senales_y_esta_config():
    with open(team_classifier.METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["feature_names"] == team_signals.feature_names()
    assert metadata["feature_settings"] == team_classifier.feature_settings()
    assert metadata["chosen_weight"] == DEFAULT["team_classifier_weight"]
    assert metadata["model_params"] == training.MODEL_PARAMS
    assert metadata["train_stride_frames"] == training.TRAIN_STRIDE_FRAMES
    assert [row[2] for row in metadata["grid"]] == list(training.WEIGHTS)


def test_cambiar_una_senal_sin_reentrenar_falla(monkeypatch):
    monkeypatch.setattr(team_signals, "SPRINT_SPEED_MS", 6.0)
    with pytest.raises(ValueError, match="team_classifier"):
        team_classifier.home_logit(_match(100))


# --------------------------------------------------------------------------- emissions

def test_peso_cero_es_el_modelo_sin_clasificador():
    features = _features()
    logit = np.random.default_rng(1).normal(0, 2, 50).astype(np.float32)
    config = merged(team_classifier_weight=0.0)
    assert np.array_equal(emissions.costs(features, config, team_logit=logit),
                          emissions.costs(features, config))


def test_el_logit_abarata_el_equipo_que_senala():
    features = _features()
    config = merged(team_classifier_weight=2.0)
    logit = np.zeros(50, dtype=np.float32)
    logit[:10], logit[10:20] = 1.5, -1.5
    base, pushed = emissions.costs(features, config), emissions.costs(features, config,
                                                                      team_logit=logit)
    # <= and not <: a frame that is already at the probability floor cannot get any cheaper.
    assert (pushed[:10, 0] <= base[:10, 0]).all() and (pushed[10:20, 1] <= base[10:20, 1]).all()
    assert (pushed[:10, 0] < base[:10, 0]).any() and (pushed[10:20, 1] < base[10:20, 1]).any()
    assert np.array_equal(pushed[20:], base[20:])
    assert np.array_equal(pushed[:, 2], base[:, 2])     # it does not touch the dead-ball cost


@pytest.mark.parametrize("bad", [np.array([0.0, np.nan, 0.0]), np.zeros((3, 1))])
def test_logits_invalidos_fallan(bad):
    with pytest.raises(ValueError):
        emissions.team_classifier_logit(bad, merged())


def test_un_logit_de_otra_longitud_falla():
    with pytest.raises(ValueError, match="team_logit"):
        emissions.costs(_features(50), merged(), team_logit=np.zeros(49))


# --------------------------------------------------------------------------- hook

def test_el_peso_no_invalida_el_detector_de_fronteras():
    assert "team_classifier_weight" in boundary_detector.DECODER_ONLY_KEYS
    assert (boundary_detector.feature_config(merged(team_classifier_weight=0.0))
            == boundary_detector.feature_config(DEFAULT))


def test_con_peso_cero_el_pipeline_no_carga_el_clasificador(monkeypatch):
    from mtah import pipeline
    from mtah.constants import N_FRAMES
    from test_pipeline_synthetic import make_tracking

    def forbidden(*args, **kwargs):
        raise AssertionError("should not be loaded")
    monkeypatch.setattr(team_classifier, "home_logit", forbidden)
    labels = pipeline.labels_from_tracking(make_tracking(N_FRAMES),
                                           merged(team_classifier_weight=0.0))
    assert len(labels) == N_FRAMES


def test_decode_score_con_peso_cero_es_la_medicion_del_detector():
    """`decode_score` without a classifier does not recompute costs: it uses the detector's."""
    from mtah.training import boundaries
    rng = np.random.default_rng(0)
    n = 2000
    labels = np.array((["H"] * 200 + ["A"] * 150 + ["D"] * 50) * 5)
    home = np.where(labels == "H", 1.0, -1.0) + rng.normal(0, 1.5, n)
    dead = np.where(labels == "D", 2.0, -2.0) + rng.normal(0, 1.0, n)
    base = {"cost": emissions.costs_from_logits(home, dead, DEFAULT), "labels": labels,
            "home_logit": home, "dead_logit": dead.astype(np.float32)}
    score = np.zeros(n)
    control = training.decode_score(base, np.zeros(n, np.float32), score, DEFAULT, 0.0)
    expected = boundaries.score_labels(
        boundaries.decode(base, DEFAULT, DEFAULT["boundary_discount_weight"],
                          DEFAULT["boundary_switch_scale"], score), base)
    assert control == expected
    truth = np.where(labels == "H", 3.0, -3.0).astype(np.float32)
    helped = training.decode_score(base, truth, score, DEFAULT, 2.0)
    assert helped["macro_f1"] >= control["macro_f1"]
