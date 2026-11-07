"""Piezas del detector de fronteras que no necesitan el IDSSE."""

import numpy as np

from mtah.features.boundaries import window_means
from mtah.training import boundaries


def test_objetivo_marca_el_radio_alrededor_de_cada_cambio():
    labels = np.array(["H"] * 20 + ["A"] * 20 + ["D"] * 20)
    target = boundaries.boundary_target(labels, half_width=3)
    assert list(np.flatnonzero(target)) == [17, 18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42]


def test_objetivo_sin_cambios_es_todo_falso():
    assert not boundaries.boundary_target(np.array(["H"] * 10), half_width=3).any()


def test_ventanas_antes_y_despues_no_incluyen_el_otro_lado():
    values = np.array([0.0] * 10 + [1.0] * 10)
    before, after = window_means(values, 4)
    assert before[10] == 0.0 and after[10] == 1.0          # justo en el cambio
    assert before[14] == 1.0 and after[6] == 0.0
    assert np.isnan(before[0])                              # nada antes del primer frame


def test_ventanas_ignoran_nan_y_dan_nan_si_no_hay_nada():
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
    before, after = window_means(values, 2)
    assert before[3] == 3.0                                  # [1, 3): NaN, 3
    assert after[0] == 1.0                                   # [0, 2): 1, NaN
    assert np.isnan(after[3])                                # [3, 5): todo NaN


def test_los_nombres_de_columna_no_dependen_de_los_datos():
    from mtah.features.boundaries import feature_names
    names = feature_names()
    assert len(names) == len(set(names)) == 109


def test_el_modelo_guardado_corresponde_a_estas_features_y_esta_config():
    import json

    from mtah.config import DEFAULT
    from mtah.features.boundaries import feature_names
    from mtah.models import boundary_detector
    with open(boundary_detector.METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["feature_names"] == feature_names()
    assert metadata["feature_config"] == boundary_detector.feature_config(DEFAULT)
    assert metadata["chosen_discount_weight"] == DEFAULT["boundary_discount_weight"]
    assert metadata["chosen_switch_scale"] == DEFAULT["boundary_switch_scale"]


def test_cambiar_una_emision_sin_reentrenar_falla():
    import pytest

    from mtah import features as feature_set
    from mtah.config import merged
    from mtah.models import boundary_detector
    from test_pipeline_synthetic import make_tracking
    config = merged(control_weight=7.0)
    features = feature_set.build_all(make_tracking(200), config)
    with pytest.raises(ValueError, match="control_weight"):
        boundary_detector.score(features, config)


def test_cambiar_solo_el_decodificador_no_invalida_el_detector():
    from mtah import features as feature_set
    from mtah.config import merged
    from mtah.models import boundary_detector
    from test_pipeline_synthetic import make_tracking
    config = merged(switch_turnover=55.0, min_duration_frames=25)
    features = feature_set.build_all(make_tracking(200), config)
    score = boundary_detector.score(features, config)
    assert score.shape == (200,) and ((score >= 0) & (score <= 1)).all()
