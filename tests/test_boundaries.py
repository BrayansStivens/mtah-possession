"""Piezas del detector de fronteras que no necesitan el IDSSE."""

import numpy as np

from mtah.training import boundaries


def test_objetivo_marca_el_radio_alrededor_de_cada_cambio():
    labels = np.array(["H"] * 20 + ["A"] * 20 + ["D"] * 20)
    target = boundaries.boundary_target(labels, half_width=3)
    assert list(np.flatnonzero(target)) == [17, 18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42]


def test_objetivo_sin_cambios_es_todo_falso():
    assert not boundaries.boundary_target(np.array(["H"] * 10), half_width=3).any()


def test_ventanas_antes_y_despues_no_incluyen_el_otro_lado():
    values = np.array([0.0] * 10 + [1.0] * 10)
    before, after = boundaries.window_means(values, 4)
    assert before[10] == 0.0 and after[10] == 1.0          # justo en el cambio
    assert before[14] == 1.0 and after[6] == 0.0
    assert np.isnan(before[0])                              # nada antes del primer frame


def test_ventanas_ignoran_nan_y_dan_nan_si_no_hay_nada():
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
    before, after = boundaries.window_means(values, 2)
    assert before[3] == 3.0                                  # [1, 3): NaN, 3
    assert after[0] == 1.0                                   # [0, 2): 1, NaN
    assert np.isnan(after[3])                                # [3, 5): todo NaN
