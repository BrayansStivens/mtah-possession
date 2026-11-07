"""Piezas de la curva de visibilidad que no necesitan el IDSSE."""

import numpy as np

from mtah.training import visibility


def test_huecos_solo_donde_la_degradacion_oculto_balon_que_existia():
    clean = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    dirty = np.array([1, 0, 0, 0, 0, 1, 0, 1], dtype=bool)
    assert visibility.hidden_gaps(dirty, clean) == [(1, 3), (4, 5), (6, 7)]


def test_se_devuelven_huecos_enteros_en_orden_y_el_ultimo_parcial():
    gaps = [(0, 4), (10, 12), (20, 30)]
    mask = visibility.frames_to_restore(gaps, order=[2, 0, 1], n_frames=30, n_needed=12)
    assert mask[20:30].all() and mask[0:2].all()
    assert mask.sum() == 12


def test_primero_lo_facil_ordena_por_balon_mas_lejos_de_todo_jugador():
    gaps = [(0, 2), (2, 4), (4, 6)]
    nearest = np.array([1.0, 1.0, 9.0, 9.0, 4.0, 4.0])
    order = visibility.recovery_order(gaps, "easy_first", nearest, np.random.default_rng(0))
    assert list(order) == [1, 2, 0]
