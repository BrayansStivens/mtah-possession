"""Descuento por frame en el coste de cambio: la entrada del detector de fronteras."""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

from mtah.config import DEFAULT
from mtah.decode import viterbi
from mtah.paths import REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "vendor", "evaluation", "results")


def baseline_raw_cost():
    with gzip.open(os.path.join(RESULTS, "baseline_raw.csv.gz"), "rt") as handle:
        labels = pd.read_csv(handle)["label"].to_numpy(dtype="<U1")
    return viterbi.costs_from_hard_labels(labels)


def default_transition():
    return viterbi.switch_costs(DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"],
                                DEFAULT["switch_from_dead"])


def test_descuento_cero_reproduce_exactamente_el_decodificador():
    cost = baseline_raw_cost()
    transition = default_transition()
    without = viterbi.decode(cost, transition)
    with_zeros = viterbi.decode(cost, transition, switch_discount=np.zeros(len(cost)))
    assert np.array_equal(without, with_zeros)


def test_el_descuento_lleva_la_frontera_a_donde_apunta_el_detector():
    """Emisiones que no distinguen entre los frames 40 y 60: el corte cae donde se abarata."""
    n_frames = 100
    cost = np.zeros((n_frames, 3))
    cost[:40, viterbi.A] = cost[:40, viterbi.D] = 1.0      # claramente H
    cost[60:, viterbi.H] = cost[60:, viterbi.D] = 1.0      # claramente A
    transition = viterbi.uniform_switch_costs(5.0)
    blocks = ((0, n_frames),)
    for boundary_frame in (45, 52, 58):
        discount = np.zeros(n_frames)
        discount[boundary_frame] = 0.4
        codes = viterbi.decode(cost, transition, blocks=blocks, switch_discount=discount)
        changes = np.flatnonzero(np.diff(codes)) + 1
        assert list(changes) == [boundary_frame]


def test_el_descuento_deja_pasar_una_posesion_corta_que_sin_el_se_borra():
    """Un robo de 1 s que las emisiones ven con poca fuerza: solo sale si el detector ayuda."""
    n_frames = 200
    cost = np.zeros((n_frames, 3))
    cost[:, viterbi.A] = cost[:, viterbi.D] = 1.0          # H en todo el tramo...
    cost[100:125, viterbi.H] = 1.5                          # ...menos 25 frames de A tibios
    cost[100:125, viterbi.A] = 0.0
    transition = viterbi.uniform_switch_costs(20.0)
    blocks = ((0, n_frames),)
    plain = viterbi.decode(cost, transition, blocks=blocks)
    assert (plain == viterbi.H).all()                       # 2 x 20 > 25 x 1.5
    discount = np.zeros(n_frames)
    discount[[100, 125]] = 0.5                              # 2 x 10 < 37.5
    guided = viterbi.decode(cost, transition, blocks=blocks, switch_discount=discount)
    assert (guided[100:125] == viterbi.A).all() and (guided[125:] == viterbi.H).all()


def test_descuento_de_uno_falla_en_vez_de_hacer_gratis_el_cambio():
    discount = np.zeros(10)
    discount[3] = 1.0
    with pytest.raises(ValueError, match="fragmentar"):
        viterbi.decode(np.zeros((10, 3)), viterbi.uniform_switch_costs(5.0),
                       blocks=((0, 10),), switch_discount=discount)


@pytest.mark.parametrize("bad_discount", [np.zeros(9), np.full(10, -0.1), np.full(10, np.nan)])
def test_descuento_invalido_falla(bad_discount):
    with pytest.raises(ValueError, match="switch_discount"):
        viterbi.decode(np.zeros((10, 3)), viterbi.uniform_switch_costs(5.0),
                       blocks=((0, 10),), switch_discount=bad_discount)
