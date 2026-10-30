"""El decodificador. Incluye una regresion contra numeros medidos con el scorer oficial."""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

from mtah import evaluate
from mtah.constants import GRADED_BLOCKS
from mtah.decode import viterbi
from mtah.paths import REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "vendor", "evaluation", "results")


def baseline(name):
    with gzip.open(os.path.join(RESULTS, name + ".csv.gz"), "rt") as handle:
        return pd.read_csv(handle)["label"].to_numpy(dtype="<U1")


def n_segments(codes):
    return len(viterbi.segment_table(codes))


def test_mas_coste_de_cambio_implica_menos_segmentos():
    cost = viterbi.costs_from_hard_labels(baseline("baseline_raw"))
    counts = [n_segments(viterbi.decode(cost, viterbi.uniform_switch_costs(c)))
              for c in (5, 20, 80, 320)]
    assert counts == sorted(counts, reverse=True), counts


def test_no_decodifica_atravesando_el_descanso():
    cost = viterbi.costs_from_hard_labels(baseline("baseline_raw"))
    codes = viterbi.decode(cost, viterbi.uniform_switch_costs(40))
    gap_start, gap_stop = GRADED_BLOCKS[0][1], GRADED_BLOCKS[1][0]
    assert (codes[gap_start:gap_stop] == viterbi.D).all()
    for start, stop, _, _ in viterbi.segment_table(codes):
        assert any(a <= start and stop <= b for a, b in GRADED_BLOCKS), (start, stop)


def test_min_duration_elimina_los_segmentos_cortos():
    cost = viterbi.costs_from_hard_labels(baseline("baseline_raw"))
    codes = viterbi.decode(cost, viterbi.uniform_switch_costs(5))
    cleaned = viterbi.enforce_min_duration(codes, 50)
    durations = [b - a for a, b, _, _ in viterbi.segment_table(cleaned)]
    assert min(durations) >= 50, min(durations)


def test_min_duration_no_toca_nada_si_el_minimo_es_uno():
    codes = np.array([0, 0, 1, 2, 2], dtype=np.int8)
    assert np.array_equal(viterbi.enforce_min_duration(codes, 1, ((0, 5),)), codes)


def test_transicion_asimetrica_es_una_matriz_valida():
    matrix = viterbi.switch_costs(turnover=30, to_dead=60, from_dead=10)
    assert matrix[viterbi.H, viterbi.A] == 30
    assert matrix[viterbi.A, viterbi.D] == 60
    assert matrix[viterbi.D, viterbi.H] == 10
    assert np.diag(matrix).tolist() == [0, 0, 0]


@pytest.mark.parametrize("name,spell,macro", [
    ("baseline_raw", 0.0378, 0.5972),
    ("baseline_smooth50", 0.3256, 0.6207),
    ("baseline_smooth125", 0.4242, 0.5812),
])
def test_reproduce_la_tabla_oficial(name, spell, macro):
    """El scorer vendorizado da los mismos numeros que publica el repo oficial."""
    result = evaluate.headline(baseline(name))
    assert result["spellF1"] == pytest.approx(spell, abs=1e-4)
    assert result["macroF1"] == pytest.approx(macro, abs=1e-4)


def test_viterbi_domina_al_voto_por_mayoria():
    """Mismo input que el baseline, mejor decodificador -> mejor en las DOS metricas.

    Numeros medidos sobre la ventana publica. No es el valor de switch_cost a usar en el
    partido completo (eso seria ajustar contra los 5 minutos, el error que EVALUATION.md
    demuestra), pero el mecanismo si tiene que seguir sosteniendose.
    """
    smooth50 = evaluate.headline(baseline("baseline_smooth50"))
    cost = viterbi.costs_from_hard_labels(baseline("baseline_raw"))
    codes = viterbi.decode(cost, viterbi.uniform_switch_costs(40))
    decoded = evaluate.headline(viterbi.to_labels(codes))

    assert decoded["spellF1"] > smooth50["spellF1"], (decoded, smooth50)
    assert decoded["macroF1"] > smooth50["macroF1"], (decoded, smooth50)
