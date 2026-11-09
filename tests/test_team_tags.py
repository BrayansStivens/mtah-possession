"""Smart Tagging como evidencia de equipo: lectura, emision y tagging simulado."""

import numpy as np
import pytest

from mtah.config import merged
from mtah.features import tagging
from mtah.io.smart_tagging import Phase
from mtah.models import emissions
from mtah.training import team_tags


def _phase(code, start_s, end_s, team):
    return Phase(code, start_s, end_s, {"Team": team})


def test_team_tags_lee_el_codigo_y_no_cuenta_doble_los_espejos():
    from mtah.constants import AWAY_TEAM, HOME_TEAM
    phases = [_phase("BUILD UP", 0, 2, HOME_TEAM),
              _phase("DEFENDING IN MIDDLE THIRD", 0, 2, AWAY_TEAM),     # espejo: local
              _phase("DEFENDING IN MIDDLE THIRD", 4, 6, HOME_TEAM),     # balon: visitante
              _phase("BALL IN THE BOX", 8, 10, HOME_TEAM)]              # no dice de quien
    tags = tagging.team_tags(phases, n_frames=300)
    assert tags.dtype == np.int8
    assert set(tags[0:50]) == {1} and set(tags[100:150]) == {-1}
    assert not tags[200:250].any()


def _features(n=50):
    rng = np.random.default_rng(0)
    return {"activity_s": rng.uniform(0, 6, n), "ball_speed_s": rng.uniform(0, 20, n),
            "ball_seen": rng.random(n) > 0.4, "ball_out": np.zeros(n, dtype=bool),
            "control_diff_s": rng.normal(0, 1, n), "spread_ratio_s": rng.uniform(0.8, 1.2, n)}


def test_peso_cero_es_el_modelo_sin_tagging():
    features = _features()
    tags = np.random.default_rng(1).integers(-1, 2, 50).astype(np.int8)
    assert np.array_equal(emissions.costs(features, merged(smart_tagging_team_weight=0.0), tags),
                          emissions.costs(features, merged(smart_tagging_team_weight=0.0)))


def test_el_tag_abarata_el_equipo_que_nombra():
    features = _features()
    config = merged(smart_tagging_team_weight=2.0)
    tags = np.zeros(50, dtype=np.int8)
    tags[:10], tags[10:20] = 1, -1
    base, tagged = emissions.costs(features, config), emissions.costs(features, config, tags)
    # <= y no <: un frame que ya esta en el suelo de probabilidad no puede abaratarse mas.
    assert (tagged[:10, 0] <= base[:10, 0]).all() and (tagged[10:20, 1] <= base[10:20, 1]).all()
    assert (tagged[:10, 0] < base[:10, 0]).any() and (tagged[10:20, 1] < base[10:20, 1]).any()
    assert np.array_equal(tagged[20:], base[20:])


def test_costs_from_logits_reproduce_costs():
    features = _features()
    config = merged()
    assert np.array_equal(emissions.costs_from_logits(emissions.home_logit(features, config),
                                                      emissions.dead_logit(features, config),
                                                      config),
                          emissions.costs(features, config))


@pytest.mark.parametrize("bad", [np.array([2, 0, 0]), np.zeros((3, 1))])
def test_team_tags_invalidos_fallan(bad):
    with pytest.raises(ValueError):
        emissions.team_tag_logit(bad, merged(smart_tagging_team_weight=1.0))


def _draws(values_flip, values_echo=None):
    values_flip = np.asarray(values_flip, dtype=float)
    echo = np.ones(len(values_flip)) if values_echo is None else np.asarray(values_echo, float)
    return {"flip": values_flip, "echo": echo}


def test_tagging_simulado_cubre_lo_pedido_con_tramos_disjuntos_y_separados():
    labels = np.array((["H"] * 300 + ["D"] * 50 + ["A"] * 300) * 10)
    home_logit = np.where(labels == "H", 1.0, -1.0)
    profile = {"coverage": 0.3, "run_lengths": np.array([100, 150, 200])}
    windows = team_tags.simulate_windows(labels, home_logit, profile, np.random.default_rng(0))
    tags = team_tags.tags_from_windows(windows, _draws(np.ones(len(windows))), len(labels))
    assert abs((tags != 0).mean() - 0.3) < 0.05
    edges = np.flatnonzero(tags[1:] != tags[:-1])
    assert not ((tags[edges] != 0) & (tags[edges + 1] != 0)).any()    # nunca H->A directo
    for start, stop, truth, _ in windows:                                # equipo mayoritario
        inside = labels[start:stop]
        assert truth == (1 if (inside == "H").sum() >= (inside == "A").sum() else -1)


def test_subir_flip_solo_anade_inversiones():
    windows = [(0, 10, 1, 1), (20, 30, -1, -1), (40, 50, 1, 1)]
    draws = _draws([0.05, 0.15, 0.5])
    low = team_tags.tags_from_windows(windows, draws, 60, flip=0.1)
    high = team_tags.tags_from_windows(windows, draws, 60, flip=0.2)
    assert list(low[[0, 20, 40]]) == [-1, -1, 1]
    assert list(high[[0, 20, 40]]) == [-1, 1, 1]


def test_eco_copia_el_equipo_de_nuestra_emision():
    windows = [(0, 10, 1, -1), (20, 30, -1, 1)]          # la emision se equivoca en los dos
    draws = _draws([1.0, 1.0], [0.3, 0.7])
    tags = team_tags.tags_from_windows(windows, draws, 40, echo=0.5)
    assert tags[0] == -1 and tags[20] == -1               # el primero copia el error


def test_exponente_lleva_los_tramos_a_donde_la_emision_esta_segura():
    labels = np.array(["H"] * 4000)
    home_logit = np.full(4000, 0.01)
    home_logit[2000:3000] = 5.0
    profile = {"coverage": 0.1, "run_lengths": np.array([50])}
    windows = team_tags.simulate_windows(labels, home_logit, profile,
                                         np.random.default_rng(0), exponent=4.0)
    centers = np.array([(a + b) // 2 for a, b, _, _ in windows])
    assert ((centers >= 1950) & (centers < 3050)).mean() > 0.9
