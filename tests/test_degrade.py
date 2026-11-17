"""Tests for the degradation. The failure that hurts most here is a silent one: producing
data that does not look like the ATD and trusting whatever number comes out, so the
marginals are checked."""

import numpy as np
import pytest

from mtah.io.atd import HOME_CODE, AWAY_CODE, Tracking
from mtah.training import degrade


def _clean(n=4000, n_players=22, seed=0):
    """Toy clean tracking: every player always visible, ball always visible."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, n_players, 2)).astype(np.float32)
    sp = rng.uniform(0.0, 8.0, size=(n, n_players)).astype(np.float32)
    team = np.array([HOME_CODE] * (n_players // 2) + [AWAY_CODE] * (n_players // 2),
                    dtype=np.int8)
    return Tracking(ball_xy=rng.uniform(0, 1, (n, 2)).astype(np.float32),
                    ball_speed=rng.uniform(0, 20, n).astype(np.float32),
                    players_xy=xy, players_speed=sp, team_of_slot=team,
                    home_name="Local", away_name="Visitante")


def test_runs_cuenta_rachas():
    assert list(degrade._runs(np.array([0, 1, 1, 0, 1, 0, 0], dtype=bool))) == [2, 1]
    assert list(degrade._runs(np.array([1, 1, 1], dtype=bool))) == [3]
    assert list(degrade._runs(np.zeros(5, dtype=bool))) == []


def test_estimate_mide_lo_que_dice_medir():
    tr = _clean()
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    # Clean tracking: the ball is never missing and every slot is always present.
    assert len(spec.ball_absent_runs) == 0
    assert spec.n_tracked_values.tolist() == [22]
    assert spec.summary()["objetos_media"] == pytest.approx(22.0)


def test_apply_produce_los_46_slots_del_reto():
    tr = _clean()
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    out = degrade.apply(tr, spec, rng=np.random.default_rng(0))
    assert out.players_xy.shape[1] == degrade.N_SLOTS_HOME + degrade.N_SLOTS_AWAY == 46
    assert (out.team_of_slot == HOME_CODE).sum() == 24
    assert (out.team_of_slot == AWAY_CODE).sum() == 22
    assert out.n_frames == tr.n_frames


def test_la_mascara_de_balon_respeta_la_tasa_objetivo():
    """The marginal has to come out right, because it is the statistic that defines the problem."""
    rng = np.random.default_rng(0)
    present = np.full(200, 60, dtype=np.int32)
    absent = np.full(200, 40, dtype=np.int32)
    mask = degrade._alternating_mask(50_000, present, absent, rng)
    assert (~mask).mean() == pytest.approx(0.4, abs=0.02)


def test_la_mascara_produce_rachas_largas_no_bernoulli():
    """The classic failure: drawing every frame independently gives runs of 1.7 frames when the
    real ones have a median of 268. A model trained against 2-frame gaps interpolates."""
    rng = np.random.default_rng(0)
    present = np.full(50, 600, dtype=np.int32)
    absent = np.full(50, 400, dtype=np.int32)
    mask = degrade._alternating_mask(50_000, present, absent, rng)
    assert degrade._runs(~mask).mean() > 100


def test_el_reparto_de_slots_no_pierde_la_mitad_de_los_objetos():
    """Regression for the real bug: allocating player by player instead of in temporal order
    left 6 objects per frame instead of 20.5."""
    tr = _clean()
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    out = degrade.apply(tr, spec, rng=np.random.default_rng(0), position_noise=False)
    visible = np.isfinite(out.players_xy[:, :, 0]).sum(axis=1)
    assert visible.mean() > 0.9 * 22


def test_apply_no_modifica_el_original():
    tr = _clean()
    before = tr.ball_xy.copy()
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    degrade.apply(tr, spec, rng=np.random.default_rng(0))
    assert np.array_equal(tr.ball_xy, before, equal_nan=True)


def test_mascara_ponderada_respeta_la_tasa_y_cae_donde_pesa():
    n = 20000
    present = np.array([300, 500, 800], dtype=np.int32)
    absent = np.array([100, 200, 400], dtype=np.int32)       # 700 / 2300 ball absent
    weights = np.full(n, 0.001)
    weights[5000:10000] = 1.0                                  # 25% of the match
    seen = degrade._weighted_mask(n, present, absent, weights, np.random.default_rng(0))
    assert abs((~seen).mean() - 700 / 2300) < 0.005
    zone = np.zeros(n, dtype=bool)
    zone[5000:10000] = True
    assert (~seen[zone]).mean() > 0.9 and (~seen[~zone]).mean() < 0.15


@pytest.mark.parametrize("bad", [np.zeros(100), -np.ones(100), np.full(100, np.nan),
                                 np.ones(99)])
def test_mascara_ponderada_rechaza_pesos_invalidos(bad):
    with pytest.raises(ValueError):
        degrade._weighted_mask(100, np.array([5]), np.array([5]), bad,
                               np.random.default_rng(0))


def test_apply_sin_pesos_es_el_proxy_de_siempre():
    clean = _clean(n=1500)
    spec = degrade.estimate(clean, ((0, 1500),))
    a = degrade.apply(clean, spec, rng=np.random.default_rng(3))
    b = degrade.apply(clean, spec, rng=np.random.default_rng(3), gap_weights=None)
    assert np.array_equal(a.ball_seen, b.ball_seen)
