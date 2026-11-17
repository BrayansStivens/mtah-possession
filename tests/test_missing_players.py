"""Tests for the thinning of players where the ball is missing and for the decision rule of
the test."""

import numpy as np
import pytest

from mtah.io.atd import HOME_CODE, AWAY_CODE, Tracking
from mtah.training import degrade, missing_players


def _tracking(n=6000, n_players=22, seed=0, unseen_every=3):
    """All the players always; the ball is missing in one of every `unseen_every` stretches
    of 100."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, n_players, 2)).astype(np.float32)
    sp = rng.uniform(0.0, 8.0, size=(n, n_players)).astype(np.float32)
    ball = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    unseen = (np.arange(n) // 100) % unseen_every == 0
    ball[unseen] = np.nan
    team = np.array([HOME_CODE] * (n_players // 2) + [AWAY_CODE] * (n_players // 2),
                    dtype=np.int8)
    return Tracking(ball_xy=ball, ball_speed=np.where(unseen, np.nan, 5.0).astype(np.float32),
                    players_xy=xy, players_speed=sp, team_of_slot=team,
                    home_name="Local", away_name="Visitante")


def _spec(values, probs):
    tr = _tracking(n=600)
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    spec.n_tracked_unseen_values = np.asarray(values, dtype=np.int32)
    spec.n_tracked_unseen_probs = np.asarray(probs, dtype=np.float64)
    return spec


def test_estimate_mide_el_histograma_sin_balon():
    tr = _tracking()
    xy = tr.players_xy.copy()
    xy[~tr.ball_seen, :4] = np.nan                       # 18 are seen without the ball
    tr = Tracking(tr.ball_xy, tr.ball_speed, xy, tr.players_speed, tr.team_of_slot,
                  tr.home_name, tr.away_name)
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    assert spec.n_tracked_unseen_values.tolist() == [18]
    assert spec.summary()["objetos_media_sin_balon"] == pytest.approx(18.0)


def test_estimate_sin_huecos_no_tiene_histograma_sin_balon():
    tr = _tracking()
    full = Tracking(np.zeros_like(tr.ball_xy), np.zeros_like(tr.ball_speed), tr.players_xy,
                    tr.players_speed, tr.team_of_slot, tr.home_name, tr.away_name)
    spec = degrade.estimate(full, ((0, full.n_frames),))
    assert spec.n_tracked_unseen_values is None
    with pytest.raises(ValueError):
        degrade.thin_where_ball_unseen(full, spec, np.random.default_rng(0))


def test_solo_quita_y_solo_donde_falta_el_balon():
    tr = _tracking()
    spec = _spec([15, 19, 21], [0.2, 0.5, 0.3])
    out = degrade.thin_where_ball_unseen(tr, spec, np.random.default_rng(0))
    before = np.isfinite(tr.players_xy[:, :, 0])
    after = np.isfinite(out.players_xy[:, :, 0])
    assert not (after & ~before).any()                                  # never invents
    assert np.array_equal(out.players_xy[tr.ball_seen], tr.players_xy[tr.ball_seen])
    assert np.array_equal(out.ball_xy, tr.ball_xy, equal_nan=True)
    # The ones that remain keep their position and speed.
    assert np.array_equal(out.players_xy[after], tr.players_xy[after])
    assert np.array_equal(np.isfinite(out.players_speed), after)


def test_iguala_el_histograma_objetivo_sin_balon():
    tr = _tracking()
    spec = _spec([15, 19, 21], [0.2, 0.5, 0.3])
    out = degrade.thin_where_ball_unseen(tr, spec, np.random.default_rng(1))
    counts = np.isfinite(out.players_xy[~tr.ball_seen, :, 0]).sum(axis=1)
    assert counts.mean() == pytest.approx(0.2 * 15 + 0.5 * 19 + 0.3 * 21, abs=0.15)
    assert (counts < 16).mean() == pytest.approx(0.2, abs=0.03)


def test_persistente_quita_los_mismos_slots_en_toda_la_racha():
    tr = _tracking()
    spec = _spec([18], [1.0])                             # 4 fewer in each ball-absent frame
    out = degrade.thin_where_ball_unseen(tr, spec, np.random.default_rng(0), persistent=True)
    gone = ~np.isfinite(out.players_xy[:, :, 0])
    for start in range(0, tr.n_frames, 300):              # ball-absent runs of 100 frames
        run = gone[start:start + 100]
        assert run.sum(axis=1).tolist() == [4] * 100
        assert (run == run[0]).all()
    # Different runs draw again.
    assert not all((gone[s] == gone[0]).all() for s in range(300, tr.n_frames, 300))


def test_persistente_y_por_frame_dan_el_mismo_histograma():
    tr = _tracking()
    spec = _spec([15, 19, 21], [0.2, 0.5, 0.3])
    counts = {}
    for persistent in (False, True):
        out = degrade.thin_where_ball_unseen(tr, spec, np.random.default_rng(1),
                                             persistent=persistent)
        counts[persistent] = np.isfinite(out.players_xy[~tr.ball_seen, :, 0]).sum(axis=1)
    assert np.array_equal(np.sort(counts[False]), np.sort(counts[True]))


def test_por_frame_parpadea():
    tr = _tracking()
    out = degrade.thin_where_ball_unseen(tr, _spec([18], [1.0]), np.random.default_rng(0))
    gone = ~np.isfinite(out.players_xy[:100, :, 0])
    assert not (gone == gone[0]).all()


def test_donde_ya_hay_menos_no_se_toca():
    tr = _tracking()
    xy = tr.players_xy.copy()
    xy[:, 12:] = np.nan                                  # only 12 in the whole match
    tr = Tracking(tr.ball_xy, tr.ball_speed, xy, tr.players_speed, tr.team_of_slot,
                  tr.home_name, tr.away_name)
    out = degrade.thin_where_ball_unseen(tr, _spec([19], [1.0]), np.random.default_rng(0))
    assert np.array_equal(out.players_xy, tr.players_xy, equal_nan=True)


def test_no_modifica_el_original():
    tr = _tracking()
    before = tr.players_xy.copy()
    degrade.thin_where_ball_unseen(tr, _spec([10], [1.0]), np.random.default_rng(0))
    assert np.array_equal(tr.players_xy, before, equal_nan=True)


def test_apply_no_cambia_con_el_histograma_nuevo():
    """The usual degradation has to come out the same: the Nov 12 measurement depends on it."""
    tr = _tracking(n=1500)
    spec = degrade.estimate(tr, ((0, tr.n_frames),))
    a = degrade.apply(tr, spec, rng=np.random.default_rng(3))
    spec.n_tracked_unseen_values = spec.n_tracked_unseen_probs = None
    b = degrade.apply(tr, spec, rng=np.random.default_rng(3))
    assert np.array_equal(a.players_xy, b.players_xy, equal_nan=True)
    assert np.array_equal(a.ball_seen, b.ball_seen)


@pytest.mark.parametrize("deltas, verdict", [
    ([8.0] * 21, "mantener"),
    ([5.0] * 15 + [-1.0] * 6, "mantener, con la ganancia reducida documentada"),
    ([12.0] * 16 + [-3.0] * 5, "mantener"),
    ([3.0] * 21, "mantener, con la ganancia reducida documentada"),
    ([1.0] * 21, "volver a peso 0"),
    ([20.0] * 10 + [-1.0] * 11, "volver a peso 0"),
    ([-2.0] * 21, "volver a peso 0"),
])
def test_regla_de_decision(deltas, verdict):
    assert missing_players.decide(deltas)["verdict"] == verdict


def test_decide_el_brazo_persistente_salvo_que_el_facil_ya_diga_volver():
    keep, revert = {"verdict": "mantener"}, {"verdict": "volver a peso 0"}
    assert missing_players.final_verdict({"thinned": keep, "persistent": revert}) \
        == "volver a peso 0"
    assert missing_players.final_verdict({"thinned": revert, "persistent": keep}) \
        == "volver a peso 0"
    assert missing_players.final_verdict({"thinned": keep, "persistent": keep}) == "mantener"


def test_desequilibrio_entre_equipos():
    n_tracked = np.array([20, 20, 18, 22])
    n_home = np.array([10, 10, 12, 11])             # without the ball: |24-18| = 6, |22-22| = 0
    seen = np.array([True, True, False, False])
    counts = missing_players.player_counts(n_tracked, n_home, seen)
    assert counts["team_gap_unseen"] == pytest.approx(3.0)
    assert counts["mean_ball_unseen"] == pytest.approx(20.0)
