"""Enlazado de picos del detector en trayectorias (sin video: candidatos sinteticos)."""

import numpy as np
import pytest

from mtah.vision import track


def _candidates(n, ball_path, clutter_rng=None, ball_score=0.8, clutter=2):
    """n frames con el balon en ball_path y `clutter` picos falsos al azar por frame."""
    rng = clutter_rng or np.random.default_rng(0)
    xy = np.zeros((n, 1 + clutter, 2))
    scores = np.zeros((n, 1 + clutter))
    xy[:, 0] = ball_path
    scores[:, 0] = ball_score
    xy[:, 1:] = rng.uniform([0, 0], [105, 68], size=(n, clutter, 2))
    scores[:, 1:] = 0.5
    return np.arange(n), xy, scores


def test_sigue_un_balon_entre_ruido_y_descarta_el_ruido():
    n = 100
    path = np.stack([np.linspace(30, 50, n), np.full(n, 34.0)], axis=1)
    frames, xy, scores = _candidates(n, path)
    filled = track.select(track.link(frames, xy, scores), n)
    assert np.isfinite(filled[:, 0]).all()
    assert np.abs(filled - path).max() < 1e-9


def test_la_puerta_fisica_corta_un_salto_imposible():
    n = 40
    path = np.stack([np.full(n, 30.0), np.full(n, 34.0)], axis=1)
    path[20:] = [80.0, 10.0]                      # 53 m en un frame
    frames, xy, scores = _candidates(n, path, clutter=0)
    tracklets = track.link(frames, xy, scores)
    assert len([t for t in tracklets if len(t.frames) >= track.MIN_DETECTIONS]) == 2


def test_interpola_huecos_cortos_y_no_inventa_fuera_del_tramo():
    n = 60
    path = np.stack([np.linspace(10, 20, n), np.full(n, 30.0)], axis=1)
    frames, xy, scores = _candidates(n, path, clutter=0)
    keep = np.ones(n, dtype=bool)
    keep[30:34] = False                           # 4 frames sin deteccion
    filled = track.select(track.link(frames[keep], xy[keep], scores[keep]), n + 20)
    assert np.allclose(filled[30:34], path[30:34])
    assert np.isnan(filled[n:, 0]).all()


def test_tramos_cortos_o_debiles_no_se_quedan():
    n = 5
    path = np.stack([np.full(n, 30.0), np.full(n, 34.0)], axis=1)
    frames, xy, scores = _candidates(n, path, clutter=0)
    assert np.isnan(track.select(track.link(frames, xy, scores), n)[:, 0]).all()


def test_to_field_deshace_la_homografia():
    H = np.array([[[1000.0, 50.0, 100.0], [20.0, 800.0, 200.0], [0.01, 0.02, 1.0]]])
    field = np.array([[0.25, 0.5], [0.8, 0.1]])
    denominator = H[0, 2, 0] * field[:, 0] + H[0, 2, 1] * field[:, 1] + 1.0
    pixel = np.stack([(1000 * field[:, 0] + 50 * field[:, 1] + 100) / denominator,
                      (20 * field[:, 0] + 800 * field[:, 1] + 200) / denominator], axis=1)
    metres = track.to_field(H, 0, pixel)
    assert metres == pytest.approx(field * np.array([105.3, 68.0]))
