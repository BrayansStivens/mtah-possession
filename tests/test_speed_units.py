"""La velocidad del ATD esta en m/s aunque el XML diga kph. Esto lo fija con datos sinteticos."""

import numpy as np
import pytest

from mtah.constants import FPS, PITCH
from mtah.io import atd


def tracking_with_speed_channel(scale_of_declared_speed, n_frames=200, n_slots=4):
    """Jugadores corriendo en x a velocidades conocidas; el canal declarado se escala."""
    true_speed_ms = np.linspace(2.0, 7.0, n_slots, dtype=np.float32)
    step_normalised = true_speed_ms / FPS / PITCH[0]
    x = 0.1 + np.arange(n_frames, dtype=np.float32)[:, None] * step_normalised[None, :]
    players_xy = np.stack([x, np.full_like(x, 0.5)], axis=-1)
    players_speed = np.tile(true_speed_ms * scale_of_declared_speed, (n_frames, 1))
    # Huecos de tracking, como en el ATD: el cociente tiene que ignorarlos.
    players_xy[50:60, 1] = np.nan
    players_speed[::7, 2] = np.nan
    return atd.Tracking(
        ball_xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
        ball_speed=np.full(n_frames, np.nan, dtype=np.float32),
        players_xy=players_xy.astype(np.float32),
        players_speed=players_speed.astype(np.float32),
        team_of_slot=np.array([0, 0, 1, 1], dtype=np.int8),
        home_name="Local", away_name="Visitante")


def test_metros_por_segundo_pasa_y_devuelve_el_cociente():
    ratio = atd.check_speed_channel_units(tracking_with_speed_channel(1.0))
    assert ratio == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("scale", [3.6, 1 / 3.6])
def test_otra_unidad_falla_en_vez_de_escalar_en_silencio(scale):
    with pytest.raises(ValueError, match="m/s"):
        atd.check_speed_channel_units(tracking_with_speed_channel(scale))


def test_sin_frames_comparables_falla():
    tracking = tracking_with_speed_channel(1.0)
    tracking.players_speed[:] = np.nan
    with pytest.raises(ValueError, match="ningun frame"):
        atd.check_speed_channel_units(tracking)
