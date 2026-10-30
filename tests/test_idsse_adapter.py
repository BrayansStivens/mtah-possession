"""Tests del adaptador IDSSE que no necesitan descargar los 370 MB.

Lo que se comprueba aqui es la parte que puede romperse en silencio: el reparto de slots
y el chequeo de unidades. La carga real se verifica a mano porque depende de la red.
"""

import numpy as np
import pytest

from mtah.io.atd import HOME_CODE, AWAY_CODE
from mtah.training import idsse


class _Player:
    def __init__(self, player_id, team):
        self.player_id = player_id
        self.team = team


class _Frame:
    def __init__(self, players):
        self.players_data = {p: None for p in players}


def test_slots_agrupados_por_equipo_locales_primero():
    home, away = "H", "A"
    players = [_Player("h1", home), _Player("a1", away), _Player("h2", home)]
    frames = [_Frame(players)]

    slot_of, team_of_slot = idsse._allocate_slots(frames, home)

    assert len(slot_of) == 3
    assert list(team_of_slot) == [HOME_CODE, HOME_CODE, AWAY_CODE]
    # Los dos locales ocupan los slots contiguos del principio.
    assert sorted([slot_of["h1"], slot_of["h2"]]) == [0, 1]
    assert slot_of["a1"] == 2


def test_un_jugador_que_aparece_en_varios_frames_ocupa_un_solo_slot():
    home = "H"
    p = _Player("h1", home)
    frames = [_Frame([p]), _Frame([p]), _Frame([p])]

    slot_of, team_of_slot = idsse._allocate_slots(frames, home)

    assert len(slot_of) == 1 and len(team_of_slot) == 1


def test_suplentes_reciben_slot_propio_al_entrar():
    home = "H"
    titular, suplente = _Player("h1", home), _Player("h2", home)
    frames = [_Frame([titular]), _Frame([titular]), _Frame([suplente])]

    slot_of, _ = idsse._allocate_slots(frames, home)

    assert slot_of["h1"] != slot_of["h2"]


def test_el_chequeo_de_unidades_acepta_velocidades_en_ms():
    # Carrera normal mas algun sprint: todo por debajo de 12 m/s.
    speed = np.random.default_rng(0).uniform(0.0, 9.0, size=(1000, 22)).astype(np.float32)
    idsse._check_speed_units(speed)          # no debe lanzar


def test_el_chequeo_de_unidades_rechaza_kmh_sin_convertir():
    """La red de seguridad del fallo caro: si kloppy deja de entregar km/h, saltar."""
    speed = np.random.default_rng(0).uniform(0.0, 33.0, size=(1000, 22)).astype(np.float32)
    with pytest.raises(ValueError, match="implausible"):
        idsse._check_speed_units(speed)


def test_el_chequeo_de_unidades_falla_si_no_hay_ninguna_velocidad():
    with pytest.raises(ValueError, match="finita"):
        idsse._check_speed_units(np.full((10, 4), np.nan, dtype=np.float32))


def test_el_factor_de_conversion_es_el_medido():
    """3.6 sale del ratio p99 fichero/derivada medido sobre J03WPY: 3.590."""
    assert idsse.KMH_TO_MS == pytest.approx(1.0 / 3.6)


def test_los_siete_partidos_publicados_estan_listados():
    assert len(idsse.MATCHES) == 7
    assert "J03WPY" in idsse.MATCHES


def test_un_partido_inexistente_falla_con_la_lista():
    with pytest.raises(KeyError, match="J03WPY"):
        idsse.load(match_id="NO_EXISTE")
