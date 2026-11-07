"""Lectura del Smart Tagging por codigo: los DEFENDING nombran al equipo sin balon."""

import numpy as np

from mtah.constants import AWAY_TEAM, FPS, HOME_TEAM
from mtah.features import tagging
from mtah.io.smart_tagging import Phase


def phase(code, start_frame, end_frame, team):
    tags = {"Team": team} if team else {}
    return Phase(code, start_frame / FPS, end_frame / FPS, tags)


def test_fase_de_posesion_vota_por_su_equipo():
    assert tagging.possession_team(phase("BUILD UP", 0, 10, HOME_TEAM)) == "H"
    assert tagging.possession_team(phase("Progression", 0, 10, AWAY_TEAM)) == "A"


def test_fase_defensiva_vota_por_el_rival():
    assert tagging.possession_team(phase("DEFENDING IN MIDDLE THIRD", 0, 10, HOME_TEAM)) == "A"
    assert tagging.possession_team(phase("DEFENDING IN DEFENSIVE THIRD", 0, 10, AWAY_TEAM)) == "H"


def test_codigos_que_no_dicen_quien_tiene_el_balon_no_votan():
    for code in ("ATTACKING TRANSITION", "BALL IN THE BOX", "SET PIECES", "GOALS"):
        assert tagging.possession_team(phase(code, 0, 10, HOME_TEAM)) is None
    assert tagging.possession_team(phase("BUILD UP", 0, 10, "N/A")) is None
    assert tagging.possession_team(phase("BUILD UP", 0, 10, None)) is None


def test_par_espejo_no_se_cancela():
    """BUILD UP local y DEFENDING del visitante sobre el mismo tramo: los dos dicen H.

    La lectura anterior sumaba +1 y -1 aqui y el frame quedaba en cero.
    """
    phases = [phase("BUILD UP", 0, 10, HOME_TEAM),
              phase("DEFENDING IN ATTACKING THIRD", 0, 10, AWAY_TEAM)]
    logit = tagging.team_logit(phases, n_frames=12)
    assert (logit[:10] == 2).all() and (logit[10:] == 0).all()


def test_etiquetas_debiles_marcan_conflicto_y_huecos():
    phases = [phase("BUILD UP", 0, 10, HOME_TEAM),
              phase("PROGRESSION", 5, 15, AWAY_TEAM)]
    labels, conflict = tagging.weak_labels(phases, n_frames=20)
    assert list(labels[:5]) == ["H"] * 5
    assert conflict[5:10].all() and (labels[5:10] == "").all()
    assert list(labels[10:15]) == ["A"] * 5
    assert (labels[15:] == "").all() and not conflict[15:].any()


def test_frames_fuera_del_partido_se_recortan():
    labels, _ = tagging.weak_labels([phase("BUILD UP", -5, 30, HOME_TEAM)], n_frames=20)
    assert (labels == "H").all()
