"""Tests del armonizador de convenio. Son puros, no necesitan datos ni red."""

import numpy as np

from mtah.training import convention as cv


def _labels(*pairs):
    """_labels(("H", 10), ("A", 2), ("H", 10)) -> array de 22 etiquetas."""
    return np.array([c for lab, n in pairs for c in lab * n], dtype="<U1")


def test_spells_detecta_rachas_maximas():
    assert cv.spells(_labels(("H", 3), ("A", 2), ("D", 1))) == [
        ("H", 0, 3), ("A", 3, 5), ("D", 5, 6)]


def test_spells_de_array_vacio():
    assert cv.spells(np.array([], dtype="<U1")) == []


def test_absorbe_un_contacto_sin_control_dentro_de_una_posesion():
    """CHALLENGE.md: un simple toque sin control no es cambio de posesion. 10 frames = 0,4 s."""
    out, info = cv.harmonise(_labels(("H", 250), ("A", 10), ("H", 250)), fps=25)
    assert set(np.unique(out)) == {"H"}
    assert info["n_poss_antes"] == 3 and info["n_poss_despues"] == 1


def test_respeta_una_posesion_controlada_del_rival():
    """4 s con el balon es posesion controlada segun el proxy de la regla: NO se absorbe."""
    labels = _labels(("H", 250), ("A", 100), ("H", 250))
    out, info = cv.harmonise(labels, fps=25)
    assert np.array_equal(out, labels)
    assert info["frames_cambiados"] == 0


def test_el_umbral_es_el_de_la_regla_y_es_estricto():
    exactly = int(cv.MIN_CONTROLLED_S * 25)
    kept, _ = cv.harmonise(_labels(("H", 250), ("A", exactly), ("H", 250)), fps=25)
    merged, _ = cv.harmonise(_labels(("H", 250), ("A", exactly - 1), ("H", 250)), fps=25)
    assert "A" in set(np.unique(kept))
    assert set(np.unique(merged)) == {"H"}


def test_no_absorbe_un_cambio_real_de_posesion():
    """Un tramo breve entre DOS equipos distintos es un cambio de verdad, no un contacto."""
    labels = _labels(("H", 250), ("A", 10), ("D", 100), ("A", 250))
    out, _ = cv.harmonise(labels, fps=25)
    assert "A" in set(np.unique(out[250:260]))


def test_el_balon_parado_nunca_se_toca():
    """La tasa de D del IDSSE ya es comparable: ahi no hay discrepancia que arreglar."""
    labels = _labels(("H", 200), ("D", 5), ("H", 200), ("D", 300), ("A", 400))
    out, _ = cv.harmonise(labels, fps=25)
    assert (out == "D").sum() == (labels == "D").sum()


def test_con_balon_parado_entre_posesiones_no_hay_nada_que_fusionar():
    """No se puentea sobre D. Si todas las posesiones van separadas por D, no cambia nada,
    y lo dice en vez de inventarse algo."""
    pieces = []
    for i in range(100):
        pieces.append(("HA"[i % 2], 50))
        pieces.append(("D", 100))
    out, info = cv.harmonise(_labels(*pieces), fps=25)
    assert info["frames_cambiados"] == 0
    assert info["dur_media_despues_s"] == info["dur_media_antes_s"]


def test_la_duracion_media_solo_puede_subir():
    rng = np.random.default_rng(0)
    pieces = []
    for _ in range(300):
        pieces.append((rng.choice(["H", "A"]), int(rng.integers(20, 200))))
        if rng.random() < 0.4:
            pieces.append(("D", int(rng.integers(50, 400))))
    out, info = cv.harmonise(_labels(*pieces), fps=25)
    assert info["dur_media_despues_s"] >= info["dur_media_antes_s"]
    assert len(out) == sum(n for _, n in pieces)


def test_no_inventa_ni_pierde_frames():
    labels = _labels(("H", 100), ("A", 7), ("H", 100), ("D", 50))
    out, _ = cv.harmonise(labels, fps=25)
    assert len(out) == len(labels)
    assert set(np.unique(out)) <= {"H", "A", "D"}
