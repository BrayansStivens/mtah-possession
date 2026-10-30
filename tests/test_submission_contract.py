"""El contrato del fichero de entrega. Un fallo aqui es puntuar cero."""

import numpy as np
import pytest

from mtah import submission
from mtah.constants import ABSTAIN, DEAD, GRADED_BLOCKS, KICKOFF_2ND, N_FRAMES


def all_dead():
    return np.full(N_FRAMES, DEAD, dtype="<U1")


def test_build_longitud_y_etiquetas():
    labels = submission.build(all_dead())
    assert len(labels) == N_FRAMES
    assert set(np.unique(labels)) <= {"H", "A", "D", "X"}


def test_fuera_de_bloques_es_abstencion():
    labels = submission.build(all_dead())
    assert labels[0] == ABSTAIN
    assert labels[GRADED_BLOCKS[0][0] - 1] == ABSTAIN
    assert labels[N_FRAMES - 1] == ABSTAIN


def test_nunca_hay_abstencion_en_frames_puntuados():
    labels = submission.build(all_dead())
    for start, stop in GRADED_BLOCKS:
        assert not (labels[start:stop] == ABSTAIN).any()


def test_build_rechaza_abstencion_dentro_de_bloques():
    bad = all_dead()
    bad[GRADED_BLOCKS[0][0] + 100] = ABSTAIN
    with pytest.raises(ValueError, match="dominada"):
        submission.build(bad)


def test_segundo_tiempo_antes_del_saque_es_balon_parado():
    source = np.full(N_FRAMES, "H", dtype="<U1")
    labels = submission.build(source)
    second_start = GRADED_BLOCKS[1][0]
    assert (labels[second_start:KICKOFF_2ND] == DEAD).all()
    assert labels[KICKOFF_2ND] == "H"


def test_build_acepta_un_array_por_bloque():
    chunks = [np.full(stop - start, "A", dtype="<U1") for start, stop in GRADED_BLOCKS]
    labels = submission.build(chunks)
    assert labels[GRADED_BLOCKS[0][0]] == "A"
    assert labels[GRADED_BLOCKS[1][1] - 1] == "A"


def test_escribir_y_validar(tmp_path):
    path = tmp_path / "equipo.csv"
    submission.write(str(path), submission.build(all_dead()))
    assert submission.validate(str(path)) is None
    assert path.read_text().splitlines()[0] == "frame,label"


def test_validate_detecta_fichero_corto(tmp_path):
    path = tmp_path / "corto.csv"
    path.write_text("frame,label\n0,D\n1,D\n")
    assert "filas" in submission.validate(str(path))


def test_validate_detecta_etiqueta_invalida(tmp_path):
    path = tmp_path / "mala.csv"
    labels = submission.build(all_dead())
    labels[5] = "D"
    submission.write(str(path), labels)
    raw = path.read_text().replace("5,D", "5,Z", 1)
    path.write_text(raw)
    assert "fuera de" in submission.validate(str(path))
