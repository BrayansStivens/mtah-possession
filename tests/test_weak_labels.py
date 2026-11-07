"""Diagnostico de acuerdo con el Smart Tagging, sobre datos sinteticos."""

import numpy as np
import pytest

from mtah.training import weak_labels


def test_acuerdo_suelo_mayoritario_y_cuartiles():
    weak_label = np.array(["H"] * 6 + ["A"] * 2 + [""] * 2)
    predicts_home = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0], dtype=bool)
    confidence = np.array([.9, .8, .7, .6, .1, .2, .5, .05, 0, 0])
    report = weak_labels.agreement(predicts_home, weak_label,
                                   np.ones(10, dtype=bool), confidence)
    assert report["frames"] == 8                      # los '' no cuentan
    assert report["agreement"] == pytest.approx(5 / 8)
    assert report["majority_class_agreement"] == pytest.approx(6 / 8)
    assert report["agreement_on_home_labels"] == pytest.approx(4 / 6)
    assert report["agreement_on_away_labels"] == pytest.approx(1 / 2)
    bands = report["agreement_by_confidence_quartile"]
    assert sum(band["frames"] for band in bands) == 8
    assert bands[-1]["agreement"] == 1.0               # lo mas confiado acierta


def test_sin_frames_comparables_falla():
    with pytest.raises(ValueError):
        weak_labels.agreement(np.ones(3, bool), np.array(["", "", ""]), np.ones(3, bool))
