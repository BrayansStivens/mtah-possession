"""Puntuacion local con el scorer *oficial* vendorizado en vendor/evaluation.

No reimplementamos la metrica a proposito: `possession_metrics.py` es la especificacion y
el fichero que decide la competicion. Si nuestro numero y el suyo discrepan, el bug es
nuestro.

Recordatorio de EVALUATION.md, que conviene tener delante cada vez que se mira un numero
de aqui: la ventana publica son 5 minutos y 17 posesiones. **No sirve para elegir entre
dos versiones de tu modelo**, solo para comprobar que el pipeline es correcto. Una
posesion fallada mueve spellF1 unos 6 puntos aqui. Diferencias por debajo de ~0.04 en
spellF1 (y ~0.01 en macroF1) son ruido incluso sobre el partido completo.
"""

import os
import sys

import numpy as np

from .paths import REPO_ROOT

VENDOR = os.path.join(REPO_ROOT, "vendor", "evaluation")
PUBLIC_GT = os.path.join(VENDOR, "gt", "BAR-ATH_possession_gt_public.csv")

if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)
import possession_metrics as pm                              # noqa: E402


def public_gt():
    """Codigos del ground truth publico (5 minutos reales, el resto X)."""
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def score(labels, gt_codes=None, bootstrap=0):
    """Dict con todas las metricas oficiales para un array de etiquetas."""
    gt_codes = public_gt() if gt_codes is None else gt_codes
    pred = pm.label_codes(np.asarray(labels, dtype="<U1"))
    return pm.evaluate(gt_codes, pred, bootstrap_n=bootstrap)


def headline(labels, gt_codes=None, bootstrap=0):
    """Los numeros que importan, en un dict pequeno."""
    result = score(labels, gt_codes, bootstrap)
    return {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_pred_poss": result["n_pred_spells"],
        "n_gt_poss": result["n_gt_spells"],
        "mean_dur_s": result["mean_spell_duration_s"],
        "gt_mean_dur_s": result["gt_mean_spell_duration_s"],
        "f1_D": result["per_class"]["D"]["f1"],
        "latency_mean": result["transitions"][25]["latency_mean"]
        if "latency_mean" in result["transitions"][25] else None,
    }


def report(labels, name="submission", gt_codes=None, bootstrap=0):
    """Imprime un resumen de una linea mas el desglose por clase."""
    result = score(labels, gt_codes, bootstrap)
    print("{}  spellF1 {:.4f}  macroF1 {:.4f}   poss {} vs {} (gt)   dur {:.1f}s vs {:.1f}s"
          .format(name, result["spell_f1_mean"], result["macro_f1"],
                  result["n_pred_spells"], result["n_gt_spells"],
                  result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    for label in ("H", "A", "D"):
        cls = result["per_class"][label]
        print("   {}  P {:.4f}  R {:.4f}  F1 {:.4f}".format(
            label, cls["precision"], cls["recall"], cls["f1"]))
    per_iou = result["spell_f1"]
    print("   spellF1 por IoU: " + "  ".join(
        "@{} {:.4f}".format(k, v["f1"]) for k, v in sorted(per_iou.items())))
    return result
