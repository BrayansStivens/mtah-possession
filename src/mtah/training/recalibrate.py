"""Re-derivacion de los hiperparametros sobre partidos EXTERNOS, sin tocar la ventana publica.

**Por que existe.** Desde el 22-10-2026 RULES.md dice que los 5 minutos publicos *no son
conjunto de ajuste* ("a rule, not just advice"). Dieciseis valores de `config.DEFAULT`
se habian derivado de ahi: medianas por clase, pesos por AUC, ventanas de suavizado, costes
de transicion y duracion minima. Aqui se vuelven a derivar con dos fuentes permitidas:

1. **Medianas por clase** medidas sobre los partidos del IDSSE *degradados* como el ATD
   (training/degrade.py), con las etiquetas llevadas al convenio del reto
   (training/convention.py). Otros partidos, etiquetas reales, misma distribucion que el ATD.
2. **Busqueda por coordenadas** del resto de valores maximizando el spellF1 medio sobre un
   subconjunto de partidos de AJUSTE, y comprobacion en otros de VALIDACION que no se han
   tocado, para saber si el arnes se ha sobreajustado a si mismo.

Los partidos de ajuste y de validacion estan fijados aqui, no se eligen por resultado.

**Que no hace.** No toca `activity_scale_ms` ni `ball_speed_scale_ms` (IQR del ATD, sin
etiquetas: limpios), ni `control_tau_m`, `count_radii_m` o `prob_floor` (fisicos o
arbitrarios). Y no toca el pipeline de produccion: el resultado es un JSON en `out/` que
alguien transcribe a `config.py` con su procedencia al lado, como pide ese fichero.

Uso:  python -m mtah.training.recalibrate   (necesita requirements-train.txt y el ATD)
"""

import json
import os
import sys
import time

import numpy as np

from .. import evaluate as ev
from .. import features as feature_set
from ..config import DEFAULT, merged
from ..decode import viterbi
from ..features import smoothing
from ..models import emissions
from ..paths import REPO_ROOT, cache_dir
from . import convention, degrade, harness, idsse

TUNE = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")   # 4 partidos: se ajusta aqui
HOLD = ("J03WPY", "J03WQQ", "J03WR9")             # 3 partidos: solo se mira al final

# Features crudos que no dependen de ningun hiperparametro barrido. Todo lo que si depende
# (suavizado, emisiones, decodificacion) se recalcula por configuracion, que es barato.
RAW_KEYS = ("activity", "ball_speed", "ball_seen", "ball_out",
            "control_home", "control_away", "spread_ratio_x")

# Rejillas. El orden importa en descenso por coordenadas: primero lo que el diagnostico
# (figura 3) senala como causa probable de perder las posesiones cortas.
GRID = [
    ("activity_smooth",     [6, 12, 25, 50, 100]),
    ("control_smooth",      [6, 12, 25, 50, 100]),
    ("shape_smooth",        [12, 25, 50, 100, 200]),
    ("ball_speed_smooth",   [6, 12, 25, 50]),
    ("min_duration_frames", [0, 6, 12, 25, 50]),
    ("switch_turnover",     [8, 14, 20, 28, 40]),
    ("switch_to_dead",      [10, 18, 26, 34, 45]),
    ("switch_from_dead",    [4, 8, 14, 22]),
    ("control_weight",      [2, 4, 6, 9, 13]),
    ("shape_weight",        [0.0, 0.5, 1.0, 1.6, 3.0]),
    ("activity_weight",     [0.5, 1.0, 1.5, 2.5]),
    ("ball_speed_weight",   [0.0, 0.3, 0.6, 1.2]),
    ("ball_unseen_logit",   [0.0, 0.25, 0.5, 1.0]),
    ("activity_center_ms",  [1.10, 1.18, 1.26, 1.35]),
]


def log(msg):
    print(msg, flush=True)


def prepare(match_id, spec, seed=0):
    """(features crudos, etiquetas) de un partido degradado, cacheado en out/cache."""
    path = os.path.join(cache_dir(), "recal_{}_s{}.npz".format(match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        return {k: blob[k] for k in RAW_KEYS}, blob["gt"]
    clean, labels = idsse.load(match_id)
    gt, _ = convention.harmonise(labels)
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed))
    feats = feature_set.build_all(dirty, DEFAULT)
    raw = {k: np.asarray(feats[k]) for k in RAW_KEYS}
    np.savez_compressed(path, gt=gt, **raw)
    return raw, gt


def _smooth(raw, cfg):
    """Lo mismo que hace features.build_all con las ventanas, sobre los crudos cacheados."""
    diff = np.where(raw["ball_seen"], raw["control_home"] - raw["control_away"],
                    np.nan).astype(np.float32)
    return {
        "activity_s": smoothing.box_mean(raw["activity"], cfg["activity_smooth"]),
        "ball_speed_s": smoothing.box_nanmean(raw["ball_speed"], cfg["ball_speed_smooth"]),
        "ball_seen": raw["ball_seen"],
        "ball_out": raw["ball_out"],
        "control_diff_s": smoothing.box_nanmean(diff, cfg["control_smooth"]),
        "spread_ratio_s": smoothing.box_nanmean(raw["spread_ratio_x"], cfg["shape_smooth"]),
    }


class Match:
    def __init__(self, match_id, raw, gt):
        self.id = match_id
        self.raw = raw
        self.gt_codes = ev.pm.label_codes(np.asarray(gt, dtype="<U1"))
        self.n = len(gt)
        self._smoothed = {}

    def score(self, cfg):
        key = (cfg["activity_smooth"], cfg["ball_speed_smooth"],
               cfg["control_smooth"], cfg["shape_smooth"])
        if key not in self._smoothed:
            self._smoothed[key] = _smooth(self.raw, cfg)
        cost = emissions.costs(self._smoothed[key], cfg)
        blocks = ((0, self.n),)
        trans = viterbi.switch_costs(cfg["switch_turnover"], cfg["switch_to_dead"],
                                     cfg["switch_from_dead"])
        codes = viterbi.decode(cost, trans, blocks=blocks)
        codes = viterbi.enforce_min_duration(codes, cfg["min_duration_frames"], blocks=blocks)
        result = ev.score(viterbi.to_labels(codes), gt_codes=self.gt_codes)
        return result["spell_f1_mean"], result["macro_f1"]


def mean_score(matches, cfg):
    s = [m.score(cfg) for m in matches]
    return float(np.mean([x[0] for x in s])), float(np.mean([x[1] for x in s]))


def class_centres(matches):
    """Cruce de medianas por clase sobre los partidos de ajuste, pooled. Mismo estimador que
    usaba config.py, pero sobre otros partidos degradados como el ATD."""
    bs_alive, bs_dead, c_h, c_a, s_h, s_a = [], [], [], [], [], []
    for m in matches:
        gt = ev.pm.label_codes  # noqa: F841  (solo para dejar claro de donde vienen las clases)
        lab = np.load(os.path.join(cache_dir(), "recal_{}_s0.npz".format(m.id)))["gt"]
        raw = m.raw
        alive = lab != "D"
        bs = raw["ball_speed"]
        ok = np.isfinite(bs)
        bs_alive.append(bs[ok & alive]); bs_dead.append(bs[ok & ~alive])
        diff = np.where(raw["ball_seen"], raw["control_home"] - raw["control_away"], np.nan)
        ok = np.isfinite(diff)
        c_h.append(diff[ok & (lab == "H")]); c_a.append(diff[ok & (lab == "A")])
        sr = raw["spread_ratio_x"]
        ok = np.isfinite(sr)
        s_h.append(sr[ok & (lab == "H")]); s_a.append(sr[ok & (lab == "A")])
    med = lambda parts: float(np.median(np.concatenate(parts)))
    detail = {
        "ball_speed_alive": med(bs_alive), "ball_speed_dead": med(bs_dead),
        "control_H": med(c_h), "control_A": med(c_a),
        "shape_H": med(s_h), "shape_A": med(s_a),
    }
    centres = {
        "ball_speed_center_ms": (detail["ball_speed_alive"] + detail["ball_speed_dead"]) / 2,
        "control_center": (detail["control_H"] + detail["control_A"]) / 2,
        "shape_center": (detail["shape_H"] + detail["shape_A"]) / 2,
    }
    return centres, detail


def coordinate_descent(matches, start, passes=2):
    best = dict(start)
    best_s, best_m = mean_score(matches, best)
    log("inicio: spellF1 %.4f  macroF1 %.4f" % (best_s, best_m))
    for p in range(1, passes + 1):
        log("--- pasada %d ---" % p)
        improved_any = False
        for name, values in GRID:
            t0 = time.time()
            results = []
            for v in values:
                if v == best[name]:
                    results.append((best_s, best_m, v)); continue
                cfg = merged(**{**best, name: v})
                s, mm = mean_score(matches, cfg)
                results.append((s, mm, v))
            s, mm, v = max(results, key=lambda r: (round(r[0], 4), round(r[1], 4)))
            tag = ""
            if v != best[name] and s > best_s + 1e-9:
                tag = "  <- cambia de %s a %s" % (best[name], v)
                best[name] = v; best_s, best_m = s, mm; improved_any = True
            log("%-20s %s  -> mejor %.4f%s  (%.0fs)" % (
                name, " ".join("%s:%.4f" % (r[2], r[0]) for r in results), best_s, tag,
                time.time() - t0))
        if not improved_any:
            log("sin mejoras en la pasada %d, paro" % p)
            break
    return best, best_s, best_m


def report(name, cfg, tune, hold):
    ts, tm = mean_score(tune, cfg)
    hs, hm = mean_score(hold, cfg)
    als, alm = mean_score(tune + hold, cfg)
    log("%-32s ajuste(4) %.4f/%.4f   validacion(3) %.4f/%.4f   todos(7) %.4f/%.4f"
        % (name, ts, tm, hs, hm, als, alm))
    return {"tune": [ts, tm], "hold": [hs, hm], "all": [als, alm]}


def main():
    t0 = time.time()
    spec = harness.real_spec()
    log("spec del ATD (sin etiquetas): %s" % {k: round(v, 2) for k, v in spec.summary().items()})
    tune = [Match(m, *prepare(m, spec)) for m in TUNE]
    hold = [Match(m, *prepare(m, spec)) for m in HOLD]
    log("partidos preparados en %.0fs" % (time.time() - t0))

    centres, detail = class_centres(tune)
    log("medianas por clase (ajuste, degradado): %s" % {k: round(v, 3) for k, v in detail.items()})
    log("centros re-derivados: %s" % {k: round(v, 3) for k, v in centres.items()})
    log("centros anteriores  : %s" % {k: DEFAULT[k] for k in centres})

    start = merged(**centres)
    best, best_s, best_m = coordinate_descent(tune, start, passes=2)

    log("=" * 78)
    out = {"config": best, "centres_detail": detail, "tune": list(TUNE), "hold": list(HOLD)}
    original = merged(activity_center_ms=1.75)      # como estaba antes del 24-10
    out["scores"] = {
        "original_pre_sesion": report("original (centro 1.75)", original, tune, hold),
        "actual_config_py":    report("config.py actual", DEFAULT, tune, hold),
        "solo_centros":        report("solo centros re-derivados", start, tune, hold),
        "recalibrado":         report("RECALIBRADO", merged(**best), tune, hold),
    }
    log("=" * 78)
    changed = {k: (DEFAULT[k], best[k]) for k in best if best[k] != DEFAULT[k]}
    log("valores que cambian respecto a config.py:")
    for k, (a, b) in changed.items():
        log("   %-22s %s -> %s" % (k, a, b))
    path = os.path.join(REPO_ROOT, "out", "recalibrated.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    log("guardado %s   (%.0f min)" % (path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
