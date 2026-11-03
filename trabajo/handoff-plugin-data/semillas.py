"""Robustez de la eleccion de config: 3 semillas de degradacion x 7 partidos, mas
ablacion de cada cambio que eligio el descenso por coordenadas.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT, merged
from mtah.training.recalibrate import Match, RAW_KEYS, TUNE, HOLD, mean_score
from mtah.paths import cache_dir

REF = json.load(open("/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/recalibracion.json"))
CENTROS = {k: REF["config"][k] for k in ("ball_speed_center_ms", "control_center", "shape_center")}

ANTES = dict(DEFAULT)
ANTES.update(activity_smooth=50, ball_speed_smooth=50, control_smooth=50, shape_smooth=50,
             ball_speed_center_ms=4.10, control_center=0.096, shape_center=0.975,
             switch_turnover=28.0)
SOLO_CENTROS = merged(**{**ANTES, **CENTROS})

def load(mid, seed):
    p = os.path.join(cache_dir(), "recal_%s_s%d.npz" % (mid, seed))
    if not os.path.exists(p):
        return None
    blob = np.load(p, allow_pickle=False)
    return Match(mid, {k: blob[k] for k in RAW_KEYS}, blob["gt"])

CFGS = [("solo_centros", SOLO_CENTROS), ("recalibrado", merged(**DEFAULT))]

print("=== 1. las tres semillas de degradacion ===")
print("%-14s %-7s %8s %8s %8s" % ("config", "semilla", "ajuste", "valid.", "todos"))
tabla = {}
for seed in (0, 1, 2):
    tune = [m for m in (load(x, seed) for x in TUNE) if m]
    hold = [m for m in (load(x, seed) for x in HOLD) if m]
    if len(tune) < 4 or len(hold) < 3:
        print("semilla %d incompleta (%d+%d), se salta" % (seed, len(tune), len(hold)))
        continue
    for nombre, cfg in CFGS:
        ts = mean_score(tune, cfg)[0]; hs = mean_score(hold, cfg)[0]
        als = mean_score(tune + hold, cfg)[0]
        tabla.setdefault(nombre, []).append((seed, ts, hs, als))
        print("%-14s s%-6d %8.4f %8.4f %8.4f" % (nombre, seed, ts, hs, als))

print()
print("delta recalibrado - solo_centros, por semilla:")
if len(tabla) == 2:
    for i, (s, ts, hs, als) in enumerate(tabla["recalibrado"]):
        _, ts0, hs0, als0 = tabla["solo_centros"][i]
        print("   semilla %d: ajuste %+0.4f  validacion %+0.4f  todos %+0.4f"
              % (s, ts - ts0, hs - hs0, als - als0))
    d_hold = [r[2] - s[2] for r, s in zip(tabla["recalibrado"], tabla["solo_centros"])]
    print("   media validacion %+0.4f   rango [%+0.4f, %+0.4f]"
          % (np.mean(d_hold), min(d_hold), max(d_hold)))

print()
print("=== 2. ablacion: revertir un solo cambio del descenso, semilla 0 ===")
tune = [load(x, 0) for x in TUNE]; hold = [load(x, 0) for x in HOLD]
base_t, base_h = mean_score(tune, DEFAULT)[0], mean_score(hold, DEFAULT)[0]
print("%-24s %8s %8s" % ("config.py completo", "%.4f" % base_t, "%.4f" % base_h))
revert = {"activity_smooth": 50, "ball_speed_smooth": 50, "shape_smooth": 50,
          "switch_turnover": 28.0,
          "ball_speed_center_ms": 4.10, "control_center": 0.096, "shape_center": 0.975}
for k, v in revert.items():
    cfg = merged(**{k: v})
    t = mean_score(tune, cfg)[0]; h = mean_score(hold, cfg)[0]
    print("  revertir %-22s %8.4f %8.4f   (ajuste %+0.4f  validacion %+0.4f)"
          % ("%s=%s" % (k, v), t, h, t - base_t, h - base_h))
