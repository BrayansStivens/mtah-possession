"""Cross-validation of the recalibration PROCEDURE, not of one particular config.

Runs the same coordinate descent of training/recalibrate.py from the same starting point
over ALL 35 possible splits of 4 fit matches / 3 validation matches, with the 7 IDSSE
matches. It answers two things a single split cannot:

  1. Stability: for each hyperparameter, which value the descent picks in each split. A
     value that changes with the split is grid noise, not a property of the problem.
  2. What the procedure gains outside its own fit: mean spellF1 on the 3 validation
     matches of EACH split, compared with that of the starting point on the same 3. It is
     the honest delta, with no match reused both to choose and to measure.

Two arms, with the SAME descent code so they are comparable:

  rejilla     the grid of training/recalibrate.py  -> datos/splits_cruzados.json          (~35 min)
  ampliada    the grid below, with the edges
              extended                             -> datos/splits_cruzados_ampliada.json (~45 min)

Results in datos/splits_cruzados*.json; the readable summary is printed by resumen_splits.py
and the consensus checks by consenso.py.

It only reads out/cache (degraded external matches). It touches neither the ATD nor the
public window.

Usage:  .venv/bin/python -I -u handoff-plugin-data/splits.py [rejilla|ampliada]
"""

import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT, merged                      # noqa: E402
from mtah.training.recalibrate import GRID, RAW_KEYS, Match, mean_score  # noqa: E402
from mtah.paths import cache_dir                             # noqa: E402

SALIDA = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/splits_cruzados.json"
REFERENCIA = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/recalibracion.json"
PARTIDOS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SPLIT_ORIGINAL = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")

CENTROS = {k: json.load(open(REFERENCIA))["config"][k]
           for k in ("ball_speed_center_ms", "control_center", "shape_center")}

# Starting point common to the 35 descents: the config.py from before Oct 28 reconstructed
# from its documented provenance, with the three IDSSE centers on top. It is the analogue of
# `start = merged(**centers)` in the original script. Identical in all splits, which is what
# makes the arrival points comparable. (The real config.py of that date is not kept: the
# repo has no commits. Reconstructed, it comes out -0.3 pp on the fit set and -0.02 pp on the
# validation set relative to the values published in recalibracion.json.)
ANTES = dict(DEFAULT)
ANTES.update(activity_smooth=50, ball_speed_smooth=50, control_smooth=50, shape_smooth=50,
             ball_speed_center_ms=4.10, control_center=0.096, shape_center=0.975,
             switch_turnover=28.0)
INICIO = merged(**{**ANTES, **CENTROS})
BARRIDOS = [n for n, _ in GRID]

# Extended grid (Oct 30). In the first batch of 35 splits, seven parameters got stuck at an
# EDGE of their own list of values, so the descent could not continue in that direction and
# the optimum fell outside the grid, unmeasured. Each one is extended with its same geometric
# step (~x1.4) until the edge that was being chosen ends up inside. Everything else is
# unchanged: same INICIO, same 14 swept parameters, same 2 passes and the same descenso()
# function, so that the only difference between the two arms is the grid.
AMPLIACION = {
    # parameter              values                                    edge it got stuck at
    "shape_smooth":        [12, 25, 50, 100, 200, 300, 450],         # 200 in 22/35
    "min_duration_frames": [0, 6, 12, 25, 50, 70, 100],              #  50 in 17/35 (mode)
    "switch_turnover":     [8, 14, 20, 28, 40, 56, 78, 110],         #  40 in 30/35
    "switch_to_dead":      [10, 18, 26, 34, 45, 63, 88, 123],        #  45 in 28/35
    "switch_from_dead":    [4, 8, 14, 22, 31, 43, 60],               #  22 in 25/35
    "ball_speed_weight":   [0.0, 0.3, 0.6, 1.2, 1.7, 2.4, 3.4],      # 1.2 in 28/35
    "ball_speed_smooth":   [1, 3, 6, 12, 25, 50],                    #   6 in 19/35, LOW edge
}
GRID_AMPLIADA = [(n, AMPLIACION.get(n, vals)) for n, vals in GRID]

REJILLAS = {
    "rejilla": (GRID, SALIDA),
    "ampliada": (GRID_AMPLIADA, SALIDA.replace(".json", "_ampliada.json")),
}


def log(msg):
    print(msg, flush=True)


def cargar(match_id, seed):
    blob = np.load(os.path.join(cache_dir(), "recal_%s_s%d.npz" % (match_id, seed)),
                   allow_pickle=False)
    return Match(match_id, {k: blob[k] for k in RAW_KEYS}, blob["gt"])


def descenso(matches, start, grid, passes=2):
    """Identical in logic to recalibrate.coordinate_descent, but silent."""
    best = dict(start)
    best_s, best_m = mean_score(matches, best)
    inicio_s = best_s
    for _ in range(passes):
        improved = False
        for name, values in grid:
            resultados = []
            for v in values:
                if v == best[name]:
                    resultados.append((best_s, best_m, v))
                    continue
                s, mm = mean_score(matches, merged(**{**best, name: v}))
                resultados.append((s, mm, v))
            s, mm, v = max(resultados, key=lambda r: (round(r[0], 4), round(r[1], 4)))
            if v != best[name] and s > best_s + 1e-9:
                best[name] = v
                best_s, best_m = s, mm
                improved = True
        if not improved:
            break
    return best, inicio_s, best_s


def main(brazo="rejilla", limite=None):
    grid, salida = REJILLAS[brazo]
    t0 = time.time()
    splits = list(itertools.combinations(PARTIDOS, 4))
    if limite:
        splits = splits[:limite]
    log("arm '%s': %d values in the grid, %d swept parameters"
        % (brazo, sum(len(v) for _, v in grid), len(grid)))
    for name, vals in grid:
        if dict(GRID)[name] != vals:
            log("   extended  %-20s %s" % (name, vals))
    log("%d splits of 4 fit / 3 validation" % len(splits))
    log("common starting point: spellF1 over the 7 = %.4f"
        % mean_score([cargar(m, 0) for m in PARTIDOS], INICIO)[0])
    log("")

    filas = []
    for i, tune_ids in enumerate(splits, 1):
        hold_ids = tuple(m for m in PARTIDOS if m not in tune_ids)
        tune = [cargar(m, 0) for m in tune_ids]
        hold = [cargar(m, 0) for m in hold_ids]
        ti = time.time()

        elegido, s_inicio_tune, s_fin_tune = descenso(tune, INICIO, grid)

        # The same hold, with the starting point and with what the descent chose.
        h_inicio = mean_score(hold, INICIO)[0]
        h_fin = mean_score(hold, elegido)[0]

        # And on the other two degradation seeds, all 7 matches: another noise axis that
        # the descent has not seen.
        otras = {}
        for seed in (1, 2):
            siete = [cargar(m, seed) for m in PARTIDOS]
            otras["s%d_inicio" % seed] = mean_score(siete, INICIO)[0]
            otras["s%d_elegido" % seed] = mean_score(siete, elegido)[0]
            del siete

        cambios = {k: elegido[k] for k in BARRIDOS if elegido[k] != INICIO[k]}
        fila = {
            "i": i, "tune": list(tune_ids), "hold": list(hold_ids),
            "es_split_original": tuple(sorted(tune_ids)) == tuple(sorted(SPLIT_ORIGINAL)),
            "tune_inicio": s_inicio_tune, "tune_elegido": s_fin_tune,
            "hold_inicio": h_inicio, "hold_elegido": h_fin,
            "elegido": {k: elegido[k] for k in BARRIDOS},
            "cambios": cambios, **otras,
        }
        filas.append(fila)
        log("%2d/%d  hold=%-24s fit %.4f->%.4f (%+.4f)  validation %.4f->%.4f (%+.4f)"
            "  [%d changes, %.0fs]%s"
            % (i, len(splits), ",".join(h[3:] for h in hold_ids),
               s_inicio_tune, s_fin_tune, s_fin_tune - s_inicio_tune,
               h_inicio, h_fin, h_fin - h_inicio, len(cambios), time.time() - ti,
               "  <- original split" if fila["es_split_original"] else ""))

        json.dump({"_en_curso": True, "brazo": brazo, "rejilla": dict(grid),
                   "inicio": INICIO, "barridos": BARRIDOS,
                   "splits": filas}, open(salida, "w"), indent=2, default=float)

    # --------------------------------------------------------------------- summary
    d_tune = np.array([f["tune_elegido"] - f["tune_inicio"] for f in filas])
    d_hold = np.array([f["hold_elegido"] - f["hold_inicio"] for f in filas])
    log("")
    log("=" * 86)
    log("PROCEDURE GAIN, over %d splits" % len(filas))
    log("  on its own fit set     : %+.4f  (from %+.4f to %+.4f)"
        % (d_tune.mean(), d_tune.min(), d_tune.max()))
    log("  on its own validation  : %+.4f  (from %+.4f to %+.4f)"
        % (d_hold.mean(), d_hold.min(), d_hold.max()))
    log("  splits where fit improves        : %d/%d" % ((d_tune > 1e-9).sum(), len(filas)))
    log("  splits where validation improves : %d/%d" % ((d_hold > 1e-9).sum(), len(filas)))
    if d_tune.mean():
        log("  transfer (mean validation / mean fit): %.0f%%"
            % (100 * d_hold.mean() / d_tune.mean()))
    for seed in (1, 2):
        d = np.array([f["s%d_elegido" % seed] - f["s%d_inicio" % seed] for f in filas])
        log("  seed %d, all 7 matches: %+.4f  (from %+.4f to %+.4f)  improves in %d/%d"
            % (seed, d.mean(), d.min(), d.max(), (d > 1e-9).sum(), len(filas)))

    log("")
    log("STABILITY PER HYPERPARAMETER")
    log("%-22s %-8s %-9s %s" % ("parametro", "inicio", "config.py", "breakdown of chosen values"))
    estabilidad = {}
    for name, valores in grid:
        cuenta = {}
        for f in filas:
            cuenta[f["elegido"][name]] = cuenta.get(f["elegido"][name], 0) + 1
        orden = sorted(cuenta.items(), key=lambda kv: -kv[1])
        moda, n_moda = orden[0]
        bordes = (valores[0], valores[-1])
        estabilidad[name] = {"reparto": {str(k): v for k, v in orden}, "moda": moda,
                             "n_moda": n_moda, "n_distintos": len(orden),
                             "inicio": INICIO[name], "config_py": DEFAULT[name],
                             "coincide_config_py": cuenta.get(DEFAULT[name], 0),
                             "valores": list(valores),
                             "moda_en_borde": moda in bordes,
                             "n_en_borde": sum(v for k, v in orden if k in bordes)}
        log("%-22s %-8s %-9s %s" % (name, INICIO[name], DEFAULT[name],
                                    "  ".join("%s:%d" % (k, v) for k, v in orden)))

    log("")
    log("how many splits choose the value that config.py has today:")
    for name in BARRIDOS:
        e = estabilidad[name]
        log("   %-22s %2d/%d" % (name, e["coincide_config_py"], len(filas)))

    log("")
    log("PARAMETERS THAT STICK TO AN EDGE OF THE GRID")
    hay_borde = False
    for name, valores in grid:
        e = estabilidad[name]
        if e["moda_en_borde"]:
            hay_borde = True
            log("   %-22s mode %-6s = edge %s  (%d/%d splits at an edge)"
                % (name, e["moda"], "bajo" if e["moda"] == valores[0] else "alto",
                   e["n_en_borde"], len(filas)))
    if not hay_borde:
        log("   none: all the modes fall inside the grid")

    json.dump({"_en_curso": False, "brazo": brazo, "rejilla": dict(grid),
               "inicio": INICIO, "barridos": BARRIDOS,
               "splits": filas, "estabilidad": estabilidad,
               "resumen": {
                   "n_splits": len(filas),
                   "d_tune_media": float(d_tune.mean()), "d_tune_min": float(d_tune.min()),
                   "d_tune_max": float(d_tune.max()),
                   "d_hold_media": float(d_hold.mean()), "d_hold_min": float(d_hold.min()),
                   "d_hold_max": float(d_hold.max()),
                   "splits_hold_mejora": int((d_hold > 1e-9).sum()),
                   "splits_tune_mejora": int((d_tune > 1e-9).sum())}},
              open(salida, "w"), indent=2, default=float)
    log("")
    log("saved %s  (%.0f min)" % (salida, (time.time() - t0) / 60))


if __name__ == "__main__":
    brazo = sys.argv[1] if len(sys.argv) > 1 else "rejilla"
    if brazo not in REJILLAS:
        raise SystemExit("unknown arm %r; use: %s" % (brazo, " | ".join(REJILLAS)))
    # Optional second argument: run only the first N splits (for checks).
    main(brazo, int(sys.argv[2]) if len(sys.argv) > 2 else None)
