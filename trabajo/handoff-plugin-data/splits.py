"""Validacion cruzada del PROCEDIMIENTO de recalibracion, no de una config concreta.

Corre el mismo descenso por coordenadas de training/recalibrate.py desde el mismo punto de
partida sobre LOS 35 splits posibles de 4 partidos de ajuste / 3 de validacion, con los 7
partidos del IDSSE. Responde dos cosas que un solo split no puede:

  1. Estabilidad: para cada hiperparametro, que valor elige el descenso en cada split. Un
     valor que cambia con el split es ruido de rejilla, no una propiedad del problema.
  2. Que gana el procedimiento fuera de su propio ajuste: media del spellF1 en los 3
     partidos de validacion de CADA split, comparada con la del punto de partida sobre los
     mismos 3. Es el delta honesto, sin reutilizar ningun partido para elegir y medir.

Dos brazos, con el MISMO codigo de descenso para que sean comparables:

  rejilla     la de training/recalibrate.py  -> datos/splits_cruzados.json          (~35 min)
  ampliada    la de abajo, con los bordes
              prolongados                   -> datos/splits_cruzados_ampliada.json (~45 min)

Resultados en datos/splits_cruzados*.json; el resumen legible lo imprime resumen_splits.py y
las comprobaciones de consenso, consenso.py.

Solo lee out/cache (partidos externos degradados). No toca el ATD ni la ventana publica.

Uso:  .venv/bin/python -I -u handoff-plugin-data/splits.py [rejilla|ampliada]
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

# Punto de partida comun a los 35 descensos: el config.py anterior al 28-10 reconstruido
# desde su procedencia documentada, con los tres centros del IDSSE encima. Es el analogo de
# `start = merged(**centres)` del script original. Identico en todos los splits, que es lo
# que hace comparables los puntos de llegada. (El config.py real de esa fecha no se
# conserva: el repo no tiene commits. Reconstruido queda a -0,3 pp en ajuste y -0,02 pp en
# validacion de los valores publicados en recalibracion.json.)
ANTES = dict(DEFAULT)
ANTES.update(activity_smooth=50, ball_speed_smooth=50, control_smooth=50, shape_smooth=50,
             ball_speed_center_ms=4.10, control_center=0.096, shape_center=0.975,
             switch_turnover=28.0)
INICIO = merged(**{**ANTES, **CENTROS})
BARRIDOS = [n for n, _ in GRID]

# Rejilla ampliada (30-10). En la primera tanda de 35 splits, siete parametros se quedaban
# clavados en un BORDE de su propia lista de valores, asi que el descenso no podia seguir en
# esa direccion y el optimo quedaba fuera de la rejilla, sin medir. Se prolonga cada uno con
# su mismo paso geometrico (~x1,4) hasta que el borde que se elegia queda por dentro. Lo
# demas no cambia: mismo INICIO, mismos 14 parametros barridos, mismas 2 pasadas y la misma
# funcion descenso(), para que la unica diferencia entre los dos brazos sea la rejilla.
AMPLIACION = {
    # parametro              valores                                   borde que se clavaba
    "shape_smooth":        [12, 25, 50, 100, 200, 300, 450],         # 200 en 22/35
    "min_duration_frames": [0, 6, 12, 25, 50, 70, 100],              #  50 en 17/35 (moda)
    "switch_turnover":     [8, 14, 20, 28, 40, 56, 78, 110],         #  40 en 30/35
    "switch_to_dead":      [10, 18, 26, 34, 45, 63, 88, 123],        #  45 en 28/35
    "switch_from_dead":    [4, 8, 14, 22, 31, 43, 60],               #  22 en 25/35
    "ball_speed_weight":   [0.0, 0.3, 0.6, 1.2, 1.7, 2.4, 3.4],      # 1,2 en 28/35
    "ball_speed_smooth":   [1, 3, 6, 12, 25, 50],                    #   6 en 19/35, borde BAJO
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
    """Identico en logica a recalibrate.coordinate_descent, pero silencioso."""
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
    log("brazo '%s': %d valores en la rejilla, %d parametros barridos"
        % (brazo, sum(len(v) for _, v in grid), len(grid)))
    for name, vals in grid:
        if dict(GRID)[name] != vals:
            log("   ampliado  %-20s %s" % (name, vals))
    log("%d splits de 4 ajuste / 3 validacion" % len(splits))
    log("punto de partida comun: spellF1 sobre los 7 = %.4f"
        % mean_score([cargar(m, 0) for m in PARTIDOS], INICIO)[0])
    log("")

    filas = []
    for i, tune_ids in enumerate(splits, 1):
        hold_ids = tuple(m for m in PARTIDOS if m not in tune_ids)
        tune = [cargar(m, 0) for m in tune_ids]
        hold = [cargar(m, 0) for m in hold_ids]
        ti = time.time()

        elegido, s_inicio_tune, s_fin_tune = descenso(tune, INICIO, grid)

        # El mismo hold, con el punto de partida y con lo que eligio el descenso.
        h_inicio = mean_score(hold, INICIO)[0]
        h_fin = mean_score(hold, elegido)[0]

        # Y en las otras dos semillas de degradacion, los 7 partidos: otro eje de ruido que
        # el descenso no ha visto.
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
        log("%2d/%d  hold=%-24s ajuste %.4f->%.4f (%+.4f)  validacion %.4f->%.4f (%+.4f)"
            "  [%d cambios, %.0fs]%s"
            % (i, len(splits), ",".join(h[3:] for h in hold_ids),
               s_inicio_tune, s_fin_tune, s_fin_tune - s_inicio_tune,
               h_inicio, h_fin, h_fin - h_inicio, len(cambios), time.time() - ti,
               "  <- split original" if fila["es_split_original"] else ""))

        json.dump({"_en_curso": True, "brazo": brazo, "rejilla": dict(grid),
                   "inicio": INICIO, "barridos": BARRIDOS,
                   "splits": filas}, open(salida, "w"), indent=2, default=float)

    # --------------------------------------------------------------------- resumen
    d_tune = np.array([f["tune_elegido"] - f["tune_inicio"] for f in filas])
    d_hold = np.array([f["hold_elegido"] - f["hold_inicio"] for f in filas])
    log("")
    log("=" * 86)
    log("GANANCIA DEL PROCEDIMIENTO, sobre %d splits" % len(filas))
    log("  en su propio ajuste    : %+.4f  (de %+.4f a %+.4f)"
        % (d_tune.mean(), d_tune.min(), d_tune.max()))
    log("  en su propia validacion: %+.4f  (de %+.4f a %+.4f)"
        % (d_hold.mean(), d_hold.min(), d_hold.max()))
    log("  splits donde el ajuste mejora    : %d/%d" % ((d_tune > 1e-9).sum(), len(filas)))
    log("  splits donde la validacion mejora: %d/%d" % ((d_hold > 1e-9).sum(), len(filas)))
    if d_tune.mean():
        log("  transferencia (media validacion / media ajuste): %.0f %%"
            % (100 * d_hold.mean() / d_tune.mean()))
    for seed in (1, 2):
        d = np.array([f["s%d_elegido" % seed] - f["s%d_inicio" % seed] for f in filas])
        log("  semilla %d, los 7 partidos: %+.4f  (de %+.4f a %+.4f)  mejora en %d/%d"
            % (seed, d.mean(), d.min(), d.max(), (d > 1e-9).sum(), len(filas)))

    log("")
    log("ESTABILIDAD POR HIPERPARAMETRO")
    log("%-22s %-8s %-9s %s" % ("parametro", "inicio", "config.py", "reparto de lo elegido"))
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
    log("cuantos splits eligen el valor que hay hoy en config.py:")
    for name in BARRIDOS:
        e = estabilidad[name]
        log("   %-22s %2d/%d" % (name, e["coincide_config_py"], len(filas)))

    log("")
    log("PARAMETROS QUE SE CLAVAN EN UN BORDE DE LA REJILLA")
    hay_borde = False
    for name, valores in grid:
        e = estabilidad[name]
        if e["moda_en_borde"]:
            hay_borde = True
            log("   %-22s moda %-6s = borde %s  (%d/%d splits en un borde)"
                % (name, e["moda"], "bajo" if e["moda"] == valores[0] else "alto",
                   e["n_en_borde"], len(filas)))
    if not hay_borde:
        log("   ninguno: todas las modas caen por dentro de la rejilla")

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
    log("guardado %s  (%.0f min)" % (salida, (time.time() - t0) / 60))


if __name__ == "__main__":
    brazo = sys.argv[1] if len(sys.argv) > 1 else "rejilla"
    if brazo not in REJILLAS:
        raise SystemExit("brazo desconocido %r; usa: %s" % (brazo, " | ".join(REJILLAS)))
    # Segundo argumento opcional: correr solo los N primeros splits (para comprobaciones).
    main(brazo, int(sys.argv[2]) if len(sys.argv) > 2 else None)
