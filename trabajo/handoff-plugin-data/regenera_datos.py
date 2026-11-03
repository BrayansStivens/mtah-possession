"""Regenera los cinco ficheros de datos/ que alimentan el dashboard, con el config.py de hoy.

Los ficheros del 28-10 se hicieron a mano en otra sesion y "recalibrado" alli era el
config.py de ese dia (activity_smooth = 12). Desde el 31-10 config.py es otro, asi que
el dashboard citaba cifras que el fichero ya no produce. Este script:

  - vuelve a puntuar las tres configuraciones comparables sobre los 7 partidos del IDSSE
    (semilla 0 de degradacion, la misma cache que uso el 28-10):
        actual        = config.py de antes del 28-10, RECONSTRUIDO como en reeval.py (ese
                        fichero no se conserva; sale 0,3 a 0,7 pp por debajo de lo medido el 28-10)
        solo_centros  = actual + los tres centros de clase re-derivados el 28-10
        recalibrado   = config.py tal cual esta HOY
    "original" (antes de la sesion del 28-10) no se puede regenerar porque no quedo su
    config; se copia tal cual del fichero viejo.
  - rehace las cuatro tablas de diagnostico para actual y recalibrado;
  - rehace la ventana publica y el partido a 1 Hz desde out/submission.csv (el de hoy).

Antes de escribir nada comprueba que, con la config "actual", reproduce el fichero viejo
exactamente (misma cache, misma config -> mismos numeros). Si eso falla, no escribe.

Uso:  .venv/bin/python handoff-plugin-data/regenera_datos.py
"""

import csv
import json
import os
import sys
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "datos")
REPO = os.path.join(AQUI, "..", "mtah-possession")
sys.path.insert(0, os.path.join(REPO, "src"))

from mtah import evaluate as ev                                   # noqa: E402
from mtah.config import DEFAULT, merged                           # noqa: E402
from mtah.decode import viterbi                                   # noqa: E402
from mtah.models import emissions                                 # noqa: E402
from mtah.paths import cache_dir                                  # noqa: E402
from mtah.training.recalibrate import (Match, RAW_KEYS, TUNE, HOLD, _smooth)  # noqa: E402

pm = ev.pm
FPS = 25.0
PARTIDOS = TUNE + HOLD
GRUPO = {**{m: "ajuste" for m in TUNE}, **{m: "validacion" for m in HOLD}}

# --- las tres configuraciones comparables -------------------------------------------------
REF = json.load(open(os.path.join(DATOS, "recalibracion.json")))
CENTROS = {k: REF["config"][k] for k in ("ball_speed_center_ms", "control_center", "shape_center")}
ANTES = dict(DEFAULT)
ANTES.update(activity_smooth=50, ball_speed_smooth=50, control_smooth=50, shape_smooth=50,
             ball_speed_center_ms=4.10, control_center=0.096, shape_center=0.975,
             switch_turnover=28.0)
CONFIGS = {
    "actual": merged(**ANTES),
    "solo_centros": merged(**{**ANTES, **CENTROS}),
    "recalibrado": DEFAULT,
}


def cargar(mid, seed=0):
    blob = np.load(os.path.join(cache_dir(), "recal_%s_s%d.npz" % (mid, seed)), allow_pickle=False)
    return Match(mid, {k: blob[k] for k in RAW_KEYS}, blob["gt"])


def decodificar(m, cfg):
    """Lo mismo que Match.score pero devolviendo los codigos predichos."""
    key = (cfg["activity_smooth"], cfg["ball_speed_smooth"], cfg["control_smooth"], cfg["shape_smooth"])
    if key not in m._smoothed:
        m._smoothed[key] = _smooth(m.raw, cfg)
    cost = emissions.costs(m._smoothed[key], cfg)
    blocks = ((0, m.n),)
    trans = viterbi.switch_costs(cfg["switch_turnover"], cfg["switch_to_dead"], cfg["switch_from_dead"])
    codes = viterbi.decode(cost, trans, blocks=blocks)
    codes = viterbi.enforce_min_duration(codes, cfg["min_duration_frames"], blocks=blocks)
    return pm.label_codes(np.asarray(viterbi.to_labels(codes), dtype="<U1"))


# --- tablas de diagnostico ----------------------------------------------------------------
def bucket(x, bordes, etiquetas):
    """Etiqueta del tramo: bordes = limites superiores inclusivos, el ultimo es abierto."""
    for b, e in zip(bordes, etiquetas):
        if x <= b:
            return e
    return etiquetas[-1]


def racha_sin_balon(seen):
    """Longitud total de la racha sin balon a la que pertenece cada frame (0 si se ve)."""
    out = np.zeros(len(seen), dtype=np.int64)
    i = 0
    while i < len(seen):
        if seen[i]:
            i += 1
            continue
        j = i
        while j < len(seen) and not seen[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def distancia_a_transicion(gt, runs):
    segs = pm.segments(gt, runs)
    bordes = np.array(sorted({s[0] for s in segs[1:]}), dtype=np.int64)
    idx = np.arange(len(gt))
    if not len(bordes):
        return np.full(len(gt), 10**9)
    pos = np.searchsorted(bordes, idx)
    izq = np.where(pos > 0, idx - bordes[np.clip(pos - 1, 0, len(bordes) - 1)], 10**9)
    der = np.where(pos < len(bordes), bordes[np.clip(pos, 0, len(bordes) - 1)] - idx, 10**9)
    return np.minimum(izq, der)


def diagnostico(partidos, cfg, nombre):
    """Las cuatro tablas en formato largo, agregadas sobre los 7 partidos."""
    err_racha, n_racha = {}, {}
    err_dist, n_dist = {}, {}
    conf = np.zeros((3, 3), dtype=np.int64)
    rec, n_rec = {}, {}
    for m in partidos:
        gt, pred = m.gt_codes, decodificar(m, cfg)
        runs = pm.valid_runs(gt)
        graded = gt != pm.X
        err = (pred != gt) & graded
        r = racha_sin_balon(np.asarray(m.raw["ball_seen"], dtype=bool))
        cat = np.where(r == 0, "balon visible",
                       np.where(r <= 25, "1-25 fr", np.where(r <= 125, "26-125 fr",
                                np.where(r <= 500, "126-500 fr", ">500 fr"))))
        for c in np.unique(cat):
            sel = (cat == c) & graded
            n_racha[c] = n_racha.get(c, 0) + int(sel.sum())
            err_racha[c] = err_racha.get(c, 0) + int(err[sel].sum())
        d = distancia_a_transicion(gt, runs)
        cat = np.where(d <= 12, "0-12 fr", np.where(d <= 25, "13-25 fr", np.where(
            d <= 75, "26-75 fr", np.where(d <= 250, "76-250 fr", ">250 fr"))))
        for c in np.unique(cat):
            sel = (cat == c) & graded
            n_dist[c] = n_dist.get(c, 0) + int(sel.sum())
            err_dist[c] = err_dist.get(c, 0) + int(err[sel].sum())
        for i in range(3):
            for j in range(3):
                conf[i, j] += int(((gt == i) & (pred == j)).sum())
        gts = pm.possession_spells(gt, runs)
        flags = pm.matched_gt_flags(pm.spell_overlaps(gts, pm.possession_spells(pred, runs)),
                                    len(gts), 0.5)
        for (a, b, _), ok in zip(gts, flags):
            s = (b - a) / FPS
            c = "<2 s" if s < 2 else "2-5 s" if s < 5 else "5-10 s" if s < 10 else "10-20 s" if s < 20 else ">20 s"
            n_rec[c] = n_rec.get(c, 0) + 1
            rec[c] = rec.get(c, 0) + int(ok)
    filas = []
    for c in ("balon visible", "1-25 fr", "26-125 fr", "126-500 fr", ">500 fr"):
        filas.append((nombre, "error_por_racha_sin_balon", c, round(100.0 * err_racha[c] / n_racha[c], 2), n_racha[c]))
    for c in ("0-12 fr", "13-25 fr", "26-75 fr", "76-250 fr", ">250 fr"):
        filas.append((nombre, "error_por_distancia_a_transicion", c, round(100.0 * err_dist[c] / n_dist[c], 2), n_dist[c]))
    for i, ri in enumerate("HAD"):
        for j, pj in enumerate("HAD"):
            filas.append((nombre, "confusion_pct_fila", "real %s -> pred %s" % (ri, pj),
                          round(100.0 * conf[i, j] / conf[i].sum(), 2), int(conf[i, j])))
    for c in ("<2 s", "2-5 s", "5-10 s", "10-20 s", ">20 s"):
        filas.append((nombre, "recall_posesiones_por_duracion", c, round(100.0 * rec[c] / n_rec[c], 2), n_rec[c]))
    return filas


def por_partido(partidos, cfg, nombre):
    filas = []
    for m in partidos:
        r = ev.score(_labels(m, cfg), gt_codes=m.gt_codes)
        filas.append((m.id, GRUPO[m.id], nombre, round(r["spell_f1_mean"], 4), round(r["macro_f1"], 4),
                      r["n_gt_spells"], r["n_pred_spells"],
                      round(r["gt_mean_spell_duration_s"], 1), round(r["mean_spell_duration_s"], 1)))
    return filas


def _labels(m, cfg):
    return np.array(list("HADX"))[decodificar(m, cfg)]


# --- ventana publica y partido a 1 Hz -----------------------------------------------------
def ventana_y_1hz(ruta_submission):
    sub = pm.read_label_csv(ruta_submission)
    lab = np.asarray(sub["label"], dtype="<U1")
    gt = ev.public_gt()
    runs_pred = pm.valid_runs(pm.label_codes(lab))          # las dos partes, donde no hay X
    assert len(runs_pred) == 2, runs_pred
    ventana = []
    g0 = int(np.flatnonzero(gt != pm.X)[0])
    gl = np.array(list("HADX"))[gt]
    for k in range(7500):
        f = g0 + k
        ventana.append((f, round(k / FPS, 2), gl[f], lab[f]))
    hz = []
    for parte, (a, b) in enumerate(runs_pred, start=1):
        for k, f in enumerate(range(a, b, int(FPS))):
            hz.append((parte, float(k), lab[f]))
    return ventana, hz


# --- comprobacion contra los ficheros viejos ----------------------------------------------
def leer_csv(nombre):
    with open(os.path.join(DATOS, nombre)) as fh:
        return list(csv.reader(fh))


def mismo(nuevo, viejo, que):
    v = [tuple(str(x) for x in f) for f in nuevo]
    w = [tuple(str(x) for x in f) for f in viejo]
    if v != w:
        for a, b in zip(v, w):
            if a != b:
                print("   DIFIERE en %s:\n      nuevo %s\n      viejo %s" % (que, a, b))
        if len(v) != len(w):
            print("   DIFIERE en %s: %d filas nuevas, %d viejas" % (que, len(v), len(w)))
        return False
    print("   ok  %s reproduce el fichero del 28-10 (%d filas)" % (que, len(v)))
    return True


def escribir_csv(nombre, cabecera, filas):
    with open(os.path.join(DATOS, nombre), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cabecera)
        w.writerows(filas)


def main():
    partidos = [cargar(m) for m in PARTIDOS]
    hoy = date.today().isoformat()

    # 1. comprobar contra los ficheros del 28-10 ------------------------------------------
    # Lo que tiene que salir EXACTO: la ventana publica y el 1 Hz (mismo out/actual.csv) y
    # los tamanos de cada categoria del diagnostico (dependen del ground truth y del balon
    # visto, no de la config). Lo que NO puede salir exacto: las puntuaciones de "actual",
    # porque el config.py anterior al 28-10 no se conserva y ANTES es una reconstruccion
    # (documentado en datos/robustez_semillas.json: -0,3 a -0,7 pp). Se imprime la
    # diferencia para que quede a la vista, y solo lo exacto corta la ejecucion.
    print("comprobando contra los ficheros del 28-10:")
    viejo_pp = leer_csv("scores_por_partido.csv")
    viejo_dx = leer_csv("diagnostico_errores.csv")
    ventana_v, hz_v = ventana_y_1hz(os.path.join(REPO, "out", "actual.csv"))
    exacto = mismo(ventana_v, leer_csv("ventana_publica_5min.csv")[1:], "ventana_publica_5min (out/actual.csv)")
    exacto &= mismo(hz_v, leer_csv("partido_completo_1hz.csv")[1:], "partido_completo_1hz (out/actual.csv)")
    dx_actual = diagnostico(partidos, CONFIGS["actual"], "actual")
    # (en la matriz de confusion n es la celda, que si depende de la prediccion: se compara
    # la suma de cada fila, que es el tamano de la clase real)
    def tamanos(filas):
        out = [(f[1], f[2], int(f[4])) for f in filas if f[1] != "confusion_pct_fila"]
        for real in "HAD":
            out.append(("confusion_pct_fila", "real %s" % real,
                        sum(int(f[4]) for f in filas if f[1] == "confusion_pct_fila" and f[2].startswith("real %s " % real))))
        return out
    exacto &= mismo(tamanos(dx_actual), tamanos([f for f in viejo_dx[1:] if f[0] == "actual"]),
                    "tamanos de categoria del diagnostico")
    pp_actual = por_partido(partidos, CONFIGS["actual"], "actual")
    exacto &= mismo([(f[0], f[5]) for f in pp_actual],
                    [(f[0], f[5]) for f in viejo_pp[1:] if f[2] == "actual"], "posesiones reales por partido")
    if not exacto:
        print("\nNO SE ESCRIBE NADA: lo que tenia que ser exacto no lo es, revisar antes.")
        sys.exit(1)
    agreg_viejo = json.load(open(os.path.join(DATOS, "agregados_por_config.json")))
    print("   la config 'actual' es una reconstruccion; diferencia frente al 28-10 (spellF1):")
    for nombre in ("actual", "solo_centros"):
        for grupo, ids in (("ajuste", TUNE), ("validacion", HOLD), ("todos", PARTIDOS)):
            s = float(np.mean([m.score(CONFIGS[nombre])[0] for m in partidos if m.id in ids]))
            ref = [a for a in agreg_viejo if a["config"] == nombre and a["grupo"] == grupo][0]["spellF1"]
            print("      %-13s %-11s hoy %.4f  28-10 %.4f  (%+.2f pp)" % (nombre, grupo, s, ref, 100 * (s - ref)))

    # 2. regenerar con config.py de hoy ----------------------------------------------------
    print("\nregenerando con config.py de hoy como 'recalibrado':")
    pp = pp_actual + por_partido(partidos, DEFAULT, "recalibrado")
    escribir_csv("scores_por_partido.csv", viejo_pp[0], pp)
    dx = dx_actual + diagnostico(partidos, DEFAULT, "recalibrado")
    escribir_csv("diagnostico_errores.csv", viejo_dx[0], dx)
    ventana, hz = ventana_y_1hz(os.path.join(REPO, "out", "submission.csv"))
    escribir_csv("ventana_publica_5min.csv", ["frame", "segundo_desde_saque", "ground_truth_publico", "modelo_actual"], ventana)
    escribir_csv("partido_completo_1hz.csv", ["parte", "segundo_de_la_parte", "modelo_actual"], hz)

    agreg = []
    for a in agreg_viejo:
        if a["config"] == "original":
            agreg.append(a)
    for orden, (nombre, etiqueta) in enumerate((("actual", "Antes del 28-10"),
                                                ("solo_centros", "Solo centros"),
                                                ("recalibrado", "Hoy (config.py)")), start=2):
        for og, (grupo, ids) in enumerate((("ajuste", TUNE), ("validacion", HOLD), ("todos", PARTIDOS)), start=1):
            sc = [m.score(CONFIGS[nombre]) for m in partidos if m.id in ids]
            agreg.append({"config": nombre, "etiqueta": etiqueta, "orden_config": orden, "grupo": grupo,
                          "orden_grupo": og, "spellF1": round(float(np.mean([x[0] for x in sc])), 4),
                          "macroF1": round(float(np.mean([x[1] for x in sc])), 4)})
    json.dump(agreg, open(os.path.join(DATOS, "agregados_por_config.json"), "w"), indent=1, ensure_ascii=False)

    for a in agreg:
        if a["config"] == "recalibrado":
            print("   recalibrado %-11s spellF1 %.4f  macroF1 %.4f" % (a["grupo"], a["spellF1"], a["macroF1"]))
    for f in pp:
        if f[2] == "recalibrado":
            print("   %s %-11s spellF1 %.4f  poses %d/%d" % (f[0], f[1], f[3], f[6], f[5]))
    from collections import Counter
    print("   ventana publica: modelo", dict(Counter(v[3] for v in ventana)),
          " 1 Hz:", dict(Counter(h[2] for h in hz)))
    print("\nescritos los 5 ficheros en datos/ (%s)" % hoy)


if __name__ == "__main__":
    main()
