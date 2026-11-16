"""Anade a los ficheros del dashboard las dos configuraciones medidas fuera de muestra.

regenera_datos.py puntua configuraciones de config.py sobre los 7 partidos del IDSSE. Desde
el 7-11 la entrega lleva ademas dos modelos entrenados en esos mismos partidos (el detector
de fronteras y el clasificador de equipo): puntuarla tal cual sobre ellos seria puntuar en
muestra. Aqui cada partido se decodifica con modelos que no lo vieron, igual que en
`mtah.training.team_classifier` (semilla 0 de degradacion):

    detector  = config.py con el detector de fronteras de los otros 6 partidos y sin
                clasificador de equipo. Es el "control" de datos/equipo_sin_balon_lomo.json
    equipo    = lo mismo mas el clasificador de equipo de los otros 6, con peso 2,0. Es el
                "elegido" de ese fichero: el peso salio 2,0 en los 7 pliegues

El IDSSE no trae Smart Tagging, asi que ninguna de las dos lo lleva.

Antes de escribir nada comprueba que cada partido reproduce exactamente el spellF1, el
macroF1 y las posesiones de datos/equipo_sin_balon_lomo.json, que las categorias del
diagnostico tienen el mismo tamano que las de regenera_datos.py (misma degradacion) y que
out/submission.csv es el que genera main. Si algo no cuadra, no escribe.

Uso:  source scripts/env.sh && python trabajo/handoff-plugin-data/regenera_equipo.py
"""

import hashlib
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import regenera_datos as R                                        # noqa: E402
from mtah import evaluate as ev                                   # noqa: E402
from mtah.config import DEFAULT                                   # noqa: E402
from mtah.models import emissions, team_classifier as modelo      # noqa: E402
from mtah.training import boundaries, harness                     # noqa: E402
from mtah.training import team_classifier as arnes                # noqa: E402

pm = ev.pm
SEMILLA = 0
SUBMISSION_SHA256 = "a43c52fb264182c06ced947cfbf47f6e86b82cf589e3d165760449d39fe2906b"
NUEVAS = (("detector", "Con detector (7-11)", 5, 0.0),
          ("equipo", "Hoy (con clasificador)", 6, 2.0))
ETIQUETA_RECALIBRADO = "Recalibrado (31-10)"
LOMO = json.load(open(os.path.join(R.DATOS, "equipo_sin_balon_lomo.json")))
SIN_BALON = os.path.join(AQUI, "..", "sin-balon-1111")
# Las entregas que se conservan en out/, para releerlas en los 5 minutos publicos.
ENTREGAS = (("31-10", "submission_31-10_73c361d5.csv", "73c361d5", "recalibrada"),
            ("8-11", "submission_0811_e7938350.csv", "e7938350", "con detector y Smart Tagging"),
            ("15-11", "submission.csv", "a43c52fb", "con clasificador de equipo"))


def etiquetas(base, logit, score, config, peso):
    """Lo mismo que `team_classifier.decode_score`, pero devolviendo las etiquetas."""
    data = base
    if peso:
        home = base["home_logit"].astype(np.float64) + peso * logit.astype(np.float64)
        data = dict(base, cost=emissions.costs_from_logits(
            home, base["dead_logit"].astype(np.float64), config))
    return np.asarray(boundaries.decode(data, config, config["boundary_discount_weight"],
                                        config["boundary_switch_scale"], score), dtype="<U1")


def decodificar_todo(config):
    """{config: {partido: (reales, predichos, balon visto, etiquetas)}}, todo fuera de muestra."""
    base, team = arnes.load_matches(harness.real_spec(), (SEMILLA,), config)
    detector = arnes.out_of_fold_detector(base)
    partidos = sorted(team[SEMILLA])
    out = {nombre: {} for nombre, _, _, _ in NUEVAS}
    for fuera in partidos:
        clasificador = arnes.fit([team[SEMILLA][m] for m in partidos if m != fuera])
        logit = modelo.antisymmetric_logit(clasificador, team[SEMILLA][fuera]["stack"])
        b = base[SEMILLA][fuera]
        for nombre, _, _, peso in NUEVAS:
            lab = etiquetas(b, logit, detector[SEMILLA][fuera], config, peso)
            out[nombre][fuera] = (pm.label_codes(b["labels"]), pm.label_codes(lab),
                                  b["ball_seen"], lab)
        print("   decodificado %s" % fuera, flush=True)
    return out


def por_partido(decodificado, nombre):
    filas = []
    for mid in R.PARTIDOS:
        gt, _, _, lab = decodificado[nombre][mid]
        r = ev.score(lab, gt_codes=gt)
        filas.append((mid, R.GRUPO[mid], nombre, round(r["spell_f1_mean"], 4),
                      round(r["macro_f1"], 4), r["n_gt_spells"], r["n_pred_spells"],
                      round(r["gt_mean_spell_duration_s"], 1), round(r["mean_spell_duration_s"], 1)))
    return filas


def reproduce_el_lomo(decodificado):
    """Cada partido tiene que dar lo que guardo la medicion del 12-11, sin redondear."""
    ok = True
    pliegues = {f["held_out"]: f for f in LOMO["folds"]}
    for nombre, clave in (("detector", "control"), ("equipo", "chosen")):
        for mid in R.PARTIDOS:
            gt, _, _, lab = decodificado[nombre][mid]
            r = ev.score(lab, gt_codes=gt)
            ref = pliegues[mid]["seeds"][str(SEMILLA)][clave]
            igual = (abs(r["spell_f1_mean"] - ref["spell_f1"]) < 1e-12
                     and abs(r["macro_f1"] - ref["macro_f1"]) < 1e-12
                     and r["n_pred_spells"] == ref["n_pred_spells"]
                     and r["n_gt_spells"] == ref["n_gt_spells"])
            if not igual:
                print("   DIFIERE %s %s: aqui %.6f, LOMO %.6f" % (
                    nombre, mid, r["spell_f1_mean"], ref["spell_f1"]))
            ok &= igual
    if any(f["choice"]["weight"] != 2.0 for f in LOMO["folds"]):
        print("   el peso elegido no es 2,0 en todos los pliegues: 'equipo' no seria el elegido")
        ok = False
    if ok:
        print("   ok  los 14 partidos-config reproducen equipo_sin_balon_lomo.json")
    return ok


def tamanos(filas):
    """Tamano de cada categoria, que depende de los datos y no de la prediccion."""
    out = [(f[1], f[2], int(f[4])) for f in filas if f[1] != "confusion_pct_fila"]
    for real in "HAD":
        out.append(("confusion_pct_fila", "real %s" % real, sum(
            int(f[4]) for f in filas
            if f[1] == "confusion_pct_fila" and f[2].startswith("real %s " % real))))
    return out


def clasificador_rows():
    """Las 21 ejecuciones del LOMO y su resumen, mas la medida con huecos realistas."""
    filas, resumen = [], []
    for fold in LOMO["folds"]:
        for semilla, fila in sorted(fold["seeds"].items()):
            c, e = fila["control"], fila["chosen"]
            filas.append({
                "partido": fold["held_out"], "grupo": R.GRUPO[fold["held_out"]],
                "semilla": int(semilla), "peso_elegido": fold["choice"]["weight"],
                "sin_clasificador": round(c["spell_f1"], 4),
                "con_clasificador": round(e["spell_f1"], 4),
                "ganancia_pp": round(100 * (e["spell_f1"] - c["spell_f1"]), 2),
                "macroF1_sin": round(c["macro_f1"], 4), "macroF1_con": round(e["macro_f1"], 4),
                "posesiones_reales": c["n_gt_spells"],
                "predichas_sin": c["n_pred_spells"], "predichas_con": e["n_pred_spells"],
                "auc_equipo": round(fila["team"]["auc"], 3),
                "auc_equipo_sin_balon": round(fila["team"]["auc_ball_unseen"], 3)})
    semillas = sorted(LOMO["folds"][0]["seeds"])
    for semilla in semillas:
        resumen.append(_resumen(semilla, [semilla]))
    resumen.append(_resumen("media", semillas))
    realista = json.load(open(os.path.join(SIN_BALON, "realista.json")))
    robusto = json.load(open(os.path.join(SIN_BALON, "robusto.json")))
    control = realista["control (entrega, detector de huecos al azar)"]["spell"]
    delta, mejor, peor = robusto["sin opp, cen, len"]["delta"]["real"]
    resumen.append({"huecos": "realistas", "semilla": "0",
                    "sin_clasificador": round(control, 4),
                    "con_clasificador": round(control + delta / 100, 4),
                    "ganancia_pp": round(delta, 2), "mejoran": mejor, "empeoran": peor,
                    "ejecuciones": mejor + peor})
    return filas, resumen


def lectura_publica():
    """Cada entrega conservada, puntuada en los 5 minutos publicos. Lectura, no criterio."""
    filas = []
    for fecha, fichero, sha, que in ENTREGAS:
        ruta = os.path.join(R.REPO, "out", fichero)
        if not os.path.exists(ruta):
            continue
        if not hashlib.sha256(open(ruta, "rb").read()).hexdigest().startswith(sha):
            raise ValueError("%s no es la entrega %s" % (ruta, sha))
        r = ev.score(np.asarray(pm.read_label_csv(ruta)["label"], dtype="<U1"))
        filas.append({"entrega": fecha, "que_lleva": que, "sha256": sha,
                      "spellF1": round(r["spell_f1_mean"], 4), "macroF1": round(r["macro_f1"], 4),
                      "posesiones_predichas": r["n_pred_spells"],
                      "posesiones_reales": r["n_gt_spells"]})
    return filas


def escribir_lectura_publica():
    json.dump({"_que_es": "spellF1 de cada entrega conservada en out/ sobre los 5 minutos con "
                          "ground truth publico. Son 17 posesiones: es una lectura, nunca un "
                          "criterio para elegir entre entregas.", "rows": lectura_publica()},
              open(os.path.join(R.DATOS, "lectura_publica.json"), "w"), indent=1,
              ensure_ascii=False)


def _resumen(nombre, semillas):
    """Media sin redondear de las ejecuciones de esas semillas, con huecos al azar."""
    pares = [(fold["seeds"][s]["control"], fold["seeds"][s]["chosen"])
             for fold in LOMO["folds"] for s in semillas]
    ganancia = [100 * (e["spell_f1"] - c["spell_f1"]) for c, e in pares]
    macro = [100 * (e["macro_f1"] - c["macro_f1"]) for c, e in pares]
    return {"huecos": "azar", "semilla": nombre,
            "sin_clasificador": round(float(np.mean([c["spell_f1"] for c, _ in pares])), 4),
            "con_clasificador": round(float(np.mean([e["spell_f1"] for _, e in pares])), 4),
            "ganancia_pp": round(float(np.mean(ganancia)), 2),
            "ganancia_min_pp": round(min(ganancia), 2), "ganancia_max_pp": round(max(ganancia), 2),
            "ganancia_macroF1_pp": round(float(np.mean(macro)), 2),
            "mejoran": sum(g > 0 for g in ganancia), "empeoran": sum(g < 0 for g in ganancia),
            "ejecuciones": len(ganancia)}


def main():
    submission = os.path.join(R.REPO, "out", "submission.csv")
    sha = hashlib.sha256(open(submission, "rb").read()).hexdigest()
    print("comprobando:")
    ok = sha == SUBMISSION_SHA256
    print("   %s out/submission.csv es %s" % ("ok " if ok else "NO ", sha[:8]))

    decodificado = decodificar_todo(DEFAULT)
    ok &= reproduce_el_lomo(decodificado)

    viejo_pp = R.leer_csv("scores_por_partido.csv")
    viejo_dx = R.leer_csv("diagnostico_errores.csv")
    previas = ("actual", "recalibrado")
    pp = [f for f in viejo_pp[1:] if f[2] in previas]
    dx = [f for f in viejo_dx[1:] if f[0] in previas]
    referencia = tamanos([f for f in dx if f[0] == "actual"])
    for nombre, _, _, _ in NUEVAS:
        nuevas = R.diagnostico_de((decodificado[nombre][m][:3] for m in R.PARTIDOS), nombre)
        ok &= R.mismo(tamanos(nuevas), referencia, "tamanos de categoria de '%s'" % nombre)
        dx += nuevas
        filas = por_partido(decodificado, nombre)
        ok &= R.mismo([(f[0], f[5]) for f in filas],
                      [(f[0], f[5]) for f in pp if f[2] == "actual"],
                      "posesiones reales por partido de '%s'" % nombre)
        pp += filas
    if not ok:
        print("\nNO SE ESCRIBE NADA: lo que tenia que ser exacto no lo es, revisar antes.")
        sys.exit(1)

    R.escribir_csv("scores_por_partido.csv", viejo_pp[0], pp)
    R.escribir_csv("diagnostico_errores.csv", viejo_dx[0], dx)
    ventana, hz = R.ventana_y_1hz(submission)
    R.escribir_csv("ventana_publica_5min.csv",
                   ["frame", "segundo_desde_saque", "ground_truth_publico", "modelo_actual"], ventana)
    R.escribir_csv("partido_completo_1hz.csv", ["parte", "segundo_de_la_parte", "modelo_actual"], hz)

    ruta = os.path.join(R.DATOS, "agregados_por_config.json")
    agreg = [a for a in json.load(open(ruta)) if a["config"] not in [n for n, _, _, _ in NUEVAS]]
    for a in agreg:
        if a["config"] == "recalibrado":
            a["etiqueta"] = ETIQUETA_RECALIBRADO
    for nombre, etiqueta, orden, _ in NUEVAS:
        for og, (grupo, ids) in enumerate((("ajuste", R.TUNE), ("validacion", R.HOLD),
                                           ("todos", R.PARTIDOS)), start=1):
            sel = [f for f in pp if f[2] == nombre and f[0] in ids]
            agreg.append({"config": nombre, "etiqueta": etiqueta, "orden_config": orden,
                          "grupo": grupo, "orden_grupo": og,
                          "spellF1": round(float(np.mean([float(f[3]) for f in sel])), 4),
                          "macroF1": round(float(np.mean([float(f[4]) for f in sel])), 4)})
    json.dump(agreg, open(ruta, "w"), indent=1, ensure_ascii=False)

    filas, resumen = clasificador_rows()
    json.dump({"_que_es": "Las 21 ejecuciones del leave-one-match-out anidado del clasificador "
                          "de equipo (7 partidos por 3 semillas de degradacion), redondeadas. "
                          "Sale de equipo_sin_balon_lomo.json.", "rows": filas},
              open(os.path.join(R.DATOS, "clasificador_equipo.json"), "w"), indent=1,
              ensure_ascii=False)
    json.dump({"_que_es": "Resumen del anterior por semilla y en media, mas la medida con los "
                          "huecos de balon colocados como en el partido real (semilla 0, "
                          "trabajo/sin-balon-1111/realista.json y robusto.json).", "rows": resumen},
              open(os.path.join(R.DATOS, "clasificador_equipo_resumen.json"), "w"), indent=1,
              ensure_ascii=False)

    escribir_lectura_publica()

    for a in agreg:
        if a["config"] in ("recalibrado", "detector", "equipo"):
            print("   %-11s %-11s spellF1 %.4f  macroF1 %.4f" % (
                a["config"], a["grupo"], a["spellF1"], a["macroF1"]))
    for r in resumen:
        print("   huecos %-9s semilla %-5s %.4f -> %.4f  %+.2f pp  %d de %d" % (
            r["huecos"], r["semilla"], r["sin_clasificador"], r["con_clasificador"],
            r["ganancia_pp"], r["mejoran"], r["ejecuciones"]))
    print("\nescritos 8 ficheros en datos/")


if __name__ == "__main__":
    main()
