"""Efecto de devolver activity_smooth de 12 a 50, medido en los 7 partidos del IDSSE.

Un solo cambio sobre config.py, todo lo demas igual. Se mide en los 4 partidos de ajuste
(donde el 12 se eligio, asi que ahi se espera perder), en los 3 de validacion que no se
usaron, y en las semillas de degradacion 1 y 2, que no vio ningun ajuste.

Uso:  .venv/bin/python -I handoff-plugin-data/activity_smooth.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data")
from splits import cargar, PARTIDOS, SPLIT_ORIGINAL          # noqa: E402

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT, merged                      # noqa: E402

AJUSTE = SPLIT_ORIGINAL
VALIDACION = tuple(m for m in PARTIDOS if m not in AJUSTE)

ANTES = merged()                       # activity_smooth = 12, tal cual config.py
DESPUES = merged(activity_smooth=50)

filas = []
for seed in (0, 1, 2):
    m = {p: cargar(p, seed) for p in PARTIDOS}
    for etiqueta, ids in (("ajuste (4)", AJUSTE), ("validacion (3)", VALIDACION),
                          ("los 7", PARTIDOS)):
        a = float(np.mean([m[p].score(ANTES)[0] for p in ids]))
        b = float(np.mean([m[p].score(DESPUES)[0] for p in ids]))
        filas.append({"semilla": seed, "grupo": etiqueta, "n": len(ids),
                      "smooth12": a, "smooth50": b, "delta_pp": 100 * (b - a)})
    del m

print("DEVOLVER activity_smooth 12 -> 50   (spellF1)")
print("%-8s %-16s %9s %9s %9s" % ("semilla", "grupo", "12", "50", "delta"))
for f in filas:
    print("%-8d %-16s %9.4f %9.4f %+8.2f pp"
          % (f["semilla"], f["grupo"], f["smooth12"], f["smooth50"], f["delta_pp"]))

print()
for grupo in ("ajuste (4)", "validacion (3)", "los 7"):
    d = np.array([f["delta_pp"] for f in filas if f["grupo"] == grupo])
    print("%-16s media de las 3 semillas %+6.2f pp   rango [%+.2f, %+.2f]"
          % (grupo, d.mean(), d.min(), d.max()))

# Por partido, en la semilla 0, para ver si la ganancia es de uno solo o general.
print()
print("por partido (semilla 0):")
m = {p: cargar(p, 0) for p in PARTIDOS}
porpartido = {}
for p in PARTIDOS:
    a, b = m[p].score(ANTES)[0], m[p].score(DESPUES)[0]
    porpartido[p] = {"smooth12": a, "smooth50": b, "delta_pp": 100 * (b - a)}
    print("   %-8s %s  %.4f -> %.4f  %+6.2f pp"
          % (p, "ajuste    " if p in AJUSTE else "validacion", a, b, 100 * (b - a)))
gana = sum(1 for v in porpartido.values() if v["delta_pp"] > 0)
print("   mejora en %d de %d partidos" % (gana, len(PARTIDOS)))

json.dump({"_que_es": "efecto de activity_smooth 12 -> 50, unico cambio sobre config.py",
           "ajuste": list(AJUSTE), "validacion": list(VALIDACION),
           "por_semilla_y_grupo": filas, "por_partido_semilla0": porpartido},
          open("/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/"
               "activity_smooth.json", "w"), indent=2, default=float)
