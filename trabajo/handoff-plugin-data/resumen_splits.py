import json, sys
import numpy as np
sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT
from mtah.training.recalibrate import GRID

RUTA = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/splits_cruzados%s.json"
brazo = sys.argv[1] if len(sys.argv) > 1 else "rejilla"
d = json.load(open(RUTA % ("" if brazo == "rejilla" else "_" + brazo)))
filas, INICIO, r = d["splits"], d["inicio"], d["resumen"]
# La rejilla efectiva de este brazo, para saber donde estan los bordes.
REJILLA = [(n, d.get("rejilla", {}).get(n, v)) for n, v in GRID]
print("brazo '%s': %d valores en la rejilla\n" % (brazo, sum(len(v) for _, v in REJILLA)))

dt = np.array([f["tune_elegido"] - f["tune_inicio"] for f in filas])
dh = np.array([f["hold_elegido"] - f["hold_inicio"] for f in filas])
print("GANANCIA DEL PROCEDIMIENTO sobre %d splits (spellF1, puntos porcentuales)" % len(filas))
print("  en su propio ajuste     %+.2f pp   rango [%+.2f, %+.2f]   mejora en %d/%d"
      % (100*dt.mean(), 100*dt.min(), 100*dt.max(), (dt > 1e-9).sum(), len(filas)))
print("  en su propia validacion %+.2f pp   rango [%+.2f, %+.2f]   mejora en %d/%d"
      % (100*dh.mean(), 100*dh.min(), 100*dh.max(), (dh > 1e-9).sum(), len(filas)))
print("  transferencia: %.0f %%" % (100*dh.mean()/dt.mean()))
print("  IC95 de la media en validacion: [%+.2f, %+.2f] pp"
      % (100*(dh.mean()-1.96*dh.std(ddof=1)/np.sqrt(len(dh))),
         100*(dh.mean()+1.96*dh.std(ddof=1)/np.sqrt(len(dh)))))
for s in (1, 2):
    x = np.array([f["s%d_elegido" % s] - f["s%d_inicio" % s] for f in filas])
    print("  semilla %d, los 7 partidos %+.2f pp   rango [%+.2f, %+.2f]   mejora en %d/%d"
          % (s, 100*x.mean(), 100*x.min(), 100*x.max(), (x > 1e-9).sum(), len(filas)))

orig = [f for f in filas if f["es_split_original"]][0]
print()
print("el split original, para situarlo:")
print("  ajuste %.4f -> %.4f (%+.2f pp)   validacion %.4f -> %.4f (%+.2f pp)"
      % (orig["tune_inicio"], orig["tune_elegido"], 100*(orig["tune_elegido"]-orig["tune_inicio"]),
         orig["hold_inicio"], orig["hold_elegido"], 100*(orig["hold_elegido"]-orig["hold_inicio"])))
pct = (dh < orig["hold_elegido"] - orig["hold_inicio"]).mean()
print("  su delta en validacion esta en el percentil %.0f de los 35" % (100*pct))

print()
print("ESTABILIDAD: que elige el descenso en cada uno de los 35 splits")
print("%-22s %-9s %-10s %-5s %-6s %s"
      % ("parametro", "partida", "config.py", "=cfg", "borde", "reparto"))
for name, valores in REJILLA:
    e = d["estabilidad"][name]
    bordes = (valores[0], valores[-1])
    moda = e["moda"]
    marca = ""
    if moda == valores[-1]:
        marca = "ALTO"
    elif moda == valores[0]:
        marca = "bajo"
    print("%-22s %-9s %-10s %2d/%-3d %-6s %s" % (
        name, e["inicio"], DEFAULT[name], e["coincide_config_py"], len(filas), marca,
        "  ".join("%s:%d" % (k, v) for k, v in e["reparto"].items())))

print()
n_cambios = [len(f["cambios"]) for f in filas]
print("cambios por descenso: media %.1f, rango %d-%d de 14 parametros"
      % (np.mean(n_cambios), min(n_cambios), max(n_cambios)))
