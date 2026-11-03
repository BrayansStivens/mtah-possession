"""Compara los dos brazos de splits.py: rejilla original vs rejilla ampliada.

Los dos brazos corren el MISMO descenso desde el MISMO punto de partida sobre los MISMOS 35
splits; lo unico que cambia es la lista de valores de siete parametros. Eso permite comparar
por pares: para cada split, cuanto mas (o menos) consigue la rejilla ampliada en los mismos
3 partidos de validacion que el descenso no vio.

Uso:  .venv/bin/python -I handoff-plugin-data/compara_rejillas.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT                                 # noqa: E402
from mtah.training.recalibrate import GRID                      # noqa: E402

BASE = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/splits_cruzados%s.json"
R = json.load(open(BASE % ""))
A = json.load(open(BASE % "_ampliada"))
assert not A.get("_en_curso"), "el brazo ampliado aun esta corriendo"
assert len(R["splits"]) == len(A["splits"]) == 35


def alineado(d):
    """Filas indexadas por el split, para comparar peras con peras."""
    return {tuple(sorted(f["tune"])): f for f in d["splits"]}


fr, fa = alineado(R), alineado(A)
assert set(fr) == set(fa), "los dos brazos no cubren los mismos splits"
claves = sorted(fr)

# El punto de partida es el mismo en los dos brazos: comprobado, no supuesto.
for k in claves:
    for campo in ("tune_inicio", "hold_inicio", "s1_inicio", "s2_inicio"):
        assert abs(fr[k][campo] - fa[k][campo]) < 1e-12, (k, campo)

n = len(claves)
ic = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def col(filas, campo):
    return np.array([filas[k][campo] for k in claves])


print("=" * 88)
print("REJILLA ORIGINAL (%d valores) vs AMPLIADA (%d valores), %d splits pareados"
      % (sum(len(v) for _, v in GRID),
         sum(len(A["rejilla"][nm]) for nm, _ in GRID), n))
print("=" * 88)
print()

print("GANANCIA DEL PROCEDIMIENTO sobre el punto de partida (pp de spellF1)")
print("%-34s %-26s %-26s" % ("", "rejilla original", "ampliada"))
ejes = [("en su propio ajuste (4 partidos)", "tune_inicio", "tune_elegido"),
        ("en su propia validacion (3)", "hold_inicio", "hold_elegido"),
        ("semilla 1, los 7 partidos", "s1_inicio", "s1_elegido"),
        ("semilla 2, los 7 partidos", "s2_inicio", "s2_elegido")]
resumen = {}
for etiqueta, ini, fin in ejes:
    dr = col(fr, fin) - col(fr, ini)
    da = col(fa, fin) - col(fa, ini)
    resumen[etiqueta] = (dr, da)
    print("%-34s %+6.2f pp  mejora %2d/%d   %+6.2f pp  mejora %2d/%d"
          % (etiqueta, 100 * dr.mean(), (dr > 1e-9).sum(), n,
             100 * da.mean(), (da > 1e-9).sum(), n))

tr = resumen["en su propia validacion (3)"][0].mean() / resumen["en su propio ajuste (4 partidos)"][0].mean()
ta = resumen["en su propia validacion (3)"][1].mean() / resumen["en su propio ajuste (4 partidos)"][1].mean()
print("%-34s %6.0f %%              %6.0f %%" % ("transferencia validacion/ajuste", 100 * tr, 100 * ta))
print()

print("DIFERENCIA PAREADA (ampliada - original), split a split")
print("%-34s %9s %18s %10s %s" % ("", "media", "IC 95 %", "gana en", "rango"))
for etiqueta, ini, fin in ejes:
    dr, da = resumen[etiqueta]
    d = 100 * (da - dr)
    print("%-34s %+6.2f pp  [%+5.2f, %+5.2f] pp  %2d/%d    [%+5.2f, %+5.2f]"
          % (etiqueta, d.mean(), d.mean() - ic(d), d.mean() + ic(d),
             (d > 1e-9).sum(), n, d.min(), d.max()))
print()
print("El eje que decide es 'en su propia validacion': son los 3 partidos que el descenso de")
print("ese split no vio. Los dos brazos parten del mismo sitio en los mismos partidos, asi que")
print("la diferencia pareada es exactamente lo que anade poder buscar mas lejos.")
print()

print("=" * 88)
print("QUE ELIGE AHORA EN LOS SIETE PARAMETROS AMPLIADOS")
print("=" * 88)
print("%-22s %-10s %-30s %s" % ("parametro", "config.py", "antes (reparto)", "ahora (reparto)"))
ampliados = [nm for nm, vals in GRID if A["rejilla"][nm] != list(vals)]
for nm in ampliados:
    er, ea = R["estabilidad"][nm], A["estabilidad"][nm]
    corto = lambda e: "  ".join("%s:%d" % (k, v) for k, v in list(e["reparto"].items())[:3])
    print("%-22s %-10s %-30s %s" % (nm, DEFAULT[nm], corto(er), corto(ea)))
print()

print("bordes: que parametros siguen clavados en un extremo de su lista")
sigue = []
for nm, vals in GRID:
    valores = A["rejilla"][nm]
    ea = A["estabilidad"][nm]
    if ea["moda"] in (valores[0], valores[-1]):
        lado = "bajo" if ea["moda"] == valores[0] else "ALTO"
        sigue.append(nm)
        print("   %-22s moda %-7s borde %-5s %2d/%d splits en un borde   %s"
              % (nm, ea["moda"], lado, ea["n_en_borde"], n,
                 "(ampliado y vuelve a topar)" if nm in ampliados else "(no se amplio)"))
if not sigue:
    print("   ninguno")
print()

print("=" * 88)
print("EL SPLIT ORIGINAL (el de config.py), para situarlo")
print("=" * 88)
k0 = [k for k in claves if fr[k]["es_split_original"]][0]
for nombre, f in (("rejilla original", fr[k0]), ("ampliada", fa[k0])):
    print("  %-18s ajuste %.4f -> %.4f (%+.2f pp)   validacion %.4f -> %.4f (%+.2f pp)"
          % (nombre, f["tune_inicio"], f["tune_elegido"],
             100 * (f["tune_elegido"] - f["tune_inicio"]),
             f["hold_inicio"], f["hold_elegido"],
             100 * (f["hold_elegido"] - f["hold_inicio"])))
print("  lo que cambia entre los dos puntos de llegada:")
for nm, _ in GRID:
    a, b = fr[k0]["elegido"][nm], fa[k0]["elegido"][nm]
    if float(a) != float(b):
        print("     %-22s %s -> %s" % (nm, a, b))
print()

nc_r = np.array([len(fr[k]["cambios"]) for k in claves])
nc_a = np.array([len(fa[k]["cambios"]) for k in claves])
print("cambios por descenso: original %.1f de 14, ampliada %.1f de 14"
      % (nc_r.mean(), nc_a.mean()))
