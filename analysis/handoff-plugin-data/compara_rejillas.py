"""Compares the two arms of splits.py: original grid vs extended grid.

Both arms run the SAME descent from the SAME starting point over the SAME 35 splits; the
only thing that changes is the list of values of seven parameters. That allows a paired
comparison: for each split, how much more (or less) the extended grid achieves on the same
3 validation matches that the descent did not see.

Usage:  .venv/bin/python -I handoff-plugin-data/compara_rejillas.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT                                 # noqa: E402
from mtah.training.recalibrate import GRID                      # noqa: E402

BASE = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/data/splits_cruzados%s.json"
R = json.load(open(BASE % ""))
A = json.load(open(BASE % "_ampliada"))
assert not A.get("_en_curso"), "the extended arm is still running"
assert len(R["splits"]) == len(A["splits"]) == 35


def alineado(d):
    """Rows indexed by split, to compare apples with apples."""
    return {tuple(sorted(f["tune"])): f for f in d["splits"]}


fr, fa = alineado(R), alineado(A)
assert set(fr) == set(fa), "the two arms do not cover the same splits"
claves = sorted(fr)

# The starting point is the same in both arms: checked, not assumed.
for k in claves:
    for campo in ("tune_inicio", "hold_inicio", "s1_inicio", "s2_inicio"):
        assert abs(fr[k][campo] - fa[k][campo]) < 1e-12, (k, campo)

n = len(claves)
ic = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def col(filas, campo):
    return np.array([filas[k][campo] for k in claves])


print("=" * 88)
print("ORIGINAL GRID (%d values) vs EXTENDED (%d values), %d paired splits"
      % (sum(len(v) for _, v in GRID),
         sum(len(A["rejilla"][nm]) for nm, _ in GRID), n))
print("=" * 88)
print()

print("PROCEDURE GAIN over the starting point (spellF1 pp)")
print("%-34s %-26s %-26s" % ("", "original grid", "ampliada"))
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
print("%-34s %6.0f%%               %6.0f%%" % ("transfer validation/fit", 100 * tr, 100 * ta))
print()

print("PAIRED DIFFERENCE (extended - original), split by split")
print("%-34s %9s %18s %10s %s" % ("", "media", "CI 95%", "wins in", "rango"))
for etiqueta, ini, fin in ejes:
    dr, da = resumen[etiqueta]
    d = 100 * (da - dr)
    print("%-34s %+6.2f pp  [%+5.2f, %+5.2f] pp  %2d/%d    [%+5.2f, %+5.2f]"
          % (etiqueta, d.mean(), d.mean() - ic(d), d.mean() + ic(d),
             (d > 1e-9).sum(), n, d.min(), d.max()))
print()
print("The deciding axis is 'en su propia validacion': the 3 matches that the descent of")
print("that split did not see. Both arms start from the same place on the same matches, so")
print("the paired difference is exactly what being able to search farther adds.")
print()

print("=" * 88)
print("WHAT IT NOW CHOOSES FOR THE SEVEN EXTENDED PARAMETERS")
print("=" * 88)
print("%-22s %-10s %-30s %s" % ("parametro", "config.py", "before (breakdown)", "now (breakdown)"))
ampliados = [nm for nm, vals in GRID if A["rejilla"][nm] != list(vals)]
for nm in ampliados:
    er, ea = R["estabilidad"][nm], A["estabilidad"][nm]
    corto = lambda e: "  ".join("%s:%d" % (k, v) for k, v in list(e["reparto"].items())[:3])
    print("%-22s %-10s %-30s %s" % (nm, DEFAULT[nm], corto(er), corto(ea)))
print()

print("edges: which parameters are still stuck at one end of their list")
sigue = []
for nm, vals in GRID:
    valores = A["rejilla"][nm]
    ea = A["estabilidad"][nm]
    if ea["moda"] in (valores[0], valores[-1]):
        lado = "bajo" if ea["moda"] == valores[0] else "ALTO"
        sigue.append(nm)
        print("   %-22s moda %-7s borde %-5s %2d/%d splits en un borde   %s"
              % (nm, ea["moda"], lado, ea["n_en_borde"], n,
                 "(extended and hits the edge again)" if nm in ampliados else "(not extended)"))
if not sigue:
    print("   none")
print()

print("=" * 88)
print("THE ORIGINAL SPLIT (the one of config.py), to put it in context")
print("=" * 88)
k0 = [k for k in claves if fr[k]["es_split_original"]][0]
for nombre, f in (("original grid", fr[k0]), ("ampliada", fa[k0])):
    print("  %-18s fit %.4f -> %.4f (%+.2f pp)   validation %.4f -> %.4f (%+.2f pp)"
          % (nombre, f["tune_inicio"], f["tune_elegido"],
             100 * (f["tune_elegido"] - f["tune_inicio"]),
             f["hold_inicio"], f["hold_elegido"],
             100 * (f["hold_elegido"] - f["hold_inicio"])))
print("  what changes between the two arrival points:")
for nm, _ in GRID:
    a, b = fr[k0]["elegido"][nm], fa[k0]["elegido"][nm]
    if float(a) != float(b):
        print("     %-22s %s -> %s" % (nm, a, b))
print()

nc_r = np.array([len(fr[k]["cambios"]) for k in claves])
nc_a = np.array([len(fa[k]["cambios"]) for k in claves])
print("changes per descent: original %.1f of 14, extended %.1f of 14"
      % (nc_r.mean(), nc_a.mean()))
