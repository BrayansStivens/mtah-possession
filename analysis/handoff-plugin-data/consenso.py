"""Two checks on the result of the 35 splits:

  1. What the descent chooses in each split, compared with config.py ON THE SAME hold.
  2. The consensus config (modal value of each parameter across the 35), measured on seeds
     1 and 2, which the descent has not seen. It controls for degradation noise, not for the
     choice of matches: they are the same 7 matches.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")
from mtah.config import DEFAULT, merged
from mtah.training.recalibrate import Match, RAW_KEYS, GRID, mean_score
from mtah.paths import cache_dir

d = json.load(open("handoff-plugin-data/datos/splits_cruzados.json"))
filas, INICIO = d["splits"], d["inicio"]
PARTIDOS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
TUNE_CFGPY = {"J03WMX", "J03WN1", "J03WOH", "J03WOY"}

def cargar(mid, seed):
    b = np.load(os.path.join(cache_dir(), "recal_%s_s%d.npz" % (mid, seed)), allow_pickle=False)
    return Match(mid, {k: b[k] for k in RAW_KEYS}, b["gt"])

cache = {}
def m(mid, seed):
    if (mid, seed) not in cache:
        cache[(mid, seed)] = cargar(mid, seed)
    return cache[(mid, seed)]

print("1. CHOSEN vs config.py, on the same hold of each split")
ganan, limpios_ganan, n_limpios = 0, 0, 0
deltas, deltas_limpios = [], []
for f in filas:
    hold = [m(x, 0) for x in f["hold"]]
    h_cfgpy = mean_score(hold, DEFAULT)[0]
    delta = f["hold_elegido"] - h_cfgpy
    deltas.append(delta)
    if delta > 1e-9:
        ganan += 1
    # "clean" hold for config.py: none of its 3 matches was used to tune config.py
    if not (set(f["hold"]) & TUNE_CFGPY):
        n_limpios += 1; deltas_limpios.append(delta)
        if delta > 1e-9: limpios_ganan += 1
deltas = np.array(deltas)
print("   mean delta (chosen - config.py) on the hold: %+.2f pp   range [%+.2f, %+.2f]"
      % (100*deltas.mean(), 100*deltas.min(), 100*deltas.max()))
print("   the chosen one beats config.py in %d/%d splits" % (ganan, len(filas)))
print("   (warning: config.py was tuned on %s, so in the holds that contain them"
      % ",".join(sorted(x[3:] for x in TUNE_CFGPY)))
print("    its score is contaminated and comes out better than it deserves)")
if n_limpios:
    dl = np.array(deltas_limpios)
    print("   in the %d splits whose hold touches NONE of those 4: %+.2f pp, wins %d/%d"
          % (n_limpios, 100*dl.mean(), limpios_ganan, n_limpios))

print()
print("2. CONSENSUS CONFIG (modal value of the 35 splits)")
consenso = dict(INICIO)
lineas = []
for name, _ in GRID:
    cuenta = collections.Counter(f["elegido"][name] for f in filas)
    valor, n = cuenta.most_common(1)[0]
    consenso[name] = valor
    marca = "" if valor == DEFAULT[name] else "   <- config.py has %s" % DEFAULT[name]
    lineas.append("   %-22s %-6s (%2d/35)%s" % (name, valor, n, marca))
print("\n".join(lineas))
consenso = merged(**consenso)

print()
print("   measured on the seeds the descent did NOT see, all 7 matches (spellF1):")
print("   %-12s %8s %8s %8s" % ("", "partida", "config.py", "consenso"))
for seed in (1, 2):
    siete = [m(x, seed) for x in PARTIDOS]
    a = mean_score(siete, INICIO)[0]; b = mean_score(siete, DEFAULT)[0]
    c = mean_score(siete, consenso)[0]
    print("   seed    %-4d %8.4f %8.4f %8.4f   (consensus - config.py %+.2f pp)"
          % (seed, a, b, c, 100*(c-b)))
siete0 = [m(x, 0) for x in PARTIDOS]
print("   seed    0    %8.4f %8.4f %8.4f   <- this one the descent DID see"
      % (mean_score(siete0, INICIO)[0], mean_score(siete0, DEFAULT)[0],
         mean_score(siete0, consenso)[0]))

print()
print("3. PARAMETERS STUCK TO THE EDGE OF THEIR GRID")
for name, valores in GRID:
    cuenta = collections.Counter(f["elegido"][name] for f in filas)
    tope = max(valores); n_tope = cuenta.get(tope, 0)
    if n_tope >= len(filas) / 2:
        print("   %-22s picks the grid maximum (%s) in %d/35; config.py has %s"
              % (name, tope, n_tope, DEFAULT[name]))

json.dump({"consenso": {k: consenso[k] for k, _ in GRID},
           "delta_elegido_menos_configpy_hold": {
               "media": float(deltas.mean()), "min": float(deltas.min()),
               "max": float(deltas.max()), "gana": int(ganan), "n": len(filas)}},
          open("handoff-plugin-data/datos/consenso_splits.json", "w"), indent=2, default=float)
print()
print("saved handoff-plugin-data/datos/consenso_splits.json")
