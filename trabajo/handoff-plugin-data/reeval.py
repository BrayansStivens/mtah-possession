"""Re-evaluacion de configuraciones sobre los 7 partidos del IDSSE, desde la cache.

No toca el ATD ni la ventana publica: solo lee out/cache/recal_<id>_s0.npz, que son los
features crudos de los partidos externos degradados, y puntua configuraciones arbitrarias.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/Users/bultoesal/Documents/Hackaton/mtah-possession/src")

from mtah.config import DEFAULT, merged
from mtah.training.recalibrate import Match, RAW_KEYS, TUNE, HOLD, mean_score
from mtah.paths import cache_dir

def load(mid, seed=0):
    blob = np.load(os.path.join(cache_dir(), "recal_%s_s%d.npz" % (mid, seed)),
                   allow_pickle=False)
    return Match(mid, {k: blob[k] for k in RAW_KEYS}, blob["gt"])

tune = [load(m) for m in TUNE]
hold = [load(m) for m in HOLD]

REF = json.load(open("/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/recalibracion.json"))
CENTROS = {k: REF["config"][k] for k in
           ("ball_speed_center_ms", "control_center", "shape_center")}

# Config anterior al 28-10, reconstruida desde la procedencia documentada en config.py.
ANTES = dict(DEFAULT)
ANTES.update(activity_smooth=50, ball_speed_smooth=50, control_smooth=50, shape_smooth=50,
             ball_speed_center_ms=4.10, control_center=0.096, shape_center=0.975,
             switch_turnover=28.0)

SOLO_CENTROS = merged(**{**ANTES, **CENTROS})

def fila(nombre, cfg):
    ts, tm = mean_score(tune, cfg)
    hs, hm = mean_score(hold, cfg)
    als, alm = mean_score(tune + hold, cfg)
    print("%-26s ajuste %.4f  validacion %.4f  todos %.4f   | macro %.4f %.4f %.4f"
          % (nombre, ts, hs, als, tm, hm, alm), flush=True)
    return {"tune": ts, "hold": hs, "all": als, "macro_all": alm}

print("referencia del JSON (spellF1 ajuste / validacion / todos):")
for k, v in REF["scores"].items():
    print("   %-22s %.4f  %.4f  %.4f" % (k, v["tune"][0], v["hold"][0], v["all"][0]))
print()
print("medido ahora desde la cache:")
got = {}
got["antes_28oct"] = fila("antes del 28-10", merged(**ANTES))
got["solo_centros"] = fila("solo_centros", SOLO_CENTROS)
got["recalibrado"] = fila("recalibrado (config.py)", DEFAULT)

print()
print("por partido (spellF1):")
print("%-8s %-6s %9s %9s %9s" % ("partido", "grupo", "antes", "centros", "recal"))
for grupo, ms in (("ajuste", tune), ("valid.", hold)):
    for m in ms:
        a = m.score(merged(**ANTES))[0]
        c = m.score(SOLO_CENTROS)[0]
        r = m.score(DEFAULT)[0]
        print("%-8s %-6s %9.4f %9.4f %9.4f" % (m.id, grupo, a, c, r))

json.dump(got, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reeval.json"), "w"), indent=2)
