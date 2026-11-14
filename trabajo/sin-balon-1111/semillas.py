"""Las tres semillas de degradacion (huecos al azar) para el clasificador de equipo sin balon.

Mismo protocolo que el detector: cada modelo se entrena con la semilla 0 de los otros 6
partidos y puntua las tres semillas del partido de fuera. Pesos fijados de antemano (los de
exp.py): aqui no se elige nada. Uso: semillas.py modelos | decode
"""

import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import modelos as M                                            # noqa: E402
import robusto as R                                            # noqa: E402
from mtah import evaluate as ev                                # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.decode import viterbi                                # noqa: E402
from mtah.models import emissions                              # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402

HERE = os.path.dirname(__file__)
TEAM = ("todas (9)", "sin opp, cen, len")
VARIANTS = {
    "control (entrega)": (None, False),
    "a todas [2.0]": ("todas (9)", False),
    "a sin opp, cen, len [2.0]": ("sin opp, cen, len", False),
    "b muerto aprendido [1.0, 0.5]": (None, True),
    "a sin opp, cen, len + b": ("sin opp, cen, len", True),
}
G = {}


def pred_path():
    return os.path.join(cache_dir(), "sinbalon_semillas.npz")


def run_modelos():
    data = {s: {m: M.feat(m, s) for m in C.MATCHES} for s in C.SEEDS}
    cached = dict(np.load(R.pred_path()))
    cached_b = np.load(M.pred_path(0))
    out = {}
    started = time.time()
    for held in C.MATCHES:
        train = [data[0][m] for m in C.MATCHES if m != held]
        for name in TEAM:
            build = R.builder(R.SUBSETS[name])
            model = R.fit(train, build)
            for s in C.SEEDS:
                out["%s|%s|%d" % (name, held, s)] = R.predict(model, data[s][held]["team"], build)
            assert np.allclose(out["%s|%s|0" % (name, held)], cached["%s|%s" % (name, held)], atol=1e-4)
        model = M.fit("b", train)
        for s in C.SEEDS:
            out["b|%s|%d" % (held, s)] = M.predict("b", model, data[s][held])
        assert np.allclose(out["b|%s|0" % held], cached_b["b_%s" % held], atol=1e-4)
        print("fuera %s  %.0fs" % (held, time.time() - started), flush=True)
    np.savez_compressed(pred_path(), **out)


def setup():
    G["pred"] = dict(np.load(pred_path()))
    G["base"] = {s: C.base(s) for s in C.SEEDS}
    G["score"] = C.oof_scores(None)
    G["gt"] = {(s, m): C.pm.label_codes(G["base"][s][m]["labels"]) for s in C.SEEDS for m in C.MATCHES}
    G["transition"] = C.SCALE * viterbi.switch_costs(
        DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"], DEFAULT["switch_from_dead"])


def run(job):
    variant, seed, match = job
    team, learned_dead = VARIANTS[variant]
    d = G["base"][seed][match]
    home = d["home_logit"].astype(np.float64)
    dead = d["dead_logit"].astype(np.float64)
    if team is not None:
        home = home + R.WEIGHT * G["pred"]["%s|%s|%d" % (team, match, seed)].astype(np.float64)
    if learned_dead:
        dead = 0.5 * G["pred"]["b|%s|%d" % (match, seed)].astype(np.float64)
    cost = emissions.costs_from_logits(home, dead, DEFAULT)
    blocks = ((0, len(cost)),)
    codes = viterbi.decode(cost, G["transition"], blocks=blocks,
                           switch_discount=C.DISCOUNT * G["score"][seed][match].astype(np.float64))
    codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"], blocks=blocks)
    r = ev.score(viterbi.to_labels(codes), gt_codes=G["gt"][(seed, match)])
    return job, (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"])


def run_decode():
    setup()
    jobs = [(v, s, m) for v in VARIANTS for s in C.SEEDS for m in C.MATCHES]
    with multiprocessing.get_context("fork").Pool(9) as pool:
        results = dict(pool.imap_unordered(run, jobs, chunksize=2))
    summary = {}
    print("%-30s | spellF1 por semilla        | delta pp por semilla (mejor/peor de 7)            | media  | macroF1 | pos pred/real" % "variante")
    for v in VARIANTS:
        spell, delta, wins = [], [], []
        for s in C.SEEDS:
            x = np.array([results[(v, s, m)][0] for m in C.MATCHES])
            c = np.array([results[("control (entrega)", s, m)][0] for m in C.MATCHES])
            spell.append(float(x.mean()))
            delta.append(float((x - c).mean() * 100))
            wins.append((int((x > c).sum()), int((x < c).sum())))
        macro = float(np.mean([results[(v, s, m)][1] for s in C.SEEDS for m in C.MATCHES]))
        pred = float(np.mean([results[(v, s, m)][2] for s in C.SEEDS for m in C.MATCHES]))
        real = float(np.mean([results[(v, s, m)][3] for s in C.SEEDS for m in C.MATCHES]))
        summary[v] = {"spell": spell, "delta_pp": delta, "wins": wins, "macro": macro,
                      "pred_spells": pred, "gt_spells": real}
        print("%-30s | %s | %s | %+6.2f | %.4f  | %.0f/%.0f" % (
            v, "  ".join("%.4f" % x for x in spell),
            "  ".join("%+6.2f (%d/%d)" % (d, w[0], w[1]) for d, w in zip(delta, wins)),
            float(np.mean(delta)), macro, pred, real))
    with open(os.path.join(HERE, "semillas.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    {"modelos": run_modelos, "decode": run_decode}[sys.argv[1]]()
