"""Prueba de estres: huecos sin balon colocados como en el ATD, segun la actividad.

En el partido del reto el balon se pierde sobre todo con poca actividad de jugadores (se ve
en el 6 % de los frames por debajo de 0,6 m/s y en el 90 % por encima de 2,6); el proxy lo
esconde al azar. Aqui se recolocan los huecos de la semilla 0 para que P(sin balon |
actividad) siga la curva del ATD (perfil.json, sin etiquetas). La degradacion de jugadores
no cambia: degrade.apply dibuja la mascara del balon lo ultimo, asi que con la misma
semilla solo cambia el balon (se comprueba).

Aqui no se elige nada: los pesos de las senales nuevas son los de la seleccion anidada con
huecos al azar (exp.py). Uso: realista.py prep | modelos | decode
"""

import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
import modelos as M                                            # noqa: E402
from mtah import evaluate as ev                                # noqa: E402
from mtah import features as feature_set                       # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.decode import viterbi                                # noqa: E402
from mtah.features import geometry, smoothing                  # noqa: E402
from mtah.features.boundaries import feature_matrix            # noqa: E402
from mtah.models import emissions                              # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402
from mtah.training import boundaries as B                      # noqa: E402
from mtah.training import convention, degrade, harness, idsse  # noqa: E402

HERE = os.path.dirname(__file__)
SEED = 0
with open(os.path.join(HERE, "perfil.json")) as _handle:
    _PROFILE = json.load(_handle)
EDGES = np.array(_PROFILE["edges"])
TARGET = 1.0 - np.array(_PROFILE["atd"]["seen"]) / np.array(_PROFILE["atd"]["n"])
G = {}


def path(match):
    return os.path.join(cache_dir(), "sinbalon_realista_%s_s%d.npz" % (match, SEED))


def pred_path():
    return os.path.join(cache_dir(), "sinbalon_realista_pred_s%d.npz" % SEED)


def unseen_by_bin(seen, bins):
    count = np.bincount(bins, minlength=len(TARGET)).astype(np.float64)
    return np.bincount(bins, weights=(~seen).astype(np.float64), minlength=len(TARGET)) \
        / np.maximum(count, 1.0), count


def calibrate(bins, spec, rounds=30, draws=3):
    """Peso por tramo de actividad para que P(sin balon | actividad) se acerque al ATD."""
    weight = TARGET / (1.0 - TARGET)
    weight /= weight.max()
    n = len(bins)
    for _ in range(rounds):
        achieved = np.zeros(len(TARGET))
        for k in range(draws):
            seen = degrade._weighted_mask(n, spec.ball_present_runs, spec.ball_absent_runs,
                                          weight[bins], np.random.default_rng(1000 + k))
            achieved += unseen_by_bin(seen, bins)[0] / draws
        weight = weight * ((TARGET + 0.02) / (achieved + 0.02)) ** 0.8
        weight = np.maximum(weight / weight.max(), 1e-5)
    return weight


def prep(match):
    started = time.time()
    clean, labels = idsse.load(match)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    spec = harness.real_spec()
    random = degrade.apply(clean, spec, rng=np.random.default_rng(SEED))
    activity = smoothing.box_mean(geometry.player_activity(random), DEFAULT["activity_smooth"])
    bins = np.clip(np.digitize(np.asarray(activity, dtype=np.float64), EDGES) - 1,
                   0, len(EDGES) - 2)
    weight = calibrate(bins, spec)
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(SEED), gap_weights=weight[bins])
    same_players = bool(np.array_equal(dirty.players_xy, random.players_xy, equal_nan=True))
    features = feature_set.build_all(dirty, DEFAULT)
    matrix, _ = feature_matrix(features, DEFAULT)
    dead, _ = F.dead_matrix(dirty, features, DEFAULT)
    ballteam, _, odd = F.ballteam_matrix(features, F.gap_context(dirty, features), DEFAULT)
    seen = dirty.ball_seen
    np.savez_compressed(path(match), matrix=matrix, labels=labels, ball_seen=seen,
                        home_logit=emissions.home_logit(features, DEFAULT),
                        dead_logit=emissions.dead_logit(features, DEFAULT),
                        dead=dead, ballteam=ballteam, odd=odd)
    unseen, count = unseen_by_bin(seen, bins)
    random_unseen, _ = unseen_by_bin(random.ball_seen, bins)
    gaps = degrade._runs(~seen)
    is_dead = labels == "D"
    return {"match": match, "seconds": round(time.time() - started, 1),
            "same_players": same_players, "count": count.tolist(),
            "unseen": (unseen * count).tolist(), "random_unseen": (random_unseen * count).tolist(),
            "n": int(len(seen)), "n_unseen": int((~seen).sum()),
            "dead_unseen": int((is_dead & ~seen).sum()), "dead_seen": int((is_dead & seen).sum()),
            "random_dead_unseen": int((is_dead & ~random.ball_seen).sum()),
            "random_n_unseen": int((~random.ball_seen).sum()),
            "gaps": int(len(gaps)), "gap_median": float(np.median(gaps)),
            "gap_p90": float(np.percentile(gaps, 90)), "gap_max": int(gaps.max()),
            "random_gaps": int(len(degrade._runs(~random.ball_seen)))}


def run_prep():
    with multiprocessing.get_context("fork").Pool(7) as pool:
        infos = pool.map(prep, C.MATCHES)
    for info in infos:
        print("%s  %.0fs  jugadores iguales que con huecos al azar: %s  sin balon %.1f%%  huecos %d (al azar %d)  mediana %.0f  p90 %.0f  max %d" % (
            info["match"], info["seconds"], info["same_players"],
            100.0 * info["n_unseen"] / info["n"], info["gaps"], info["random_gaps"],
            info["gap_median"], info["gap_p90"], info["gap_max"]), flush=True)
        assert info["same_players"]
    count = np.sum([i["count"] for i in infos], axis=0)
    unseen = np.sum([i["unseen"] for i in infos], axis=0)
    random_unseen = np.sum([i["random_unseen"] for i in infos], axis=0)
    print("actividad (m/s) | sin balon: ATD real  IDSSE al azar  IDSSE realista")
    for b in range(len(TARGET)):
        print("  %.1f - %-4s    |           %5.1f%%        %5.1f%%          %5.1f%%" % (
            EDGES[b], "%.1f" % EDGES[b + 1] if EDGES[b + 1] < 99 else "mas",
            100 * TARGET[b], 100 * random_unseen[b] / count[b], 100 * unseen[b] / count[b]))
    total = {k: sum(i[k] for i in infos) for k in (
        "n", "n_unseen", "dead_unseen", "dead_seen", "random_dead_unseen", "random_n_unseen")}
    print("P(muerto | sin balon)  al azar %.1f%%   realista %.1f%%     P(muerto | visto) realista %.1f%%" % (
        100.0 * total["random_dead_unseen"] / total["random_n_unseen"],
        100.0 * total["dead_unseen"] / total["n_unseen"],
        100.0 * total["dead_seen"] / (total["n"] - total["n_unseen"])))
    with open(os.path.join(HERE, "realista_prep.json"), "w") as handle:
        json.dump({"matches": infos, "target": TARGET.tolist(), "total": total}, handle, indent=1)


def load(match):
    blob = np.load(path(match))
    data = {k: blob[k] for k in blob.files}
    data["team"] = M.feat(match, SEED)["team"]      # los jugadores no cambian
    return data


def run_modelos():
    """Scores y logits del partido de fuera con huecos realistas: modelos entrenados con
    huecos al azar (`rand`, lo que tendriamos hoy) y con huecos realistas (`real`)."""
    base = C.base(SEED)
    rand = {m: M.feat(m, SEED) for m in C.MATCHES}
    real = {m: load(m) for m in C.MATCHES}
    for m in C.MATCHES:
        assert (rand[m]["labels"] == real[m]["labels"]).all()
    out = {}
    started = time.time()
    for held in C.MATCHES:
        others = [m for m in C.MATCHES if m != held]
        for source, train_base, train_feat in (("rand", base, rand), ("real", real, real)):
            model = B.fit([train_base[m] for m in others])
            out["det_%s_%s" % (source, held)] = B.boundary_score(model, real[held]).astype(np.float32)
            for kind in ("b", "c"):
                model = M.fit(kind, [train_feat[m] for m in others])
                out["%s_%s_%s" % (kind, source, held)] = M.predict(kind, model, real[held])
        print("fuera %s  %.0fs" % (held, time.time() - started), flush=True)
        np.savez_compressed(pred_path(), **out)


# --------------------------------------------------------------------------- decodificar

CTL = (("det", "rand"),)
U_SHIFTS = (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0)
VARIANTS = {
    "control (entrega, detector de huecos al azar)": CTL,
    "sin detector (config anterior)": (("det", "none"),),
    "detector reentrenado con huecos realistas": (("det", "real"),),
    "a  equipo sin balon [2.0]": CTL + (("a", 2.0),),
    "c  equipo con contexto, sustituye [3.0]": CTL + (("c_rep", "rand", 3.0),),
    "c  idem, reentrenado realista": CTL + (("c_rep", "real", 3.0),),
    "b  muerto aprendido [1.0, 0.5]": CTL + (("b", "rand", 1.0, 0.5),),
    "b  idem, reentrenado realista": CTL + (("b", "real", 1.0, 0.5),),
    "a + b": CTL + (("a", 2.0), ("b", "rand", 1.0, 0.5)),
    "a + b reentrenado realista": CTL + (("a", 2.0), ("b", "real", 1.0, 0.5)),
    "c + b": CTL + (("c_rep", "rand", 3.0), ("b", "rand", 1.0, 0.5)),
    "c + b, los dos reentrenados realista": CTL + (("c_rep", "real", 3.0), ("b", "real", 1.0, 0.5)),
    "todo reentrenado realista (detector, a, b)": (("det", "real"), ("a", 2.0), ("b", "real", 1.0, 0.5)),
}
for _d in U_SHIFTS:
    VARIANTS["u  empuje a muerto sin balon [%+.1f]" % _d] = CTL + (("u", _d),)
    VARIANTS["a + u [%+.1f]" % _d] = CTL + (("a", 2.0), ("u", _d))


def setup():
    G["real"] = {m: load(m) for m in C.MATCHES}
    G["pred"] = dict(np.load(pred_path()))
    G["a"] = dict(np.load(M.pred_path(SEED)))
    G["gt"] = {m: C.pm.label_codes(G["real"][m]["labels"]) for m in C.MATCHES}
    G["switch"] = viterbi.switch_costs(
        DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"], DEFAULT["switch_from_dead"])


def run(job):
    name, match = job
    d = G["real"][match]
    home = d["home_logit"].astype(np.float64).copy()
    dead = d["dead_logit"].astype(np.float64).copy()
    seen = d["ball_seen"]
    scale, discount = C.SCALE, None
    for part in VARIANTS[name]:
        kind = part[0]
        if kind == "det":
            if part[1] == "none":
                scale, discount = 1.0, None
            else:
                discount = C.DISCOUNT * G["pred"]["det_%s_%s" % (part[1], match)].astype(np.float64)
        elif kind == "a":
            home = home + part[1] * G["a"]["a_%s" % match].astype(np.float64)
        elif kind == "c_rep":
            home = part[2] * G["pred"]["c_%s_%s" % (part[1], match)].astype(np.float64)
        elif kind == "b":
            learned = G["pred"]["b_%s_%s" % (part[1], match)].astype(np.float64)
            dead = (1.0 - part[2]) * dead + part[2] * part[3] * learned
        elif kind == "u":
            dead = dead + part[1] * ~seen
        else:
            raise ValueError(kind)
    cost = emissions.costs_from_logits(home, dead, DEFAULT)
    blocks = ((0, len(cost)),)
    codes = viterbi.decode(cost, scale * G["switch"], blocks=blocks, switch_discount=discount)
    codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"], blocks=blocks)
    labels = viterbi.to_labels(codes)
    r = ev.score(labels, gt_codes=G["gt"][match])
    gt = d["labels"]
    return job, {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
                 "n_gt": r["n_gt_spells"],
                 "acc_seen": float((labels[seen] == gt[seen]).mean()),
                 "acc_unseen": float((labels[~seen] == gt[~seen]).mean()),
                 "pred_dead_unseen": float((labels[~seen] == "D").mean()),
                 "gt_dead_unseen": float((gt[~seen] == "D").mean())}


def run_decode():
    setup()
    jobs = [(name, m) for name in VARIANTS for m in C.MATCHES]
    started = time.time()
    with multiprocessing.get_context("fork").Pool(9) as pool:
        results = dict(pool.imap_unordered(run, jobs, chunksize=2))
    print("%d decodificaciones en %.0fs" % (len(jobs), time.time() - started))

    def col(name, key):
        return np.array([results[(name, m)][key] for m in C.MATCHES])

    control = "control (entrega, detector de huecos al azar)"
    base = col(control, "spell")
    print("%-46s spellF1  delta pp  mej/peor  macroF1  pos pred/real  acierto visto  sin balon  %%D pred sin balon (real %.1f%%)" % (
        "variante (huecos realistas, semilla 0)", 100 * col(control, "gt_dead_unseen").mean()))
    summary = {}
    for name in VARIANTS:
        s = col(name, "spell")
        d = (s - base) * 100
        summary[name] = {"spell": float(s.mean()), "delta_pp": float(d.mean()),
                         "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                         "macro": float(col(name, "macro").mean()),
                         "n_pred": float(col(name, "n_pred").mean()),
                         "n_gt": float(col(name, "n_gt").mean()),
                         "acc_seen": float(col(name, "acc_seen").mean()),
                         "acc_unseen": float(col(name, "acc_unseen").mean()),
                         "pred_dead_unseen": float(col(name, "pred_dead_unseen").mean()),
                         "per_match": s.tolist()}
        print("%-46s %.4f   %+6.2f    %d/%d     %.4f   %5.0f/%-5.0f    %5.1f%%     %5.1f%%     %5.1f%%" % (
            name, s.mean(), d.mean(), (d > 0).sum(), (d < 0).sum(), col(name, "macro").mean(),
            col(name, "n_pred").mean(), col(name, "n_gt").mean(), 100 * col(name, "acc_seen").mean(),
            100 * col(name, "acc_unseen").mean(), 100 * col(name, "pred_dead_unseen").mean()))

    # El empuje `u` es un escalar: se elige dejando un partido fuera (en los otros 6) y se
    # evalua en el de fuera. No usa modelos, asi que no hace falta mas anidado.
    for prefix, label in (("u  empuje a muerto sin balon [%+.1f]", "u"), ("a + u [%+.1f]", "a + u")):
        zero = control if label == "u" else "a  equipo sin balon [2.0]"
        options = [zero] + [prefix % d for d in U_SHIFTS]
        table = np.array([col(o, "spell") for o in options])
        deltas, picks = [], {}
        for h in range(len(C.MATCHES)):
            inner = np.delete(table, h, axis=1).mean(axis=1)
            best = int(np.argmax(inner))
            picks[options[best]] = picks.get(options[best], 0) + 1
            deltas.append(table[best, h] - base[h])
        deltas = np.array(deltas) * 100
        print("ELEGIDO DEJANDO UNO FUERA  %-6s %+.2f pp vs control  (%d mejor / %d peor)  %s" % (
            label, deltas.mean(), (deltas > 0).sum(), (deltas < 0).sum(),
            ", ".join("%s x%d" % kv for kv in picks.items())))
        summary["lomo " + label] = {"delta_pp": float(deltas.mean()), "picks": picks}
    with open(os.path.join(HERE, "realista.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    {"prep": run_prep, "modelos": run_modelos, "decode": run_decode}[sys.argv[1]]()
