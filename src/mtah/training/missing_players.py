"""Prueba de estres: el clasificador de equipo cuando faltan jugadores donde falta el balon.

**Que comprueba.** El +16,1 pp del clasificador de equipo (training/team_classifier.py) se
midio con el IDSSE degradado, que conserva a los jugadores cuando falta el balon: 20,4
objetos de media con y sin balon. En el ATD del reto no es asi: 21,4 con balon, 19,2 sin
el, y el 14,3 % de los frames sin balon tienen menos de 16 (ESTADO.md §6). Es justo donde
el clasificador tiene que trabajar. Aqui se repite la evaluacion del partido de fuera con
el histograma de objetos del ATD *sin balon* impuesto en los frames sin balon
(`degrade.thin_where_ball_unseen`), y nada mas cambia: mismo balon, mismos jugadores en
los frames con balon, mismas semillas.

**Que no se toca.** Los modelos son los de la medicion del 12-11 *tal cual se
entregaron*: para cada partido de fuera, el clasificador y el detector de fronteras se
entrenan con los otros 6 degradados de la forma de siempre (semilla 0). El peso es el de
config.py (`team_classifier_weight`), fijo; no se elige nada aqui. La pregunta es si lo que
se entrego aguanta este dato, no que configuracion lo aguantaria mejor.

**Comprobacion de que el arnes es el mismo.** La misma ejecucion puntua tambien la
degradacion de siempre, que tiene que reproducir `datos/equipo_sin_balon_lomo.json` (el
peso elegido fue 2,0 en los 7 pliegues). Si no lo reproduce, el resultado no vale.

**Regla de decision, escrita el 15-11 antes de correr.** Sobre las 21 ejecuciones (7
partidos x 3 semillas), diferencia de spellF1 con el peso de config.py contra peso 0, en
la degradacion con jugadores perdidos:

- media >= +5 pp y al menos 15 de 21 mejoran: **se mantiene** la entrega `a43c52fb...`;
- media <= +1 pp, o menos de 11 de 21 mejoran: **se vuelve** a `e7938350...`
  (`team_classifier_weight = 0`);
- entre medias: se mantiene, y ESTADO.md cuenta la ganancia reducida.

Con esto no se reajusta nada: ni el peso, ni los costes de cambio, ni las senales.

**Lo que sigue sin ver.** Los jugadores que se quitan se eligen al azar; en el ATD
probablemente faltan los que estan lejos de la camara o amontonados. En los frames con
balon el IDSSE degradado se queda en ~20,4 objetos frente a 21,4 del ATD (solo se puede
quitar, no inventar), lo que hace esta prueba algo mas dura de lo real. Los huecos sin
balon siguen al azar (la prueba de huecos colocados por actividad es otra, `realista.py`).

Uso:  python -m mtah.training.missing_players      -> out/missing_players.json
      (necesita MTAH_DATA_DIR para medir el ATD, requirements-train.txt y scikit-learn)
"""

import json
import os
import time

import numpy as np

from .. import features as feature_set
from ..config import DEFAULT
from ..features import team_signals
from ..features.boundaries import feature_matrix
from ..models import emissions
from ..models import team_classifier as shipped_classifier
from ..paths import REPO_ROOT, cache_dir
from . import boundaries, convention, degrade, harness, idsse
from . import team_classifier as team_training
from .boundaries import SEEDS, log

THINNED_KEYS = boundaries.PREPARED_KEYS + ("stack", "n_tracked")

# Umbrales de la regla de decision del docstring. No se cambian despues de ver el resultado.
KEEP_MIN_PP, KEEP_MIN_BETTER = 5.0, 15
REVERT_MAX_PP, REVERT_MIN_BETTER = 1.0, 11

FEW_PLAYERS = 16

LOMO_TABLE = os.path.join(REPO_ROOT, "trabajo", "handoff-plugin-data", "datos",
                          "equipo_sin_balon_lomo.json")


# --------------------------------------------------------------------------- datos

def _thinning_rng(seed):
    """Generador propio del adelgazado, distinto del de `degrade.apply` con esa semilla."""
    return np.random.default_rng([seed, 7])


def prepare_thinned(match_id, spec, seed, config):
    """Lo de `boundaries.prepare` y `team_classifier.prepare` sobre el tracking adelgazado."""
    path = os.path.join(cache_dir(), "thinned_{}_s{}.npz".format(match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(THINNED_KEYS) <= set(blob.files):
            return {key: blob[key] for key in THINNED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed), gap_weights=None)
    thinned = degrade.thin_where_ball_unseen(dirty, spec, _thinning_rng(seed))
    features = feature_set.build_all(thinned, config)
    matrix, _ = feature_matrix(features, config)
    data = {"cost": emissions.costs(features, config), "matrix": matrix, "labels": labels,
            "ball_seen": thinned.ball_seen,
            "home_logit": emissions.home_logit(features, config),
            "dead_logit": emissions.dead_logit(features, config),
            "stack": team_signals.smoothed(thinned),
            "n_tracked": np.isfinite(thinned.players_xy[:, :, 0]).sum(axis=1).astype(np.int16)}
    np.savez_compressed(path, **data)
    return data


def player_counts(n_tracked, ball_seen):
    """Objetos de media con y sin balon, y fraccion de frames sin balon con menos de 16."""
    unseen = n_tracked[~ball_seen]
    return {"mean_ball_seen": float(n_tracked[ball_seen].mean()),
            "mean_ball_unseen": float(unseen.mean()) if len(unseen) else None,
            "unseen_below_16": float((unseen < FEW_PLAYERS).mean()) if len(unseen) else None}


# --------------------------------------------------------------------------- evaluacion

def evaluate(base, team, thinned, config):
    """Por partido de fuera y semilla: control y peso de config, degradacion normal y adelgazada."""
    weight = config["team_classifier_weight"]
    matches = sorted(team[SEEDS[0]])
    folds = []
    for held_out in matches:
        started = time.time()
        inner = [m for m in matches if m != held_out]
        classifier = team_training.fit([team[SEEDS[0]][m] for m in inner])
        detector = boundaries.fit([base[SEEDS[0]][m] for m in inner])
        fold = {"held_out": held_out, "seeds": {}}
        for seed in SEEDS:
            row = {}
            for name, data, stack in (
                    ("standard", base[seed][held_out], team[seed][held_out]["stack"]),
                    ("thinned", thinned[seed][held_out], thinned[seed][held_out]["stack"])):
                logit = shipped_classifier.antisymmetric_logit(classifier, stack)
                score = boundaries.boundary_score(detector, data)
                control = team_training.decode_score(data, logit, score, config, 0.0)
                chosen = team_training.decode_score(data, logit, score, config, weight)
                quality = team_training.team_quality(
                    {held_out: logit}, {held_out: {"labels": data["labels"],
                                                   "ball_seen": data["ball_seen"]}})
                row[name] = {"control": control, "weighted": chosen, "team": quality,
                             "delta_spell_f1_pp": 100 * (chosen["spell_f1"]
                                                         - control["spell_f1"])}
            row["players_thinned"] = player_counts(thinned[seed][held_out]["n_tracked"],
                                                   thinned[seed][held_out]["ball_seen"])
            fold["seeds"][seed] = row
            log("%s s%d  normal %+.2f pp  adelgazado %+.2f pp  (control %.4f -> %.4f)  "
                "AUC sin balon %.3f -> %.3f  objetos sin balon %.1f, <16 %.1f %%" % (
                    held_out, seed, row["standard"]["delta_spell_f1_pp"],
                    row["thinned"]["delta_spell_f1_pp"],
                    row["standard"]["control"]["spell_f1"],
                    row["thinned"]["control"]["spell_f1"],
                    row["standard"]["team"]["auc_ball_unseen"],
                    row["thinned"]["team"]["auc_ball_unseen"],
                    row["players_thinned"]["mean_ball_unseen"],
                    100 * row["players_thinned"]["unseen_below_16"]))
        log("   %s en %.0fs" % (held_out, time.time() - started))
        folds.append(fold)
    return folds


def check_reproduces_lomo(folds, weight, path=LOMO_TABLE, tolerance=1e-9):
    """Lista de discrepancias de la degradacion normal contra la tabla del 12-11 (vacia = ok)."""
    if not os.path.exists(path):
        return ["no esta {}".format(path)]
    with open(path, encoding="utf-8") as handle:
        reference = {f["held_out"]: f for f in json.load(handle)["folds"]}
    problems = []
    for fold in folds:
        ref = reference.get(fold["held_out"])
        if ref is None:
            problems.append("%s no esta en la tabla" % fold["held_out"])
            continue
        for seed, row in fold["seeds"].items():
            ref_row = ref["seeds"][str(seed)]
            pairs = (("control", row["standard"]["control"], ref_row["control"]),
                     ("peso %g" % weight, row["standard"]["weighted"],
                      ref_row["by_weight"]["%g" % weight]))
            for name, mine, theirs in pairs:
                if abs(mine["spell_f1"] - theirs["spell_f1"]) > tolerance:
                    problems.append("%s s%s %s: %.6f aqui, %.6f en la tabla" % (
                        fold["held_out"], seed, name, mine["spell_f1"], theirs["spell_f1"]))
    return problems


def decide(deltas_pp):
    """La regla del docstring sobre las diferencias de spellF1 (pp) de las 21 ejecuciones."""
    deltas_pp = np.asarray(deltas_pp, dtype=np.float64)
    mean, better = float(deltas_pp.mean()), int((deltas_pp > 0).sum())
    if mean >= KEEP_MIN_PP and better >= KEEP_MIN_BETTER:
        verdict = "mantener"
    elif mean <= REVERT_MAX_PP or better < REVERT_MIN_BETTER:
        verdict = "volver a peso 0"
    else:
        verdict = "mantener, con la ganancia reducida documentada"
    return {"mean_pp": mean, "better": better, "runs": int(len(deltas_pp)),
            "verdict": verdict}


def summarise(folds):
    out = {}
    for name in ("standard", "thinned"):
        deltas = [f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds for s in f["seeds"]]
        per_seed = {s: float(np.mean([f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds]))
                    for s in SEEDS}
        out[name] = dict(decide(deltas), per_seed_mean_pp=per_seed,
                         min_pp=float(np.min(deltas)), max_pp=float(np.max(deltas)))
        log("%-9s %+.2f pp de media (%d de %d mejoran; de %+.2f a %+.2f)  por semilla %s" % (
            name, out[name]["mean_pp"], out[name]["better"], out[name]["runs"],
            out[name]["min_pp"], out[name]["max_pp"],
            "  ".join("%+.2f" % v for v in per_seed.values())))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    log("ATD del reto: %s" % json.dumps(spec.summary()))
    base, team = team_training.load_matches(spec, SEEDS, config)
    thinned = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for match_id in sorted(idsse.MATCHES):
            data = prepare_thinned(match_id, spec, seed, config)
            if not np.array_equal(data["ball_seen"], base[seed][match_id]["ball_seen"]):
                raise ValueError("thinned_{0}_s{1}.npz no sale de la misma degradacion que "
                                 "boundary_{0}_s{1}.npz; borra las dos y vuelve a correr"
                                 .format(match_id, seed))
            thinned[seed][match_id] = data
    log("partidos preparados (%.0fs)" % (time.time() - started))

    folds = evaluate(base, team, thinned, config)
    log("=" * 78)
    problems = check_reproduces_lomo(folds, config["team_classifier_weight"])
    for problem in problems:
        log("NO REPRODUCE: " + problem)
    summary = summarise(folds)
    log("reproduce la tabla del 12-11: %s" % ("si" if not problems else "NO, no vale"))
    log("regla de decision sobre la degradacion adelgazada: %s"
        % summary["thinned"]["verdict"].upper())

    path = os.path.join(REPO_ROOT, "out", "missing_players.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"folds": folds, "summary": summary, "reproduces_lomo": not problems,
                   "lomo_problems": problems, "spec": spec.summary(),
                   "settings": {"weight": config["team_classifier_weight"],
                                "keep": [KEEP_MIN_PP, KEEP_MIN_BETTER],
                                "revert": [REVERT_MAX_PP, REVERT_MIN_BETTER]}},
                  handle, indent=2, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
