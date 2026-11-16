"""Prueba de estres: el clasificador de equipo cuando faltan jugadores donde falta el balon.

**Que comprueba.** El +16,1 pp del clasificador de equipo (training/team_classifier.py) se
midio con el IDSSE degradado, que conserva a los jugadores cuando falta el balon: 20,4
objetos de media con y sin balon. En el ATD del reto no es asi: 21,4 con balon, 19,2 sin
el, y el 14,3 % de los frames sin balon tienen menos de 16 (ESTADO.md §6). Es justo donde
el clasificador tiene que trabajar. Aqui se repite la evaluacion del partido de fuera con
el histograma de objetos del ATD *sin balon* impuesto en los frames sin balon
(`degrade.thin_where_ball_unseen`), y nada mas cambia: mismo balon, mismos jugadores en
los frames con balon, mismas semillas.

Dos brazos con el mismo histograma por frame:

- `thinned`: los jugadores que se quitan se sortean en cada frame. Parpadean, y las
  ventanas de 1 a 8 s de las senales promedian buena parte de ese ruido: es la version
  facil.
- `persistent`: se sortean una vez por racha sin balon y faltan durante toda la racha,
  como un track perdido del ATD. **La decision sale de este brazo.**

**Que no se toca.** Los modelos son los de la medicion del 12-11 *tal cual se
entregaron*: para cada partido de fuera, el clasificador y el detector de fronteras se
entrenan con los otros 6 degradados de la forma de siempre (semilla 0). El peso es el de
config.py (`team_classifier_weight`), fijo; no se elige nada aqui. La pregunta es si lo que
se entrego aguanta este dato, no que configuracion lo aguantaria mejor.

**Comprobacion de que el arnes es el mismo.** La misma ejecucion puntua tambien la
degradacion de siempre (`standard`), que tiene que reproducir
`datos/equipo_sin_balon_lomo.json` (el peso elegido fue 2,0 en los 7 pliegues). Las
diferencias de los brazos son pareadas contra su propio control con los mismos modelos,
asi que un desajuste pequeno (otra maquina reentrenando el boosting) no las invalida, pero
se informa.

**Regla de decision, escrita el 15-11 antes de ver ningun resultado.** Sobre las 21
ejecuciones del brazo `persistent` (7 partidos x 3 semillas), diferencia de spellF1 con el
peso de config.py contra peso 0:

- media >= +5 pp y al menos 15 de 21 mejoran: **se mantiene** la entrega `a43c52fb...`;
- media <= +1 pp, o menos de 11 de 21 mejoran: **se vuelve** a `e7938350...`
  (`team_classifier_weight = 0`);
- entre medias: se mantiene, y ESTADO.md cuenta la ganancia reducida.

Si `thinned` (la version facil) ya da "volver", eso basta para volver. Con esto no se
reajusta nada: ni el peso, ni los costes de cambio, ni las senales.

**Lo que sigue sin ver.** Que jugadores faltan: aqui se eligen al azar y sin mirar el
equipo; en el ATD probablemente faltan los lejanos a la camara o los amontonados, y puede
faltar mas de un equipo que de otro (se compara sin etiquetas el desequilibrio local -
visitante del ATD con el de aqui y se informa). En los frames con balon el IDSSE degradado
se queda en ~20,4 objetos frente a 21,4 del ATD (solo se puede quitar, no inventar). Los
huecos sin balon siguen al azar (la prueba de huecos colocados por actividad es otra,
`trabajo/sin-balon-1111/realista.py`). No es una cota del peor caso.

Uso:  python -m mtah.training.missing_players      -> out/missing_players.json
      (necesita MTAH_DATA_DIR para medir el ATD, requirements-train.txt y scikit-learn)
"""

import json
import os
import time

import numpy as np

from .. import features as feature_set
from ..config import DEFAULT
from ..constants import GRADED_BLOCKS
from ..features import geometry, team_signals
from ..features.boundaries import feature_matrix
from ..io import atd
from ..models import emissions
from ..models import team_classifier as shipped_classifier
from ..paths import REPO_ROOT, cache_dir
from . import boundaries, convention, degrade, harness, idsse
from . import team_classifier as team_training
from .boundaries import SEEDS, log

# Brazo -> sortear los que faltan una vez por racha sin balon (True) o en cada frame.
ARMS = {"thinned": False, "persistent": True}
DECISION_ARM = "persistent"

THINNED_KEYS = boundaries.PREPARED_KEYS + ("stack", "n_tracked", "n_home")

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


def _tracked(tracking):
    """(objetos por frame, objetos del local por frame)."""
    live = np.isfinite(tracking.players_xy[:, :, 0])
    home = live[:, tracking.team_of_slot == geometry.HOME_CODE]
    return live.sum(axis=1).astype(np.int16), home.sum(axis=1).astype(np.int16)


def prepare_thinned(match_id, spec, seed, config, persistent):
    """Lo de `boundaries.prepare` y `team_classifier.prepare` sobre el tracking adelgazado."""
    name = "thinned_persistent" if persistent else "thinned"
    path = os.path.join(cache_dir(), "{}_{}_s{}.npz".format(name, match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(THINNED_KEYS) <= set(blob.files):
            return {key: blob[key] for key in THINNED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed), gap_weights=None)
    thinned = degrade.thin_where_ball_unseen(dirty, spec, _thinning_rng(seed),
                                             persistent=persistent)
    features = feature_set.build_all(thinned, config)
    matrix, _ = feature_matrix(features, config)
    n_tracked, n_home = _tracked(thinned)
    data = {"cost": emissions.costs(features, config), "matrix": matrix, "labels": labels,
            "ball_seen": thinned.ball_seen,
            "home_logit": emissions.home_logit(features, config),
            "dead_logit": emissions.dead_logit(features, config),
            "stack": team_signals.smoothed(thinned),
            "n_tracked": n_tracked, "n_home": n_home}
    np.savez_compressed(path, **data)
    return data


def player_counts(n_tracked, n_home, ball_seen):
    """Objetos con y sin balon, frames sin balon con menos de 16, y desequilibrio entre equipos.

    `team_gap_unseen` es la media de |local - visitante| en objetos seguidos, sin balon.
    """
    unseen = ~ball_seen
    n_tracked, n_home = n_tracked.astype(np.int64), n_home.astype(np.int64)
    gap = np.abs(2 * n_home - n_tracked)[unseen]
    return {"mean_ball_seen": float(n_tracked[ball_seen].mean()),
            "mean_ball_unseen": float(n_tracked[unseen].mean()) if unseen.any() else None,
            "unseen_below_16": float((n_tracked[unseen] < FEW_PLAYERS).mean())
            if unseen.any() else None,
            "team_gap_unseen": float(gap.mean()) if unseen.any() else None}


def atd_player_counts():
    """Lo mismo medido en los frames puntuados del ATD del reto, sin etiquetas."""
    tracking = atd.load()
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, end in GRADED_BLOCKS:
        graded[start:end] = True
    n_tracked, n_home = _tracked(tracking)
    return player_counts(n_tracked[graded], n_home[graded], tracking.ball_seen[graded])


# --------------------------------------------------------------------------- evaluacion

def evaluate(base, team, arms, config):
    """Por partido de fuera y semilla: control y peso de config en cada degradacion."""
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
            sources = [("standard", base[seed][held_out], team[seed][held_out]["stack"])]
            sources += [(arm, arms[arm][seed][held_out], arms[arm][seed][held_out]["stack"])
                        for arm in ARMS]
            row = {}
            for name, data, stack in sources:
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
                if name in ARMS:
                    row[name]["players"] = player_counts(data["n_tracked"], data["n_home"],
                                                         data["ball_seen"])
            fold["seeds"][seed] = row
            log("%s s%d  " % (held_out, seed) + "  ".join(
                "%s %+.2f pp (AUC sin balon %.3f)" % (
                    name, row[name]["delta_spell_f1_pp"], row[name]["team"]["auc_ball_unseen"])
                for name in ("standard",) + tuple(ARMS)))
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


def final_verdict(summary):
    """La del brazo persistente, salvo que el facil ya diga volver."""
    if summary["thinned"]["verdict"] == "volver a peso 0":
        return "volver a peso 0"
    return summary[DECISION_ARM]["verdict"]


def summarise(folds):
    out = {}
    for name in ("standard",) + tuple(ARMS):
        deltas = [f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds for s in f["seeds"]]
        per_seed = {s: float(np.mean([f["seeds"][s][name]["delta_spell_f1_pp"] for f in folds]))
                    for s in SEEDS}
        out[name] = dict(decide(deltas), per_seed_mean_pp=per_seed,
                         min_pp=float(np.min(deltas)), max_pp=float(np.max(deltas)))
        log("%-10s %+.2f pp de media (%d de %d mejoran; de %+.2f a %+.2f)  por semilla %s" % (
            name, out[name]["mean_pp"], out[name]["better"], out[name]["runs"],
            out[name]["min_pp"], out[name]["max_pp"],
            "  ".join("%+.2f" % v for v in per_seed.values())))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    real_counts = atd_player_counts()
    log("ATD del reto: %s" % json.dumps(spec.summary()))
    log("ATD del reto, objetos: %s" % json.dumps(real_counts))
    base, team = team_training.load_matches(spec, SEEDS, config)
    arms = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    for arm, persistent in ARMS.items():
        for seed in SEEDS:
            for match_id in sorted(idsse.MATCHES):
                data = prepare_thinned(match_id, spec, seed, config, persistent)
                if not np.array_equal(data["ball_seen"], base[seed][match_id]["ball_seen"]):
                    raise ValueError("el brazo {0} de {1} s{2} no sale de la misma degradacion "
                                     "que boundary_{1}_s{2}.npz; borra las caches y vuelve a "
                                     "correr".format(arm, match_id, seed))
                arms[arm][seed][match_id] = data
    log("partidos preparados (%.0fs)" % (time.time() - started))

    folds = evaluate(base, team, arms, config)
    log("=" * 78)
    problems = check_reproduces_lomo(folds, config["team_classifier_weight"])
    for problem in problems:
        log("NO REPRODUCE: " + problem)
    summary = summarise(folds)
    simulated = {arm: player_counts(
        np.concatenate([arms[arm][s][m]["n_tracked"] for s in SEEDS for m in arms[arm][s]]),
        np.concatenate([arms[arm][s][m]["n_home"] for s in SEEDS for m in arms[arm][s]]),
        np.concatenate([arms[arm][s][m]["ball_seen"] for s in SEEDS for m in arms[arm][s]]))
        for arm in ARMS}
    for arm, counts in simulated.items():
        log("objetos en %-10s %s" % (arm, json.dumps(counts)))
    verdict = final_verdict(summary)
    log("reproduce la tabla del 12-11: %s" % ("si" if not problems else
                                              "NO (%d discrepancias)" % len(problems)))
    log("regla de decision (brazo %s): %s" % (DECISION_ARM, verdict.upper()))

    path = os.path.join(REPO_ROOT, "out", "missing_players.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"folds": folds, "summary": summary, "verdict": verdict,
                   "reproduces_lomo": not problems, "lomo_problems": problems,
                   "spec": spec.summary(), "atd_players": real_counts,
                   "simulated_players": simulated,
                   "settings": {"weight": config["team_classifier_weight"],
                                "decision_arm": DECISION_ARM,
                                "keep": [KEEP_MIN_PP, KEEP_MIN_BETTER],
                                "revert": [REVERT_MAX_PP, REVERT_MIN_BETTER]}},
                  handle, indent=2, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
