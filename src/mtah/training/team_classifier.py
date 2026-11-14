"""Clasificador de equipo sin balon, entrenado en el IDSSE y medido con LOMO anidado.

**Que intenta arreglar.** En los 7 partidos del IDSSE degradados como el ATD, el modelo
acierta el 87,5 % de los frames con balon y el 74,5 % de los que no lo tienen (el 41 % del
total), y de las posesiones reales que pasan casi enteras sin balon recupera el 15 %. Sin
balon las emisiones a mano solo arrastran la ultima senal vista. Este clasificador lee
quien tiene la posesion de la forma y el movimiento de los dos equipos
(`features/team_signals.py`) y suma su logit al de equipo de las emisiones.

**Como se mide, y por que asi.** No por frame: en spellF1, dentro del decodificador de la
entrega. Emisiones a mano mas `peso x logit`, con el detector de fronteras puntuando fuera
de muestra y el descuento y la escala de config.py. Leave-one-match-out *anidado* sobre
los 7 partidos: para cada partido de fuera, el clasificador se entrena con los otros 6 y
el peso se elige tambien sobre esos 6, cada uno puntuado por un clasificador que no vio ni
a ese partido ni al de fuera. El partido de fuera no influye ni en el modelo ni en el
peso. Las semillas de degradacion 1 y 2 solo comprueban que el resultado aguanta otro
patron de oclusion, con el modelo y el peso ya fijados en la semilla 0.

El detector de fronteras que acompana a cada partido se entreno con los otros 6, de modo
que el de un partido de dentro si vio al de fuera. No es lo que se elige aqui, y el
partido de fuera siempre se evalua con un detector que no lo vio.

Ninguna etiqueta del partido del reto entra aqui. El ATD solo aporta la distribucion de
oclusiones (`harness.real_spec`, sin etiquetas).

**Lo que esta medicion no ve.** El IDSSE degradado pierde el balon al azar y conserva a
los jugadores; el ATD lo pierde sobre todo con los jugadores parados y entonces le faltan
tambien jugadores (19,2 seguidos de media sin balon, 21,4 con el). Un clasificador de
dominio separa las senales del partido del reto de las del IDSSE con AUC 0,72 (0,53 entre
partidos del IDSSE), asi que la ganancia de aqui es una cota optimista de la del partido
del reto. Detalle en trabajo/sin-balon-1111/RESUMEN.md.

Uso:  python -m mtah.training.team_classifier           medicion -> out/team_classifier.json
      python -m mtah.training.team_classifier export    artefacto de la entrega
      (necesita requirements-train.txt + scikit-learn)
"""

import json
import os
import time

import numpy as np

from ..config import DEFAULT
from ..features import team_signals
from ..models import emissions, team_classifier
from ..paths import REPO_ROOT, cache_dir
from . import boundaries, convention, degrade, harness, idsse
from .boundaries import SEEDS, TRAIN_STRIDE_FRAMES, log

# Rejilla fija del peso sobre el logit del clasificador; 0 es la entrega sin el. Es la del
# experimento del 11-11 y no se amplia por arriba aunque gane el borde: el peso manda mas
# cuanto menos se parezca el partido del reto al IDSSE, y eso aqui no se puede medir.
WEIGHTS = (0.0, 0.5, 1.0, 2.0)

# Los mismos del detector de fronteras; no se han buscado.
MODEL_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                "min_samples_leaf": 200, "l2_regularization": 1.0, "early_stopping": False,
                "random_state": 0}

PREPARED_KEYS = ("stack", "labels", "ball_seen")


# --------------------------------------------------------------------------- datos

def _settings():
    return json.dumps(team_classifier.feature_settings(), sort_keys=True)


def prepare(match_id, spec, seed):
    """{senales suavizadas, etiquetas, balon visto} de un partido degradado, cacheado.

    La misma degradacion que `boundaries.prepare` (misma semilla, huecos al azar): los
    logits de las emisiones y estas senales tienen que salir del mismo tracking.
    """
    path = os.path.join(cache_dir(), "team_{}_s{}.npz".format(match_id, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(PREPARED_KEYS) <= set(blob.files) and str(blob["settings"]) == _settings():
            return {key: blob[key] for key in PREPARED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed), gap_weights=None)
    data = {"stack": team_signals.smoothed(dirty),
            "labels": np.asarray(labels, dtype="<U1"), "ball_seen": dirty.ball_seen}
    np.savez_compressed(path, settings=np.array(_settings()), **data)
    return data


def load_matches(spec, seeds, config):
    """(base, team): `boundaries.prepare` y `prepare` de cada semilla y partido."""
    base = {seed: {} for seed in seeds}
    team = {seed: {} for seed in seeds}
    for seed in seeds:
        for match_id in sorted(idsse.MATCHES):
            base[seed][match_id] = boundaries.prepare(match_id, spec, seed, config)
            team[seed][match_id] = prepare(match_id, spec, seed)
            if not np.array_equal(base[seed][match_id]["ball_seen"],
                                  team[seed][match_id]["ball_seen"]):
                raise ValueError("out/cache/boundary_{0}_s{1}.npz y team_{0}_s{1}.npz no salen "
                                 "de la misma degradacion; borralas y vuelve a preparar"
                                 .format(match_id, seed))
    return base, team


# --------------------------------------------------------------------------- modelo

def fit(match_data):
    """Clasificador entrenado con las dos perspectivas de los frames vivos de esos partidos."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows, targets = [], []
    for data in match_data:
        labels = data["labels"][::TRAIN_STRIDE_FRAMES]
        alive = labels != "D"
        for perspective, mine in ((0, "H"), (1, "A")):
            rows.append(team_signals.matrix(data["stack"], perspective)
                        [::TRAIN_STRIDE_FRAMES][alive])
            targets.append(labels[alive] == mine)
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(np.concatenate(rows), np.concatenate(targets))
    return model


def out_of_fold_detector(base):
    """{semilla: {partido: score de frontera}} con el detector de los otros 6 (semilla 0)."""
    matches = sorted(base[SEEDS[0]])
    scores = {seed: {} for seed in base}
    for match_id in matches:
        model = boundaries.fit([base[SEEDS[0]][j] for j in matches if j != match_id])
        for seed in base:
            scores[seed][match_id] = boundaries.boundary_score(model, base[seed][match_id])
    return scores


def team_quality(logit_by_match, team_by_match):
    """AUC del logit contra el equipo real en los frames vivos; aparte, en los que no hay balon."""
    from sklearn.metrics import roc_auc_score

    logit, truth, unseen = [], [], []
    for match_id, data in team_by_match.items():
        alive = data["labels"] != "D"
        logit.append(logit_by_match[match_id][alive])
        truth.append(data["labels"][alive] == "H")
        unseen.append(~data["ball_seen"][alive])
    logit, truth, unseen = (np.concatenate(v) for v in (logit, truth, unseen))
    return {"auc": float(roc_auc_score(truth, logit)),
            "auc_ball_unseen": float(roc_auc_score(truth[unseen], logit[unseen]))}


# --------------------------------------------------------------------------- decodificacion

def decode_score(base, logit, detector_score, config, weight):
    """spellF1 y macroF1 de un partido en el decodificador de la entrega, con ese peso.

    El logit se suma fuera de `home_logit`, como en `emissions.costs`: el detector de
    fronteras no lo ve. Con peso 0 es exactamente la medicion del detector.
    """
    data = base
    if weight:
        home = base["home_logit"].astype(np.float64) + weight * logit.astype(np.float64)
        data = dict(base, cost=emissions.costs_from_logits(
            home, base["dead_logit"].astype(np.float64), config))
    labels = boundaries.decode(data, config, config["boundary_discount_weight"],
                               config["boundary_switch_scale"], detector_score)
    return boundaries.score_labels(labels, base)


def choose_weight(base_by_match, logit_by_match, detector_by_match, config, weights=WEIGHTS):
    """Peso de `weights` con mejor spellF1 medio sobre esos partidos. Desempate macroF1."""
    results = []
    for weight in weights:
        scores = [decode_score(base_by_match[m], logit_by_match[m], detector_by_match[m],
                               config, weight) for m in base_by_match]
        results.append((round(float(np.mean([s["spell_f1"] for s in scores])), 6),
                        round(float(np.mean([s["macro_f1"] for s in scores])), 6), weight))
    best = max(results, key=lambda r: (r[0], r[1]))
    return {"weight": best[2], "inner_spell_f1": best[0], "inner_macro_f1": best[1],
            "grid": [list(r) for r in results]}


# --------------------------------------------------------------------------- evaluacion

def nested_leave_one_match_out(base, team, detector, config):
    """Ver el docstring del modulo. `base[seed][match]`, `team[seed][match]` preparados."""
    matches = sorted(team[SEEDS[0]])
    models = {}

    def model_for(train_ids):
        key = frozenset(train_ids)
        if key not in models:
            models[key] = fit([team[SEEDS[0]][m] for m in sorted(key)])
        return models[key]

    folds = []
    for held_out in matches:
        started = time.time()
        inner = [m for m in matches if m != held_out]
        inner_logit = {m: team_classifier.antisymmetric_logit(
            model_for([j for j in inner if j != m]), team[SEEDS[0]][m]["stack"]) for m in inner}
        choice = choose_weight({m: base[SEEDS[0]][m] for m in inner}, inner_logit,
                               {m: detector[SEEDS[0]][m] for m in inner}, config)
        outer_model = model_for(inner)
        fold = {"held_out": held_out, "choice": choice, "seeds": {}}
        log("== fuera: %s   peso elegido en los 6 de dentro: %.1f (dentro %.4f)   %.0fs"
            % (held_out, choice["weight"], choice["inner_spell_f1"], time.time() - started))
        for seed in SEEDS:
            logit = team_classifier.antisymmetric_logit(outer_model, team[seed][held_out]["stack"])
            by_weight = {"%g" % weight: decode_score(base[seed][held_out], logit,
                                                     detector[seed][held_out], config, weight)
                         for weight in WEIGHTS}
            row = {"control": by_weight["0"], "chosen": by_weight["%g" % choice["weight"]],
                   "by_weight": by_weight,
                   "team": team_quality({held_out: logit}, {held_out: team[seed][held_out]})}
            fold["seeds"][seed] = row
            log("   semilla %d  control %.4f  elegido %+.2f  " % (
                seed, row["control"]["spell_f1"],
                100 * (row["chosen"]["spell_f1"] - row["control"]["spell_f1"]))
                + "  ".join("peso %s %+.2f" % (key, 100 * (value["spell_f1"]
                                                           - row["control"]["spell_f1"]))
                            for key, value in by_weight.items() if key != "0")
                + "  AUC %.3f (sin balon %.3f)" % (row["team"]["auc"],
                                                   row["team"]["auc_ball_unseen"]))
        folds.append(fold)
    return folds


def _paired(folds, seed, pick):
    """Diferencia de spellF1 y macroF1 contra el control, partido a partido."""
    spell = np.array([pick(f["seeds"][seed])["spell_f1"] - f["seeds"][seed]["control"]["spell_f1"]
                      for f in folds])
    macro = np.array([pick(f["seeds"][seed])["macro_f1"] - f["seeds"][seed]["control"]["macro_f1"]
                      for f in folds])
    return {"delta_spell_f1_pp": 100 * float(spell.mean()),
            "matches_better": int((spell > 0).sum()), "matches_worse": int((spell < 0).sum()),
            "delta_macro_f1_pp": 100 * float(macro.mean())}


def summarise(folds):
    """El peso elegido anidado contra el control y, como referencia, cada peso fijo."""
    out = {}
    arms = [("chosen", lambda row: row["chosen"])]
    arms += [("weight_%g" % w, lambda row, key="%g" % w: row["by_weight"][key])
             for w in WEIGHTS if w]
    for seed in SEEDS:
        out[seed] = {"control_spell_f1": float(np.mean(
            [f["seeds"][seed]["control"]["spell_f1"] for f in folds]))}
        for name, pick in arms:
            out[seed][name] = _paired(folds, seed, pick)
            log("semilla %d  control %.4f  %-10s %+.2f pp spellF1 (%d mejor / %d peor)  "
                "%+.2f pp macroF1" % (seed, out[seed]["control_spell_f1"], name,
                                      out[seed][name]["delta_spell_f1_pp"],
                                      out[seed][name]["matches_better"],
                                      out[seed][name]["matches_worse"],
                                      out[seed][name]["delta_macro_f1_pp"]))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    base, team = load_matches(harness.real_spec(), SEEDS, config)
    log("partidos preparados (%.0fs)" % (time.time() - started))
    detector = out_of_fold_detector(base)
    log("detector de fronteras fuera de muestra (%.0fs)" % (time.time() - started))

    folds = nested_leave_one_match_out(base, team, detector, config)
    log("=" * 78)
    summary = summarise(folds)
    path = os.path.join(REPO_ROOT, "out", "team_classifier.json")
    with open(path, "w") as handle:
        json.dump({"folds": folds, "summary": summary,
                   "settings": {"weights": WEIGHTS, "stride": TRAIN_STRIDE_FRAMES,
                                "model": MODEL_PARAMS,
                                "features": team_classifier.feature_settings()}},
                  handle, indent=2, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))


def export(config=None):
    """Entrena el clasificador de la entrega con los 7 partidos y elige su peso.

    El mismo procedimiento que se midio, sin partido de fuera: el peso se elige sobre los 7
    con logits fuera de muestra (cada partido puntuado por un clasificador entrenado con
    los otros 6), y el modelo final se entrena con los 7. Semilla 0, como en la medicion.
    Escribe models/team_classifier.{joblib,json}; el peso elegido se transcribe a config.py
    (`team_classifier_weight`) con su procedencia.
    """
    config = config or DEFAULT
    started = time.time()
    seed = SEEDS[0]
    base, team = load_matches(harness.real_spec(), (seed,), config)
    base, team = base[seed], team[seed]
    matches = sorted(team)
    detector = out_of_fold_detector({seed: base})[seed]
    out_of_fold = {m: team_classifier.antisymmetric_logit(
        fit([team[j] for j in matches if j != m]), team[m]["stack"]) for m in matches}
    choice = choose_weight(base, out_of_fold, detector, config)
    control = next(row for row in choice["grid"] if row[2] == 0)
    quality = team_quality(out_of_fold, team)
    model = fit([team[m] for m in matches])
    metadata = team_classifier.save(model, {
        "trained_on": matches, "degradation_seed": seed,
        "train_stride_frames": TRAIN_STRIDE_FRAMES,
        "model_params": MODEL_PARAMS,
        "feature_names": team_signals.feature_names(),
        "feature_settings": team_classifier.feature_settings(),
        "chosen_weight": choice["weight"],
        "out_of_fold_spell_f1": choice["inner_spell_f1"],
        "out_of_fold_macro_f1": choice["inner_macro_f1"],
        "control_spell_f1": control[0], "control_macro_f1": control[1],
        "out_of_fold_auc": quality["auc"],
        "out_of_fold_auc_ball_unseen": quality["auc_ball_unseen"],
        "grid": choice["grid"],
    })
    log("elegido sobre los 7 (fuera de muestra): peso %.1f -> spellF1 %.4f (sin clasificador "
        "%.4f), AUC de equipo %.4f" % (choice["weight"], choice["inner_spell_f1"], control[0],
                                       quality["auc"]))
    log("guardado %s (sha256 %s..., %.0f min)" % (
        team_classifier.ARTIFACT_PATH, metadata["artifact_sha256"][:12],
        (time.time() - started) / 60))
    return metadata


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
