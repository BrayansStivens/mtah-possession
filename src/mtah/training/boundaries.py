"""Detector de fronteras de posesion, entrenado en el IDSSE y medido con LOMO anidado.

**Que intenta arreglar.** El error del modelo se concentra junto a los cambios: a +-12
frames de una transicion real falla uno de cada dos frames, y las posesiones de menos de
5 s se recuperan un ~8 %. Las emisiones estan suavizadas a +-50 frames, asi que ven *que*
equipo domina un tramo pero no *donde* empieza. Este detector mira ventanas cortas a cada
lado de cada frame (que cambia entre antes y despues) y devuelve un
`transition_boundary_score` en [0, 1]. El decodificador lo usa como descuento sobre el
coste de cambiar de estado (`viterbi.decode(..., switch_discount=...)`): donde el detector
ve una frontera, cortar sale mas barato.

Plan de origen: `plan-fronteras-segmentos.md` (Dia 1 y Dia 2). El nombre es "score" y no
"probabilidad" porque nadie ha validado su calibracion, y el decodificador no lo trata
como una.

**Como se mide, y por que asi.** Leave-one-match-out *anidado* sobre los 7 partidos:
para cada partido de fuera, el detector se entrena con los otros 6, y el peso del
descuento se elige tambien sobre esos 6 con puntuaciones fuera de muestra (cada uno
puntuado por un detector que no lo vio). El partido de fuera no influye ni en el modelo
ni en el peso. Las tres semillas de degradacion de un mismo partido no son partidos
independientes: la 1 y la 2 solo se usan para ver si el resultado aguanta otro patron de
oclusion, con el modelo y el peso ya fijados en la semilla 0.

Ninguna etiqueta del partido del reto entra aqui. El ATD solo aporta la distribucion de
oclusiones (`harness.real_spec`, sin etiquetas).

Uso:  python -m mtah.training.boundaries      (necesita requirements-train.txt + scikit-learn)
"""

import itertools
import json
import os
import time

import numpy as np

from .. import evaluate as ev
from .. import features as feature_set
from ..config import DEFAULT
from ..decode import viterbi
from ..features import geometry
from ..features.boundaries import WINDOWS_FRAMES, feature_matrix, feature_names
from ..models import boundary_detector, emissions
from ..paths import REPO_ROOT, cache_dir
from . import convention, degrade, harness, idsse

# Un frame es positivo si cae a menos de esto de un cambio de etiqueta. 12 frames (0,48 s)
# es el radio en el que el diagnostico de errores mide la mitad de los fallos.
BOUNDARY_HALF_WIDTH_FRAMES = 12

# Uno de cada N frames para entrenar. Los vecinos son casi identicos: un millon de frames
# correlacionados no son un millon de ejemplos, y asi cada ajuste tarda segundos.
TRAIN_STRIDE_FRAMES = 4

# Rejilla del decodificador, elegida dentro de cada pliegue. `switch_scale` multiplica los
# tres costes de cambio de config.py: con el detector abaratando las fronteras, puede
# convenir encarecer el resto. (1.0, 0.0) es el modelo actual tal cual.
DISCOUNT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 0.95)
SWITCH_SCALES = (1.0, 1.25, 1.5, 2.0)

# Tres brazos con la misma eleccion anidada, para saber de donde sale la mejora. Encarecer
# los cambios ya mejora algo sin detector, y sin este control esa parte se le atribuiria.
ARMS = {
    "scale_only": [(0.0, scale) for scale in SWITCH_SCALES],
    "discount_only": [(weight, 1.0) for weight in DISCOUNT_WEIGHTS],
    "both": list(itertools.product(DISCOUNT_WEIGHTS, SWITCH_SCALES)),
}

SEEDS = (0, 1, 2)

PREPARED_KEYS = ("cost", "matrix", "labels", "ball_seen", "home_logit", "dead_logit")

# Donde caen los huecos sin balon al degradar. Se entrena y se elige siempre con `random`;
# las otras dos son pruebas de estres para el partido de fuera (training/evidence.py).
GAP_PLACEMENTS = ("random", "crowded", "transitions")
CROWD_RADIUS_M = 5.0
TRANSITION_RADIUS_FRAMES = 12
GAP_WEIGHT_FLOOR = 0.02


def log(message):
    print(message, flush=True)


# --------------------------------------------------------------------------- objetivo

def boundary_target(labels, half_width=BOUNDARY_HALF_WIDTH_FRAMES):
    """(n,) bool: True a menos de `half_width` frames de un cambio de etiqueta."""
    labels = np.asarray(labels)
    change_frames = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    near = np.zeros(len(labels) + 1, dtype=np.int32)
    np.add.at(near, np.clip(change_frames - half_width, 0, len(labels)), 1)
    np.add.at(near, np.clip(change_frames + half_width, 0, len(labels)), -1)
    return np.cumsum(near)[:-1] > 0


# --------------------------------------------------------------------------- datos

def gap_weights(placement, clean, labels):
    """Peso por frame para centrar ahi un hueco sin balon. None = al azar, el de siempre.

    `crowded` no mira etiquetas: pesa el numero de jugadores a menos de CROWD_RADIUS_M del
    balon (al cuadrado), que es como describe DATA.md donde pierde el balon el ATD. Peor
    caso, `transitions`: centra los huecos junto a los cambios de posesion reales.
    """
    if placement == "random":
        return None
    if placement == "crowded":
        with np.errstate(invalid="ignore"):
            crowd = (geometry.ball_distances(clean) < CROWD_RADIUS_M).sum(axis=1)
        return GAP_WEIGHT_FLOOR + crowd.astype(np.float64) ** 2
    if placement == "transitions":
        return np.where(boundary_target(labels, TRANSITION_RADIUS_FRAMES), 1.0,
                        GAP_WEIGHT_FLOOR)
    raise ValueError("colocacion desconocida {!r}; hay {}".format(placement, GAP_PLACEMENTS))


def prepare(match_id, spec, seed, config=None, placement="random"):
    """{coste, matriz, etiquetas, balon visto, logits de la emision} de un partido degradado, cacheado."""
    config = config or DEFAULT
    suffix = "" if placement == "random" else "_" + placement
    path = os.path.join(cache_dir(), "boundary_{}{}_s{}.npz".format(match_id, suffix, seed))
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=False)
        if set(PREPARED_KEYS) <= set(blob.files):
            return {key: blob[key] for key in PREPARED_KEYS}
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    labels = np.asarray(labels, dtype="<U1")
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed),
                          gap_weights=gap_weights(placement, clean, labels))
    features = feature_set.build_all(dirty, config)
    matrix, _ = feature_matrix(features, config)
    data = {"cost": emissions.costs(features, config), "matrix": matrix, "labels": labels,
            "ball_seen": dirty.ball_seen,
            "home_logit": emissions.home_logit(features, config),
            "dead_logit": emissions.dead_logit(features, config)}
    np.savez_compressed(path, **data)
    return data


# --------------------------------------------------------------------------- modelo

def fit(match_data):
    """Clasificador de frontera entrenado sobre una lista de partidos preparados."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = [d["matrix"][::TRAIN_STRIDE_FRAMES] for d in match_data]
    targets = [boundary_target(d["labels"])[::TRAIN_STRIDE_FRAMES] for d in match_data]
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    model.fit(np.concatenate(rows), np.concatenate(targets))
    return model


def boundary_score(model, data):
    return model.predict_proba(data["matrix"])[:, 1]


def decode(data, config, discount_weight=0.0, switch_scale=1.0, score=None):
    """Etiquetas del pipeline actual, con o sin la guia del detector."""
    blocks = ((0, len(data["cost"])),)
    transition = switch_scale * viterbi.switch_costs(
        config["switch_turnover"], config["switch_to_dead"], config["switch_from_dead"])
    discount = None if score is None or discount_weight == 0 else discount_weight * score
    codes = viterbi.decode(data["cost"], transition, blocks=blocks, switch_discount=discount)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"], blocks=blocks)
    return viterbi.to_labels(codes)


def score_labels(predicted, data):
    gt_codes = ev.pm.label_codes(data["labels"])
    result = ev.score(predicted, gt_codes=gt_codes)
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"]}


def detector_quality(score, data):
    """AUC y precision media del score contra el objetivo, mas el acierto de fronteras."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    target = boundary_target(data["labels"])
    return {"auc": float(roc_auc_score(target, score)),
            "average_precision": float(average_precision_score(target, score)),
            "positive_rate": float(target.mean())}


# --------------------------------------------------------------------------- evaluacion

def choose_decoder(data_by_match, score_by_match, config, grid):
    """(peso, escala) de `grid` con mejor spellF1 medio sobre esos partidos. Desempate macroF1."""
    results = []
    for weight, scale in grid:
        scores = [score_labels(decode(data_by_match[m], config, weight, scale,
                                      score_by_match[m]), data_by_match[m])
                  for m in data_by_match]
        results.append((round(float(np.mean([s["spell_f1"] for s in scores])), 6),
                        round(float(np.mean([s["macro_f1"] for s in scores])), 6),
                        weight, scale))
    best = max(results, key=lambda r: (r[0], r[1]))
    return {"discount_weight": best[2], "switch_scale": best[3],
            "inner_spell_f1": best[0], "inner_macro_f1": best[1],
            "grid": [list(r) for r in results]}


def nested_leave_one_match_out(data, config):
    """Ver el docstring del modulo. `data[seed][match]` preparado."""
    matches = sorted(data[SEEDS[0]])
    models = {}

    def model_for(train_ids):
        key = frozenset(train_ids)
        if key not in models:
            started = time.time()
            models[key] = fit([data[SEEDS[0]][m] for m in sorted(key)])
            log("   detector entrenado con %s (%.0fs)" % (",".join(sorted(key)),
                                                         time.time() - started))
        return models[key]

    folds = []
    for held_out in matches:
        log("== fuera: %s" % held_out)
        inner = [m for m in matches if m != held_out]
        inner_scores = {m: boundary_score(model_for([j for j in inner if j != m]),
                                          data[SEEDS[0]][m]) for m in inner}
        inner_data = {m: data[SEEDS[0]][m] for m in inner}
        choices = {arm: choose_decoder(inner_data, inner_scores, config, grid)
                   for arm, grid in ARMS.items()}
        for arm, choice in choices.items():
            log("   %-13s elegido en los 6 de dentro: peso %.2f escala %.2f (dentro %.4f)"
                % (arm, choice["discount_weight"], choice["switch_scale"],
                   choice["inner_spell_f1"]))

        outer_model = model_for(inner)
        fold = {"held_out": held_out, "choices": choices, "seeds": {}}
        for seed in SEEDS:
            held = data[seed][held_out]
            score = boundary_score(outer_model, held)
            row = {"control": score_labels(decode(held, config), held),
                   "detector": detector_quality(score, held)}
            for arm, choice in choices.items():
                row[arm] = score_labels(decode(held, config, choice["discount_weight"],
                                               choice["switch_scale"], score), held)
            fold["seeds"][seed] = row
            log("   semilla %d  control %.4f  " % (seed, row["control"]["spell_f1"])
                + "  ".join("%s %+.2f" % (arm, 100 * (row[arm]["spell_f1"]
                                                      - row["control"]["spell_f1"]))
                            for arm in ARMS)
                + "  AUC %.3f" % row["detector"]["auc"])
        folds.append(fold)
    return folds


def _paired(folds, seed, arm, reference):
    """Diferencia de spellF1 y macroF1, partido a partido, entre dos brazos."""
    spell = np.array([f["seeds"][seed][arm]["spell_f1"] - f["seeds"][seed][reference]["spell_f1"]
                      for f in folds])
    macro = np.array([f["seeds"][seed][arm]["macro_f1"] - f["seeds"][seed][reference]["macro_f1"]
                      for f in folds])
    return {"delta_spell_f1_pp": 100 * float(spell.mean()),
            "matches_better": int((spell > 0).sum()), "matches_worse": int((spell < 0).sum()),
            "delta_macro_f1_pp": 100 * float(macro.mean())}


def summarise(folds):
    """Cada brazo contra el control y, lo que decide, `both` contra `scale_only`."""
    out = {}
    comparisons = [(arm, "control") for arm in ARMS] + [("both", "scale_only")]
    for seed in SEEDS:
        out[seed] = {"control_spell_f1": float(np.mean(
            [f["seeds"][seed]["control"]["spell_f1"] for f in folds]))}
        for arm, reference in comparisons:
            key = "%s_vs_%s" % (arm, reference)
            out[seed][key] = _paired(folds, seed, arm, reference)
            log("semilla %d  %-24s %+.2f pp spellF1 (%d mejor / %d peor)  %+.2f pp macroF1"
                % (seed, key, out[seed][key]["delta_spell_f1_pp"],
                   out[seed][key]["matches_better"], out[seed][key]["matches_worse"],
                   out[seed][key]["delta_macro_f1_pp"]))
    return out


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    data = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for match_id in sorted(idsse.MATCHES):
            data[seed][match_id] = prepare(match_id, spec, seed, config)
        log("semilla %d preparada (%.0fs)" % (seed, time.time() - started))

    folds = nested_leave_one_match_out(data, config)
    log("=" * 78)
    summary = summarise(folds)
    path = os.path.join(REPO_ROOT, "out", "boundaries.json")
    with open(path, "w") as handle:
        json.dump({"folds": folds, "summary": summary,
                   "settings": {"half_width": BOUNDARY_HALF_WIDTH_FRAMES,
                                "windows": WINDOWS_FRAMES, "stride": TRAIN_STRIDE_FRAMES,
                                "arms": ARMS}},
                  handle, indent=2, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))


def export(config=None):
    """Entrena el detector de la entrega con los 7 partidos y elige peso y escala.

    El mismo procedimiento que se midio, sin partido de fuera: peso y escala se eligen
    sobre los 7 con puntuaciones fuera de muestra (cada partido puntuado por un detector
    entrenado con los otros 6), y el modelo final se entrena con los 7. Semilla 0, como en
    la medicion. Escribe models/boundary_detector.{joblib,json}; los dos valores elegidos
    se transcriben a config.py con su procedencia, y `score()` comprueba al cargar que las
    features se calculan con la misma config que aqui.
    """
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    matches = sorted(idsse.MATCHES)
    data = {m: prepare(m, spec, SEEDS[0], config) for m in matches}
    out_of_fold = {m: boundary_score(fit([data[j] for j in matches if j != m]), data[m])
                   for m in matches}
    choice = choose_decoder(data, out_of_fold, config, ARMS["both"])
    model = fit([data[m] for m in matches])
    names = feature_names()
    if {d["matrix"].shape[1] for d in data.values()} != {len(names)}:
        raise ValueError("la cache out/cache/boundary_*.npz es de otra version de las "
                         "features; borrala y vuelve a preparar")
    metadata = boundary_detector.save(model, {
        "trained_on": matches, "degradation_seed": SEEDS[0],
        "half_width_frames": BOUNDARY_HALF_WIDTH_FRAMES,
        "train_stride_frames": TRAIN_STRIDE_FRAMES,
        "feature_names": names,
        "feature_config": boundary_detector.feature_config(config),
        "chosen_discount_weight": choice["discount_weight"],
        "chosen_switch_scale": choice["switch_scale"],
        "out_of_fold_spell_f1": choice["inner_spell_f1"],
        "out_of_fold_macro_f1": choice["inner_macro_f1"],
        "grid": choice["grid"],
    })
    log("elegido sobre los 7 (fuera de muestra): peso %.2f escala %.2f -> spellF1 %.4f"
        % (choice["discount_weight"], choice["switch_scale"], choice["inner_spell_f1"]))
    log("guardado %s (sha256 %s..., %.0f min)" % (
        boundary_detector.ARTIFACT_PATH, metadata["artifact_sha256"][:12],
        (time.time() - started) / 60))
    return metadata


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
