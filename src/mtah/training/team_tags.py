"""Peso del Smart Tagging como evidencia de "que equipo tiene el balon", medido en el IDSSE.

**Por que.** El 60 % de los frames que fallamos en el IDSSE (71 % con el balon visible) son
local por visitante, no vivo por muerto. El Smart Tagging de BAR-ATH dice de quien es el
balon en fases de ~18 s que cubren el 40 % de los frames puntuados, y el pipeline no lo
usaba. Entra como un sumando del logit de local (`models/emissions.team_tag_logit`).

**El problema: el IDSSE no tiene Smart Tagging**, asi que su peso no se puede medir
directamente. Se simula un tagging sobre las etiquetas del IDSSE que se parezca al real en
lo que se puede medir del real **sin etiquetas**: tramos disjuntos con las longitudes de los
tramos reales, la misma cobertura, centrados en frames vivos, separados entre si (en
BAR-ATH nunca hay un cambio directo de equipo dentro del tagging), y con el equipo
mayoritario del tramo (un tramo que cruza un cambio arrastra error, como una fase real).

**Lo que no se sabe es cuanto acierta el tagging real.** Lo unico medible sin etiquetas es
cuanto coincide con nuestra emision: 0,762 en BAR-ATH. Y un tagging simulado *perfecto*
coincide con nuestra emision en el IDSSE solo 0,706. El real coincide mas que uno perfecto,
asi que el "acierta X" no se puede calibrar con ruido independiente. Hay tres explicaciones,
y cada una es un escenario con su parametro calibrado para dar 0,762:

- `independent`: errores independientes de los nuestros (`flip`: probabilidad de invertir
  un tramo). Si ni con flip 0 se llega, es la cota optimista: tags perfectos.
- `echo`: Metrica etiqueta con el mismo tracking que nosotros, asi que sus errores
  coinciden con los nuestros. Con probabilidad `echo` el tramo lleva el equipo que dice
  *nuestra* emision en vez del verdadero. El escenario pesimista: lo que repite lo que ya
  sabemos no aporta.
- `easy`: las fases se detectan donde el juego es claro, que es donde nuestra emision ya
  acierta. Los tramos se centran con peso |logit de local|^k.

El peso se elige con el mismo LOMO anidado que el detector de fronteras, encima del
pipeline entregado (detector incluido, con el peso y escala de cada pliegue), en cada
escenario por separado. Se entrega el peso del escenario que menos gana, y solo si ahi
sigue ganando.

Uso:  python -m mtah.training.team_tags                    (los tres escenarios; ~20 min)
      python -m mtah.training.team_tags robust             (ajuste por cuartiles + pesos fijos; ~15 min)
      python -m mtah.training.team_tags export <escenario> (peso para config.py)
"""

import json
import os
import sys
import time

import numpy as np

from .. import features as feature_set
from ..config import DEFAULT
from ..constants import graded_mask
from ..features import tagging
from ..models import emissions
from . import boundaries, degrade, evidence, harness, idsse

# Cada escenario: (parametro que se calibra, su rejilla). Los otros dos quedan neutros.
SCENARIOS = {
    "independent": ("flip", tuple(round(0.025 * k, 3) for k in range(13))),     # 0 .. 0,30
    "echo": ("echo", tuple(round(0.05 * k, 2) for k in range(21))),             # 0 .. 1
    "easy": ("exponent", (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)),
}
NEUTRAL = {"flip": 0.0, "echo": 0.0, "exponent": 0.0}
# Con los pesos elegidos en `independent`, hasta que ruido independiente sigue ganando.
BREAK_EVEN_FLIPS = (0.10, 0.20, 0.30)
TAG_WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TAG_SEED_OFFSET = 1000
MAX_PLACEMENT_ATTEMPTS = 200_000
WEIGHT_FLOOR = 1e-6
N_CONFIDENCE_BANDS = 4


def log(message):
    print(message, flush=True)


# --------------------------------------------------------------------------- BAR-ATH

def agreement(home_logit, tags, mask):
    """(acuerdo, acuerdo por cuartil de |logit|) entre la emision y el tagging en `mask`."""
    mask = mask & (tags != 0)
    if not mask.any():
        raise ValueError("ningun frame con tag en la mascara")
    correct = (home_logit[mask] > 0) == (tags[mask] > 0)
    confidence = np.abs(home_logit[mask])
    edges = np.quantile(confidence, np.linspace(0, 1, N_CONFIDENCE_BANDS + 1))
    band = np.clip(np.searchsorted(edges, confidence, side="right") - 1,
                   0, N_CONFIDENCE_BANDS - 1)
    return float(correct.mean()), [float(correct[band == q].mean())
                                   for q in range(N_CONFIDENCE_BANDS)]


def real_profile(config=None):
    """Lo que se mide del tagging real de BAR-ATH sin ninguna etiqueta."""
    from ..io import atd, smart_tagging

    config = config or DEFAULT
    tracking = atd.load()
    n = tracking.n_frames
    tags = tagging.team_tags(smart_tagging.load(), n)
    graded = graded_mask(n)
    home = emissions.home_logit(feature_set.build_all(tracking, config), config)
    overall, by_quartile = agreement(home, tags, graded)
    return {"run_lengths": degrade._runs(tags[graded] != 0),
            "coverage": float((tags[graded] != 0).mean()),
            "agreement": overall, "agreement_by_quartile": by_quartile}


# --------------------------------------------------------------------------- simulacion

def _majority(values_are_home):
    return 1 if values_are_home.sum() * 2 >= len(values_are_home) else -1


def simulate_windows(labels, home_logit, profile, rng, exponent=0.0):
    """[(inicio, fin, equipo verdadero, equipo de nuestra emision)] con +-1 por equipo.

    `exponent` > 0 centra los tramos con peso |home_logit|^exponent: donde nuestra emision
    esta segura (escenario `easy`).
    """
    labels = np.asarray(labels)
    n = len(labels)
    live = np.flatnonzero(labels != "D")
    # Suelo para que ningun frame vivo quede con peso exactamente cero.
    weights = (np.abs(home_logit[live]) ** exponent if exponent
               else np.ones(len(live))) + WEIGHT_FLOOR
    # Los centros se sortean solo entre frames libres: si no, con el peso concentrado los
    # sorteos caen una y otra vez en tramos ya puestos y nunca se llega a la cobertura.
    cumulative = np.cumsum(weights)
    taken = np.zeros(n, dtype=bool)
    target = int(round(profile["coverage"] * n))
    windows, covered = [], 0
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        if covered >= target:
            return windows
        length = int(rng.choice(profile["run_lengths"]))
        draw = rng.random() * cumulative[-1]
        center = int(live[min(np.searchsorted(cumulative, draw, side="right"), len(live) - 1)])
        start = max(0, min(center - length // 2, n - length))
        stop = start + length
        if taken[max(0, start - 1):stop + 1].any():     # disjuntos y con hueco
            continue
        inside = labels[start:stop]
        alive = inside != "D"
        if not alive.any():
            continue
        windows.append((start, stop, _majority(inside[alive] == "H"),
                        _majority(home_logit[start:stop] > 0)))
        taken[start:stop] = True
        covered += length
        cumulative = np.cumsum(np.where(taken[live], 0.0, weights))
    raise RuntimeError("no se llego a cubrir el %.0f %% de los frames en %d intentos"
                       % (100 * profile["coverage"], MAX_PLACEMENT_ATTEMPTS))


def tags_from_windows(windows, draws, n, flip=0.0, echo=0.0):
    """(n,) int8. Por tramo: equipo de nuestra emision si su sorteo de eco < `echo`, si no
    el verdadero; despues invertido si su sorteo de flip < `flip`.

    Los sorteos son fijos por tramo, asi que subir `flip` o `echo` solo anade cambios: los
    niveles son anidados y se comparan sobre los mismos tramos.
    """
    tags = np.zeros(n, dtype=np.int8)
    for (start, stop, truth, ours), flip_draw, echo_draw in zip(windows, draws["flip"],
                                                                draws["echo"]):
        team = ours if echo_draw < echo else truth
        tags[start:stop] = -team if flip_draw < flip else team
    return tags


class Simulation:
    """Tramos y sorteos de un escenario para todos los (semilla, partido), reproducibles."""

    def __init__(self, data, profile, exponent):
        self.data = data
        self._cache = {}
        for seed in data:
            for index, m in enumerate(sorted(data[seed])):
                rng = np.random.default_rng(TAG_SEED_OFFSET + 10 * seed + index)
                windows = simulate_windows(data[seed][m]["labels"], data[seed][m]["home_logit"],
                                           profile, rng, exponent)
                draws = {"flip": rng.random(len(windows)), "echo": rng.random(len(windows))}
                self._cache[seed, m] = (windows, draws)

    def tags(self, seed, m, flip=0.0, echo=0.0):
        windows, draws = self._cache[seed, m]
        return tags_from_windows(windows, draws, len(self.data[seed][m]["labels"]), flip, echo)


def scenario_settings(name, value):
    """{flip, echo, exponent} del escenario con su parametro a `value`."""
    parameter, _ = SCENARIOS[name]
    return dict(NEUTRAL, **{parameter: value})


# --------------------------------------------------------------------------- decodificar

def tagged_data(data, tags, weight, config):
    """El partido preparado con la evidencia del tagging sumada a su coste."""
    tag_config = dict(config, smart_tagging_team_weight=weight)
    home = data["home_logit"] + emissions.team_tag_logit(tags, tag_config)
    return dict(data, cost=emissions.costs_from_logits(home, data["dead_logit"], config))


def check_costs_rebuild(data, config):
    """Reconstruir el coste desde los logits tiene que dar el de la cache, al bit."""
    if not np.array_equal(emissions.costs_from_logits(data["home_logit"], data["dead_logit"],
                                                      config), data["cost"]):
        raise RuntimeError("costs_from_logits no reproduce el coste cacheado")


def decode_with_tags(data, tags, weight, fold_choice, score, config):
    discount_weight, switch_scale = fold_choice
    return boundaries.decode(tagged_data(data, tags, weight, config), config,
                             discount_weight, switch_scale, score)


# --------------------------------------------------------------------------- medicion

def calibrate(name, data, profile):
    """(valor calibrado, en_el_borde, tabla, simulacion) del parametro del escenario."""
    parameter, grid = SCENARIOS[name]
    table, simulations = [], {}
    for value in grid:
        settings = scenario_settings(name, value)
        exponent = settings["exponent"]
        if exponent not in simulations:
            simulations[exponent] = Simulation(data, profile, exponent)
        sim = simulations[exponent]
        overall, quartiles = [], []
        for m in sorted(data[0]):
            tags = sim.tags(0, m, settings["flip"], settings["echo"])
            a, q = agreement(data[0][m]["home_logit"], tags, np.ones(len(tags), dtype=bool))
            overall.append(a)
            quartiles.append(q)
        table.append({parameter: value, "agreement": float(np.mean(overall)),
                      "agreement_by_quartile": list(np.mean(quartiles, axis=0))})
    best = min(table, key=lambda row: abs(row["agreement"] - profile["agreement"]))
    value = best[parameter]
    at_edge = value in (grid[0], grid[-1]) and abs(best["agreement"] - profile["agreement"]) > 0.01
    return value, at_edge, table, simulations[scenario_settings(name, value)["exponent"]]


def choose_weight(data_by_match, tags_by_match, score_by_match, choice_by_match, config):
    """Peso de TAG_WEIGHTS con mejor spellF1 medio en esos partidos; desempate macroF1."""
    rows = []
    for weight in TAG_WEIGHTS:
        scores = [boundaries.score_labels(decode_with_tags(
            data_by_match[m], tags_by_match[m], weight, choice_by_match[m],
            score_by_match[m], config), data_by_match[m]) for m in data_by_match]
        rows.append((round(float(np.mean([s["spell_f1"] for s in scores])), 6),
                     round(float(np.mean([s["macro_f1"] for s in scores])), 6), weight))
    best = max(rows, key=lambda r: (r[0], r[1]))
    return best[2], [list(r) for r in rows]


def prepared(config):
    spec = harness.real_spec()
    matches = sorted(idsse.MATCHES)
    data = {seed: {m: boundaries.prepare(m, spec, seed, config) for m in matches}
            for seed in boundaries.SEEDS}
    for seed in boundaries.SEEDS:
        for m in matches:
            check_costs_rebuild(data[seed][m], config)
    return matches, data


class Detectors:
    """Detectores de frontera por conjunto de entrenamiento, como en el LOMO."""

    def __init__(self, data):
        self.data = data
        self._models = {}

    def model(self, train_ids):
        key = frozenset(train_ids)
        if key not in self._models:
            self._models[key] = boundaries.fit([self.data[0][j] for j in sorted(key)])
        return self._models[key]


def nested_lomo(name, settings, simulation, data, matches, detectors, fold_choice, config,
                extra_flips=()):
    """Peso elegido en los 6 de dentro, evaluado en el de fuera con las 3 semillas."""
    folds = []
    for held_out in matches:
        inner = [m for m in matches if m != held_out]
        inner_scores = {m: boundaries.boundary_score(
            detectors.model([j for j in inner if j != m]), data[0][m]) for m in inner}
        weight, grid = choose_weight(
            {m: data[0][m] for m in inner},
            {m: simulation.tags(0, m, settings["flip"], settings["echo"]) for m in inner},
            inner_scores, {m: fold_choice[held_out] for m in inner}, config)
        outer_model = detectors.model(inner)
        fold = {"held_out": held_out, "weight": weight, "grid": grid, "seeds": {}}
        for seed in boundaries.SEEDS:
            held = data[seed][held_out]
            score = boundaries.boundary_score(outer_model, held)
            row = {"control": evidence.mechanism(
                boundaries.decode(held, config, *fold_choice[held_out], score), held["labels"])}
            for flip in (settings["flip"],) + tuple(extra_flips):
                tags = simulation.tags(seed, held_out, flip, settings["echo"])
                row["flip_%.3f" % flip] = evidence.mechanism(decode_with_tags(
                    held, tags, weight, fold_choice[held_out], score, config), held["labels"])
            fold["seeds"][seed] = row
        log("   [%s] %s peso %.1f  " % (name, held_out, weight) + "  ".join(
            "s%d %+.2f" % (s, 100 * (fold["seeds"][s]["flip_%.3f" % settings["flip"]]["spell_f1"]
                                    - fold["seeds"][s]["control"]["spell_f1"]))
            for s in boundaries.SEEDS))
        folds.append(fold)
    return folds


def summarise(folds, key):
    rng = np.random.default_rng(0)
    seed_mean = [np.mean([f["seeds"][s][key]["spell_f1"] - f["seeds"][s]["control"]["spell_f1"]
                          for s in f["seeds"]]) for f in folds]
    macro = [np.mean([f["seeds"][s][key]["macro_f1"] - f["seeds"][s]["control"]["macro_f1"]
                      for s in f["seeds"]]) for f in folds]
    out = {"spell_f1_mean_of_seeds": evidence.paired_statistics(seed_mean, rng),
           "macro_f1_mean_of_seeds_pp": 100 * float(np.mean(macro))}
    for metric in ("transition_f1_12fr", "error_near_change", "error_elsewhere"):
        out[metric] = {arm: float(np.mean([f["seeds"][s][arm][metric] for f in folds
                                           for s in f["seeds"]])) for arm in ("control", key)}
    return out


def measure(config=None):
    config = config or DEFAULT
    started = time.time()
    profile = real_profile(config)
    log("BAR-ATH: tagging en %.1f %% de los frames, tramo mediano %.0f fr, acuerdo con la "
        "emision %.3f (por cuartil %s)" % (
            100 * profile["coverage"], np.median(profile["run_lengths"]),
            profile["agreement"], " ".join("%.3f" % q for q in profile["agreement_by_quartile"])))
    matches, data = prepared(config)
    choices, _ = evidence.lomo_choices()
    fold_choice = {m: choices[m]["both"] for m in matches}
    detectors = Detectors(data)

    path = os.path.join(evidence.DATA_DIR, "smart_tagging_lomo.json")
    result = {"profile": {k: v for k, v in profile.items() if k != "run_lengths"},
              "settings": {"tag_weights": TAG_WEIGHTS, "scenarios": SCENARIOS,
                           "break_even_flips": BREAK_EVEN_FLIPS,
                           "tag_seed_offset": TAG_SEED_OFFSET},
              "scenarios": {}}
    for name, (parameter, _) in SCENARIOS.items():
        value, at_edge, table, simulation = calibrate(name, data, profile)
        settings = scenario_settings(name, value)
        log("== %s: %s calibrado %s%s" % (name, parameter, value,
                                         "  (EN EL BORDE: no llega al acuerdo real)"
                                         if at_edge else ""))
        for row in table:
            log("   %s %-5s acuerdo %.3f  cuartiles %s" % (
                parameter, row[parameter], row["agreement"],
                " ".join("%.3f" % q for q in row["agreement_by_quartile"])))
        extra = BREAK_EVEN_FLIPS if name == "independent" else ()
        folds = nested_lomo(name, settings, simulation, data, matches, detectors,
                            fold_choice, config, extra)
        summary = {"flip_%.3f" % f: summarise(folds, "flip_%.3f" % f)
                   for f in (settings["flip"],) + tuple(extra)}
        for key, s in summary.items():
            st = s["spell_f1_mean_of_seeds"]
            log("   %s %s: %+.2f pp spellF1 [%+.2f, %+.2f], %d/7 mejor, p=%.3f; macroF1 %+.2f pp"
                % (name, key, st["mean_pp"], st["ci95_pp"][0], st["ci95_pp"][1],
                   st["matches_better"], st["sign_test_p_one_sided"],
                   s["macro_f1_mean_of_seeds_pp"]))
        result["scenarios"][name] = {"parameter": parameter, "calibrated": value,
                                     "calibration_at_edge": at_edge, "calibration": table,
                                     "folds": folds, "summary": summary}
        # Tras cada escenario, para que un fallo en el siguiente no se lleve este.
        with open(path, "w") as handle:
            json.dump(result, handle, indent=1, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))
    return result


def export(name, config=None):
    """El peso para config.py en un escenario: el mismo procedimiento sobre los 7."""
    config = config or DEFAULT
    profile = real_profile(config)
    matches, data = prepared(config)
    value, at_edge, _, simulation = calibrate(name, data, profile)
    settings = scenario_settings(name, value)
    tags = {m: simulation.tags(0, m, settings["flip"], settings["echo"]) for m in matches}
    detectors = Detectors(data)
    out_of_fold = {m: boundaries.boundary_score(
        detectors.model([j for j in matches if j != m]), data[0][m]) for m in matches}
    choice = (config["boundary_discount_weight"], config["boundary_switch_scale"])
    weight, grid = choose_weight({m: data[0][m] for m in matches}, tags, out_of_fold,
                                 {m: choice for m in matches}, config)
    log("%s: %s calibrado %s%s; peso elegido sobre los 7: %.1f" % (
        name, SCENARIOS[name][0], value, " (en el borde)" if at_edge else "", weight))
    for row in grid:
        log("   peso %.1f  spellF1 %.4f  macroF1 %.4f" % (row[2], row[0], row[1]))
    return weight


# --------------------------------------------------------------------------- robustez

# Calibracion conjunta: el acuerdo por cuartil de confianza tiene cuatro numeros, no uno, y
# distingue escenarios que el acuerdo medio confunde (eco 0,5 y facil 8 dan los dos 0,762).
MIX_ECHO = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MIX_EXPONENT = (0.0, 1.0, 2.0, 4.0, 8.0)
# Pesos fijos, sin eleccion: si uno pierde poco en el peor escenario y gana en los demas,
# se puede encender sin saber cual es el verdadero.
FIXED_WEIGHTS = (0.5, 1.0)


def calibrate_mixed(data, profile):
    """(tabla ordenada por ajuste, simulaciones por exponente) de la rejilla eco x exponente.

    El ajuste es el error cuadratico medio entre el acuerdo por cuartil simulado (medio
    sobre las 3 semillas y los 7 partidos) y el de BAR-ATH.
    """
    target = np.array(profile["agreement_by_quartile"])
    simulations = {k: Simulation(data, profile, k) for k in MIX_EXPONENT}
    table = []
    for exponent in MIX_EXPONENT:
        for echo in MIX_ECHO:
            overall, quartiles = [], []
            for seed in data:
                for m in sorted(data[seed]):
                    tags = simulations[exponent].tags(seed, m, echo=echo)
                    a, q = agreement(data[seed][m]["home_logit"], tags,
                                     np.ones(len(tags), dtype=bool))
                    overall.append(a)
                    quartiles.append(q)
            quartiles = np.mean(quartiles, axis=0)
            table.append({"echo": echo, "exponent": exponent,
                          "agreement": float(np.mean(overall)),
                          "agreement_by_quartile": [float(q) for q in quartiles],
                          "quartile_rmse": float(np.sqrt(np.mean((quartiles - target) ** 2)))})
    table.sort(key=lambda row: row["quartile_rmse"])
    return table, simulations


def fixed_weight_rows(scenarios, simulations, data, matches, detectors, fold_choice, config):
    """Cambio de spellF1 del partido de fuera con cada peso fijo, por escenario y semilla."""
    rows = []
    for held_out in matches:
        model = detectors.model([m for m in matches if m != held_out])
        for seed in boundaries.SEEDS:
            held = data[seed][held_out]
            score = boundaries.boundary_score(model, held)
            control = boundaries.score_labels(boundaries.decode(
                held, config, *fold_choice[held_out], score), held)
            for name, settings in scenarios.items():
                tags = simulations[settings["exponent"]].tags(
                    seed, held_out, settings["flip"], settings["echo"])
                for weight in FIXED_WEIGHTS:
                    result = boundaries.score_labels(decode_with_tags(
                        held, tags, weight, fold_choice[held_out], score, config), held)
                    rows.append({"match": held_out, "seed": seed, "scenario": name,
                                 "weight": weight,
                                 "delta_spell_f1": result["spell_f1"] - control["spell_f1"],
                                 "delta_macro_f1": result["macro_f1"] - control["macro_f1"]})
        log("   pesos fijos: %s hecho" % held_out)
    return rows


def summarise_fixed(rows, scenarios, matches):
    rng = np.random.default_rng(0)
    out = {}
    for name in scenarios:
        for weight in FIXED_WEIGHTS:
            subset = [r for r in rows if r["scenario"] == name and r["weight"] == weight]
            per_match = [np.mean([r["delta_spell_f1"] for r in subset if r["match"] == m])
                         for m in matches]
            stats = evidence.paired_statistics(per_match, rng)
            stats["macro_f1_pp"] = 100 * float(np.mean([r["delta_macro_f1"] for r in subset]))
            out["%s|%.1f" % (name, weight)] = stats
            log("   %-22s peso %.1f: %+.2f pp [%+.2f, %+.2f] %d/7, macroF1 %+.2f pp" % (
                name, weight, stats["mean_pp"], stats["ci95_pp"][0], stats["ci95_pp"][1],
                stats["matches_better"], stats["macro_f1_pp"]))
    return out


def robustness(config=None):
    config = config or DEFAULT
    started = time.time()
    profile = real_profile(config)
    matches, data = prepared(config)
    choices, _ = evidence.lomo_choices()
    fold_choice = {m: choices[m]["both"] for m in matches}
    detectors = Detectors(data)

    table, simulations = calibrate_mixed(data, profile)
    log("BAR-ATH por cuartil: %s" % " ".join(
        "%.3f" % q for q in profile["agreement_by_quartile"]))
    log("mejores ajustes por cuartil (eco, exponente):")
    for row in table[:8]:
        log("   eco %.1f exp %.0f  acuerdo %.3f  cuartiles %s  rmse %.4f" % (
            row["echo"], row["exponent"], row["agreement"],
            " ".join("%.3f" % q for q in row["agreement_by_quartile"]), row["quartile_rmse"]))
    best = table[0]
    mixed = {"flip": 0.0, "echo": best["echo"], "exponent": best["exponent"]}

    log("== LOMO anidado con el mejor ajuste conjunto")
    folds = nested_lomo("mixed", mixed, simulations[mixed["exponent"]], data, matches,
                        detectors, fold_choice, config)
    nested = summarise(folds, "flip_0.000")
    st = nested["spell_f1_mean_of_seeds"]
    log("   mixed: %+.2f pp spellF1 [%+.2f, %+.2f], %d/7 mejor, p=%.3f; macroF1 %+.2f pp" % (
        st["mean_pp"], st["ci95_pp"][0], st["ci95_pp"][1], st["matches_better"],
        st["sign_test_p_one_sided"], nested["macro_f1_mean_of_seeds_pp"]))

    calibrated = {}
    for name in ("echo", "easy"):
        value = calibrate(name, data, profile)[0]
        calibrated[name] = scenario_settings(name, value)
        if calibrated[name]["exponent"] not in simulations:
            raise RuntimeError("exponente calibrado fuera de MIX_EXPONENT")
    scenarios = {
        "perfect": dict(NEUTRAL),
        "independent_flip_0.1": dict(NEUTRAL, flip=0.1),
        "independent_flip_0.2": dict(NEUTRAL, flip=0.2),
        "echo_calibrated": calibrated["echo"],
        "easy_calibrated": calibrated["easy"],
        "mixed_best_fit": mixed,
        "echo_total": dict(NEUTRAL, echo=1.0),
    }
    log("== pesos fijos %s en cada escenario" % (FIXED_WEIGHTS,))
    rows = fixed_weight_rows(scenarios, simulations, data, matches, detectors, fold_choice,
                             config)
    fixed = summarise_fixed(rows, scenarios, matches)

    path = os.path.join(evidence.DATA_DIR, "smart_tagging_robustez.json")
    with open(path, "w") as handle:
        json.dump({"profile": {k: v for k, v in profile.items() if k != "run_lengths"},
                   "mixed_calibration": table, "mixed_best": mixed,
                   "mixed_nested_lomo": {"folds": folds, "summary": nested},
                   "scenarios": scenarios, "fixed_weights": fixed, "fixed_rows": rows},
                  handle, indent=1, default=float)
    log("guardado %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2])
    elif sys.argv[1:2] == ["robust"]:
        robustness()
    else:
        measure()
