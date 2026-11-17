"""Weight of Smart Tagging as evidence of "which team has the ball", measured on the IDSSE.

**Why.** 60% of the frames we get wrong on the IDSSE (71% with the ball visible) are
home versus away, not live versus dead. The BAR-ATH Smart Tagging says who owns the
ball in phases of ~18 s that cover 40% of the scored frames, and the pipeline did not
use it. It enters as a term added to the home logit (`models/emissions.team_tag_logit`).

**The problem: the IDSSE has no Smart Tagging**, so its weight cannot be measured
directly. We simulate a tagging on the IDSSE labels that resembles the real one in
what can be measured about the real one **without labels**: disjoint windows with the
lengths of the real windows, the same coverage, centered on live frames, separated from
each other (in BAR-ATH there is never a direct change of team inside the tagging), and
with the window's majority team (a window that crosses a change carries error, like a
real phase).

**What we do not know is how accurate the real tagging is.** The only thing measurable
without labels is how much it agrees with our emission: 0.762 on BAR-ATH. And a *perfect*
simulated tagging agrees with our emission on the IDSSE only 0.706. The real one agrees
more than a perfect one, so "it is X% accurate" cannot be calibrated with independent
noise. There are three explanations, and each is a scenario with its parameter calibrated
to give 0.762:

- `independent`: errors independent of ours (`flip`: probability of flipping a window).
  If even flip 0 does not get there, it is the optimistic bound: perfect tags.
- `echo`: Metrica labels with the same tracking we use, so its errors coincide with
  ours. With probability `echo` the window carries the team that *our* emission says
  instead of the true one. The pessimistic scenario: what repeats what we already know
  adds nothing.
- `easy`: phases are detected where play is clear, which is where our emission is already
  right. Windows are centered with weight |home logit|^k.

The weight is chosen with the same nested LOMO as the boundary detector, on top of the
submitted pipeline (detector included, with each fold's weight and scale), in each
scenario separately. We submit the weight of the scenario that gains least, and only if
it still gains there.

Usage:  python -m mtah.training.team_tags                    (the three scenarios; ~20 min)
        python -m mtah.training.team_tags robust             (fit by quartiles + fixed weights; ~15 min)
        python -m mtah.training.team_tags export <scenario>  (weight for config.py)
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

# Each scenario: (parameter being calibrated, its grid). The other two stay neutral.
SCENARIOS = {
    "independent": ("flip", tuple(round(0.025 * k, 3) for k in range(13))),     # 0 .. 0.30
    "echo": ("echo", tuple(round(0.05 * k, 2) for k in range(21))),             # 0 .. 1
    "easy": ("exponent", (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)),
}
NEUTRAL = {"flip": 0.0, "echo": 0.0, "exponent": 0.0}
# With the weights chosen in `independent`, up to how much independent noise it still wins.
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
    """(agreement, agreement per |logit| quartile) between emission and tagging in `mask`."""
    mask = mask & (tags != 0)
    if not mask.any():
        raise ValueError("no tagged frame in the mask")
    correct = (home_logit[mask] > 0) == (tags[mask] > 0)
    confidence = np.abs(home_logit[mask])
    edges = np.quantile(confidence, np.linspace(0, 1, N_CONFIDENCE_BANDS + 1))
    band = np.clip(np.searchsorted(edges, confidence, side="right") - 1,
                   0, N_CONFIDENCE_BANDS - 1)
    return float(correct.mean()), [float(correct[band == q].mean())
                                   for q in range(N_CONFIDENCE_BANDS)]


def real_profile(config=None):
    """What is measured about the real BAR-ATH tagging without any labels."""
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


# --------------------------------------------------------------------------- simulation

def _majority(values_are_home):
    return 1 if values_are_home.sum() * 2 >= len(values_are_home) else -1


def simulate_windows(labels, home_logit, profile, rng, exponent=0.0):
    """[(start, end, true team, team of our emission)] with +-1 per team.

    `exponent` > 0 centers the windows with weight |home_logit|^exponent: where our emission
    is confident (`easy` scenario).
    """
    labels = np.asarray(labels)
    n = len(labels)
    live = np.flatnonzero(labels != "D")
    # Floor so that no live frame ends up with exactly zero weight.
    weights = (np.abs(home_logit[live]) ** exponent if exponent
               else np.ones(len(live))) + WEIGHT_FLOOR
    # Centers are drawn only among free frames: otherwise, with the weight concentrated, the
    # draws fall again and again in windows already placed and the coverage is never reached.
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
        if taken[max(0, start - 1):stop + 1].any():     # disjoint and with a gap
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
    raise RuntimeError("could not cover %.0f%% of the frames in %d attempts"
                       % (100 * profile["coverage"], MAX_PLACEMENT_ATTEMPTS))


def tags_from_windows(windows, draws, n, flip=0.0, echo=0.0):
    """(n,) int8. Per window: the team of our emission if its echo draw < `echo`, otherwise
    the true one; then flipped if its flip draw < `flip`.

    The draws are fixed per window, so raising `flip` or `echo` only adds changes: the
    levels are nested and compared on the same windows.
    """
    tags = np.zeros(n, dtype=np.int8)
    for (start, stop, truth, ours), flip_draw, echo_draw in zip(windows, draws["flip"],
                                                                draws["echo"]):
        team = ours if echo_draw < echo else truth
        tags[start:stop] = -team if flip_draw < flip else team
    return tags


class Simulation:
    """Windows and draws of a scenario for all (seed, match) pairs, reproducible."""

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
    """{flip, echo, exponent} of the scenario with its parameter set to `value`."""
    parameter, _ = SCENARIOS[name]
    return dict(NEUTRAL, **{parameter: value})


# --------------------------------------------------------------------------- decoding

def tagged_data(data, tags, weight, config):
    """The match prepared with the tagging evidence added to its cost."""
    tag_config = dict(config, smart_tagging_team_weight=weight)
    home = data["home_logit"] + emissions.team_tag_logit(tags, tag_config)
    return dict(data, cost=emissions.costs_from_logits(home, data["dead_logit"], config))


def check_costs_rebuild(data, config):
    """Rebuilding the cost from the logits must give the cached one, bit for bit."""
    if not np.array_equal(emissions.costs_from_logits(data["home_logit"], data["dead_logit"],
                                                      config), data["cost"]):
        raise RuntimeError("costs_from_logits does not reproduce the cached cost")


def decode_with_tags(data, tags, weight, fold_choice, score, config):
    discount_weight, switch_scale = fold_choice
    return boundaries.decode(tagged_data(data, tags, weight, config), config,
                             discount_weight, switch_scale, score)


# --------------------------------------------------------------------------- measurement

def calibrate(name, data, profile):
    """(calibrated value, at_edge, table, simulation) of the scenario's parameter."""
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
    """Weight from TAG_WEIGHTS with the best mean spellF1 on those matches; macroF1 breaks ties."""
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
    """Boundary detectors per training set, as in the LOMO."""

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
    """Weight chosen on the 6 inside, evaluated on the held-out one with the 3 seeds."""
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
        log("   [%s] %s weight %.1f  " % (name, held_out, weight) + "  ".join(
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
    log("BAR-ATH: tagging on %.1f%% of the frames, median window %.0f fr, agreement with the "
        "emission %.3f (by quartile %s)" % (
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
        log("== %s: %s calibrated %s%s" % (name, parameter, value,
                                         "  (AT THE EDGE: does not reach the real agreement)"
                                         if at_edge else ""))
        for row in table:
            log("   %s %-5s agreement %.3f  quartiles %s" % (
                parameter, row[parameter], row["agreement"],
                " ".join("%.3f" % q for q in row["agreement_by_quartile"])))
        extra = BREAK_EVEN_FLIPS if name == "independent" else ()
        folds = nested_lomo(name, settings, simulation, data, matches, detectors,
                            fold_choice, config, extra)
        summary = {"flip_%.3f" % f: summarise(folds, "flip_%.3f" % f)
                   for f in (settings["flip"],) + tuple(extra)}
        for key, s in summary.items():
            st = s["spell_f1_mean_of_seeds"]
            log("   %s %s: %+.2f pp spellF1 [%+.2f, %+.2f], %d/7 better, p=%.3f; macroF1 %+.2f pp"
                % (name, key, st["mean_pp"], st["ci95_pp"][0], st["ci95_pp"][1],
                   st["matches_better"], st["sign_test_p_one_sided"],
                   s["macro_f1_mean_of_seeds_pp"]))
        result["scenarios"][name] = {"parameter": parameter, "calibrated": value,
                                     "calibration_at_edge": at_edge, "calibration": table,
                                     "folds": folds, "summary": summary}
        # After each scenario, so that a failure in the next one does not take this one with it.
        with open(path, "w") as handle:
            json.dump(result, handle, indent=1, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))
    return result


def export(name, config=None):
    """The weight for config.py in one scenario: the same procedure over the 7."""
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
    log("%s: %s calibrated %s%s; weight chosen over the 7: %.1f" % (
        name, SCENARIOS[name][0], value, " (at the edge)" if at_edge else "", weight))
    for row in grid:
        log("   weight %.1f  spellF1 %.4f  macroF1 %.4f" % (row[2], row[0], row[1]))
    return weight


# --------------------------------------------------------------------------- robustness

# Joint calibration: the agreement per confidence quartile has four numbers, not one, and
# it tells apart scenarios that the mean agreement confuses (echo 0.5 and easy 8 both give
# 0.762).
MIX_ECHO = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MIX_EXPONENT = (0.0, 1.0, 2.0, 4.0, 8.0)
# Fixed weights, no choice: if one loses little in the worst scenario and wins in the others,
# it can be switched on without knowing which one is the true one.
FIXED_WEIGHTS = (0.5, 1.0)


def calibrate_mixed(data, profile):
    """(table ordered by fit, simulations per exponent) of the echo x exponent grid.

    The fit is the mean squared error between the simulated agreement per quartile (mean
    over the 3 seeds and the 7 matches) and the BAR-ATH one.
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
    """Change in spellF1 of the held-out match with each fixed weight, per scenario and seed."""
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
        log("   fixed weights: %s done" % held_out)
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
            log("   %-22s weight %.1f: %+.2f pp [%+.2f, %+.2f] %d/7, macroF1 %+.2f pp" % (
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
    log("BAR-ATH by quartile: %s" % " ".join(
        "%.3f" % q for q in profile["agreement_by_quartile"]))
    log("best fits by quartile (echo, exponent):")
    for row in table[:8]:
        log("   echo %.1f exp %.0f  agreement %.3f  quartiles %s  rmse %.4f" % (
            row["echo"], row["exponent"], row["agreement"],
            " ".join("%.3f" % q for q in row["agreement_by_quartile"]), row["quartile_rmse"]))
    best = table[0]
    mixed = {"flip": 0.0, "echo": best["echo"], "exponent": best["exponent"]}

    log("== nested LOMO with the best joint fit")
    folds = nested_lomo("mixed", mixed, simulations[mixed["exponent"]], data, matches,
                        detectors, fold_choice, config)
    nested = summarise(folds, "flip_0.000")
    st = nested["spell_f1_mean_of_seeds"]
    log("   mixed: %+.2f pp spellF1 [%+.2f, %+.2f], %d/7 better, p=%.3f; macroF1 %+.2f pp" % (
        st["mean_pp"], st["ci95_pp"][0], st["ci95_pp"][1], st["matches_better"],
        st["sign_test_p_one_sided"], nested["macro_f1_mean_of_seeds_pp"]))

    calibrated = {}
    for name in ("echo", "easy"):
        value = calibrate(name, data, profile)[0]
        calibrated[name] = scenario_settings(name, value)
        if calibrated[name]["exponent"] not in simulations:
            raise RuntimeError("calibrated exponent outside MIX_EXPONENT")
    scenarios = {
        "perfect": dict(NEUTRAL),
        "independent_flip_0.1": dict(NEUTRAL, flip=0.1),
        "independent_flip_0.2": dict(NEUTRAL, flip=0.2),
        "echo_calibrated": calibrated["echo"],
        "easy_calibrated": calibrated["easy"],
        "mixed_best_fit": mixed,
        "echo_total": dict(NEUTRAL, echo=1.0),
    }
    log("== fixed weights %s in each scenario" % (FIXED_WEIGHTS,))
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
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2])
    elif sys.argv[1:2] == ["robust"]:
        robustness()
    else:
        measure()
