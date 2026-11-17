"""Measures the new signals inside the submission decoder. Seed 0, 7 IDSSE matches.

Control: the submission config (discount 0.95 and scale 2.0 with the out-of-sample
detector). Treatments: swap the team logit or the dead-ball logit with the classifiers
from modelos.py. Each weight is chosen **nested**: for the held-out match h the variant
with the best mean over the other 6 is picked, scored with models that saw neither h nor
the match itself, and it is evaluated on h with the model that did not see h. The public
window is not used.
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
from mtah import evaluate as ev                                # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.decode import viterbi                                # noqa: E402
from mtah.models import emissions                              # noqa: E402

HERE = os.path.dirname(__file__)
G = {}
CTL = ()

MASKS = ("all", "unseen", "blind")
FAMILIES = {
    "a  equipo sin balon, suma": [(("a", m, w),) for m in MASKS for w in (0.5, 1.0, 2.0)],
    "c  equipo con contexto, suma": [(("c_add", m, w),) for m in MASKS for w in (0.5, 1.0, 2.0)],
    "c  equipo con contexto, sustituye": [(("c_rep", m, w),) for m in MASKS for w in (1.0, 2.0, 3.0)],
    "b  muerto aprendido": [(("b", m, a, k),) for m in ("all", "unseen") for a in (0.5, 1.0)
                            for k in (0.5, 1.0, 2.0)],
    "u  empuje a muerto sin balon": [(("u", d),) for d in (-1.0, -0.5, 0.5)],
}
TEAM_FAMILIES = [k for k in FAMILIES if k[0] in "ac"]
NEEDS_PRED = {"a", "c_add", "c_rep", "b"}


def setup(seed):
    G["base"] = C.base(seed)
    G["score"] = C.oof_scores(None)[seed]
    G["pred"] = dict(np.load(M.pred_path(seed)))
    G["mask"], G["gt"] = {}, {}
    for m in C.MATCHES:
        seen = G["base"][m]["ball_seen"]
        G["mask"][m] = {"all": np.ones(len(seen), bool), "unseen": ~seen,
                        "blind": C.no_ball_in_window(seen)}
        G["gt"][m] = C.pm.label_codes(G["base"][m]["labels"])
    G["transition"] = C.SCALE * viterbi.switch_costs(
        DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"], DEFAULT["switch_from_dead"])


def uses_pred(variant):
    return any(part[0] in NEEDS_PRED for part in variant)


def logits(variant, match, other):
    b = G["base"][match]
    home = b["home_logit"].astype(np.float64).copy()
    dead = b["dead_logit"].astype(np.float64).copy()

    def p(kind):
        key = "%s_%s" % (kind, match) if other is None else "%s_%s_%s" % (kind, match, other)
        return G["pred"][key].astype(np.float64)

    for part in variant:
        name = part[0]
        if name in ("a", "c_add"):
            home = home + part[2] * p(name[0]) * G["mask"][match][part[1]]
        elif name == "c_rep":
            home = np.where(G["mask"][match][part[1]], part[2] * p("c"), home)
        elif name == "b":
            mixed = (1.0 - part[2]) * dead + part[2] * part[3] * p("b")
            dead = np.where(G["mask"][match][part[1]], mixed, dead)
        elif name == "u":
            dead = dead + part[1] * G["mask"][match]["unseen"]
        else:
            raise ValueError(name)
    return home, dead


def run(job):
    variant, match, other = job
    home, dead = logits(variant, match, other)
    cost = emissions.costs_from_logits(home, dead, DEFAULT)
    blocks = ((0, len(cost)),)
    codes = viterbi.decode(cost, G["transition"], blocks=blocks,
                           switch_discount=C.DISCOUNT * G["score"][match])
    codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"], blocks=blocks)
    r = ev.score(viterbi.to_labels(codes), gt_codes=G["gt"][match])
    return job, (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"])


def key(job):
    variant, match, other = job
    if not uses_pred(variant):
        other = None
    return (variant, match, other)


def jobs_for(variant):
    out = [(variant, m, None) for m in C.MATCHES]
    if uses_pred(variant):
        out += [(variant, m, h) for h in C.MATCHES for m in C.MATCHES if m != h]
    return out


class Runner:
    def __init__(self, seed, workers=9):
        self.path = os.path.join(HERE, "exp_s%d.json" % seed)
        self.results = {}
        if os.path.exists(self.path):
            with open(self.path) as handle:
                self.results = {self._parse(k): tuple(v) for k, v in json.load(handle).items()}
        self.pool = multiprocessing.get_context("fork").Pool(workers)

    @staticmethod
    def _parse(text):
        variant, match, other = json.loads(text)
        return (tuple(tuple(p) for p in variant), match, other)

    def ensure(self, variants):
        todo = sorted({key(j) for v in variants for j in jobs_for(v)} - set(self.results),
                      key=repr)
        if not todo:
            return
        started = time.time()
        for job, value in self.pool.imap_unordered(run, todo, chunksize=2):
            self.results[job] = value
        with open(self.path, "w") as handle:
            json.dump({json.dumps(k): v for k, v in self.results.items()}, handle)
        print("   %d decodings in %.0fs" % (len(todo), time.time() - started), flush=True)

    def spell(self, variant, match, other=None):
        return self.results[key((variant, match, other))][0]

    def inner(self, variant, held):
        return float(np.mean([self.spell(variant, m, held) for m in C.MATCHES if m != held]))

    def outer(self, variant):
        return float(np.mean([self.spell(variant, m) for m in C.MATCHES]))


def name(variant):
    return "control" if variant == CTL else " + ".join(
        "%s[%s]" % (p[0], ",".join(str(x) for x in p[1:])) for p in variant)


def nested(runner, family, title):
    """Choose on the 6 inner matches, evaluate on the held-out one. Returns {match: variant}."""
    options = [CTL] + list(family)
    deltas, chosen = [], {}
    for held in C.MATCHES:
        best = max(options, key=lambda v: runner.inner(v, held))
        chosen[held] = best
        deltas.append(runner.spell(best, held) - runner.spell(CTL, held))
    deltas = np.array(deltas) * 100
    picks = {}
    for v in chosen.values():
        picks[name(v)] = picks.get(name(v), 0) + 1
    print("NESTED  %-36s %+.2f pp  (%d better / %d worse)  per match %s" % (
        title, deltas.mean(), (deltas > 0).sum(), (deltas < 0).sum(),
        " ".join("%+.1f" % d for d in deltas)))
    print("        chosen: %s" % ", ".join("%s x%d" % kv for kv in sorted(picks.items(), key=lambda kv: -kv[1])))
    return chosen, float(deltas.mean())


def main(seed=0):
    setup(seed)
    runner = Runner(seed)
    runner.ensure([CTL])
    control = runner.outer(CTL)
    print("CONTROL spellF1 %.4f  per match %s" % (
        control, " ".join("%.4f" % runner.spell(CTL, m) for m in C.MATCHES)), flush=True)
    summary = {"control": control, "families": {}}
    chosen = {}
    for title, family in FAMILIES.items():
        runner.ensure(family)
        print("== %s   (out-of-sample mean, WITHOUT nested choice: optimistic)" % title)
        for v in family:
            d = np.array([runner.spell(v, m) - runner.spell(CTL, m) for m in C.MATCHES]) * 100
            print("   %-34s %.4f  %+.2f pp  (%d/%d)" % (
                name(v), runner.outer(v), d.mean(), (d > 0).sum(), (d < 0).sum()))
        chosen[title], delta = nested(runner, family, title)
        summary["families"][title] = {
            "nested_pp": delta,
            "variants": {name(v): runner.outer(v) - control for v in family}}
        sys.stdout.flush()

    # Team: the best of the three team families, chosen nested.
    team_all = [v for t in TEAM_FAMILIES for v in FAMILIES[t]]
    chosen_team, delta = nested(runner, team_all, "team (a or c, either)")
    summary["team_any_nested_pp"] = delta
    dead_all = FAMILIES["b  muerto aprendido"] + FAMILIES["u  empuje a muerto sin balon"]
    chosen_dead, delta = nested(runner, dead_all, "dead ball (b or u, either)")
    summary["dead_any_nested_pp"] = delta

    # Combination: per held-out match, the chosen team variant + the chosen dead-ball variant;
    # and the choice is made nested among control, team only, dead ball only and both.
    combos = {h: chosen_team[h] + chosen_dead[h] for h in C.MATCHES}
    runner.ensure(set(combos.values()))
    deltas, picks = [], {}
    for h in C.MATCHES:
        options = [CTL, chosen_team[h], chosen_dead[h], combos[h]]
        best = max(options, key=lambda v: runner.inner(v, h))
        picks[name(best)] = picks.get(name(best), 0) + 1
        deltas.append(runner.spell(best, h) - runner.spell(CTL, h))
    deltas = np.array(deltas) * 100
    print("NESTED  %-36s %+.2f pp  (%d better / %d worse)  per match %s" % (
        "team + dead ball", deltas.mean(), (deltas > 0).sum(), (deltas < 0).sum(),
        " ".join("%+.1f" % d for d in deltas)))
    print("        chosen: %s" % ", ".join("%s x%d" % kv for kv in picks.items()))
    summary["combo_nested_pp"] = float(deltas.mean())
    with open(os.path.join(HERE, "exp_resumen_s%d.json" % seed), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
