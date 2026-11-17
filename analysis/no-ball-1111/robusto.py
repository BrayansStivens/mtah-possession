"""Ball-absent team classifier (a) with fewer signals: which ones transfer to the ATD.

deriva_senal.py says that `opp` (and somewhat `cen` and `len`) are distributed differently in
the ATD than in the degraded IDSSE, and that width, the signal that weighs the most, is not.
Here we drop what drifts and measure what it costs. The weight (2.0) is the one from the
nested selection with all the signals: nothing is chosen here. Usage: robusto.py modelos | decode
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
from mtah.config import DEFAULT                                # noqa: E402
from mtah.decode import viterbi                                # noqa: E402
from mtah.models import emissions                              # noqa: E402
from mtah.paths import REPO_ROOT, cache_dir                    # noqa: E402

HERE = os.path.dirname(__file__)
WEIGHT = 2.0
SUBSETS = {
    "todas (9)": F.TEAM_SIGNALS,
    "sin opp": ("cen", "len", "wid", "dfl", "atl", "tide", "spd", "spr"),
    "sin opp, cen, len": ("wid", "dfl", "atl", "tide", "spd", "spr"),
    "anchura y lineas": ("wid", "dfl", "atl"),
    "solo anchura": ("wid",),
}
G = {}


def builder(signals):
    index = [F.TEAM_SIGNALS.index(s) for s in signals]
    sums = [F.TEAM_SIGNALS.index(s) for s in F.SUM_SIGNALS if s in signals]

    def build(team, perspective):
        p, q = perspective, 1 - perspective
        cols = []
        for w in range(team.shape[2]):
            cols += [team[p, index, w].T, team[q, index, w].T,
                     (team[p, index, w] - team[q, index, w]).T]
            if sums:
                cols.append((team[p, sums, w] + team[q, sums, w]).T)
        return np.concatenate(cols, axis=1)
    return build


def fit(train, build):
    xs, ys = [], []
    for data in train:
        labels = data["labels"][::M.STRIDE]
        alive = labels != "D"
        for perspective, mine in ((0, "H"), (1, "A")):
            xs.append(build(data["team"], perspective)[::M.STRIDE][alive])
            ys.append(labels[alive] == mine)
    return M.new_model().fit(np.concatenate(xs), np.concatenate(ys))


def predict(model, team, build):
    return (0.5 * (model.decision_function(build(team, 0))
                   - model.decision_function(build(team, 1)))).astype(np.float32)


def pred_path():
    return os.path.join(cache_dir(), "sinbalon_robusto.npz")


def run_modelos():
    import deriva as D
    import deriva_senal as DS
    from sklearn.metrics import roc_auc_score
    data = {m: M.feat(m, 0) for m in C.MATCHES}
    real_team, real_seen, graded = DS.atd_team()
    realistic = {m: np.load(os.path.join(cache_dir(), "sinbalon_realista_%s_s0.npz" % m))["ball_seen"]
                 for m in C.MATCHES}
    assert np.array_equal(builder(F.TEAM_SIGNALS)(data[C.MATCHES[0]]["team"], 1),
                          M.team_matrix(data[C.MATCHES[0]]["team"], 1), equal_nan=True)
    out, info = {}, {}
    started = time.time()
    for name, signals in SUBSETS.items():
        build = builder(signals)
        scores, truth, unseen = [], [], []
        for held in C.MATCHES:
            model = fit([data[m] for m in C.MATCHES if m != held], build)
            logit = predict(model, data[held]["team"], build)
            out["%s|%s" % (name, held)] = logit
            alive = data[held]["labels"] != "D"
            scores.append(logit[alive])
            truth.append(data[held]["labels"][alive] == "H")
            unseen.append(~data[held]["ball_seen"][alive])
        out["%s|ATD" % name] = predict(fit(list(data.values()), build), real_team, build)
        scores, truth, unseen = (np.concatenate(v) for v in (scores, truth, unseen))
        teams = [data[m]["team"] for m in C.MATCHES]
        row = {"columns": int(build(real_team[:, :, :, :2], 0).shape[1]),
               "auc_all": float(roc_auc_score(truth, scores)),
               "auc_unseen": float(roc_auc_score(truth[unseen], scores[unseen]))}
        for region, keep in (("seen", True), ("unseen", False)):
            masks = [realistic[m] == keep for m in C.MATCHES]
            row["domain_" + region] = D.domain_auc(
                real_team, graded & (real_seen == keep), teams[:6], masks[:6], build)
            row["domain_ref_" + region] = float(np.mean([
                D.domain_auc(teams[j], masks[j], [teams[k] for k in range(7) if k != j],
                             [masks[k] for k in range(7) if k != j], build) for j in (0, 3, 6)]))
        info[name] = row
        print("%-20s %2d columns  IDSSE team AUC %.4f (ball-absent %.4f)  ATD-IDSSE domain seen %.3f (ref %.3f)  ball-absent %.3f (ref %.3f)  %.0fs" % (
            name, row["columns"], row["auc_all"], row["auc_unseen"], row["domain_seen"],
            row["domain_ref_seen"], row["domain_unseen"], row["domain_ref_unseen"],
            time.time() - started), flush=True)
    np.savez_compressed(pred_path(), **out)
    with open(os.path.join(HERE, "robusto_modelos.json"), "w") as handle:
        json.dump(info, handle, indent=1)


# --------------------------------------------------------------------------- decode

def setup():
    import realista as R
    G["pred"] = dict(np.load(pred_path()))
    G["rand"] = C.base(0)
    G["rand_score"] = C.oof_scores(None)[0]
    G["real"] = {m: R.load(m) for m in C.MATCHES}
    realistic = dict(np.load(R.pred_path()))
    G["real_score"] = {m: realistic["det_rand_%s" % m] for m in C.MATCHES}
    G["gt"] = {m: C.pm.label_codes(G["rand"][m]["labels"]) for m in C.MATCHES}
    G["transition"] = C.SCALE * viterbi.switch_costs(
        DEFAULT["switch_turnover"], DEFAULT["switch_to_dead"], DEFAULT["switch_from_dead"])


def run(job):
    name, gaps, match = job
    d = G[gaps][match]
    home = d["home_logit"].astype(np.float64)
    if name is not None:
        home = home + WEIGHT * G["pred"]["%s|%s" % (name, match)].astype(np.float64)
    cost = emissions.costs_from_logits(home, d["dead_logit"].astype(np.float64), DEFAULT)
    blocks = ((0, len(cost)),)
    codes = viterbi.decode(cost, G["transition"], blocks=blocks,
                           switch_discount=C.DISCOUNT * G[gaps + "_score"][match].astype(np.float64))
    codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"], blocks=blocks)
    return job, ev.score(viterbi.to_labels(codes), gt_codes=G["gt"][match])["spell_f1_mean"]


def real_match():
    """Reading on the challenge match, as in real.py: {subset: row}."""
    import real as RM
    from mtah import features as feature_set
    from mtah import submission
    from mtah.constants import graded_mask
    from mtah.features import tagging
    from mtah.io import atd, smart_tagging
    from mtah.models import boundary_detector
    tracking = atd.load()
    n = tracking.n_frames
    graded = graded_mask(n)
    phases = smart_tagging.load()
    weak, conflict = tagging.weak_labels(phases, n)
    usable = graded & ~conflict & (weak != "")
    features = feature_set.build_all(tracking, DEFAULT)
    hand_home = emissions.home_logit(features, DEFAULT)
    hand_dead = emissions.dead_logit(features, DEFAULT)
    tag_logit = emissions.team_tag_logit(tagging.team_tags(phases, n), DEFAULT)
    discount = boundary_detector.switch_discount(features, DEFAULT)
    delivered = np.asarray(ev.pm.read_label_csv(
        os.path.join(REPO_ROOT, "out", "submission.csv"))["label"], dtype="<U1")
    seen = tracking.ball_seen
    rows = {}
    for name in [None] + list(SUBSETS):
        home = hand_home.copy()
        if name is not None:
            home = home + WEIGHT * G["pred"]["%s|ATD" % name].astype(np.float64)
        cost = emissions.costs_from_logits(home + tag_logit, hand_dead, DEFAULT)
        codes = viterbi.decode(cost, G["transition"], switch_discount=discount)
        codes = viterbi.enforce_min_duration(codes, DEFAULT["min_duration_frames"])
        labels = submission.build(viterbi.to_labels(codes))
        if name is None:
            assert (labels == delivered).all()
        r = ev.score(labels)
        count, duration, share = RM.spells(labels, graded)
        alive = usable & (labels != "D")
        rows[name or "entrega"] = {
            "spell_public": r["spell_f1_mean"], "macro_public": r["macro_f1"],
            "n_pred_public": r["n_pred_spells"], "n_spells": count,
            "same": float((labels[graded] == delivered[graded]).mean()),
            "same_unseen": float((labels[graded & ~seen] == delivered[graded & ~seen]).mean()),
            "weak": float((labels[alive] == weak[alive]).mean())}
    return rows


def run_decode():
    setup()
    jobs = [(name, gaps, m) for name in [None] + list(SUBSETS) for gaps in ("rand", "real")
            for m in C.MATCHES]
    with multiprocessing.get_context("fork").Pool(9) as pool:
        results = dict(pool.imap_unordered(run, jobs, chunksize=2))
    rows = real_match()
    with open(os.path.join(HERE, "robusto_modelos.json")) as handle:
        info = json.load(handle)
    summary = {}
    print("%-20s | IDSSE: team AUC  delta random  delta realistic (win/loss) | ATD domain seen / ball-absent | real match (READING): spellF1 macroF1 pos/17  possessions  same as submission  phase agreement" % "signals for (a)")
    for name in [None] + list(SUBSETS):
        delta = {}
        for gaps in ("rand", "real"):
            d = np.array([results[(name, gaps, m)] - results[(None, gaps, m)] for m in C.MATCHES]) * 100
            delta[gaps] = (float(d.mean()), int((d > 0).sum()), int((d < 0).sum()))
        r = rows[name or "entrega"]
        i = info.get(name, {})
        summary[name or "entrega"] = {"delta": delta, "real_match": r, "models": i}
        print("%-20s |        %s      %+6.2f       %+6.2f (%d/%d)     |        %s / %s        |      %.4f  %.4f  %2d/17      %3d        %5.1f%%          %5.1f%%" % (
            name or "entrega", "%.4f" % i["auc_all"] if i else "  -   ", delta["rand"][0],
            delta["real"][0], delta["real"][1], delta["real"][2],
            "%.3f" % i["domain_seen"] if i else "  -  ", "%.3f" % i["domain_unseen"] if i else "  -  ",
            r["spell_public"], r["macro_public"], r["n_pred_public"], r["n_spells"],
            100 * r["same"], 100 * r["weak"]))
    with open(os.path.join(HERE, "robusto.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    {"modelos": run_modelos, "decode": run_decode}[sys.argv[1]]()
