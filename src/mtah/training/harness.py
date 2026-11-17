"""Validation harness: measure the pipeline on a FULL match with labels.

The problem it solves. The only ground truth we have for the challenge match is 5 minutes
and 17 possessions, and EVALUATION.md is explicit that it is **no use for choosing between
two versions of the model**: the baseline that wins there loses on the full match. Up to
here every number in this project comes from that window.

A degraded IDSSE match does give a full-match spellF1 against real labels. Seven times.

**What it measures well and what it does not.** It measures *relative* differences well: if
a change improves on all seven matches, it improves. The *absolute* value is only
indicative, for two reasons that cannot be removed: the DFL `BallStatus` convention is not
identical to the challenge's `D`, and the Bundesliga is not this Barcelona-Athletic. Also
the degradation spreads the ball-absent runs at random whereas the ATD concentrates them in
the hard frames, so the proxy is **optimistic** by construction.

Before believing any number, `compare_stats()` checks that the degraded data looks like the
ATD on the axes that were measured. If it does not, the number is worthless.
"""

import numpy as np

from .. import evaluate as ev
from .. import features as feature_set
from .. import submission
from ..config import DEFAULT
from ..constants import GRADED_BLOCKS
from ..decode import viterbi
from ..io import atd
from ..models import emissions
from . import convention, degrade, idsse


def real_spec(source="tactical"):
    """Degradation parameters measured on the real ATD of the challenge match."""
    return degrade.estimate(atd.load(source=source), GRADED_BLOCKS)


def _stats(tracking, blocks=None):
    n = tracking.n_frames
    mask = np.ones(n, dtype=bool)
    if blocks is not None:
        mask[:] = False
        for a, b in blocks:
            mask[a:b] = True
    visible = ~np.isnan(tracking.players_xy[:, :, 0])
    n_tracked = visible[mask].sum(axis=1)
    runs = degrade._runs(~tracking.ball_seen[mask])
    return {
        "balon_visible_pct": 100.0 * tracking.ball_seen[mask].mean(),
        "objetos_media": float(n_tracked.mean()),
        "objetos_p50": float(np.median(n_tracked)),
        "objetos_max": int(n_tracked.max()),
        "rachas_sin_balon_n": len(runs),
        "racha_media_fr": float(runs.mean()) if len(runs) else 0.0,
        "racha_p50_fr": float(np.median(runs)) if len(runs) else 0.0,
        "racha_max_fr": int(runs.max()) if len(runs) else 0,
    }


def compare_stats(degraded, source="tactical"):
    """Print side by side the statistics of the real ATD and those of the degraded proxy."""
    real = _stats(atd.load(source=source), GRADED_BLOCKS)
    fake = _stats(degraded)
    print("%-22s %>12s %12s" .replace(">", "") % ("statistic", "real ATD", "degraded"))
    for k in real:
        print("%-22s %12.2f %12.2f" % (k, real[k], fake[k]))
    return real, fake


def predict(tracking, config=None):
    """(labels, features) of the pipeline on any tracking, in a single block.

    A single block because the IDSSE file does not include the halftime frames: there is no
    gap to separate and the decoder carries nothing across anything. It also returns the
    features so that error analysis does not have to recompute them.

    Without the boundary detector, on purpose: it was trained on these same 7 matches and
    applying it here would be measuring it on its training set. Its honest measurement is
    the one in `training/boundaries.py` (nested leave-one-match-out).
    """
    config = config or DEFAULT
    blocks = ((0, tracking.n_frames),)
    features = feature_set.build_all(tracking, config)
    cost = emissions.costs(features, config)
    transition = viterbi.switch_costs(config["switch_turnover"], config["switch_to_dead"],
                                      config["switch_from_dead"])
    codes = viterbi.decode(cost, transition, blocks=blocks)
    codes = viterbi.enforce_min_duration(codes, config["min_duration_frames"], blocks=blocks)
    return viterbi.to_labels(codes), features


def run_match(match_id, config=None, seed=0, spec=None, verbose=True, harmonise=True):
    """Degrade an IDSSE match, run the pipeline and score it against its labels.

    `harmonise=True` first brings the labels to the challenge's possession convention (see
    convention.py). Without that, the model is asked to get 421 possessions right when it is
    designed to produce ~220, and the spellF1 comes out sunk for a reason that has nothing
    to do with the model's quality.
    """
    config = config or DEFAULT
    spec = spec or real_spec()
    clean, labels = idsse.load(match_id)
    if harmonise:
        labels, _ = convention.harmonise(labels)

    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(seed))
    pred, _ = predict(dirty, config)

    gt_codes = ev.pm.label_codes(np.asarray(labels, dtype="<U1"))
    result = ev.score(pred, gt_codes=gt_codes)
    if verbose:
        print("%-8s spellF1 %.4f  macroF1 %.4f   poss %d vs %d   dur %.1fs vs %.1fs" % (
            match_id, result["spell_f1_mean"], result["macro_f1"],
            result["n_pred_spells"], result["n_gt_spells"],
            result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    return result


def run_all(config=None, seed=0, matches=None, spec=None, harmonise=True):
    """The seven matches. Returns (list of results, mean spellF1, mean macroF1).

    The mean over seven matches is the number to rely on when deciding, not that of any
    single match nor that of the public window.
    """
    spec = spec or real_spec()
    out = []
    for m in (matches or sorted(idsse.MATCHES)):
        out.append(run_match(m, config=config, seed=seed, spec=spec,
                             harmonise=harmonise))
    spell = float(np.mean([r["spell_f1_mean"] for r in out]))
    macro = float(np.mean([r["macro_f1"] for r in out]))
    print("-" * 64)
    print("MEAN of %d matches   spellF1 %.4f   macroF1 %.4f" % (len(out), spell, macro))
    return out, spell, macro
