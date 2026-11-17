"""Sensitivity curve to ball visibility: Step zero of plan-detector-balon.md.

**The question.** A ball detector of our own on the video is only worth it if seeing more
ball raises the spellF1. The harness answers it without building anything: IDSSE is degraded
as always (ball visible ~59%, like the ATD) and then ball is *given back* up to 70, 80, 90
and 100%, with the `config.py` model untouched. Flat curve: the detector is not started.
Steep curve: it is the best investment left.

**Two ways of giving the ball back**, because a real detector does not recover at random:

- `random`: whole gaps in random order.
- `easy_first`: first the gaps with the ball far from every player (open play), and the
  crowded ones last -- duels, boxes, tackles --, which is where a detector fails and where
  possession is decided. This is the honest estimate; `random` is the optimistic one.

Whole gaps are recovered and not isolated frames: the model averages +-50 frames and a few
scattered frames inside a long gap would help it more than a detector that loses the ball
in the same places as the tracker would.

What matters is the **slope**. The level is inflated because the degradation places the
gaps at random and the ATD concentrates them in the hard frames.

Usage:  python -m mtah.training.visibility
"""

import json
import os
import time

import numpy as np

from .. import evaluate as ev
from ..config import DEFAULT
from ..features import geometry
from ..io.atd import Tracking
from ..paths import REPO_ROOT
from . import convention, degrade, harness, idsse

TARGET_VISIBILITY = (0.70, 0.80, 0.90, 1.00)
STRATEGIES = ("random", "easy_first")
SEED = 0


def log(message):
    print(message, flush=True)


def hidden_gaps(dirty_seen, clean_seen):
    """[(start, end)] of runs in which the degradation hid a ball that did exist."""
    hidden = ~dirty_seen & clean_seen
    edges = np.diff(np.concatenate(([0], hidden.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def recovery_order(gaps, strategy, nearest_player_m, rng):
    """Order in which a detector would give the gaps back."""
    if strategy == "random":
        return list(rng.permutation(len(gaps)))
    if strategy == "easy_first":
        openness = [float(np.nanmean(nearest_player_m[a:b])) for a, b in gaps]
        return list(np.argsort(openness)[::-1])
    raise ValueError("unknown strategy {!r}".format(strategy))


def restore_ball(dirty, clean, frames):
    """Degraded tracking with the clean ball given back at `frames` (bool mask)."""
    ball_xy, ball_speed = dirty.ball_xy.copy(), dirty.ball_speed.copy()
    ball_xy[frames] = clean.ball_xy[frames]
    ball_speed[frames] = clean.ball_speed[frames]
    return Tracking(ball_xy=ball_xy, ball_speed=ball_speed, players_xy=dirty.players_xy,
                    players_speed=dirty.players_speed, team_of_slot=dirty.team_of_slot,
                    home_name=dirty.home_name, away_name=dirty.away_name)


def frames_to_restore(gaps, order, n_frames, n_needed):
    """Mask with whole gaps in `order` until reaching `n_needed` frames (the last, partial)."""
    mask = np.zeros(n_frames, dtype=bool)
    remaining = n_needed
    for index in order:
        if remaining <= 0:
            break
        start, stop = gaps[index]
        take = min(stop - start, remaining)
        mask[start:start + take] = True
        remaining -= take
    return mask


def curve_for_match(match_id, spec, config):
    clean, labels = idsse.load(match_id)
    labels, _ = convention.harmonise(labels)
    gt_codes = ev.pm.label_codes(np.asarray(labels, dtype="<U1"))
    dirty = degrade.apply(clean, spec, rng=np.random.default_rng(SEED))
    nearest = geometry.nearest_per_team(geometry.ball_distances(clean), clean.team_of_slot)
    nearest_player_m = np.nanmin(nearest, axis=1)
    gaps = hidden_gaps(dirty.ball_seen, clean.ball_seen)
    n = dirty.n_frames
    base_visibility = float(dirty.ball_seen.mean())

    def run(tracking):
        predicted, _ = harness.predict(tracking, config)
        result = ev.score(predicted, gt_codes=gt_codes)
        return result["spell_f1_mean"], result["macro_f1"]

    spell, macro = run(dirty)
    rows = [{"strategy": "base", "target": None, "visibility": base_visibility,
             "spell_f1": spell, "macro_f1": macro}]
    for strategy in STRATEGIES:
        order = recovery_order(gaps, strategy, nearest_player_m,
                               np.random.default_rng(SEED + 1))
        for target in TARGET_VISIBILITY:
            needed = int(round((target - base_visibility) * n))
            restored = restore_ball(dirty, clean, frames_to_restore(gaps, order, n, needed))
            spell, macro = run(restored)
            rows.append({"strategy": strategy, "target": target,
                         "visibility": float(restored.ball_seen.mean()),
                         "spell_f1": spell, "macro_f1": macro})
    for row in rows:
        log("%s  %-10s visible %.3f  spellF1 %.4f  macroF1 %.4f" % (
            match_id, row["strategy"], row["visibility"], row["spell_f1"], row["macro_f1"]))
    return rows


def main(config=None):
    config = config or DEFAULT
    started = time.time()
    spec = harness.real_spec()
    by_match = {m: curve_for_match(m, spec, config) for m in sorted(idsse.MATCHES)}

    log("=" * 70)
    summary = []
    base = [rows[0] for rows in by_match.values()]
    summary.append({"strategy": "base", "target": None,
                    "visibility": float(np.mean([r["visibility"] for r in base])),
                    "spell_f1": float(np.mean([r["spell_f1"] for r in base])),
                    "macro_f1": float(np.mean([r["macro_f1"] for r in base]))})
    for strategy in STRATEGIES:
        for target in TARGET_VISIBILITY:
            rows = [r for match_rows in by_match.values() for r in match_rows
                    if r["strategy"] == strategy and r["target"] == target]
            spell = [r["spell_f1"] for r in rows]
            summary.append({"strategy": strategy, "target": target,
                            "visibility": float(np.mean([r["visibility"] for r in rows])),
                            "spell_f1": float(np.mean(spell)),
                            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                            "matches_above_base": int(sum(
                                s > b["spell_f1"] for s, b in zip(spell, base)))})
    for row in summary:
        log("MEAN   %-10s visible %.3f  spellF1 %.4f  macroF1 %.4f" % (
            row["strategy"], row["visibility"], row["spell_f1"], row["macro_f1"]))
    path = os.path.join(REPO_ROOT, "out", "visibility.json")
    with open(path, "w") as handle:
        json.dump({"by_match": by_match, "summary": summary, "seed": SEED}, handle,
                  indent=2, default=float)
    log("saved %s (%.0f min)" % (path, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
