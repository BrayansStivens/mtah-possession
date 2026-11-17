"""Degrade clean tracking until it looks like the challenge's automatic ATD.

**Why it exists.** Two uses, and the second matters more than the first:

1. Training. A model fitted on corrected tracking learns to lean on signals that do not
   exist at inference time -- ball always visible, stable identity, 22 objects.
2. **Measuring.** The public ground truth is 5 minutes and 17 possessions, and EVALUATION.md
   says it is no use for choosing between two versions of the model. Degrading an IDSSE
   match, which does have a per-frame label for the full match, gives a spellF1 over an
   entire match. Seven times over. That turns "we cannot measure" into "we can measure on
   seven proxy matches", and right now that is the biggest risk the project has.

**The principle.** No parameter here is made up: they all come out of `estimate()`, which
measures them on the real ATD of the match **without using any label**. That is legitimate
-- it does not touch the private ground truth, only the marginal distribution of the input
data, which is published -- and it is what separates this from eyeballed domain
randomization.

**It is applied at load time, with a different seed each time.** A degraded and frozen
corpus has a fixed occlusion pattern that a model learns to memorize.

What this version does NOT do, and worth keeping in mind when reading a number that comes
out of here:

- **Ball-absent runs are placed at random.** In the real ATD they concentrate in crowded
  areas and aerial duels (DATA.md), that is, in the hard frames. By spreading them
  uniformly, the proxy ends up **optimistic**: it loses the ball at easier moments than
  the real tracker does.
- **It does not model identity swaps between slots**, only fragmentation. A slot that
  changes person in the middle of a segment is an error that does not happen here.
- Position noise is homogeneous AR(1), with no dependence on distance to the camera.
"""

import numpy as np

from ..io.atd import HOME_CODE, AWAY_CODE, Tracking

# Slot split of the tactical ATD, from DATA.md: 46 = 24 home + 22 away.
N_SLOTS_HOME, N_SLOTS_AWAY = 24, 22


class Spec:
    """Degradation parameters measured on a real tracking. They are all empirical samples,
    not fitted distributions: resampling from reality does not require choosing a
    parametric family or getting it right."""

    __slots__ = ("ball_absent_runs", "ball_present_runs", "slot_runs",
                 "n_tracked_values", "n_tracked_probs", "step_sigma_m",
                 "n_tracked_unseen_values", "n_tracked_unseen_probs")

    def __init__(self, ball_absent_runs, ball_present_runs, slot_runs,
                 n_tracked_values, n_tracked_probs, step_sigma_m,
                 n_tracked_unseen_values=None, n_tracked_unseen_probs=None):
        self.ball_absent_runs = ball_absent_runs
        self.ball_present_runs = ball_present_runs
        self.slot_runs = slot_runs
        self.n_tracked_values = n_tracked_values
        self.n_tracked_probs = n_tracked_probs
        self.step_sigma_m = step_sigma_m
        # Histogram of objects only in the ball-absent frames. `thin_where_ball_unseen` uses
        # it; `apply` does not look at it, so the usual degradation does not change.
        self.n_tracked_unseen_values = n_tracked_unseen_values
        self.n_tracked_unseen_probs = n_tracked_unseen_probs

    def summary(self):
        a, p = self.ball_absent_runs, self.ball_present_runs
        return {
            "rachas_sin_balon": len(a),
            "media_sin_balon_fr": float(a.mean()) if len(a) else 0.0,
            "max_sin_balon_fr": int(a.max()) if len(a) else 0,
            "fraccion_sin_balon": float(a.sum() / (a.sum() + p.sum())) if len(a) else 0.0,
            "tramos_de_slot": len(self.slot_runs),
            "media_tramo_fr": float(self.slot_runs.mean()) if len(self.slot_runs) else 0.0,
            "objetos_media": float((self.n_tracked_values * self.n_tracked_probs).sum()),
            "objetos_media_sin_balon": (
                float((self.n_tracked_unseen_values * self.n_tracked_unseen_probs).sum())
                if self.n_tracked_unseen_values is not None else None),
            "step_sigma_m": float(self.step_sigma_m),
        }


def _runs(mask):
    """Lengths of the runs of True in a boolean array."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return np.array(out, dtype=np.int32)


def estimate(tracking, blocks):
    """Measure the parameters on a real tracking. It uses no labels anywhere."""
    n = tracking.n_frames
    graded = np.zeros(n, dtype=bool)
    for a, b in blocks:
        graded[a:b] = True

    seen = tracking.ball_seen[graded]
    visible = ~np.isnan(tracking.players_xy[:, :, 0])

    # Continuous segments of each slot: how long a track lasts before breaking.
    slot_runs = [r for s in range(visible.shape[1]) for r in _runs(visible[graded, s])]

    n_tracked = visible[graded].sum(axis=1)
    values, counts = np.unique(n_tracked, return_counts=True)
    # The same histogram only where the ball is missing: in the ATD players are missing
    # right there (19.2 on average without the ball against 21.4 with it, STATUS.md §6).
    unseen_values = unseen_probs = None
    if (~seen).any():
        unseen_values, unseen_counts = np.unique(n_tracked[~seen], return_counts=True)
        unseen_values = unseen_values.astype(np.int32)
        unseen_probs = (unseen_counts / unseen_counts.sum()).astype(np.float64)

    # Position noise: standard deviation of the step between consecutive frames, which in a
    # clean tracking is real movement and in a noisy one is movement plus jitter. It is used
    # as an upper bound on the noise to inject, not as its exact value.
    step = np.diff(tracking.players_xy[:, :, 0], axis=0)
    step = step[np.isfinite(step)]
    step_sigma = float(np.std(step)) if step.size else 0.0

    return Spec(
        ball_absent_runs=_runs(~seen),
        ball_present_runs=_runs(seen),
        slot_runs=np.array(slot_runs, dtype=np.int32),
        n_tracked_values=values.astype(np.int32),
        n_tracked_probs=(counts / counts.sum()).astype(np.float64),
        step_sigma_m=step_sigma,
        n_tracked_unseen_values=unseen_values,
        n_tracked_unseen_probs=unseen_probs,
    )


def _alternating_mask(n, present_runs, absent_runs, rng, start_present=True):
    """Mask of n frames alternating runs resampled from the two distributions.

    Resampling whole lengths is the difference between a 2-frame gap and a 300-frame one.
    An i.i.d. Bernoulli with the same marginal rate gives runs averaging 1.7 frames; the
    real ATD has median 268 (10.7 s) and maximum 5537 (3.7 min).
    """
    target_absent = absent_runs.sum() / float(absent_runs.sum() + present_runs.sum())

    def draw(start):
        mask = np.empty(n, dtype=bool)
        i, present = 0, start
        while i < n:
            pool = present_runs if present else absent_runs
            length = int(rng.choice(pool)) if len(pool) else 1
            mask[i:i + length] = present
            i += length
            present = not present
        return mask

    # The runs have a very long tail (max 5537 frames) and only ~140 fit in a match, so a
    # single draw easily strays 5-7 points from the target rate. It is redrawn until it gets
    # close, instead of trimming the runs -- which is exactly what cannot be touched, because
    # run length is the parameter that matters.
    best, best_err = None, float("inf")
    for _ in range(40):
        mask = draw(start_present)
        err = abs((~mask).mean() - target_absent)
        if err < best_err:
            best, best_err = mask, err
        if err <= 0.01:
            break
        start_present = bool(rng.integers(2))
    return best


def _weighted_mask(n, present_runs, absent_runs, weights, rng):
    """Presence mask with the ball-absent runs centered where `weights` is heavy.

    Same run lengths and same ball-absent fraction as `_alternating_mask`; only where they
    fall changes. It is for stress tests: put the gaps where the real tracker puts them (or
    worse) and see whether what was measured with random gaps holds up. Two runs that land
    on top of each other merge into a longer one, so the number of runs goes down: this is
    reported, not corrected, because correcting it would require moving gaps away from where
    the weight asks for them.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any() \
            or weights.sum() <= 0:
        raise ValueError("gap_weights must be ({},) finite, >= 0 and not all zero".format(n))
    target = int(round(n * absent_runs.sum() / float(absent_runs.sum() + present_runs.sum())))
    probability = weights / weights.sum()
    absent = np.zeros(n, dtype=bool)
    missing = target
    while missing > 0:
        length = min(int(rng.choice(absent_runs)), missing, n)
        center = int(rng.choice(n, p=probability))
        start = max(0, min(center - length // 2, n - length))
        absent[start:start + length] = True
        missing = target - int(absent.sum())
    return ~absent


def _reassign_slots(players_xy, players_speed, team_of_slot, spec, rng):
    """Chop up each player and distribute the segments among the 46 identity-less slots.

    A slot is not a person: it is an object with a lifetime. When a segment ends the slot
    becomes free and can later receive another player, which is exactly what the tracker
    does when it loses someone and reacquires another.

    The distribution is done **in global temporal order**, not player by player. Doing it
    per player looks equivalent and is not: the first one processed occupies slots until the
    end of the match and the following ones are left without room from frame 0. The first
    version of this had that flaw and gave 6 objects per frame instead of 20.5.
    """
    n = players_xy.shape[0]
    n_out = N_SLOTS_HOME + N_SLOTS_AWAY
    out_xy = np.full((n, n_out, 2), np.nan, dtype=np.float32)
    out_sp = np.full((n, n_out), np.nan, dtype=np.float32)
    target_team = np.array([HOME_CODE] * N_SLOTS_HOME + [AWAY_CODE] * N_SLOTS_AWAY,
                           dtype=np.int8)

    for team in (HOME_CODE, AWAY_CODE):
        src = np.flatnonzero(team_of_slot == team)
        dst = np.flatnonzero(target_team == team)

        # 1) All segments of all the team's players, chopped up.
        pieces = []
        for s in src:
            present = np.isfinite(players_xy[:, s, 0])
            i = 0
            while i < n:
                if not present[i]:
                    i += 1
                    continue
                end = i
                while end < n and present[end]:
                    end += 1
                while i < end:
                    length = int(rng.choice(spec.slot_runs)) if len(spec.slot_runs) else end - i
                    stop = min(i + max(length, 1), end)
                    pieces.append((i, stop, s))
                    i = stop
        # 2) In start order, which is what makes the distribution correct.
        pieces.sort()

        busy = np.zeros(len(dst), dtype=np.int64)
        for begin, stop, s in pieces:
            free = np.flatnonzero(busy <= begin)
            if not len(free):
                continue            # with no free slot the segment is lost, as in the ATD
            k = int(free[rng.integers(len(free))])
            slot = dst[k]
            out_xy[begin:stop, slot] = players_xy[begin:stop, s]
            out_sp[begin:stop, slot] = players_speed[begin:stop, s]
            busy[k] = stop
    return out_xy, out_sp, target_team


def _thin_to_histogram(players_xy, players_speed, spec, rng):
    """Drop objects per frame until matching the real ATD's histogram of objects.

    An object cannot be invented where there is none, so this only removes. That is why it
    is applied after fragmentation, which has already removed some on its own.
    """
    n = players_xy.shape[0]
    target = rng.choice(spec.n_tracked_values, size=n, p=spec.n_tracked_probs)
    for i in range(n):
        live = np.flatnonzero(np.isfinite(players_xy[i, :, 0]))
        excess = len(live) - int(target[i])
        if excess > 0:
            drop = rng.choice(live, size=excess, replace=False)
            players_xy[i, drop] = np.nan
            players_speed[i, drop] = np.nan
    return players_xy, players_speed


def thin_where_ball_unseen(tracking, spec, rng, persistent=False):
    """New tracking with fewer players **only in the ball-absent frames**, as in the ATD.

    `apply` matches the ATD's histogram of objects over the whole match, but does so the same
    with and without the ball; in the ATD players are missing right when the ball is missing.
    This is applied after `apply` and touches nothing else: same ball, same players in the
    frames with the ball, and in the ball-absent frames it only removes.

    The target number for each ball-absent frame comes from matching quantiles: the frame
    with the fewest players gets the lowest target of the ball-absent histogram, and so on in
    order. That way the final distribution is the ATD's (except where there were already
    fewer, which stays as it is), instead of the minimum of two independent draws, which
    would remove too many. The ones removed are chosen at random, as in `_thin_to_histogram`.

    `persistent=False` draws the ones removed in each frame: they flicker, and the 1 to 8 s
    windows of the team signals average out that noise. `persistent=True` draws once per
    ball-absent run: the same slots disappear for the whole run, which is how the ATD loses
    a track (segments of ~2700 frames on average). Same histogram per frame.
    """
    if spec.n_tracked_unseen_values is None:
        raise ValueError("the spec has no ball-absent histogram; measure it again with estimate()")
    xy = tracking.players_xy.copy()
    sp = tracking.players_speed.copy()
    frames = np.flatnonzero(~tracking.ball_seen)
    if len(frames):
        live = np.isfinite(xy[frames, :, 0])
        count = live.sum(axis=1)
        target_sorted = np.sort(rng.choice(spec.n_tracked_unseen_values, size=len(frames),
                                           p=spec.n_tracked_unseen_probs))
        order = np.lexsort((rng.random(len(frames)), count))
        target = np.empty(len(frames), dtype=np.int64)
        target[order] = target_sorted
        excess = np.maximum(count - target, 0)
        if persistent:
            run_id = np.cumsum(np.diff(frames, prepend=frames[0] - 2) > 1) - 1
            keys = rng.random((int(run_id[-1]) + 1, live.shape[1]))[run_id]
        else:
            keys = rng.random(live.shape)
        keys[~live] = np.inf
        rank = np.argsort(np.argsort(keys, axis=1), axis=1)
        drop = rank < excess[:, None]
        rows, slots = np.nonzero(drop)
        xy[frames[rows], slots] = np.nan
        sp[frames[rows], slots] = np.nan
    return Tracking(ball_xy=tracking.ball_xy, ball_speed=tracking.ball_speed, players_xy=xy,
                    players_speed=sp, team_of_slot=tracking.team_of_slot,
                    home_name=tracking.home_name, away_name=tracking.away_name)


def apply(tracking, spec, rng=None, position_noise=True, rho=0.9, gap_weights=None):
    """Return a new, degraded Tracking. It does not modify the original.

    `gap_weights=None` places the ball-absent runs at random (the usual proxy); a per-frame
    array centers them with that relative probability (see `_weighted_mask`).
    """
    rng = rng or np.random.default_rng()
    n = tracking.n_frames

    xy, sp, team_of_slot = _reassign_slots(
        tracking.players_xy, tracking.players_speed, tracking.team_of_slot, spec, rng)
    xy, sp = _thin_to_histogram(xy, sp, spec, rng)

    if position_noise and spec.step_sigma_m > 0:
        # AR(1) and not white noise: white noise is filtered out with a moving average and
        # the model learns to smooth instead of to distrust.
        sigma = 0.25 * spec.step_sigma_m
        noise = np.zeros((n, xy.shape[1], 2), dtype=np.float32)
        step = rng.normal(0.0, sigma, size=noise.shape).astype(np.float32)
        noise[0] = step[0]
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + step[i]
        xy = xy + noise

    ball_xy = tracking.ball_xy.copy()
    ball_speed = tracking.ball_speed.copy()
    if gap_weights is None:
        seen = _alternating_mask(n, spec.ball_present_runs, spec.ball_absent_runs, rng,
                                 start_present=bool(rng.integers(2)))
    else:
        seen = _weighted_mask(n, spec.ball_present_runs, spec.ball_absent_runs, gap_weights,
                              rng)
    ball_xy[~seen] = np.nan
    ball_speed[~seen] = np.nan

    return Tracking(ball_xy=ball_xy, ball_speed=ball_speed, players_xy=xy,
                    players_speed=sp, team_of_slot=team_of_slot,
                    home_name=tracking.home_name, away_name=tracking.away_name)
