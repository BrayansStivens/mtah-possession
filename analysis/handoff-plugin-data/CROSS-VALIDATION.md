# Cross-validation of the recalibration procedure — 35 splits

**Date:** Oct 29, 2026 · **What was measured:** the *procedure* of `training/recalibrate.py`, not a
specific configuration. **How:** the same coordinate descent, from the same starting point, over
**all 35 possible splits** of 4 fit matches / 3 validation matches from the 7 IDSSE
matches. Each descent is measured on the 3 matches that descent did not see, and also on
degradation seeds 1 and 2, which none of them saw.

> **Added on Oct 31:** the grid was to be expanded on this report's advice. That is now done, and the
> result is that it **is no use**: it gains +0.62 pp on the fit set and −0.13 pp outside it. See the first
> section; it qualifies the finding that "the grid falls short" without invalidating the rest.

Scripts: `splits.py` (the 35 descents) and the consensus check. Data:
`data/splits_cruzados.json`, `data/consenso_splits.json`. Only `out/cache` was read (degraded
external matches): neither the ATD nor the public window was touched.

---

## Oct 31: expanding the grid is no use

**Verdict:** the grid did fall short, and fixing it **improves nothing outside the fit set**.
All the extra search capacity goes into overfitting. The grid in
`training/recalibrate.py` should not be expanded.

**Seven** parameters were expanded, not the five of the next section: when reviewing the stability table
two more appeared that also got pinned to an edge — `min_duration_frames` (50 is
its cap and is the mode, 17/35) and `ball_speed_smooth`, which gets pinned to the **minimum** of its list
(6 in 19/35). Each list was extended with its same geometric step (~×1.4) until the
value being chosen was left inside: 64 → 82 values. Nothing else changed — same starting point, the same 14
swept parameters, the same 2 passes and the same `descenso()` function, which now receives the
grid as an argument. Before launching it was checked that with the old grid it reproduces the original
split to the sixth decimal and chooses the same 14 values, so the parametrization moved nothing.

The two arms run over the same 35 splits from the same starting point, so the
comparison is **paired** split by split:

| axis | original grid | expanded | paired difference | 95% CI | wins in |
|---|---|---|---|---|---|
| on its own fit set (4) | +3.27 pp | +3.89 pp | **+0.62 pp** | [+0.34, +0.90] | 26/35 |
| **on its own validation set (3)** | +1.71 pp | +1.58 pp | **−0.13 pp** | [−0.52, +0.26] | 14/35 |
| seed 1, all 7 | +1.93 pp | +1.91 pp | −0.02 pp | [−0.34, +0.30] | 18/35 |
| seed 2, all 7 | +1.74 pp | +1.84 pp | +0.10 pp | [−0.18, +0.38] | 20/35 |
| transfer | 52% | **41%** | | | |

It is a clean result, and it is negative. On the fit set the improvement is real and significant: +0.62 pp with
the whole interval above zero. On the three matches the descent did not see, the interval crosses
zero and the improvement occurs in **14 of 35** splits, that is, fewer than half: indistinguishable from
a coin flip. The two degradation seeds say the same. Transfer drops from 52% to 41%,
which is exactly the signature of expanding the search space without more data: it fits the
four matches at hand better and not the problem better.

The diagnosis in the next section was correct —**no** parameter gets pinned to an edge anymore,
so the optimum on the fit set really was outside the old grid. What does not hold up
is the implication: that optimum does not generalize. And the newly reachable region is visibly noise.
`switch_from_dead` is now spread across **seven** distinct values (14:12, 8:6, 22:5, 31:4,
60:4, 4:2, 43:2) and `switch_to_dead` between 63:12 and 123:8, a range of almost double. Where before
there was an edge that looked like signal, there is a flat plateau.

On the original split, the one of `config.py`, the expanded grid reaches higher on the fit set
(+3.14 pp versus +2.72) and **lower** on validation (+2.07 pp versus +2.82). The same
pattern, on the only split that had been looked at before.

### What to take from this about `config.py`

With the expanded grid, the splits that choose the value that is in `config.py` today collapse:
`shape_smooth` 2/35, `activity_smooth` 3/35, `min_duration_frames` 5/35, `switch_turnover` 6/35,
`switch_to_dead` 6/35, `ball_speed_weight` 6/35. **That is not an argument for changing them.** If the
expanded arm does not win outside its fit set, what it chooses is not better; it is just different. The only
values that hold up in both arms are `control_weight` (34/35), `shape_weight` (32/35),
`control_smooth` (31/35) and `activity_weight` (27/35).

The `activity_smooth` = 12 matter stands as what it was: 1/35 with the old grid, 3/35 with the
expanded one, and the mode is never 12. That conclusion did not depend on the grid.

### What this comparison cannot say

*Where* to expand was decided by looking at the 35 splits of the previous day. That decision used only the
values the descent chose on the **fit** matches of each split, never validation
scores — but since every match enters the fit set of some split, there is a weak leak.
Given that the result is negative, the leak plays in favor of the expanded grid and it still
loses, so the verdict is not at risk from that side.

The real cost was **5 h 48 min** of CPU, not the hour I estimated: the new parameters include
large smoothing windows, which multiply the entries of the `Match._smoothed` cache and
make each evaluation more expensive. If someone repeats this with another grid, expect that order
of magnitude.

Scripts: `splits.py ampliada` (the new arm) and `compara_rejillas.py` (the paired table
above). Data: `data/splits_cruzados_ampliada.json`.

---

## Oct 31: `activity_smooth` goes back to 50 (applied)

The only change to `config.py` that has come out of all this. Measured with the three degradation seeds,
without touching anything else:

| | fit (4) | validation (3) | all 7 |
|---|---|---|---|
| seed 0 | −0.27 pp | **+1.14 pp** | +0.33 pp |
| seed 1 | +0.59 pp | **+1.40 pp** | +0.94 pp |
| seed 2 | +0.05 pp | **+0.81 pp** | +0.37 pp |
| mean | +0.12 pp | **+1.11 pp** | +0.55 pp |

All three seeds have the same sign on validation, and it improves on 5 of the 7 matches. On the
fit set the change is neutral (+0.12 pp), not the loss that the sweep documented in
`config.py` suggested — that sweep gave 12 an advantage of 0.0014 over 50, with a total
range of 0.14 pp between 6 and 50. It distinguished nothing.

Three independent measurements choose 50 and none chooses 12: the distribution over the 35 splits
(12 comes up 1/35, and 3/35 with the expanded grid; the mode is 50), the ablation above, and the
flatness of the sweep itself.

**Provenance, which matters here more than usual.** 50 coincides with the value there was before
Oct 28, and that one came from the public window, which `RULES.md` forbids as a fit set.
The coincidence is just that: the current 50 is justified with the IDSSE matches
and nothing else. Had the measurement given 25, it would be 25. It is written that way in `config.py`
so that nobody reads the change as a return to the old value.

Script: `activity_smooth.py`. Data: `data/activity_smooth.json`.

### Checked against the real match

The change was applied to `config.py` and the submission was regenerated with the ATD assets. The 55
tests pass and the CSV contract is valid. Two more readings, neither of which decided
anything — the decision had already been made with the IDSSE:

- **Dead-ball rate:** 45.8% → **44.9%** of the 158,857 scored frames, against the
  44.9% mean of the seven IDSSE matches. `activity_center_ms` = 1.26 was set by
  percentile on the *smoothed* signal, so it is coupled to `activity_smooth`: changing
  the window moves the rate. It moved it toward the target. If the smoothing is touched again,
  this rate has to be measured again.
- **5-minute public window:** spellF1 0.4875 → **0.4903**; macroF1 0.8108 → 0.8009.
  It is a reading, not a criterion: `RULES.md` forbids it as a fit set. It is looked at
  to check that the change breaks nothing there, and it does not.

### A misquoted number in `config.py`

The docstring said *"fit 0.3905, validation 0.3957"*. The 0.3905 was correct; the 0.3957
was not what the file produced. It came from the dump of `recalibrate.py`, which stores the three class
centers unrounded; with the centers rounded as they are in `config.py` the
validation gives **0.3968**. That is 0.001 of luck in the rounding, not a measurement error, but the
quoted number has to be the one the file produces. Corrected, and it now says what the file gives
as it stands: fit 0.3878, validation 0.4081, all seven 0.3965.

## Correction of the correction

On the morning of Oct 29 I wrote, in `VALIDATION.md`, that coordinate descent **"transfers
nothing"**. That is false, and the error is mine, a method error: I deduced it by comparing two fixed
configurations (`recalibrado` and `solo_centros`) on three seeds. That measures the difference
between two end points, which is noisy; it does not measure what the procedure gains. Running it 35
times:

| | mean | range | improves in |
|---|---|---|---|
| on its own fit set (4 matches) | **+3.27 pp** | +1.49 to +4.58 | 35/35 |
| **on its own validation set (3 matches)** | **+1.71 pp** | −0.58 to +3.51 | **32/35** |
| on seed 1, all 7 matches | +1.93 pp | +0.78 to +3.15 | 35/35 |
| on seed 2, all 7 matches | +1.74 pp | +0.20 to +2.66 | 35/35 |

95% CI of the mean on validation: **[+1.37, +2.05] pp**. The recalibration gains something real outside
where it was tuned. The transfer is **52%**: of every two points it buys on the fit set, one is lost
along the way. That is partial overfitting, which is normal and what should have been said from the
start, not "transfers nothing".

What remains true of the previous report: the 0.3927 in the technical document mixes fit and
validation and is not a prediction, and the original split turned out lucky — its validation delta
(+2.82 pp) is at the **83rd percentile** of the 35.

---

## The finding that matters: the grid falls short

> **Measured on Oct 31:** true, it falls short — but expanding it does not improve validation. The title
> of this section promises more than it ended up being worth. See "expanding the grid is no use" above.

In each descent it was recorded which value each hyperparameter chose. Five of them choose
**the maximum of their own grid** in most splits, which means the search is hitting the
edge and the optimum lies beyond:

| parameter | picks the cap | in | `config.py` has |
|---|---|---|---|
| `switch_turnover` | 40 | 30/35 | 40 ✅ |
| `ball_speed_weight` | 1.2 | 28/35 | **0.6** |
| `switch_to_dead` | 45 | 28/35 | **34** |
| `switch_from_dead` | 22 | 25/35 | **14** |
| `shape_smooth` | 200 | 22/35 | 200 ✅ |

`config.py` documents `switch_to_dead` and `switch_from_dead` as *"Confirmed"* because on the
original split the sweep did not move them. In 28 and 25 of the 35 splits it does move them, and to the cap. The three
values in bold are candidates to be wrong by construction of the sweep, not by bad measurement.

## Stability per parameter

How many of the 35 splits choose the value that is in `config.py` today:

| parameter | matches | distribution of what was chosen |
|---|---|---|
| `switch_turnover` 40 | **30/35** | 40:30 · 28:4 · 20:1 |
| `control_smooth` 50 | **28/35** | 50:28 · 100:7 |
| `control_weight` 6.0 | **27/35** | 6:27 · 9:4 · 13:3 · 4:1 |
| `shape_smooth` 200 | 22/35 | 200:22 · 100:6 · 25:5 · 12:2 |
| `shape_weight` 1.6 | 22/35 | 1.6:22 · 1.0:12 · 0.5:1 |
| `activity_weight` 1.0 | 22/35 | 1.0:22 · 0.5:13 |
| `ball_speed_smooth` 6 | 19/35 | 6:19 · 12:11 · 25:3 · 50:2 |
| `min_duration_frames` 12 | 12/35 | 50:17 · 12:12 · 0:4 · 25:2 |
| `activity_center_ms` 1.26 | 12/35 | 1.18:14 · 1.26:12 · 1.10:7 · 1.35:2 |
| `ball_unseen_logit` 0.5 | 11/35 | 0.0:12 · 0.25:12 · 0.5:11 |
| `switch_from_dead` 14 | 8/35 | 22:25 · 14:8 · 8:2 |
| `ball_speed_weight` 0.6 | 7/35 | 1.2:28 · 0.6:7 |
| `switch_to_dead` 34 | 6/35 | 45:28 · 34:6 · 26:1 |
| **`activity_smooth` 12** | **1/35** | 50:14 · 6:8 · 100:8 · 25:4 · **12:1** |

Three groups:

**Truly confirmed.** `switch_turnover` 40, `control_smooth` 50 and `control_weight` 6.0 are
chosen by the descent in ~80% of the splits. They are the three that the ablation in the previous report already
pointed to as the ones that transfer, and the first two also have mechanical justification written
in `config.py`. Nothing to touch.

**`activity_smooth` = 12 has no backing at all: 1 of 35.** The mode is 50, the value it had
before Oct 28. And it is consistent with everything else: the sweep documented in `config.py` itself
(6:.3754 12:.3761 25:.3755 50:.3747) has a range of 0.14 pp, that is, nothing, and the ablation
measured +1.14 pp on validation when returning it to 50. Three independent signals pointing to the same
place: **that 12 is grid noise**.

**Indifferent.** `ball_unseen_logit` splits 12/12/11 between 0, 0.25 and 0.5: the parameter does
nothing, exactly what `config.py` itself reasons when explaining that adding it counts the
evidence twice. `activity_center_ms` splits between 1.10/1.18/1.26, but here we must keep
1.26 **anyway**, because it does not come from a sweep but from matching the dead-ball rate
to that of full matches, which is an argument that does not depend on this measurement.

## What if the modal values are taken?

The "consensus config" (the most-voted value of each parameter across the 35 splits) scores, on the
two degradation seeds the descent did not see, over the 7 matches:

| | start | `config.py` | consensus |
|---|---|---|---|
| seed 1 | 0.3778 | 0.3793 | **0.3935** (+1.42 pp) |
| seed 2 | 0.3996 | 0.4092 | **0.4171** (+0.79 pp) |
| seed 0 *(seen)* | 0.3797 | 0.3932 | 0.4018 |

It wins on both unseen seeds. **But it is not a clean result and must not be submitted as such:**
the modal values were chosen by looking at the 7 matches, so seeds 1 and 2 control for degradation
noise but **not** for the choice of matches — they are the same 7. As an estimate of what it would
deliver on the competition match it is not valid.

## What to do

1. ~~**Expand the grid**~~ **DONE on Oct 31, and the answer is no.** Seven
   parameters were expanded (64 → 82 values): +0.62 pp on the fit set, −0.13 pp on validation, 14/35. Do not
   expand the grid in `recalibrate.py`, and do not change `switch_to_dead`, `switch_from_dead` or
   `ball_speed_weight` to what the expanded arm chooses. Details in the first section. What
   remains open is not the grid: it is that with 7 fit matches there is no signal to distinguish
   those values, and searching more does not fix that.
2. **`activity_smooth`:** 1/35 is enough to stop defending the 12. But the replacement
   value has to be justified with the full cross-validation, not with this column.
3. **In the technical document**, replace "it did not overfit" with the number that can be defended:
   *the recalibration gains +1.7 pp (95% CI [+1.4, +2.1]) on matches that were not used
   to tune it, over 35 partitions; on the fit matches it gains +3.3 pp, so roughly
   half of the gain is overfitting.*

## Untested coverage

The 7 matches are all Bundesliga with the same tracking, degraded to resemble the ATD; none
of this measures the jump to the league, camera and tracker of the competition match. Seeds
1 and 2 vary the degradation noise, not the matches. Neither the ATD nor the public window has
been touched, and no class center has been re-derived: the three are still the IDSSE medians
over the 4 fit matches of the original split.
