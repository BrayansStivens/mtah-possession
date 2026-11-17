# Validation of the spellF1 analysis — current vs recalibrated

**Date:** Oct 28, 2026 · **Scope:** the 5 files in `handoff-plugin-data/datos/`.
**Method:** read-only; all numbers were recalculated independently from the CSV/JSON files
(script: `verificar.py`, in the session scratchpad). Nothing in `data/` was touched.

## Verdict: **share with caveats, and correct two claims before sending it**

> **Oct 29, later:** the claim that the descent "transfers nothing" **is false**;
> running the procedure over the 35 splits gains +1.7 pp outside its fit set. See
> [CROSS-VALIDATION.md](CROSS-VALIDATION.md), which supersedes the correction section
> below on everything concerning overfitting.
>
> **Oct 29:** finding 1 was measured again with the three degradation seeds and its
> recommendation to change config is **withdrawn**. Read *Correction of Oct 29* before
> finding 1. The rest of the report stands.

The package's calculations are correct: the JSON aggregates exactly reproduce the mean of the per-match
CSVs, the rows of the confusion matrix sum to 100%, the `n` values match across tables
(1,002,644 frames, 1,945 real possessions) and the recalibrated centers are indeed the midpoint
of the per-class medians. There are no calculation errors.

---

## Correction of Oct 29, 2026 — finding 1 does not survive its own check

When trying to bring `solo_centros` into `config.py`, the IDSSE harness was measured again, this time with the
**three degradation seeds** that exist in `out/cache` instead of just one. Two things change:

**a) `solo_centros` is not "config.py without the extra tuning".** The script that generated `recalibracion.json`
defines it as `merged(**centers)` on top of the `DEFAULT` **of that moment**, that is, the `config.py`
from before Oct 28: smoothing windows ±50, `switch_turnover` 28 and the centers of the public
window, with only the three new centers on top. That file **is not preserved** (the repo has
no commits at all), so `solo_centros` cannot be reproduced exactly. Rebuilt from the provenance
documented in `config.py` it comes out at −0.3 pp on the fit set and −0.02 pp on the validation set relative to the
published values — close, but not the same.

**b) the −0.22 pp advantage is within seed-to-seed noise.** spellF1 delta
(`recalibrado` − `solo_centros`), measuring both configs with the same code and the same seed:

| seed | fit (4) | validation (3) |
|---|---|---|
| 0 | +1.91 pp | +0.60 pp |
| 1 | +0.59 pp | −0.44 pp |
| 2 | +2.13 pp | −0.61 pp |
| **mean** | **+1.54 pp** | **−0.15 pp** |

The sign on validation changes with the seed and the range is 1.2 pp: seven times the −0.22 pp
the recommendation rested on. The bias of the reconstruction is a
constant offset and does not affect that range.

**What does remain demonstrated, and more firmly than before:** on the fit set the recalibration wins on all three
seeds (+0.6 to +2.1 pp) and on validation it wins on none (mean −0.15 pp). Coordinate
descent buys points where it was tuned and **transfers nothing**. Finding 2 stands; what falls is
the corollary that `solo_centros` is better. The two are indistinguishable outside the fit set.

**What to do about this:** do not change the submission config. There is no evidence to prefer either
of the two, and `config.py` has the provenance of each value written next to it, which is what
RULES.md requires. What has to be corrected is the **narrative**: the recalibration is worth +1.5 pp where it was tuned and
0 pp outside.

### Where the gain comes from, parameter by parameter

Reverting **a single value** of `config.py` to what it was before Oct 28 (seed 0, everything else the same):

| reverted | Δ fit | Δ validation |
|---|---|---|
| `control_center` 0.008 → 0.096 | −0.06 pp | **−1.55 pp** |
| `switch_turnover` 40 → 28 | −0.37 pp | **−1.27 pp** |
| `shape_smooth` 200 → 50 | −0.69 pp | −0.63 pp |
| `ball_speed_center` 3.14 → 4.10 | −0.52 pp | +0.06 pp |
| `ball_speed_smooth` 6 → 50 | −1.10 pp | +0.33 pp |
| `activity_smooth` 12 → 50 | −0.27 pp | **+1.14 pp** |
| `shape_center` 0.976 → 0.975 | 0.00 pp | 0.00 pp |

It separates cleanly into two groups. What **transfers** are the two values with mechanical
justification: the control center (symmetric between teams by construction) and the ball turnover
cost (which `config.py` already documents as a plateau above 40, not a peak). What does **not
transfer** is `activity_smooth` = 12: it costs 1.1 pp on validation and only buys 0.27 pp on the fit set
— it is noise picked by grid. `ball_speed_smooth` = 6 goes in the same direction, more mildly.
`shape_center` is indistinguishable: rounding to three decimals does not move a single frame.

Method warning: that last table is measured **on the validation set**. It serves to
diagnose where the overfitting comes from; using it to choose `activity_smooth` = 50 would be making
the same mistake again, with the last clean set left. If you want to decide that properly,
the 7 matches have to be re-split and re-derived, not read off this column.

Measured with `handoff-plugin-data/semillas.py` and `reeval.py`; numbers in
`datos/robustez_semillas.json`. Only the harness was re-run over `out/cache` (degraded external
matches): neither the ATD nor the public window was touched.

---

What does not hold up is the reading: **"0.3767 → 0.3927 and it did not overfit" is too optimistic, and
the configuration the package recommends is not the one that generalizes best.**

---

## Findings

### 1. [HIGH] `solo_centros` beats `recalibrado` on the validation matches

It is in `recalibracion.json`, in the `scores` block, and the package does not mention it. spellF1:

| config | fit (4) | validation (3) | all 7 |
|---|---|---|---|
| `original_pre_sesion` | 0.3110 | 0.3272 | 0.3179 |
| `actual_config_py` | 0.3714 | 0.3837 | 0.3767 |
| `solo_centros` | 0.3747 | **0.3979** | 0.3847 |
| `recalibrado` | **0.3905** | 0.3957 | 0.3927 |

Read by columns: re-centering and nothing else gains +1.42 pp on validation; adding the rest of the
recalibration on top gains +1.58 pp **on the fit set** and loses **−0.22 pp on validation**. This is the
classic signature of overfitting: the extra parameters buy points where they were fitted and do not transfer them.

The fit−validation gap confirms it: −0.0232 with `solo_centros` → −0.0052 with `recalibrado`. The
fit set "catches up" with the validation set, which is what happens when you fit on it.

**Practical consequence** — *corrected on Oct 29, see the correction section*: the
−0.22 pp advantage is a single degradation seed and changes sign with the other two. **There is no evidence
to change config.** What must be said is that the advantage of `recalibrado` over
`solo_centros` is not demonstrated outside the fit set: there they are indistinguishable.

### 2. [HIGH] "It did not overfit" is not what the data say

The improvement is +1.91 pp on the fit set and +1.19 pp on validation: it transfers 62%, not 100%. And one of
the three validation matches **gets worse** (J03WPY, −0.0035).

Per-match deltas (recalibrated − current):

| match | group | current | recalibrated | Δ |
|---|---|---|---|---|
| J03WMX | fit | 0.3556 | 0.3606 | +0.0050 |
| J03WN1 | fit | 0.3227 | 0.3508 | +0.0281 |
| J03WOH | fit | 0.4515 | 0.4570 | +0.0055 |
| J03WOY | fit | 0.3556 | 0.3936 | +0.0380 |
| J03WPY | validation | 0.3824 | 0.3789 | **−0.0035** |
| J03WQQ | validation | 0.4210 | 0.4357 | +0.0147 |
| J03WR9 | validation | 0.3478 | 0.3724 | +0.0246 |

The two big jumps (+0.028 and +0.038) both fall in the fit set.

Significance with n=7: paired t p=0.029, Wilcoxon p=0.031, but sign test p=0.125 (6/7).
95% CI of the mean delta: **[+0.002, +0.030]** — the interval grazes zero. With the 3 validation matches
alone (2/3 positive, p=1.0) nothing can be claimed statistically.

**Honest wording:** "+1.2 pp on 3 matches not used for tuning, with 1 of the 3 going against; the
sample does not allow ruling out that part of the jump is noise."

### 3. [HIGH] The validation set is easier than the fit set, so 0.3957 is not a prediction

In **all four** configs, validation scores above fit (between +0.5 and +2.3 pp). It is a
property of those 3 matches, not of the model. Quoting 0.3957 as "what we expect in the competition"
inherits that favorable bias. On top of that, the 7 matches are Bundesliga, with a different tracking
provider and a different camera than the competition match: **none of these numbers is a
prediction for the submission**, only evidence that the margin over the 0.2433 minimum is wide
(the worst match, 0.3227, exceeds it by +33%).

### 4. [MEDIUM] There is aggregation bias in categories with few frames

`error_por_racha_sin_balon`, which is the table where the most "improvement" shows:

| category | n | weight | current error | recalibrated | Δ pp | frames saved |
|---|---|---|---|---|---|---|
| 1-25 fr | 512 | 0.05% | 11.52% | 4.10% | −7.42 | **38** |
| 26-125 fr | 854 | 0.09% | 20.73% | 19.56% | −1.17 | 10 |
| 126-500 fr | 131,727 | 13.1% | 26.58% | 26.23% | −0.35 | 461 |
| >500 fr | 276,897 | 27.6% | 27.69% | 27.39% | −0.30 | 831 |
| ball visible | 592,654 | 59.1% | 15.27% | 14.27% | −1.00 | 5,927 |

The 1-25 fr bin is 0.05% of the data: its −7.42 pp is **38 frames**, pure noise. Drawn
as five equal-size bars, the chart suggests an average improvement of −2.05 pp; weighting by
frames the real improvement is **−0.72 pp**. Almost all of the effect (82%) comes from the "ball visible" bin.

→ In the dashboard, the two small bins are labeled with their `n` and marked as unreliable.

Same warning in the opposite direction: in `error_por_distancia_a_transicion`, the largest bin (>250 fr,
291,549 frames, 29% of the total) **gets worse** 7.22% → 7.48%. It is not in the package summary.

### 5. [MEDIUM] The gain is almost all precision, and recall has a ceiling

Decomposing spellF1 from the recall-by-duration table and the predicted possessions:

| config | hits | recall | precision | reconstructed F1 |
|---|---|---|---|---|
| current | ~644 / 1,945 | 0.331 | 0.414 | 0.368 |
| recalibrated | ~660 / 1,945 | 0.339 | 0.458 | 0.390 |

(it reconstructs 0.368 and 0.390 versus the official 0.3767 and 0.3927 — the difference is that the official one
averages F1 per match instead of pooling; consistent.)

Recall rises +0.8 pp and precision +4.4 pp: the recalibration does not hit more possessions, it
emits fewer (1,555 → 1,441 predicted, versus 1,945 real) and longer ones (mean duration 14.1 s → 15.2 s
versus 11.8 s real). It improves the metric by moving away from the real structure of the match.

**The ceiling this imposes:** with recall 0.339, even with perfect precision spellF1 ≈ **0.507**. The
possessions shorter than 5 s are 764 of 1,945 (**39%**) and are recovered at 9%. As long as that does not
change, further tuning of thresholds yields less and less. And the mechanism is visible in the config itself:
`min_duration_frames: 12` (0.48 s) and `switch_turnover: 40` (1.6 s) make it structurally impossible
to emit the short possessions. That is the big lever, not the centers.

### 6. [MEDIUM] The recalibration does not reduce H/A confusion: it rebalances it toward H

| | current | recalibrated |
|---|---|---|
| true H → pred H | 66.3% | 72.8% |
| true A → pred A | 73.2% | **67.9%** |
| H↔A over the total | 11.76% | 10.90% |

It gains 6.5 pp on H and loses 5.4 pp on A; net −8,630 frames (−0.9 pp). The cause is direct: the
`control` and `shape` centers are the midpoint of the **H and A medians measured on the 4 fit
matches** (verified: 0.008298 and 0.975781 are exactly those midpoints). That H/A asymmetry
gets baked into the config. If in the competition match the H/A split is different, the bias
works against us. The package does not include the spread of those medians, so it cannot be known whether the
classes are well separated or overlap — with 20% H/A confusion, they overlap a lot.

### 7. [LOW] Informational observations, not actionable

- **Mean of means:** the aggregates average spellF1 per match unweighted. Weighting by
  real possessions the difference is ≤ 0.25 pp in all cases. Checked and with no impact.
- **`n_frames_o_posesiones` mixes two meanings:** in three tables it is the size of the category
  (identical across configs, which makes the comparison fair); in the confusion matrix it is the count
  of the cell, which does vary. The rows sum the same (292,078 / 261,027 / 449,539), so it is
  consistent, but it should be clarified in the legend.
- **The 5-min window is not representative:** 19.6% D versus 44.8% in the Bundesliga
  matches. It is 5 min of continuous play after a restart. The per-frame agreement there is 81.0% and the
  model under-predicts D (13.0% vs 19.6% real). Do not extrapolate from that window.
- **The halves of the full match last 53 min each** (3,178 s and 3,175 s; 105.9 min in total,
  versus 95.5 min average per match in the external set). It probably includes padding before the
  kick-off or the break, which would be labeled D and inflates the 44.8% of D. It deserves a check of
  whether the submission window covers those minutes.

---

## Checks done, and their result

| check | result |
|---|---|
| JSON aggregates = mean of the CSVs | ✅ exact in all 6 cases (tune/hold/all × 2 configs) |
| Confusion matrix rows sum to 100% | ✅ (99.99–100.00 due to rounding) |
| Frame totals consistent across tables | ✅ 1,002,644 in the three frame tables |
| Possessions in the recall table = sum of `posesiones_reales` | ✅ 1,945 |
| Category `n` identical across configs (fair comparison) | ✅ in the 3 tables where it must be |
| Centers = midpoint of the declared medians | ✅ control, shape and ball_speed |
| spellF1 reconstructed from precision/recall | ✅ 0.368 / 0.390 vs 0.377 / 0.393 (consistent) |
| Public window: 7,500 rows at 25 fps = 300 s | ✅ |
| All matches exceed the 0.2433 minimum | ✅ the worst is 0.3227 |
| Simple vs weighted mean | ✅ difference ≤ 0.25 pp, immaterial |

---

## What to change before sharing it

1. **Replace "it did not overfit"** with the breakdown from finding 2 (+1.91 pp fit / +1.19 pp
   validation, 1 of 3 against, CI [+0.002, +0.030]).
2. **Add the `solo_centros` row** to the results table and say which one goes into the submission.
   It is the product decision that comes out of this validation.
3. **Stop quoting 0.3957 as an expectation** for the competition match: validation is easier
   than fit in all four configs, and on top of that it is another league and another tracking.
4. **Put the `n` on all diagnostic charts** and flag the bins of 512 and 854 frames.
5. **Mention the 0.507 ceiling** and the 39% of short possessions as the limit to further tuning.

## What could not be evaluated with this package

- **There is no ground truth for the competition match** (it is private), so there is no way to measure
  there: everything above is transfer from Bundesliga.
- **`ventana_publica_5min.csv` and `partido_completo_1hz.csv` only carry `modelo_actual`.** There is no
  recalibrated column, so the time views cannot be compared across configs.
- **There is no spread** (IQR/deviation) of the per-class medians, only the midpoint: H/A separability
  cannot be judged.
- **There are no per-match intervals** (a single evaluation per match and config), so the variance of the method
  cannot be separated from the variance between matches.
- The model was not re-run and the scorer's code was not verified: its outputs were audited.
