# Possessions without the ball: what was measured on November 11 and 12

Working folder, outside `src/`. The submission does not change: `out/submission.csv`
is still `e793835071b54afb`. **Update from Nov 12:** idea 1 of §8 was integrated into the
`equipo-sin-balon` branch (`features/team_signals.py`, `models/team_classifier.py`,
`training/team_classifier.py`, 17 tests). The integrated measurement reproduces the reduced (a)
row of §4 down to the decimal: +16.01, +17.13 and +15.18 pp. Details in `STATUS.md`, §2.
Everything below is measured on the 7 degraded IDSSE matches, out of sample per match. The
challenge's public window (5 min, 17 possessions) is only read: one possession moves about
6 points and decides nothing.

## 1. What was already there reproduces

| Check | Result |
|---|---|
| Tests | 103 pass |
| `scripts/reproduce.sh` | regenerates the submission byte for byte |
| Public reading of the submission | spellF1 0.4148, macroF1 0.8003, 10 possessions predicted out of 17 |
| Nested LOMO of Simon's detector | +1.08, +0.87 and +1.56 pp per seed, mean +1.17 pp |

## 2. Where the score is lost (`diag.py`, seed 0, random gaps)

| Segment | Frames | Per-frame accuracy |
|---|---|---|
| Ball seen | 59.1% | 87.5% |
| Ball not seen | 40.9% | 74.5% |

| Real possessions recovered at IoU 0.5 | |
|---|---|
| With 75 to 100% of frames without the ball | 15.0% |
| With 25 to 50% | 55.2% |
| Shorter than 3 s | 5.8% |
| Longer than 25 s | 69.0% |

| Real changes found at 1 s | With ball | Without ball |
|---|---|---|
| To dead ball | 82.0% | 36.9% |
| Restart | 69.1% | 28.4% |
| Steal | 18.5% | 4.1% |

The submission predicts 179 possessions on average against 278 real ones. The dominant failure
without the ball is the team: 13.3% of those frames have the team swapped, against 8.2% of
invented dead ball and 4.1% of lost dead ball.

## 3. The shape of the teams says who has the ball (`feats.py`, `modelos.py`, `auc.py`)

Out-of-sample AUC, seed 0.

| Signal | AUC |
|---|---|
| Team, ball-absent frames at 2 s: the submission's hand-built logit | 0.6996 |
| Team, same frames: classifier (a), players only | 0.9536 |
| Dead ball, ball-absent frames: hand-built logit | 0.9359 |
| Dead ball, same frames: classifier (b) | 0.9735 |

(a) uses 9 signals per team and attack direction (centroid, length, width, defense and attack
lines, tide, speed, players in the opponent's half, sprints) in windows of 0.5, 1.5 and 4 s.
The width difference alone reaches 0.90.

## 4. Inside the decoder, random gaps, three seeds (`exp.py`, `semillas.py`)

Models trained with seed 0 of the other 6 matches. Weights set by the nested selection of
`exp.py`.

| Variant | Seed 0 | Seed 1 | Seed 2 | Mean | Matches improved |
|---|---|---|---|---|---|
| Submission | 0.4111 | 0.4042 | 0.4290 | | |
| (a) all signals, weight 2.0 | +16.00 | +17.29 | +15.29 | +16.19 pp | 21 of 21 |
| (a) without `opp`, `cen`, `len`, weight 2.0 | +16.01 | +17.13 | +15.18 | +16.11 pp | 21 of 21 |
| (b) learned dead ball | +4.33 | +3.70 | +2.44 | +3.49 pp | 20 of 21 |
| reduced (a) + (b) | +20.86 | +21.74 | +19.91 | +20.84 pp | 21 of 21 |

With reduced (a) + (b) the mean spellF1 goes from 0.41 to 0.62 and the macroF1 from 0.79 to 0.89.

## 5. Gaps like those of the real match (`perfil.py`, `realista.py`, seed 0)

The ATD loses the ball mostly when the players are standing still: 93.9% of frames without the
ball in the least active segment and 9.8% in the most active one. The proxy hid it at random,
close to 41% in all segments. `realista.py` repositions the gaps according to activity without
touching the players: 84.8% and 19.4% in those same segments, a softer version of the real
curve. With that, 68.3% of the frames without the ball are stopped play, against 44.2% at
random.

| Variant, weights fixed in advance | spellF1 | Difference | Matches improved |
|---|---|---|---|
| Submission | 0.4511 | | |
| Submission without boundary detector | 0.4372 | −1.38 pp | 1 of 7 |
| Detector retrained with realistic gaps | 0.4514 | +0.03 pp | 4 of 7 |
| (a) all, weight 2.0 | 0.5780 | +12.70 pp | 7 of 7 |
| (a) without `opp`, `cen`, `len` | 0.5766 | +12.55 pp | 7 of 7 |
| (b) learned dead ball | 0.4898 | +3.87 pp | 7 of 7 |
| (a) + (b) | 0.6273 | +17.62 pp | 7 of 7 |
| (a) + (b) retrained with realistic gaps | 0.6321 | +18.10 pp | 7 of 7 |

Moving `ball_unseen_logit` in either direction makes it worse: from −0.25 to −2.84 pp. The
submission already marks dead ball on 73.5% of the frames without the ball and the real value
is 68.3%.

## 6. The real match is not quite like the IDSSE (`deriva.py`, `deriva_senal.py`, `robusto.py`)

Without labels: a classifier tries to tell the ATD apart from the degraded IDSSE using the team
signals. 0.5 is indistinguishable. The reference is one IDSSE match against the others.

| Signals of (a) | Columns | IDSSE team AUC | Domain, ball seen | Domain, without ball | Random gain | Realistic gain |
|---|---|---|---|---|---|---|
| Reference between IDSSE matches | | | 0.53 | 0.59 | | |
| All 9 | 87 | 0.9518 | 0.933 | 0.873 | +16.00 | +12.70 |
| Without `opp` | 78 | 0.9512 | 0.923 | 0.843 | +16.51 | +13.04 |
| Without `opp`, `cen`, `len` | 57 | 0.9506 | 0.724 | 0.659 | +16.01 | +12.55 |
| Width and lines | 27 | 0.9180 | 0.701 | 0.659 | +7.98 | +5.35 |
| Width only | 9 | 0.9059 | 0.602 | 0.516 | +5.18 | +2.67 |

The drift is in `opp` (0.91 on its own), `cen` and `len` (0.70). Removing them costs nothing in
validation and leaves the classifier much closer to the distribution of the real match. The
subset was chosen by looking at the unlabeled distribution of the challenge match, not its score.

Another difference that the proxy does not reproduce: in the ATD, players are missing exactly
when the ball is missing. 19.2 tracked players on average without the ball against 21.4 with
the ball, and 14.3% of frames with fewer than 16 against 1.0%. In the degraded IDSSE it is 20.4
and 6.4% in both cases.

## 7. Reading on the real match (`real.py`, `robusto.py`), not a criterion

| Candidate | Public spellF1 | Possessions out of 17 | Possessions in the match | Frames equal to the submission |
|---|---|---|---|---|
| Submission | 0.4148 | 10 | 202 | 100% |
| (a) all | 0.6143 | 11 | 185 | 89.7% |
| (a) without `opp`, `cen`, `len` | 0.6286 | 11 | 190 | 89.9% |
| (a) width and lines | 0.3333 | 7 | 190 | 90.3% |
| (a) width only | 0.3333 | 7 | 171 | 91.1% |
| (b) learned dead ball | 0.4000 | | | |

The two weak classifiers go up on IDSSE and down on the public reading: this confirms that
17 possessions do not rank candidates.

## 8. Ideas, in order of expected value

1. **Adopt reduced (a), weight 2.0.** It is the only thing with a large and stable gain: +16 pp
   over three seeds, +12.6 pp with realistic gaps, 28 of 28 comparisons in favor. It requires a
   new artifact trained on the IDSSE, its reproduction test and the license credit.
2. **Re-tune the switch costs with (a) in place.** 40, 34 and 14 were tuned when the team without
   the ball was almost noise. Only 18.5% of steals are found with the ball and with (a) 206
   possessions out of 278 are predicted. Nested experiment, not run yet.
3. **Remove players when the ball is missing while degrading.** It would close the gap from
   point 6 and tell us whether (a) holds up with 16 to 19 players. Not run.
4. **Normalize the signals per match.** The residual drift of reduced (a) is 0.72 and 0.66
   against 0.53 and 0.59. Centering each signal with the mean of its own match uses no labels.
   Not run.
5. **(b) learned dead ball.** +3.5 pp alone and about +4.7 pp on top of (a) on the IDSSE, but it
   does not improve the real reading and depends on the ball context, which is where the proxy
   differs most. Second step, not first.
6. **Smart Tagging set pieces as dead-ball anchors** in stretches without the ball. It is Simon's
   task and fits with point 2.

Discarded based on measurement: moving `ball_unseen_logit`, retraining the detector with
realistic gaps, a width-only classifier.

## Still to check

- Realistic gaps only on seed 0, and the curve is smoother than the real one.
- (a) against the Smart Tagging phases on frames without the ball gives AUC 0.61 to 0.70. It is
  a weak diagnostic, the phases last about 18 s.
- Only reduced (a) is integrated in `src/`, in the `equipo-sin-balon` branch. The rest is not.
