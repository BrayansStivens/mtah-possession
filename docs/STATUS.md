# Project status — November 16, 2026

Document for the team. This repository has two parts:

- **The submission** (`src/`, `tests/`, `vendor/`, `scripts/`): the code that generates the CSV that
  gets scored. The repo has been public on GitHub since Oct 30 (until Nov 14 this file said it
  was private, and it was not; see §5). Read `README.md`.
- **The analysis** (`trabajo/`): analysis, documents, measurement scripts and result tables.
  It is the material the decisions were made with. Read `trabajo/README.md`.

Everything claimed here is measured, and the file that backs it is named next to it. If something
is not measured, it says so.

---

## 1. At a glance

| | |
|---|---|
| What is submitted on **Nov 17** | a CSV of 199,319 rows (H/A/D per frame), this repository, public, and a short technical document |
| Ranking metric | **spellF1** (possessions matched by temporal overlap). Tiebreaker: per-frame macroF1 |
| Official minimum to beat | **0.2433** (`baseline_smooth50`) |
| **Where we stand, measured** | **0.4073** on average over 7 Bundesliga matches degraded to look like our data, leaving each match out of the detector's training (0.3965 without the boundary detector). Smart Tagging (Nov 8) adds **+0.5 to +0.9 pp estimated** with a simulated tagging: IDSSE has no real one, so that part is not fully measurable (§2) |
| **With the team classifier** | **0.5712** on those same 7 matches with the **team classifier without the ball** (from Nov 12, in `main` since Nov 15), against 0.4111 for the same model without it: **+16.0 pp**, with the classifier and its weight chosen without seeing the match being scored. Over three degradation seeds, +16.1 pp on average and better in all 21 runs. **It is an optimistic bound**: with gaps placed as in the real match it drops to +12.6 pp (§2). Removing players where the ball is missing the way our match loses them (Nov 16), **it holds up: +16.3 pp, 21 of 21** (§2) |
| In the 5 public minutes | 0.4148 (0.4444 without the tagging, 0.4903 without the detector); with the team classifier, 0.6286 — **a reading, not a criterion**: 17 possessions, one miss moves 6 points |
| What `main` generates | Since Nov 15, `config.py` exactly as it stands generates the CSV with the team classifier (sha256 `a43c52fb…`; passes the official validator; 139 tests green since Nov 16; **reproduced byte for byte on Nov 16 from a fresh clone with a freshly created venv, on Windows and on Linux**, §2). **The CSV that gets sent has to be the one `main` generates.** The previous submission, from Nov 8 (`e7938350…`), comes out with `team_classifier_weight = 0`. With `smart_tagging_team_weight = 0` as well you get the Nov 7 one (`5d97eccc…`), and with `boundary_discount_weight = 0` as well, the Oct 31 one (`73c361d5…`) |
| Which configuration it carries | the one in `config.py`; each value has its origin written next to it. **None comes from the 5 public minutes anymore** (forbidden since Oct 22) |

If everything stopped today, there is something valid to submit and it is 15 points above the minimum.

---

## 2. What's done

### The program (`src/mtah/`)

- Reading of the tactical tracking (`io/atd.py`), homography (`io/homography.py`) and Smart Tagging
  (`io/smart_tagging.py`). The tracking and, since Nov 8, the team from the Smart Tagging phases
  are wired into the pipeline; the homography has its reader written and is not used.
- Geometric per-frame features (`features/geometry.py`): ball control per team, player
  activity, team shape, ball speed.
- Two-halves model (`models/emissions.py`): *live/dead* comes from player kinematics and does not
  need the ball; *home/away* comes from ball-player geometry. They are separated because the ball
  is missing in **41%** of the scored frames and the first half keeps working without it.
- Since Nov 12 (in `main` since Nov 15): shape and movement signals of each team in its own
  attacking direction (`features/team_signals.py`) and a classifier trained on IDSSE that reads
  from them who has possession, without looking at the ball
  (`models/team_classifier.{py,joblib,json}`). It adds to the home logit with
  `team_classifier_weight`; with weight 0 it is not loaded and the Nov 8 submission comes out
  to the byte.
- Offline Viterbi decoding (`decode/viterbi.py`) with switch costs and minimum duration.
  It beats the baseline's majority vote on both metrics at once (table in `README.md`).
- `X` is never written on a scored frame: in spellF1 it makes no difference and in macroF1 it costs.
- CLI: `python -m mtah status | build | validate | evaluate`.
- Vendored official scorer (`vendor/evaluation`, MIT) with a regression test that reproduces
  the published numbers of the four official baselines.

### The measurement harness (`src/mtah/training/`, outside the reproduction path)

It is the piece that turned "we can't measure" into "we can measure on seven full matches":

- `idsse.py`: downloads and adapts the IDSSE/Bundesliga dataset (Bassek et al., *Sci Data* 2025,
  CC-BY 4.0): 7 matches with possession and ball status labeled per frame at 25 Hz.
- `convention.py`: brings its possession convention (it changes at every touch) to the challenge's
  (it changes only with controlled possession).
- `degrade.py`: spoils the clean tracking until it resembles ours (ball absent 41%, mixed-up
  identities, noise). Three seeds (`out/cache/recal_*_s{0,1,2}.npz`).
- `harness.py`: scores the pipeline over a full match with real labels.
- `recalibrate.py`: re-derives the hyperparameters on those matches (per-class medians +
  coordinate descent on 4 fit matches, check on 3 validation matches).
- `team_classifier.py` (Nov 12): the nested leave-one-match-out of the team classifier and,
  with `export`, the saved model. Same structure as `boundaries.py` for the detector.
- `missing_players.py` (Nov 16): the stress test of the classifier with missing players
  where the ball is absent (`degrade.thin_where_ball_unseen`), with its decision rule in the
  docstring. It measures the ATD, so it needs `MTAH_DATA_DIR`.

### The measured decisions (chronology)

| Date | What | Result | Where |
|---|---|---|---|
| Oct 22-23 | The rules change: the 5 public min are not a fit set; no manual labeling | 16 values of `config.py` became inadmissible | `docs/analysis-and-proposal.md` §1 |
| Oct 28 | Full recalibration on IDSSE, without the public window | 7 values changed; +1.2 pp on the 3 validation matches | `trabajo/handoff-plugin-data/datos/recalibracion.json` |
| Oct 29 | Audit of the analysis | "it did not overfit" **did not hold up**; validation is easier than the fit | `trabajo/handoff-plugin-data/VALIDATION.md` |
| Oct 29-30 | Cross-validation: the whole procedure, over the **35 possible partitions** of 4 fit / 3 validation | **+1.71 pp** outside the fit set (95% CI [+1.37, +2.05], 32/35) and +3.27 pp inside → it transfers, and **half of the gain is overfitting** | `CROSS-VALIDATION.md`, `datos/splits_cruzados.json` |
| Oct 31 | Expanded grid (64 → 82 values), same 35 partitions, paired comparison | +0.62 pp on fit, **−0.13 pp on validation** (14/35): discarded | `datos/splits_cruzados_ampliada.json`, `compara_rejillas.py` |
| Oct 31 | `activity_smooth` 12 → 50 | +1.11 pp on validation, positive in all 3 seeds, neutral on fit; dead-ball rate 44.9% like IDSSE | `datos/activity_smooth.json` |
| Oct 31 | Three wrongly quoted numbers in `config.py` fixed; rounding convention for the centers | the file produces exactly what it says | `config.py` docstring, commits `0e1560e`, `d805776` |
| Nov 2 | Dashboard data regenerated with today's `config.py` | they reproduce those of Oct 28 in everything that does not depend on the config | `regenera_datos.py` |
| Nov 7 | Ported the useful parts of `DeteccionPosesionBalon` (see below). **The submission does not change**: CSV identical byte for byte before and after (sha256 `73c361d5…`) | 73 tests green (55 + 18) | `io/atd.py`, `features/tagging.py`, `decode/viterbi.py`, `training/weak_labels.py` |
| Nov 7 | **Boundary detector** trained on IDSSE, measured with nested leave-one-match-out and three arms (see below) | **+1.17 pp** spellF1 and +1.12 pp macroF1 over the previous model (21 runs, 16 improve); **+1.07 pp over making changes more expensive without a detector, positive in all 7 matches** | `training/boundaries.py`, `datos/fronteras_lomo_anidado.json` |
| Nov 7 | **The detector goes into the submission.** Trained with the 7 matches; weight 0.95 and scale 2.0 chosen with the same procedure, out of sample | re-exporting gives the same model byte for byte; the submission changes in 3.8% of the scored frames (369 → 333 segments) | `models/boundary_detector.{joblib,json}`, `config.py` |
| Nov 7 | Ball visibility curve (Step zero of `plan-detector-balon.md`) | **it is not flat**: from 59% to 70 / 80 / 90% visible ball it gives +2.2 / +3.5 / +6.3 pp, recovering the easy ones first, in all 7 matches | `training/visibility.py`, `datos/curva_visibilidad.json` |
| Nov 11-12 | Diagnosis of the ball-absent frames, and nine team signals that do not look at the ball | per-frame accuracy 87.5% with the ball seen and 74.5% without it; of the possessions with 75-100% of ball-absent frames, 15.0% are recovered; without the ball the dominant failure is the swapped team (13.3% of those frames) | `trabajo/sin-balon-1111/SUMMARY.md`, `diag.json` |
| Nov 12 | **Team classifier without the ball**, integrated as a candidate in the `equipo-sin-balon` branch (see below). **The submission does not change** | **+16.11 pp** spellF1 and +6.46 pp macroF1 over the model without it, nested LOMO, 7 matches × 3 seeds, **21 of 21 improve** (from +10.9 to +20.1 pp); +12.6 pp with gaps placed as in the real match (seed 0, 7 of 7) | `features/team_signals.py`, `models/team_classifier.{py,joblib,json}`, `training/team_classifier.py`, `datos/equipo_sin_balon_lomo.json` |
| Nov 15 | The `equipo-sin-balon` branch is merged into `main`: the team classifier enters what `config.py` generates | `main` generates `a43c52fb…` (passes the validator, 120 tests); with `team_classifier_weight = 0` it comes out `e7938350…` byte for byte. Nothing new was measured: the numbers and limits of Nov 12 hold | `config.py` |
| Nov 15 | Dashboard regenerated with today's submission: the tables now carry "with detector" and "today (with classifier)", each match scored with models that did not see it | the 14 match-configs reproduce `equipo_sin_balon_lomo.json` exactly; by possessions, recall goes from 0.33 to 0.50 and the spellF1 ceiling from 0.50 to 0.67; the possessions under 5 s that are recovered, from one in twelve to one in five | `regenera_equipo.py`, `datos/clasificador_equipo*.json`, `datos/lectura_publica.json` |
| Nov 15 | **Reproduction on a second machine** (Simon, Windows 11): `main` at `f49dc9c`, Python 3.12.2 with the exact versions of `requirements.txt`, data read from the organization's Drive and the tracking cache set aside to parse the `_atd.txt` from scratch | `build` gives **`a43c52fb…` byte for byte** and passes the validator; with `team_classifier_weight = 0`, `e7938350…` byte for byte; 120 tests green; the two models load with their sha256. `scripts/reproduce.sh` **does not start on Windows from Git Bash** (see §7) | `config.py`, `scripts/reproduce.sh` |
| Nov 16 | **`reproduce.sh` and `env.sh` fixed for Windows** (Git Bash): with `cygpath` they pass `PYTHONPATH` in Windows format and with `;`; on Linux and macOS nothing changes. `reproduce.sh` stops if Python is < 3.12 and accepts `PYTHON=...` | with Python 3.11 it stops with the warning; with the venv it gives `a43c52fb…` | `scripts/reproduce.sh`, `scripts/env.sh` |
| Nov 16 | **Clean reproduction on two systems**, as the organizers would do it: `git clone` from GitHub (branch `medicion-jugadores-sin-balon`, `68fb042`), freshly created venv, `pip install -r requirements-dev.txt` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.8.0, pytest 9.0.1), `./scripts/reproduce.sh FC_Medallo` with no previous cache. (1) Windows 11, Python 3.12.2, Git Bash, data read from the Drive. (2) Linux, Ubuntu 24.04 on WSL2, Python 3.12.3, clone on the Linux file system, the four tactical files copied from the Drive (no video) | **both give `a43c52fb…` byte for byte**, pass the validator and **139 tests green**. 46 s on Windows and 17 s on Linux. On Ubuntu `python3.12-venv` was not installed; the venv was made with `--without-pip` and `get-pip.py` (no sudo). A machine other than Simon's is still missing | `scripts/reproduce.sh` |
| Nov 15-16 | **Stress test of the team classifier with missing players without the ball** (see below). The same models that were submitted and fixed weight 2.0; decision rule written and committed before running. **Nothing in the submission changes** | **+16.28 pp** spellF1 with the same players absent for the whole ball-absent run, **21 of 21 improve** (from +12.5 to +19.9 pp); +15.84 pp if they are drawn at every frame; the usual degradation reproduces the Nov 12 table to the decimal (+16.11 pp). Decision: **`a43c52fb…` is kept** | `training/missing_players.py`, `degrade.thin_where_ball_unseen`, `datos/equipo_sin_balon_jugadores.json` |

**What came in on Nov 7 from `DeteccionPosesionBalon`** (the other line, done without access to this repo):

- **Speed units, now with an explicit failure.** `atd.load()` compares the speed channel with the derivative of the
  position and fails if the ratio falls outside [0.8, 1.25]. Measured on BAR-ATH: **0.992**, so it is m/s even though the XML says kph.
  If someone divides by 3.6 trusting the metadata, the pipeline fails instead of predicting `D` almost everywhere.
- **Smart Tagging is read by code.** The `DEFENDING IN *` phases name the team **without** the ball (the submission agrees with the
  tagged team 16-26% of the time in those phases, against 84% in `BUILD UP`/`PROGRESSION`) and are mirror images of the
  possession phases. `features/tagging.py::team_logit` was adding them up as if they belonged to the team with the ball; it was not
  wired in, so it did not affect any submission, but it would have once wired in.
- **First measurement of "whose ball it is" on the challenge match** (`python -m mtah.training.weak_labels`), against the
  Smart Tagging phases (63,682 scored frames, phases of ~18 s, not ground truth):

  | | agreement | majority class on the same frames |
  |---|---|---|
  | per-frame emission | **0.765** | 0.659 |
  | emission, ball visible | 0.755 | 0.674 |
  | emission, ball not visible | 0.812 | 0.592 |
  | decoded output (live frames) | **0.791** | 0.676 |

  And it rises with the emission's confidence: 0.60 → 0.76 → 0.84 → 0.86 by quartiles. The harness's signal holds on
  BAR-ATH. It is a diagnostic: **weights are not chosen by maximizing this agreement.**
- **The slot for the boundary detector.** `viterbi.decode(..., switch_discount=...)` multiplies each frame's switch costs
  by `1 - discount` where a detector sees a transition. A fraction and not a subtraction: changes cost very different amounts
  (steal 40, restart 14) and a subtraction bounded by the cheapest would barely move a steal. With zeros it reproduces the
  decoder bit for bit (test); a discount of 1 or more fails instead of making fragmenting free.

**The boundary detector, measured on Nov 7** (`python -m mtah.training.boundaries`, 8 min with the cache):

- **What it is.** Gradient boosting (`HistGradientBoostingClassifier`) on 109 features. For 13 per-frame signals
  (ball-player control, distance to the nearest one per team, activity, ball speed and direction, speed
  of each centroid, shape, and the two logits of the current emission): the mean in a window *after* the frame and its
  difference with the one from *before*, at 0.24 / 0.48 / 1 / 2 s; plus the time since and until the last time the
  ball was seen. Target: being within ±12 frames of a label change (in the challenge's convention). Out of sample: AUC
  0.90-0.92 and average precision 0.55-0.61, with 6% positives.
- **How it was measured.** For each left-out match, the detector is trained on the other 6, and the discount weight and the
  scale of the switch costs are chosen on those 6 with out-of-sample scores. The left-out one influences
  nothing. Degradation seeds 1 and 2 use the model and the choice already fixed. The control reproduces exactly
  the 0.3965 of §3. Three arms with the same nested choice, because making changes more expensive could improve things by itself:

  | over 7 matches × 3 seeds | mean | runs that improve | matches that improve (seed mean) |
  |---|---|---|---|
  | only making changes more expensive | +0.10 pp | 12/21 | 4/7 |
  | only the detector's discount | +0.48 pp | 15/21 | 4/7 |
  | **both** | **+1.17 pp** | 16/21 | 5/7 |
  | **both versus only making them more expensive** | **+1.07 pp** | 16/21 | **7/7** |

  Seed 0: 0.3965 → **0.4073**, and macroF1 rises at the same time (+1.12 pp). Making changes more expensive on its own is noise; the detector
  pays off when the decoder cuts *only* where it sees a boundary. The two matches that lose against the control
  (J03WOH, J03WOY) chose scale 2.0, the edge of the grid. It is not widened: on Oct 31 it was already seen that widening
  grids overfits.
- **What it does not say.** It is degraded Bundesliga; on BAR-ATH there is no way to measure it (the public window is not a criterion).
  What could be checked, on Nov 7, is in the next point.
- **The three tests showing it contributes** (`python -m mtah.training.evidence stress | tagging`; tables in
  `trabajo/handoff-plugin-data/datos/evidencia_*.json`). With random gaps the run reproduces the LOMO above
  bit for bit, and the script fails if it does not.
  1. **Stress: the ball is lost where it hurts.** The same detector, weight and scale of each fold (trained and
     chosen with random gaps, like the submission), but the left-out match is degraded with the gaps centered
     where there are people < 5 m from the ball (`crowded`, label-free: this is how DATA.md describes the ATD) or, in the worst case, next
     to the real possession changes (`transitions`). Same run length and same 59% visible; with the real
     runs, much more than 57% of the changes cannot be hidden. Mean of seeds per match:

     | gaps | changes without ball | AUC | both − control | both − only making changes more expensive |
     |---|---|---|---|---|
     | random | 41% | 0.91 | +1.17 pp [+0.21, +2.07], 5/7 | +1.07 pp [+0.51, +1.55], 7/7 (p 0.008) |
     | crowded | 49% | 0.91 | +1.07 pp [−0.22, +2.20], 6/7 | +1.14 pp [+0.41, +1.86], 6/7 |
     | next to the changes | 57% | 0.90 | +0.41 pp [−0.16, +0.96], 4/7 | +0.81 pp [+0.40, +1.20], 6/7 |

     95% bootstrap CI over the 7 matches. **With the gaps where the ATD puts them the gain holds**; in
     the worst case it shrinks to +0.4 pp and can no longer be distinguished from zero, but it is not negative, and what the
     detector adds over making changes more expensive remains clear. There is no reason to withdraw it; if it is ever withdrawn, it
     is `boundary_discount_weight = 0`.
  2. **Mechanism: it fixes what the diagnosis said.** Same run, mean of 7 matches × 3 seeds, random gaps
     (in the other two placements the pattern is the same):

     | | control | only making them more expensive | both |
     |---|---|---|---|
     | changes: precision / recall at ±0.5 s | 0.30 / 0.24 | 0.31 / 0.24 | **0.42 / 0.33** |
     | changes: F1 at ±1 s | 0.35 | 0.35 | **0.42** |
     | error within ±12 frames of a real change | 48.6% | 48.7% | 46.9% |
     | error elsewhere | 17.4% | 17.1% | 16.3% |
     | recall of possessions < 2 s / 2-5 s / > 10 s | 1.8 / 13.3 / 59.6% | 1.4 / 12.4 / 59.8% | 3.1 / 14.0 / 59.5% |

     The gain comes from **where** it cuts (+0.10 in changes F1 with the same number of possessions as only
     making them more expensive), not from cutting more or less. Short possessions are barely recovered: the ceiling is still there.
  3. **BAR-ATH without labels: neutral.** The set pieces (`SET PIECES`) and the transitions (`ATTACKING / DEFENSIVE
     TRANSITION`) of the Smart Tagging, which the pipeline does not use, as ±1 s windows. Of the cuts the detector
     adds (34), 44% fall inside them, and of those it removes (70) also 44%, versus 34% at random. Transition windows
     with some cut: 21 → 19 of 24. Neither for nor against: the tagging is too coarse (set-piece windows
     of 5 to 35 s) to tell this change apart. Nothing is chosen with this.
- **In the submission since Nov 7.** `python -m mtah.training.boundaries export` trains it with the 7 matches and chooses
  weight and scale with the same procedure (each match scored by a detector that did not see it): **weight 0.95, scale
  2.0**, out-of-sample spellF1 0.4111 versus 0.4089 with scale 1.5. Both values are the edge of their grid,
  as in the 2 folds that lost; it is not widened (on Oct 31 it was seen that widening overfits). The model lives in the
  repo (`models/boundary_detector.joblib`, 420 KB) with its sha256, the scikit-learn version and the features config
  it was trained with; if any of the three does not match, `build` fails instead of scoring garbage. Re-exporting gives the
  same file byte for byte.
- **Label-free checks on BAR-ATH**, which choose nothing but would warn of a gross domain jump: the mean
  score on the scored frames is 0.063 (0.061-0.064 on IDSSE) and 2.2% exceeds 0.5 (2.7% on IDSSE).
  It changes 3.8% of the scored frames and drops from 369 to 333 segments, the same direction as in the harness. On the
  public window it drops from 0.4903 to 0.4444 with macroF1 equal: it predicts 10 possessions instead of 14 (17 real) and, at
  IoU 0.5, gets 6 right instead of 8 with 4 false instead of 6 (counted on Nov 11). It is two possessions out of 17; the window
  is not a criterion, but it is on record.

**The visibility curve, measured on Nov 7** (`python -m mtah.training.visibility`, 1 min). Same model and same
degradations, returning entire ball gaps. Mean spellF1 of the 7 matches:

| visible ball | random gaps | easy ones first (ball far from everyone) |
|---|---|---|
| 59% (like the ATD) | 0.3965 | 0.3965 |
| 70% | 0.4255 | 0.4181 |
| 80% | 0.4460 | 0.4310 |
| 90% | 0.4747 | 0.4596 |
| 100% | 0.4977 | 0.4977 |

It improves in all 7 matches at every level. The column that counts is the right-hand one: about 2-3 pp for every 10 points of
visibility. The slope is the reliable part; the level is inflated because the degradation spreads the gaps at random.

**Smart Tagging as team evidence, since Nov 8** (`python -m mtah.training.team_tags` and `... robust`;
tables in `datos/smart_tagging_lomo.json` and `smart_tagging_robustez.json`).

- **Why.** In IDSSE, 60% of the missed frames are home mistaken for away (71% with the ball visible). The
  Metrica phases with a team (BUILD UP, PROGRESSION, CREATING CHANCES, LONG BALL and the DEFENDING IN * as mirror images)
  say whose ball it is in 40% of the scored frames, in stretches of ~20 s.
- **How it comes in.** `features/tagging.team_tags` gives +1/−1/0 per frame and `models/emissions.team_tag_logit` adds it
  to the home logit with `smart_tagging_team_weight`. It goes *after* the boundary detector, which reads `home_logit` and
  was trained without tagging: adding it before would change its inputs. Weight 0 = the Nov 7 submission to the byte.
- **The problem: IDSSE has no Smart Tagging.** It is simulated on its labels, copying what is measured on the real one
  without labels: stretch lengths, coverage, separate stretches (never a direct change of team) and the
  stretch's majority team. What is not known is how accurate it is. The only measurable thing is how much it agrees with our
  emission on BAR-ATH: **0.762**, and by confidence quartile 0.60 / 0.75 / 0.84 / 0.86. A *perfect* simulated tagging
  agrees only 0.706, so the real one agrees more than a perfect one: either it copies part of our errors
  (Metrica labels with the same tracking), or it falls on the easy stretches, or our emission is more accurate here.
- **What was measured.** Nested LOMO on the submitted pipeline (detector included), 7 matches × 3 seeds:

  | tagging scenario | compatible with BAR-ATH? | fixed weight 0.5 | fixed weight 1.0 |
  |---|---|---|---|
  | perfect | no (agreement 0.706) | +1.89 pp | +2.64 pp |
  | 10% / 20% of stretches inverted | no | +1.24 / +0.67 | +1.56 / +0.48 |
  | copies our team in half of the stretches | only the mean agreement | +0.47 | +0.41 |
  | falls where our emission is sure | yes | +0.90 | +0.59 |
  | **best fit to the four quartiles** (easy + 20% copy) | **yes, the best** | **+0.61** [−0.15, +1.51], 5/7 | +0.10 |
  | copies our emission entirely | no (it would give 0.807) | −1.12 | −2.06 |

- **Decision: 0.5.** It wins in every scenario compatible with what was measured and only loses in one that the agreement
  rules out. It was chosen between two fixed weights for robustness, looking at these results: it is a choice made on
  the evaluation matches and has to be reported that way. The automatic per-fold choice came out noisier (+0.39
  in the best fit; −0.20 in the copy one, where it almost always chose 0).
- **What it does not say.** It is a simulation with an assumption about the real tagging's errors, calibrated with a single
  number per quartile. None of the intervals of the best-fitting scenario excludes zero. On BAR-ATH it changes
  2.1% of the scored frames and drops from 334 to 316 segments. On the public window it drops from 0.4444 to 0.4148: the
  same 10 predicted possessions and the same 6 hits at IoU 0.5, with worse overlap on one (counted on Nov 11).

**The team classifier without the ball, from Nov 12, in `main` since Nov 15** (`python -m mtah.training.team_classifier` and
`... export`; table in `datos/equipo_sin_balon_lomo.json`; the earlier exploration, with what was discarded, in
`trabajo/sin-balon-1111/SUMMARY.md`). It was integrated as a candidate in the `equipo-sin-balon` branch and on Nov 15 it was
merged into `main`. Below, "the candidate" is the output with the classifier (`a43c52fb…`) and "the submission" is the
Nov 8 one (`e7938350…`).

- **Why.** The ball is missing in 41% of the frames and there the ball-player geometry has nothing to read: the
  emission drags along the last thing it saw. On degraded IDSSE the per-frame accuracy is 87.5% with the ball seen and
  74.5% without it, and of the possessions that pass with 75% or more of their frames without the ball only 15.0% are
  recovered. Without the ball the dominant failure is the swapped team (13.3% of those frames), ahead of the invented
  dead ball (8.2%) and the missed dead ball (4.1%) (`sin-balon-1111/diag.json`, seed 0).
- **What it is.** Six signals per team, measured on each one's attack axis: width, defensive line, attacking line,
  mean speed toward the opposing goal, mean speed and fraction of players sprinting, smoothed
  over windows of 1, 3 and 8 s (`features/team_signals.py`, 57 columns). The attacking direction comes from the
  tracking itself, without metadata. A gradient boosting turns them into a logit of "possession is the home team's".
  It is trained with both perspectives and half the difference between them is used: swapping the team names flips
  its sign exactly, so it cannot learn "the home team usually has the ball". Out of sample it separates the teams
  with AUC 0.919 to 0.963 depending on the match (0.911 to 0.966 on the ball-absent frames); the hand-made emission,
  on ball-absent frames at 2 s, 0.70.
- **How it comes in.** `models/team_classifier.home_logit` gives the per-frame logit and `emissions.costs` adds it to the
  home one with `team_classifier_weight`, like the Smart Tagging: *after* the boundary detector, which reads
  `home_logit` and was trained without it. Weight 0 = the Nov 8 submission to the byte, without loading the model (there is
  a test). On loading, the model checks its sha256, the scikit-learn version and that the signals are those of the training.
- **What was measured.** Nested LOMO inside the submitted decoder, detector included and scoring out of
  sample: for each left-out match, classifier trained on the other 6 and weight chosen on those 6 with
  classifiers that saw neither that match nor the left-out one. Model and weight are fixed on seed 0; seeds
  1 and 2 only change the occlusion pattern. IDSSE has no Smart Tagging, so the control is the model without
  tagging.

  | seed | without classifier | with classifier | difference | matches better | macroF1 | fixed weight 0.5 | fixed weight 1.0 |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.4111 | 0.5712 | **+16.01 pp** | 7 of 7 | +6.32 pp | +11.41 | +13.77 |
  | 1 | 0.4042 | 0.5755 | **+17.13 pp** | 7 of 7 | +6.75 pp | +12.02 | +14.97 |
  | 2 | 0.4290 | 0.5808 | **+15.18 pp** | 7 of 7 | +6.32 pp | +10.91 | +13.33 |
  | **mean** | | | **+16.11 pp** | **21 of 21** | +6.46 pp | +11.45 | +14.02 |

  The chosen weight was 2.0 in all 7 folds. The run that gains least, +10.9 pp; the one that gains most, +20.1 pp.
  It predicts 206 possessions on average per match where before 179; the real ones are 278. The mean macroF1
  goes from 0.793 to 0.858.
- **Decision: weight 2.0, and the grid is not widened.** It is the edge (0.5 / 1.0 / 2.0) and the gain keeps growing
  with the weight, but the measurement is optimistic because of what is below and a larger weight would bet more on it. On
  Oct 31 it was already seen that half of each grid gain was overfitting.
- **Why six signals and not the nine of the experiment.** A domain classifier tells the BAR-ATH signals apart from
  the IDSSE ones with AUC 0.93 using all nine and 0.72 with these six (0.53 between IDSSE matches). The three that are
  dropped (centroid, block length, players in the opposing half) cost nothing on validation
  (+16.00 pp with nine, +16.01 with six, seed 0). The subset was chosen by looking at the label-free distribution of
  the match, not at its score.
- **What it does not say.** Three things, in order of importance:
  1. *The harness keeps the players and BAR-ATH does not.* On BAR-ATH there are 19.2 players tracked on average when
     the ball is missing and 21.4 when it is there, and 14.3% of the ball-absent frames have fewer than 16 (1.0% with
     the ball). On degraded IDSSE they are 20.4 and 6.4% in both cases. The classifier was measured with almost
     complete teams exactly where players are missing here. **Measured on Nov 16 (next block): it loses nothing.** It
     remains open that on BAR-ATH more players are missing from one team than from the other, more so than in the test.
  2. *The harness gaps fall at random.* BAR-ATH loses the ball mostly while the players are standing still (93.9% of
     ball-absent frames in the lowest-activity bracket, 9.8% in the highest). Relocating the gaps according to
     activity the gain drops to **+12.55 pp** (7 of 7), on seed 0 only and with a smoother curve than the
     real one (`sin-balon-1111/realista.json`).
  3. *The residual drift is not zero*: 0.72 versus 0.53. And against the Smart Tagging phases on ball-absent frames
     the nine-signal classifier gives AUC 0.70 where the hand-made emission gives 0.85, and added together 0.83
     (`sin-balon-1111/atd_check.json`; with the ball seen it is the other way around, 0.91 versus 0.80). It is a weak
     diagnostic (phases of ~18 s, and Metrica labels with the same tracking), but it is not a confirmation.
- **On BAR-ATH** (label-free, only described): the candidate changes 10.1% of the scored frames with respect
  to the submission and predicts 190 possessions where the submission 202. **It changes more where the ball is seen** (12.5% of
  those frames) **than where it is missing** (6.6%): 73.5% of the frames that change have the ball. Here the submission already
  marks 84.2% of the ball-absent frames as dead and the classifier only decides between teams, so in this
  match it acts mainly as a second opinion with the ball in view. Of the changes, 73.5% are from one
  team to the other, 19.5% from dead to team and 7.0% from team to dead. H / A / D split:
  0.358 / 0.207 / 0.436, versus 0.357 / 0.194 / 0.449. On the public window it reads 0.6286 spellF1 (0.4148 the
  submission), macroF1 0.8643 (0.8003) and 11 possessions predicted out of 17 (10). **A reading, not a criterion**: in the same
  experiment two weaker classifiers went up on IDSSE and dropped to 0.3333 on that window.

**The test of missing players without the ball, Nov 15-16** (`python -m mtah.training.missing_players`, 7 min with
the caches built; full table in `datos/equipo_sin_balon_jugadores.json`; branch `medicion-jugadores-sin-balon`).
It closes point 1 of "What it does not say" above. **The submission does not change.**

- **What was done.** `degrade.thin_where_ball_unseen` removes players only in the ball-absent frames down to the
  histogram of objects that BAR-ATH has without the ball, matching quantiles. It does not touch the ball or the frames with
  the ball, and the usual degradation does not change (there is a test). Two arms with the same per-frame histogram:
  `thinned`, which draws at every frame which players are missing (they flicker and the 1 to 8 s windows of the signals
  average out the noise: the easy version), and `persistent`, which draws once per ball-absent run and leaves them out for
  the whole run, like a lost ATD track (stretches of ~2,700 frames on average). **`persistent` decides.**
- **What was not touched.** The models are those of the Nov 12 measurement as they were: for each left-out match,
  classifier and detector trained on the other 6 with the usual degradation (seed 0). Weight 2.0 from
  `config.py`, fixed. Nothing was chosen.
- **Decision rule, written and committed before running** (`bc6e2c5`, `7a33004`). Over the 21 runs
  of `persistent`: mean ≥ +5 pp and ≥ 15 of 21 improve → `a43c52fb…` is kept; mean ≤ +1 pp or < 11 of 21 →
  go back to `e7938350…`; in between, keep it and document it. If `thinned` already said to go back, that was enough.
- **The harness is the same.** The usual degradation, scored in the same run, reproduces
  `datos/equipo_sin_balon_lomo.json` without any discrepancy (21 runs, control and weight 2.0).
- **The players come out as on BAR-ATH** (scored ATD frames against the 7 matches × 3 seeds):

  | | objects with ball | objects without ball | without ball, < 16 | \|home − away\| without ball |
  |---|---|---|---|---|
  | BAR-ATH (ATD, label-free) | 21.39 | 19.17 | 14.3% | 1.34 |
  | IDSSE degraded as usual | ~20.4 | ~20.4 | ~6.4% | |
  | Test, `thinned` and `persistent` | 20.37 | 19.08 | 14.2% | 0.86 and 0.87 |

- **Result.** spellF1 difference with the classifier (weight 2.0) versus without it, left-out match:

  | degradation | seed 0 | seed 1 | seed 2 | mean | improve | worst / best run | macroF1 | spellF1 without → with |
  |---|---|---|---|---|---|---|---|---|
  | as usual (Nov 12) | +16.01 | +17.13 | +15.18 | +16.11 pp | 21 of 21 | +10.9 / +20.1 | +6.46 pp | 0.415 → 0.576 |
  | `thinned` | +15.50 | +17.25 | +14.75 | +15.84 pp | 21 of 21 | +10.7 / +20.4 | +6.37 pp | 0.417 → 0.575 |
  | **`persistent`** | **+16.36** | **+17.04** | **+15.44** | **+16.28 pp** | **21 of 21** | **+12.5 / +19.9** | +6.42 pp | 0.416 → 0.579 |

  By match (`persistent`, seeds 0 / 1 / 2): J03WMX +18.0 / +16.1 / +13.8; J03WN1 +12.8 / +15.0 / +13.5;
  J03WOH +13.7 / +16.9 / +15.9; J03WOY +19.5 / +19.9 / +18.2; J03WPY +19.9 / +19.4 / +19.7; J03WQQ +13.9 / +12.9 /
  +14.5; J03WR9 +16.7 / +19.1 / +12.5. The team AUC on the ball-absent frames barely moves: 0.907 to 0.966
  versus 0.911 to 0.966. It predicts 207 possessions per match (206 before; 278 real).
- **Decision: `a43c52fb…` is kept.** Losing 1 or 2 of ~20 players barely changes a team's width, lines or
  mean speeds over 1 to 8 s, which is what the classifier reads.
- **What it still does not see.** (1) On BAR-ATH the absences are more unbalanced between teams: 1.34 players of
  mean difference without the ball versus 0.87 in the test, which removes players without looking at the team. With the AUC so
  stable it does not seem that it will change the decision, but it is not measured. (2) Who is missing: here at random; in
  the ATD probably those far from the camera or the crowded ones. (3) The ball-absent gaps are still random; this test has not
  been combined with the one of gaps placed by activity (+12.6 pp, seed 0 only). (4) On the frames with the
  ball the test stays at 20.4 objects versus 21.4 on BAR-ATH: players can only be removed, not invented.

### The documents (`trabajo/`)

- `plain-summary.md` — the challenge without jargon. Start here.
- `analysis-and-proposal.md` — analysis of the challenge, proposal, what was tried and what did not work.
- `external-data.md` — which external datasets exist, which one to use (IDSSE) and how to degrade them.
- `handoff-plugin-data/VALIDATION.md` and `CROSS-VALIDATION.md` — the audits. The second
  corrects the first on the overfitting point.
- `handoff-plugin-data/datos/` — all the tables; `handoff-plugin-data/*.py` the scripts that
  produce them.
- `sin-balon-1111/SUMMARY.md` — the diagnosis of the ball-absent frames (Nov 11 and 12), what was
  tried, what was discarded and the ideas that remain; next to it, its scripts and aggregate tables.
- There is also an interactive dashboard (private claude.ai artifact, "Posesión FC Medallo")
  fed with exactly those tables. Ask Brayan for the link.

---

## 3. The numbers, and how to read them

Degradation seed 0, `config.py` of Oct 31 (`trabajo/handoff-plugin-data/datos/scores_por_partido.csv`):

| match | group | before Oct 28 | today |
|---|---|---|---|
| J03WMX | fit | 0.3434 | 0.3607 |
| J03WN1 | fit | 0.3254 | 0.3435 |
| J03WOH | fit | 0.4442 | 0.4678 |
| J03WOY | fit | 0.3598 | 0.3790 |
| J03WPY | validation | 0.3807 | 0.3889 |
| J03WQQ | validation | 0.4209 | 0.4584 |
| J03WR9 | validation | 0.3488 | 0.3771 |
| **mean of the 7** | | 0.3748 | **0.3965** |

Four warnings so as not to misread them:

1. **They are matches from another league with another tracker, degraded by hand.** They measure
   *relative* differences between versions of the model; the absolute level in the competition match
   may be different.
2. **The validation group is easier than the fit group** in all configurations
   (0.41 versus 0.39). The 0.41 is not a prediction of the score.
3. **The "before Oct 28" column is a reconstruction.** That `config.py` was not kept (the
   repo had no commits until Oct 31). The reconstruction scores 0.3–0.7 pp below what was
   measured that day (`datos/robustez_semillas.json`, `_procedencia`).
4. **A single partition does not distinguish configurations.** With the other two seeds, differences of
   less than a point change sign. The only thing that distinguishes them is cross-validation over
   the 35 partitions (§2).

Where the model fails today (`datos/diagnostico_errores.csv`, today's config):

- **Dead ball is detected well**: 92% of the D frames correct.
- **Which team has the ball, so-so**: real H → predicted H 74%, real A → A 69%. The H/A error is
  ~20% of the live frames, and the recalibration rebalanced it toward H, it did not eliminate it.
- **Short possessions are lost**: those under 5 s are **39%** of all of them and ~8% are
  recovered (3% of those <2 s, 11% of those 2-5 s). With that recall, even if precision
  were perfect spellF1 would not exceed ≈0.50. **It is the real ceiling of the current model.**
- **The error is concentrated next to the transitions**: within ±12 frames of a real change one
  frame in two fails; beyond 250 frames, 7%.
- **When the ball has gone unseen for more than 5 s, the error rises to 27%** (versus 14% with the
  ball visible). The ball is not lost in short occlusions: it is lost in blocks of minutes.

---

## 4. What's left

### Plan until the 17th (agreed on Nov 11, after merging `fronteras-deteccion` into `main`; on Nov 12 the fourth candidate is added and on Nov 15 it is merged into `main`)

Six days left. What gets done, by whom, and in what order:

| when | what | who | how we know it is done |
|---|---|---|---|
| until the 14th | **The Smart Tagging set pieces as the team taking the restart** (corners and goal kicks; throw-ins go with `N/A`). It is what is missing in the long gaps: without the ball the emission gives P(H) = P(A) and the model lives off inertia, whether the gap lasts 5 s or 266 s. Unlike the phases, this **is measured for real**: IDSSE brings the set-piece events with team, nothing has to be simulated | Simon | same nested LOMO as §2; it goes in only if it wins on validation and is reported with its interval |
| until the 14th | **Reproduction on a clean machine**: clone, `pip install -r requirements.txt`, `source scripts/env.sh`, `build`, and the sha256 `a43c52fb…` has to come out on `main` (since the Nov 15 merge; before, `e7938350…`). The submission now depends on two pickles with an exact scikit-learn version and on Python ≥ 3.12; if it fails, it fails in the organizers' hands. Careful: `python -m mtah` needs `PYTHONPATH=src` (`env.sh` sets it) or installing the package. **Done on Nov 15 on Simon's machine: `a43c52fb…` comes out byte for byte** (§2, chronology), and **on Nov 16 clean**: fresh clone from GitHub and freshly created venv, on Windows 11 and on Linux (Ubuntu 24.04, WSL2); both give `a43c52fb…` byte for byte and 139 tests green | Brayan / Simon | the sha matches on a machine that is not ours ✅ |
| **Friday 14** | **Decide and freeze the submission.** Four candidates, reproducible with three values of `config.py`: the Oct 31 one (`73c361d5…`, 0.3965 measured), with detector (`5d97eccc…`, 0.4073 measured out of sample), with detector + tagging (`e7938350…`, +0.6 pp *estimated* on top, interval that includes zero) and, since Nov 12, with the team classifier without the ball on top of everything (`a43c52fb…`, branch `equipo-sin-balon`, +16.1 pp measured out of sample on IDSSE, +12.6 pp with gaps like the real ones; optimistic bound, §2). **On Nov 15 the branch was merged into `main`**: `config.py` exactly as it stands generates `a43c52fb…`. Going back to the Nov 8 one means setting `team_classifier_weight = 0`. After the 14th no hyperparameter is touched: on Oct 31 it was seen that half of each grid gain was overfitting | everyone | `config.py` frozen and the submission's sha written in §1 |
| 14-16 | **The short technical document** (2-3 pages): what the model does, how it was measured, what we do not know. It is drawn from this file | Brayan | PDF in `trabajo/` |
| 16 | **Public repo: it has been public since Oct 30** (checked on Nov 14). What remains: audit of `data/` (done on every push), attribution to the DFL (already there), `vendor/NOTICE.md`, decide whether `trabajo/` stays | Brayan | the repo is public and the validator passes on the submitted CSV |
| 15-16 | **Test of missing players without the ball** on the submission, with the decision rule written before running. **Done: `a43c52fb…` is kept** (+16.28 pp, 21 of 21; §2) | Simon | `datos/equipo_sin_balon_jugadores.json`; STATUS up to date ✅ |
| — | **`detector-balon` is shelved** without merging. The only thing rescued is its documentation section (what was tried and why it does not give more), so as not to repeat the experiment. If it is resumed, its next step is written down: simulate the filling-in *with its errors* on IDSSE before touching the pipeline | — | — |

Team conventions, so that the history is uniform: commits in Spanish and without `Co-Authored-By`
lines or tool marks.

### The rest, in the order I would do it

None of this has been started unless it says so.

1. **The final technical document.** It does not exist as such; the material is in `trabajo/`. The
   sentence about overfitting that goes into it is the one from §2 (Oct 29-30), not "it did not overfit".
2. **Getting the repository ready to be made public.** Before publishing: check once more
   that nothing from `data/` gets in (today: 0 files, audited on every push), cite the
   IDSSE paper and credit the DFL (**a condition of the CC-BY license, not a courtesy**), review
   `vendor/NOTICE.md`, and test reproduction on a clean machine (**done on Nov 16**:
   fresh clone and freshly created venv on Windows and on Linux, same sha256; see §2).
3. **Our own ball detector on the video** (the vision line). The visibility curve
   (§2, Nov 7) says it is worth it *if it arrives*: going up to 70% visible ball is worth about
   +2 pp, and to 80% about +3.5 pp. But the plan estimates it at 7.5 days, 10 remain until the
   submission, and its abandonment criterion (50% recall at 90% precision at three days)
   still stands. It has not been started.
4. **The Smart Tagging set pieces** as a clue for restarts. The phases' team is already
   wired in (§2, Nov 8). The 59 set pieces are not: their windows are 5 to 35 s, and on Nov 7 the detector's
   cuts fell in them just as they do at random, so as a clue for where to cut they are
   too coarse. What they could give is the team taking the restart (goal kicks and
   corners; throw-ins go with `N/A`).
5. **Improve the degradation** so that the ball-absent runs fall where they really fall
   (crowded zones, aerial balls). Since Nov 7 `degrade.apply(gap_weights=...)` places them
   by weight and is used as the detector's stress test (§2); training and choices
   continue with random gaps. Moving everything to `crowded` is a harness change that
   would move all the numbers, and it has not been done.
6. **A learned emission** on the degraded IDSSE matches (phase C of
   `external-data.md`). Quick test on Nov 7 with the detector's features (half a match to
   train, half to measure): per-frame accuracy 0.783, the same as the geometric one; mixed
   with it, changes of the same size as the noise. It is not a shortcut; if it is done, with the same nested
   LOMO.
7. **Validate on another domain** (SkillCorner, phase D) to find out whether we have tuned to IDSSE.

What is **no longer** worth it: continuing to fine-tune the hyperparameters of the current model. With 7
fit matches there is no signal to tell more values apart (the expanded grid showed it:
5 h 48 min of compute for −0.13 pp), and the 0.50 ceiling is set by the recall of short
possessions, not by the thresholds.

---

## 5. Problems we ran into

What went wrong, how it was detected and how it ended. It is here so that nobody falls for it again.

| Problem | How it was seen | What was done |
|---|---|---|
| 16 values of `config.py` came from the 5 public minutes, which since Oct 22 cannot be used for tuning | reading `RULES.md` after the change | re-derived on Oct 28 from IDSSE; `config.py` states the provenance of each value |
| "The recalibration did not overfit" | audit of Oct 29: it gained +1.9 on fit and +1.2 on validation with 1 of 3 matches against | cross-validation over 35 partitions: +3.3 / +1.7; half of it is overfitting. Sentence corrected in all the documents |
| An intermediate claim of Oct 29 said the descent "transfers nothing" | it was false; it came from looking at a single split | withdrawn; the cross-validation replaces it |
| `config.py` quoted three numbers that the file did not produce (validation 0.3957; D rate 45.3%; "300 and 450 already go down") | when re-measuring on Oct 31 | corrected; convention: the class centers are written rounded (the sixteen digits of the script are seed noise) |
| Seven hyperparameters were pinned to the edge of their grid | per-parameter stability over the 35 splits | expanded grid and measured: it does not improve validation. Correct diagnosis, wrong implication |
| `activity_smooth` = 12 was chosen by only 1 of 35 partitions | same analysis | put back to 50, justified with the three seeds (+1.1 pp validation) |
| The configuration from before Oct 28 was not kept | the repo had no commits | it was reconstructed; the reconstruction runs 0.3–0.7 pp lower and is labeled as such wherever it appears |
| The validation set is easier than the fit set | in all four configurations | the 0.41 is not quoted as an expectation |
| Aggregation bias in two categories of 512 and 854 frames (0.14% of the data) | `VALIDATION.md` finding 4 | they are shown with their `n` and flagged; nothing is concluded from them |
| The recalibration improves spellF1 by emitting **fewer** and longer possessions (~1,540 → ~1,390 predicted versus 1,945 real) | `VALIDATION.md` finding 5 | it is known and said: it improves the metric by moving away from the real structure; the 0.50 ceiling comes from there |
| I estimated 50 min for the expanded grid; it took 348 | large windows multiply the cache per match | documented for next time |
| This file said the repo was private and that it would be made public on submission day. It has been public since Oct 30 | when pushing the `equipo-sin-balon` branch on Nov 14, the visibility on GitHub was checked: the page opens without credentials | corrected here. No match data is versioned (the index was reviewed: no video, no tracking, no homography, no Smart Tagging). The approach and `trabajo/` have been in plain view since then |
| An example command with a placeholder (`USUARIO`) was run as is and invited an unrelated GitHub user to the repo (believed to be private) | seen in the output | invitation revoked on the spot; verified 0 pending and a single collaborator. They never got access (invitations require acceptance) |

Open question, partly answered on Nov 8: the pipeline uses the tactical tracking and the team
from the Smart Tagging phases. The homography and the broadcast tracking are read but not
used, for the reasons in `README.md` (they do not separate play from stoppages; a different timeline).

---

## 6. What we don't see (untested coverage)

- **There is no ground truth for the competition match.** Everything measured is transfer from
  Bundesliga: another league, another camera, another tracker. Nobody knows how much is lost in the jump.
- **Seven matches are few.** The seeds vary the degradation noise, not the matches.
- **The degradation is our own model of the ATD.** If the real ATD loses the ball some other way,
  the harness is optimistic or pessimistic and we do not know which.
- **The dispersion of the class centers has not been measured** (only the median), so one
  cannot judge how separable H and A are by those signals.
- **There are no per-match intervals**: one evaluation per match and configuration.
- **Clean reproduction is done, but on a single computer.** On Nov 16, from a fresh clone
  from GitHub and a freshly created venv, `./scripts/reproduce.sh` gives `a43c52fb…` byte for byte on
  Windows 11 (Python 3.12.2) and on Linux (Ubuntu 24.04 on WSL2, Python 3.12.3), with 139 tests
  green (§2). Both systems run on Simon's machine; it has not been tried on macOS or on
  another physical machine. On Linux `python3.12-venv` was missing and it was solved without sudo (§7).
- **SAST, linting in CI and load tests have not been run**; the tests are unit and
  contract tests.
- **There is no record of the reader having been tested with Metrica's sample data** (phase E of
  `external-data.md`).
- **The supervised route is only measured in the harness**: the boundary detector wins +1.1 pp
  on degraded Bundesliga and +0.4 to +1.1 pp with the gaps placed adversarially; whether it
  transfers to BAR-ATH cannot be checked (the Smart Tagging comes out neutral).
- **The team classifier (in `main` since Nov 15) is measured on degraded Bundesliga, not here.**
  The lack of players without the ball is already measured (Nov 16): with BAR-ATH's histogram and the same
  players absent for the whole run it gains +16.28 pp, the same as without removing them (§2). Still not measured: the
  imbalance between teams in those absences (1.34 players on BAR-ATH, 0.87 in the test), the
  gaps placed by activity on seeds 1 and 2 (and combined with the missing players), the
  classifier alongside the Smart Tagging (IDSSE does not have it) and the switch costs readjusted with the
  classifier in place. The +16 pp are still an upper bound; with realistic gaps it was +12.6.
- **The Smart Tagging weight comes from a simulated tagging**, not the real one: how accurate the
  Metrica one is cannot be measured without labels, only bounded by its agreement with our emission.

---

## 7. How to review and reproduce it

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # numpy 2.5.4 requires Python >= 3.12
pip install -r requirements-dev.txt
python -m pytest -q                      # 139 tests
```

To regenerate the submission you need the match data (they are not in the repo, they cannot be:
5 GB and not redistributable). Download them from the organization's Drive and:

```bash
source scripts/env.sh                    # MTAH_DATA_DIR, defaults to ~/Documents/Hackaton/data
python -m mtah status
./scripts/reproduce.sh FC_Medallo        # -> out/FC_Medallo.csv
```

`reproduce.sh` works on Linux, macOS and Windows with Git Bash (since Nov 16; before, it failed
on Windows with `No module named mtah`). It stops if Python is < 3.12; `PYTHON=...` picks the
interpreter. What was run on Nov 16 for the clean reproduction:

```bash
# Windows, Git Bash (the system 'python' may not be 3.12: the venv's is used)
git clone https://github.com/BrayansStivens/mtah-possession.git && cd mtah-possession
py -3.12 -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-dev.txt
export MTAH_DATA_DIR='<Drive folder that contains tactical/>'
PYTHON=.venv/Scripts/python.exe ./scripts/reproduce.sh FC_Medallo   # ~45 s
sha256sum out/FC_Medallo.csv                                         # a43c52fb…

# Linux (Ubuntu 24.04). If python3.12-venv is missing and there is no sudo:
#   python3.12 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
export MTAH_DATA_DIR=<folder with tactical/>    # atd.xml, atd.txt and smart_tagging.xml are enough
./scripts/reproduce.sh FC_Medallo                                    # ~20 s
sha256sum out/FC_Medallo.csv                                         # a43c52fb…
```

To redo the harness measurements you also need `pip install -r requirements-train.txt`
(kloppy, to download IDSSE). The first call to `idsse.load` downloads each match (~2 min
each) and leaves it in `out/cache/`; after that `python -m mtah.training.boundaries`,
`python -m mtah.training.visibility`, `python -m mtah.training.team_classifier` and
`python -m mtah.training.missing_players` (this one with `MTAH_DATA_DIR`; ~4 min of preparation and 3 of
evaluation on Simon's machine) run in minutes. The
scripts in `trabajo/handoff-plugin-data/` were written to run from the original working
folder (`~/Documents/Hackaton/`) with absolute paths; each one says at the top what it does
and what file it produces.

---

## 8. Rules that cannot be broken

- **The match data is not redistributed** (video, tracking, homography, Smart Tagging).
  `.gitignore` excludes it; do not un-ignore it. Every push is audited.
- **No excerpt of the match is uploaded to a cloud LLM.** Outputs of our model,
  the public ground truth and aggregates over public matches are fine.
- **The 5 public minutes are neither a fit set nor a selection set.** They are looked at, not used
  to choose.
- **No manual labeling.**
- **The public repo has to regenerate the submitted CSV exactly**, which is why
  `requirements.txt` only carries numpy and pandas pinned and `training/` stays off the path.
