# External data for a supervised possession model

**More than a Hack 2027 — Phase 1 · research prior to deciding on v2**
Date: October 20, 2026 · Submission: November 17, 2026 (28 days)

---

## 1. Verdict

**Yes, it is worth it, and there is a clear winner: the IDSSE/Bundesliga dataset (Bassek et al.,
*Scientific Data* 2025).** It provides `BallPossession` and `BallStatus` **per frame at 25 Hz**,
which is literally the challenge's `H`/`A`/`D` label, over 7 full matches (~665 min ≈ 1.0 M
labeled frames, versus the 7,500 of the public GT: **×133**). It is CC-BY 4.0, a direct download
from figshare with no registration, and `kloppy` already loads it with one call.

But with one condition that decides the outcome: **train only the emissions, not the
segmentation.** IDSSE's possession convention is finer-grained than the challenge's (I measured
5.20 possessions/min versus 3.40 in the public window), and `spellF1` scores exactly that. The
repo's current architecture (`models/emissions.py` + `decode/viterbi.py`) already separates those
two halves, so it fits without a redesign: **external data → emissions; 5-minute public GT →
transition costs.**

Secondary, in order of value:

- **SkillCorner opendata** (MIT, 20 matches): not so much for training as for **measuring** how a
  real automatic tracking degrades. It is the only candidate that *already* is computer vision on
  broadcast, with gaps and extrapolation.
- **Metrica sample-data**: same provider and same EPTS format as the challenge, which makes it
  ideal for validating the *parser*. As training data it has a serious leak (§5.2).
- **SoccerNet, StatsBomb, SoccerTrack v2**: discarded, with reasons in §4.

---

## 2. Scope and method

What I did: verification of primary sources, without touching `mtah-possession/` or `repo-mtah/`
or the match assets (they are not needed and were not used).

- I read `RULES.md`, `DATA.md` and `EVALUATION.md` from the official repo to pin down the
  requirements.
- For each candidate I downloaded the **actual files** and measured statistics, instead of
  trusting the documentation: one full SkillCorner match (97 MB), the full tracking and events of
  Metrica Sample Game 3 (53 MB), and the tail of the position XML of one IDSSE match (70 MB,
  enough to isolate the ball's `FrameSet` for one half).
- I checked the `kloppy` source code to confirm the field mapping, not just the documentation.
- I characterized the label convention of the 5-minute public GT so I could compare it with
  that of each external source.

Everything I call **"measured"** comes from those files. What I call **"documented"** comes from
the paper, the README or `DATA.md`, and I haven't checked it first-hand.

### 2.1 The two requirements in RULES.md

> 1. *"The organizers can access and inspect it … a public model, a commercial API, an open dataset."*
> 2. *"Your technical document explains it."*

This does **not** demand a permissive license: it demands accessibility. A dataset with open
registration that the organizers can also complete qualifies; what falls outside is "a private
in-house model or an unavailable internal dataset". Even so, I rank the candidates penalizing
access friction, because every extra step is a question a judge may not want to ask.

None of this affects the other relevant prohibition: **do not upload the match assets to any
cloud LLM**. The external data is public and unrelated to the match, so it doesn't fall under that
rule; the challenge's data still doesn't leave the machine.

---

## 3. Comparison table

| | **IDSSE / Bundesliga** | **SkillCorner opendata** | **Metrica sample-data** | **SoccerNet (GSR / BAS)** | **StatsBomb open** | **PFF FC WC2022** | **SoccerTrack v2** |
|---|---|---|---|---|---|---|---|
| **License** | CC-BY 4.0 (+ DFL authorization) | **MIT** | No license file; README asks for responsible use and attribution | SoccerNet NDA, no redistribution | User agreement, non-commercial | Not published | CC-BY 4.0 |
| **Access** | figshare, direct, no registration | GitHub + Git LFS | GitHub, direct | HF *gated*, **manual approval** | GitHub, direct | Form | HF *gated* |
| **Inspectable by the organizers** | Yes, trivial | Yes, trivial | Yes, trivial | Yes, but by signing an NDA | Yes | Only by filling in the form | Yes, by accepting terms |
| **Tracking of 22 players + ball** | Yes | Yes (broadcast) | Yes | Only 30 s clips | **No** | Yes | Yes (interpolated ball) |
| **Frequency** | **25 Hz** | 10 Hz | **25 Hz** | 25 Hz | — | n/a | n/a |
| **Possession label** | **`BallPossession` per frame** | **`possession.group` per frame** | Derivable from events at exact frame | No | Derivable, but without tracking | n/a | No (events only) |
| **Dead / live ball** | **`BallStatus` per frame** | Via `phases_of_play` | Derivable (gaps between blocks) | No | Partial | n/a | No |
| **Player identity** | Yes (official) | Yes (~97%) | Yes | Yes | — | Yes | Yes |
| **Labeled volume** | **~665 min** (7 matches) | ~1,600 useful min (20 matches) | ~287 min (3 matches) | 100 min with no possession label | — | ~6,100 min | 934 min with no label |
| **Conversion effort** | **Low** (kloppy) | Medium (JSONL + resampling) | Low (EPTS) / Medium (CSV) | High and useless | — | Medium | High |
| **Verdict** | **Use** | **Use, to measure degradation** | Use to validate the parser | Discard | Discard | Optional | Discard |

---

## 4. Dataset by dataset

### 4.1 IDSSE / Bundesliga — **the recommendation**

Bassek, M., Rein, R., Weber, H. et al. *An integrated dataset of spatiotemporal and event data
in elite soccer*. Sci Data 12, 195 (2025). Article DOI `10.1038/s41597-025-04505-y`;
data on figshare `10.6084/m9.figshare.28196177`.

- **Content** (documented): 7 full matches, Bundesliga first and second divisions 2022/23,
  with metadata, official event data and TRACAB position data at 25 Hz.
- **License** (verified on the figshare API): `CC BY 4.0`, `is_public: true`,
  `is_embargoed: false`, 21 files, 2.63 GB. It asks to cite the paper and credit the DFL.
- **Format** (verified by downloading the XML): DFL format. One `FrameSet` per person and per
  half, with `<Frame N= T= X= Y= D= S= A= M=/>`. Coordinates in **meters**, origin at the center
  of the pitch, `PitchSize X="105.00" Y="68.00"`, `S` in m/s, intervals of 0.040 s → **exactly
  25 Hz**, same as the challenge's tactical video.
- **The label** (verified): the `FrameSet` with `TeamId="BALL"` carries, in every frame,
  `BallPossession` (`1`=home, `2`=away) and `BallStatus` (`1`=live, `0`=dead). The mapping
  to the challenge's label space is direct and unambiguous:

  ```
  BallStatus == 0                     -> D
  BallStatus == 1 and Possession == 1 -> H
  BallStatus == 1 and Possession == 2 -> A
  ```

- **Two advantages that no other candidate has** (both verified in the file):
  1. The ball carries **`Z` (height)** in all frames. This makes it possible to identify aerial
     balls, which is exactly where `DATA.md` says the ATD loses the ball. Without this column,
     the degradation's bias toward aerial duels (§5.1) would have to be made up.
  2. The ball has coordinates **even when `BallStatus=0`**. That is, there is a ball position
     during stoppages. Metrica doesn't have it, and that is what disqualifies it as the main
     source (§5.2).
- **Loading** (verified in the kloppy source code, `kloppy/_providers/sportec.py` and
  `kloppy/infra/serializers/tracking/sportec/deserializer.py`): `sportec.load_open_tracking_data(match_id=...)`
  exists, with the 7 IDSSE `match_id`s wired in and the figshare URL built; the deserializer maps
  `BallStatus` → `ball_state` (`ALIVE`/`DEAD`) and `BallPossession` → `ball_owning_team`. The
  `coordinates="metrica"` parameter returns normalized 0–1 coordinates with the origin at the
  top-left over a 105×68 pitch (verified in `MetricaCoordinateSystem`): **exactly the challenge's
  convention**, with no conversion of our own.

  ```python
  from kloppy import sportec
  ds = sportec.load_open_tracking_data(match_id="J03WPY", coordinates="metrica")
  ```

- **Measured** (second half of `J03WMX`, 75,259 ball frames = 50.2 min):

  | | |
  |---|---|
  | Raw split | `H` 26.2% · `A` 24.5% · `D` 49.3% |
  | Possessions | 261 → **5.20/min**, mean 5.8 s, median 2.8 s |
  | `D` runs | 62 → 1.24/min, mean 23.9 s |

- **Real limitation**: 7 Bundesliga 2022/23 matches, of which 5 have the same home team
  (Fortuna Düsseldorf, second division). Tactical diversity is scarce and the LaLiga/Barcelona
  style is not represented. For learning "what it looks like when the home team has the ball" in
  geometric terms that matters little; for anything that depends on the specific team shape, it
  matters.

### 4.2 SkillCorner opendata — the domain mirror

- **License** (verified, `LICENSE` file): **MIT**. The most permissive of all.
- **Volume** (verified in the repo tree): **20 matches**, not 10 as the README says — it is
  out of date. All with `tracking_extrapolated.jsonl`, `dynamic_events.csv`,
  `phases_of_play.csv` and `match.json`. Australian A-League 2024/25. The tracking goes through
  Git LFS (~97 MB per match).
- **Format** (verified): JSONL, one line per frame, **10 fps**. Coordinates in meters with
  the origin at the center of the pitch. Each frame carries `ball_data` (with `is_detected`),
  `player_data` (list with `x`, `y`, `player_id`, `is_detected`) and **`possession: {player_id,
  group}`** with `group ∈ {"home team", "away team", null}`.
- **Measured** (match `1925299`, 61,282 frames with `period` = 102.1 min):

  | | |
  |---|---|
  | Ball with coordinates | 78.2% |
  | Ball **detected** (not extrapolated) | **63.0%** |
  | Frames with no tracked object | 21.8% (broadcast cuts and replays) |
  | `player_data` | either 22 objects or 0 — never in between |
  | **Detected** objects per frame (in frames with tracking) | mean 12.73 · p50 14 · p90 18 · max 21 |
  | `possession.group` | `home` 36.5% · `away` 21.6% · `null` 41.9% |
  | Of the `null`s, outside a phase of play | 94.3% |
  | Runs with no detected ball | n=214, mean 10.6 s, p50 3.4 s, p90 34.2 s, max 142.8 s |
  | Team spells | n=335, mean 10.6 s |

- **What it is really for.** It is the only candidate that is **already automatic vision
  tracking**, with the ball being lost through real occlusion. Its 63.0% of detected ball is
  remarkably close to the challenge ATD's 56.9%, and its distribution of ball-absent runs is an
  empirical reference for the most important degradation parameter (§5.1). Use it to
  **estimate the shape of the degradations**, and as a validation set in a different domain.
- **Why it is not the main source**: 10 Hz versus 25 Hz. Upsampling to 25 Hz interpolates
  smoothness that the ATD doesn't have, and that would teach the model to trust a false
  continuity. The 21.8% of frames without tracking are production cuts that the tactical video
  doesn't have.

### 4.3 Metrica sample-data — same format, but with a leak

- **License** (verified): **there is no `LICENSE` file** (404). The README asks for responsible
  use and attribution. Public download from GitHub, so it meets the inspection requirement, but
  it is better to say so as is in the technical document instead of calling it "open source".
- **Content** (verified): 3 matches. Sample Game 1 and 2 in Metrica's own CSV; **Sample
  Game 3 in EPTS/FIFA format, identical to the challenge's** (`<FrameRate>25</FrameRate>`,
  `FieldSize 105×68`, 0–1 coordinates). 143,761 frames = **95.8 min** per match; ~287 min in
  total.
- **Events** (verified, SG3): 3,620 events with exact `start.frame` and `end.frame` —
  `CARRY` 1,395, `PASS` 1,121, `CHALLENGE` 335, `RECOVERY` 294, `BALL LOST` 288, `SET PIECE` 90
  (with subtypes `THROW IN`, `CORNER KICK`, `FREE KICK`, `KICK OFF`…), `BALL OUT` 64, `SHOT` 20.
  Each one with a team. Deriving possession is mechanical and frame-accurate.
- **The problem, measured**:

  | | |
  |---|---|
  | Objects per frame | **22 in 100% of frames.** No gaps, no fragmentation |
  | Ball present, overall | 67.78% |
  | Ball present **inside an on-ball block** | **100.0%** |
  | Ball present **outside an on-ball block** | **5.2%** |

  The pattern is **the exact inverse of the ATD**. In the challenge the ball is lost in crowded
  play; in Metrica the ball is always there during play and disappears during stoppages,
  because they simply stop tracking it. Training on this without correcting it teaches the model
  that "ball present ⇒ play is live", with near-perfect accuracy in training and catastrophic
  on the ATD. And it can't be fixed by adding noise: **you can't invent a ball position where
  there is none.** You can only dilute the correlation by dropping the ball during play too,
  which leaves the set permanently biased on the axis that matters most.
- **What it is good for**: as a **parser test**. SG3 goes in almost directly through `io/atd.py`
  (22 slots instead of 46), and it is the only way to verify the EPTS reader against a file from
  the same provider without spending the public window.

### 4.4 Discarded

- **SoccerNet.** GSR is 200 clips of **30 s** (100 min) with pitch coordinates, role, team
  and jersey number, but **no possession label**. And 30 seconds don't contain possessions: the
  mean in the public GT is 14.2 s, so each clip would give one or two, truncated at both ends —
  useless for calibrating anything related to `spellF1`. Ball Action Spotting labels
  actions on video but doesn't come with tracking. On top of that, access is **NDA via Hugging
  Face with manual approval**, which adds latency and a redistribution restriction, in exchange
  for nothing the others don't give better.
- **StatsBomb open data.** It is event data. The 360 *freeze frames* are snapshots at the
  events, not continuous tracking, and 360 coverage is partial. There is no way to build the
  46-slot-per-frame representation. It is still useful as a reference for the convention of what
  counts as a possession, nothing more.
- **SoccerTrack v2.** CC-BY 4.0 and 934 min, but university football, *gated* on HF, the ball
  track is an **interpolated reconstruction between events** and there is no possession label.
- **PFF FC World Cup 2022.** 64 matches with tracking and events, free via a form, and
  `kloppy` has a `PFF` provider. It is by far the largest volume available. I leave it as
  **optional**: I didn't find published license terms, and form-based access means that an
  organizer who wants to inspect it has to fill it in themselves. If at some point more volume
  is needed, this is the place to go — but the terms have to be read first.

---

## 5. Domain randomization strategy

The problem in one sentence: **all external data is corrected and has identity; the challenge's
ATD does not.** A model trained on clean data learns to rely on signals that don't exist at
inference.

The principle that organizes everything that follows: **every degradation parameter is
estimated from the real ATD of the match, without labels.** That is legitimate —it doesn't touch
the private ground truth, only the marginal distribution of the input data, which is published—
and it's what turns this into measured *domain adaptation* instead of *domain randomization* by
eye. The statistics in section 4.2 (SkillCorner) serve as a sanity check: if a parameter
estimated on the ATD falls far from what a real automatic tracking gives, it is probably
badly estimated.

> Implementation note: the degradation must be applied **at load time of each epoch, with a
> different seed**, not once on disk. A degraded and frozen corpus is a corpus with a fixed
> occlusion pattern that the network learns to memorize.

### 5.1 Ball loss — the degradation that decides the outcome

**Target** (documented in `DATA.md`): ball absent in **43.1%** of all frames and **40.7%** of
the scored ones.

**What not to do:** drop each frame independently with p=0.43. That gives geometric runs with a
mean of 1.75 frames (0.07 s). Real occlusions last seconds. I measured those of
SkillCorner: mean 10.6 s, median 3.4 s, p90 34.2 s, max 142.8 s — a very long tail that an
i.i.d. Bernoulli never produces. A model trained against 2-frame gaps learns to
interpolate; against 300-frame gaps it has to learn to survive without the ball, which is what is
needed in 41% of the scored frames.

**Proposed model: two-state Markov chain with explicit duration, modulated by context.**

1. **Marginal.** Set `P(absent) = 0.407` over scored frames.
2. **Run length distribution.** Estimate from `features.geometry.ball_gap_lengths`
   applied to the real ATD (the function already exists in the repo). Fit a log-normal or a
   negative binomial to the absence **and** presence run lengths separately, and sample from
   there instead of using a per-frame rate. Two distributions, four parameters.
   - *Source statistic:* `NaN` runs in `tracking.ball_xy` over the scored blocks.
3. **Context bias.** `DATA.md` says the ball is lost in "occlusions, crowded boxes
   and balls in the air". That bias has to be reproduced, because otherwise the set of frames
   without a ball ends up made of easy situations and the model underestimates how hard the real
   case is. Fit a logistic regression

   ```
   logit P(ball absent in frame t) = b0 + b1·density(t) + b2·ball_height(t) + b3·center_dist(t)
   ```

   where `density(t)` is the number of players within a 5 m radius around the ball
   (`features.geometry.count_within`, which is already implemented).
   - *Source statistic:* this regression is **fit on the real ATD without labels**. The
     players are observed even when the ball is missing, so `density` is measurable in
     both cases; the dependent variable is simply `isnan(ball_xy)`. `b2` can't be
     estimated from the ATD (there is no height), so it is estimated from IDSSE: `P(absent | Z > 2 m)`
     against `P(absent | Z ≤ 2 m)` using the already fitted model, and transferred. **This is the
     concrete reason IDSSE wins: it is the only one that brings `Z`.**
4. **Ball false positives.** The ATD sometimes places the ball where it isn't. Introduce, at a
   small rate, frames in which the ball position jumps to an incorrect point.
   - *Source statistic:* frequency of frames in the real ATD where the implied ball speed
     between consecutive frames exceeds `constants.BALL_MAX_SPEED` (40 m/s). The
     threshold is already defined in the repo.

### 5.2 Mandatory correction if Metrica is used

Before any other degradation, the **ball↔live-play correlation must be broken**
(§4.3: 100% versus 5.2%). Options, in order of preference:

1. **Don't use Metrica for training**, only to validate the parser. This is what I recommend.
2. If it is used: discard from training all the frames outside an on-ball block. That loses
   33% of the match and, with it, almost the entire `D` class — which leaves Metrica
   contributing only `H`/`A` examples, which is the half of the problem that needs it least.
3. What is **not** valid: applying the §5.1 degradation on top and assuming it is enough. It
   dilutes the correlation but doesn't eliminate it, and leaves a residual bias that can't be
   measured.

IDSSE doesn't have this problem: verified that the frames with `BallStatus=0` carry real `X`,
`Y`, `Z`.

### 5.3 Identity destruction and slot assignment

**Target**: 46 slots (24 home, 22 away) for 22 players, with no identity, that
"appear, disappear and swap identity".

1. **Assignment.** A slot is not a player: it is an *object with a lifetime*. Cut each
   player's trajectory into segments; assign each segment to a free slot of the right team;
   when a segment ends the slot becomes free and can be reused later for another player.
2. **Segment length distribution.** This is what sets how long a track's continuity lasts
   before it breaks.
   - *Source statistic:* for each of the 46 slots of the real ATD, count the maximal non-`NaN`
     runs and their duration. That directly gives the segment length distribution and
     the slot occupancy pattern, without any assumption.
3. **Identity swaps.** When two players of the same team cross within `d`
   meters, swap their slots with probability `p_swap`.
   - *Source statistic:* `p_swap` is not directly observable without ground truth, so use
     as a proxy the **teleport rate**: frames where a slot's position jumps
     more than a plausible human speed allows (e.g. > 12 m/s) between
     consecutive frames. Each jump is an identity swap or a reacquisition; the rate bounds
     the two together, which for this purpose is what matters. Cross-check with SkillCorner, which
     documents ~97% correct identity.
4. **Team error.** The ATD fixes the team per slot in the XML, so a swap between
   teams is impossible in the representation — but the tracker can put a detection of the
   wrong team into a slot. Model it at a low rate (on the order of 1–3%, anchored to
   SkillCorner's 97%), because it is the error that does the most damage: it directly turns `H`
   into `A`.

### 5.4 Number of tracked objects

**Target** (documented): mean of **20.5** objects over scored frames, and
"occasionally exceeds 22 when the tracker duplicates someone".

1. Per frame, sample the number of objects present from the **empirical histogram** of non-`NaN`
   slots per scored frame of the real ATD. Not from a uniform 20–23 range: the shape of the tail
   matters, and the histogram is free to compute.
2. **Which players drop out is not uniform.** Bias toward (a) players inside clusters
   and (b) players far from the ball or from the center of the action. A tracker loses
   first whoever overlaps with another.
   - *Source statistic:* fit `P(slot absent | no. of players within < 2 m)` on the real
     ATD, with `count_within(radii=(2.0,))`, which is already in `geometry.py`.
3. **Duplicates.** With small probability, a player occupies two slots at once with a
   displacement of ~0.5–1.5 m. This is what produces the frames with more than 22 objects.
   - *Source statistic:* fraction of scored ATD frames with more than 22 active slots.

### 5.5 Position and velocity noise

i.i.d. Gaussian noise is a classic mistake here: it is trivially filtered with a
moving average, so the model learns to smooth and never learns to distrust.

1. **Use temporally correlated noise** (AR(1)): `e_t = ρ·e_{t-1} + N(0, σ²)`, applied per
   track. Two parameters, `ρ` and `σ`.
2. **Make `σ` position-dependent.** The error of a detection projected through a homography
   grows with the distance to the camera. Scale `σ` with the distance to the center of the pitch
   along the long axis.
3. *Source statistic:* there are no true positions for the match, so the estimation is
   **spectral**. Compute the autocovariance of the displacement between frames (lags 1 and 2) of
   the real ATD's tracks and of IDSSE's clean tracks; fit `ρ` and `σ` so that the
   autocovariance of the degraded IDSSE matches that of the ATD. Cross-check: the distribution
   of the speed `s` (which the ATD gives directly, in m/s) must match in the two sets
   after the degradation. That comparison is the only one that requires no assumption.

### 5.6 Geometric harmonization

| | IDSSE | Challenge |
|---|---|---|
| Units | meters, origin at the center | normalized 0–1 |
| Pitch | 105.0 × 68.0 | 105.3 × 68.0 (`constants.PITCH`) |
| Frequency | 25 Hz | 25 Hz |

- The conversion is done by `kloppy` with `coordinates="metrica"`. It is worth verifying it
  against a conversion of our own before trusting it.
- The 0.3 m difference in length is 0.3%: irrelevant unless absolute distances in meters are
  compared across sets. Since `geometry.py` works in meters, the right thing is to
  normalize with each source's pitch and go back to meters with the challenge's pitch.
- **Attack direction.** It is necessary to align which team attacks in which direction in each
  half, or all the team-shape features end up mirrored. Metrica's EPTS carries
  `attack_direction_first_half`; DFL also records it. It is a silent and costly error:
  a test that verifies it on every imported match is worthwhile.

### 5.7 What **not** to degrade

- **Do not** reproduce SkillCorner's cuts and replays (21.8% of frames without tracking). The
  challenge's tactical video doesn't have them; its homography is valid in 100% of frames.
- **Do not** degrade the validation set with a different seed from the one that will be used to
  decide. The comparison between configurations has to be on the same degradation.
- **Do not** degrade the 5-minute public window. It is the only real labeled data there is and
  it is used as is.

---

## 6. The label convention: where the real risk lies

This is what surprised me most in the research and what changes how the external data
should be used.

I characterized the public window (frames 11,185–18,684) and compared it with raw IDSSE:

| | Public GT (5 min) | Raw IDSSE (50 min) |
|---|---|---|
| Possessions per minute | **3.40** | **5.20** |
| Mean possession duration | 14.2 s | 5.8 s |
| Median | 11.8 s (`H`) / 7.4 s (`A`) | 2.8 s |
| `D` runs per minute | 0.80 | 1.24 |
| % of frames in `D` | 19.6% | 49.3% |

The actual sequence of the public window is:
`A7 H8 A3 H42 A7 D6 H29 A3 H42 A20 H1 A16 H1 A4 D20 A36 H3 D18 A8 D14 H12` (seconds).

There are 42-second possessions next to 1-second ones. The challenge's convention does **not**
split a long possession at every opponent contact, but it **does** record very brief touches.
The DFL's `BallPossession` changes at every individual touch, and its `BallStatus=0` covers any
referee-called interruption. They are different conventions.

I tested whether a morphological smoothing (minimum possession duration, minimum `D` duration,
bridging over short opponent spells) brings one closer to the other. With `min_poss=3 s,
min_dead=3 s, bridge=3 s` IDSSE drops to 3.31 possessions/min — very close to the 3.40 target —
but the **mean duration stays at 9.2 s versus 14.2 s**. The smoothing fixes the count and not the
shape.

**Two honest caveats about those target numbers:**

1. **The public window is a biased sample.** It is the first 5 minutes after the kick-off,
   the stretch of the match with the fewest interruptions. Its 19.6% of `D` is almost certainly
   far below that of the full match; IDSSE's 49.3% over an entire half is more
   representative of what football is. Calibrating the convention against that 19.6% would be
   fitting to an unrepresentative window.
2. **17 possessions are 17 observations.** `EVALUATION.md` warns that in `spellF1`
   differences below ~0.04 are noise over the full match; over 5 minutes the noise
   is much greater. These four numbers serve to rule out absurd configurations, not to
   fine-tune.

**Operational conclusion, and it is the central recommendation of this report:**

> **Don't learn the segmentation from the external data.** Train with it only the
> **emission model** — the question "given this frame, what does it look like for the home team
> to have it, the away team, or for it to be dead?" —, which is local to the frame and where the
> convention barely matters. Leave the segmentation where it already is: in the **Viterbi
> transition costs**, tuned against the 5 public minutes and against the prior knowledge of how
> long a possession lasts.

This fits exactly with how the repo is built. `models/emissions.py` produces an `(n, 3)`
cost matrix and `decode/viterbi.py` segments it. Replacing the former with a trained version
**doesn't touch the latter**. And the factorization that `emissions.py` already documents
—`P(H) = P(live)·P(home|live)`— is still the right one: `BallStatus` trains the first half
and `BallPossession` the second, each with its own set of valid frames.

---

## 7. Effort

Estimates for one person, assuming the current pipeline works and there are 28 days.

| Phase | What it includes | Effort | Value |
|---|---|---|---|
| **A. Calibrate the geometric v1** | Load IDSSE with kloppy, adapter to `Tracking`, extract labels, tune the ~6 scalars of `config.DEFAULT` (`tau`, `activity_center_ms`, `activity_scale_ms`, `ball_gap_weight`…) by direct search against the external labels | **1–2 days** | **High.** Today those values are guesses. Tuning them is robust to the domain shift because there are few parameters and they have physical meaning |
| **B. Degradation module** | §5.1–5.6, plus the scripts that estimate each parameter from the real ATD | **3–4 days** | **High.** Without this, phase C is counterproductive. And the estimation scripts are by themselves a useful characterization of the ATD |
| **C. Learned emission** | Logistic regression or *gradient boosting* on the `geometry.py` features, trained on degraded IDSSE. Two heads: live/dead and home/away | **2–3 days** | **Medium-high.** This is where the headroom is: the 41% of scored frames without the ball needs a team shape and activity model that is hard to design by hand |
| **D. SkillCorner as validation** | JSONL parser, coordinate conversion, labels from `possession.group` + `phases_of_play` | **1–2 days** | **Medium.** Validating on a different domain is the only defense against having overfitted to IDSSE |
| **E. Metrica as a parser test** | SG3 goes in almost directly through `io/atd.py` | **0.5 days** | **Medium.** Cheap and verifies the EPTS reader without spending the public window |
| **F. Sequence model** | Replace Viterbi with something learned | 5+ days | **Low / negative.** It is exactly where the label convention doesn't transfer (§6). I wouldn't do it on this schedule |

**Recommended route: A → B → E → C → D.** About 8–11 days, leaving a margin. The stopping
criterion is explicit: **measure each phase against the 5 public minutes and against the official
floor of 0.2433. If C doesn't beat A, we submit A.** Phase A has value on its own and doesn't
depend on the rest working out.

**Main risk:** that B eats the schedule. The degradation has many pieces and each one invites
refinement. Mitigation: implement §5.1 (ball loss) and §5.3 (identity) first, which are the two
that really separate the domains, and treat §5.4 and §5.5 as optional refinements.

---

## 8. Coverage: what I haven't evaluated

So that nobody reads this report with more confidence than it deserves:

- **I haven't run anything from the team's pipeline or touched the match data.** All the
  ATD statistics I cite come from `DATA.md`, not from measuring them. The §5 parameters are
  specified but **none is estimated**: those calculations have to be run on the real ATD, and
  it is possible that some estimate comes out ill-conditioned.
- **I haven't trained or evaluated any model.** There is no empirical evidence here that the
  supervised route beats the geometric v1. The argument is about headroom, not about results.
- **The IDSSE statistics come from one half of one match** (75,259 frames of `J03WMX`),
  not from all seven. The `H`/`A`/`D` split and the possession rates could vary between matches.
- **SkillCorner's come from one match out of twenty.**
- **I haven't verified PFF FC's terms** or downloaded its dataset.
- **I haven't checked that `kloppy` downloads correctly from figshare** — I read the code that
  builds the URLs and verified by hand that those URLs exist and serve the file, but I didn't
  run `load_open_tracking_data`.
- **The Metrica sample-data license is ambiguous**, neither nonexistent nor permissive: there is
  no license file. If it is used, it is better to say so in the technical document and not dress
  it up.

---

## 9. Sources

- Official challenge repo: <https://github.com/metrica-sports/more-than-a-hack-2027> (`RULES.md`, `DATA.md`, `EVALUATION.md`)
- Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Sci Data 12, 195 — <https://doi.org/10.1038/s41597-025-04505-y> · data: <https://doi.org/10.6084/m9.figshare.28196177> · code: <https://github.com/spoho-datascience/idsse-data>
- SkillCorner open data — <https://github.com/SkillCorner/opendata> (MIT license)
- Metrica Sports sample data — <https://github.com/metrica-sports/sample-data>
- kloppy — <https://github.com/PySport/kloppy> (`kloppy/_providers/sportec.py`, `kloppy/infra/serializers/tracking/sportec/deserializer.py`)
- SoccerNet — <https://www.soccer-net.org/data> · GSR: <https://arxiv.org/abs/2404.11335>
- StatsBomb open data — <https://github.com/statsbomb/open-data>
- PFF FC 2022 World Cup dataset — <https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset>
- SoccerTrack v2 — <https://arxiv.org/abs/2508.01802> · <https://huggingface.co/datasets/atomscott/soccertrack-v2>
