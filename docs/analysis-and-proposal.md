# More than a Hack 2027 — challenge analysis and proposal

**FC Medallo** · Work day 1 · October 24, 2026

**In short:** the pipeline already produces a valid submission of 199,319 rows. It stands at **spellF1
0.3902** measured on seven full matches, against an official floor of **0.2433**. The
finding of the day is that **the 5 minutes of ground truth they publish don't represent the
match** (88.2% ball visible versus 59.3% for the whole match), and that was miscalibrating the
dead-ball detector. The rest of this document is the challenge explained from top to bottom,
what I propose, and what I tried.

---

# PART 1 — What we're being asked

## The problem, no frills

One match: **Barcelona – Athletic**, LaLiga 2024-25, matchday 2. They give us the video and the
automatic tracking, and we have to say, **for each one of the 199,319 frames**, who has
the ball.

Four labels:

| | |
|---|---|
| `H` | Barça has the ball, with live play |
| `A` | Athletic has the ball, with live play |
| `D` | dead ball, not in play |
| `X` | no answer |

The submission is a CSV of 199,319 rows. That's it. That's what gets scored.

Of those frames, **158,857 are scored** (about 106 minutes): both halves. What comes before
kick-off, half-time and what comes after the final whistle doesn't count, so there we can safely put
`X`.

## What makes the problem hard

Three things, and it's worth being clear on them because they shape everything:

**1. The ball is absent in 41% of the scored frames.** And it's not that it can't be seen in the
video: Metrica's automatic tracker loses it. It gets lost exactly where it matters
most — crowded penalty areas, aerial duels, tackles.

**2. There is no player identity.** There are **46 tracking "slots" for 22 players**. The
slots appear, disappear and swap identities as the tracker loses and recovers people. In other words: you can't say "player 7 has the ball", you can only reason
collectively.

**3. They only publish 5 minutes of correct answers.** 7,500 frames, 17 possessions. The
rest of the ground truth is private until the competition closes.

## What data we get

| | **tactical** | **broadcast** |
|---|---|---|
| role | **the reference — row *i* of the submission is frame *i*** | optional, **not synchronized** |
| video | 199,319 frames, 25 fps | its own timeline, with cuts and repeats |
| tracking (ATD) | 199,319 records | 164,488 |
| homography | 100% usable | 65.8% |
| Smart Tagging | 638 instances | — |
| ball visible | 56.9% | 50.5% |

- **ATD (Automatic Tracking Data)**: position and velocity of each object per frame,
  normalized coordinates 0–1, velocity in m/s, `NaN` where missing. It's automatic and **uncorrected**.
- **Smart Tagging**: 638 tactical phases detected automatically (Build Up, Progression,
  Set Pieces...), each with a team. **It is not ground truth**: Metrica generates it from the same
  video we have, without review. It's a free hint, not an answer.
- **Homography**: 3×3 matrix per frame to go from pitch position to pixel. It's only
  needed if we're going to read the video.

> Watch out for a detail that sinks submissions: if your file comes out with ~164k rows,
> you indexed the broadcast video by mistake.

## How we're scored

Two metrics, and the one that rules **isn't the obvious one**.

**`macroF1`** — mean F1 over the three classes, frame by frame. It's the classic one.
It's only a tiebreaker.

**`spellF1`** — **this is the one that ranks.** It treats each possession as **one object**, not as
a pile of frames. It matches each possession you predict with a real one by
temporal overlap, and averages over five thresholds (0.3 to 0.7).

The difference matters enormously. Look at the baselines they themselves publish — it's
**the same algorithm four times**, changing only the smoothing window:

| baseline | macroF1 | spellF1 |
|---|---|---|
| `baseline_raw` | 0.6532 | 0.0486 |
| `baseline_smooth25` | **0.6752** | 0.1996 |
| `baseline_smooth50` | 0.6724 | **0.2433** |
| `baseline_smooth125` | 0.6356 | 0.2242 |

`macroF1` moves 0.04 and prefers one version. `spellF1` **moves by a factor of
five** and prefers a different one. A system can get 97% of the frames right and end up terrible if it
reports triple the possessions that there were.

**The official floor to beat is `baseline_smooth50`, spellF1 = 0.2433.**

Two more things about the metric:

- A 1-frame offset is **free**. Being a second late costs you a third of the result.
- A constant ("all `H`") scores exactly **0**.

### What the `X` does

This matters for design:

- For `macroF1`, an `X` counts as a miss.
- For `spellF1`, **the scorer fills it in with your previous label** from that stretch.

In other words: staying quiet equals repeating the last thing you said. **Inside a possession it's
free** — if you lose the ball for ten seconds, staying quiet is better than making things up and splitting the
possession in two. **At a possession change it's very expensive**, because you carry the old
label into the new state.

### There's no jury

This changed compared to 2026. **Nothing but the file is scored.** Not originality, not
presentation, not use of LLMs. The repo and the technical document aren't scored: they exist so that
the number means something and can be reproduced.

## The possession rules (what they really ask for)

This is in `CHALLENGE.md` and it's worth reading in full, but the summary:

**Possession change** = a player from the opposing team gains **controlled** possession.

- While play is live, **there is always a team in possession**. There's no "contested".
- In phases of loose or contested ball, **possession stays with the last team that
  had it**.
- After a stoppage there's always a clear possessor. The hard part is detecting when it changes afterwards.

**What does NOT count as a possession change:**
- a mere touch without control
- a deflection of a pass that doesn't intercept it
- an aerial duel won without subsequent control

**What's already settled:**
- Three states and no more. There's no "contested" or "unknown".
- **A possession is a maximal run of one team with the live ball.** `H H H A A` is two
  possessions. `H H D D H H` is two possessions with a stoppage in between, not one.
- Dead ball is a stoppage, **never** a possession.
- **They don't ask whose ball it is when it's dead.** If it's dead, the answer
  is `D`, full stop.

**And the most useful fact of all:** the ground truth **is derived mechanically from the corrected
tracking**. It's not a person watching the video. A frame is `D` where the corrected tracking
marks the ball dead, and otherwise it takes the team the tracking says has it. **Nothing is
smoothed or interpolated afterwards.**

So the question isn't "how did the human label it", it's "what does the corrected tracking
understand by dead and by possessing". And the 5 public minutes are where you can look.

## What can be used and what can't

**Almost everything is allowed**: any language, framework, pretrained model, external data,
LLMs. With two conditions: that the organizers can access and inspect it, and that the
technical document explains it.

Even **improving or replacing the tracking they give us is explicitly legitimate**. They give it to us
so nobody has to build detection and calibration before getting to the real question.

**What is NOT allowed, and disqualifies:**

1. **Labeling by hand, in any amount and for any use, on this match.**
   Not for the submission, not for training, not for validating. And **manually correcting the tracking**
   of this match (ball positions, identity swaps, whatever) counts as hand labeling. On
   **other matches** you can: label whatever you want, as long as the
   organizers can access it and the document explains it.

   > This changed on **October 22 and 23**. It used to say that labeling by hand for training or
   > validating was allowed. No longer. Whoever handles the vision part has to know this.

   And there are two new rules along the same line:

   - **The 5 public minutes are not a training or tuning set.** They are for
     checking that the pipeline is correct. They say so literally: *«That is a rule, not
     just advice»*. That includes drawing medians, weights or constants from there.
   - **Nothing produced by humans about this match, from any source**: no Opta, StatsBomb,
     Wyscout or LaLiga events, no corrected tracking, no annotations. On other
     matches, yes.
2. **Using the private ground truth**, or anything derived from it, beyond the 5
   published minutes.
3. **Submitting a file that the repo can't regenerate.** The same file, not a similar one.

And one that affects us directly: **the terms forbid uploading the video, the data "or
any extract" to a cloud LLM.** Using them to write code is no problem;
pasting ATD rows into them is.

## Schedule

| | |
|---|---|
| data released | October 20 |
| **submission** | **November 17** |
| results | before November 26 |

**24 days** left.

---

# PART 2 — What we propose

## The idea

A **geometric, training-free** model that reads only the tracking. Three pieces:
geometric features → cost per frame and state → a Viterbi that segments.

The design decision that holds everything up is **splitting the question in two**:

```
P(H) = P(live) · P(home | live)
P(A) = P(live) · P(away | live)
P(D) = P(dead)
```

Why? Because the two halves rely on different things:

- **Is the ball in play?** It's answered with the **players' kinematics**. When play is
  stopped people walk; with the live ball they run. **It doesn't need to see the ball.**
- **Whose is it?** It's answered with the **ball-player geometry**. That one does need it.

Since the ball is missing in 41% of the scored frames, putting both questions into a single
classifier means throwing away the half that does work in those frames. Kept separate,
when there is no ball the second one stays at 0.5 (the frame has no opinion) and the decoder
carries over the previous state — which is exactly what `spellF1` rewards.

And ball control is computed as a **sum of `exp(−d/τ)` over all the players of the
team**, not as "the closest wins". With 46 slots that appear and disappear, a decision based on a
single player jumps teams as soon as that slot is lost for one frame.

## Why not a supervised model from the start

Because there are **17 possessions** of ground truth. That's not enough to train. And as shown below,
those 5 minutes moreover don't represent the match.

---

# PART 3 — How I'm going about it

## The rule

Everything I've done today follows from a single methodological decision:

> **No parameter may come from the 5 public minutes.**
> Every value must come either from **labeled external data** (other matches), or from an
> **unlabeled measurement on the full match**.

Since October 22 this is a rule, not a criterion. And it has to be said clearly: **today the model
doesn't fully comply.** I re-derived the dead-ball threshold cleanly, but the
rest of the constants come from before and come out of the public window: the per-class medians
of each signal, the weights chosen by AUC, the smoothing windows, the transition costs
(*«12 of the 16 changes are direct»*) and the minimum duration (*«6 of 17 last
less than 4 s»*). They are **sixteen values** in the configuration file, spread over about ten decisions. All of them have to be re-derived from the 7 external matches and from unlabeled statistics — and that is the first thing to do,
because the technical document has to be able to state that nothing is left from the window.

## The underlying problem: we had nothing to measure with

Until today, any decision of ours rested on 17 possessions. That's very little and, what's more,
misleading. So I spent a good part of the day **building a measuring instrument**.

---

# PART 4 — What I tried and what came out

## First: finding external data

I reviewed the public football tracking candidates (Metrica sample data, SkillCorner,
SoccerNet, StatsBomb, PFF, SoccerTrack). The clear winner is the **IDSSE dataset**:

- 7 full Bundesliga matches, **CC-BY 4.0** license, direct download.
- Tracking at **25 Hz**, same as our video.
- And the important part: each frame carries **`BallPossession`** (1 home, 2 away) and
  **`BallStatus`** (1 live, 0 dead). That is, **the literal `H`/`A`/`D` label, per frame**.
- That's ~1.0 million labeled frames against the 7,500 we have. **×133.**

> A trap that almost cost me everything: the library that loads it returns **the speed in km/h,
> unconverted**. If you don't catch it, the "dead ball" threshold calls every frame
> live and the model trains against garbage without failing anywhere. I detected it by comparing
> with the speed derived from position: ratio 3.590 ≈ 3.6.

## Finding 1: the 5 public minutes don't represent the match

This is the one that supports everything else. Measured on the real ATD:

| | public window | full match |
|---|---|---|
| Ball visible | **88.2%** | **59.3%** |
| Dead-ball frames | **19.6%** | ~45% |

They are the first 5 minutes after kick-off: the stretch with the fewest interruptions and the best
camera tracking. **Any statistic measured there and carried over to the whole match
lands in the wrong place.**

## Finding 2: that was breaking the dead-ball detector

The "dead ball" threshold was set as the crossover of the per-class medians of the two
classes on player activity. The medians were **well measured**. The problem
is that **a crossover of medians depends on the class mix of the sample**.

Transplanted to the full match, that threshold fell at the **56th percentile** of the real
activity, and the model ended up saying `D` in **55.4%** of the frames.

I fixed it by using as the target the dead-ball rate measured on the 7 full matches
of IDSSE (**44.9%**) and the real activity distribution of our match, which is measured
**without using any label**. It's the change that gave the most today.

## Finding 3: the ball isn't lost in occlusions, it's lost in blocks of minutes

```
150 ball-absent blocks | median 10.7 s | max 3.7 minutes
only 9% of the runs last 5 frames or fewer
```

The average possession lasts 14.2 s. **A typical gap eats up almost a whole possession.** There's no
need to ride out half-second occlusions: we have to ride out ten-second blindnesses.

## The POC: a validation set of 7 full matches

The idea: **take a clean IDSSE match and dirty it until its statistics
match those of the ATD**, run our pipeline on it and score against its real labels.

The degradation parameters are measured on our ATD **without using labels**. How the
imitation came out:

| | real ATD | degraded |
|---|---|---|
| Ball visible | 59.31% | 59.39% |
| Objects per frame | 20.49 | 20.46 |
| Ball-absent run, median | 268 frames | 275 |
| Ball-absent run, max | 5,537 frames | 5,537 |

We also had to **align the convention**: the German dataset splits possession at every touch
(421 possessions per match, 7.7 s on average) and the challenge merges them (17 possessions, 14.2 s).
Without correcting that we were asking the model to get 421 possessions right when it's designed to
produce ~220.

## Results

```
spellF1 public 5-min window ............ 0.4727   (we started the day at 0.3200)
spellF1 7 full matches ................. 0.3902   (we started at 0.3156)
official floor ......................... 0.2433
```

**The good number is 0.3902**, not 0.4727. The public window is very optimistic.

And one important point of honesty: **0.3902 isn't a prediction of our score either**.
The proxy has biases in both directions and none is quantified:

- *Optimistic*: it places the ball-absent runs at random, whereas the ATD concentrates them in the
  hard frames. It also doesn't imitate identity swaps between slots.
- *Uncertain*: the Bundesliga is not this match.

It serves to **compare versions against each other**, which is already infinitely more than we had.

## What I tried that did NOT work

**Raising the weight of "ball not visible".** Its measured likelihood ratio is +1.79 and the
model uses +0.50. It looked underweighted. At matched dead-ball rate, **it doesn't improve**, and
beyond 2.5 it collapses.

**Lowering the weight of team shape.** Its real AUC is 0.632 and not the 0.812 that the public
window said, so I lowered it. Validated with 7 matches × 3 seeds: **indistinguishable**.
Reverted.

> **The lesson, which came up twice:** the model sums logits, and that assumes the
> signals are independent. **They are not.** Ball visibility goes from 9.7% to 90.0%
> depending on player activity, because the same thing causes both: the tracker in trouble.
> An AUC or a marginal likelihood **doesn't tell you the right weight**.

---

# PART 5 — Where we are and what's next

Today the pipeline produces a valid CSV of 199,319 rows that passes the official validator, with
55 tests passing. If everything stopped now, there is something to submit.

What I would do next, in order:

0. ~~Re-derive the sixteen values inherited from the public window~~ — **done on Oct 28**
   with `training/recalibrate.py`: class centers by medians over 4 degraded external matches, the rest by sweep over those 4 and checking on 3 held out. 7
   values changed; `config.py` documents the provenance of each one and **none depends any more on the
   public window.** Validated Oct 29 to 31 over the 35 possible splits: the
   procedure gains +1.7 pp (95% CI [+1.4, +2.1]) on matches that were not used to
   tune it and +3.3 pp on the fit matches, so half the gain is overfitting.
   Widening the grid doesn't improve validation (−0.13 pp) and was discarded. On Oct 31
   `activity_smooth` went back to 50 (+1.1 pp on validation with the three seeds). With
   today's file: fit 0.3878, validation 0.4081, all seven 0.3965.
1. **Take the optimism out of the validation set**, making the ball-absent runs
   concentrate where they really concentrate (crowded areas, aerial balls).
2. **Our own ball detector on the video.** The absent 41% is not that the ball can't be
   seen: it's that the tracker loses it. Recovering part of those frames would improve the half of the
   model that is blind today. This connects directly with the computer vision line.
3. **Connect Smart Tagging** as a hint for restarts.
4. **A learned emission** on the degraded external data — but only if the harness says
   there's room.

## How this fits with the other lines

We agreed that each of us would look at one alternative and then compare. From what I've seen
today, **this is not a competition between the three**: the vision one and mine feed each other.

The 41% of frames without ball is the bottleneck of my whole model. It's where the half that
decides "whose is it" goes blind and the only thing left is to carry over the previous state. And
that 41% **isn't that the ball can't be seen in the video** — it's that Metrica's tracker
loses it. Our own detector that recovers even part of those frames **goes straight
into my pipeline without touching the design**, and it would also change the parameters of the
validation set: we'd have to degrade less.

The same goes for Smart Tagging. Its 59 set pieces mark restarts, which is exactly where the model
goes most wrong today.

So when we compare, the useful question may not be *which of the three is better* but
*what each one gives to the others*.

## Things I need someone to look at with a critical eye

Three decisions of mine from today. They have reasoning behind them, but none is a truth:

1. That the dead-ball rate of the German dataset (44.9%) is comparable to what the challenge
   calls `D`.
2. ~~Aligning the convention by aiming at a 14.2 s average duration~~ — withdrawn: that number came
   from the public window. The merge rule now comes from the challenge's definition of *controlled
   possession* (a contact by the opponent of less than a few seconds, without control, doesn't
   change possession), applied to the external matches. The resulting average duration is
   a measurement, not a target.
3. That the pipeline is using **only the tactical ATD**. The homography, Smart Tagging and the
   broadcast ATD have the reading code written but **are not connected**.
   A decision, or was it left half-done?
