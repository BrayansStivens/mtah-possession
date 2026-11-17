"""Every hyperparameter in one place, with the provenance of each number next to it.

A Python dict and not a YAML on purpose: one dependency fewer, and the deliverable requires
that this repo reproduce the exact CSV.

**How NOT to choose these numbers.** EVALUATION.md shows that tuning a single
hyperparameter against the 5 public minutes picks the wrong value for the full match
(`baseline_smooth125` wins there and loses on the match). So:

**Since Oct 22, 2026 this is a rule, not advice**: RULES.md says the 5 public minutes
"are not training data and not a tuning set". Until Oct 28 sixteen values here came from
that window (per-class medians, AUC weights, windows, transition costs, minimum duration).
They were re-derived that day with `training/recalibrate.py`, and
**no value in this file depends on the public window any more**. There are two sources:

- **Label-free statistics of the real ATD** (global IQRs over the 158,857 scored frames,
  object histogram, ball-absent runs). They use no label at all.
- **The seven IDSSE matches** (Bundesliga, CC-BY 4.0, see `training/idsse.py`),
  degraded to resemble the ATD (`training/degrade.py`) and with the challenge's possession
  convention (`training/convention.py`). The class centers are medians measured there. The
  rest was chosen or confirmed by coordinate-descent search maximizing mean spellF1 on
  4 FIT matches (J03WMX, J03WN1, J03WOH, J03WOY) and checked on 3 VALIDATION matches
  (J03WPY, J03WQQ, J03WR9) that had not been touched. Compared with the previous version
  the improvement was +0.004 and +0.009 spellF1 on two new degradation seeds:
  small, but with the same sign on all three.

  With this file **exactly as it stands**, degradation seed 0: fit 0.3878,
  validation 0.4081, all seven 0.3965. (Until Oct 31 this read "fit 0.3905,
  validation 0.3957". The 0.3905 was correct; the 0.3957 did not come from this file but
  from the `recalibrate.py` dump, which stores the three class centers unrounded.
  Rounded as they are below they give 0.3968. That is 0.001 of luck in the rounding, not a
  measurement error, but the number quoted has to be the one the file produces.)

**Validation of the procedure, Oct 29-31.** What was measured was the whole coordinate
descent, not one configuration: the same procedure over the 35 possible splits of 4
fit matches / 3 validation matches. It gains **+1.7 pp** (95% CI [+1.4, +2.1]) on the
matches that each descent did not use to tune itself and +3.3 pp on those it did, so
roughly half of the gain is overfitting. It is not valid to say there was no
overfitting. Seven hyperparameters were pinned at an edge of their value list, so
the grid was widened (64 -> 82 values) and the 35 splits were repeated: +0.62 pp on the
fit set and **-0.13 pp** on validation, improving in 14/35. That is why the grid of
`training/recalibrate.py` stays as it is. The detail does not live in this repo, which is
the submission one, but in the team's working material (`CROSS-VALIDATION.md`).

Each value has its source next to it. RULES.md requires that the technical document be able
to explain any external data that enters the method, and SUBMISSION.md that it list every
labeled dataset: source, who labeled it and how much. Here is the answer.
"""

DEFAULT = {
    # ---------------------------------------------------------------- features
    "control_tau_m": 3.0,          # control radius of the exponential kernel, in meters
    "count_radii_m": (2.0, 5.0),

    # Widths of the centered smoothing windows, in frames (25 fps). Until Oct 28 all four
    # were +-50, chosen by AUC on the public window. Re-derived in the IDSSE harness
    # (sweep 6/12/25/50/100, plus 200 for shape):
    #   activity   +-50  (6:.3754 12:.3761 25:.3755 50:.3747 100:.3657). **12 -> 50 on
    #              Oct 31.** That sweep is flat: between 6 and 50 there is 0.14 pp and 12 beat
    #              50 by 0.0014, which distinguishes nothing. Three independent measurements
    #              choose 50 and none chooses 12: (a) over the 35 splits the descent picks 12
    #              in 1/35 -- 3/35 with the widened grid -- and the mode is 50; (b) setting 50
    #              and touching nothing else gives +1.11 pp on average on the 3 validation
    #              matches, with the three degradation seeds of the same sign (+1.14 / +1.40 /
    #              +0.81) and improving on 5 of the 7 matches; (c) on the fit set the change is
    #              neutral, +0.12 pp on average, not the loss the sweep suggested.
    #              PROVENANCE WARNING: 50 coincides with the value that was there before Oct 28,
    #              which came from the public window and was therefore inadmissible. The
    #              coincidence is only that. The current 50 is justified with the IDSSE
    #              matches, in (a), (b) and (c), without looking at the public window; if the
    #              measurement had given another number, another number would be here.
    #   ball speed +-6   (6:.3871 12:.3848 25:.3838 50:.3770). The shorter the better;
    #              2 and 4 give +0.001, noise. With the grid widened downward the mode drops
    #              to 3 (15/35) and 1 (12/35), but that arm does not improve on validation:
    #              it stays.
    #   control    +-50  (6:.3304 12:.3419 25:.3608 50:.3761 100:.3735). The other way round: with
    #              the ball absent in runs of 10 s median, the nanmean needs a wide window
    #              to bridge the gaps. Confirmed, unchanged: 31/35 splits choose
    #              it, the most stable of the four.
    #   shape      +-200 (12:.3769 ... 100:.3816 200:.3838; 300:.3871 450:.3867).
    #              CORRECTION of Oct 31: this used to say that 300 and 450 "already drop" and
    #              that is false, those two numbers are higher than the one for 200. The widened
    #              grid also prefers them (300 in 16/35, 450 in 11/35). It stays at 200 anyway,
    #              because that arm gains +0.62 pp on the fit set and -0.13 pp on the matches
    #              it did not see: what it picks does not generalize. It is a slow, weak signal
    #              (AUC 0.63): it only works as an 8 s prior.
    "activity_smooth": 50,
    "ball_speed_smooth": 6,
    "control_smooth": 50,
    "shape_smooth": 200,

    # ------------------------------------------------- dead-ball detector
    # 1.75 -> 1.26 on 2026-10-24. Structural failure, not a measurement one.
    #
    # The 1.75 came from the crossing of per-class medians on the public window (live 2.296,
    # dead 1.204) and those two medians are correct. The problem is that a crossing of
    # medians depends on the CLASS MIX of the sample, and the public window has
    # 19.6% dead ball whereas a full match is around 45%. Transplanting that crossing to
    # another mix puts it at the wrong percentile.
    #
    # Measured: 1.75 falls at the **56.3rd percentile** of the match's real activity, and the
    # model ended up predicting D on 55.4% of the scored frames. It matches exactly.
    #
    # The target rate comes from the 7 IDSSE matches (training/idsse.py), which are
    # full matches and therefore have the right class mix:
    #     41.5  51.7  44.3  47.6  40.7  46.5  41.9   ->  mean 44.9%, range 40.7-51.7
    # The value 1.26 is the one that makes the full pipeline predict D at the expected rate.
    # Measured on Oct 31 over the 158,857 scored frames of the ATD: **44.9%**, against the
    # 44.9% mean of the seven IDSSE matches. Spot on.
    #
    # This used to read "45.3%" and that was another number the file no longer produced: with
    # activity_smooth at 12, as it was until Oct 31, the measured rate is 45.8%. The
    # activity smoothing and this center are coupled -- the center is set by percentile over
    # the smoothed signal -- so changing the window moves the rate. It moved in our favor:
    # from 45.8% to 44.9%, closer to the target. If activity_smooth is touched again, this
    # rate has to be measured again.
    #
    # Both ingredients are independent of the public window: the rate comes from
    # external data and the activity distribution is measured over the 158,857 scored frames
    # without using any label. It was not chosen by maximizing spellF1 there.
    # (For the record: on the public window spellF1 rises from 0.3385 to 0.4824 and macroF1
    # from 0.7181 to 0.8021. The improvement is monotonic over the whole 1.75 -> 1.26 sweep,
    # not a peak, and of a size far above the ~0.04 noise band of EVALUATION.md.)
    #
    # The scale still comes from the global IQR (1.037 - 2.305), which does not depend on the
    # mix. Sweep in the harness 1.10/1.18/1.26/1.35 -> 1.26 (.3872/.3845/.3905/.3815): the
    # value that comes from matching the prior of D is also the best on external matches.
    "activity_center_ms": 1.26,
    "activity_scale_ms": 0.45,
    "activity_weight": 1.0,        # IDSSE harness: 0.5/1.0/1.5/2.5 -> .3793/.3905/.3753/.3599

    # CONVENTION FOR THE THREE CLASS CENTERS (this one, control_center and shape_center):
    # they are written rounded to two or three significant figures, and if they are re-derived
    # with training/recalibrate.py they are rounded the same way before pasting them here. The
    # script dumps them with sixteen digits, but they are the median of a float32 signal over
    # matches degraded with one seed: from the third digit on they change with the seed, so
    # writing them in full feigns a precision that the measurement does not have. Checked on
    # Oct 31 that rounded and unrounded give the same spellF1 to four decimals on the
    # fit set, validation and all seven matches. (Until that day the docstring cited a
    # validation number that only came out with the unrounded centers; hence this note.)
    #
    # Crossing of per-class medians over the 4 IDSSE fit matches, degraded:
    # live 6.279 m/s, dead 0.000 -> 3.14. (Before 4.10, from the public window.) Dead
    # gives 0 because a truly dead ball does not move; on the public window it came out 1.84
    # because there "dead" included the ball rolling toward the restart. The scale remains
    # the global IQR of the ATD (0.0 - 6.8), without labels.
    "ball_speed_center_ms": 3.14,
    "ball_speed_scale_ms": 2.00,
    "ball_speed_weight": 0.6,      # IDSSE harness: 0/0.3/0.6/1.2 -> .3524/.3752/.3905/.3899

    # Not seeing the ball is weak evidence of a stoppage. (The previous version justified it
    # with rates from the public window, 93.5% live / 66.4% dead, which moreover cannot hold
    # for the full match: with them the overall visibility would be 0.81 and it is 0.59.)
    # Value confirmed in the IDSSE harness: 0/0.25/0.5/1.0 -> .3820/.3884/.3905/.3744.
    # Tried 2026-10-24: raising it to the implied likelihood ratio (+1.79, measured on the
    # real ATD) does NOT improve. With the D rate matched by bisection, spellF1 drops from
    # 0.4824 to 0.4686 and from 2.5 on it sinks. Reason: that +1.79 is a MARGINAL ratio
    # and here the logits are summed, which assumes conditional independence. The ball's
    # visibility goes from 9.7% to 90.0% depending on activity -- both signals are caused by
    # the same thing, the tracker in trouble. Raising it is counting the evidence twice.
    "ball_unseen_logit": 0.5,      # AUC 0.7700
    # Ball tracked outside the pitch: hard evidence but of low coverage, because
    # when the ball goes out the tracker usually loses it instead of following it outside.
    "ball_out_logit": 2.5,

    # ------------------------------------------------------ team ownership
    # Crossing of medians of the control difference over the 4 IDSSE fit matches,
    # degraded: H 0.212, A -0.196 -> 0.008. (Before 0.096 from the public window,
    # H 0.356 / A -0.164: with only 9 away-team possessions the A side came out short.) That the
    # crossing lands at ~0 is what is expected of a signal symmetric between teams.
    # The ATD's global IQR is narrow (-0.041 to 0.173), so the slope has to be high
    # or the signal ends up flat.
    "control_center": 0.008,
    "control_weight": 6.0,         # IDSSE harness: 2/4/6/9/13 -> .3559/.3861/.3905/.3871/.3755

    # Team shape: the only team signal that does NOT need the ball, and that is why it
    # carries the 41% of scored frames without it. Crossing of medians over the 4 IDSSE
    # fit matches, degraded: H 1.046, A 0.905 -> 0.976. (The public window
    # gave 0.975 with H 1.195 / A 0.756: same center, classes much more separated than
    # they really are -- the real AUC is 0.63, not 0.81.) Global IQR 0.742 - 1.652.
    "shape_center": 0.976,

    # shape_weight: stays at 1.6, but the documented AUC was wrong.
    #
    # 2026-10-24. The 0.8124 of the previous version was measured on the public window (17
    # possessions) and is noise: measured over the SEVEN IDSSE matches (~670 min, see
    # training/idsse.py) the real AUC is 0.632, range 0.586-0.670, no match above
    # 0.68.
    #
    # With that figure I lowered the weight to 0.6 by rescaling with d-prime. When validating
    # it on the 7 degraded matches (training/harness.py), 3 seeds each, it turns out that it
    # **makes no difference**:
    #     1.6 -> spellF1 0.2880 +- 0.0067   macroF1 0.7638 +- 0.0032
    #     0.6 -> spellF1 0.2892 +- 0.0056   macroF1 0.7616 +- 0.0011
    # The difference (+0.0013 / -0.0022) is much smaller than the dispersion between seeds.
    # They are indistinguishable, so 1.6 is kept, which is the value with provenance in the
    # method documented in this file.
    #
    # The lesson, which holds for ALL the weights here: this is a sum of logits, that is, it
    # assumes conditional independence, and the features are not independent. The marginal
    # AUC of team shape is low because it is averaged over all frames,
    # but its value is concentrated in the 41% of frames without the ball, where it is the only
    # team signal left. A marginal statistic does not determine the right weight.
    # The same mistake was made with ball_unseen_logit (see below).
    #
    # Correction of Oct 28: the above was measured with shape_smooth = 50. With the
    # re-derived window (+-200) the signal DOES matter: 0/0.5/1.0/1.6/3.0 -> .3629/.3739/.3866/
    # .3905/.3829. A signal of AUC 0.63 frame by frame is worth nothing per frame, but
    # averaged over 8 s it is a useful team prior right where there is no ball.
    "shape_weight": 1.6,           # IDSSE harness, with shape_smooth 200

    "prob_floor": 0.02,            # probability floor, so that no frame is infinite

    # ------------------------------------------------------------ decoder
    # Asymmetric transition costs, in nats. The asymmetry (leaving D cheap, entering it
    # expensive) is kept; the values come from the IDSSE harness, not from the public window:
    #   turnover   8/14/20/28/40 -> .3673/.3854/.3852/.3871/.3905; 55/75/100 give .3918,
    #              a plateau: above 40 the exact value does not matter. Before 28.
    #   to_dead    10/18/26/34/45 -> .3728/.3828/.3860/.3905/.3896. Confirmed.
    #   from_dead  4/8/14/22 -> .3845/.3882/.3905/.3887. Confirmed.
    "switch_turnover": 40.0,
    "switch_to_dead": 34.0,
    "switch_from_dead": 14.0,

    # Minimum segment duration. 12 frames = 0.5 s. In the IDSSE harness 0, 6 and 12 give
    # exactly the same (.3871) and 25 / 50 get worse (.3865 / .3857): with the transition
    # costs above, the decoder no longer produces segments shorter than 0.5 s, so
    # this value is a safety belt, not a lever. It is left at 12.
    "min_duration_frames": 12,

    # ------------------------------------------------------------ boundary detector
    # models/boundary_detector.py: a per-frame score of "possession changes here". The
    # decoder multiplies each frame's switch costs by (1 - weight * score), and
    # all switch costs by the scale. Measured with nested leave-one-match-out on
    # the 7 IDSSE matches x 3 seeds (STATUS.md, Nov 7): +1.17 pp spellF1 over the model
    # without the detector, and +1.07 pp over just raising the scale (positive on all 7
    # matches). The two values are chosen by `python -m mtah.training.boundaries export` over
    # the 7 with out-of-sample scores; they are copied from models/boundary_detector.json.
    # Chosen on Nov 7: weight 0.95 and scale 2.0 (out-of-sample spellF1 0.4111 against
    # 0.4089 with scale 1.5). Both are at the edge of their grid, as in the 2 folds of 7
    # that lost in the measurement; it is not widened: on Oct 31 it was seen that widening overfits.
    # Weight 0 = the Oct 31 model exactly, without loading the detector.
    "boundary_discount_weight": 0.95,
    "boundary_switch_scale": 2.0,

    # ------------------------------------------------------------ Smart Tagging
    # features/tagging.team_tags: +1/-1 where a Metrica phase says which team has the
    # ball. It is added to the home logit after the detector (models/emissions.py). The IDSSE
    # has no tagging, so the weight is chosen with a simulated tagging over its
    # labels (training/team_tags.py). 0 = no tagging, the previous model exactly.
    # Chosen on Nov 8 between two fixed weights (0.5 and 1.0) for robustness, not for maximum:
    # with 0.5 the LOMO over the 7 matches x 3 seeds wins in all the scenarios compatible with
    # the agreement measured on BAR-ATH (+0.47 to +0.90 pp; +0.61 in the one that best fits the
    # four quartiles) and only loses if the tagging fully copied our emission (-1.12), which
    # the measured agreement rules out (it would give 0.807 against 0.762). 1.0 wins more with
    # good tags and drops to +0.10 at the best fit. The automatic per-fold choice was noisier
    # (+0.39). Tables in trabajo/handoff-plugin-data/datos/smart_tagging_robustez.json.
    "smart_tagging_team_weight": 0.5,

    # ------------------------------------------------------------ team classifier
    # models/team_classifier.py: a "does the home team have possession" logit read from the
    # shape and movement of the two teams, without the ball (features/team_signals.py). It is
    # added to the home logit after the detector, like the Smart Tagging (models/emissions.py).
    # It targets the frames without the ball: 41% of the total, where the previous model gets
    # 74.5% right against 87.5% with the ball. Measured with nested leave-one-match-out on the
    # 7 IDSSE matches x 3 seeds (STATUS.md, Nov 12): +16.11 pp spellF1 over the model without
    # the classifier, better in all 21 runs. The weight is chosen by
    # `python -m mtah.training.team_classifier export` over the 7 with out-of-sample logits;
    # it is copied from models/team_classifier.json.
    # Chosen on Nov 12: weight 2.0 (out-of-sample spellF1 0.5712 against 0.4111 without the
    # classifier; 0.5251 with 0.5 and 0.5488 with 1.0). It is the edge of the grid and is not
    # widened: the degraded IDSSE keeps the players and the challenge match does not always,
    # and its signals are distinguishable from the IDSSE ones (domain AUC 0.72), so the gain
    # above is an optimistic bound and a larger weight would bet more on it.
    # Weight 0 = the Nov 8 model exactly, without loading the classifier.
    "team_classifier_weight": 2.0,
}


def merged(**overrides):
    """Copy of DEFAULT with overrides, failing if a name does not exist."""
    unknown = sorted(set(overrides) - set(DEFAULT))
    if unknown:
        raise KeyError("unknown hyperparameters: {}".format(unknown))
    config = dict(DEFAULT)
    config.update(overrides)
    return config
