"""Smart Tagging turned into per-frame features.

It is the biggest piece of free information in the bundle and the easiest to misuse. What
it is: 638 tactical phases detected automatically, 602 with a team tag, and 59 set pieces,
each of which marks a **restart**, that is, the end of a dead-ball run.

What it is NOT, and DATA.md says so explicitly:

- **It is not ground truth.** Metrica generates it from the same video and tracking we
  have, with no manual review. It has false positives and negatives.
- **`Team` is not always a team.** 36 instances come with `N/A`, and among them are
  *the 26 throw-ins*, exactly where knowing the side would be worth the most.
- **The intervals are phases, not possessions.** A Build Up can cover one possession,
  fall short or span two. Copying an interval as a label imports its errors.

Hence everything here goes into the model as *evidence with a measured weight*, never as
an answer. The set-piece tags, for example, cover a window of about 5 s around the
restart and not the instant itself: checked against the two initial restarts, whose tags
(11097-11222 and 113660-113785) contain the actual frames 11185 and 113742.

**The `Team` of a phase is not always the team with the ball.** It depends on the code,
and reading it wrong inverts the evidence. Measured on Nov 7 on BAR-ATH, the fraction of
live frames of each phase in which the current submission predicts the tagged team (no
labels: it is our model against the tag, only to read the semantics of the code):

    BUILD UP .84   PROGRESSION .84   CREATING CHANCES .74   LONG BALL .66
    DEFENDING IN ATTACKING THIRD .16   ... MIDDLE THIRD .16   ... DEFENSIVE THIRD .26

The `DEFENDING IN *` codes name the team WITHOUT the ball. And they mirror the others:
they cover the same intervals as BUILD UP / PROGRESSION / CREATING CHANCES with the
opposite team (same frame counts, to the frame). The other codes with a team are left out:
BALL IN FINAL THIRD / BALL IN THE BOX / PLAYERS IN THE BOX name the attacker but agree
only .65-.72, and ATTACKING / DEFENSIVE TRANSITION agree .55 / .45 -- their interval
straddles the change of possession, so they do not say whose ball it is (perhaps where it
changes).
The per-code reading comes from DeteccionPosesionBalon/tactical_tagging.py.
"""

import numpy as np

from ..constants import N_FRAMES

# Codes whose `Team` is the team with the ball.
CODES_NAMING_TEAM_IN_POSSESSION = frozenset({
    "build up", "progression", "creating chances", "long ball"})
# Codes whose `Team` is the team WITHOUT the ball: the vote goes to the opponent.
CODES_NAMING_TEAM_OUT_OF_POSSESSION = frozenset({
    "defending in attacking third", "defending in middle third",
    "defending in defensive third"})


def possession_team(phase):
    """'H', 'A' or None: the team with the ball according to the phase, reading its code.

    None if the phase has no team or if its code does not say who has the ball.
    """
    tagged = phase.team_label
    if tagged is None:
        return None
    code = phase.code.strip().lower()
    if code in CODES_NAMING_TEAM_IN_POSSESSION:
        return tagged
    if code in CODES_NAMING_TEAM_OUT_OF_POSSESSION:
        return "A" if tagged == "H" else "H"
    return None


def team_votes(phases, n_frames=N_FRAMES):
    """(home_votes, away_votes), each (n,) int16: phases that give the ball to each team.

    Since the DEFENDING phases mirror the possession phases, a frame covered by a mirror
    pair counts two votes for the same team. It is the same evidence twice: the counts say
    how many phases speak, not how much independent information there is.
    """
    home_votes = np.zeros(n_frames, dtype=np.int16)
    away_votes = np.zeros(n_frames, dtype=np.int16)
    for phase in phases:
        team = possession_team(phase)
        if team is None:
            continue
        start = max(0, phase.start_frame)
        stop = min(n_frames, phase.end_frame)
        if stop > start:
            (home_votes if team == "H" else away_votes)[start:stop] += 1
    return home_votes, away_votes


def team_logit(phases, n_frames=N_FRAMES):
    """(n,) votes for the home team minus votes for the away team, reading each code.

    See above. They are summed instead of taking the last one because the phases overlap. Phases of
    different teams coinciding is a contradiction and it cancels out.
    """
    home_votes, away_votes = team_votes(phases, n_frames)
    return (home_votes - away_votes).astype(np.float32)


def team_tags(phases, n_frames=N_FRAMES):
    """(n,) int8: +1 where the tagging gives the ball to home, -1 to away, 0 with no vote.

    The sign of `team_logit`: mirror pairs do not count double and tied contradictions
    cancel out. It is what goes into the model (models/emissions.team_tag_logit).
    """
    return np.sign(team_logit(phases, n_frames)).astype(np.int8)


def weak_labels(phases, n_frames=N_FRAMES):
    """(n,) 'H', 'A' or '' per frame, and (n,) bool for conflict.

    '' where no phase says whose ball it is and also where they contradict each other (the
    conflict is flagged separately). It is not ground truth: they are ~18 s phases generated
    by Metrica from the same video. They are only good as a diagnostic of the model on this
    match.
    """
    home_votes, away_votes = team_votes(phases, n_frames)
    conflict = (home_votes > 0) & (away_votes > 0)
    labels = np.full(n_frames, "", dtype="<U1")
    labels[(home_votes > 0) & ~conflict] = "H"
    labels[(away_votes > 0) & ~conflict] = "A"
    return labels, conflict


def set_piece_window(phases, n_frames=N_FRAMES, pad=0):
    """(n,) True inside a set-piece interval, optionally widened by `pad` frames.

    Where this is True there is a restart nearby, so there is a dead-ball to live-ball
    transition nearby. It is a hint about *where* a transition happens, not about which
    label to assign.
    """
    out = np.zeros(n_frames, dtype=bool)
    for phase in phases:
        if not phase.is_set_piece:
            continue
        out[max(0, phase.start_frame - pad):min(n_frames, phase.end_frame + pad)] = True
    return out


def restart_team_logit(phases, n_frames=N_FRAMES):
    """(n,) team that restarts play, only inside set-piece windows with a team tag.

    Kept apart from team_logit because at a restart the tag is much more reliable: the team
    taking a goal kick or a corner has the ball by definition. Throw-ins are left out
    because they all come with `N/A`.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    for phase in phases:
        if not phase.is_set_piece:
            continue
        label = phase.team_label
        if label is None:
            continue
        start = max(0, phase.start_frame)
        stop = min(n_frames, phase.end_frame)
        out[start:stop] += 1.0 if label == "H" else -1.0
    return out


def build(phases, n_frames=N_FRAMES, set_piece_pad=0):
    return {
        "tag_team": team_logit(phases, n_frames),
        "tag_set_piece": set_piece_window(phases, n_frames, set_piece_pad),
        "tag_restart_team": restart_team_logit(phases, n_frames),
    }
