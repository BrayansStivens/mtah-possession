"""Bring the IDSSE labels to the challenge's possession convention.

**The problem.** The DFL `BallPossession` changes on every individual touch: J03WPY has
421 possessions averaging 7.7 s. The challenge does not: CHALLENGE.md defines a change of
possession as the moment a rival gains **controlled** possession, and says that a simple
touch without control, a deflection that does not intercept, or an aerial duel with no
subsequent control are **not** changes of possession. They are two different conventions
for the same phenomenon.

As long as this is not corrected, the harness's spellF1 is sunk for a structural reason that
has nothing to do with the model's quality: it is asked to get 421 possessions right when it
is designed to produce ~220.

**Where the merge rule comes from -- and where it does NOT.** It comes from the challenge's
definition, applied to the external matches. CHALLENGE.md proposes as a proxy for controlled
possession that *«a player keeps the ball for a few seconds»*. Here that is read as: a
rival run shorter than `MIN_CONTROLLED_S`, wedged inside someone else's possession, is a
contact without control and gets absorbed. A run equal or longer is a real change and is
respected.

An earlier version of this module searched for the threshold that gave a mean duration of
14.2 s, which is the mean of the 17 possessions of the public window. It was withdrawn:
since October 22 the rules say that window is not a tuning set, and even if it were only
to interpret the convention, 17 possessions are a thin basis for a constant. Now the
resulting mean duration is a **measurement** that gets reported, not a target that is
sought.

**What is adjusted and what is not.** Only team possessions separated by brief rival
contacts are merged. **Dead ball is not touched**: the `D` rate of IDSSE is already
comparable to the one expected for the full match, and there is no discrepancy to fix there.

**What this reading leaves out, on purpose.** CHALLENGE.md gives a second proxy, *«a team
completes more than two consecutive passes»*, which would need the DFL event data (kloppy
loads it with `sportec.load_open_event_data`). It is not implemented. A rival who completes
three passes in less than `MIN_CONTROLLED_S` is a rare case, but it exists, and this
version would absorb it wrongly.
"""

import numpy as np

# "A few seconds" from CHALLENGE.md, read as 3. It is a reading of the rule's text, not a
# measurement on any match: if someone has better judgement for the number, this is the
# only place that needs touching.
MIN_CONTROLLED_S = 3.0


def spells(labels):
    """[(label, start, end_exclusive)] of maximal runs."""
    labels = np.asarray(labels, dtype="<U1")
    if not len(labels):
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    return [(labels[bounds[i]], int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1)]


def _absorb_uncontrolled(sp, max_len):
    """Absorb team runs shorter than `max_len` flanked by the SAME rival.

    Only when both neighbours are the same team: that is a brief contact inside someone
    else's possession. A brief touch between two different teams, or next to a dead ball,
    is left alone: there is no one else's possession wrapping it there.
    """
    changed = True
    while changed:
        changed = False
        out, i = [], 0
        while i < len(sp):
            lab, a, b = sp[i]
            if (0 < i < len(sp) - 1 and lab in "HA" and (b - a) < max_len
                    and sp[i - 1][0] == sp[i + 1][0] and sp[i - 1][0] in "HA"):
                prev = out[-1]
                out[-1] = (prev[0], prev[1], sp[i + 1][2])
                i += 2
                changed = True
                continue
            out.append((lab, a, b))
            i += 1
        sp = out
    return sp


def _mean_possession_s(sp, fps):
    lengths = [b - a for lab, a, b in sp if lab in "HA"]
    return (float(np.mean(lengths)) / fps) if lengths else 0.0


def harmonise(labels, fps=25, min_controlled_s=MIN_CONTROLLED_S):
    """Labels with the challenge's convention. Returns (labels, info) so it can be audited."""
    labels = np.asarray(labels, dtype="<U1")
    base = spells(labels)
    final = _absorb_uncontrolled(base, min_controlled_s * fps)

    out = labels.copy()
    for lab, a, b in final:
        out[a:b] = lab

    info = {
        "min_controlled_s": float(min_controlled_s),
        "n_poss_antes": sum(1 for l, _, _ in base if l in "HA"),
        "n_poss_despues": sum(1 for l, _, _ in final if l in "HA"),
        "dur_media_antes_s": _mean_possession_s(base, fps),
        "dur_media_despues_s": _mean_possession_s(final, fps),
        "frames_cambiados": int((out != labels).sum()),
        "pct_frames_cambiados": 100.0 * float((out != labels).mean()),
    }
    return out, info
