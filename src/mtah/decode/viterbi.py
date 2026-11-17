"""Temporal decoding: from per-frame scores to possession segments.

Why this module is the core of the challenge
--------------------------------------------
The ranking is `spellF1`: each possession is matched 1 to 1 with the truth by temporal IoU.
That punishes two things at once and symmetrically -- inventing possessions and missing them.

A majority vote over a window (what the official baselines do) controls the number
of segments *indirectly*, via the window size, and to avoid fragmenting a long possession
it has to erase the short ones. They are two different objectives and a single window cannot
separate them: that is why `baseline_smooth125` wins on number of possessions and loses on
macroF1, and `baseline_smooth25` the opposite.

Viterbi penalizes the number of transitions *directly*, so it can keep a 1 s
possession and at the same time not chop up a 40 s one. Measured with the official scorer on
the public window, starting from the SAME raw label as the baseline:

    majority vote: baseline_smooth50     spellF1 0.3256   macroF1 0.6207   26 poss
    majority vote: baseline_smooth125    spellF1 0.4242   macroF1 0.5812   16 poss
    Viterbi switch_cost=40               spellF1 0.4000   macroF1 0.6686   16 poss

It beats smooth50 on both metrics at once and exceeds the macroF1 of the four baselines.

WARNING: 40 is the value that wins on the 5 public minutes, and that is exactly the mistake
that EVALUATION.md demonstrates with smooth125. The value for the full match has to be
chosen by a criterion other than tuning against that window -- distribution of
durations, cross-validation on external data, or bootstrap.

Latency
-------
Measured by degrading the public ground truth: +1 frame of delay is free (spellF1 1.0000),
+25 frames (1 s) drops to 0.7765 and +50 frames (2 s) to 0.5882. Any causal filter
(backward moving average, EMA, online decoding) is not viable. Viterbi is offline and
centered by construction: it uses the whole sequence, and for that reason it introduces no delay.
"""

import numpy as np

from ..constants import GRADED_BLOCKS

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2
N_STATES = 3


def uniform_switch_costs(switch_cost):
    """Transition cost matrix (3, 3) with the same price for every switch."""
    matrix = np.full((N_STATES, N_STATES), float(switch_cost))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def switch_costs(turnover, to_dead, from_dead):
    """Asymmetric transition matrix, which is how football really behaves.

    In the 5 public minutes, 12 of the 16 possession changes are *direct* (H->A with no
    dead ball in between) and only 4 go through a stoppage. That is: the dominant problem
    is turnovers in live play, not restarts, and charging the same for H->A as for H->D->A
    biases the decoder toward inventing stoppages.

        turnover    H <-> A direct
        to_dead     H|A -> D   (the ball goes dead)
        from_dead   D -> H|A   (restart)
    """
    matrix = np.zeros((N_STATES, N_STATES))
    matrix[H, A] = matrix[A, H] = float(turnover)
    matrix[H, D] = matrix[A, D] = float(to_dead)
    matrix[D, H] = matrix[D, A] = float(from_dead)
    return matrix


def costs_from_hard_labels(labels, miss_cost=1.0, unknown_cost=0.0):
    """Hard per-frame labels -> cost matrix (n, 3), to decode a baseline.

    Cost 0 for the observed state and `miss_cost` for the other two. Where the
    observation is unknown (`X` or anything outside H/A/D) the cost is
    `unknown_cost` for all three, that is: the frame has no opinion and lets its neighbors decide.
    """
    labels = np.asarray(labels)
    cost = np.full((len(labels), N_STATES), float(miss_cost))
    unknown = np.ones(len(labels), dtype=bool)
    for code, name in enumerate(STATES):
        hit = labels == name
        cost[hit, code] = 0.0
        unknown &= ~hit
    cost[unknown] = float(unknown_cost)
    return cost


def check_switch_discount(switch_discount, n_frames):
    """Validates a per-frame discount, as a fraction, on the cost of changing state.

    `switch_discount[t]` multiplies the six switch costs between frames t-1 and t by
    `1 - switch_discount[t]`; `switch_discount[0]` is not used. It is the input of the
    boundary detector (training/boundaries.py): where it sees a transition, switching is
    cheaper.

    A fraction and not a subtraction, because switches cost very different amounts (a
    turnover 40, a restart 14): a subtraction would have to stay below 14 and would barely
    move a turnover. It has to be < 1: with 1 the switch is free and the decoder would
    fragment there without any emission asking for it.
    """
    discount = np.asarray(switch_discount, dtype=np.float64)
    if discount.shape != (n_frames,):
        raise ValueError("switch_discount must have shape ({},), has {}".format(
            n_frames, discount.shape))
    if not np.isfinite(discount).all() or (discount < 0).any():
        raise ValueError("switch_discount must be finite and non-negative")
    if discount.size and discount.max() >= 1.0:
        raise ValueError(
            "switch_discount reaches {:.3f}: with 1 or more a switch is free and makes "
            "fragmenting free. Lower the detector weight.".format(discount.max()))
    return discount


def _viterbi_block(cost, transition, switch_discount=None):
    """Viterbi over a contiguous block. cost (n, 3) cost, lower is better.

    Pure Python loop on purpose: with 3 states it is 9 additions and 3 minima per frame, and
    the per-call overhead of numpy on 3-element arrays is greater than the computation.

    `switch_discount` (n,) scales the six switch costs on entering each frame (see
    check_switch_discount). Without it, or with zeros, the result is bit-for-bit identical to
    what it was before it existed: multiplying by 1.0 changes no float.
    """
    n = len(cost)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    emit = cost.tolist()
    keep = ([1.0] * n if switch_discount is None
            else (1.0 - np.asarray(switch_discount, float)).tolist())
    t00, t01, t02 = transition[0]
    t10, t11, t12 = transition[1]
    t20, t21, t22 = transition[2]

    d0, d1, d2 = emit[0]
    back0, back1, back2 = [], [], []

    for t in range(1, n):
        e0, e1, e2 = emit[t]
        k = keep[t]

        c0, c1, c2 = d0 + t00, d1 + t10 * k, d2 + t20 * k
        if c0 <= c1 and c0 <= c2:
            n0, b0 = c0, 0
        elif c1 <= c2:
            n0, b0 = c1, 1
        else:
            n0, b0 = c2, 2

        c0, c1, c2 = d0 + t01 * k, d1 + t11, d2 + t21 * k
        if c0 <= c1 and c0 <= c2:
            n1, b1 = c0, 0
        elif c1 <= c2:
            n1, b1 = c1, 1
        else:
            n1, b1 = c2, 2

        c0, c1, c2 = d0 + t02 * k, d1 + t12 * k, d2 + t22
        if c0 <= c1 and c0 <= c2:
            n2, b2 = c0, 0
        elif c1 <= c2:
            n2, b2 = c1, 1
        else:
            n2, b2 = c2, 2

        d0, d1, d2 = n0 + e0, n1 + e1, n2 + e2
        back0.append(b0)
        back1.append(b1)
        back2.append(b2)

    out = np.empty(n, dtype=np.int8)
    state = 0 if (d0 <= d1 and d0 <= d2) else (1 if d1 <= d2 else 2)
    out[n - 1] = state
    backs = (back0, back1, back2)
    for t in range(n - 1, 0, -1):
        state = backs[state][t - 1]
        out[t - 1] = state
    return out


def decode(cost, transition, blocks=GRADED_BLOCKS, switch_discount=None):
    """Viterbi per scored block. Returns int8 codes (0=H, 1=A, 2=D).

    It never decodes across the boundary between blocks: the half-time gap is
    20,156 frames and a phantom transition there is an invented possession.
    Frames outside the blocks are left as D, which is not scored.

    `switch_discount` is optional and per frame (see check_switch_discount). The first frame
    of each block is not a transition, so its discount is not used.
    """
    transition = np.asarray(transition, dtype=np.float64)
    if switch_discount is not None:
        switch_discount = check_switch_discount(switch_discount, len(cost))
    out = np.full(len(cost), D, dtype=np.int8)
    for start, stop in blocks:
        block_discount = None if switch_discount is None else switch_discount[start:stop]
        out[start:stop] = _viterbi_block(cost[start:stop], transition, block_discount)
    return out


def enforce_min_duration(codes, min_frames, blocks=GRADED_BLOCKS):
    """Absorbs segments shorter than `min_frames` into the cheapest neighbor.

    Careful not to overdo it: in the 5 public minutes, 6 of the 17 possessions last less than
    4 s and the shortest 0.76 s (19 frames). Measured on the ground truth, erasing
    possessions shorter than 2 s costs 0.17 of spellF1 and those shorter than 5 s costs 0.44.
    A minimum above ~0.5 s starts to destroy real possessions.
    """
    if min_frames <= 1:
        return codes.copy()

    out = codes.copy()
    for start, stop in blocks:
        window = out[start:stop]
        changed = True
        while changed:
            changed = False
            breaks = np.flatnonzero(np.diff(window)) + 1
            bounds = np.concatenate(([0], breaks, [len(window)]))
            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                if b - a >= min_frames:
                    continue
                before = window[a - 1] if a > 0 else None
                after = window[b] if b < len(window) else None
                if before is None and after is None:
                    continue
                if before is None:
                    window[a:b] = after
                elif after is None:
                    window[a:b] = before
                else:
                    # Into the longer neighbor, to avoid chaining cascading absorptions.
                    len_before = a - int(bounds[i - 1]) if i > 0 else 0
                    len_after = int(bounds[i + 2]) - b if i + 2 < len(bounds) else 0
                    window[a:b] = before if len_before >= len_after else after
                changed = True
                break
        out[start:stop] = window
    return out


def to_labels(codes):
    """Int codes -> array of letters 'H'/'A'/'D'."""
    return np.array(STATES, dtype="<U1")[np.asarray(codes, dtype=np.int64)]


def segment_table(codes, blocks=GRADED_BLOCKS, fps=25):
    """[(start, stop, label, duration_s)] per block. To inspect by eye."""
    rows = []
    for start, stop in blocks:
        window = codes[start:stop]
        breaks = np.flatnonzero(np.diff(window)) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            a, b = start + int(bounds[i]), start + int(bounds[i + 1])
            rows.append((a, b, STATES[int(codes[a])], (b - a) / float(fps)))
    return rows
