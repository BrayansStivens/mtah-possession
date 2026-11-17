"""Construction, writing and validation of the submission file.

Contract (SUBMISSION.md): exact header `frame,label`, 199,319 data rows, `frame` column
from 0 to 199,318 in order, label in {H, A, D, X}, no empty cells.

Design decision: **we never write X on a scored frame.**

This is measured with the official scorer. The scorer forward-fills the X's with your last
label of the same half *before* matching possessions, so in spellF1 an X is
exactly "repeat my previous answer". But macroF1 charges it as a miss. Abstaining
on 20% of the scored frames versus writing that same filler yourself:

    with 20% X                spellF1 0.3610   macroF1 0.5438
    same but filled by hand   spellF1 0.3610   macroF1 0.6192

Identical on the ranking metric, 0.075 worse on the tie-breaker. X is strictly
dominated inside the scored blocks. Outside them nothing is scored, and there X is the
honest answer.
"""

import os

import numpy as np

from .constants import (ABSTAIN, DEAD, GRADED_BLOCKS, KICKOFF_2ND, LABELS, N_FRAMES)

HEADER = "frame,label"


def build(labels_graded, n_frames=N_FRAMES, blocks=GRADED_BLOCKS,
          outside=ABSTAIN, dead_before_2nd_kickoff=True):
    """Full-length label array built from those of the scored blocks.

    `labels_graded` can be:
      - an array of length n_frames, of which only the scored blocks are respected
      - a list of arrays, one per block, in the order of `blocks`
    """
    out = np.full(n_frames, outside, dtype="<U1")

    if isinstance(labels_graded, (list, tuple)) and len(labels_graded) == len(blocks) \
            and not isinstance(labels_graded[0], str):
        for (start, stop), chunk in zip(blocks, labels_graded):
            chunk = np.asarray(chunk, dtype="<U1")
            if len(chunk) != stop - start:
                raise ValueError("block {}-{}: expected {} labels, got {}"
                                 .format(start, stop, stop - start, len(chunk)))
            out[start:stop] = chunk
    else:
        source = np.asarray(labels_graded, dtype="<U1")
        if len(source) != n_frames:
            raise ValueError("expected {} labels, got {}".format(n_frames, len(source)))
        for start, stop in blocks:
            out[start:stop] = source[start:stop]

    if dead_before_2nd_kickoff:
        # The second half is scored from 110,808 but the kick-off is not taken until 113,742.
        # That is 2,934 frames of dead ball, and the fact is public (FAQ of the official repo).
        second_start = blocks[1][0]
        out[second_start:KICKOFF_2ND] = DEAD

    # Design invariant: no X inside the scored blocks.
    for start, stop in blocks:
        abstained = out[start:stop] == ABSTAIN
        if abstained.any():
            raise ValueError(
                "{} scored frames were left as X (e.g. {}). X is dominated inside "
                "the blocks: write your own filler."
                .format(int(abstained.sum()), int(start + np.flatnonzero(abstained)[0])))
    return out


def write(path, labels):
    """Writes the submission CSV. Returns the path."""
    labels = np.asarray(labels, dtype="<U1")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(HEADER + "\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def validate(path, n_expected=N_FRAMES):
    """None if the file is a valid submission, or the reason in one line.

    Reimplements the same checks as the official validator, reading the file
    from disk: what is validated is the file that will be uploaded, not the in-memory array.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(path, dtype={"label": "object"})
    except Exception as error:                              # noqa: BLE001
        return "cannot read: {}".format(error)

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "frame" not in frame.columns or "label" not in frame.columns:
        return "expected columns 'frame' and 'label', got {}".format(list(frame.columns))
    if len(frame) != n_expected:
        return "{} rows != {}".format(len(frame), n_expected)

    frames = frame["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        return "the 'frame' column is not integer (dtype {})".format(frames.dtype)
    if not np.array_equal(frames, np.arange(n_expected)):
        return "the 'frame' column is not 0..{} in order".format(n_expected - 1)

    values = frame["label"].values
    blank = pd.isna(values)
    if blank.any():
        return "empty label in {} rows; write X to abstain".format(int(blank.sum()))

    cleaned = np.array([str(v).strip().upper() for v in values], dtype=object)
    bad = sorted(set(cleaned) - set(LABELS))
    if bad:
        return "labels outside {}: {}".format(list(LABELS), bad[:5])
    return None


def counts(labels):
    labels = np.asarray(labels, dtype="<U1")
    return {label: int(np.sum(labels == label)) for label in LABELS}
