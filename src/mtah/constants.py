"""Constants of the match and of the submission contract.

Everything that appears here is publicly documented in the organizers' repo
(DATA.md, FAQ.md) or derived from the video itself. Nothing comes from the private ground truth.
"""

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199_319          # frames of the tactical video == rows of the submission
N_TRACK_SLOTS = 46          # 24 Barcelona + 22 Athletic, they are NOT players
PITCH = (105.3, 68.0)       # meters, measured on the corrected tracking (games.py)

HOME, AWAY, DEAD, ABSTAIN = "H", "A", "D", "X"
LABELS = (HOME, AWAY, DEAD, ABSTAIN)

HOME_TEAM, AWAY_TEAM = "Barcelona", "Atletic Club"   # the first Team in the XML is the home team

# Scored blocks, in video frames. Half-open [start, end).
# Published in DATA.md: they are boundaries, not labels.
GRADED_BLOCKS = ((11_185, 90_652), (110_808, 190_198))
N_GRADED = sum(b - a for a, b in GRADED_BLOCKS)       # 158_857

# The second half is scored from 110_808 but the kick-off is not taken until 113_742: those
# 2_934 frames are dead ball. Fact published in the FAQ of the official repo.
KICKOFF_2ND = 113_742

# Maximum plausible ball speed in m/s. Used to discard impossible tracking jumps when
# filling gaps.
BALL_MAX_SPEED = 40.0


def is_graded(frame):
    """True if the frame counts for the score."""
    return any(a <= frame < b for a, b in GRADED_BLOCKS)


def graded_mask(n_frames=N_FRAMES):
    """Boolean mask over all frames, True where scored."""
    import numpy as np
    mask = np.zeros(n_frames, dtype=bool)
    for a, b in GRADED_BLOCKS:
        mask[a:b] = True
    return mask
