"""Smart Tagging: *automatic* tactical annotation of the tactical video.

SportsCode XML, 638 instances in this match. Each one is an interval with a `code`
and a set of grouped tags (`Team`, `Half`, `Type`, ...).

Time base verified by the organizers: `<start>` and `<end>` are seconds from the
first frame of the tactical video, so `round(s * 25)` is a video frame and there is nothing
else to correct.

Three warnings that cost money if ignored:

- **It is not ground truth.** Metrica generates it automatically from the same video and tracking
  that you have, with no manual review. False positives and negatives included.
- **`Team` is not always a team.** 36 instances carry `N/A`, among them *all* the
  throw-ins, exactly where knowing the side would be worth the most.
- **The intervals are phases, not possessions.** A Build Up can span one possession,
  fall short or cover two. Copying an interval as a label imports its errors.

Codes are in uppercase (`SET PIECES`) and tag values in Title Case
(`Kick Off (Start)`), so everything is compared case-insensitively.
"""

import xml.etree.ElementTree as ET

from ..constants import FPS, GAME, HOME_TEAM, AWAY_TEAM
from ..paths import require

SET_PIECES = "set pieces"
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick",
                 "kick off (after goal)", "kick off (start)")


class Phase:
    """One tagging instance, already converted to video frames [start, end)."""

    __slots__ = ("code", "start_frame", "end_frame", "start_s", "end_s", "tags")

    def __init__(self, code, start_s, end_s, tags, fps=FPS):
        self.code = code or ""
        self.start_s = start_s
        self.end_s = end_s
        self.start_frame = int(round(start_s * fps))
        self.end_frame = int(round(end_s * fps))
        self.tags = tags

    def tag(self, group, default=None):
        """Value of a tag group, comparing the name case-insensitively."""
        for key, value in self.tags.items():
            if (key or "").strip().lower() == group.strip().lower():
                return value
        return default

    @property
    def team_label(self):
        """'H', 'A' or None. None includes the explicit `N/A`, which is not a team."""
        value = (self.tag("Team") or "").strip().lower()
        if value == HOME_TEAM.lower():
            return "H"
        if value == AWAY_TEAM.lower():
            return "A"
        return None

    @property
    def is_set_piece(self):
        return self.code.strip().lower() == SET_PIECES

    @property
    def restart_type(self):
        value = (self.tag("Type") or "").strip().lower()
        return value if value in RESTART_TYPES else None

    def __repr__(self):
        return "Phase({!r}, {}-{}, team={})".format(
            self.code, self.start_frame, self.end_frame, self.team_label)


def load(source="tactical", game=GAME, fps=FPS):
    """List of Phase in file order."""
    path = require(["smart_tagging"], source=source, game=game)[0]
    phases = []
    for instance in ET.parse(path).iter("instance"):
        start = instance.findtext("start")
        end = instance.findtext("end")
        if start is None or end is None:
            continue
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        phases.append(Phase(instance.findtext("code"), float(start), float(end), tags, fps))
    return phases


def set_pieces(phases):
    """Only the set pieces: each one marks a restart, that is the end of a dead ball."""
    return [p for p in phases if p.is_set_piece]


def code_counts(phases):
    counts = {}
    for phase in phases:
        counts[phase.code] = counts.get(phase.code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
