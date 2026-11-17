"""Reading of the Automatic Tracking Data (Metrica's FIFA/EPTS export).

Two files: a metadata XML (teams and which team each track slot belongs to) and
a raw TXT with one line per frame:

    <frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>
                 players (46 slots)         ball

Coordinates are normalized 0-1 over the pitch and the speed is in m/s. Missing
observations are NaN, both for players and for the ball.

Two things that are easy to get wrong and expensive to detect:

1. `frame_id` counts from 1 and the submission from 0  ->  video_frame = frame_id - 1.
2. The pitch is not square (105.3 x 68 m). Comparing distances without scaling both axes
   picks the wrong player near the touchlines.

Parsing the TXT takes tens of seconds, so it is cached in out/cache/*.npz.
Delete the cache if you change the bundle.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import FPS, GAME, N_FRAMES, N_TRACK_SLOTS, PITCH
from ..paths import cache_dir, require

HOME_CODE, AWAY_CODE = 0, 1
_TRANSLATE = str.maketrans({":": ",", ";": ","})

# Acceptable band for (speed in the file / speed derived from position). In m/s it comes
# out ~1.0 (measured 0.992 on BAR-ATH); in km/h it would be ~3.6 and in km/h divided by 3.6
# twice, ~0.28. The band is wide on purpose: it separates units, it does not measure precision.
SPEED_RATIO_ACCEPTED = (0.8, 1.25)
# Below this derived speed the ratio is dominated by tracking jitter.
SPEED_RATIO_MIN_DERIVED_MS = 0.5


class Tracking:
    """Tracking of a match, aligned to the video frame (index 0..N_FRAMES-1).

    Attributes (all float32, NaN where there is no observation):
        ball_xy        (n, 2)      normalized 0-1
        ball_speed     (n,)        m/s
        players_xy     (n, slots, 2)
        players_speed  (n, slots)
        team_of_slot   (slots,)    int8, 0 = home, 1 = away
        ball_seen      (n,)        bool, True where the tracker sees the ball
    """

    __slots__ = ("ball_xy", "ball_speed", "players_xy", "players_speed",
                 "team_of_slot", "home_name", "away_name")

    def __init__(self, ball_xy, ball_speed, players_xy, players_speed,
                 team_of_slot, home_name, away_name):
        self.ball_xy = ball_xy
        self.ball_speed = ball_speed
        self.players_xy = players_xy
        self.players_speed = players_speed
        self.team_of_slot = team_of_slot
        self.home_name = home_name
        self.away_name = away_name

    @property
    def n_frames(self):
        return len(self.ball_xy)

    @property
    def ball_seen(self):
        return ~np.isnan(self.ball_xy[:, 0])

    def xy_metres(self, xy):
        """Converts normalized coordinates to meters, scaling each axis."""
        return xy * np.asarray(PITCH, dtype=np.float32)

    def slots_of(self, team):
        """Indices of a team's slots (0 home, 1 away)."""
        return np.flatnonzero(self.team_of_slot == team)

    def summary(self):
        seen = self.ball_seen
        n_tracked = np.sum(~np.isnan(self.players_xy[:, :, 0]), axis=1)
        return {
            "n_frames": int(self.n_frames),
            "slots": int(self.players_xy.shape[1]),
            "home": self.home_name,
            "away": self.away_name,
            "ball_seen_pct": float(100.0 * seen.mean()),
            "mean_tracked_objects": float(n_tracked.mean()),
            "max_tracked_objects": int(n_tracked.max()),
        }


def read_metadata(meta_path):
    """(team_of_slot, home_name, away_name) from the metadata XML.

    The first <Team> in the file is the home team: that is the FIFA/EPTS convention and the one
    kloppy follows. For this match that is Barcelona, which DATA.md confirms separately.
    """
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, got {}".format(meta_path, len(teams)))

    home_id, home_name = teams[0]
    _, away_name = teams[1]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no <Player>".format(meta_path))

    team_of_slot = np.array(
        [HOME_CODE if tid == home_id else AWAY_CODE for tid in slots], dtype=np.int8)
    return team_of_slot, home_name, away_name


def speed_channel_ratio(tracking, min_derived_speed_ms=SPEED_RATIO_MIN_DERIVED_MS):
    """Median of (player speed in the file / speed derived from position).

    The derivative is computed between consecutive frames in meters, so it is NaN in any
    gap and does not cross any absence. Identity swaps between slots produce absurd jumps
    in the derivative, but they are few and the median does not see them.
    """
    xy_metres = tracking.players_xy * np.asarray(PITCH, dtype=np.float32)
    step = xy_metres[1:] - xy_metres[:-1]
    derived_speed = np.sqrt((step ** 2).sum(axis=-1)) * float(FPS)
    declared_speed = tracking.players_speed[1:]
    comparable = (np.isfinite(derived_speed) & np.isfinite(declared_speed)
                  & (derived_speed > min_derived_speed_ms))
    if not comparable.any():
        raise ValueError("no frame allows comparing the speed in the file with the "
                         "speed derived from position")
    return float(np.median(declared_speed[comparable] / derived_speed[comparable]))


def check_speed_channel_units(tracking):
    """Fails if the player speed in the file is not in m/s. Returns the ratio.

    `atd.xml` declares `<Unit>kph</Unit>` and that is false: measured against position the
    channel is in m/s. The next person who reads the metadata will "fix" it by dividing by 3.6,
    and since the players' activity -- the dead-ball detector -- comes from this channel, the
    pipeline would keep running and predicting D almost everywhere without failing anywhere.
    This turns it into an explicit error. (Ported from DeteccionPosesionBalon.)
    """
    ratio = speed_channel_ratio(tracking)
    low, high = SPEED_RATIO_ACCEPTED
    if not low <= ratio <= high:
        raise ValueError(
            "the player speed in the file is {:.3f} times the speed derived from "
            "position; in m/s it would give ~1.0 (km/h would give ~3.6). Do not convert units "
            "by trusting the metadata: measure them again.".format(ratio))
    return ratio


def _parse_raw(raw_path, n_slots, n_frames):
    """Parses the raw TXT into a matrix (n_frames, 1 + 3*n_slots + 3).

    Line by line instead of loading the whole file: it is ~200 MB and doing a replace
    over the whole text triples the memory without gaining noticeable speed.
    """
    n_cols = 1 + 3 * n_slots + 3
    out = np.full((n_frames, n_cols), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            values = line.translate(_TRANSLATE).split(",")
            if len(values) != n_cols:
                raise ValueError(
                    "{}:{}: expected {} fields ({} slots), got {}. "
                    "Check that the file matches this metadata XML."
                    .format(os.path.basename(raw_path), lineno, n_cols, n_slots, len(values)))
            row = np.array(values, dtype=np.float32)
            frame = int(row[0]) - 1          # frame_id counts from 1
            if not 0 <= frame < n_frames:
                raise ValueError("{}:{}: frame_id {} out of range"
                                 .format(os.path.basename(raw_path), lineno, int(row[0])))
            out[frame] = row
            seen[frame] = True

    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError("{}: missing {} frames out of {}".format(
            os.path.basename(raw_path), missing, n_frames))
    return out


def load(source="tactical", game=GAME, n_frames=None, use_cache=True):
    """Loads the tracking. Caches the parse in out/cache for next times."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], source=source, game=game)
    team_of_slot, home_name, away_name = read_metadata(meta_path)
    n_slots = len(team_of_slot)
    if source == "tactical" and n_slots != N_TRACK_SLOTS:
        # Not an error: it only warns that the bundle is not the expected one.
        print("warning: {} slots in the XML, expected {}".format(n_slots, N_TRACK_SLOTS))
    if n_frames is None:
        n_frames = N_FRAMES if source == "tactical" else 164_488

    cache = os.path.join(cache_dir(), "{}_{}_atd.npz".format(game, source))
    stamp = os.path.getmtime(raw_path)
    if use_cache and os.path.exists(cache):
        blob = np.load(cache)
        if float(blob["stamp"]) == stamp and int(blob["n_slots"]) == n_slots:
            tracking = Tracking(blob["ball_xy"], blob["ball_speed"], blob["players_xy"],
                                blob["players_speed"], team_of_slot, home_name, away_name)
            check_speed_channel_units(tracking)
            return tracking

    matrix = _parse_raw(raw_path, n_slots, n_frames)
    players = matrix[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = matrix[:, 1 + 3 * n_slots:]

    tracking = Tracking(
        ball_xy=np.ascontiguousarray(ball[:, :2]),
        ball_speed=np.ascontiguousarray(ball[:, 2]),
        players_xy=np.ascontiguousarray(players[:, :, :2]),
        players_speed=np.ascontiguousarray(players[:, :, 2]),
        team_of_slot=team_of_slot, home_name=home_name, away_name=away_name)
    check_speed_channel_units(tracking)

    if use_cache:
        np.savez_compressed(
            cache, ball_xy=tracking.ball_xy, ball_speed=tracking.ball_speed,
            players_xy=tracking.players_xy, players_speed=tracking.players_speed,
            stamp=np.float64(stamp), n_slots=np.int32(n_slots))
    return tracking
