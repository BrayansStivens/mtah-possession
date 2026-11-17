"""IDSSE dataset: seven Bundesliga matches with possession labeled per frame.

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer". Sci Data 12, 195 (2025).  doi:10.1038/s41597-025-04505-y
Data on figshare doi:10.6084/m9.figshare.28196177, **CC-BY 4.0** license with
authorization from the DFL. If this ends up in the submission, the paper has to be cited
and the DFL credited -- it is a condition of the license, not a courtesy.

**Why this dataset and not another.** It has `BallPossession` (1 home, 2 away) and
`BallStatus` (1 live, 0 stopped) in every frame of the ball track, at 25 Hz, which is
exactly the challenge's H/A/D label and exactly the frequency of the tactical video. That
is ~1.0 M labeled frames against the 7,500 of the public ground truth.

Three traps, all three verified by measuring and not by reading documentation:

1. **Speed comes in km/h, not m/s.** kloppy passes the DFL `S` field through without
   converting. Measured on J03WPY: the file's p99 is 22.88 and the p99 of the speed derived
   from position is 6.37, ratio **3.590** -- that is, 3.6 on the nose. Without dividing, the
   mean activity comes out 6.53 instead of 1.81 and `activity_center_ms = 1.75` calls
   everything live. It is the costliest silent failure this import has.

2. **The slots are not the challenge's.** Here there is one slot per player who steps on the
   pitch (~15-16 per team counting substitutes) and identity is correct. The ATD has 46
   identity-less slots that fragment and swap. Turning this into something resembling the
   ATD is the degradation module's job, not this one's: this module delivers the clean data.

3. **The pitch measures 105 x 68 and the challenge's 105.3 x 68.** The coordinates come out
   normalized 0-1 in both cases, so passing them through `features.geometry` scales them
   with the challenge's `PITCH` and every x distance ends up 0.3% long. Below the noise, but
   worth knowing before comparing absolute distances between the two sets.

The ball has coordinates in **every** frame, including those where play is stopped --
verified: 0 frames without a ball position in J03WPY. That is what rules out Metrica
sample-data as the main source, where the ball only exists while play is live.
"""

import os

import numpy as np

from ..io.atd import HOME_CODE, AWAY_CODE, Tracking
from ..paths import cache_dir

# The seven published matches. The ids are the ones kloppy accepts.
MATCHES = {
    "J03WMX": "1. FC Koln vs FC Bayern Munchen",
    "J03WN1": "VfL Bochum 1848 vs Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Dusseldorf vs 1. FC Nurnberg",
    "J03WOH": "Fortuna Dusseldorf vs SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Dusseldorf vs FC St. Pauli",
    "J03WOY": "Fortuna Dusseldorf vs F.C. Hansa Rostock",
    "J03WR9": "Fortuna Dusseldorf vs 1. FC Kaiserslautern",
}

# See trap 1 in the docstring. Measured, not assumed.
KMH_TO_MS = 1.0 / 3.6

# Tolerance of the units check: an elite sprint is around 10 m/s, so any maximum above this
# means the file is not in m/s.
MAX_PLAUSIBLE_SPEED_MS = 12.0


def _allocate_slots(frames, home_team):
    """(slot_of_player, team_of_slot). One slot per player, home players first.

    The home players go before the away players so that `team_of_slot` comes out in blocks,
    as in the ATD XML, and `slots_of()` returns contiguous indices.
    """
    home_ids, away_ids = [], []
    seen = set()
    for frame in frames:
        for player in frame.players_data:
            if player.player_id in seen:
                continue
            seen.add(player.player_id)
            (home_ids if player.team == home_team else away_ids).append(player.player_id)

    order = home_ids + away_ids
    slot_of = {pid: i for i, pid in enumerate(order)}
    team_of_slot = np.array([HOME_CODE] * len(home_ids) + [AWAY_CODE] * len(away_ids),
                            dtype=np.int8)
    return slot_of, team_of_slot


def _to_arrays(frames, home_team, slot_of, n_slots):
    """Walk the frames once and fill the Tracking arrays + the labels."""
    from kloppy.domain import BallState

    n = len(frames)
    players_xy = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    labels = np.empty(n, dtype="<U1")

    for i, frame in enumerate(frames):
        for player, data in frame.players_data.items():
            j = slot_of[player.player_id]
            if data.coordinates is not None:
                players_xy[i, j] = (data.coordinates.x, data.coordinates.y)
            if data.speed is not None:
                players_speed[i, j] = data.speed * KMH_TO_MS

        ball = frame.ball_coordinates
        if ball is not None and ball.x is not None:
            ball_xy[i] = (ball.x, ball.y)
        if frame.ball_speed is not None:
            ball_speed[i] = frame.ball_speed * KMH_TO_MS

        # BallStatus=0 is a referee stoppage, and it overrides possession: the DFL keeps
        # saying who had the ball when play was stopped, but for the challenge that frame
        # is D. That "who had it" is extra signal that is deliberately discarded here,
        # because the challenge's label does not account for it.
        if frame.ball_state == BallState.DEAD:
            labels[i] = "D"
        elif frame.ball_owning_team is None:
            labels[i] = "X"
        else:
            labels[i] = "H" if frame.ball_owning_team == home_team else "A"

    return players_xy, players_speed, ball_xy, ball_speed, labels


def _check_speed_units(players_speed):
    """Fail if the speed does not look like it is in m/s after converting.

    It exists because the conversion depends on kloppy still delivering km/h. The day they
    fix it upstream, this trips instead of silently poisoning the calibration.
    """
    finite = players_speed[np.isfinite(players_speed)]
    if finite.size == 0:
        raise ValueError("no player speed is finite")
    top = float(np.percentile(finite, 99.9))
    if top > MAX_PLAUSIBLE_SPEED_MS:
        raise ValueError(
            "implausible player speed after converting: p99.9 = {:.2f} m/s.\n"
            "kloppy probably already returns m/s and the division by 3.6 is redundant. "
            "Check KMH_TO_MS.".format(top))


def load(match_id="J03WPY", use_cache=True):
    """(Tracking, labels) of an IDSSE match.

    `labels` is an array of 'H'/'A'/'D' of the same length as the tracking, with the same
    convention as `decode.viterbi.STATES`. The first call downloads ~370 MB from figshare
    and takes a minute and a half; afterwards it comes from the cache in `out/cache`.
    """
    if match_id not in MATCHES:
        raise KeyError("unknown match {!r}; available: {}".format(
            match_id, sorted(MATCHES)))

    cache = os.path.join(cache_dir(), "idsse_{}.npz".format(match_id))
    if use_cache and os.path.exists(cache):
        blob = np.load(cache, allow_pickle=False)
        tracking = Tracking(
            blob["ball_xy"], blob["ball_speed"], blob["players_xy"],
            blob["players_speed"], blob["team_of_slot"],
            str(blob["home_name"]), str(blob["away_name"]))
        return tracking, blob["labels"]

    from kloppy import sportec

    # coordinates="metrica" returns 0-1 with origin at top-left over 105x68, which is the
    # challenge's convention. Verified against kloppy's MetricaCoordinateSystem.
    dataset = sportec.load_open_tracking_data(match_id=match_id, coordinates="metrica")
    home_team, away_team = dataset.metadata.teams

    slot_of, team_of_slot = _allocate_slots(dataset.frames, home_team)
    arrays = _to_arrays(dataset.frames, home_team, slot_of, len(team_of_slot))
    players_xy, players_speed, ball_xy, ball_speed, labels = arrays
    _check_speed_units(players_speed)

    tracking = Tracking(
        ball_xy=ball_xy, ball_speed=ball_speed,
        players_xy=players_xy, players_speed=players_speed,
        team_of_slot=team_of_slot,
        home_name=home_team.name, away_name=away_team.name)

    if use_cache:
        np.savez_compressed(
            cache, ball_xy=ball_xy, ball_speed=ball_speed, players_xy=players_xy,
            players_speed=players_speed, team_of_slot=team_of_slot, labels=labels,
            home_name=home_team.name, away_name=away_team.name)
    return tracking, labels
