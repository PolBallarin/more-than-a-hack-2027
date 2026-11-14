"""The tracking as arrays indexed by video frame, plus the shared signals built directly on it.

Everything downstream works on a Tracking rather than on the raw file, so the parsing and the
frame alignment live here and nowhere else:

    from mqc import atd
    data = atd.load()               # tactical, the submission's timeline
    data.players[frame, slot]       # (x, y, speed): x, y normalised 0-1, speed m/s, NaN if missing
    data.ball[frame]                # (x, y, s), NaN if missing
    data.slot_team[slot]            # 0 = home (H), 1 = away (A)

Index i is video frame i, which is row i of a submission: the raw file counts from 1, and load()
checks that it is exactly 1..n with nothing missing before subtracting one.
"""

import collections
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames

# Params
HOME, AWAY, NOBODY = 0, 1, -1
# Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" -- all three separators become one, so a line
# is a flat run of 1 + 3 * n_slots + 3 numbers.
FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","

Tracking = collections.namedtuple("Tracking", ["players", "ball", "slot_team", "team_names"])


def slot_teams(meta_path):
    """(slot_team, (home_name, away_name)) from the metadata XML.

    The first <Team> is home -- the FIFA/EPTS convention, and Barcelona for this match.
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slot_team = np.array([HOME if player.get("teamId") == home_id else AWAY
                          for player in root.iter("Player")], dtype=np.int8)
    return slot_team, tuple(team.findtext("Name") for team in teams)


def load(source=DEFAULT_SOURCE, game=GAME, data_dir=None):
    """The whole ATD of one source as a Tracking. A few seconds for the tactical file."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    slot_team, team_names = slot_teams(meta_path)
    n_frames = source_frames(source, game)
    width = 1 + 3 * len(slot_team) + 3

    flat = np.empty((n_frames, width), dtype=np.float64)
    with open(raw_path) as handle:
        for row, line in enumerate(handle):
            if row >= n_frames:
                raise ValueError("{}: more than {} records".format(raw_path, n_frames))
            values = (line.rstrip("\n").replace(FIELD_SEP, VALUE_SEP)
                      .replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP))
            if len(values) != width:
                raise ValueError("{}: line {} has {} values, expected {}".format(
                    raw_path, row + 1, len(values), width))
            flat[row] = values
    if row + 1 != n_frames:
        raise ValueError("{}: {} records, expected {}".format(raw_path, row + 1, n_frames))
    if not np.array_equal(flat[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{} in order".format(raw_path, n_frames))

    players = flat[:, 1:-3].reshape(n_frames, len(slot_team), 3).astype(np.float32)
    ball = flat[:, -3:].astype(np.float32)
    return Tracking(players, ball, slot_team, team_names)


def nearest_team(data, pitch=GAMES[GAME]["pitch"]):
    """Per-frame team of the tracked player nearest the ball; NOBODY where either is missing.

    Distances are compared in metres, because the pitch is not square: in normalised units the
    short axis would be stretched and the wrong player picked near the touchlines. Ties go to the
    lowest slot, as in starter/baseline.py.
    """
    length, width = pitch
    ball_x = data.ball[:, 0].astype(np.float64)[:, None] * length
    ball_y = data.ball[:, 1].astype(np.float64)[:, None] * width
    dx = data.players[:, :, 0].astype(np.float64) * length - ball_x
    dy = data.players[:, :, 1].astype(np.float64) * width - ball_y
    distance = dx * dx + dy * dy  # NaN wherever the ball or that player is missing

    out = np.full(len(distance), NOBODY, dtype=np.int8)
    answered = ~np.all(np.isnan(distance), axis=1)
    out[answered] = data.slot_team[np.nanargmin(distance[answered], axis=1)]
    return out


def centred_count(mask, half_width):
    """For every frame, how many frames of `mask` are True within +/- half_width of it.

    Cumulative sums, so the cost does not depend on the window. Clipped at both ends of the array.
    """
    cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    index = np.arange(len(mask))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(mask), index + half_width + 1)
    return cumulative[high] - cumulative[low]
