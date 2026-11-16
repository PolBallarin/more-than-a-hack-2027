"""Which team has the ball on the frames where it is alive (PLAN.md, task T3).

The contract: assign(data, dead, tags) returns one label per video frame, 'D' on exactly the
frames where `dead` is True and 'H', 'A' or 'X' everywhere else.

Version 1 -- possession is CONTROL, and it only changes hands when the rival controls the ball.

  * Control. A player controls the ball when it is within CONTROL_RADIUS of them -- a stride's
    reach plus tracking noise -- and no opponent is that close too. With players of both teams
    that close the ball is contested, which (CHALLENGE.md) leaves possession where it was.
  * Change of possession. The team changes only when the other team holds control for at least
    SUSTAIN: long enough to be a reception, not a deflection or a player running past the ball.
    The change is dated to the START of that control, which is when the ball changed hands.
  * Nothing else changes it. A pass in flight, a loose or contested ball, a stretch where the
    tracker lost the ball: the team stays the one that last controlled it.
  * After every stoppage the restart has a clear possessor (CHALLENGE.md), so each live stretch
    starts with the team of its first control, carried back to the restart frame. This matters
    for the score: an X right after a D would be forward-filled as D by spellF1 and erase the
    start of the possession.
  * A live stretch with no control at all (ball never close to anyone) takes the team of the
    player nearest the ball on most of its visible frames, and X if the ball is never seen.

X is otherwise not used: inside a possession it is free on spellF1 but a certain miss on macroF1,
while holding the team is right most of the time. Every constant is a physical quantity, none is
tuned on the public five minutes or tied to frames of this match.
"""

import numpy as np

from games import GAME, GAMES
from mqc import atd

# Params
FPS = GAMES[GAME]["fps"]
# A player reaches the ball within ~1 m (a stride / a leg), and player and ball positions each
# carry a few tens of cm of noise: 1.5 m is "can play it".
CONTROL_RADIUS = 1.5
# A reception -- the ball arriving and being taken -- keeps ball and player together for at least
# ~0.3 s; a deflection or a player running past is shorter. The public window has possessions as
# short as 0.8 s, so this must stay well under that.
SUSTAIN = int(round(0.3 * FPS))
# Control runs of one team separated by up to this many frames of noise are one control.
CONTROL_GAP = 2
NOBODY = atd.NOBODY


def _runs(mask):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def control(data, pitch=GAMES[GAME]["pitch"]):
    """Per frame: HOME/AWAY if exactly one team has a player within CONTROL_RADIUS, else NOBODY."""
    length, width = pitch
    bx = data.ball[:, 0].astype(np.float64)[:, None] * length
    by = data.ball[:, 1].astype(np.float64)[:, None] * width
    dx = data.players[:, :, 0].astype(np.float64) * length - bx
    dy = data.players[:, :, 1].astype(np.float64) * width - by
    with np.errstate(invalid="ignore"):
        close = (dx * dx + dy * dy) < CONTROL_RADIUS ** 2       # NaN -> False
    home = close[:, data.slot_team == atd.HOME].any(axis=1)
    away = close[:, data.slot_team == atd.AWAY].any(axis=1)
    out = np.full(len(close), NOBODY, dtype=np.int8)
    out[home & ~away] = atd.HOME
    out[away & ~home] = atd.AWAY
    return out


def _control_spells(ctrl):
    """[(start, stop, team)] runs of one team's control, bridging gaps of <= CONTROL_GAP frames."""
    frames = np.flatnonzero(ctrl != NOBODY)
    if not len(frames):
        return []
    teams = ctrl[frames]
    # A new spell starts where the team changes or the gap since the last control frame is long.
    breaks = np.flatnonzero((teams[1:] != teams[:-1]) | (np.diff(frames) > CONTROL_GAP + 1)) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(frames)]))
    return [(int(frames[a]), int(frames[b - 1]) + 1, int(teams[a])) for a, b in zip(starts, stops)]


def _live_stretch(ctrl, nearest, start, stop):
    """Team code per frame of one live stretch [start, stop)."""
    spells = _control_spells(ctrl[start:stop])
    out = np.full(stop - start, NOBODY, dtype=np.int8)

    current, since = None, 0
    for s, e, team in spells:
        if current is None:
            current, since = team, 0               # the restart's possessor, carried back
        elif team != current and e - s >= SUSTAIN:
            out[since:s] = current
            current, since = team, s
    if current is not None:
        out[since:] = current
        return out

    seen = nearest[start:stop]
    seen = seen[seen != NOBODY]
    if len(seen):
        home = int(np.sum(seen == atd.HOME))
        out[:] = atd.HOME if home >= len(seen) - home else atd.AWAY
    return out


def assign(data, dead, tags):
    """np.ndarray of '<U1', one label per video frame."""
    ctrl = control(data)
    nearest = atd.nearest_team(data)
    codes = np.full(len(dead), NOBODY, dtype=np.int8)
    for start, stop in _runs(~dead):
        codes[start:stop] = _live_stretch(ctrl, nearest, start, stop)

    labels = np.full(len(dead), "X", dtype="<U1")
    labels[codes == atd.HOME] = "H"
    labels[codes == atd.AWAY] = "A"
    labels[dead] = "D"
    return labels
