"""Which frames have the ball out of play (PLAN.md, task T2).

The contract: detect(data, tags) returns a bool array over video frames, True where the ball is
dead. possession.assign() writes D on exactly those frames and a team everywhere else.

Version 2 -- physics, not the ball's visibility. "The tracker lost the ball" is not "the ball is
dead": on BAR-ATH the ball is missing on ~41% of graded frames in long blocks during live play,
and is visible during stoppages. v0 equated the two; v1/v2 drop that signal entirely.

What a stoppage looks like in tracking, and the signal used for each:

  1. Everybody walks, and at some point nearly stops. Between the whistle and the restart players
     reposition at walking pace; during live play the median player jogs. A frame is "walking"
     when the MEDIAN speed of the tracked players, averaged over +/- SPEED_HALF_WINDOW, is below
     WALK_SPEED. The median, not the mean, so one glitching tracklet cannot swing it. Slow live
     build-up is also "walking" for a few seconds, so a walking run only counts as a stoppage if
     the median speed drops below STAND_SPEED somewhere in it (hysteresis).
  2. The ball is off the pitch. A visible ball beyond a touch or goal line by more than
     OUT_MARGIN is dead by definition. Those frames are dead, and they seed a stoppage: the
     walking run that follows them is part of the same stoppage even if it starts a little late,
     because players need a couple of seconds to decelerate after the ball leaves.
  3. The ball is placed. A visible ball on the pitch that stays still (< STILL_SPEED) for at least
     MIN_STILL is being placed for a set piece: in live play a ball at rest that long is not
     being played. Those frames are dead.
  4. The restart is a kick. When a still ball is then struck (> KICK_SPEED within KICK_LAG of it
     ceasing to rest), that strike is the restart instant, and the ball is alive from there --
     even if the players are still slow, which they are for a couple of seconds after any
     restart. The stretch between the ball leaving rest and the strike (the taker's touch) is
     dead too. Where no kick is seen, the stoppage ends where the players speed up again, which
     is RESTART_LAG after the real restart; that lag is taken off.
  5. Smart Tagging SET PIECES are a prior that a stoppage exists. The tagged window (~35 s, 5 s for
     kick-offs) surrounds the restart but does not mark it, so the instant is looked for in the
     tracking: a kicked still ball in the window, else the end of a detected stoppage in the
     window, else the moment the players speed back up to walking after their slowest point in
     the window. Everything from that slowest point (extended back over the walking run around
     it) to the restart is dead; everything from the restart to the end of the window is left
     to the other signals. This recovers stoppages the speed signal misses -- quick throw-ins and
     goal kicks where players never slow down much -- without trusting the tag's boundaries.
  6. Stoppages last. A restart needs the ball retrieved and placed, so a dead run shorter than
     MIN_DEAD is noise and is dropped -- a stray D splits a possession, which is what spellF1
     punishes most. Symmetrically, a live run shorter than MIN_ALIVE between two dead runs is
     treated as part of the stoppage.

Every constant is a physical quantity with its reason next to it, none is tuned on the public
five minutes or tied to frames of this match (PLAN.md rules 2 and 3). `tags` may be None (the
validation game has no Smart Tagging); step 5 is then skipped.
"""

import numpy as np

from games import GAME, GAMES
from mqc import atd

# Params
FPS = GAMES[GAME]["fps"]
# Preferred human walking speed is ~1.3-1.4 m/s (Bohannon 1997; Mohler et al. 2007). Below this
# the typical player is walking or standing, which is what a stoppage looks like.
WALK_SPEED = 1.4
# Slow-walk lower bound (~1.0 m/s): a stoppage is a walking stretch in which the typical player at
# some point (almost) stops. Slow live build-up stays a little above walking-but-stopping, so a
# walking run is only dead if it dips below this somewhere (hysteresis: enter below STAND_SPEED,
# stay while below WALK_SPEED).
STAND_SPEED = 1.0
# 1 s either side: long enough to average out a single stride or tracking jitter, short next to
# the shortest real stoppages (~3 s).
SPEED_HALF_WINDOW = FPS
# Ball diameter is 0.22 m and the ball is out only when wholly over the line; 1 m covers that
# plus position noise, so a ball rolling along the line is not called out.
OUT_MARGIN = 1.0
# The quickest restarts (a throw-in taken at once) still take ~2 s from the ball leaving play.
MIN_DEAD = 2 * FPS
# A restart is followed by at least a pass; a live stretch of under 1 s between two stoppages is
# the speed signal flickering, not play.
MIN_ALIVE = FPS
# Players decelerate for a few seconds after the ball goes out (validation: mean speed falls
# from ~1.9 to ~1.4 m/s over the first 2 s). A ball-out frame extends the stoppage forwards to
# the walking run that starts within this lag.
OUT_LAG = 3 * FPS
# Ball speed is a frame-to-frame difference of positions with ~0.1 m noise, i.e. ~2.5 m/s of noise
# per frame; +/- 2 frames (0.2 s) of averaging brings that under the still threshold.
BALL_HALF_WINDOW = 2
# A ball "at rest": well under a walking player carrying it (~1.4 m/s) and above residual noise.
STILL_SPEED = 0.5
# A placement takes at least a second; a ball merely changing direction is still for a few frames.
MIN_STILL = FPS
# A kick, not a hand carry or a roll: a short pass already leaves the foot at > 5 m/s.
KICK_SPEED = 5.0
# The taker usually touches or rolls the ball before striking it, so the kick comes up to ~2 s
# after the ball stops being at rest; the restart instant is the kick itself, not the first touch.
KICK_LAG = 2 * FPS
# After a restart the players need a couple of seconds to accelerate back above walking pace, so
# a stoppage found from their speed ends late. Measured on the Metrica sample data (the declared
# training data, Sample_Game_3): the median detected end is 1.9 s after the event-log restart.
# Rounded to 2 s and removed from every stoppage end that is not an observed kick.
RESTART_LAG = 2 * FPS
SET_PIECE_CODE = "set pieces"


def _runs(mask):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _smoothed(x, half_width):
    """Centred mean of x over +/- half_width, ignoring NaN; NaN where nothing was observed."""
    ok = np.isfinite(x)
    # atd.centred_count sums in int64, so it is only used for the count; the sum is done here.
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    index = np.arange(len(x))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(x), index + half_width + 1)
    total = cumulative[high] - cumulative[low]
    count = atd.centred_count(ok, half_width)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def median_speed(data):
    """Per-frame median speed (m/s) of the tracked players; NaN where none is tracked."""
    speed = data.players[:, :, 2].astype(np.float64)
    out = np.full(len(speed), np.nan)
    tracked = np.isfinite(speed).any(axis=1)
    out[tracked] = np.nanmedian(speed[tracked], axis=1)
    return out


def _ball_metres(data, pitch):
    length, width = pitch
    return (data.ball[:, 0].astype(np.float64) * length,
            data.ball[:, 1].astype(np.float64) * width)


def ball_out(data, pitch=GAMES[GAME]["pitch"]):
    """True where the ball is visible and more than OUT_MARGIN beyond a line."""
    length, width = pitch
    x, y = _ball_metres(data, pitch)
    with np.errstate(invalid="ignore"):
        return ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))


def ball_speed(data, pitch=GAMES[GAME]["pitch"]):
    """Ball speed in m/s between consecutive visible frames, averaged over +/- BALL_HALF_WINDOW."""
    x, y = _ball_metres(data, pitch)
    raw = np.full(len(x), np.nan)
    raw[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS     # NaN across any missing frame
    return _smoothed(raw, BALL_HALF_WINDOW)


def ball_restarts(data, out, pitch=GAMES[GAME]["pitch"]):
    """(still mask, [restart frames]): placed-ball stretches, and the kicks that end them."""
    speed = ball_speed(data, pitch)
    with np.errstate(invalid="ignore"):
        resting = (speed < STILL_SPEED) & ~out
        kicked = speed > KICK_SPEED
    still = np.zeros(len(speed), dtype=bool)
    restarts = []
    for start, stop in _runs(resting):
        if stop - start < MIN_STILL:
            continue
        still[start:stop] = True
        kick = np.flatnonzero(kicked[stop:stop + KICK_LAG])
        if len(kick):
            restarts.append(int(stop + kick[0]))
    return still, restarts


def _set_pieces(tags):
    return [(t["start_frame"], t["end_frame"]) for t in (tags or [])
            if (t.get("code") or "").strip().lower() == SET_PIECE_CODE]


def _apply_set_piece(dead, speed, walking, window, restarts):
    """Make sure a stoppage ends at a restart found inside one tagged set-piece window."""
    first, last = max(0, window[0]), min(len(dead), window[1])
    if last - first < 2 or not np.isfinite(speed[first:last]).any():
        return
    slowest = first + int(np.nanargmin(speed[first:last]))

    kicks = [r for r in restarts if slowest <= r < last]
    ends = [i for i in range(max(slowest, first + 1), last) if dead[i - 1] and not dead[i]]
    if kicks:
        restart = kicks[0]
    elif ends:
        restart = ends[0]
    else:
        with np.errstate(invalid="ignore"):
            moving = np.flatnonzero(speed[slowest:last] >= WALK_SPEED)
        restart = slowest + int(moving[0]) if len(moving) else last

    begin = slowest
    while begin > 0 and walking[begin - 1]:
        begin -= 1
    dead[begin:restart] = True


def _end_at_kicks(dead, restarts):
    """The ball is alive from a kicked restart on, whatever the players' speed says."""
    for restart in restarts:
        if restart < len(dead) and dead[restart]:
            stop = restart
            while stop < len(dead) and dead[stop]:
                stop += 1
            dead[restart:stop] = False


def detect(data, tags, pitch=GAMES[GAME]["pitch"]):
    """np.ndarray of bool, one per video frame, True where the ball is dead."""
    speed = _smoothed(median_speed(data), SPEED_HALF_WINDOW)
    with np.errstate(invalid="ignore"):
        walking = speed < WALK_SPEED                                 # NaN -> False
        standing = speed < STAND_SPEED
    out = ball_out(data, pitch)
    still, restarts = ball_restarts(data, out, pitch)

    # 1-2. Hysteresis: a walking run is a stoppage only if players (almost) stop in it, or the
    # ball leaves the pitch during it.
    dead = np.zeros(len(speed), dtype=bool)
    for start, stop in _runs(walking):
        if standing[start:stop].any() or out[start:stop].any():
            dead[start:stop] = True
    dead |= out
    # A ball-out frame carries the stoppage forward over the deceleration that follows it.
    for start, stop in _runs(out):
        nxt = np.flatnonzero(walking[stop:stop + OUT_LAG])
        if len(nxt):
            dead[stop:stop + nxt[0]] = True

    # 3. A placed ball, up to the kick that restarts play when there is one.
    dead |= still
    for start, stop in _runs(still):
        kick = next((r for r in restarts if stop <= r <= stop + KICK_LAG), None)
        if kick is not None:
            dead[stop:kick] = True

    # 5. Tagged set pieces: a stoppage must end at a restart inside each window.
    for window in _set_pieces(tags):
        _apply_set_piece(dead, speed, walking, window, restarts)

    # 4. A kicked restart ends the stoppage there.
    _end_at_kicks(dead, restarts)

    # 4b. Other stoppages end where the players' speed says, which is RESTART_LAG after the
    # restart: move those ends back, never below MIN_DEAD.
    kicks = set(restarts)
    for start, stop in _runs(dead):
        if stop in kicks or stop >= len(dead):
            continue
        new_stop = max(start + MIN_DEAD, stop - RESTART_LAG)
        if new_stop < stop:
            dead[new_stop:stop] = False

    # 6. Fill short live gaps inside a stoppage, then drop stoppages too short to be real.
    for start, stop in _runs(~dead):
        if stop - start < MIN_ALIVE and start > 0 and stop < len(dead):
            dead[start:stop] = True
    for start, stop in _runs(dead):
        if stop - start < MIN_DEAD:
            dead[start:stop] = False
    return dead
