"""Which frames have the ball out of play. Owner: Pol (PLAN.md, task T2).

The contract: detect(data, tags) returns a bool array over video frames, True where the ball is
dead. possession.assign() writes D on exactly those frames and a team everywhere else.

Version 0 reproduces baseline_smooth50's D: a frame is dead when the tracking located no ball
within +/-50 frames of it. That equates "the tracker lost the ball" with "the ball is not in play",
which is wrong in both directions -- the ball goes missing in long blocks during live play, and is
visible during stoppages -- and is exactly what T2 replaces.
"""

from mqc import atd

# Params
WINDOW = 50  # +/- frames, as in baseline_smooth50


def detect(data, tags):
    """np.ndarray of bool, one per video frame, True where the ball is dead."""
    located = atd.nearest_team(data) != atd.NOBODY
    return atd.centred_count(located, WINDOW) == 0
