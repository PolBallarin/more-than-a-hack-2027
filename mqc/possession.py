"""Which team has the ball on the frames where it is alive. Owner: Pau (PLAN.md, task T3).

The contract: assign(data, dead, tags) returns one label per video frame, 'D' on exactly the
frames where `dead` is True and 'H', 'A' or 'X' everywhere else.

Version 0 reproduces baseline_smooth50's team call: the team of the player nearest the ball, by
majority vote over +/-50 frames, ties to home. It has no notion of control -- a touch flips the
vote as easily as a pass -- which is what T3 replaces.
"""

import numpy as np

from mqc import atd

# Params
WINDOW = 50  # +/- frames, as in baseline_smooth50


def assign(data, dead, tags):
    """np.ndarray of '<U1', one label per video frame."""
    nearest = atd.nearest_team(data)
    home = atd.centred_count(nearest == atd.HOME, WINDOW)
    away = atd.centred_count(nearest == atd.AWAY, WINDOW)

    labels = np.where(home >= away, "H", "A")
    labels[dead] = "D"
    return labels
