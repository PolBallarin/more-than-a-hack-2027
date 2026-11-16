"""Smart Tagging intervals in video frames, for the modules that use them as a prior.

A thin wrapper: starter/load_smart_tagging.py already reads the XML and converts seconds to video
frames (round(seconds * 25), end frame exclusive), and checks that time base against both kickoffs.
Each interval is a dict with code, start_frame, end_frame, start_s, end_s and tags.

Two things to remember when using them (see DATA.md): Team is 'N/A' on every throw-in
and kick-off, and on the DEFENDING codes Team is the defending side -- each one mirrors an attacking
phase with the same interval and the opposite team.
"""

from games import GAME
from starter import load_smart_tagging


def load(game=GAME, data_dir=None):
    """[{code, start_frame, end_frame, start_s, end_s, tags}] in file order."""
    return load_smart_tagging.load(game, data_dir)
