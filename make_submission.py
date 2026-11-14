"""MesQueUnClub submission: one command from the data bundle to the file we upload.

    pip install -r requirements.txt
    export MTAH_DATA_DIR=/path/to/the/unpacked/bundle
    python make_submission.py               # writes evaluation/results/MesQueUnClub.csv
    python evaluation/validate_submission.py evaluation/results/MesQueUnClub.csv

The pipeline is four steps, each in its own module of mqc/ so that two people can work on it
without touching the same file:

    tracking -> tagging -> dead ball (bool per frame) -> possession (H/A/D/X per frame)

Version 0 of every step reproduces baseline_smooth50 byte for byte, so the file this writes is the
published floor until a module is replaced. Whatever the modules become, this stays a single
deterministic command: the file uploaded is the file this script writes.
"""

import argparse
import os

import numpy as np

from mqc import atd, dead_ball, possession, tagging
from starter.baseline import write_submission

# Params
TEAM = "MesQueUnClub"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "evaluation", "results", TEAM + ".csv")


def build_labels():
    """One H/A/D/X label per tactical video frame."""
    data = atd.load()
    tags = tagging.load()
    dead = dead_ball.detect(data, tags)
    labels = possession.assign(data, dead, tags)

    # The contract between the two modules, checked rather than trusted: D is the dead-ball
    # detector's call and nobody else's.
    if not np.array_equal(labels == "D", dead):
        raise ValueError("possession.assign must write D on exactly the frames dead_ball.detect "
                         "marks dead")
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT_PATH, help="where to write the submission")
    args = parser.parse_args()

    path = write_submission(args.out, build_labels())
    print("wrote {}".format(path))
    print("\nNow validate it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, HERE)))
