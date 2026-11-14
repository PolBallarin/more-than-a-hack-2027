"""MesQueUnClub submission: one command from the data bundle to the file we upload.

    pip install -r requirements.txt
    export MTAH_DATA_DIR=/path/to/the/unpacked/bundle
    python make_submission.py               # writes evaluation/results/MesQueUnClub.csv
    python evaluation/validate_submission.py evaluation/results/MesQueUnClub.csv

Version 0 is the published floor -- starter/baseline.py's nearest-player labels, smoothed by a
centred majority vote over +/-50 frames -- so there is a valid, reproducible file to submit before
any of our own modelling exists. Each improvement replaces a step of build_labels() and must keep
this a single deterministic command: the file uploaded is the file this script writes.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "starter"))
import baseline  # noqa: E402  (needs the path above)

# Params
TEAM = "MesQueUnClub"
SMOOTH_WINDOW = 50  # +/- frames of the majority vote, as in baseline_smooth50
OUT_PATH = os.path.join(HERE, "evaluation", "results", TEAM + ".csv")


def build_labels():
    """One H/A/D label per tactical video frame."""
    codes = baseline.build()
    return baseline.smooth(codes, SMOOTH_WINDOW)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT_PATH, help="where to write the submission")
    args = parser.parse_args()

    path = baseline.write_submission(args.out, build_labels())
    print("wrote {}".format(path))
    print("\nNow validate it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, HERE)))
