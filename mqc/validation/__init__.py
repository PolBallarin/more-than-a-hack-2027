"""Validation set: Metrica Sample_Game_3, ball degraded like BAR-ATH, labels from its events.

    from mqc.validation import load, score
    tracking, gt = load()                 # Tracking (ball degraded) + '<U1' H/A/D/X labels
    result = score(gt, my_labels)         # evaluation.possession_metrics.evaluate, unedited

Files: download.py (pinned fetch), sample_game.py (format + labelling rules), degrade.py (ball
removal from BAR-ATH run lengths), run.py (go/no-go: the four baselines on this set).

Smart Tagging does not exist for this game, so `tags` is None here; code under test must cope.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from degrade import SEED, degrade  # noqa: E402
from sample_game import load as load_sample  # noqa: E402


def periods(gt_labels):
    """Half-open row spans of the graded runs (the two periods), derived from the labels."""
    return pm.valid_runs(pm.label_codes(gt_labels))


def load(seed=SEED, degraded=True):
    """(tracking, gt_labels) for the validation game."""
    tracking, gt, _, _ = load_sample()
    if degraded:
        tracking, _ = degrade(tracking, periods(gt), seed=seed)
    return tracking, gt


def score(gt_labels, pred_labels):
    """The official metrics dict for one prediction, from the unedited scorer."""
    return pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(np.asarray(pred_labels)))


def pipeline_labels(tracking, tags=None):
    """The current mqc pipeline (dead_ball.detect -> possession.assign) on a validation Tracking.

    Same calls and same D-contract check as make_submission.build_labels, minus the file I/O.
    """
    from mqc import dead_ball, possession
    dead = dead_ball.detect(tracking, tags)
    labels = possession.assign(tracking, dead, tags)
    if not np.array_equal(labels == "D", dead):
        raise ValueError("possession.assign must write D exactly where dead_ball.detect is True")
    return labels


def score_pipeline(seed=SEED):
    """Official metrics of the current mqc pipeline on the (degraded) validation game."""
    tracking, gt = load(seed=seed)
    return score(gt, pipeline_labels(tracking))
