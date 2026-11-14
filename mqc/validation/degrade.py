"""Remove the validation ball the way OUR tracking loses it.

Sample_Game_3's ball is almost perfect while play is alive (missing on ~3% of alive frames) and
almost always absent while it is dead. BAR-ATH loses it on ~41% of graded frames, in long blocks
(median ~11 s) that do not line up with stoppages. A validation set that keeps the sample ball
would reward "no ball = dead ball", which is false on our match.

So the degradation is an alternating renewal process -- present block, missing block, present
block, ... -- whose block lengths are drawn (bootstrapped) from the present/missing run lengths
measured on BAR-ATH's graded frames. It is:

  * derived from our data, never set by hand: the run lengths are measured, not typed in;
  * blind to the labels: it never looks at the validation GT, so it cannot leak it;
  * overlaid on the sample's own missingness (union), not a replacement for it;
  * seeded: same SEED, same mask, same score, on every machine.

It is NEVER to be changed to make a score look better (PLAN.md, T1).

Only the ball is degraded. Our player tracking is also worse than the sample's (46 tracklets,
gaps, no identity); that is a known limitation, reported rather than simulated.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, half_video_blocks, require, source_frames  # noqa: E402

# Params
SEED = 2027
STATS_VERSION = 1   # bump if the measurement below changes, so stale caches are not reused


def _runs(mask):
    """(lengths of True runs, lengths of False runs) of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    true_runs = stops - starts
    inv = np.diff(np.concatenate(([0], (~mask).astype(np.int8), [0])))
    false_runs = np.flatnonzero(inv == -1) - np.flatnonzero(inv == 1)
    return true_runs, false_runs


def measure_bar_ath(data_dir=None):
    """Ball missing/present run lengths (frames) on BAR-ATH's graded halves. Cached.

    Only the ball field of each ATD line is inspected, and only aggregate run lengths leave this
    function. Runs are cut at the half boundaries, so a run touching one is censored; it is kept,
    as the halves are long enough (~53 min) for that to be a handful of runs.
    """
    root = data_dir or DATA_DIR
    cache = os.path.join(root, "cache", "bar_ath_ball_runs_v{}.npz".format(STATS_VERSION))
    if os.path.isfile(cache):
        stored = np.load(cache)
        return stored["missing"], stored["present"]

    (raw_path,) = require(["atd_raw"], GAME, data_dir)
    n = source_frames("tactical", GAME)
    missing = np.ones(n, dtype=bool)
    with open(raw_path) as handle:
        for line in handle:
            head, _, rest = line.partition(":")
            ball = rest.rpartition(":")[2]
            if ball and ball[0] != "N":
                missing[int(head) - 1] = False      # ATD frame_id counts from 1

    miss_runs, pres_runs = [], []
    for _, first, last in half_video_blocks(GAME):
        m, p = _runs(missing[first:last + 1])
        miss_runs.append(m)
        pres_runs.append(p)
    miss, pres = np.concatenate(miss_runs), np.concatenate(pres_runs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, missing=miss, present=pres)
    return miss, pres


def summary(missing_runs, present_runs, fps=25):
    total_missing, total = missing_runs.sum(), missing_runs.sum() + present_runs.sum()
    return {"missing_share": float(total_missing) / total,
            "n_missing_runs": int(len(missing_runs)),
            "missing_median_s": float(np.median(missing_runs)) / fps,
            "missing_p90_s": float(np.percentile(missing_runs, 90)) / fps,
            "missing_max_s": float(missing_runs.max()) / fps,
            "present_median_s": float(np.median(present_runs)) / fps}


def mask(n_frames, spans, missing_runs, present_runs, seed=SEED):
    """Boolean (n_frames,) True where the ball is removed, generated inside each span.

    spans: [(start, stop)] half-open row ranges (the validation periods). The first block of a
    span is missing with probability equal to the measured missing time share.
    """
    rng = np.random.default_rng(seed)
    share = missing_runs.sum() / float(missing_runs.sum() + present_runs.sum())
    out = np.zeros(n_frames, dtype=bool)
    for start, stop in spans:
        pos, gone = start, bool(rng.random() < share)
        while pos < stop:
            length = int(rng.choice(missing_runs if gone else present_runs))
            if gone:
                out[pos:min(pos + length, stop)] = True
            pos += length
            gone = not gone
    return out


def degrade(tracking, spans, seed=SEED, data_dir=None):
    """Copy of tracking with the ball set to NaN on the generated mask; also returns the mask."""
    missing_runs, present_runs = measure_bar_ath(data_dir)
    removed = mask(len(tracking.ball), spans, missing_runs, present_runs, seed)
    ball = tracking.ball.copy()
    ball[removed] = np.nan
    return tracking._replace(ball=ball), removed
