"""T1 go/no-go: does the validation set rank the four baselines like the official full match?

    python mqc/validation/run.py                  # fixed seed, the decision
    python mqc/validation/run.py --seeds 10       # also: how stable is that order across masks

Official order on the private full match (from the private GT; used ONLY as this check and
declared in the technical document):

    baseline_smooth50 0.2433 > baseline_smooth125 0.2242 > baseline_smooth25 0.1996 > raw 0.0486

GO if the validation spellF1 puts them in the same order; then it may be used to choose between
our versions. NO-GO otherwise: abandon it and fix parameters by documented physical arguments.

The baseline is re-run here on the validation Tracking with the starter's exact rule: nearest
player (in metres, `mqc.atd.nearest_team`) to the visible ball, D where the ball is not visible,
then `starter.baseline.smooth`, imported unchanged. The current mqc pipeline (v0 = smooth50) is
scored too, through the same calls as make_submission.py.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)

import baseline  # noqa: E402  (starter/baseline.py, unedited)
from degrade import SEED, degrade, measure_bar_ath, summary  # noqa: E402
from mqc import atd  # noqa: E402
from mqc.validation import periods, pipeline_labels, score  # noqa: E402
from sample_game import load as load_sample  # noqa: E402

# Params
VARIANTS = baseline.VARIANTS                         # raw, 25, 50, 125
OFFICIAL_ORDER = ("baseline_smooth50", "baseline_smooth125", "baseline_smooth25", "baseline_raw")
OFFICIAL_SPELL = {"baseline_raw": 0.0486, "baseline_smooth25": 0.1996,
                  "baseline_smooth50": 0.2433, "baseline_smooth125": 0.2242}


def baseline_scores(tracking, gt, pitch):
    codes = atd.nearest_team(tracking, pitch)
    return {name: score(gt, baseline.smooth(codes, window)) for name, window in VARIANTS}


def order_of(results):
    return tuple(sorted(results, key=lambda name: -results[name]["spell_f1_mean"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=0,
                        help="also report the order over this many other seeds (info only)")
    args = parser.parse_args()

    clean, gt, meta, report = load_sample()
    pitch = meta["pitch"]
    spans = periods(gt)
    print("labels:", report)
    counts = {k: int((gt == k).sum()) for k in "HADX"}
    print("GT frames:", counts, " periods:", spans)
    print("BAR-ATH ball runs:", {k: round(v, 3) for k, v in summary(*measure_bar_ath()).items()})

    tracking, removed = degrade(clean, spans, seed=SEED)
    graded = gt != "X"
    alive = (gt == "H") | (gt == "A")
    miss = np.isnan(tracking.ball[:, 0])
    print("validation ball missing: graded {:.1%}, alive {:.1%}, dead {:.1%} (seed {})\n".format(
        miss[graded].mean(), miss[alive].mean(), miss[gt == "D"].mean(), SEED))

    for label, track in (("clean ball (reference only)", clean), ("degraded ball", tracking)):
        results = baseline_scores(track, gt, pitch)
        print("{}:".format(label))
        print("  {:<20} {:>8} {:>8} {:>9} {:>9}   official spellF1".format(
            "", "spellF1", "macroF1", "n_spells", "gt_spells"))
        for name, _ in VARIANTS:
            r = results[name]
            print("  {:<20} {:>8.4f} {:>8.4f} {:>9} {:>9}   {:.4f}".format(
                name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                OFFICIAL_SPELL[name]))
        print("  order: {}\n".format(" > ".join(order_of(results))))

    degraded = baseline_scores(tracking, gt, pitch)
    decided = order_of(degraded)
    go = decided == OFFICIAL_ORDER
    print("official order: {}".format(" > ".join(OFFICIAL_ORDER)))
    print("GO/NO-GO (seed {}): {}\n".format(SEED, "GO" if go else "NO-GO"))

    mine = score(gt, pipeline_labels(tracking))
    print("mqc pipeline (current dead_ball + possession), degraded ball:")
    print("  spellF1 {:.4f}  macroF1 {:.4f}  n_spells {} (gt {})  vs baseline_smooth50 {:.4f}"
          .format(mine["spell_f1_mean"], mine["macro_f1"], mine["n_pred_spells"],
                  mine["n_gt_spells"], degraded["baseline_smooth50"]["spell_f1_mean"]))

    if args.seeds:
        hits = 0
        for seed in range(SEED + 1, SEED + 1 + args.seeds):
            track, _ = degrade(clean, spans, seed=seed)
            order = order_of(baseline_scores(track, gt, pitch))
            hits += order == OFFICIAL_ORDER
            print("  seed {}: {}".format(seed, " > ".join(order)))
        print("same order as official on {}/{} other seeds (info only, not the decision)".format(
            hits, args.seeds))


if __name__ == "__main__":
    main()
