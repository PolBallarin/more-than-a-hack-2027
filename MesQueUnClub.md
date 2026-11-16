# MesQueUnClub — More than a Hack 2027, Phase 1

Team: MesQueUnClub (Pol Ballarin, Pau Canosa). Repository: this fork of
`metrica-sports/more-than-a-hack-2027`. Submission file: `MesQueUnClub.csv`.

## 1. Approach

A rule-based pipeline on the tactical tracking, with no learned model. Every threshold is a
physical quantity with its justification written next to it in the code; none was tuned on the
public five minutes, and none refers to specific frames of the match.

```
mqc/atd.py         tracking -> arrays indexed by video frame (frame_id - 1)
mqc/tagging.py     Smart Tagging intervals in video frames
mqc/dead_ball.py   bool per frame: is the ball out of play?
mqc/possession.py  H / A per live frame; D exactly where dead_ball says
make_submission.py chains them and writes the CSV
```

We split the problem in two because the metric does: a stray `D` or a flicker to the other team
splits a possession, and spellF1 punishes that more than anything else. So both modules aim for
few, well-placed transitions rather than per-frame accuracy.

**Why not "no ball = dead ball"** (what the baseline does): on this match the ball is missing on
~41% of graded frames, in long blocks (median 10.7 s) during live play, and it is visible during
many stoppages. Visibility is therefore not used as a dead-ball signal at all.

## 2. Dead ball (`mqc/dead_ball.py`)

What a stoppage looks like in tracking, and the signal used for each:

1. **Everybody walks, and at some point nearly stops.** Median speed of the tracked players,
   averaged over ±1 s. A walking stretch (< 1.4 m/s, preferred human walking speed) is a stoppage
   only if it dips below 1.0 m/s (slow-walk lower bound) somewhere; this hysteresis separates
   stoppages from slow build-up. The median, not the mean, so one glitching tracklet cannot swing
   it.
2. **The ball is off the pitch** by more than 1 m (ball size plus position noise). The stoppage
   extends over the 3 s the players need to decelerate after it.
3. **The ball is placed**: visible, on the pitch, below 0.5 m/s for at least 1 s.
4. **The restart is a kick**: the first frame above 5 m/s within 2 s after a placed ball. From
   there the ball is alive whatever the players' speed. Where no kick is seen, the stoppage ends
   where the players speed up again, which is ~2 s after the real restart (players accelerate);
   that lag, measured on the training data (§4), is taken off.
5. **Smart Tagging SET PIECES** as a prior: each tagged window (~35 s, 5 s for kick-offs) must
   contain a stoppage ending at a restart found in the tracking (a kick, else the end of a detected
   stoppage, else the moment players speed back up after their slowest point in the window). The
   tag's own boundaries are not trusted.
6. **Stoppages last**: dead runs under 2 s are dropped, live gaps under 1 s inside a stoppage are
   filled.

## 3. Possession (`mqc/possession.py`)

- **Control**: a player within 1.5 m of the ball (reach plus tracking noise) with no opponent that
  close. Players of both teams that close = contested ball.
- **Change of possession** only when the rival holds control for ≥ 0.3 s (a reception, not a
  deflection or a player running past the ball). The change is dated to the start of that control.
- **Loose ball, pass in flight, contested ball, ball not tracked**: possession stays with the last
  team that controlled it (CHALLENGE.md).
- **Restarts**: each live stretch starts with the team of its first control, carried back to the
  restart frame. This avoids an `X` right after a `D`, which the scorer would fill as `D`.
- **No control in a whole live stretch**: the Smart Tagging possession phase covering most of it
  (PROGRESSION, BUILD UP, CREATING CHANCES, ATTACKING TRANSITION, whose Team is the side with the
  ball); else the team of the nearest player on most visible frames; else `X` (~1% of live frames).
- `X` is not used inside possessions: it is free on spellF1 but a certain miss on macroF1.

## 4. Assets and training data

| asset | used? | how |
|---|---|---|
| tactical tracking (ATD) | yes | everything above |
| tactical Smart Tagging | yes, as a fallback only | set-piece windows (dead ball), attacking phases (possession without control) |
| tactical homography | no | the ATD is already in pitch coordinates |
| broadcast tracking / homography | no | not synchronised to tactical, weak (DATA.md) |
| videos | no | |
| Smart Tagging DEFENDING mirrors, unmirrored codes (LONG BALL, BALL IN FINAL THIRD, ...) | dropped | mirrors add nothing; unmirrored codes do not say who has the ball (58–70% agreement) |

**Training / validation data**: only **Metrica Sports Sample_Game_3**
(github.com/metrica-sports/sample-data, commit `e706dd5`, SHA-256 checked; one match, 143,761
frames, 3,620 events labelled by Metrica Sports). Nothing comes from the provided game.

- Per-frame labels derived from its event log by rules in `mqc/validation/sample_game.py`: team of
  the last control event (PASS, CARRY, RECOVERY, SET PIECE, SHOT, BALL LOST); `D` from BALL OUT
  (end), FAULT RECEIVED, goal or shot out, offside or hand ball, until the next SET PIECE or goal
  kick.
- Its ball is degraded to look like ours: removed in blocks bootstrapped from the ball
  present/missing run lengths of our tracking (40.7% missing, median gap 10.7 s), seed 2027,
  blind to the labels.
- **Go/no-go**: we only allowed ourselves to use it for choosing between versions if it ranked
  the four published baselines like the official full-match table (50 > 125 > 25 > raw). It gave
  50 > 25 > 125 > raw (25 and 125 within 0.007), so it was **not** used to choose between
  variants. That official table comes from the private ground truth as published in
  `starter/baseline.py`; it was used for this check only. The sample game was used to diagnose
  failure modes and to measure one physical quantity: the 2 s restart lag of §2.4.
- The public five minutes were used only as pass/fail checks of the pipeline.

Indicative, on the degraded validation game (not used for selection): spellF1 0.2706 for the
baseline pipeline, 0.3697 for ours.

## 5. Limitations and what we would do next

- **Dead-ball boundaries are the main loss.** With perfect stoppages our possession module scores
  ~0.51 spellF1 on validation instead of ~0.37. The stoppage start is the weak point: the
  players' deceleration lags the whistle by a variable amount.
- **Quick possession changes are missed.** On validation we predict ~245 possessions against 401
  (~350 on our match): short possessions without a sustained control are merged into the previous
  one.
- **The validation game is easier than ours**: its players are clean tracks of real people, and
  its ball is absent during almost every stoppage. Only the ball was degraded.
- **The ball-based dead-ball signals (placed ball, kick restart) and the Smart Tagging fallback
  cannot be measured** without the private ground truth; they are included on physical grounds and
  change few frames.
- With more time: learn the control and dead-ball rules from several event-labelled public
  matches, use the homography/video to see the ball when the tracking loses it, and model the
  referee's whistle from collective deceleration.

## 6. Reproduction

```bash
git clone https://github.com/PolBallarin/more-than-a-hack-2027.git
cd more-than-a-hack-2027
python -m venv .venv                               # tested with Python 3.12.10
.venv/bin/python -m pip install -r requirements.txt   # pinned: numpy, pandas (+ deps)
export MTAH_DATA_DIR=/path/to/unpacked/bundle      # holds tactical/ (see data/README.md)
.venv/bin/python make_submission.py                # -> evaluation/results/MesQueUnClub.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/MesQueUnClub.csv
```

Deterministic: no randomness in the pipeline (the validation set uses a fixed seed). Verified by
regenerating from a clean clone and comparing the file byte for byte.

Validation set (optional, downloads ~55 MB of public Metrica data into `$MTAH_DATA_DIR/cache/`):

```bash
.venv/bin/python mqc/validation/run.py --seeds 10
```
