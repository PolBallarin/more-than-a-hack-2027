# MesQueUnClub — how we tackled possession

Team: MesQueUnClub (Pol Ballarin, Pau Canosa, Oriol Bracons, Robert Cuello)
Repo: https://github.com/PolBallarin/more-than-a-hack-2027 · File: `MesQueUnClub.csv`

## TL;DR

No neural nets, no training loop. We built a small rule-based pipeline on top of the tactical
tracking, where every threshold is a number we can defend physically ("people walk at ~1.4 m/s"),
not one we found by trying values until the score went up. It beats the baseline everywhere we can
measure it, it regenerates the exact same CSV from a clean clone, and the whole thing runs in a
couple of minutes on a laptop.

## Why rules and not a model

We had one long weekend and one match with no labels we were allowed to train on. The only
labelled data we could legally use was a single public Metrica sample game, with cleaner tracking
than ours. Training a model on one match and hoping it transfers to a noisier one felt like a good
way to fool ourselves.

So we went the other way: work out what a stoppage and a possession actually *look like* in
tracking data, write that down as code, and keep it simple enough that we can tell why it got a
frame wrong.

The pipeline is four small modules plus one entry point:

```
mqc/atd.py          parse the tracking into arrays indexed by video frame
mqc/tagging.py      Smart Tagging intervals in video frames
mqc/dead_ball.py    is the ball dead on this frame? (bool)
mqc/possession.py   H or A on live frames, D exactly where dead_ball says so
make_submission.py  glue + write the CSV
```

We split it in two on purpose, because the metric punishes two different mistakes. A stray `D` in
the middle of a possession cuts it in half, and so does a one-frame flicker to the other team.
spellF1 hates both, so both modules optimise for *few, well-placed transitions* rather than
per-frame accuracy.

## Dead ball: stop trusting the ball

The baseline says "no ball, dead ball". On this match that's just wrong: the ball goes missing on
about 41% of graded frames, in long blocks (median ~11 s) during live play, and it's often
perfectly visible during stoppages. So we stopped using visibility as a signal at all and looked
at what the players do instead.

What we use, roughly in order of how much it matters:

1. **Everyone slows down.** During a stoppage people walk around; during play the typical player
   jogs. We take the *median* speed of the tracked players (median, so one glitchy tracklet doesn't
   wreck it), smooth it over ±1 s, and call it walking below 1.4 m/s. That alone over-fires on
   slow build-up play, so we added hysteresis: a walking stretch only counts as a stoppage if at
   some point it drops below 1.0 m/s, i.e. people actually stop. That one change removed most of
   our false stoppages.
2. **The ball is off the pitch.** More than 1 m past a line is out, no discussion. We also extend
   the stoppage over the ~3 s the players need to brake after that.
3. **The ball is being placed.** A ball sitting still (< 0.5 m/s) on the pitch for a second or
   more is almost always someone setting up a set piece.
4. **The restart is a kick.** First frame above 5 m/s within 2 s after a placed ball. From there
   the ball is alive, even if everyone is still strolling. Our first version waited for the kick
   right at the end of the still stretch and caught 1 restart out of 47; the taker usually nudges
   the ball first, so we gave it 2 s and got 23.
5. **Smart Tagging set pieces as a hint, not as truth.** Every SET PIECES window (~35 s) must
   contain a stoppage, but we look for the actual restart instant in the tracking. We never trust
   the tag's start and end.
6. **Stoppages take time.** Anything under 2 s gets dropped, and tiny live gaps (< 1 s) inside a
   stoppage get filled.

One last fix: when there's no kick to anchor the restart, the "players speed up again" signal
fires about 2 s late, because people need a moment to accelerate. We measured that lag on the
sample game (median 1.9 s), rounded it to 2 s and subtracted it. That's the only number we took
from data instead of from first principles, and we're upfront about it.

## Possession: who actually has it

The rule we settled on is basically how you'd explain it to a coach:

- **Control** = a player within 1.5 m of the ball (a stride plus tracking noise) with no opponent
  that close. If both teams are that close, it's a 50/50 and nothing changes.
- **Possession only flips when the other team holds the ball for 0.3 s.** That's a real
  reception, not a deflection or someone running past. When it flips, we date the change to the
  *start* of that control, not to when we became sure about it.
- **Loose ball, pass in the air, scramble, tracker lost the ball:** possession stays with whoever
  had it last. That's what the challenge rules say, and it's also what keeps us from flickering.
- **After every restart**, the stretch starts with the team of the first control, carried back to
  the restart frame. This one matters more than it looks: the scorer fills an `X` with the
  previous label, and right after a stoppage that previous label is `D`, so an early `X` quietly
  deletes the start of a possession.
- **No control anywhere in a live stretch** (rare, ~1% of live frames): we take the team of the
  Smart Tagging attacking phase covering it if there is one, otherwise whoever is closest to the
  ball most of the time, otherwise `X`.

We don't use `X` inside possessions. It's free on spellF1, but it's a guaranteed miss on macroF1
(the tie-break), and holding the last team is right most of the time anyway.

## What we used, what we threw away

| asset | verdict |
|---|---|
| Tactical tracking | the backbone of everything |
| Smart Tagging | kept, but only as a fallback: set-piece windows for dead ball, attacking phases when tracking has nothing |
| Smart Tagging DEFENDING codes | dropped, they just mirror the attacking ones |
| LONG BALL, BALL IN FINAL THIRD, PLAYERS IN THE BOX… | dropped, they don't say who has the ball (58–70% agreement with us) |
| Homography | not needed, tracking is already in pitch coordinates |
| Broadcast feed | dropped: not synced to tactical, and much weaker |
| Video | not used |

## Training and validation data

Everything labelled that we touched comes from **one public game: Metrica Sports
Sample_Game_3** (github.com/metrica-sports/sample-data, pinned to commit `e706dd5`, SHA-256
checked). That's 143,761 frames and 3,620 events, labelled by Metrica Sports. Nothing comes from
the competition match.

How we turned it into a validation set:

- **Labels from the event log.** The team is the one of the last event with the ball (PASS, CARRY,
  RECOVERY, SET PIECE, SHOT, BALL LOST). Dead ball runs from BALL OUT / FAULT RECEIVED / goal /
  offside until the next SET PIECE or goal kick. All of it is written in
  `mqc/validation/sample_game.py`.
- **Made the ball as bad as ours.** The sample game's ball is way too good, so we removed it in
  blocks whose lengths are bootstrapped from *our* tracking's ball gaps (40.7% missing, median gap
  10.7 s). Fixed seed, and it never looks at the labels.
- **A sanity check before trusting it.** We said up front we'd only use it to pick between versions
  if it ranked the four published baselines in the same order as the official full-match table
  (50 > 125 > 25 > raw). It didn't: it swapped 25 and 125, which were 0.007 apart. So we did
  **not** use it to choose variants. We used it to understand failure modes, and to measure that
  one 2 s lag. (That official table comes from the private ground truth as published in
  `starter/baseline.py`; we only used it for this check.)
- **The public five minutes** were only ever a "does the pipeline run and look sane" check.

For reference only, on the degraded sample game: the baseline pipeline scores 0.27 spellF1, ours
0.37.

## Where it breaks (and what we'd do with another week)

- **Stoppage boundaries are where most of the score goes.** If we hand our possession module the
  *true* stoppages on the sample game, it jumps from ~0.37 to ~0.51. The start of a stoppage is
  the weak spot, because players don't all brake at the same time after the whistle.
- **We miss quick turnovers.** We predict ~245 possessions on the sample game against 401 real
  ones. Short possessions without a clean 0.3 s control get merged into the previous one.
- **Our validation set is easier than the real thing.** Its players are clean tracks of real
  people and the ball vanishes during almost every stoppage. We only degraded the ball.
- **Some signals we couldn't measure at all.** The placed-ball, kick-restart and Smart Tagging
  pieces only show up on our match, and we don't have its ground truth. They're in because the
  physics makes sense, and they touch relatively few frames.
- **With more time:** label a few more public games and learn the control and stoppage rules
  instead of hand-writing them, use the video or homography to recover the ball when the tracker
  loses it, and model the whistle from how the whole team decelerates.

## Reproducing it

```bash
git clone https://github.com/PolBallarin/more-than-a-hack-2027.git
cd more-than-a-hack-2027
python -m venv .venv                                  # we used Python 3.12.10
.venv/bin/python -m pip install -r requirements.txt   # pinned with ==
export MTAH_DATA_DIR=/path/to/unpacked/bundle         # folder that contains tactical/
.venv/bin/python make_submission.py                   # -> evaluation/results/MesQueUnClub.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/MesQueUnClub.csv
```

There's nothing random in the pipeline. We checked it the boring way: fresh clone, fresh venv,
regenerate, compare hashes. Same file, byte for byte.

If you want to rerun our validation set (it downloads ~55 MB of public Metrica data into
`$MTAH_DATA_DIR/cache/`):

```bash
.venv/bin/python mqc/validation/run.py --seeds 10
```
