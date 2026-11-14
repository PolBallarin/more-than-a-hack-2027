"""Metrica Sample_Game_3 as a `Tracking` plus per-frame H/A/D/X labels derived from its events.

Why this game: it is the only public Metrica match with both tracking and a full event log, in
the same EPTS family as our ATD. It is close enough to reuse, different enough to check first:

  ours (BAR-ATH ATD)                       Sample_Game_3
  -------------------------------------    ---------------------------------------------
  "<frame>:<x,y,s>;... 46 slots:<x,y,s>"   "<frame>:<x,y>;... 22 slots:<x,y>"  (no speed)
  slots are tracklets                      slots are real players, but the slot -> player
                                           mapping changes at every substitution (11
                                           DataFormatSpecification blocks)
  Barcelona / Atletic Club                 Team A (local, -> H) / Team B (-> A)

So the loader checks the line layout against the metadata before parsing anything, gives every
PERSON a fixed slot (35 players, NaN when off the pitch) so `slot_team` stays constant, and
computes speed from positions.

Labels come from the event log with the rules below -- the whole labelling policy is in
`labels_from_events` and these constants:

  * Team in possession = team of the last CONTROL event (PASS, CARRY, RECOVERY, SET PIECE, SHOT,
    BALL LOST) at or before the frame. A CHALLENGE, a loose ball or a pass in flight does not
    change it, which matches the challenge GT ("a loose or contested ball stays with the last
    team"). At one frame, BALL LOST is applied before the other events, so a simultaneous
    RECOVERY by the opponent wins.
  * D from a STOPPAGE event until the next RESTART event:
      stoppages  BALL OUT (at its end frame: start is the kick, end is the ball leaving),
                 FAULT RECEIVED, a SHOT that ends in a GOAL or OUT (end frame),
                 BALL LOST with subtype OFFSIDE or HAND BALL (start frame, the whistle);
      restarts   SET PIECE, and any event with subtype GOAL KICK (goal kicks are logged as a
                 PASS, not a SET PIECE).
    A restart that arrives while the ball is still alive (no stoppage logged since the previous
    one) starts its D at the end frame of the event before it: some stoppage happened, the log
    just did not name it.
  * X outside play: before each period's first restart (the kick-off) and after its last event.
    The two periods are therefore the two graded runs, as in our GT.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _path in (ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from download import fetch  # noqa: E402
from mqc.atd import AWAY, HOME, Tracking  # noqa: E402  (Pol's, the pipeline's own type)

# Params
FPS = 25
# Centred finite difference for speed, +/- this many frames. Positions are 0.04 s apart and
# quantised to 1e-5 of the pitch (~1 mm), so one frame is fine for noise; 2 frames (0.16 s
# baseline) is a physical compromise between noise and lag, not a tuned value.
SPEED_HALF_WINDOW = 2
# Above any human sprint (~12.4 m/s peak, Bolt 2009): such a value is a tracking jump (the file
# splices the halves and substitutions at layout boundaries), so it is set to NaN, not kept.
MAX_SPEED = 13.0

CONTROL = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST")
RESTART_TYPES = ("SET PIECE",)
RESTART_SUBTYPES = ("GOAL KICK",)
STOP_AT_END = {"BALL OUT": None, "SHOT": ("GOAL", "OUT")}      # type -> required subtype (any)
STOP_AT_START = {"FAULT RECEIVED": None, "BALL LOST": ("OFFSIDE", "HAND BALL")}
LOSS_FIRST = ("BALL LOST",)  # applied before other events on the same frame


# -- metadata -------------------------------------------------------------------------------------

def read_metadata(path):
    """Teams, people, pitch, periods and per-frame-range slot layouts from the EPTS XML."""
    root = ET.parse(path).getroot()
    meta = root.find("Metadata")

    params = {p.findtext("Name"): p.findtext("Value")
              for p in meta.iter("ProviderParameter")}
    periods = [(int(params["first_half_start"]), int(params["first_half_end"])),
               (int(params["second_half_start"]), int(params["second_half_end"]))]

    score = meta.find(".//Score")
    home_id = score.get("idLocalTeam")
    teams = {t.get("id"): t.findtext("Name") for t in meta.iter("Team")}
    if len(teams) != 2 or home_id not in teams:
        raise ValueError("{}: expected 2 teams incl. local {}".format(path, home_id))

    people = [(p.get("id"), p.get("teamId")) for p in meta.iter("Player")]
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}

    size = meta.find(".//FieldSize")
    pitch = (float(size.findtext("Width")), float(size.findtext("Height")))

    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        xs, ys = refs[0::2], refs[1::2]
        if not all(x.endswith("_x") for x in xs) or not all(y.endswith("_y") for y in ys):
            raise ValueError("unexpected channel order in layout {}".format(spec.attrib))
        if len(list(spec.iter("BallChannelRef"))) != 2:
            raise ValueError("expected ball x,y in layout {}".format(spec.attrib))
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                        [channel_player[x] for x in xs]))
    return {"home_id": home_id, "teams": teams, "people": people, "pitch": pitch,
            "periods": periods, "layouts": layouts}


# -- tracking -------------------------------------------------------------------------------------

def _check_line(line, n_players):
    """Refuse to parse a file whose line layout is not the one the metadata promises."""
    parts = line.rstrip("\n").split(":")
    if len(parts) != 3:
        raise ValueError("expected 'frame:players:ball', got {} fields".format(len(parts)))
    groups = parts[1].split(";")
    if len(groups) != n_players or any(len(g.split(",")) != 2 for g in groups):
        raise ValueError("expected {} 'x,y' players, got {!r}".format(n_players, parts[1][:80]))
    if len(parts[2].split(",")) != 2:
        raise ValueError("expected ball 'x,y', got {!r}".format(parts[2]))


def speed(xy, pitch, half_window=SPEED_HALF_WINDOW, fps=FPS):
    """Speed in m/s from (n, k, 2) normalised positions by centred difference, NaN at gaps.

    Edges and frames next to a NaN fall back to whatever side is available; a frame with no
    neighbour on either side stays NaN. Never interpolates across missing positions.
    """
    metres = xy * np.asarray(pitch, dtype=np.float64)
    n = len(metres)
    out = np.full(xy.shape[:-1], np.nan)
    for h in range(half_window, 0, -1):
        lo = np.clip(np.arange(n) - h, 0, n - 1)
        hi = np.clip(np.arange(n) + h, 0, n - 1)
        dist = np.linalg.norm(metres[hi] - metres[lo], axis=-1)
        dt = (hi - lo)[:, None] / float(fps)
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = dist / dt
        fill = np.isnan(out) & np.isfinite(estimate)
        out[fill] = estimate[fill]
    out[out > MAX_SPEED] = np.nan
    return out.astype(np.float32)


def load_tracking(paths=None):
    """Sample_Game_3 as an `mqc.atd.Tracking` with one slot per person (35), plus the metadata.

    Same fields and dtypes as `atd.load()`; index = row = frame - 1. The ball's third channel is
    always NaN (the sample has none). The pitch size is in meta["pitch"] (105 x 68), since the
    Tracking itself does not carry it.
    """
    paths = paths or fetch(quiet=True)
    meta = read_metadata(paths["Sample_Game_3_metadata.xml"])
    slot_of = {pid: i for i, (pid, _) in enumerate(meta["people"])}
    slot_team = np.array([HOME if team == meta["home_id"] else AWAY
                          for _, team in meta["people"]], dtype=np.int8)

    n = meta["periods"][-1][1]
    layout_slots = np.full(n + 1, -1, dtype=np.int16)
    slot_maps = []
    for k, (start, stop, players) in enumerate(meta["layouts"]):
        layout_slots[start:stop + 1] = k
        slot_maps.append(np.array([slot_of[p] for p in players]))

    xy = np.full((n, len(slot_team), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    with open(paths["Sample_Game_3_tracking.txt"]) as handle:
        first = handle.readline()
        _check_line(first, len(meta["layouts"][0][2]))
        handle.seek(0)
        for line in handle:
            frame, players, ball_xy = line.rstrip("\n").split(":")
            frame = int(frame)
            row = frame - 1
            values = np.array(players.replace(";", ",").split(","), dtype=np.float32)
            xy[row, slot_maps[layout_slots[frame]]] = values.reshape(-1, 2)
            ball[row, :2] = np.array(ball_xy.split(","), dtype=np.float32)

    players = np.concatenate([xy, speed(xy, meta["pitch"])[..., None]], axis=-1)
    names = (meta["teams"][meta["home_id"]],
             next(name for tid, name in meta["teams"].items() if tid != meta["home_id"]))
    return Tracking(players=players, ball=ball, slot_team=slot_team, team_names=names), meta


# -- labels ---------------------------------------------------------------------------------------

def _subtypes(event):
    sub = event.get("subtypes")
    if not sub:
        return ()
    if isinstance(sub, dict):
        sub = [sub]
    return tuple(s["name"].upper() for s in sub)


def _matches(rule, event):
    kind = event["type"]["name"].upper()
    if kind not in rule:
        return False
    need = rule[kind]
    return need is None or any(s in need for s in _subtypes(event))


def labels_from_events(events, n_frames, home_team_id):
    """'<U1' (n_frames,) H/A/D/X from the event log, by the rules in the module docstring.

    Returns (labels, report) where report counts how often each rule fired, so the policy can be
    audited from the output.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    report = {"periods": 0, "stoppages": 0, "restarts": 0, "restarts_without_stoppage": 0}

    for period in sorted({e["period"] for e in events}):
        evs = [e for e in events if e["period"] == period and e["start"]["frame"] is not None]
        evs.sort(key=lambda e: (e["start"]["frame"],
                                0 if e["type"]["name"].upper() in LOSS_FIRST else 1,
                                e["index"]))
        restarts = [i for i, e in enumerate(evs) if e["type"]["name"].upper() in RESTART_TYPES
                    or any(s in RESTART_SUBTYPES for s in _subtypes(e))]
        if not restarts:
            continue
        report["periods"] += 1
        begin = evs[restarts[0]]["start"]["frame"]
        end = max(e["end"]["frame"] or e["start"]["frame"] for e in evs)

        # Walk the events once, writing the label in force from each change point onwards.
        changes = []           # (frame, label)
        dead_since = None
        for i, e in enumerate(evs[restarts[0]:], start=restarts[0]):
            kind = e["type"]["name"].upper()
            start = e["start"]["frame"]
            if i in restarts:
                report["restarts"] += 1
                if dead_since is None and i != restarts[0]:
                    report["restarts_without_stoppage"] += 1
                    prev = evs[i - 1]
                    changes.append((min(prev["end"]["frame"] or start, start), "D"))
                dead_since = None
            elif dead_since is not None:
                continue       # nothing revives a dead ball except a restart
            if _matches(STOP_AT_START, e):
                dead_since = start
            elif _matches(STOP_AT_END, e):
                dead_since = e["end"]["frame"] or start
            if kind in CONTROL:
                team = "H" if e["team"]["id"] == home_team_id else "A"
                changes.append((start, team))
            if dead_since is not None:
                report["stoppages"] += 1
                changes.append((dead_since, "D"))

        # Change points in frame order; at one frame the later-listed change wins.
        changes.sort(key=lambda c: c[0])
        for (frame, label), nxt in zip(changes, changes[1:] + [(end + 1, None)]):
            lo, hi = max(frame, begin), min(nxt[0], end + 1)
            if hi > lo:
                labels[lo - 1:hi - 1] = label   # event frames count from 1, rows from 0
    return labels, report


def load(paths=None):
    """(tracking, gt_labels, meta, report) for Sample_Game_3."""
    paths = paths or fetch(quiet=True)
    tracking, meta = load_tracking(paths)
    with open(paths["Sample_Game_3_events.json"]) as handle:
        events = json.load(handle)["data"]
    labels, report = labels_from_events(events, len(tracking.ball), meta["home_id"])
    return tracking, labels, meta, report
