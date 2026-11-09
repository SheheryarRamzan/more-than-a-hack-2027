"""Smart Tagging as evidence: restarts, goals, and phase pseudo-labels for self-checks.

Smart Tagging is Metrica's automatic output, not ground truth, so it is used in two limited ways:

  * Set pieces mark restarts. The tag window is ~5 s around the restart, so the restart frame
    itself is located in the tracking as the moment the ball goes from rest to moving. Only then is
    it anchored, as "dead just before, alive just after, and the tagged team (if any) restarts".
  * Phases (Build Up, Defending in ... Third, ...) name a team in or out of possession. They are
    NOT fed to the decoder; they are used only to measure how consistent our output is with an
    independent automatic signal (report.py), which needs no ground truth.
"""

import difflib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

HOME, AWAY = 0, 1

# Phases whose Team tag is the team IN possession, and those whose Team tag is the team OUT of it.
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box"}
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}
# Phases that can only happen with the ball in play. The long, territorial ones ("Defending in
# ... Third", "Players in the Box", "Ball in Final Third", "Build Up") also run through stoppages
# -- they cover 64% of graded time, more than the ball is ever alive -- so they are not used as
# "alive" examples.
ALIVE_PHASES = {"attacking transition", "defensive transition", "progression",
                "creating chances", "long ball", "ball in the box"}


def team_code(value, home, away):
    """HOME / AWAY for a Team tag value, None for N/A. Fuzzy, because the tag spells the away club
    differently from the metadata."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("", "n/a", "na", "none", "-"):
        return None
    r_home = difflib.SequenceMatcher(None, v, home.strip().lower()).ratio()
    r_away = difflib.SequenceMatcher(None, v, away.strip().lower()).ratio()
    if max(r_home, r_away) < 0.5:
        return None
    return HOME if r_home >= r_away else AWAY


def load_events(home, away):
    """Smart Tagging instances with lower-cased code/type, a team code and video-frame bounds."""
    events = []
    for ev in load_smart_tagging():
        events.append({
            "code": (ev["code"] or "").strip().lower(),
            "type": (ev["tags"].get("Type") or "").strip().lower(),
            "team": team_code(ev["tags"].get("Team"), home, away),
            "start": int(ev["start_frame"]),
            "end": int(ev["end_frame"]),
        })
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events


def half_of(frame, halves=None):
    """(half, first, last) of the half containing frame, or None."""
    for half, first, last in (halves or half_video_blocks(GAME)):
        if first <= frame <= last:
            return half, first, last
    return None


def locate_restarts(events, feats, cfg):
    """One restart per set-piece tag: the frame the ball is struck, the tagged team, the type.

    Prefers a strict kick (ball seen at rest, then moving) inside the tag window; then the nearest
    strict-or-loose kick within the padded window; otherwise falls back to the window centre and
    marks the restart unlocated, which the decoder then trusts much less.
    """
    n = feats["n"]
    out = []
    for ev in events:
        if ev["code"] != "set pieces":
            continue
        start, end = max(ev["start"], 0), min(ev["end"], n)
        lo, hi = max(start - cfg.tag_search_pad, 0), min(end + cfg.tag_search_pad, n)
        centre = (start + end) // 2
        frame, located = None, True
        inner = np.flatnonzero(feats["kick"][start:end])
        if len(inner):
            frame = start + int(inner[0])
        else:
            cand = np.flatnonzero(feats["kick"][lo:hi] | feats["kick_loose"][lo:hi])
            if len(cand):
                frame = lo + int(cand[np.argmin(np.abs(lo + cand - centre))])
        if frame is None:
            frame, located = centre, False
        if half_of(frame, feats["halves"]) is None and ev["type"] == "kick off (start)":
            # The first half is graded from its kick-off frame, so a kick found a few frames early
            # lands outside it. Clamp to the start of the half the tag window overlaps.
            for _, first, last in feats["halves"]:
                if lo <= first <= hi:
                    frame = first
        if half_of(frame, feats["halves"]) is None:
            continue
        out.append({"frame": frame, "located": located, "team": ev["team"], "type": ev["type"]})
    out.sort(key=lambda r: r["frame"])
    return out


def goal_dead_spans(events, restarts, feats, cfg):
    """[(first_dead, kickoff)] for every goal followed by its kick-off.

    The stoppage starts when the ball is seen in the goal mouth past the line, searched from the
    tag start to a little after its end; if it is never seen there, from the tag end.
    """
    kickoffs = [r for r in restarts if r["type"].startswith("kick off (after goal")]
    spans = []
    for ev in events:
        if ev["code"] != "goals":
            continue
        nxt = next((r for r in kickoffs if r["frame"] > ev["start"]
                    and r["frame"] - ev["end"] <= cfg.goal_ko_max_gap), None)
        if nxt is None:
            continue
        lo = max(ev["start"], 0)
        hi = min(ev["end"] + cfg.goal_search_after, nxt["frame"])
        hits = np.flatnonzero(feats["in_goal"][lo:hi]) if hi > lo else np.empty(0, dtype=int)
        first = lo + int(hits[0]) if len(hits) else ev["end"]
        if first < nxt["frame"]:
            spans.append((first, nxt["frame"]))
    return spans


def dead_pseudo_labels(events, restarts, goal_spans, graded, cfg, halves=None):
    """(y int8[n], weight float[n]) for the self-supervised dead-ball model.

    y = 1 (dead): the run-up to every located restart, every goal stoppage, and the graded frames
    before a half's kick-off. y = 0 (alive): inside tagged phases, away from set pieces and goals.
    y = -1 elsewhere. Every label here comes from Smart Tagging plus a fixed rule -- no person
    looked at any frame to decide one.

    Each dead episode gets the same total weight, so one long goal celebration does not teach the
    model that "dead" means "everyone standing still" at the expense of a quick throw-in.
    """
    n = len(graded)
    y = np.full(n, -1, dtype=np.int8)
    trim = cfg.calib_phase_trim
    for ev in events:
        if ev["code"] in ALIVE_PHASES:
            a, b = max(ev["start"] + trim, 0), min(ev["end"] - trim, n)
            if b > a:
                y[a:b] = 0
    for ev in events:
        if ev["code"] in ("set pieces", "goals"):
            y[max(ev["start"] - cfg.calib_exclude_before, 0):
              min(ev["end"] + cfg.calib_exclude_after, n)] = -1

    episodes = []
    for r in restarts:
        if r["located"]:
            episodes.append((r["frame"] - cfg.calib_dead_before, r["frame"] - cfg.calib_dead_gap))
            if r["type"] == "kick off (start)":
                for _, first, last in (halves or half_video_blocks(GAME)):
                    if first < r["frame"] - cfg.calib_dead_gap and r["frame"] <= last:
                        episodes.append((first, r["frame"] - cfg.calib_dead_gap))
    episodes.extend(goal_spans)

    weight = np.zeros(n)
    for a, b in episodes:
        a, b = max(a, 0), min(b, n)
        if b > a:
            y[a:b] = 1
    y[~graded] = -1
    for a, b in episodes:
        a, b = max(a, 0), min(b, n)
        m = (y[a:b] == 1)
        if m.any():
            weight[a:b][m] += 1.0 / m.sum()
    alive = y == 0
    weight[alive] = 1.0 / max(alive.sum(), 1)
    weight[y == 1] = np.maximum(weight[y == 1], 1e-12)
    # Balance the classes.
    for cls in (0, 1):
        m = y == cls
        if m.any():
            weight[m] *= 0.5 / weight[m].sum()
    return y, weight


def team_pseudo_labels(events, feats, cfg):
    """(y int8[n], weight float[n]) for the collective team model: 1 = home in possession.

    Two independent automatic sources: Smart Tagging phases with a team, and frames where one
    team's player is unmistakably on a slow ball. Frames where they disagree are dropped. The
    collective model never sees the control evidence as a feature, so this is not circular.
    """
    n = feats["n"]
    phase = phase_pseudo_labels(events, n)
    ev = feats["team_ev"]
    speed = feats["speed"]
    confident = (np.abs(ev) >= cfg.team_confident_control) & np.isfinite(speed) \
        & (speed < cfg.team_confident_speed)
    control = np.where(confident, np.where(ev > 0, 1, 0), -1).astype(np.int8)
    y = np.where(phase >= 0, (phase == 0).astype(np.int8), -1).astype(np.int8)
    both = (y >= 0) & (control >= 0)
    y[both & (y != control)] = -1
    y[(y < 0) & (control >= 0) & ~both] = control[(y < 0) & (control >= 0) & ~both]
    y[~feats["graded"]] = -1
    weight = np.zeros(n)
    for cls in (0, 1):
        m = y == cls
        if m.any():
            weight[m] = 0.5 / m.sum()
    return y, weight


def phase_pseudo_labels(events, n):
    """int8[n]: HOME/AWAY where a phase tag implies who has the ball, -1 elsewhere or on conflict."""
    votes = np.zeros((n, 2), dtype=np.int16)
    for ev in events:
        if ev["team"] is None:
            continue
        if ev["code"] in IN_POSSESSION:
            side = ev["team"]
        elif ev["code"] in OUT_OF_POSSESSION:
            side = 1 - ev["team"]
        else:
            continue
        votes[max(ev["start"], 0):min(ev["end"], n), side] += 1
    out = np.full(n, -1, dtype=np.int8)
    out[(votes[:, 0] > 0) & (votes[:, 1] == 0)] = HOME
    out[(votes[:, 1] > 0) & (votes[:, 0] == 0)] = AWAY
    return out
