"""Turn per-frame evidence into H / A / D labels.

Stage 1 decides alive vs dead, stage 2 decides home vs away inside each alive stretch. Splitting
them keeps one failure from causing the other: a throw-in taker holding the ball is "close to the
ball" for seconds, which would read as possession if the two questions were asked together.

Both stages are the same 2-state Viterbi: one state earns score[t] per frame relative to the
other, and changing state costs a fixed amount. That cost is what turns flickering per-frame
evidence into possessions -- a switch happens only where the evidence for it, summed over time,
outweighs the cost, and the switch is placed where that evidence starts.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from possession.features import runs  # noqa: E402

LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"


def viterbi2(score1, switch, init1=0.0):
    """Best bool path for a 2-state chain: state True earns score1[t], False earns 0, every change
    of state costs `switch`. init1 is an extra score for starting in state True. Ties stay put, so
    the result is deterministic."""
    s = np.asarray(score1, dtype=np.float64).tolist()
    T = len(s)
    if T == 0:
        return np.zeros(0, dtype=bool)
    from0 = bytearray(T)   # 1 if the best way into state False at t came from True
    from1 = bytearray(T)   # 1 if the best way into state True at t came from True
    v0, v1 = 0.0, s[0] + init1
    for t in range(1, T):
        jump0 = v1 - switch
        if jump0 > v0:
            n0 = jump0
            from0[t] = 1
        else:
            n0 = v0
        jump1 = v0 - switch
        if v1 >= jump1:
            n1 = v1 + s[t]
            from1[t] = 1
        else:
            n1 = jump1 + s[t]
        v0, v1 = n0, n1
    path = np.zeros(T, dtype=bool)
    state = 1 if v1 > v0 else 0
    for t in range(T - 1, -1, -1):
        path[t] = bool(state)
        if t:
            state = from1[t] if state else from0[t]
    return path


def anchors(feats, restarts, goal_spans, cfg):
    """(dead_w, alive_w): per-frame anchor scores from located restarts, kick-offs and goals."""
    n = feats["n"]
    dead_w = np.zeros(n)
    alive_w = np.zeros(n)
    for r in restarts:
        w = cfg.anchor_weight if r["located"] else cfg.anchor_weight_unlocated
        f = r["frame"]
        lo = max(f - cfg.anchor_dead_before, 0)
        dead_w[lo:f] = np.maximum(dead_w[lo:f], w)
        alive_w[f:f + cfg.anchor_alive_after] = np.maximum(alive_w[f:f + cfg.anchor_alive_after], w)
        if r["type"] == "kick off (start)":
            # Everything graded before the kick-off of a half is dead -- the second half is graded
            # from 2,934 frames before its kick-off (FAQ.md).
            for _, first, last in feats["halves"]:
                if first <= f <= last:
                    dead_w[first:f] = np.maximum(dead_w[first:f], w)
    for first, kickoff in goal_spans:
        dead_w[first:kickoff] = np.maximum(dead_w[first:kickoff], cfg.anchor_weight)
    return dead_w, alive_w


def enforce_min_durations(dead, lo, hi, cfg, locked_dead, locked_alive):
    """Drop stoppages shorter than min_dead and live stretches shorter than min_alive, in [lo, hi)."""
    for value, min_len, locked in ((True, cfg.min_dead, locked_dead),
                                   (False, cfg.min_alive, locked_alive)):
        window = dead[lo:hi]
        for a, b in runs(window == value):
            if b - a < min_len and not locked[lo + a:lo + b].any():
                window[a:b] = not value
    return dead


def snap_edges(dead, lo, hi, feats, cfg, locked):
    """Move unanchored edges onto the physical event that explains them.

    alive -> dead snaps to the first frame the ball is seen past a line nearby; dead -> alive snaps
    to the nearest kick. An edge with no such event nearby is left where the decoder put it.
    """
    out_frames = feats["out"]
    kicks = feats["kick"]
    window = dead[lo:hi].copy()
    edges = np.flatnonzero(window[1:] != window[:-1]) + 1
    for e in edges.tolist():
        f = lo + e
        if locked[max(f - 2, 0):f + 2].any():
            continue
        if dead[f]:                                   # alive -> dead at f
            a, b = max(f - cfg.snap_out_before, lo), min(f + cfg.snap_out_after, hi)
            hits = np.flatnonzero(out_frames[a:b])
            if len(hits):
                g = a + int(hits[0])
                if g < f:
                    dead[g:f] = True
                elif g > f:
                    dead[f:g] = False
        else:                                         # dead -> alive at f
            a, b = max(f - cfg.snap_kick_window, lo), min(f + cfg.snap_kick_window, hi)
            hits = a + np.flatnonzero(kicks[a:b])
            if len(hits):
                g = int(hits[np.argmin(np.abs(hits - f))])
                if g < f:
                    dead[g:f] = False
                elif g > f:
                    dead[f:g] = True
    return dead


def decode_alive_dead(feats, ldead, dead_w, alive_w, cfg):
    """bool[n], True = dead, decided per half."""
    n = feats["n"]
    dead = np.zeros(n, dtype=bool)
    score = ldead + dead_w - alive_w
    locked_dead = dead_w >= cfg.anchor_weight
    locked_alive = alive_w >= cfg.anchor_weight
    for _, first, last in feats["halves"]:
        lo, hi = first, last + 1
        dead[lo:hi] = viterbi2(score[lo:hi], cfg.alive_dead_switch)
        enforce_min_durations(dead, lo, hi, cfg, locked_dead, locked_alive)
        snap_edges(dead, lo, hi, feats, cfg, locked_dead | locked_alive)
        enforce_min_durations(dead, lo, hi, cfg, locked_dead, locked_alive)
    return dead


def decode_alive_dead_at_share(feats, ldead, dead_w, alive_w, cfg):
    """(dead, offset): shift the dead-ball evidence until the decoded dead share over the graded
    frames matches cfg.target_dead_share (a prior from other games, see config.py). Bisection on
    a constant offset; the decoded share is monotone in it up to the min-duration rounding."""
    graded = feats["graded"]
    if cfg.target_dead_share <= 0:
        return decode_alive_dead(feats, ldead, dead_w, alive_w, cfg), 0.0
    lo, hi = -8.0, 8.0
    for _ in range(cfg.share_search_iters):
        mid = 0.5 * (lo + hi)
        share = decode_alive_dead(feats, ldead + mid, dead_w, alive_w, cfg)[graded].mean()
        if share > cfg.target_dead_share:
            hi = mid
        else:
            lo = mid
    offset = 0.5 * (lo + hi)
    return decode_alive_dead(feats, ldead + offset, dead_w, alive_w, cfg), offset


def decode_teams(feats, dead, restarts, cfg, team_score=None):
    """bool[n], True = home, meaningful on alive frames only."""
    n = feats["n"]
    rw = cfg.restart_window
    score = (feats["team_ev"] if team_score is None else team_score).copy()
    # Sustained bias towards the team Smart Tagging names on a located restart.
    for r in restarts:
        if r["located"] and r["team"] is not None:
            sign = 1.0 if r["team"] == 0 else -1.0
            score[max(r["frame"] - rw, 0):r["frame"] + rw] += sign * cfg.tag_team_bias

    if cfg.rule_bias > 0:
        score += restart_rule_bias(feats, dead, cfg)

    home = np.zeros(n, dtype=bool)
    for _, first, last in feats["halves"]:
        lo, hi = first, last + 1
        for a, b in runs(~dead[lo:hi]):
            s, e = lo + a, lo + b
            # Decode from rw frames before the restart: whoever stands over the dead ball (the
            # taker) sets the starting team, then only the alive part is kept.
            lead = max(s - rw, lo)
            home[s:e] = viterbi2(score[lead:e], cfg.team_switch)[s - lead:]
            if cfg.team_snap_window > 0:
                snap_team_edges(home, s, e, feats["team_ev"], cfg)
    return home


def snap_team_edges(home, s, e, team_ev, cfg):
    """Move each home<->away change inside [s, e) to the first frame of clear control by the team
    that gains the ball, if one lies within team_snap_window frames. The decoder places a change
    where the smoothed evidence tips over, which can lag the moment the ball is actually won.
    A change never moves past a neighbouring change."""
    window = home[s:e]
    edges = (np.flatnonzero(window[1:] != window[:-1]) + 1).tolist()
    w = cfg.team_snap_window
    for i, t in enumerate(edges):
        new_home = bool(window[t])
        lo = max(t - w, edges[i - 1] + 1 if i else 1)
        hi = min(t + w, edges[i + 1] - 1 if i + 1 < len(edges) else len(window) - 1)
        ev = team_ev[s + lo:s + hi + 1]
        hits = np.flatnonzero(ev >= cfg.team_snap_threshold) if new_home \
            else np.flatnonzero(ev <= -cfg.team_snap_threshold)
        if not len(hits):
            continue
        f = lo + int(hits[0])
        if f < t:
            window[f:t] = new_home
        elif f > t:
            window[t:f] = not new_home


def restart_kind(x, y, length, width):
    """Restart type from where the ball is when play resumes, or None."""
    if abs(x - length / 2) < 3.0 and abs(y - width / 2) < 3.0:
        return "kick off"
    if (x < 3.0 or x > length - 3.0) and (y < 4.0 or y > width - 4.0):
        return "corner"
    if y < 2.0 or y > width - 2.0:
        return "throw-in"
    if (x < 7.0 or x > length - 7.0) and abs(y - width / 2) < 11.0:
        return "goal kick"
    return None


def restart_rule_bias(feats, dead, cfg):
    """Per-frame bias towards the team the Laws of the Game give each restart to.

      corner    -> the team attacking that end         (known from the attack direction)
      goal kick -> the team defending that end
      throw-in  -> the opponents of whoever touched it last before it went out
      kick-off  -> the opponents of whoever touched it last (after a goal: the team that conceded)

    "Touched it last" is the team of the nearest player on the last frames the ball was seen
    before the stoppage began. Free kicks are left to the evidence: the Laws do not say who
    fouled whom.
    """
    n = feats["n"]
    length, width = feats["pitch"]
    bx, by, known = feats["bx"], feats["by"], feats["known"]
    team_ev, direction = feats["team_ev"], feats["direction"]
    rw = cfg.restart_window
    bias = np.zeros(n)
    for _, first, last in feats["halves"]:
        lo, hi = first, last + 1
        alive_runs = runs(~dead[lo:hi])
        for i, (a, _) in enumerate(alive_runs):
            s = lo + a
            if s == lo:
                continue
            near = np.flatnonzero(known[max(s - cfg.rule_ball_window, lo):s + 6])
            if not len(near):
                continue
            f = max(s - cfg.rule_ball_window, lo) + int(near[np.argmin(
                np.abs(max(s - cfg.rule_ball_window, lo) + near - s))])
            kind = restart_kind(bx[f], by[f], length, width)
            if kind is None:
                continue
            attacker_of_end = (0 if direction[s] > 0 else 1) if bx[f] > length / 2 \
                else (1 if direction[s] > 0 else 0)
            if kind == "corner":
                team = attacker_of_end
            elif kind == "goal kick":
                team = 1 - attacker_of_end
            else:
                if i == 0:
                    continue
                stop = lo + alive_runs[i - 1][1]            # the stoppage began here
                window = team_ev[max(stop - cfg.rule_touch_window, lo):stop]
                touched = window[window != 0]
                if not len(touched):
                    continue
                last_team = 0 if touched[-1] > 0 else 1
                team = 1 - last_team
            sign = 1.0 if team == 0 else -1.0
            bias[max(s - rw, lo):s + rw] += sign * cfg.rule_bias
    return bias


def labels_from(dead, home, graded):
    """object array of 'H' / 'A' / 'D' / 'X'."""
    out = np.full(len(dead), LABEL_X, dtype="<U1")
    out[graded & dead] = LABEL_D
    out[graded & ~dead & home] = LABEL_H
    out[graded & ~dead & ~home] = LABEL_A
    return out


def decode(feats, restarts, goal_spans, cfg, ldead=None, team_score=None):
    """Labels for every frame, plus the intermediate arrays for reporting."""
    dead_w, alive_w = anchors(feats, restarts, goal_spans, cfg)
    dead, offset = decode_alive_dead_at_share(
        feats, feats["ldead"] if ldead is None else ldead, dead_w, alive_w, cfg)
    home = decode_teams(feats, dead, restarts, cfg, team_score)
    return labels_from(dead, home, feats["graded"]), {"dead": dead, "home": home,
                                                       "dead_offset": offset}
