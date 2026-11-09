"""Aggregate checks that need no ground truth.

Everything printed here is a count, a share or a mean -- never frame-level data. Two kinds:

  * plausibility -- how many possessions and stoppages, how long, what share is dead ball. A
    football match has a few hundred possessions and the ball is dead for a large share of the
    graded time; a submission far outside that is wrong whatever it scores anywhere.
  * consistency with Smart Tagging -- an independent automatic signal. Agreement with it is not
    accuracy (it has its own errors), but a change that lowers it a lot is suspect.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession.features import runs  # noqa: E402
from possession.tags import phase_pseudo_labels  # noqa: E402

FPS = GAMES[GAME]["fps"]


def segment_stats(labels, halves=None):
    """Possession and stoppage counts and durations, inside the halves only."""
    halves = halves or half_video_blocks(GAME)
    spells, stoppages = [], []
    for _, first, last in halves:
        window = labels[first:last + 1]
        for value, bucket in (("H", spells), ("A", spells), ("D", stoppages)):
            bucket.extend(b - a for a, b in runs(window == value))
    graded = np.concatenate([labels[f:l + 1] for _, f, l in halves])
    alive = np.isin(graded, ["H", "A"])
    spells_s = np.array(spells, dtype=float) / FPS
    stops_s = np.array(stoppages, dtype=float) / FPS
    return {
        "graded_frames": int(len(graded)),
        "share_H": float(np.mean(graded == "H")),
        "share_A": float(np.mean(graded == "A")),
        "share_D": float(np.mean(graded == "D")),
        "home_share_of_alive": float(np.mean(graded[alive] == "H")) if alive.any() else 0.0,
        "n_possessions": int(len(spells_s)),
        "possession_mean_s": float(spells_s.mean()) if len(spells_s) else 0.0,
        "possession_median_s": float(np.median(spells_s)) if len(spells_s) else 0.0,
        "possessions_under_2s": int(np.sum(spells_s < 2.0)),
        "n_stoppages": int(len(stops_s)),
        "stoppage_mean_s": float(stops_s.mean()) if len(stops_s) else 0.0,
    }


def tagging_consistency(labels, events, restarts):
    """Agreement with Smart Tagging phases and set pieces (no ground truth involved)."""
    n = len(labels)
    pseudo = phase_pseudo_labels(events, n)
    tagged = pseudo >= 0
    ours_alive = np.isin(labels, ["H", "A"])
    both = tagged & ours_alive
    ours_home = labels == "H"
    agree = np.mean(ours_home[both] == (pseudo[both] == 0)) if both.any() else 0.0

    located = [r for r in restarts if r["located"]]
    dead_before = alive_after = team_ok = team_n = 0
    for r in located:
        f = r["frame"]
        dead_before += labels[max(f - 12, 0)] == "D"
        after = labels[min(f + 25, n - 1)]
        alive_after += after in ("H", "A")
        if r["team"] is not None and after in ("H", "A"):
            team_n += 1
            team_ok += (after == "H") == (r["team"] == 0)
    k = max(len(located), 1)
    return {
        "phase_frames_tagged": int(tagged.sum()),
        "phase_team_agreement": float(agree),
        "phase_frames_we_call_dead": float(np.mean(labels[tagged] == "D")) if tagged.any() else 0.0,
        "set_pieces": len(restarts),
        "set_pieces_located": len(located),
        "restart_dead_before": float(dead_before) / k,
        "restart_alive_after": float(alive_after) / k,
        "restart_team_matches_tag": float(team_ok) / max(team_n, 1),
    }


def print_block(title, values):
    print("\n" + title)
    for key, value in values.items():
        if isinstance(value, float):
            print("  {:<34} {:.4f}".format(key, value))
        else:
            print("  {:<34} {}".format(key, value))
