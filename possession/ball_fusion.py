"""Turn per-frame video ball candidates into a clean ball track that can fill the ATD's gaps.

Single-frame detections of a 4-9 px ball are noisy: heads, socks and line markings fire too. A real
ball moves along a continuous path, so candidates are linked over time into tracklets and only
tracklets that persist and move are kept:

  link      greedy, frame by frame: a candidate joins the nearest live tracklet if it is within
            max_speed * (frames since its last point) on the pitch; otherwise it starts a new one;
            a tracklet dies after max_gap frames without a point.
  keep      tracklets at least min_len frames long, with mean confidence >= min_mean_score, that
            move at least min_travel metres in total (static detections are markings or a ball at
            rest for a set piece -- dead ball, irrelevant to possession).
  choose    one ball per frame: the kept tracklet with the highest mean confidence.

Pixels go to the pitch through the inverse homography. Only aggregates are ever printed.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

FPS = 25


def candidates_on_pitch(cand, H, valid, pitch=None):
    """(x_m, y_m, score) of shape (n, K) from pixel candidates (n, K, 3)."""
    from starter.load_homography import project
    length, width = pitch or GAMES[GAME]["pitch"]
    n, k, _ = cand.shape
    inv = np.full_like(H, np.nan)
    inv[valid] = np.linalg.inv(H[valid])
    xs, ys = np.full((n, k), np.nan), np.full((n, k), np.nan)
    for j in range(k):
        x, y = project(inv.transpose(1, 2, 0), cand[:, j, 0].astype(float),
                       cand[:, j, 1].astype(float))
        xs[:, j], ys[:, j] = np.asarray(x) * length, np.asarray(y) * width
    return xs, ys, cand[:, :, 2].astype(float)


def link(xs, ys, sc, min_score=0.1, max_speed=40.0, max_gap=3):
    """List of tracklets, each a list of (frame, x, y, score)."""
    step = max_speed / FPS
    live, done = [], []
    n, k = xs.shape
    for f in range(n):
        order = np.argsort(-np.nan_to_num(sc[f], nan=-1.0))
        used = set()
        for j in order:
            s = sc[f, j]
            if not np.isfinite(s) or s < min_score or not np.isfinite(xs[f, j]):
                continue
            best, best_d = None, None
            for t, tr in enumerate(live):
                if t in used:
                    continue
                lf, lx, ly, _ = tr[-1]
                d = np.hypot(xs[f, j] - lx, ys[f, j] - ly)
                if d <= step * (f - lf) + 0.5 and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                live.append([(f, xs[f, j], ys[f, j], s)])
                used.add(len(live) - 1)
            else:
                live[best].append((f, xs[f, j], ys[f, j], s))
                used.add(best)
        still = []
        for tr in live:
            (done if f - tr[-1][0] > max_gap else still).append(tr)
        live = still
    return done + live


def ball_track(xs, ys, sc, min_score=0.1, max_speed=40.0, max_gap=3, min_len=8,
               min_mean_score=0.2, min_travel=1.0):
    """(bx, by) float[n] in metres: the fused video ball, NaN where no kept tracklet."""
    n = xs.shape[0]
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    best = np.full(n, -1.0)
    for tr in link(xs, ys, sc, min_score, max_speed, max_gap):
        if len(tr) < min_len:
            continue
        a = np.array(tr)
        mean_s = a[:, 3].mean()
        travel = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2])).sum()
        if mean_s < min_mean_score or travel < min_travel:
            continue
        frames = a[:, 0].astype(int)
        better = mean_s > best[frames]
        bx[frames[better]], by[frames[better]] = a[better, 1], a[better, 2]
        best[frames[better]] = mean_s
    return bx, by
