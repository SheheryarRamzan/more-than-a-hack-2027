"""How good are the video ball detections? Checked against the provided tracking, no ground truth.

    python detector/eval_detections.py [data/cache/ball_detections_s1.npz]

On frames where the provided tracking sees the ball, a detection is "right" if it lands within
1.5 m of it on the pitch (pixels -> pitch through the inverse homography). That gives, per
confidence threshold:

  agree      share of confident detections that agree with the tracking's ball (precision proxy)
  recall     share of tracked-ball frames with an agreeing confident detection
  gap_fill   share of frames where the tracking has NO ball but a confident detection exists

Only aggregates are printed.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES  # noqa: E402
from possession import atd  # noqa: E402
from possession.features import graded_mask  # noqa: E402
from starter.load_homography import load as load_h, project  # noqa: E402


def detections_on_pitch(cand, H, valid):
    """(x_m, y_m, score) arrays of shape (n, K): candidates projected to pitch metres."""
    L, W = GAMES[GAME]["pitch"]
    n, k, _ = cand.shape
    inv = np.full_like(H, np.nan)
    inv[valid] = np.linalg.inv(H[valid])
    xs, ys = np.full((n, k), np.nan), np.full((n, k), np.nan)
    for j in range(k):
        x, y = project(inv.transpose(1, 2, 0), cand[:, j, 0].astype(float), cand[:, j, 1].astype(float))
        xs[:, j], ys[:, j] = np.asarray(x) * L, np.asarray(y) * W
    return xs, ys, cand[:, :, 2].astype(float)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "cache", "ball_detections_s1.npz")
    z = np.load(path)
    cand, done = z["cand"], z["done"]
    L, W = GAMES[GAME]["pitch"]
    H, _, valid = load_h()
    trk = atd.load(verbose=False)
    bx = trk["ball"][:, 0].astype(float) * L
    by = trk["ball"][:, 1].astype(float) * W
    xs, ys, sc = detections_on_pitch(cand, H, valid)

    use = done & graded_mask(len(done))
    seen = use & np.isfinite(bx)
    gap = use & ~np.isfinite(bx)
    top_d = np.hypot(xs[:, 0] - bx, ys[:, 0] - by)
    any_d = np.nanmin(np.where(np.isfinite(xs), np.hypot(xs - bx[:, None], ys - by[:, None]), np.inf), axis=1)
    print("{}: {} graded frames processed ({} with a tracked ball, {} without)".format(
        os.path.basename(path), int(use.sum()), int(seen.sum()), int(gap.sum())))
    print("median distance top candidate -> tracked ball: {:.1f} m".format(
        float(np.nanmedian(top_d[seen & np.isfinite(top_d)])) if (seen & np.isfinite(top_d)).any() else float("nan")))
    print("any of the top-{} candidates within 1.5 m of the tracked ball: {:.1%}".format(
        cand.shape[1], float(np.mean(any_d[seen] < 1.5)) if seen.any() else 0.0))
    print("\nthreshold   agree  recall  gap_fill")
    for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        conf = np.isfinite(sc[:, 0]) & (sc[:, 0] >= t)
        both = seen & conf
        agree = float(np.mean(top_d[both] < 1.5)) if both.any() else float("nan")
        recall = float(np.mean(conf[seen] & (top_d[seen] < 1.5))) if seen.any() else float("nan")
        fill = float(np.mean(conf[gap])) if gap.any() else float("nan")
        print("  {:<8}  {:.3f}   {:.3f}   {:.3f}".format(t, agree, recall, fill))


if __name__ == "__main__":
    main()
