"""Run the FootAndBall ball detector over the tactical video on the GPU.

    .venv-detect/Scripts/python detector/detect_ball.py --start 20000 --end 25000   # a pilot
    .venv-detect/Scripts/python detector/detect_ball.py                             # whole video

Why: the provided tracking loses the ball on 41 % of graded frames, mostly in long stretches
(median gap 10.7 s). Improving the provided data with our own algorithm is explicitly allowed
(RULES.md); this re-detects the ball from the video so those gaps can be filled.

The detector is FootAndBall (Komorowski et al., VISAPP 2020; MIT licence), built for long-shot
fixed-camera football video where the ball is only a few pixels across -- here it is ~4.4 px. Its
published weights are vendored in third_party/footandball and loaded with weights_only=True.

Output: data/cache/ball_detections_s<scale>.npz with the top-K ball candidates per video frame
(pixel x, pixel y, confidence) in ORIGINAL-resolution pixels. It stays under data/ (gitignored):
detections are derived from the match video and must not be published. Only aggregate counts are
printed.

Determinism: FP32, cuDNN deterministic mode, fixed frame order. GPU inference can still differ in
the last bits between machines; the downstream fusion thresholds a confidence, so a borderline
candidate is the only thing that could move.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAB = os.path.join(ROOT, "third_party", "footandball")
for p in (FAB, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import network.footandball as footandball  # noqa: E402
from data.augmentation import BALL_LABEL  # noqa: E402
from games import DATA_DIR, asset_paths, source_frames  # noqa: E402

WEIGHTS = os.path.join(FAB, "models", "model_20201019_1416_final.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(device, max_ball, threshold):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = footandball.model_factory("fb1", "detect", max_ball_detections=max_ball,
                                      ball_threshold=threshold, player_threshold=1.01)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def to_batch(frames, scale, device):
    """BGR uint8 frames -> normalised RGB tensor, resized by `scale`, padded to a multiple of 16."""
    if scale != 1.0:
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                  for f in frames]
    h, w = frames[0].shape[:2]
    H, W = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    arr = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        arr[i, :h, :w] = f[:, :, ::-1]                      # BGR -> RGB
    x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="exclusive")
    parser.add_argument("--scale", type=float, default=1.0, help="upscale before detection")
    parser.add_argument("--top", type=int, default=3, help="candidates kept per frame")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = source_frames("tactical")
    end = min(args.end or n, n)
    out = args.out or os.path.join(DATA_DIR, "cache", "ball_detections_s{:g}.npz".format(args.scale))
    if os.path.isfile(out):
        z = np.load(out)
        cand, done = z["cand"], z["done"]
    else:
        cand = np.full((n, args.top, 3), np.nan, dtype=np.float32)
        done = np.zeros(n, dtype=bool)

    model = build_model(device, max_ball=args.top, threshold=args.min_score)
    video = cv2.VideoCapture(asset_paths()["video"])
    if args.start:
        video.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_no, batch, idx = args.start, [], []
    t0, last_report = time.time(), time.time()

    def flush():
        with torch.no_grad():
            dets = model(to_batch(batch, args.scale, device))
        for k, d in zip(idx, dets):
            keep = (d["labels"] == BALL_LABEL).to(d["boxes"].device)
            boxes, scores = d["boxes"][keep].cpu().numpy(), d["scores"][keep].cpu().numpy()
            order = np.argsort(-scores)[:args.top]
            for j, o in enumerate(order):
                x1, y1, x2, y2 = boxes[o]
                cand[k, j] = ((x1 + x2) / 2 / args.scale, (y1 + y2) / 2 / args.scale, scores[o])
            done[k] = True

    while frame_no < end:
        ok, frame = video.read()
        if not ok:
            break
        batch.append(frame)
        idx.append(frame_no)
        frame_no += 1
        if len(batch) == args.batch:
            flush()
            batch, idx = [], []
        if time.time() - last_report > 60:
            rate = (frame_no - args.start) / (time.time() - t0)
            print("  frame {} / {}  ({:.1f} fps, ~{:.0f} min left)".format(
                frame_no, end, rate, (end - frame_no) / max(rate, 1e-9) / 60), flush=True)
            last_report = time.time()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, cand=cand, done=done)
    if batch:
        flush()
    video.release()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, cand=cand, done=done)
    span = done[args.start:end]
    found = np.isfinite(cand[args.start:end, 0, 2])
    print("processed {} frames ({:.1f} fps); a ball candidate on {:.1%} of them -> {}".format(
        int(span.sum()), span.sum() / (time.time() - t0), found[span].mean() if span.any() else 0,
        os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
