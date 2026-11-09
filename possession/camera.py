"""Where the tactical camera is aimed, from the homography: a coarse ball position.

The tactical camera pans to follow play. Projecting the image centre back onto the pitch through
the inverse homography gives the point it is aimed at; along the length of the pitch that point
tracks the ball (correlation 0.83, median error 12 m over the frames where the ball is seen, and
the camera trails the ball by about a second or two). That is far coarser than a ball detection,
but it exists on every frame -- including the long stretches where the tracking has lost the ball.

Only the x coordinate is used: the image centre lands beyond the far touchline, so its y says
nothing about the ball.

For the open matches used in validation there is no video, so synthetic_aim() builds a camera
with the same character from their true ball position: lagged, smoothed, and with slow noise sized
so that its correlation and median error match the real camera's.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES  # noqa: E402

CACHE = os.path.join(DATA_DIR, "cache", "{}_tactical_aim_v1.npy".format(GAME))
IMAGE_CENTRE = (960.0, 540.0)


def aim_x(use_cache=True):
    """float[n]: x (metres) of the pitch point at the centre of the tactical image."""
    if use_cache and os.path.isfile(CACHE):
        return np.load(CACHE)
    from starter.load_homography import load, project
    matrices, _, valid = load()
    inverse = np.full_like(matrices, np.nan)
    inverse[valid] = np.linalg.inv(matrices[valid])
    x, _ = project(inverse.transpose(1, 2, 0), *IMAGE_CENTRE)
    out = np.asarray(x, dtype=np.float64) * GAMES[GAME]["pitch"][0]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.save(CACHE, out)
    return out


def synthetic_aim(true_ball_x, length, seed, lag=25, smooth=25, noise_m=18.0, noise_tau=125):
    """A camera aim like the real one, from an external match's true ball x (metres).

    noise_m=18 gives correlation ~0.85 and median error ~14 m against the true ball; the real
    camera has 0.83 and 12 m (heavier tails). No single level matches both, so this errs on the
    pessimistic side: if the feature helps in validation, it should help on the real camera.
    """
    from possession.features import moving_nanmean
    rng = np.random.RandomState(seed)
    n = len(true_ball_x)
    lagged = np.concatenate((np.full(lag, np.nan), true_ball_x[:n - lag]))
    base = moving_nanmean(lagged, -smooth, smooth + 1)
    # Slow (AR(1)) noise: the operator's framing wanders over seconds, not frames.
    a = np.exp(-1.0 / noise_tau)
    e = rng.randn(n) * noise_m * np.sqrt(1 - a * a)
    noise = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = a * acc + e[i]
        noise[i] = acc
    aim = base + noise
    return np.clip(aim, 0.15 * length, 0.85 * length)
