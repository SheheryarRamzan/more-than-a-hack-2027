"""Per-frame evidence computed from the tracking: everything the decoder looks at.

Two signals come out of here:

  team_ev  in [-1, 1]   positive = home is controlling the ball on this frame, negative = away,
                        0 = nobody is close enough to control it (or the ball is not seen). A loose
                        ball gives no evidence, so the decoder keeps the last team -- which is the
                        challenge's own rule for contested and loose balls.
  ldead    real         log-odds-like score that the ball is dead on this frame.

plus the ball-speed jumps ("kicks") and out-of-bounds frames used to place restarts and to snap
segment edges.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
GOAL_HALF_WIDTH = 4.5   # metres either side of the centre line of the goal (posts at 3.66)


# -- small helpers -------------------------------------------------------------------------------

def moving_nanmean(x, lo, hi):
    """Mean of x over the window [t+lo, t+hi), ignoring NaN. NaN where the window has no value."""
    n = len(x)
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(n)
    a = np.clip(index + lo, 0, n)
    b = np.clip(index + hi, 0, n)
    k = count[b] - count[a]
    out = np.full(n, np.nan)
    ok = k > 0
    out[ok] = (total[b] - total[a])[ok] / k[ok]
    return out


def runs(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def long_runs(mask, min_len):
    """mask restricted to the runs at least min_len long."""
    out = np.zeros(len(mask), dtype=bool)
    for a, b in runs(mask):
        if b - a >= min_len:
            out[a:b] = True
    return out


def default_match():
    """The competition match: its halves (inclusive video frames) and pitch size in metres."""
    return {"name": GAME, "halves": half_video_blocks(GAME), "pitch": GAMES[GAME]["pitch"]}


def graded_mask(n, halves=None):
    """True on the frames of the halves (the only ones that are scored)."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in (halves or half_video_blocks(GAME)):
        mask[first:last + 1] = True
    return mask


# -- ball ----------------------------------------------------------------------------------------

def clean_ball(bx, by, cfg):
    """(x, y, observed): drop isolated impossible jumps, bridge short gaps linearly.

    observed is the raw detection mask after spike removal; x and y are also filled on the bridged
    frames. Long gaps stay NaN on purpose.
    """
    n = len(bx)
    obs = np.isfinite(bx) & np.isfinite(by)
    index = np.flatnonzero(obs)
    if len(index) >= 3:
        step = np.hypot(np.diff(bx[index]), np.diff(by[index])) * FPS / np.diff(index)
        too_fast = step > cfg.ball_max_speed
        spike = np.zeros(len(index), dtype=bool)
        spike[1:-1] = too_fast[:-1] & too_fast[1:]    # impossible both into and out of it
        obs[index[spike]] = False

    frames = np.arange(n)
    prev = np.maximum.accumulate(np.where(obs, frames, -1))
    nxt = np.minimum.accumulate(np.where(obs, frames, n)[::-1])[::-1]
    gap = nxt - prev - 1
    bridge = ~obs & (prev >= 0) & (nxt < n) & (gap <= cfg.ball_interp_max_gap)

    x = np.where(obs, bx, np.nan)
    y = np.where(obs, by, np.nan)
    if bridge.any():
        p, q = prev[bridge], nxt[bridge]
        w = (frames[bridge] - p) / (q - p)
        x[bridge] = bx[p] + w * (bx[q] - bx[p])
        y[bridge] = by[p] + w * (by[q] - by[p])
    return x, y, obs


def ball_speed(x, y, halfwin):
    """Central-difference speed in m/s; NaN where either end is unknown."""
    n = len(x)
    h = int(halfwin)
    speed = np.full(n, np.nan)
    if n > 2 * h:
        vx = (x[2 * h:] - x[:-2 * h]) * FPS / (2.0 * h)
        vy = (y[2 * h:] - y[:-2 * h]) * FPS / (2.0 * h)
        speed[h:n - h] = np.hypot(vx, vy)
    return speed


# -- intensity -----------------------------------------------------------------------------------

def player_speeds(px, py, ps):
    """Per-slot speed in m/s: the tracker's own channel, or differenced positions if it is absent."""
    tracked = np.isfinite(px) & np.isfinite(py)
    if np.isfinite(ps).sum() >= 0.5 * tracked.sum():
        return np.where(tracked, ps, np.nan)
    vx = np.gradient(px, axis=0) * FPS
    vy = np.gradient(py, axis=0) * FPS
    return np.hypot(vx, vy)


def touch_events(feats, rate=15.0, k=3, near=2.0, nms=5):
    """(touch bool[n], team float[n]): abrupt changes of ball velocity with a player close by.

    After Link & Hoernig (2017, "Individual ball possession in soccer"): a player touches the ball
    when its velocity changes sharply while they are within reach. Positions are lightly smoothed,
    velocity is a central difference, and the change is measured between t-k and t+k; candidates
    need a player within `near` metres and are thinned to local maxima. team is +1 for a home
    player, -1 for away, 0 where there is no touch. Only frames where the ball is seen qualify.

    The velocity threshold is NOT a constant: tracking systems smooth the ball differently (at a
    fixed 4 m/s the degraded open matches show ~5.5x more touches than the provided ATD), so each
    match keeps its `rate` strongest touches per minute of visible graded ball. "Touch" then means
    the same thing in training and in the competition match.
    """
    mm = moving_nanmean
    n = feats["n"]
    sx, sy = mm(feats["bx"], -1, 2), mm(feats["by"], -1, 2)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[1:-1], vy[1:-1] = (sx[2:] - sx[:-2]) * 12.5, (sy[2:] - sy[:-2]) * 12.5
    dv = np.full(n, np.nan)
    dv[k:n - k] = np.hypot(vx[2 * k:] - vx[:n - 2 * k], vy[2 * k:] - vy[:n - 2 * k])
    d_home, d_away = feats["d_home"], feats["d_away"]
    base = feats["known"] & np.isfinite(dv) & (np.minimum(d_home, d_away) <= near)

    def peaks(threshold):
        cand = base & (dv >= threshold)
        score = np.where(cand, dv, 0.0)
        padded = np.concatenate((np.zeros(nms), score, np.zeros(nms)))
        local = np.max(np.stack([padded[i:i + n] for i in range(2 * nms + 1)]), axis=0)
        return cand & (score >= local)

    target = rate * (feats["known"] & feats["graded"]).sum() / (25.0 * 60.0)
    lo, hi = 0.5, 40.0                                   # bisection on the threshold (m/s)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if (peaks(mid) & feats["graded"]).sum() > target:
            lo = mid
        else:
            hi = mid
    peak = peaks(hi)
    team = np.where(peak, np.where(d_home <= d_away, 1.0, -1.0), 0.0)
    return peak, team


def gmm_llr(values, fit_mask, clip):
    """Log-likelihood ratio low-vs-high component of a 2-Gaussian mixture on log(values).

    Fitted to this match's own intensity over the graded frames, so the dead/alive threshold is a
    property of the data rather than a constant someone picked. Deterministic: fixed initialisation
    at the quartiles, fixed number of EM steps.
    """
    logv = np.log(np.where(np.isfinite(values), values, np.nan) + 0.1)
    sample = logv[fit_mask & np.isfinite(logv)]
    if len(sample) < 100:
        return np.zeros(len(values)), None
    mu = np.percentile(sample, [25.0, 75.0])
    sd = np.full(2, sample.std() / 2.0 + 1e-3)
    pi = np.array([0.5, 0.5])
    for _ in range(200):
        logp = -0.5 * ((sample[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(pi)
        logp -= logp.max(axis=1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-9
        pi = nk / len(sample)
        mu = (resp * sample[:, None]).sum(axis=0) / nk
        sd = np.sqrt((resp * (sample[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-3
    low, high = (0, 1) if mu[0] <= mu[1] else (1, 0)

    def component(k, v):
        return -0.5 * ((v - mu[k]) / sd[k]) ** 2 - np.log(sd[k]) + np.log(pi[k])

    llr = component(low, logv) - component(high, logv)
    llr = np.where(np.isfinite(llr), np.clip(llr, -clip, clip), 0.0)
    params = {"low_mean_mps": float(np.exp(mu[low]) - 0.1),
              "high_mean_mps": float(np.exp(mu[high]) - 0.1),
              "low_weight": float(pi[low])}
    return llr, params


# -- collective team shape -----------------------------------------------------------------------

def collective_features(px, py, home, bx, by, known, cfg, halves, aim=None, half_length=52.5):
    """(team_x float[n, 3], direction float[n], stats).

    direction is +1 where home attacks towards x = LENGTH. It is read per half from the
    goalkeepers: the team defending the x = 0 goal always has someone near it, so the median over
    the half of each team's rearmost x tells the two apart. No identities needed.

    The columns, all oriented so that positive favours HOME in possession:
      forward_velocity_diff  home's forward speed minus away's, team centroids, smoothed. The team
                             with the ball pushes up; the team without it drops back. It flips at
                             a turnover even when the ball itself is not seen.
      goal_side_diff         share of away players goal-side of the ball minus share of home's.
                             The defending team gets behind the ball.
      goal_side_valid        1 where the ball position (or one seen in the last 2 s) exists.
    """
    import warnings
    n = px.shape[0]
    tracked = np.isfinite(px) & np.isfinite(py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xs_h = np.where(tracked[:, home], px[:, home], np.nan)
        xs_a = np.where(tracked[:, ~home], px[:, ~home], np.nan)
        cx_h, cx_a = np.nanmean(xs_h, axis=1), np.nanmean(xs_a, axis=1)
        min_h, min_a = np.nanmin(xs_h, axis=1), np.nanmin(xs_a, axis=1)

    direction = np.zeros(n)
    orient = {}
    for half, first, last in halves:
        sl = slice(first, last + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home_rear, away_rear = np.nanmedian(min_h[sl]), np.nanmedian(min_a[sl])
        direction[sl] = 1.0 if home_rear < away_rear else -1.0
        orient["half{}_home_attacks_towards_x_max".format(half)] = bool(home_rear < away_rear)

    k = cfg.collective_smooth

    def forward_velocity(cx, sign):
        c = moving_nanmean(cx, -k, k + 1)
        v = np.gradient(c) * FPS * sign
        return moving_nanmean(v, -k, k + 1)

    fwd = forward_velocity(cx_h, direction) - forward_velocity(cx_a, -direction)

    # Last ball position seen within ball_hold_frames.
    frames = np.arange(n)
    last = np.maximum.accumulate(np.where(known, frames, -1))
    held = (last >= 0) & (frames - last <= cfg.ball_hold_frames)
    hx = np.where(held, bx[np.maximum(last, 0)], np.nan)
    behind_h = ((px[:, home] - hx[:, None]) * direction[:, None] < 0) & tracked[:, home]
    behind_a = ((px[:, ~home] - hx[:, None]) * -direction[:, None] < 0) & tracked[:, ~home]
    n_h = tracked[:, home].sum(axis=1)
    n_a = tracked[:, ~home].sum(axis=1)
    valid = held & (n_h >= 5) & (n_a >= 5)
    goal_side = np.where(valid, behind_a.sum(axis=1) / np.maximum(n_a, 1)
                         - behind_h.sum(axis=1) / np.maximum(n_h, 1), 0.0)

    # Shape: the team with the ball stretches the pitch (wide and long), the team without it
    # compacts; and the relative position of the two blocks along the attack direction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ys_h = np.where(tracked[:, home], py[:, home], np.nan)
        ys_a = np.where(tracked[:, ~home], py[:, ~home], np.nan)
        width = np.nanstd(ys_h, axis=1) - np.nanstd(ys_a, axis=1)
        length = np.nanstd(xs_h, axis=1) - np.nanstd(xs_a, axis=1)
    centroid = (cx_h - cx_a) * direction
    shape = [moving_nanmean(v, -k, k + 1) for v in (width, length, centroid)]
    # Slower versions of the two strongest signals (±2 s): turnovers show as a sustained flip.
    slow = [moving_nanmean(fwd, -50, 51), moving_nanmean(goal_side, -50, 51)]

    cols = [fwd, goal_side, valid.astype(float)] + shape + slow
    names = list(TEAM_X_NAMES)
    if aim is not None:
        # The camera aim stands in for the ball wherever no ball was seen in the last 2 s.
        proxy = np.where(held, hx, aim)
        ok = np.isfinite(proxy) & (n_h >= 5) & (n_a >= 5)
        b_h = ((px[:, home] - proxy[:, None]) * direction[:, None] < 0) & tracked[:, home]
        b_a = ((px[:, ~home] - proxy[:, None]) * -direction[:, None] < 0) & tracked[:, ~home]
        gs_cam = np.where(ok, b_a.sum(axis=1) / np.maximum(n_a, 1)
                          - b_h.sum(axis=1) / np.maximum(n_h, 1), 0.0)
        # Where play is, along home's attack direction: +1 at away's goal line, -1 at home's.
        territory = np.where(ok, (proxy - half_length) * direction / half_length, 0.0)
        cols += [moving_nanmean(gs_cam, -k, k + 1), moving_nanmean(territory, -k, k + 1)]
        names += ["goal_side_camera", "territory_camera"]
    team_x = np.column_stack([np.where(np.isfinite(c), c, 0.0) for c in cols])
    return team_x, direction, orient, names


TEAM_X_NAMES = ["forward_velocity_diff", "goal_side_diff", "goal_side_valid", "width_diff",
                "length_diff", "centroid_diff_forward", "forward_velocity_diff_2s",
                "goal_side_diff_2s"]


# -- everything ----------------------------------------------------------------------------------

def compute(trk, cfg, match=None):
    """dict of per-frame arrays, all length n_frames."""
    match = match or default_match()
    halves = match["halves"]
    LENGTH, WIDTH = match["pitch"]
    players = trk["players"]
    n = players.shape[0]
    graded = graded_mask(n, halves)

    px = players[..., 0].astype(np.float64) * LENGTH
    py = players[..., 1].astype(np.float64) * WIDTH
    ps = players[..., 2].astype(np.float64)
    raw_bx = trk["ball"][:, 0].astype(np.float64) * LENGTH
    raw_by = trk["ball"][:, 1].astype(np.float64) * WIDTH

    bx, by, observed = clean_ball(raw_bx, raw_by, cfg)
    known = np.isfinite(bx) & np.isfinite(by)
    speed = ball_speed(bx, by, cfg.ball_speed_halfwin)

    # Nearest player of each team to the ball.
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    home = trk["team"] == 0
    d_home = dist[:, home].min(axis=1)
    d_away = dist[:, ~home].min(axis=1)
    del dist

    d_min = np.minimum(d_home, d_away)
    proximity = np.clip((cfg.control_zero_radius - d_min)
                        / (cfg.control_zero_radius - cfg.control_full_radius), 0.0, 1.0)
    gap = np.minimum(d_away, 50.0) - np.minimum(d_home, 50.0)      # positive: home is closer
    contest = np.tanh(gap / cfg.contest_scale)
    fast = np.where(np.isfinite(speed) & (speed > cfg.ball_fast_speed), cfg.ball_fast_weight, 1.0)
    team_ev = np.where(known, proximity * contest * fast, 0.0)

    # Collective intensity: mean speed of the tracked players, smoothed ±1 s, centred.
    speeds = np.clip(player_speeds(px, py, ps), 0.0, cfg.player_speed_clip)
    count = np.isfinite(speeds).sum(axis=1)
    total = np.nansum(speeds, axis=1)
    mean_speed = np.where(count >= cfg.min_tracked_players, total / np.maximum(count, 1), np.nan)
    k = cfg.intensity_smooth
    intensity = moving_nanmean(mean_speed, -k, k + 1)
    llr_int, gmm = gmm_llr(intensity, graded, cfg.intensity_llr_clip)

    # Where the ball is.
    m = cfg.out_margin
    inside_raw = ((raw_bx >= -m) & (raw_bx <= LENGTH + m) & (raw_by >= -m) & (raw_by <= WIDTH + m))
    out = observed & ~inside_raw
    in_goal = observed & ((raw_bx < 0.0) | (raw_bx > LENGTH)) \
        & (np.abs(raw_by - WIDTH / 2.0) < GOAL_HALF_WIDTH)
    inside = known & (bx >= 0.0) & (bx <= LENGTH) & (by >= 0.0) & (by <= WIDTH)
    finite_speed = np.isfinite(speed)
    still = known & finite_speed & (speed < cfg.stationary_speed)
    still_long = long_runs(still, cfg.stationary_min)
    fast_inside = inside & finite_speed & (speed > cfg.fast_inside_speed)

    ldead = (cfg.w_intensity * llr_int
             + cfg.w_out * out
             + cfg.w_stationary * still_long
             - cfg.w_fast_inside * fast_inside
             + cfg.w_missing * ~known)

    # The same evidence, and a little more, as a feature matrix for the self-supervised model.
    running = np.where(count >= cfg.min_tracked_players,
                       (speeds > cfg.running_speed).sum(axis=1) / np.maximum(count, 1), np.nan)
    running = moving_nanmean(running, -k, k + 1)
    log_int = np.log(np.where(np.isfinite(intensity), intensity, np.nan) + 0.1)
    # Standardised within the match, so a model trained on one provider's speeds transfers to
    # another's (absolute speed levels differ between tracking systems).
    ref = log_int[graded & np.isfinite(log_int)]
    if len(ref) > 100:
        log_int = (log_int - np.median(ref)) / (ref.std() + 1e-9)
    moving_inside = inside & finite_speed & (speed > 2.0) & (speed <= cfg.fast_inside_speed)
    missing_long = long_runs(~known, cfg.missing_long)
    columns = [
        ("intensity_llr", llr_int),
        ("log_intensity", log_int),
        ("running_share", running),
        ("ball_out", out.astype(float)),
        ("ball_still", still_long.astype(float)),
        ("ball_fast_inside", fast_inside.astype(float)),
        ("ball_moving_inside", moving_inside.astype(float)),
        ("ball_missing", (~known).astype(float)),
        ("ball_missing_long", missing_long.astype(float)),
    ]
    dead_x = np.column_stack([c for _, c in columns])
    for j in range(dead_x.shape[1]):                     # NaN -> the column's graded median
        col = dead_x[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            fill = np.nanmedian(col[graded]) if np.isfinite(col[graded]).any() else 0.0
            col[bad] = fill

    # Kicks: the ball goes from slow to fast. "strict" needs the ball seen before the kick;
    # "loose" also accepts a ball that reappears already moving (a throw-in from off camera).
    pre = moving_nanmean(speed, -cfg.kick_pre_win, 0)
    post = moving_nanmean(speed, 0, cfg.kick_post_win)
    kick = (post > cfg.kick_post_min) & (pre < cfg.kick_pre_max)
    kick_loose = (post > cfg.kick_post_min) & ~(pre >= cfg.kick_pre_max)

    # -- collective team shape: signals that do not need the ball to be seen ----------------------
    aim = trk.get("aim_x") if cfg.use_camera else None
    team_x, direction, collective_stats, team_names = collective_features(
        px, py, home, bx, by, known, cfg, halves, aim, LENGTH / 2.0)

    stats = {
        "collective": collective_stats,
        "ball_observed_graded": float(observed[graded].mean()),
        "ball_known_after_bridging_graded": float(known[graded].mean()),
        "mean_tracked_players_graded": float(count[graded].mean()),
        "intensity_mixture": gmm,
    }
    return {
        "n": n, "halves": halves, "pitch": (LENGTH, WIDTH), "graded": graded, "bx": bx, "by": by, "team_ev": team_ev, "ldead": ldead, "known": known,
        "dead_x": dead_x, "dead_x_names": [name for name, _ in columns],
        "team_x": team_x, "team_x_names": team_names,
        "direction": direction,
        "speed": speed, "out": out, "in_goal": in_goal, "kick": kick, "kick_loose": kick_loose,
        "d_home": d_home, "d_away": d_away, "intensity": intensity, "stats": stats,
    }
