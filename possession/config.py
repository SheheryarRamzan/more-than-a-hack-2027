"""Every tunable number in the pipeline, in one place.

None of these values were fitted to the public ground truth: the rules forbid it, and the
repository shows it picks the wrong answer anyway. They come from three places:

  * physics and football priors -- how fast a ball travels, how close a player has to be to
    control it, how long a restart takes;
  * unsupervised statistics of this match's own tracking (the intensity threshold is a two-component
    mixture fitted to the match, not a constant);
  * self-consistency checks against Smart Tagging, which is algorithmic output and allowed.

Units: metres, metres per second, and frames at 25 fps (25 frames = 1 s).
"""

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    # -- ball cleaning ---------------------------------------------------------------------------
    # A hard shot is ~35 m/s. An observation that implies a faster jump both into and out of it is
    # a detection error, not a ball.
    ball_max_speed: float = 40.0
    # Gaps up to this many frames (0.6 s) are bridged linearly. Longer gaps stay missing: a ball
    # unseen for over half a second may have changed direction, and guessing where it went is how a
    # possession gets invented.
    ball_interp_max_gap: int = 15
    # Half-window of the central difference used for ball velocity.
    ball_speed_halfwin: int = 2

    # -- ball control (team evidence) ------------------------------------------------------------
    # A player within control_full_radius of the ball gives full evidence for their team, fading
    # linearly to nothing at control_zero_radius.
    control_full_radius: float = 1.0
    control_zero_radius: float = 3.0
    # When both teams have a player near the ball, the evidence is tanh(distance gap / scale):
    # 1 m closer is a clear win, 10 cm is a contest.
    contest_scale: float = 1.0
    # A ball travelling faster than this is passing a player, not being controlled by them.
    ball_fast_speed: float = 15.0
    ball_fast_weight: float = 0.2

    # -- collective intensity (dead-ball evidence) -----------------------------------------------
    player_speed_clip: float = 10.0       # tracker speed spikes above this are clipped
    min_tracked_players: int = 8          # below this, intensity is not measured on that frame
    intensity_smooth: int = 25            # half-window of the centred moving average (±1 s)
    intensity_llr_clip: float = 3.0
    w_intensity: float = 1.0

    # -- other dead-ball evidence ----------------------------------------------------------------
    out_margin: float = 0.5               # metres past the line before the ball counts as out
    w_out: float = 2.0
    stationary_speed: float = 0.5         # m/s
    stationary_min: int = 25              # the ball must be still this long to count
    w_stationary: float = 0.8
    fast_inside_speed: float = 6.0        # a ball moving this fast on the pitch is in play
    w_fast_inside: float = 1.0
    w_missing: float = 0.3                # ball not seen (and not bridged)
    running_speed: float = 3.0            # a player faster than this is running, not walking
    missing_long: int = 50                # ball unseen this long is its own feature

    # -- self-supervised calibration of the dead-ball evidence -----------------------------------
    # The hand weights above are only the fallback. By default a logistic model is fitted on
    # pseudo-labels produced by Smart Tagging (an algorithm, never a person): "alive" inside tagged
    # phases away from set pieces, "dead" in the run-up to every located restart, after goals and
    # before a half's kick-off. No ground truth is involved.
    calibrate_dead: int = 1
    calib_l2: float = 1.0
    calib_dead_before: int = 62           # pseudo-dead window before a restart: [r-62, r-12)
    calib_dead_gap: int = 12
    calib_exclude_before: int = 150       # no pseudo-alive within 6 s before a set-piece tag...
    calib_exclude_after: int = 50         # ...or 2 s after it
    dead_prior_logodds: float = 0.0       # balanced classes, then this prior
    dead_evidence_clip: float = 6.0
    calib_phase_trim: int = 12            # phase edges are fuzzy; drop 0.5 s at each end
    # Share of graded time the ball is dead, as a prior from OTHER games: LaLiga's effective
    # playing time is ~53-56 min, and the graded window here is 105.9 min including 2 min before
    # the second-half kick-off, so dead ~ 0.47-0.50. The alive/dead threshold is shifted until the
    # decoded share matches. 0 disables this and keeps the model's own threshold.
    # Rule fixed before looking at the competition match: keep the trained model's own threshold if
    # the dead share it gives there is plausible (0.43-0.53), otherwise force 0.48. v3 and v4 gave
    # 0.416 / 0.417 (forced to 0.48); v5 gave 0.434 and v6 gives 0.438, so the model's own
    # threshold is kept -- which is also what scored best on validation (v5: 0.5558 vs 0.5480).
    target_dead_share: float = 0.0
    share_search_iters: int = 18

    # -- supervised models trained on OTHER matches (possession/validate.py, train.py) ------------
    use_trained_models: int = 1           # 1: use possession/models/idsse_models.json
    model_hgb: int = 1                    # [validated] boosted trees: 0.4747 vs logistic 0.4596
    # [validated] OFF: +0.001 held-out spellF1 (noise), and it relies on a synthetic camera for the
    # open matches -- not worth the risk.
    use_camera: int = 0                   # 1: camera aim (homography) as a ball proxy, see camera.py
    context_features: int = 1             # [validated] bracketing / slow context: 0.4897 -> 0.5231
    hgb_iter: int = 200                   # boosted trees: number of trees
    hgb_lr: float = 0.05                  # learning rate
    hgb_leaves: int = 63                  # [validated] leaves per tree (31: 0.5326, 63: 0.5348)
    hgb_min_leaf: int = 200               # minimum samples per leaf
    hgb_l2: float = 1.0                   # L2 regularisation
    stack: int = 1                        # [validated] second stage on temporal aggregates: 0.5350 -> 0.5485
    stack_iter: int = 100                 # second stage: number of trees
    stack_leaves: int = 31                # second stage: leaves per tree
    # [validated] ball touches (validate.touch_events, after Link & Hoernig 2017), with the
    # threshold normalised per match to 15 touches per minute of visible ball: 0.5558 -> 0.5939
    touch_features: int = 1
    # [validated] members on disjoint frame subsamples, logits averaged: 0.5939 -> 0.6025 (tuned)
    ensemble: int = 3
    model_dead_balanced: int = 0          # 0: learn the dead-ball base rate from the training games
    model_dead_scale: float = 1.0
    model_team_scale: float = 1.0
    model_team_clip: float = 6.0

    # -- collective team signals (work while the ball is unseen) ---------------------------------
    collective_smooth: int = 12           # half-window for centroid smoothing and its velocity
    ball_hold_frames: int = 50            # last known ball position is reused this long
    w_collective: float = 0.15            # per-frame weight of the collective team log-odds
    collective_clip: float = 4.0
    team_confident_control: float = 0.8   # |control evidence| that counts as a confident pseudo-label
    team_confident_speed: float = 8.0

    # -- stage 1: alive / dead decoding ----------------------------------------------------------
    # Decoder settings below marked [validated] were chosen by leave-one-match-out spellF1 on the
    # seven open IDSSE matches (possession/tune.py), never on the competition match.
    alive_dead_switch: float = 32.0       # [validated] evidence needed to flip alive <-> dead
    min_dead: int = 25                    # [validated] a stoppage shorter than 1 s is not one
    min_alive: int = 38                   # [validated] live play shorter than 1.5 s between stoppages

    # -- restarts from Smart Tagging -------------------------------------------------------------
    tag_search_pad: int = 25              # frames searched either side of a set-piece window
    kick_pre_win: int = 12                # ball must be slow over the 0.5 s before a kick...
    kick_pre_max: float = 2.0
    kick_post_win: int = 8                # ...and fast over the 0.3 s after it
    kick_post_min: float = 5.0
    anchor_dead_before: int = 38          # frames forced dead before a located restart
    anchor_alive_after: int = 50          # frames forced alive after it
    anchor_weight: float = 50.0           # per-frame score of a located anchor
    anchor_weight_unlocated: float = 3.0  # per-frame score when the kick could not be found
    goal_search_after: int = 75           # frames after a GOALS window searched for the ball in the net
    goal_ko_max_gap: int = 4500           # a kick-off over 3 min after a goal is not its restart

    # -- stage 2: home / away decoding -----------------------------------------------------------
    # [validated] Evidence (in team-model log-odds summed over frames) needed to hand possession
    # to the other team: a touch or a deflection does not pay it, sustained control does.
    team_switch: float = 4.0
    # Each alive stretch is decoded together with the restart_window frames before it, so the
    # player standing over the dead ball -- the taker -- decides who starts with it.
    restart_window: int = 25
    # Per-frame bias towards the team Smart Tagging names on a located restart, applied over
    # [r - restart_window, r + restart_window). Sustained, not a one-frame head start: a head start
    # only buys the first frame, since one switch is cheaper than it. [validated]
    tag_team_bias: float = 0.5
    # Restarts by the Laws of the Game (decode.restart_rule_bias). [validated] OFF: it slightly
    # lowered held-out spellF1 at every strength tried -- "last toucher" read from noisy tracking
    # is less reliable than who stands over the ball at the restart.
    rule_bias: float = 0.0
    rule_ball_window: int = 50            # ball position read within 2 s before the restart
    rule_touch_window: int = 50           # last touch read within 2 s before the stoppage
    # Snap home<->away changes to the first frame of clear control (decode.snap_team_edges).
    # [validated] frames either side searched; 0 disables. Helped v5 (+0.009); with touch features
    # the models place team changes well enough on their own (0: 0.6025, 12: 0.6006).
    team_snap_window: int = 0
    team_snap_threshold: float = 0.5      # |control evidence| that counts as clear control

    # -- boundary refinement ---------------------------------------------------------------------
    snap_kick_window: int = 50            # a dead->alive edge snaps to a kick within ±2 s
    snap_out_before: int = 75             # an alive->dead edge snaps to the ball leaving the pitch
    snap_out_after: int = 25


def with_overrides(cfg, pairs):
    """Copy of cfg with KEY=VALUE overrides from the command line, cast to each field's type."""
    types = {f.name: f.type for f in fields(cfg)}
    changes = {}
    for pair in pairs or ():
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in types:
            raise KeyError("unknown config key {!r}".format(key))
        cast = {"int": int, "float": float}.get(getattr(types[key], "__name__", types[key]), float)
        changes[key] = cast(value)
    return replace(cfg, **changes)
