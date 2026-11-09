"""Leave-one-match-out validation on degraded open matches from OTHER games (IDSSE).

    python -m possession.validate                  # score the current config, and the baseline
    python -m possession.validate --set team_switch=16 rule_bias=1

For each of the seven matches: fit the per-frame models on the other six, run the full decoder on
the held-out one, and score it with the official metric code (evaluation/possession_metrics.py)
against its DFL labels converted to controlled possession. Every number is averaged over three
plausible readings of "controlled" (spells under 0.5 / 1 / 2 s merged), so a choice that only wins
under one reading does not get picked.

Realism check: the organisers' baseline (nearest player + majority vote over +/-2 s) is scored the
same way. It reaches 0.2433 on the competition match; if it lands near that here, the degraded
matches are about as hard as the real one.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
from possession import atd, calibrate, decode, external, features, tags  # noqa: E402
from possession.config import Config, with_overrides  # noqa: E402

CONTROL_LEVELS = (0.5, 1.0, 2.0)
TRAIN_LEVEL = 1.0
SUBSAMPLE = 5           # consecutive frames are near-duplicates; every 5th is plenty to fit on
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def _last_next(mask):
    """(index of the last True at or before t, index of the next True at or after t)."""
    n = len(mask)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, idx, -1))
    nxt = np.minimum.accumulate(np.where(mask, idx, n)[::-1])[::-1]
    return last, nxt


def _ages(mask, cap_s=60.0):
    """(seconds since mask was last True, seconds until it is next True, last, next), capped."""
    n = len(mask)
    idx = np.arange(n)
    last, nxt = _last_next(mask)
    since = np.where(last >= 0, (idx - last) / 25.0, cap_s)
    until = np.where(nxt < n, (nxt - idx) / 25.0, cap_s)
    return np.minimum(since, cap_s), np.minimum(until, cap_s), last, nxt


def team_context(feats):
    """Bracketing and slow context for the team model (positive favours home).

    Inside a long ball gap the best evidence is which team clearly had the ball just before it and
    just after it, and how long ago / how soon that was; plus slower versions of the shape signals.
    The match is processed offline, so looking ahead is legitimate.
    """
    ev = feats["team_ev"]
    n = len(ev)
    since, until, last, nxt = _ages(np.abs(ev) >= 0.5)
    last_sign = np.where(last >= 0, np.sign(ev[np.maximum(last, 0)]), 0.0)
    next_sign = np.where(nxt < n, np.sign(ev[np.minimum(nxt, n - 1)]), 0.0)
    ball_since, ball_until, _, _ = _ages(feats["known"])
    tx = feats["team_x"]
    mm = features.moving_nanmean
    cols = [last_sign, next_sign, np.log1p(since), np.log1p(until),
            last_sign * np.exp(-since / 5.0), next_sign * np.exp(-until / 5.0),
            mm(ev, -50, 0), mm(ev, 1, 51),
            np.log1p(ball_since), np.log1p(ball_until),
            mm(tx[:, 1], -125, 126), mm(tx[:, 1], -250, 251),
            mm(tx[:, 0], -125, 126), mm(tx[:, 5], -125, 126), mm(tx[:, 3], -125, 126)]
    names = ["last_control_team", "next_control_team", "log_s_since_control",
             "log_s_until_control", "last_control_decay5s", "next_control_decay5s",
             "control_past_2s", "control_future_2s", "log_s_since_ball", "log_s_until_ball",
             "goal_side_5s", "goal_side_10s", "forward_velocity_5s", "centroid_5s", "width_5s"]
    return cols, names


def dead_context(feats):
    """Context for the dead-ball model: ball sightings, acceleration into a restart, recent outs."""
    dx = feats["dead_x"]
    mm = features.moving_nanmean
    ball_since, ball_until, _, _ = _ages(feats["known"])
    log_int = dx[:, 1]
    cols = [np.log1p(ball_since), np.log1p(ball_until),
            mm(log_int, 1, 51) - mm(log_int, -50, 0),       # players speeding up = restart
            mm(log_int, -125, 126), mm(dx[:, 2], -125, 126),
            mm(dx[:, 3], -125, 1), mm(dx[:, 3], 0, 126),    # ball out in the last / next 5 s
            mm(dx[:, 7], -125, 126)]
    names = ["log_s_since_ball", "log_s_until_ball", "intensity_acceleration_2s",
             "log_intensity_5s", "running_share_5s", "ball_out_past_5s", "ball_out_next_5s",
             "ball_missing_share_5s"]
    return cols, names


def team_matrix(feats, cfg=None):
    """Per-frame inputs of the team model. Columns oriented so positive favours home."""
    ev = feats["team_ev"]
    cols = [ev,
            features.moving_nanmean(ev, -12, 13),
            features.moving_nanmean(ev, -50, 51),
            features.moving_nanmean(ev, -125, 126)]
    X = np.column_stack(cols + [feats["team_x"]])
    if cfg is not None and cfg.context_features:
        X = np.column_stack([X] + team_context(feats)[0])
    return np.where(np.isfinite(X), X, 0.0)


def dead_matrix(feats, cfg=None):
    """Per-frame inputs of the dead-ball model."""
    X = feats["dead_x"]
    if cfg is not None and cfg.context_features:
        X = np.column_stack([X] + dead_context(feats)[0])
    return np.where(np.isfinite(X), X, 0.0)


CONTROL_NAMES = ["control", "control_1s", "control_4s", "control_10s"]
TEAM_NAMES = CONTROL_NAMES + features.TEAM_X_NAMES


def team_names(feats, cfg=None):
    names = CONTROL_NAMES + list(feats["team_x_names"])
    if cfg is not None and cfg.context_features:
        names += team_context(feats)[1]
    return names


def dead_names(feats, cfg=None):
    names = list(feats["dead_x_names"])
    if cfg is not None and cfg.context_features:
        names += dead_context(feats)[1]
    return names


def prepare(cfg, matches=None, verbose=True, degrade_kw=None):
    """Degraded features, synthetic tags and harmonised truths for each external match."""
    matches = matches or [m for m in external.MATCHES if external.path_of(m, "tracking")]
    trk_atd = atd.load(verbose=False)
    run_lengths = external.atd_run_lengths(trk_atd, features.default_match()["halves"])
    out = []
    for i, mid in enumerate(matches):
        t = time.time()
        m = external.load(mid)
        deg = external.degrade(m, run_lengths, seed=1000 + external.MATCHES.index(mid),
                               **(degrade_kw or {}))
        if cfg.use_camera:
            from possession import camera
            deg["aim_x"] = camera.synthetic_aim(m["ball"][:, 0].astype(float) * m["pitch"][0],
                                                m["pitch"][0],
                                                seed=3000 + external.MATCHES.index(mid))
        match = {"name": mid, "halves": m["halves"], "pitch": m["pitch"]}
        feats = features.compute(deg, cfg, match)
        events = external.synth_tags(m, seed=2000 + external.MATCHES.index(mid))
        truths = {lvl: external.harmonise(m["labels"], m["halves"], lvl) for lvl in CONTROL_LEVELS}
        out.append({"id": mid, "feats": feats, "events": events, "truths": truths,
                    "team_X": team_matrix(feats, cfg), "dead_X": dead_matrix(feats, cfg),
                    "deg": deg})
        if verbose:
            print("  prepared {}  ({:.0f} s)".format(mid, time.time() - t))
    return out


def _rows(train, dead_mats, team_mats):
    """Stacked training rows (every SUBSAMPLE-th graded frame) for the dead and team models."""
    dx, dy, tx, ty = [], [], [], []
    for p, dX, tX in zip(train, dead_mats, team_mats):
        truth = p["truths"][TRAIN_LEVEL]
        idx = np.flatnonzero(p["feats"]["graded"])[::SUBSAMPLE]
        dx.append(dX[idx])
        dy.append((truth[idx] == "D").astype(float))
        alive = idx[np.isin(truth[idx], ["H", "A"])]
        tx.append(tX[alive])
        ty.append((truth[alive] == "H").astype(float))
    return tuple(map(np.concatenate, (dx, dy, tx, ty)))


def _fit_pair(dx, dy, tx, ty, cfg, stage2=False):
    """(dead_model, team_model): dead with the training games' base rate, team balanced."""
    w_dead = np.full(len(dy), 1.0 / len(dy))
    if cfg.model_dead_balanced:
        for c in (0.0, 1.0):
            w_dead[dy == c] = 0.5 / max((dy == c).sum(), 1)
    w_team = np.zeros(len(ty))
    for c in (0.0, 1.0):
        w_team[ty == c] = 0.5 / max((ty == c).sum(), 1)
    if cfg.model_hgb:
        return _fit_hgb(dx, dy, w_dead, cfg, stage2), _fit_hgb(tx, ty, w_team, cfg, stage2)
    return (calibrate.fit_logistic(dx, dy, w_dead * 2000.0, cfg.calib_l2),
            calibrate.fit_logistic(tx, ty, w_team * 2000.0, cfg.calib_l2))


def stage2_matrix(feats, l_dead, l_team):
    """Inputs of the second-stage models: temporal aggregates of the first-stage log-odds.

    Lets a model learn patterns over time that one frame cannot show -- evidence that flips and
    flips back (a deflection), team evidence fading into a stoppage, a restart's build-up.
    """
    mm = features.moving_nanmean
    cols = []
    for l in (l_dead, l_team):
        flips = np.abs(np.diff(np.sign(l), prepend=np.sign(l[:1]))) > 0
        cols += [l, mm(l, -25, 0), mm(l, 1, 26), mm(l, -125, 0), mm(l, 1, 126),
                 mm(l, -375, 376), mm(flips.astype(float), -50, 51)]
    cols += [feats["team_ev"], feats["known"].astype(float)]
    X = np.column_stack(cols)
    return np.where(np.isfinite(X), X, 0.0)


STAGE2_NAMES = [p + s for p in ("dead_", "team_") for s in
                ("logit", "past_1s", "next_1s", "past_5s", "next_5s", "15s", "flip_rate_2s")] \
    + ["control", "ball_known"]


def fit_models(train, cfg):
    """(dead, team) models fitted on the training matches; with cfg.stack also the second stage,
    trained on OUT-OF-FOLD first-stage predictions (each training match predicted by a first
    stage fitted without it), returned as (dead1, team1, dead2, team2)."""
    first = _fit_pair(*_rows(train, [p["dead_X"] for p in train], [p["team_X"] for p in train]),
                      cfg)
    if not cfg.stack:
        return first
    mats = []
    for j, p in enumerate(train):
        others = train[:j] + train[j + 1:]
        d1, t1 = _fit_pair(*_rows(others, [q["dead_X"] for q in others],
                                  [q["team_X"] for q in others]), cfg)
        mats.append(stage2_matrix(p["feats"], _logit(p["dead_X"], d1), _logit(p["team_X"], t1)))
    second = _fit_pair(*_rows(train, mats, mats), cfg, stage2=True)
    return first + second


def _fit_hgb(X, y, w, cfg, stage2=False):
    """Gradient-boosted trees (scikit-learn), deterministic: fixed seed, no early stopping."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=cfg.stack_iter if stage2 else cfg.hgb_iter,
                                           learning_rate=cfg.hgb_lr,
                                           max_leaf_nodes=cfg.stack_leaves if stage2
                                           else cfg.hgb_leaves,
                                           min_samples_leaf=cfg.hgb_min_leaf,
                                           l2_regularization=cfg.hgb_l2,
                                           early_stopping=False, random_state=0)
    model.fit(X, y, sample_weight=w * len(w))
    return model


def _logit(X, model):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return calibrate.predict_logit(X, model)


def model_scores(feats, team_X, models, cfg, dead_X=None):
    """(ldead, team_score) per frame from fitted models."""
    dead_model, team_model = models[0], models[1]
    dead_X = dead_matrix(feats, cfg) if dead_X is None else dead_X
    for X, model, what in ((dead_X, dead_model, "dead"), (team_X, team_model, "team")):
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError("{} model expects {} features, got {}: the config does not match the "
                             "saved models (context_features?)".format(what, expected, X.shape[1]))
    l_dead, l_team = _logit(dead_X, dead_model), _logit(team_X, team_model)
    if len(models) == 4:                                  # second stage on temporal aggregates
        X2 = stage2_matrix(feats, l_dead, l_team)
        l_dead, l_team = _logit(X2, models[2]), _logit(X2, models[3])
    ldead = np.clip(l_dead * cfg.model_dead_scale
                    + cfg.dead_prior_logodds, -cfg.dead_evidence_clip, cfg.dead_evidence_clip)
    team = np.clip(l_team, -cfg.model_team_clip,
                   cfg.model_team_clip) * cfg.model_team_scale
    team = np.where(feats["direction"] != 0, team, 0.0)
    return ldead, team


def score(labels, truths):
    """{metric: mean over control levels} with the official metric code."""
    pred = np.array([CODE[v] for v in labels], dtype=np.int8)
    rows = []
    for lvl, truth in truths.items():
        gt = np.array([CODE[v] for v in truth], dtype=np.int8)
        r = pm.evaluate(gt, pred)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                     r["alive_dead_f1"], r["team_acc_both_alive"]))
    a = np.array(rows, dtype=float).mean(axis=0)
    return {"spellF1": a[0], "macroF1": a[1], "pred_poss": a[2], "gt_poss": a[3],
            "aliveDeadF1": a[4], "teamAcc": a[5]}


def baseline_labels(p):
    """The organisers' baseline on the degraded match: nearest player, vote over +/-50 frames."""
    f = p["feats"]
    seen = np.isfinite(p["deg"]["ball"][:, 0])
    codes = np.where(~seen, -1, np.where(f["d_home"] <= f["d_away"], 0, 1))
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - 50), np.minimum(len(codes), i + 51)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    lab = np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))
    return np.where(f["graded"], lab, "X")


def evaluate(prepared, cfg, models=None, verbose=True):
    """Leave-one-match-out over the prepared matches. Returns (mean row, per-match rows)."""
    rows = []
    for k, p in enumerate(prepared):
        fitted = models[k] if models else fit_models(prepared[:k] + prepared[k + 1:], cfg)
        ldead, team = model_scores(p["feats"], p["team_X"], fitted, cfg, p["dead_X"])
        restarts = tags.locate_restarts(p["events"], p["feats"], cfg)
        labels, _ = decode.decode(p["feats"], restarts, [], cfg, ldead=ldead,
                                  team_score=team)
        r = score(labels, p["truths"])
        r["id"] = p["id"]
        rows.append(r)
        if verbose:
            print("  {id}  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {pred_poss:.0f} "
                  "(gt {gt_poss:.0f})  aliveDeadF1 {aliveDeadF1:.3f}  teamAcc {teamAcc:.3f}".format(**r))
    keys = ["spellF1", "macroF1", "pred_poss", "gt_poss", "aliveDeadF1", "teamAcc"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return mean, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = with_overrides(Config(), args.set)

    print("preparing degraded external matches ...")
    prepared = prepare(cfg)

    print("\norganisers' baseline (realism check; 0.2433 on the competition match):")
    base = [score(baseline_labels(p), p["truths"]) for p in prepared]
    print("  mean spellF1 {:.4f}  macroF1 {:.4f}".format(np.mean([b["spellF1"] for b in base]),
                                                         np.mean([b["macroF1"] for b in base])))

    print("\nour pipeline, leave-one-match-out:")
    mean, _ = evaluate(prepared, cfg)
    print("  MEAN  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {pred_poss:.0f} (gt "
          "{gt_poss:.0f})  aliveDeadF1 {aliveDeadF1:.3f}  teamAcc {teamAcc:.3f}".format(**mean))


if __name__ == "__main__":
    main()
