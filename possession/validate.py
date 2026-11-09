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


def team_matrix(feats):
    """Per-frame inputs of the team model. Columns oriented so positive favours home."""
    ev = feats["team_ev"]
    cols = [ev,
            features.moving_nanmean(ev, -12, 13),
            features.moving_nanmean(ev, -50, 51),
            features.moving_nanmean(ev, -125, 126)]
    X = np.column_stack(cols + [feats["team_x"]])
    return np.where(np.isfinite(X), X, 0.0)


CONTROL_NAMES = ["control", "control_1s", "control_4s", "control_10s"]
TEAM_NAMES = CONTROL_NAMES + features.TEAM_X_NAMES


def team_names(feats):
    return CONTROL_NAMES + list(feats["team_x_names"])


def prepare(cfg, matches=None, verbose=True):
    """Degraded features, synthetic tags and harmonised truths for each external match."""
    matches = matches or [m for m in external.MATCHES if external.path_of(m, "tracking")]
    trk_atd = atd.load(verbose=False)
    run_lengths = external.atd_run_lengths(trk_atd, features.default_match()["halves"])
    out = []
    for i, mid in enumerate(matches):
        t = time.time()
        m = external.load(mid)
        deg = external.degrade(m, run_lengths, seed=1000 + external.MATCHES.index(mid))
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
                    "team_X": team_matrix(feats), "deg": deg})
        if verbose:
            print("  prepared {}  ({:.0f} s)".format(mid, time.time() - t))
    return out


def fit_models(train, cfg):
    """(dead_model, team_model) fitted on the training matches."""
    dx, dy, tx, ty = [], [], [], []
    for p in train:
        truth = p["truths"][TRAIN_LEVEL]
        g = p["feats"]["graded"]
        idx = np.flatnonzero(g)[::SUBSAMPLE]
        dx.append(p["feats"]["dead_x"][idx])
        dy.append((truth[idx] == "D").astype(float))
        alive = idx[np.isin(truth[idx], ["H", "A"])]
        tx.append(p["team_X"][alive])
        ty.append((truth[alive] == "H").astype(float))
    dx, dy, tx, ty = map(np.concatenate, (dx, dy, tx, ty))
    w_dead = np.full(len(dy), 1.0 / len(dy))
    if cfg.model_dead_balanced:
        for c in (0.0, 1.0):
            w_dead[dy == c] = 0.5 / max((dy == c).sum(), 1)
    w_team = np.zeros(len(ty))
    for c in (0.0, 1.0):
        w_team[ty == c] = 0.5 / max((ty == c).sum(), 1)
    if cfg.model_hgb:
        return _fit_hgb(dx, dy, w_dead), _fit_hgb(tx, ty, w_team)
    dead_model = calibrate.fit_logistic(dx, dy, w_dead * 2000.0, cfg.calib_l2)
    team_model = calibrate.fit_logistic(tx, ty, w_team * 2000.0, cfg.calib_l2)
    return dead_model, team_model


def _fit_hgb(X, y, w):
    """Gradient-boosted trees (scikit-learn), deterministic: fixed seed, no early stopping."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)
    model.fit(X, y, sample_weight=w * len(w))
    return model


def _logit(X, model):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return calibrate.predict_logit(X, model)


def model_scores(feats, team_X, models, cfg):
    """(ldead, team_score) per frame from fitted models."""
    dead_model, team_model = models
    ldead = np.clip(_logit(feats["dead_x"], dead_model) * cfg.model_dead_scale
                    + cfg.dead_prior_logodds, -cfg.dead_evidence_clip, cfg.dead_evidence_clip)
    team = np.clip(_logit(team_X, team_model), -cfg.model_team_clip,
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
        ldead, team = model_scores(p["feats"], p["team_X"], fitted, cfg)
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
