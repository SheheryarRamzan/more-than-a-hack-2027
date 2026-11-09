"""Self-supervised calibration of the dead-ball evidence.

A weighted, L2-regularised logistic regression fitted by Newton's method (IRLS) on the
pseudo-labels from tags.dead_pseudo_labels. numpy only, deterministic. Its output replaces the
hand-weighted dead-ball score: same inputs, but the weights come from the match's own automatic
tagging instead of from someone's guess.
"""

import numpy as np


def fit_logistic(X, y, w, l2, iters=100):
    """(beta, mu, sd): intercept-first coefficients on standardised features."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    penalty = np.full(Z.shape[1], l2)
    penalty[0] = 0.0                                   # never shrink the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
        grad = Z.T @ (w * (y - p)) - penalty * beta
        hess = (Z * (w * p * (1.0 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta, mu, sd


def predict_logit(X, model):
    beta, mu, sd = model
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ beta


def collective_team_evidence(feats, y, w, cfg):
    """(log-odds home-in-possession float[n], report) from the collective shape features."""
    X = feats["team_x"]
    use = y >= 0
    n_home, n_away = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_home < 200 or n_away < 200:
        return np.zeros(len(X)), {"fitted": False}
    model = fit_logistic(X[use], y[use].astype(np.float64), w[use] * 2000.0, cfg.calib_l2)
    logit = np.clip(predict_logit(X, model), -cfg.collective_clip, cfg.collective_clip)
    # Outside the halves the orientation is undefined; no evidence there.
    logit = np.where(feats["direction"] != 0, logit, 0.0)
    fitted = predict_logit(X[use], model) > 0
    truth = y[use] == 1
    report = {
        "fitted": True,
        "pseudo_home": n_home,
        "pseudo_away": n_away,
        "balanced_accuracy_on_pseudo": float(0.5 * (np.mean(fitted[truth])
                                                    + np.mean(~fitted[~truth]))),
        "coefficients": {name: round(float(b), 3) for name, b in
                         zip(["intercept"] + feats["team_x_names"], model[0])},
    }
    return logit, report


def calibrated_dead_evidence(feats, y, w, cfg):
    """(ldead float[n], report dict). Falls back to the hand-weighted score if too few labels."""
    X = feats["dead_x"]
    use = y >= 0
    n_dead, n_alive = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_dead < 200 or n_alive < 200:
        return feats["ldead"], {"calibrated": False, "pseudo_dead": n_dead, "pseudo_alive": n_alive}

    # Weights are tiny after normalisation; scale them so the L2 term has a sensible size relative
    # to the data term (equivalent to a per-class sample size of ~1000).
    model = fit_logistic(X[use], y[use].astype(np.float64), w[use] * 2000.0, cfg.calib_l2)
    logit = predict_logit(X, model) + cfg.dead_prior_logodds
    ldead = np.clip(logit, -cfg.dead_evidence_clip, cfg.dead_evidence_clip)

    fitted = predict_logit(X[use], model) > 0
    truth = y[use] == 1
    report = {
        "calibrated": True,
        "pseudo_dead": n_dead,
        "pseudo_alive": n_alive,
        "recall_on_pseudo_dead": float(np.mean(fitted[truth])),
        "recall_on_pseudo_alive": float(np.mean(~fitted[~truth])),
        "coefficients": {name: round(float(b), 3) for name, b in
                         zip(["intercept"] + feats["dead_x_names"], model[0])},
    }
    return ldead, report
