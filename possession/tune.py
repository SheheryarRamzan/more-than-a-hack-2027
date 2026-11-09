"""Choose the decoder's settings on held-out OTHER matches, never on the competition match.

    python -m possession.tune

Coordinate descent over a small grid: each setting in turn is set to every candidate value with
the others held fixed, and the value with the best mean leave-one-match-out spellF1 (macroF1 to
break ties) is kept; two passes. The per-frame models are fitted once per held-out match, since
they do not depend on the decoder settings. The chosen values are printed so they can be written
into config.py with their provenance.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import replace  # noqa: E402

from possession import validate  # noqa: E402
from possession.config import Config  # noqa: E402

GRID = {
    "team_switch": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    "alive_dead_switch": [3.0, 6.0, 12.0, 24.0, 48.0],
    "min_dead": [25, 50, 100],
    "min_alive": [12, 38, 75],
    "rule_bias": [0.0, 0.5, 1.0, 2.0],
    "tag_team_bias": [0.5, 1.0, 2.0, 4.0],
    "restart_window": [12, 25, 50],
    "model_team_scale": [0.5, 1.0, 2.0],
}


def key(mean):
    return (round(mean["spellF1"], 4), round(mean["macroF1"], 4))


def main():
    cfg = Config(target_dead_share=0.0)
    print("preparing ...")
    prepared = validate.prepare(cfg)

    base = [validate.score(validate.baseline_labels(p), p["truths"]) for p in prepared]
    print("organisers' baseline: spellF1 {:.4f}  macroF1 {:.4f}   (0.2433 / 0.6724 on the "
          "competition match)".format(sum(b["spellF1"] for b in base) / len(base),
                                      sum(b["macroF1"] for b in base) / len(base)))

    # Which per-frame model family, at the default decoder settings.
    choice = None
    for hgb in (0, 1):
        trial = replace(cfg, model_hgb=hgb)
        t = time.time()
        models = [validate.fit_models(prepared[:k] + prepared[k + 1:], trial)
                  for k in range(len(prepared))]
        mean, _ = validate.evaluate(prepared, trial, models, verbose=False)
        print("{:<10} spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {pred_poss:.0f} (gt "
              "{gt_poss:.0f})  aliveDeadF1 {aliveDeadF1:.3f}  teamAcc {teamAcc:.3f}  ({t:.0f} s)"
              .format("hgb" if hgb else "logistic", t=time.time() - t, **mean))
        if choice is None or key(mean) > key(choice[1]):
            choice = (trial, mean, models)
    cfg, best, models = choice
    for sweep in range(2):
        for name, values in GRID.items():
            t = time.time()
            results = []
            for value in values:
                trial = replace(cfg, **{name: value})
                mean, _ = validate.evaluate(prepared, trial, models, verbose=False)
                results.append((key(mean), value, mean))
            top = max(results, key=lambda r: r[0])
            if top[0] > key(best):
                cfg, best = replace(cfg, **{name: top[1]}), top[2]
            print("pass {} {:<18} ".format(sweep + 1, name)
                  + "  ".join("{}:{:.4f}".format(v, r[0][0]) for r in results for v in [r[1]])
                  + "   -> {} ({:.0f} s)".format(getattr(cfg, name), time.time() - t))
    print("\nbest  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {pred_poss:.0f} "
          "(gt {gt_poss:.0f})  aliveDeadF1 {aliveDeadF1:.3f}  teamAcc {teamAcc:.3f}".format(**best))
    print("chosen: model_hgb={}, ".format(cfg.model_hgb)
          + ", ".join("{}={}".format(k, getattr(cfg, k)) for k in GRID))
    print("\nper match at the chosen settings:")
    validate.evaluate(prepared, cfg, models, verbose=True)


if __name__ == "__main__":
    main()
