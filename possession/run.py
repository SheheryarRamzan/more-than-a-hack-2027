"""Build the possession file from the provided data.

    python -m possession.run                      # writes evaluation/results/matrix_ae.csv
    python -m possession.run --name v2 --set team_switch=10 alive_dead_switch=16

Then validate and score it exactly as the organisers will:

    python evaluation/validate_submission.py evaluation/results/matrix_ae.csv
    python evaluation/run_evaluation.py

Deterministic: the same data and the same config give the same file, byte for byte.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession import atd, calibrate, decode, features, report, tags  # noqa: E402
from possession.config import Config, with_overrides  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="matrix_ae", help="submission name (file stem)")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, see possession/config.py")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw tracking")
    args = parser.parse_args(argv)

    started = time.time()
    cfg = with_overrides(Config(), args.set)
    labels, ctx = build(cfg, use_cache=not args.no_cache)
    feats, events, restarts, calib, team_model, goal_spans = (
        ctx["feats"], ctx["events"], ctx["restarts"], ctx["calib"], ctx["team_model"],
        ctx["goal_spans"])

    path = write_submission(os.path.join(args.out_dir, args.name + ".csv"), labels)

    summary = {"features": feats["stats"],
               "dead_ball_model": calib,
               "team_model": team_model,
               "segments": report.segment_stats(labels),
               "smart_tagging": report.tagging_consistency(labels, events, restarts),
               "goal_stoppages_anchored": len(goal_spans)}
    report.print_block("tracking", {k: v for k, v in feats["stats"].items()
                                    if not isinstance(v, dict)})
    if feats["stats"]["intensity_mixture"]:
        report.print_block("intensity mixture (unsupervised)", feats["stats"]["intensity_mixture"])
    report.print_block("dead-ball model (self-supervised on Smart Tagging)",
                       {k: v for k, v in calib.items() if k != "coefficients"})
    if "coefficients" in calib:
        report.print_block("  coefficients (standardised)", calib["coefficients"])
    report.print_block("collective team model (self-supervised)",
                       {k: v for k, v in team_model.items() if k != "coefficients"})
    if "coefficients" in team_model:
        report.print_block("  coefficients (standardised)", team_model["coefficients"])
    report.print_block("attack direction (from goalkeeper positions)",
                       feats["stats"]["collective"])
    report.print_block("segments (graded frames)", summary["segments"])
    report.print_block("consistency with Smart Tagging", summary["smart_tagging"])

    with open(os.path.join(args.out_dir, args.name + ".json"), "w") as handle:
        json.dump({"name": args.name, "config": asdict(cfg), "summary": summary}, handle, indent=2)

    try:
        shown = os.path.relpath(path, ROOT)
    except ValueError:                                    # --out-dir on another Windows drive
        shown = path
    print("\nwrote {}  ({:.1f} s)".format(shown, time.time() - started))
    print("next:  python evaluation/validate_submission.py {}".format(shown))
    return path


def build(cfg, use_cache=True, verbose=True):
    """(labels, context): the whole pipeline, data in, labels out. Shared with diagnostics."""
    trk = atd.load(use_cache=use_cache, verbose=verbose)
    if verbose:
        print("teams: H = {}, A = {}".format(trk["home"], trk["away"]))
        for note in trk["notes"]:
            print("  " + note)

    if cfg.use_camera:
        from possession import camera
        trk = dict(trk)
        trk["aim_x"] = camera.aim_x()
    feats = features.compute(trk, cfg)
    events = tags.load_events(trk["home"], trk["away"])
    restarts = tags.locate_restarts(events, feats, cfg)
    goal_spans = tags.goal_dead_spans(events, restarts, feats, cfg)

    if cfg.use_trained_models:
        # Models trained on the open IDSSE matches (other games), see possession/train.py.
        from possession import train, validate
        models, payload = train.load_models()
        ldead, team_score = validate.model_scores(feats, validate.team_matrix(feats, cfg),
                                                  models, cfg)
        calib = {"source": payload["trained_on"]}
        team_model = {"source": payload["trained_on"]}
    else:
        # Fallback: self-supervised on Smart Tagging pseudo-labels (v2).
        ldead, calib = feats["ldead"], {"calibrated": False}
        if cfg.calibrate_dead:
            y, w = tags.dead_pseudo_labels(events, restarts, goal_spans, feats["graded"], cfg,
                                           feats["halves"])
            ldead, calib = calibrate.calibrated_dead_evidence(feats, y, w, cfg)
        ty, tw = tags.team_pseudo_labels(events, feats, cfg)
        collective, team_model = calibrate.collective_team_evidence(feats, ty, tw, cfg)
        team_score = feats["team_ev"] + cfg.w_collective * collective

    labels, inner = decode.decode(feats, restarts, goal_spans, cfg, ldead=ldead,
                                  team_score=team_score)
    calib["dead_threshold_offset"] = round(float(inner["dead_offset"]), 3)

    ctx = {"trk": trk, "feats": feats, "events": events, "restarts": restarts,
           "goal_spans": goal_spans, "ldead": ldead, "calib": calib,
           "team_score": team_score, "team_model": team_model, "inner": inner}
    return labels, ctx


if __name__ == "__main__":
    main()
