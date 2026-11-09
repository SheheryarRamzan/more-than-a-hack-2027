"""Fit the per-frame models on the seven open IDSSE matches (other games) and save them.

    python -m possession.train          # downloads IDSSE if missing (~2.6 GB), ~10 min

The saved file, possession/models/idsse_models.json, holds only logistic-regression coefficients
learned from OTHER matches -- no data from the competition match, which is why it can be committed
to a public repository. possession.run reads it, so regenerating the submission does not need the
download; this script is how the file itself is regenerated, deterministically.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession import external, validate  # noqa: E402
from possession.config import Config  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "possession", "models", "idsse_models.json")


def _pack(model, names):
    if not isinstance(model, tuple):                    # gradient-boosted trees
        import base64
        import pickle
        return {"names": names, "kind": "hgb",
                "pickle_b64": base64.b64encode(pickle.dumps(model, protocol=4)).decode("ascii")}
    beta, mu, sd = model
    return {"names": names, "beta": [float(v) for v in beta],
            "mu": [float(v) for v in mu], "sd": [float(v) for v in sd]}


def _unpack(d):
    if d.get("kind") == "hgb":
        import base64
        import pickle
        return pickle.loads(base64.b64decode(d["pickle_b64"]))
    return (np.array(d["beta"]), np.array(d["mu"]), np.array(d["sd"]))


def save_models(models, names, path=MODEL_PATH, matches=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"trained_on": "IDSSE open data (CC BY 4.0), matches " + ", ".join(matches),
               "dead": _pack(models[0], names[0]), "team": _pack(models[1], names[1])}
    with open(path, "w", newline="\n") as handle:
        json.dump(payload, handle, indent=1)
    return path


def load_models(path=MODEL_PATH):
    with open(path) as handle:
        payload = json.load(handle)
    return (_unpack(payload["dead"]), _unpack(payload["team"])), payload


def main():
    cfg = Config()
    for mid in external.MATCHES:
        external.download(mid)
    prepared = validate.prepare(cfg, list(external.MATCHES))
    models = validate.fit_models(prepared, cfg)
    names = (validate.dead_names(prepared[0]["feats"], cfg),
             validate.team_names(prepared[0]["feats"], cfg))
    path = save_models(models, names, matches=external.MATCHES)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    for label, model, cols in (("dead", models[0], names[0]), ("team", models[1], names[1])):
        if not isinstance(model, tuple):
            continue
        print("\n{} model (standardised coefficients)".format(label))
        for name, b in zip(["intercept"] + list(cols), model[0]):
            print("  {:<26} {:+.3f}".format(name, b))


if __name__ == "__main__":
    main()
