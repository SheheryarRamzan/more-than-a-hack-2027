"""Validation and training data from OTHER matches: the IDSSE open dataset.

Seven full German Bundesliga / 2. Bundesliga matches (2022/23) with official DFL tracking, released
under CC BY 4.0 by Bassek, Weber, Rein & Memmert, "An integrated dataset of synchronized
spatiotemporal and event data in elite soccer" (Scientific Data, 2025). RULES.md allows external
data about other games, and this is public, so the organisers can access it.

Every ball frame in the DFL tracking carries BallStatus (alive/dead) and BallPossession (home/away)
from the corrected official tracking -- the same kind of per-frame label the challenge's ground
truth is derived from. That gives ~2,500 possessions to measure against instead of 17.

Three steps turn a DFL match into something the pipeline can be tested on honestly:

  load()      -> the same arrays atd.load() returns (normalised 0-1 positions, speed in m/s),
                 plus the per-frame H/A/D truth and the match's halves and pitch size.
  degrade()   -> make the clean official tracking look like the automatic ATD: the ball is removed
                 in long stretches whose lengths are drawn from the ATD's own gap statistics, and
                 players drop out and jitter. Without this the validation would flatter us.
  synth_tags()-> Smart Tagging-like set-piece tags at the true restarts, with the same coverage
                 pattern the provided tagging has (all corners, some throw-ins with no team, few
                 free kicks).

Nothing here touches the competition match's labels; the only thing read from the provided data is
the run-length statistics of the ATD ball (how long it stays seen and lost), never a label.
"""

import glob
import os
import re
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

EXT_DIR = os.path.join(DATA_DIR, "external", "idsse")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FPS = 25
HALF_GAP = 250          # ungraded frames inserted between the halves of an external match

# figshare file ids, from kloppy's IDSSE table (kloppy/_providers/sportec.py).
IDSSE = {
    "J03WPY": {"meta": 51643487, "event": 51643505, "tracking": 51643526},
    "J03WN1": {"meta": 51643472, "event": 51643496, "tracking": 51643517},
    "J03WMX": {"meta": 51643475, "event": 51643493, "tracking": 51643514},
    "J03WOH": {"meta": 51643478, "event": 51643499, "tracking": 51643520},
    "J03WQQ": {"meta": 51643484, "event": 51643508, "tracking": 51643529},
    "J03WOY": {"meta": 51643481, "event": 51643502, "tracking": 51643523},
    "J03WR9": {"meta": 51643490, "event": 51643511, "tracking": 51643532},
}
MATCHES = sorted(IDSSE)
TOKENS = {"meta": "matchinformation", "event": "events_raw", "tracking": "positions_raw_observed"}


# -- files ---------------------------------------------------------------------------------------

def path_of(match_id, kind):
    hits = glob.glob(os.path.join(EXT_DIR, "*{}*{}.xml".format(TOKENS[kind], match_id)))
    return hits[0] if hits else None


def download(match_id, kinds=("meta", "tracking")):
    """Fetch missing files through figshare's public API (the documented programmatic route)."""
    os.makedirs(EXT_DIR, exist_ok=True)
    for kind in kinds:
        if path_of(match_id, kind):
            continue
        url = "https://api.figshare.com/v2/file/download/{}".format(IDSSE[match_id][kind])
        request = urllib.request.Request(url, headers={"User-Agent": "mtah2027-possession"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            disposition = response.headers.get("Content-Disposition", "")
            match = re.search(r'filename="?([^";]+)"?', disposition)
            name = os.path.basename(match.group(1)) if match else "{}_{}.xml".format(
                TOKENS[kind], match_id)
            with open(os.path.join(EXT_DIR, name), "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)


# -- parsing -------------------------------------------------------------------------------------

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _teams(meta_path):
    """(home TeamId, guest TeamId) from the match information file."""
    text = open(meta_path, encoding="utf-8").read()
    roles = {}
    for team in re.finditer(r"<Team [^>]*>", text):
        attrs = dict(_ATTR.findall(team.group(0)))
        roles[attrs.get("Role")] = attrs["TeamId"]
    if "home" not in roles or "guest" not in roles:
        raise ValueError("no home/guest team in {}".format(meta_path))
    return roles["home"], roles["guest"]


def _parse_positions(path):
    """{(section, team, person): dict of arrays} and the pitch size, by a line scan."""
    sets, current, pitch = {}, None, (105.0, 68.0)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            s = line.lstrip()
            if s.startswith("<Frame "):
                a = dict(_ATTR.findall(s))
                current["N"].append(int(a["N"]))
                current["X"].append(float(a["X"]))
                current["Y"].append(float(a["Y"]))
                current["S"].append(float(a.get("S", "nan")))
                if "BallStatus" in a:
                    current["status"].append(int(a["BallStatus"]))
                    current["poss"].append(int(a.get("BallPossession", "0") or 0))
            elif s.startswith("<FrameSet "):
                a = dict(_ATTR.findall(s))
                key = (a["GameSection"], a["TeamId"], a["PersonId"])
                current = sets.setdefault(key, {"N": [], "X": [], "Y": [], "S": [],
                                                "status": [], "poss": []})
            elif s.startswith("<PitchSize "):
                a = dict(_ATTR.findall(s))
                pitch = (float(a["X"]), float(a["Y"]))
    return sets, pitch


def load(match_id, use_cache=True):
    """dict like atd.load() plus 'labels' (H/A/D/X per frame), 'halves', 'pitch', 'ball_true'."""
    cache = os.path.join(CACHE_DIR, "idsse_{}_v2.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        return {"players": z["players"], "ball": z["ball"], "team": z["team"],
                "labels": z["labels"], "halves": [tuple(h) for h in z["halves"].tolist()],
                "pitch": tuple(z["pitch"].tolist()), "home": "home", "away": "away",
                "notes": ["IDSSE {} (cached)".format(match_id)], "name": match_id}

    download(match_id)
    home_id, guest_id = _teams(path_of(match_id, "meta"))
    sets, (length, width) = _parse_positions(path_of(match_id, "tracking"))
    # Some matches also track the three referees; they are not players of either team.
    sets = {k: v for k, v in sets.items() if k[1] in (home_id, guest_id, "BALL")}

    sections = ["firstHalf", "secondHalf"]
    ball = {sec: next(v for (s, t, p), v in sets.items() if s == sec and t == "BALL")
            for sec in sections}
    first_n = {sec: min(ball[sec]["N"]) for sec in sections}
    sizes = {sec: max(ball[sec]["N"]) - first_n[sec] + 1 for sec in sections}
    offset = {"firstHalf": 0, "secondHalf": sizes["firstHalf"] + HALF_GAP}
    n = offset["secondHalf"] + sizes["secondHalf"]

    persons = sorted({p for (s, t, p) in sets if t != "BALL"})
    slot = {p: i for i, p in enumerate(persons)}
    team = np.zeros(len(persons), dtype=np.int8)
    players = np.full((n, len(persons), 3), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    labels = np.full(n, "X", dtype="<U1")

    for (sec, team_id, person), v in sets.items():
        idx = np.array(v["N"]) - first_n[sec] + offset[sec]
        x = (np.array(v["X"]) + length / 2.0) / length
        y = (np.array(v["Y"]) + width / 2.0) / width
        if team_id == "BALL":
            ball_xy[idx, 0], ball_xy[idx, 1] = x, y
            status, poss = np.array(v["status"]), np.array(v["poss"])
            labels[idx] = np.where(status == 0, "D", np.where(poss == 1, "H", "A"))
        else:
            k = slot[person]
            team[k] = 0 if team_id == home_id else 1
            players[idx, k, 0], players[idx, k, 1] = x, y
            players[idx, k, 2] = np.array(v["S"]) / 3.6          # km/h -> m/s

    halves = [(1, offset["firstHalf"], offset["firstHalf"] + sizes["firstHalf"] - 1),
              (2, offset["secondHalf"], offset["secondHalf"] + sizes["secondHalf"] - 1)]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, players=players, ball=ball_xy, team=team, labels=labels,
             halves=np.array(halves), pitch=np.array([length, width]))
    return load(match_id, use_cache=True)


# -- label definition ----------------------------------------------------------------------------

def harmonise(labels, halves, min_control_s):
    """Copy of DFL labels with the challenge's notion of CONTROLLED possession.

    DFL's BallPossession changes on contact; the challenge does not count "a simple touch without
    control" or "a deflection that does not intercept" (CHALLENGE.md). So an interior spell
    shorter than min_control_s, with the same opposing team on both sides inside one alive
    stretch, is given back to that team. Spells next to a stoppage are kept: after a restart
    there is always a clear possessor. Shortest first, so merges cascade correctly.
    """
    out = labels.copy()
    limit = int(round(min_control_s * FPS))
    if limit <= 0:
        return out
    for _, first, last in halves:
        window = out[first:last + 1]
        changed = True
        while changed:
            changed = False
            breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(window)]))
            segs = [(bounds[i], bounds[i + 1], window[bounds[i]]) for i in range(len(bounds) - 1)]
            best = None
            for i in range(1, len(segs) - 1):
                a, b, k = segs[i]
                before, after = segs[i - 1][2], segs[i + 1][2]
                if k in ("H", "A") and before == after and before in ("H", "A") \
                        and before != k and b - a < limit:
                    if best is None or b - a < best[1] - best[0]:
                        best = (a, b, before)
            if best is not None:
                window[best[0]:best[1]] = best[2]
                changed = True
    return out


# -- making it look like the automatic tracking --------------------------------------------------

def atd_run_lengths(trk_atd, halves):
    """(visible run lengths, missing run lengths) of the ATD ball inside the halves.

    Statistics of the provided tracking's *detection* process only -- how long the ball stays seen
    and how long it stays lost. No label is involved.
    """
    seen = np.isfinite(trk_atd["ball"][:, 0]) & np.isfinite(trk_atd["ball"][:, 1])
    vis, gap = [], []
    for _, first, last in halves:
        window = seen[first:last + 1]
        edges = np.diff(np.concatenate(([0], window.astype(np.int8), [0])))
        a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        vis += (b - a).tolist()
        window = ~window
        edges = np.diff(np.concatenate(([0], window.astype(np.int8), [0])))
        a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        gap += (b - a).tolist()
    return np.array(vis), np.array(gap)


def degrade(trk, run_lengths, seed, player_drop=(1 / 500.0, 1 / 40.0), ball_noise=0.25,
            player_noise=0.10, missing_alive=0.28, missing_dead=0.55):
    """Copy of an external match made to look like the ATD. Deterministic for a given seed.

    The ATD loses the ball more when it is dead (out of play, being retrieved) than in play.
    Measured on the provided tracking against Smart Tagging pseudo-labels only: 18.5 % missing in
    the most dynamic live phases, 34-36 % in the run-up to restarts, 40.7 % overall -- which a
    28 % (alive) / 55 % (dead) split reproduces. Lost stretches keep the ATD's run lengths,
    stretched or shrunk to hit each rate.
    """
    rng = np.random.RandomState(seed)
    vis_runs, gap_runs = run_lengths
    length, width = trk["pitch"]
    n = len(trk["ball"])
    ball = trk["ball"].astype(np.float64).copy()
    players = trk["players"].astype(np.float64).copy()

    def keep_mask(share):
        """Alternating seen / lost stretches whose long-run lost share is `share`."""
        stretch = share * vis_runs.mean() / (gap_runs.mean() * (1.0 - share))
        p_seen = 1.0 - share
        keep = np.zeros(n, dtype=bool)
        for _, first, last in trk["halves"]:
            t, seen = first, rng.rand() < p_seen
            while t <= last:
                span = int(rng.choice(vis_runs)) if seen \
                    else max(1, int(round(rng.choice(gap_runs) * stretch)))
                if seen:
                    keep[t:min(t + span, last + 1)] = True
                t += span
                seen = not seen
        return keep

    dead_true = trk["labels"] == "D"
    keep = np.where(dead_true, keep_mask(missing_dead), keep_mask(missing_alive))
    ball[~keep] = np.nan
    ball[:, 0] += rng.randn(n) * ball_noise / length
    ball[:, 1] += rng.randn(n) * ball_noise / width

    # Players: each drops out and comes back (a two-state chain), and jitters.
    leave, come_back = player_drop
    present = np.isfinite(players[:, :, 0])
    for k in range(players.shape[1]):
        draws = rng.rand(n)
        on, mask = True, np.ones(n, dtype=bool)
        flips = np.flatnonzero(draws < max(leave, come_back))
        # Walk only the candidate flip frames; cheap and exact for the two-state chain.
        last = 0
        for f in flips.tolist():
            p = leave if on else come_back
            if draws[f] < p:
                mask[last:f] = on
                on, last = not on, f
        mask[last:] = on
        players[~(mask & present[:, k]), k, :] = np.nan
    players[:, :, 0] += rng.randn(n, players.shape[1]) * player_noise / length
    players[:, :, 1] += rng.randn(n, players.shape[1]) * player_noise / width

    out = dict(trk)
    out["players"] = players.astype(np.float32)
    out["ball"] = ball.astype(np.float32)
    return out


def synth_tags(trk, seed, coverage=None):
    """Smart Tagging-like set-piece events at the true restarts of an external match.

    Mirrors the provided tagging: a window from 3.5 s before to 1.5 s after the restart, the
    restarting team named except on throw-ins and kick-offs after goals, and the provided file's
    coverage pattern by type (it has every corner but only a handful of free kicks).
    """
    coverage = coverage or {"throw-in": 0.6, "corner kick": 1.0, "goal kick": 0.6,
                            "free kick": 0.2, "kick off (after goal)": 1.0,
                            "kick off (start)": 1.0}
    rng = np.random.RandomState(seed)
    length, width = trk["pitch"]
    labels = trk["labels"]
    x = trk["ball"][:, 0].astype(float) * length
    y = trk["ball"][:, 1].astype(float) * width
    events = []
    for _, first, last in trk["halves"]:
        window = labels[first:last + 1]
        alive = (window == "H") | (window == "A")
        starts = np.flatnonzero(alive[1:] & ~alive[:-1]) + 1
        restarts = ([0] if alive[0] else []) + starts.tolist()
        for i, r in enumerate(restarts):
            f = first + r
            bx, by = x[f], y[f]
            if i == 0:
                kind = "kick off (start)"
            elif abs(bx - length / 2) < 3 and abs(by - width / 2) < 3:
                kind = "kick off (after goal)"
            elif (bx < 2.5 or bx > length - 2.5) and (by < 3.5 or by > width - 3.5):
                kind = "corner kick"
            elif by < 1.5 or by > width - 1.5:
                kind = "throw-in"
            elif (bx < 6.5 or bx > length - 6.5) and abs(by - width / 2) < 10.5:
                kind = "goal kick"
            else:
                kind = "free kick"
            if rng.rand() >= coverage.get(kind, 0.0):
                continue
            team = None if kind in ("throw-in", "kick off (after goal)") else (
                0 if labels[f] == "H" else 1)
            events.append({"code": "set pieces", "type": kind, "team": team,
                           "start": f - 88, "end": f + 37})
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events
