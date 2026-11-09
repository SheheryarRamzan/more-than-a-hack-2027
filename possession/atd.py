"""Read the automatic tracking (FIFA/EPTS export) into numpy arrays, cached on disk.

The raw file is one line per frame, "<frame_id>:<slot>;<slot>;...:<ball>", each slot "x,y,speed"
in normalised pitch units with NaN for a missing observation. Parsing 199k lines takes a few
seconds; the arrays are then cached under data/cache/, which is gitignored with the rest of data/
(the cache is derived tracking and must never be committed).

Slot -> team comes from the metadata's DataFormatSpecification (the order the PlayerChannelRefs
appear in), cross-checked against the order of the Player elements, which is what the starter
baseline assumes.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, require, source_frames  # noqa: E402

CACHE_VERSION = 1
HOME, AWAY = 0, 1


def _local(tag):
    """Tag name without an XML namespace."""
    return tag.rsplit("}", 1)[-1]


def read_metadata(meta_path):
    """(team_of_slot int8[S], home_name, away_name, notes).

    The first Team in the file is home (the FIFA/EPTS convention, and Barcelona per DATA.md).
    """
    root = ET.parse(meta_path).getroot()
    teams = [el for el in root.iter() if _local(el.tag) == "Team"]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in the ATD metadata, found {}".format(len(teams)))
    home_id = teams[0].get("id")
    names = [next((c.text for c in t if _local(c.tag) == "Name"), t.get("id")) for t in teams]

    players = [el for el in root.iter() if _local(el.tag) == "Player"]
    team_of_player = {p.get("id"): (HOME if p.get("teamId") == home_id else AWAY)
                      for p in players}
    order_by_player_elements = [p.get("id") for p in players]

    # Slot order as the raw file actually lays it out: the first DataFormatSpecification's
    # PlayerChannelRefs, mapped through PlayerChannels to player ids.
    channel_to_player = {el.get("id"): el.get("playerId")
                         for el in root.iter() if _local(el.tag) == "PlayerChannel"}
    order_by_spec = []
    spec = next((el for el in root.iter() if _local(el.tag) == "DataFormatSpecification"), None)
    if spec is not None:
        for el in spec.iter():
            if _local(el.tag) == "PlayerChannelRef":
                player = channel_to_player.get(el.get("playerChannelId"))
                if player is not None and player not in order_by_spec:
                    order_by_spec.append(player)

    notes = []
    if order_by_spec and set(order_by_spec) <= set(team_of_player):
        order = order_by_spec
        notes.append("slot order from DataFormatSpecification; matches Player order: {}".format(
            order_by_spec == order_by_player_elements))
    else:
        order = order_by_player_elements
        notes.append("slot order from Player elements (no usable DataFormatSpecification)")

    team = np.array([team_of_player[p] for p in order], dtype=np.int8)
    return team, names[0], names[1], notes


def _layout(raw_path):
    """(n_slots, n_ball) from the file itself.

    The slot count is the number of ';'-groups on a line. The ball field count is read from the
    first line where the ball is actually observed, because a missing ball may be written as a
    single NaN rather than one NaN per field.
    """
    n_slots, n_ball = None, None
    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            if n_slots is None:
                n_slots = parts[1].count(";") + 1
            if parts[2][:1] not in ("N", ""):
                n_ball = parts[2].count(",") + 1
                break
    if n_slots is None or n_ball is None:
        raise ValueError("could not read the ATD layout (no well-formed line with a ball)")
    return n_slots, n_ball


def _parse_raw(raw_path, n_frames):
    """(players float32[n, S, 3], ball float32[n, B], present bool[n], n_bad) by video frame."""
    n_slots, n_ball = _layout(raw_path)
    width = 1 + 3 * n_slots + n_ball

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, n_ball), np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    table = str.maketrans({":": ",", ";": ","})
    n_bad = 0

    def place(rows):
        nonlocal n_bad
        frame = rows[:, 0].astype(np.int64) - 1          # ATD counts from 1, video from 0
        ok = (frame >= 0) & (frame < n_frames)
        n_bad += int((~ok).sum())
        frame, rows = frame[ok], rows[ok]
        players[frame] = rows[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
        ball[frame] = rows[:, 1 + 3 * n_slots:]
        present[frame] = True

    def parse_line(line):
        """One line, group by group: tolerant of a missing group written as a bare NaN."""
        parts = line.rstrip("\r\n").split(":")
        if len(parts) != 3:
            return None
        groups = parts[1].split(";")
        if len(groups) != n_slots:
            return None
        row = np.full(width, np.nan)
        try:
            row[0] = float(parts[0])
            for i, group in enumerate(groups):
                if group[:1] in ("N", ""):
                    continue
                values = group.split(",")[:3]
                row[1 + 3 * i:1 + 3 * i + len(values)] = [float(v) for v in values]
            if parts[2][:1] not in ("N", ""):
                values = parts[2].split(",")[:n_ball]
                row[1 + 3 * n_slots:1 + 3 * n_slots + len(values)] = [float(v) for v in values]
        except ValueError:
            return None
        return row

    def flush(chunk):
        nonlocal n_bad
        text = ",".join(line.rstrip("\r\n").translate(table) for line in chunk)
        try:
            values = np.array(text.split(","), dtype=np.float64)
        except ValueError:
            values = np.empty(0)
        if values.size == len(chunk) * width:
            place(values.reshape(len(chunk), width))
            return
        # Field counts vary inside this chunk: fall back to the group-aware line parser.
        rows = []
        for line in chunk:
            row = parse_line(line)
            if row is None:
                n_bad += 1
            else:
                rows.append(row)
        if rows:
            place(np.vstack(rows))

    chunk = []
    with open(raw_path, "r") as handle:
        for line in handle:
            if line.strip():
                chunk.append(line)
            if len(chunk) == 20000:
                flush(chunk)
                chunk = []
    if chunk:
        flush(chunk)
    return players, ball, present, n_bad


def load(source="tactical", use_cache=True, verbose=True):
    """Tracking arrays for one source, as a dict. Coordinates stay normalised (0-1)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None, source)
    n_frames = source_frames(source)
    team, home, away, notes = read_metadata(meta_path)

    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path)), CACHE_VERSION])
    cache = os.path.join(DATA_DIR, "cache", "{}_{}_atd_v{}.npz".format(GAME, source, CACHE_VERSION))
    if use_cache and os.path.isfile(cache):
        stored = np.load(cache)
        if np.array_equal(stored["stamp"], stamp):
            if verbose:
                print("tracking: loaded cache {}".format(os.path.relpath(cache, ROOT)))
            return {"players": stored["players"], "ball": stored["ball"],
                    "present": stored["present"], "team": team, "home": home, "away": away,
                    "notes": notes}

    if verbose:
        print("tracking: parsing {} ...".format(os.path.basename(raw_path)))
    players, ball, present, n_bad = _parse_raw(raw_path, n_frames)
    if players.shape[1] != len(team):
        raise ValueError("raw file has {} slots but metadata names {}".format(
            players.shape[1], len(team)))
    notes.append("frames present {} / {}, malformed lines skipped {}".format(
        int(present.sum()), n_frames, n_bad))
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, players=players, ball=ball, present=present, stamp=stamp)
    return {"players": players, "ball": ball, "present": present, "team": team,
            "home": home, "away": away, "notes": notes}
