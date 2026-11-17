"""Prepares the new signals of each degraded match. Usage: build.py [seed ...]"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import comun as C                                              # noqa: E402
import feats as F                                              # noqa: E402
from mtah import features as feature_set                       # noqa: E402
from mtah.config import DEFAULT                                # noqa: E402
from mtah.paths import cache_dir                               # noqa: E402


def path(match, seed):
    return os.path.join(cache_dir(), "sinbalon_feat_%s_s%d.npz" % (match, seed))


def build(args):
    match, seed = args
    started = time.time()
    clean, dirty, labels = C.dirty(match, seed)
    features = feature_set.build_all(dirty, DEFAULT)
    direction, report = F.attack_direction(dirty)
    clean_direction, clean_report = F.attack_direction(clean)
    smoothed = F.team_smoothed(dirty, direction)
    team = np.stack([[[smoothed[(t, s, w)] for w in F.TEAM_WINDOWS] for s in F.TEAM_SIGNALS]
                     for t in (0, 1)]).astype(np.float32)          # (2, signals, windows, n)
    dead, dead_names = F.dead_matrix(dirty, features, DEFAULT)
    context = F.gap_context(dirty, features)
    ballteam, ballteam_names, odd = F.ballteam_matrix(features, context, DEFAULT)
    np.savez(path(match, seed), team=team, dead=dead, ballteam=ballteam, odd=odd,
             direction=direction.astype(np.int8), labels=labels,
             ball_seen=dirty.ball_seen)
    info = {"match": match, "seed": seed, "seconds": round(time.time() - started, 1),
            "direction_dirty": report, "direction_clean": clean_report,
            "direction_same_as_clean": float((direction == clean_direction).mean()),
            "dead_names": dead_names, "ballteam_names": ballteam_names}
    return info


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    jobs = [(m, s) for s in seeds for m in C.MATCHES if not os.path.exists(path(m, s))]
    with Pool(7) as pool:
        for info in pool.imap_unordered(build, jobs):
            print("%s s%d  %.0fs  direction %s  same as clean %.4f" % (
                info["match"], info["seed"], info["seconds"],
                [(r["block"], r["direction"], round(r["agreement"], 3))
                 for r in info["direction_dirty"]], info["direction_same_as_clean"]),
                flush=True)
            with open(os.path.join(os.path.dirname(__file__), "build_%s_s%d.json"
                                   % (info["match"], info["seed"])), "w") as handle:
                json.dump(info, handle, indent=1)
