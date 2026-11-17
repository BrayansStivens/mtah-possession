"""Command-line interface.

    python -m mtah status            which assets are present and which are missing
    python -m mtah build             regenerates the submission in out/
    python -m mtah evaluate <csv>    scores with the official scorer (public window)
    python -m mtah validate <csv>    checks the file against the contract before uploading it
"""

import argparse
import os
import sys

import numpy as np


def cmd_status(args):
    from .io import atd
    from .paths import asset_paths, available, data_dir

    print("MTAH_DATA_DIR -> {}".format(data_dir()))
    for source in ("tactical", "broadcast"):
        print("\n[{}]".format(source))
        paths = asset_paths(source)
        for name, present in available(source).items():
            size = ""
            if present:
                bytes_ = os.path.getsize(paths[name])
                size = " ({:.1f} MB)".format(bytes_ / 1e6) if bytes_ > 1e6 \
                    else " ({:.0f} KB)".format(bytes_ / 1e3)
            print("  {} {:<14}{}".format("ok  " if present else "MISSING", name, size))

    if available("tactical")["atd_meta"]:
        team_of_slot, home, away = atd.read_metadata(asset_paths("tactical")["atd_meta"])
        print("\nteams: H = {}  |  A = {}".format(home, away))
        print("track slots: {} ({} home / {} away)".format(
            len(team_of_slot), int((team_of_slot == 0).sum()), int((team_of_slot == 1).sum())))
    return 0


def cmd_build(args):
    from . import pipeline, submission
    from .config import merged

    labels = pipeline.run(config=merged(), use_cache=not args.no_cache)
    path = submission.write(args.out, labels)
    print("written {}".format(path))
    print("breakdown: {}".format(submission.counts(labels)))

    reason = submission.validate(path)
    if reason:
        print("INVALID: {}".format(reason), file=sys.stderr)
        return 1
    print("contract: ok")

    from . import evaluate
    evaluate.report(labels, name=os.path.basename(path))
    return 0


def cmd_evaluate(args):
    import pandas as pd

    from . import evaluate
    labels = pd.read_csv(args.csv)["label"].to_numpy(dtype="<U1")
    evaluate.report(labels, name=os.path.basename(args.csv), bootstrap=args.bootstrap)
    if args.bootstrap:
        print("   (bootstrap interval over 17 possessions: indicative, not a ranking)")
    return 0


def cmd_validate(args):
    from . import submission
    reason = submission.validate(args.csv)
    if reason:
        print("INVALID: {}".format(reason), file=sys.stderr)
        return 1
    print("{}: contract ok".format(args.csv))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mtah", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="which assets are downloaded").set_defaults(func=cmd_status)

    build = sub.add_parser("build", help="regenerate the submission")
    build.add_argument("--out", default=os.path.join("out", "submission.csv"))
    build.add_argument("--no-cache", action="store_true",
                       help="re-parse the tracking instead of using out/cache")
    build.set_defaults(func=cmd_build)

    ev = sub.add_parser("evaluate", help="score a csv with the official scorer")
    ev.add_argument("csv")
    ev.add_argument("--bootstrap", type=int, default=0)
    ev.set_defaults(func=cmd_evaluate)

    va = sub.add_parser("validate", help="check the file against the contract")
    va.add_argument("csv")
    va.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)
