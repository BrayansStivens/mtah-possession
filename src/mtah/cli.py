"""Interfaz de linea de comandos.

    python -m mtah status            que assets hay y que falta
    python -m mtah build             regenera la entrega en out/
    python -m mtah evaluate <csv>    puntua con el scorer oficial (ventana publica)
    python -m mtah validate <csv>    comprueba el contrato del fichero antes de subirlo
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
            print("  {} {:<14}{}".format("ok  " if present else "FALTA", name, size))

    if available("tactical")["atd_meta"]:
        team_of_slot, home, away = atd.read_metadata(asset_paths("tactical")["atd_meta"])
        print("\nequipos: H = {}  |  A = {}".format(home, away))
        print("track slots: {} ({} local / {} visitante)".format(
            len(team_of_slot), int((team_of_slot == 0).sum()), int((team_of_slot == 1).sum())))
    return 0


def cmd_build(args):
    from . import pipeline, submission
    from .config import merged

    labels = pipeline.run(config=merged(), use_cache=not args.no_cache)
    path = submission.write(args.out, labels)
    print("escrito {}".format(path))
    print("reparto: {}".format(submission.counts(labels)))

    reason = submission.validate(path)
    if reason:
        print("INVALIDO: {}".format(reason), file=sys.stderr)
        return 1
    print("contrato: ok")

    from . import evaluate
    evaluate.report(labels, name=os.path.basename(path))
    return 0


def cmd_evaluate(args):
    import pandas as pd

    from . import evaluate
    labels = pd.read_csv(args.csv)["label"].to_numpy(dtype="<U1")
    evaluate.report(labels, name=os.path.basename(args.csv), bootstrap=args.bootstrap)
    if args.bootstrap:
        print("   (intervalo bootstrap sobre 17 posesiones: orientativo, no un ranking)")
    return 0


def cmd_validate(args):
    from . import submission
    reason = submission.validate(args.csv)
    if reason:
        print("INVALIDO: {}".format(reason), file=sys.stderr)
        return 1
    print("{}: contrato ok".format(args.csv))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mtah", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="que assets hay descargados").set_defaults(func=cmd_status)

    build = sub.add_parser("build", help="regenera la entrega")
    build.add_argument("--out", default=os.path.join("out", "submission.csv"))
    build.add_argument("--no-cache", action="store_true",
                       help="reparsea el tracking en lugar de usar out/cache")
    build.set_defaults(func=cmd_build)

    ev = sub.add_parser("evaluate", help="puntua un csv con el scorer oficial")
    ev.add_argument("csv")
    ev.add_argument("--bootstrap", type=int, default=0)
    ev.set_defaults(func=cmd_evaluate)

    va = sub.add_parser("validate", help="comprueba el contrato del fichero")
    va.add_argument("csv")
    va.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)
