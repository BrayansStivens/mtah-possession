#!/usr/bin/env bash
# The single command that regenerates the submitted file. The submission requires it:
# running this repo against the data has to produce THE SAME file.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MTAH_DATA_DIR:?Point MTAH_DATA_DIR at the match bundle (see README)}"

# In Git Bash (Windows) Python is the Windows one and does not understand POSIX paths
# (/d/...) or the ':' separator; cygpath exists there and not on Linux or macOS.
if command -v cygpath >/dev/null 2>&1; then
    export PYTHONPATH="$(cygpath -w "$PWD/src")${PYTHONPATH:+;$PYTHONPATH}"
else
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# numpy 2.5.4 requires Python >= 3.12; with any other it fails later and worse. PYTHON picks
# the interpreter.
PY="${PYTHON:-python}"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || {
    echo "Python >= 3.12 is required ($("$PY" --version 2>&1)); activate the venv or set PYTHON=..." >&2
    exit 1
}

TEAM="${1:-submission}"
"$PY" -m mtah status
"$PY" -m mtah build --out "out/${TEAM}.csv"
"$PY" -m mtah validate "out/${TEAM}.csv"
