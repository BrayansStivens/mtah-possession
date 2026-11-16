#!/usr/bin/env bash
# El comando unico que regenera el fichero entregado. El entregable lo exige:
# ejecutar este repo contra los datos tiene que producir EL MISMO fichero.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MTAH_DATA_DIR:?Apunta MTAH_DATA_DIR al bundle del partido (ver README)}"

# En Git Bash (Windows) el Python es de Windows y no entiende rutas POSIX (/d/...) ni el
# separador ':'; cygpath existe ahi y no en Linux ni en macOS.
if command -v cygpath >/dev/null 2>&1; then
    export PYTHONPATH="$(cygpath -w "$PWD/src")${PYTHONPATH:+;$PYTHONPATH}"
else
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# numpy 2.5.4 pide Python >= 3.12; con otro falla mas tarde y peor. PYTHON elige el interprete.
PY="${PYTHON:-python}"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || {
    echo "hace falta Python >= 3.12 ($("$PY" --version 2>&1)); activa el venv o pon PYTHON=..." >&2
    exit 1
}

TEAM="${1:-submission}"
"$PY" -m mtah status
"$PY" -m mtah build --out "out/${TEAM}.csv"
"$PY" -m mtah validate "out/${TEAM}.csv"
