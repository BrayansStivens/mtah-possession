#!/usr/bin/env bash
# El comando unico que regenera el fichero entregado. El entregable lo exige:
# ejecutar este repo contra los datos tiene que producir EL MISMO fichero.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MTAH_DATA_DIR:?Apunta MTAH_DATA_DIR al bundle del partido (ver README)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

TEAM="${1:-submission}"
python -m mtah status
python -m mtah build --out "out/${TEAM}.csv"
python -m mtah validate "out/${TEAM}.csv"
