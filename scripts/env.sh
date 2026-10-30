# Carga esto antes de trabajar:  source scripts/env.sh
# Los assets del partido viven fuera del repo: son ~5 GB y no son redistribuibles.
export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/Documents/Hackaton/data}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src:$PYTHONPATH"
echo "MTAH_DATA_DIR=$MTAH_DATA_DIR"
