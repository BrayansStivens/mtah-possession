# Carga esto antes de trabajar:  source scripts/env.sh
# Los assets del partido viven fuera del repo: son ~5 GB y no son redistribuibles.
export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/Documents/Hackaton/data}"
_mtah_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
# En Git Bash (Windows) el Python es de Windows: ruta en formato Windows y separador ';'.
if command -v cygpath >/dev/null 2>&1; then
    export PYTHONPATH="$(cygpath -w "$_mtah_src")${PYTHONPATH:+;$PYTHONPATH}"
else
    export PYTHONPATH="$_mtah_src${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _mtah_src
echo "MTAH_DATA_DIR=$MTAH_DATA_DIR"
