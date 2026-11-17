# Load this before working:  source scripts/env.sh
# The match assets live outside the repo: they are ~5 GB and not redistributable.
export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/Documents/Hackaton/data}"
_mtah_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
# In Git Bash (Windows) Python is the Windows one: Windows-format path and ';' separator.
if command -v cygpath >/dev/null 2>&1; then
    export PYTHONPATH="$(cygpath -w "$_mtah_src")${PYTHONPATH:+;$PYTHONPATH}"
else
    export PYTHONPATH="$_mtah_src${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _mtah_src
echo "MTAH_DATA_DIR=$MTAH_DATA_DIR"
