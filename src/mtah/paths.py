"""Resolucion de rutas a los assets del partido.

El bundle se organiza por grabacion, igual que lo entregan los organizadores:

    <raiz>/tactical/BAR-ATH_tactical_atd.txt
    <raiz>/broadcast/BAR-ATH_broadcast_atd.txt

La raiz se toma de MTAH_DATA_DIR si existe; si no, de ./data en la raiz del repo.
Nada de este repo escribe un nombre de fichero a mano fuera de este modulo.
"""

import os

from .constants import GAME

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}

ASSETS = {
    "tactical": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    "broadcast": ("video", "atd_meta", "atd_raw", "homography"),
}


def data_dir():
    return os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO_ROOT, "data")


def cache_dir():
    path = os.path.join(REPO_ROOT, "out", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def asset_paths(source="tactical", game=GAME, root=None):
    """Dict asset -> ruta. No comprueba existencia; para eso esta require()."""
    if source not in ASSETS:
        raise KeyError("fuente desconocida {!r}; hay {}".format(source, sorted(ASSETS)))
    folder = os.path.join(root or data_dir(), source)
    stem = "{}_{}".format(game, source)
    return {name: os.path.join(folder, stem + SUFFIXES[name]) for name in ASSETS[source]}


def require(names, source="tactical", game=GAME, root=None):
    """Devuelve las rutas pedidas, o falla con un mensaje que dice que descargar."""
    paths = asset_paths(source, game, root)
    missing = [(n, paths[n]) for n in names if not os.path.exists(paths[n])]
    if missing:
        detail = "\n".join("  falta {:<14} -> {}".format(n, p) for n, p in missing)
        raise FileNotFoundError(
            "Assets del partido no encontrados en {}\n{}\n"
            "Descargalos del Drive de la competicion y descomprime manteniendo la\n"
            "estructura <raiz>/tactical/..., o apunta MTAH_DATA_DIR a donde los tengas."
            .format(data_dir(), detail))
    return [paths[n] for n in names]


def available(source="tactical", game=GAME, root=None):
    """Dict asset -> bool, para saber con que se puede trabajar ya."""
    paths = asset_paths(source, game, root)
    return {name: os.path.exists(path) for name, path in paths.items()}
