"""Refinado de las etiquetas positivas con el propio modelo (autoentrenamiento del plan).

    python -m mtah.vision.refine [--model ballnet_h1]

La etiqueta gratis es el balon del ATD proyectado al suelo: ~12 px de ruido de mediana, y
en una parte de los frames no hay balon donde dice (balon en el aire, tapado, o el
tracker siguiendo otra cosa). Esos positivos ensenan a la red que el cesped o una marca
del campo son un balon. Aqui, en cada recorte positivo, se busca el pico del modelo a
menos de SEARCH_PX de la etiqueta: si es claro, pasa a ser la etiqueta; si no, el recorte
se descarta (ni positivo ni negativo).

Salida: <vision_dir>/refined_targets.npy (N, 2), NaN donde no hay etiqueta refinada (y en
todos los recortes que no son positivos); la lee train.py --labels refined.
"""

import argparse
import os

import numpy as np
import torch

from . import dataset
from .detect import load_model
from .net import OUTPUT_STRIDE, to_image, to_output

SEARCH_PX = 24
MIN_PROBABILITY = 0.15
BATCH = 256


@torch.no_grad()
def refine(name="ballnet_h1"):
    directory = dataset.vision_dir()
    meta = np.load(os.path.join(directory, "meta.npz"))
    patches = np.load(os.path.join(directory, "patches.npy"), mmap_mode="r")
    model = load_model(name)
    device = next(model.parameters()).device
    positive = np.flatnonzero(meta["kind"] == dataset.KINDS["ball"])
    refined = np.full((len(meta["kind"]), 2), np.nan, dtype=np.float32)
    size = dataset.PATCH // OUTPUT_STRIDE
    grid = np.arange(size)
    peak_probability = []
    for start in range(0, len(positive), BATCH):
        rows = positive[start:start + BATCH]
        batch = torch.from_numpy(np.ascontiguousarray(patches[rows])).to(device).float()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(batch.contiguous(
                memory_format=torch.channels_last)).float()).cpu().numpy()
        label = to_output(meta["target"][rows].astype(np.float64))
        radius = SEARCH_PX / OUTPUT_STRIDE
        inside = ((grid[None, None, :] - label[:, 0, None, None]) ** 2
                  + (grid[None, :, None] - label[:, 1, None, None]) ** 2) <= radius ** 2
        masked = np.where(inside, probability, 0.0).reshape(len(rows), -1)
        best = masked.argmax(1)
        score = masked[np.arange(len(rows)), best]
        peak_probability.append(score)
        keep = score >= MIN_PROBABILITY
        cells = np.stack([best % size, best // size], 1).astype(np.float64)
        refined[rows[keep]] = to_image(cells[keep])
    patches._mmap.close()
    peak_probability = np.concatenate(peak_probability)
    kept = np.isfinite(refined[positive, 0])
    shift = np.hypot(*(refined[positive][kept] - meta["target"][positive][kept]).T)
    print("positivos %d: se quedan %.1f%% (pico >= %.2f a <%d px); desplazamiento p50 %.1f px, "
          "p90 %.1f px" % (len(positive), 100 * kept.mean(), MIN_PROBABILITY, SEARCH_PX,
                          np.median(shift), np.percentile(shift, 90)))
    np.save(os.path.join(directory, "refined_targets.npy"), refined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ballnet_h1")
    refine(parser.parse_args().model)
