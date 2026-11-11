"""Ejecuta BallNet sobre frames enteros y guarda los TOP_K picos de cada uno.

    python -m mtah.vision.detect holdout [--model ballnet]   validacion (bloques apartados)
    python -m mtah.vision.detect fill [--model ballnet]      lo que necesita el enlazado
    python -m mtah.vision.detect all [--model ballnet]       todo el partido (~1-2 h)

Salida: <vision_dir>/peaks_<conjunto>_<modelo>.npz con
    frame  (n,)        indice de frame de video
    peaks  (n, K, 3)   x, y en pixel y probabilidad, ordenados por probabilidad
"""

import argparse
import os
import queue
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..constants import N_FRAMES, graded_mask
from ..io import atd, homography
from . import dataset, frames
from .net import OUTPUT_STRIDE, BallNet, to_image
from .train import model_path

TOP_K = 5
PEAK_RADIUS = 12                 # supresion de no-maximos: picos a mas de esto entre si
BATCH = 4
HOLDOUT_STRIDE = 5
GAP_EDGE_FRAMES = 10
PARTS = 8
ATTEMPTS = 3


def load_model(name="ballnet", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path(name), map_location=device)
    torch.backends.cudnn.benchmark = True
    model = BallNet().to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model.to(memory_format=torch.channels_last)


@torch.no_grad()
def peaks(model, stacks):
    """(B, CHANNELS, H, W) uint8 -> (B, TOP_K, 3) numpy con x, y, probabilidad."""
    device = next(model.parameters()).device
    batch = torch.from_numpy(stacks).to(device).float()
    batch = batch.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        probability = torch.sigmoid(model(batch).float())
    radius = PEAK_RADIUS // OUTPUT_STRIDE
    pooled = F.max_pool2d(probability[:, None], 2 * radius + 1, stride=1,
                          padding=radius)[:, 0]
    probability = torch.where(probability == pooled, probability,
                              torch.zeros_like(probability))
    flat = probability.flatten(1)
    score, where = flat.topk(TOP_K, dim=1)
    width = probability.shape[-1]
    out = torch.stack([to_image((where % width).float()), to_image((where // width).float()),
                       score], -1)
    return out.cpu().numpy()


def _prepared_stacks(wanted, H, out_queue):
    """Hilo productor: decodifica y apila mientras la GPU trabaja (OpenCV suelta el GIL)."""
    try:
        for t, frames_t in frames.neighbourhoods(wanted):
            out_queue.put((t, frames.stack(H, t, frames_t)))
    except BaseException as error:              # que el consumidor se entere y falle
        out_queue.put(error)
        return
    out_queue.put(None)


def run(wanted, name="ballnet", label="all", parts=PARTS, attempts=ATTEMPTS):
    """Por tramos de frames: cada tramo se guarda al acabar y uno que falla al leer el
    video (el video vive en Google Drive y la lectura en streaming se ha cortado a mitad,
    'partial file') se reintenta. Volver a lanzar el comando salta los tramos ya hechos."""
    started = time.time()
    model = load_model(name)
    H = homography.load()[0]
    directory = dataset.vision_dir()
    wanted = np.unique(np.asarray(wanted, dtype=np.int64))
    chunks = np.array_split(wanted, parts)
    part_paths = []
    for i, chunk in enumerate(chunks):
        path = os.path.join(directory, "peaks_%s_%s.part%d.npz" % (label, name, i))
        part_paths.append(path)
        if os.path.exists(path):
            print("tramo %d/%d ya hecho" % (i + 1, parts), flush=True)
            continue
        for attempt in range(1, attempts + 1):
            try:
                found_frames, found_peaks = _run_chunk(model, H, chunk)
                break
            except RuntimeError as error:
                print("tramo %d/%d, intento %d fallo: %s (%s)" % (
                    i + 1, parts, attempt, error, error.__cause__), flush=True)
                if attempt == attempts:
                    raise
                time.sleep(30)
        np.savez(path, frame=found_frames, peaks=found_peaks)
        print("tramo %d/%d guardado (%d frames, %.1f min)" % (
            i + 1, parts, len(found_frames), (time.time() - started) / 60), flush=True)
    blobs = [np.load(p) for p in part_paths]
    path = os.path.join(directory, "peaks_%s_%s.npz" % (label, name))
    found_frames = np.concatenate([b["frame"] for b in blobs])
    np.savez(path, frame=found_frames, peaks=np.concatenate([b["peaks"] for b in blobs]))
    for blob, part in zip(blobs, part_paths):
        blob.close()
        os.remove(part)
    print("guardado %s (%d frames, %.1f min)" % (path, len(found_frames),
                                                 (time.time() - started) / 60))
    return path


def _run_chunk(model, H, wanted):
    started = time.time()
    found_frames, found_peaks, pending, pending_frames = [], [], [], []
    prepared = queue.Queue(maxsize=4 * BATCH)
    producer = threading.Thread(target=_prepared_stacks, args=(wanted, H, prepared),
                                daemon=True)
    producer.start()

    def flush():
        found_peaks.append(peaks(model, np.stack(pending)))
        found_frames.extend(pending_frames)
        pending.clear()
        pending_frames.clear()

    done = 0
    while True:
        item = prepared.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise RuntimeError("fallo preparando frames") from item
        t, stacked = item
        pending.append(stacked)
        pending_frames.append(t)
        if len(pending) == BATCH:
            flush()
        if done % 2000 == 0:
            print("  %d/%d frames (%.1f min)" % (done, len(wanted),
                                                 (time.time() - started) / 60), flush=True)
        done += 1
    if pending:
        flush()
    producer.join()
    return (np.array(found_frames, dtype=np.int32),
            np.concatenate(found_peaks).astype(np.float32))


def holdout_frames():
    """Frames de validacion: dentro de los bloques apartados, uno de cada HOLDOUT_STRIDE.

    Incluye frames con y sin balon del ATD: los primeros se puntuan, los segundos dicen
    cuanto recupera el detector donde el tracker falla.
    """
    return np.flatnonzero(dataset.holdout_mask())[::HOLDOUT_STRIDE]


def fill_frames(margin=GAP_EDGE_FRAMES):
    """Todos los frames de los bloques apartados (para puntuar trayectorias contra el ATD)
    mas los frames puntuados sin balon del ATD y unos pocos alrededor de cada hueco."""
    tracking = atd.load()
    missing = ~tracking.ball_seen & graded_mask(tracking.n_frames)
    grown = np.convolve(missing, np.ones(2 * margin + 1), mode="same") > 0
    return np.flatnonzero(grown | dataset.holdout_mask(tracking.n_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("holdout", "fill", "all"))
    parser.add_argument("--model", default="ballnet")
    args = parser.parse_args()
    wanted = {"holdout": holdout_frames, "fill": fill_frames,
              "all": lambda: np.arange(N_FRAMES)}[args.which]()
    run(wanted, args.model, args.which)
