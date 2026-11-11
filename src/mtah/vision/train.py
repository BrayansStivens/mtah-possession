"""Entrena BallNet con los recortes de dataset.build().

    python -m mtah.vision.train [--epochs N] [--sigma S] [--labels refined]

El objetivo es una gaussiana de SIGMA px en la posicion etiquetada. La etiqueta del ATD
tiene ~10 px de ruido; --labels refined usa las posiciones refinadas por una pasada
anterior del propio modelo (autoentrenamiento, ver refine.py).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from . import dataset
from .net import OUTPUT_STRIDE, BallNet, focal_heatmap_loss, to_output

SIGMA = 4.0
BATCH = 128
LEARNING_RATE = 2e-3


def heatmaps(target, has_ball, size, sigma):
    """(N, size, size) gaussianas con pico exacto 1 en el pixel redondeado; 0 sin balon."""
    grid = torch.arange(size, dtype=torch.float32, device=target.device)
    peak = target.round()
    dx = grid[None, None, :] - peak[:, 0, None, None]
    dy = grid[None, :, None] - peak[:, 1, None, None]
    out = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    return out * has_ball[:, None, None].float()


def output_targets(target):
    """(etiquetas en la rejilla de salida, hay balon). En pixel de imagen, -1 es 'sin balon'."""
    return to_output(target), target[:, 0] >= 0


def augment(batch, target, generator):
    """Volteo horizontal y brillo/contraste. Nada de volteo vertical: la gravedad existe."""
    flip = torch.rand(len(batch), generator=generator, device="cpu") < 0.5
    flip = flip.to(batch.device)
    batch = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
    size = batch.shape[-1]
    target = target.clone()
    flipped_x = size - 1 - target[:, 0]
    target[:, 0] = torch.where(flip & (target[:, 0] >= 0), flipped_x, target[:, 0])
    gain = 1 + 0.2 * (torch.rand(len(batch), 1, 1, 1, generator=generator) - 0.5)
    bias = 20 * (torch.rand(len(batch), 1, 1, 1, generator=generator) - 0.5)
    batch = (batch * gain.to(batch.device) + bias.to(batch.device)).clamp(0, 255)
    return batch, target


def model_path(name="ballnet"):
    return os.path.join(dataset.vision_dir(), name + ".pt")


def _load_patches(directory, index, hard):
    """Recortes uint8 (patches[index] y, con hard, los negativos duros) en un unico array.

    Se copia por trozos desde los ficheros mapeados y se cierran: ni dos copias enteras a
    la vez ni las paginas del mapeo contando como memoria del proceso (con ~8 GB el
    sistema mato dos entrenamientos el 9-11).
    """
    patches = np.load(os.path.join(directory, "patches.npy"), mmap_mode="r")
    sources = [(patches, index)]
    if hard:
        hard_meta = np.load(os.path.join(directory, "hard_meta.npz"))
        hard_patches = np.load(os.path.join(directory, "hard_patches.npy"), mmap_mode="r")
        sources.append((hard_patches, np.arange(int(hard_meta["count"]))))
    total = sum(len(rows) for _, rows in sources)
    data = np.empty((total,) + patches.shape[1:], dtype=np.uint8)
    cursor = 0
    for source, rows in sources:
        for start in range(0, len(rows), 4096):
            chunk = rows[start:start + 4096]
            data[cursor:cursor + len(chunk)] = source[chunk]
            cursor += len(chunk)
        source._mmap.close()
    return data


def train(epochs=12, sigma=SIGMA, labels="atd", name="ballnet", seed=0, hard=False,
          init=None):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = dataset.vision_dir()
    meta = np.load(os.path.join(directory, "meta.npz"))
    target = meta["target"]
    if labels == "refined":
        refined = np.load(os.path.join(directory, "refined_targets.npy"))
        keep = ~np.isnan(refined[:, 0])
        target = np.where(keep[:, None], refined, target)
        # positivos que el refinamiento no confirma: fuera (ni positivo ni negativo)
        usable = keep | (meta["kind"] != dataset.KINDS["ball"])
    else:
        usable = np.ones(len(target), dtype=bool)
    index = np.flatnonzero(usable)
    target = target[index]
    if hard:
        hard_meta = np.load(os.path.join(directory, "hard_meta.npz"))
        target = np.concatenate([target, hard_meta["target"]])
    print("cargando %d recortes en memoria..." % len(target), flush=True)
    data = torch.from_numpy(_load_patches(directory, index, hard))
    target = torch.from_numpy(target.astype(np.float32))
    index = np.arange(len(target))

    torch.backends.cudnn.benchmark = True
    model = BallNet().to(device).to(memory_format=torch.channels_last)
    if init:
        model.load_state_dict(torch.load(model_path(init), map_location=device)["state"])
        print("partiendo de", model_path(init), flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    steps = epochs * (len(index) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, LEARNING_RATE, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    first_epoch = 0
    resume_path = model_path(name) + ".partial"
    if os.path.exists(resume_path):
        # Un pase cortado (p. ej. por falta de memoria) sigue en la epoca siguiente.
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["state"])
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"].cpu())     # map_location lo mueve a la GPU
        history, first_epoch = state["history"], state["epoch"]
        print("reanudando tras la epoca %d" % first_epoch, flush=True)
    started = time.time()
    for epoch in range(first_epoch, epochs):
        order = torch.randperm(len(index), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order) - BATCH + 1, BATCH):
            chosen = order[start:start + BATCH]
            batch = data[chosen].to(device, non_blocking=True).float()
            batch, batch_target = augment(batch, target[chosen].to(device), generator)
            goal = heatmaps(*output_targets(batch_target),
                            batch.shape[-1] // OUTPUT_STRIDE, sigma / OUTPUT_STRIDE)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(batch.contiguous(memory_format=torch.channels_last))
            loss = focal_heatmap_loss(logits.float(), goal)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        print("epoca %d: perdida %.4f (%.1f min)" % (epoch + 1, history[-1],
                                                    (time.time() - started) / 60), flush=True)
        torch.save({"state": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "schedule": schedule.state_dict(), "scaler": scaler.state_dict(),
                    "generator": generator.get_state(), "history": history,
                    "epoch": epoch + 1}, resume_path)
    torch.save({"state": model.state_dict(), "sigma": sigma, "labels": labels,
                "epochs": epochs, "history": history, "hard": hard, "init": init},
               model_path(name))
    os.remove(resume_path)
    print("guardado", model_path(name))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--labels", choices=("atd", "refined"), default="atd")
    parser.add_argument("--name", default="ballnet")
    parser.add_argument("--hard", action="store_true", help="anade los negativos duros")
    parser.add_argument("--init", default=None, help="modelo del que partir")
    args = parser.parse_args()
    train(args.epochs, args.sigma, args.labels, args.name, hard=args.hard, init=args.init)
