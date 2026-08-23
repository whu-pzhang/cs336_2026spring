from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    assert x.shape[0] > context_length, "The input sequence is too short for the given context length."
    valid_range = x.shape[0] - context_length
    start_indices = np.random.randint(0, valid_range, size=batch_size)

    x_batch = np.stack([x[i : i + context_length] for i in start_indices])
    y_batch = np.stack([x[i + 1 : i + context_length + 1] for i in start_indices])
    # indices = start_indices[:, None] + np.arange(context_length)
    # x_batch = x[indices]
    # y_batch = x[indices + 1]

    x_batch = torch.from_numpy(x_batch).to(dtype=torch.long, device=device)
    y_batch = torch.from_numpy(y_batch).to(dtype=torch.long, device=device)

    return x_batch, y_batch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | Path | BinaryIO,
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | Path | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration
