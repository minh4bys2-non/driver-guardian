"""Versioned, self-contained checkpoints; replace files only after a complete write."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .model import DrowsyClassifier


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint: model_config and training config are required; legacy weights need explicit migration")
    required = {"model", "model_config", "config", "epoch"}
    if not required <= state.keys():
        raise ValueError(f"Checkpoint missing fields: {required - state.keys()}")
    return state


def load_model(path, device="cpu"):
    state = load_checkpoint(path)
    model = DrowsyClassifier(**state["model_config"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state
