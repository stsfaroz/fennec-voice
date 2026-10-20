"""Known Fennec models. The package version and the model versions are separate:
a new model is a new entry here, and "latest" moves to it only when it is the better default."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    task: str           # "interrupt" or "handoff"
    input_format: str   # how the model input is built, see interrupt.py / handoff.py
    description: str


MODELS = {m.name: m for m in (
    ModelInfo("fennec-interrupt-150m-v1", "SalmanFaroz/fennec-interrupt-150m-v1", "interrupt", "fi-2.0",
              "The agent is talking and the caller makes a sound: nod or interruption? English."),
    ModelInfo("fennec-handoff-150m-v1", "SalmanFaroz/fennec-handoff-150m-v1", "handoff", "plain-1",
              "The agent is quiet and the caller paused: finished or not? English."),
)}

LATEST = {"interrupt": "fennec-interrupt-150m-v1", "handoff": "fennec-handoff-150m-v1"}


def list_models(task: str | None = None) -> list[ModelInfo]:
    """All known models, optionally only one task ("interrupt" or "handoff")."""
    return [m for m in MODELS.values() if task in (None, m.task)]


def resolve(model: str | None, task: str) -> tuple[str, ModelInfo | None]:
    """model: None/"latest", a name from MODELS, a Hugging Face repo id, or a local folder.
    -> (repo id or local path, ModelInfo if it is a known model)"""
    if model in (None, "latest"):
        model = LATEST[task]
    if model in MODELS:
        info = MODELS[model]
        if info.task != task:
            raise ValueError(f"{model} is a {info.task} model, not a {task} model")
        return info.repo, info
    if os.path.isdir(model):
        return model, None
    known = {m.repo: m for m in MODELS.values()}
    if model in known:
        return resolve(known[model].name, task)
    if "/" in model:
        return model, None
    raise ValueError(f"unknown model {model!r}; known: {', '.join(MODELS)}")
