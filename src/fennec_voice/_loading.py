from __future__ import annotations

import json
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .registry import ModelInfo, resolve


def pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load(model: str | None, task: str, device: str, fp16: bool | None, revision: str | None):
    """-> (tokenizer, model, heads.json dict, device, ModelInfo or None)"""
    repo_or_path, info = resolve(model, task)
    path = repo_or_path if os.path.isdir(repo_or_path) else snapshot_download(repo_or_path, revision=revision)
    heads = json.load(open(os.path.join(path, "heads.json")))
    if not str(heads.get("task", task)).startswith(task):
        raise ValueError(f"{repo_or_path} is not a {task} model (heads.json task: {heads.get('task')!r})")
    device = pick_device(device)
    if fp16 is None:
        fp16 = device.startswith("cuda")
    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
    net = (net.half() if fp16 else net).to(device)
    return tok, net, heads, device, info
