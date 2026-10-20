from __future__ import annotations

import re
from typing import Iterable

import torch

from ._loading import load
from .types import HandoffResult


def _clean(s: str | None) -> str:
    return " ".join(re.sub(r"[^a-z' ]", "", (s or "").lower()).split())


class FennecHandoff:
    """The agent is quiet and the caller paused. Has the caller finished (reply) or not (wait for more)?

        handoff = FennecHandoff()
        handoff.predict("I want to fly from Mumbai to um")
    """

    def __init__(self, model: str | None = "latest", *, device: str = "auto", fp16: bool | None = None,
                 threads: int | None = None, revision: str | None = None, threshold: float = 0.5):
        """Same options as FennecInterrupt. threshold: p_unfinished at or above it means "not finished"."""
        self.tok, self.model, h, self.device, self.info = load(model, "handoff", device, fp16, revision)
        fmt = h.get("input_format", "plain-1")
        if fmt != "plain-1":
            raise ValueError(f"this fennec-voice version cannot build input format {fmt!r}; upgrade fennec-voice")
        self.T, self.max_len = float(h.get("temperature", 1.0)), h.get("max_len", 48)
        self.threads, self.threshold = threads, threshold

    def predict(self, caller: str) -> HandoffResult:
        """caller: the caller's words since the agent stopped."""
        return self.predict_batch([caller])[0]

    @torch.inference_mode()
    def predict_batch(self, texts: Iterable[str]) -> list[HandoffResult]:
        clean = [_clean(t) for t in texts]
        out: list[HandoffResult] = [HandoffResult(finished=None, p_unfinished=None)] * len(clean)
        todo = [i for i, t in enumerate(clean) if t]
        if todo:
            if self.threads:
                torch.set_num_threads(self.threads)
            enc = self.tok([clean[i] for i in todo], truncation=True, max_length=self.max_len, padding=True,
                           return_tensors="pt").to(self.device)
            ps = torch.sigmoid(self.model(**enc).logits.float()[:, 0] / self.T).cpu().tolist()
            for i, p in zip(todo, ps):
                out[i] = HandoffResult(finished=p < self.threshold, p_unfinished=p)
        return out
