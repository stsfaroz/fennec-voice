from __future__ import annotations

import re
from typing import Iterable

import torch

from ._loading import load
from .text.format_input import build_input, normalise
from .types import InterruptResult


def plain(s: str | None) -> str:
    """Lowercase, no punctuation (apostrophes kept), single spaces."""
    return " ".join(re.sub(r"[^\w\s']", " ", (s or "").lower()).split())


class FennecInterrupt:
    """The agent is talking and the caller makes a sound. Is it a nod (keep talking) or does the caller take the turn?

        interrupt = FennecInterrupt()
        interrupt.predict(said="Your flight from Mumbai to Delhi is", unsaid="on the seventeenth of October.", caller="mm hmm")
    """

    def __init__(self, model: str | None = "latest", *, device: str = "auto", fp16: bool | None = None,
                 threads: int | None = None, revision: str | None = None):
        """model: "latest", a model name (see fennec_voice.list_models()), a Hugging Face repo id or a local folder.
        device: "auto" (GPU if available), "cuda" or "cpu". fp16: default on for GPU. threads: CPU threads (torch default if None).
        revision: a Hugging Face commit or tag, to pin the exact model files."""
        self.tok, self.model, h, self.device, self.info = load(model, "interrupt", device, fp16, revision)
        fmt = h.get("input_format", "fi-2.0")
        if fmt != "fi-2.0":
            raise ValueError(f"this fennec-voice version cannot build input format {fmt!r}; upgrade fennec-voice")
        self.labels: list[str] = h["acts"]
        self.stop_labels: set[str] = set(h["stop_acts"])
        self.max_len, self.max_tokens, self.threads = h["max_len"], h.get("max_input_tokens", 96), threads

    def model_input(self, said: str, unsaid: str, caller: str) -> str:
        """The exact text the model reads: [BOT SAID] ... / [BOT NOT YET SAID] ... / [CALLER] ..."""
        return build_input(plain(said), plain(unsaid), plain(caller),
                           count_tokens=lambda s: len(self.tok(s).input_ids), max_tokens=self.max_tokens).text

    def predict(self, said: str, unsaid: str, caller: str) -> InterruptResult:
        """said: the agent's words the caller had already heard. unsaid: the rest of the agent's sentence.
        caller: what the caller said (so far)."""
        return self.predict_batch([(said, unsaid, caller)])[0]

    @torch.inference_mode()
    def predict_batch(self, items: Iterable[tuple[str, str, str]]) -> list[InterruptResult]:
        """items: (said, unsaid, caller) tuples. One forward pass for all of them."""
        items = list(items)
        out: list[InterruptResult | None] = [None] * len(items)
        todo = []
        for i, (said, unsaid, caller) in enumerate(items):
            if normalise(caller or ""):
                todo.append(i)
            else:                                   # no words to judge (noise, junk speech-to-text)
                out[i] = InterruptResult(label=None, keep_talking=True, p_stop=0.0, probs={})
        if todo:
            if self.threads:
                torch.set_num_threads(self.threads)
            texts = [self.model_input(*items[i]) for i in todo]
            enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float()[:, :len(self.labels)], -1).cpu().tolist()
            for i, p in zip(todo, probs):
                pr = dict(zip(self.labels, p))
                label = max(pr, key=pr.get)
                out[i] = InterruptResult(label=label, keep_talking=label == "BACKCHANNEL",
                                         p_stop=sum(v for k, v in pr.items() if k in self.stop_labels), probs=pr)
        return out
