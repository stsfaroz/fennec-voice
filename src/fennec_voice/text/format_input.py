# From https://huggingface.co/ozonetg/bargein-classifier-ettin-150m (MIT License, see NOTICE).
# The Fennec v1 models read text built by exactly this code. Only the imports changed for packaging.
"""The input formatter for the binary barge-in classifier (FORMAT_VERSION fi-2.0).

Train = eval = runtime: the model was trained on exactly this text, so never build a model input any other way.
One exchange only, no history.

    [BOT SAID] <agent_said: what the caller heard>
    [BOT NOT YET SAID] <agent_unsaid: the rest of the bot's line, not heard; context only>
    [CALLER] <caller_said: ASR style>

- The caller text goes LAST and is never cut.
- Agent text: textnorm.normalise_agent ("light": lowercase, numbers as words, but , . ? ! kept as tokens so the
  question cue survives). Caller text: textnorm.normalise (idempotent on ASR output).
- Truncation, only when the full text exceeds `max_tokens`:
    1. agent_unsaid is cut from its END down to UNSAID_MIN_HEAD words (its first words tell what comes next);
    2. agent_said is cut from its START down to SAID_MIN_TAIL words (the tail is what the caller just heard);
    3. agent_unsaid is cut to nothing;
    4. agent_said is cut down to SAID_HARD_MIN words.
  "..." marks a cut. If it still does not fit, over_budget=True (the tokenizer's left truncation is the last net;
  the caller text is at the end so it survives even that).
- Empty agent segments render as "(nothing)", an empty caller segment as "(empty)".
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

from .textnorm import normalise, normalise_agent  # noqa: F401

FORMAT_VERSION = "fi-2.0"
M_SAID = "[BOT SAID]"
M_UNSAID = "[BOT NOT YET SAID]"
M_CALLER = "[CALLER]"
EMPTY_AGENT = "(nothing)"
EMPTY_CALLER = "(empty)"
ELLIPSIS = "..."
SAID_MIN_TAIL = 24
SAID_HARD_MIN = 8
UNSAID_MIN_HEAD = 8


class FormattedInput(NamedTuple):
    text: str
    truncated: bool
    over_budget: bool


class _Seg:
    """keep='tail' cuts from the start, keep='head' cuts from the end."""
    __slots__ = ("words", "keep", "n")

    def __init__(self, words, keep):
        self.words, self.keep, self.n = words, keep, len(words)

    def render(self) -> str:
        if not self.words:
            return EMPTY_AGENT
        if self.n >= len(self.words):
            return " ".join(self.words)
        if self.n <= 0:
            return ELLIPSIS
        if self.keep == "tail":
            return ELLIPSIS + " " + " ".join(self.words[-self.n:])
        return " ".join(self.words[:self.n]) + " " + ELLIPSIS


def _assemble(said: _Seg, unsaid: _Seg, caller: str) -> str:
    return (f"{M_SAID} {said.render()}\n{M_UNSAID} {unsaid.render()}\n{M_CALLER} {caller or EMPTY_CALLER}")


def build_input(agent_said: str, agent_unsaid: str, caller_said: str, *,
                count_tokens: Optional[Callable[[str], int]] = None,
                max_tokens: Optional[int] = None) -> FormattedInput:
    said = _Seg(normalise_agent(agent_said or "").split(), "tail")
    unsaid = _Seg(normalise_agent(agent_unsaid or "").split(), "head")
    caller = normalise(caller_said or "")
    text = _assemble(said, unsaid, caller)
    if max_tokens is None or count_tokens is None or count_tokens(text) <= max_tokens:
        return FormattedInput(text, False, False)

    def fits() -> bool:
        return count_tokens(_assemble(said, unsaid, caller)) <= max_tokens

    def shrink(seg: _Seg, floor: int) -> bool:
        floor = max(0, min(floor, seg.n))
        hi = seg.n
        seg.n = floor
        if not fits():
            return False
        lo = floor
        while lo < hi:
            mid = (lo + hi + 1) // 2
            seg.n = mid
            if fits():
                lo = mid
            else:
                hi = mid - 1
        seg.n = lo
        return True

    ok = False
    for seg, floor in ((unsaid, UNSAID_MIN_HEAD), (said, SAID_MIN_TAIL), (unsaid, 0), (said, SAID_HARD_MIN)):
        if shrink(seg, floor):
            ok = True
            break
    return FormattedInput(_assemble(said, unsaid, caller), True, not ok)
