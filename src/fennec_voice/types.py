from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class InterruptResult:
    label: str | None            # one of FennecInterrupt.labels, or None when the caller text has no words
    keep_talking: bool           # True only for BACKCHANNEL (a nod) or no words
    p_stop: float                # total probability of the labels that mean "stop and listen"
    probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HandoffResult:
    finished: bool | None        # None when the caller text has no words
    p_unfinished: float | None

    def to_dict(self) -> dict:
        return asdict(self)
