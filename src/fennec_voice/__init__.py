"""fennec-voice: small turn-taking models for voice agents.

    from fennec_voice import FennecInterrupt, FennecHandoff
    interrupt = FennecInterrupt()   # the agent is talking and the caller makes a sound: nod or interruption?
    handoff = FennecHandoff()       # the agent is quiet and the caller paused: finished or not?
"""
from .handoff import FennecHandoff
from .interrupt import FennecInterrupt
from .registry import LATEST, MODELS, ModelInfo, list_models
from .types import HandoffResult, InterruptResult

__version__ = "0.1.1"
__all__ = ["FennecInterrupt", "FennecHandoff", "InterruptResult", "HandoffResult", "list_models", "MODELS", "LATEST",
           "ModelInfo", "__version__"]
