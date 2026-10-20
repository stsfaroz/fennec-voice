"""Downloads both models from Hugging Face. Run with: FENNEC_TEST_MODELS=1 pytest"""
import os

import pytest

from fennec_voice import FennecHandoff, FennecInterrupt

pytestmark = [pytest.mark.models,
              pytest.mark.skipif(not os.environ.get("FENNEC_TEST_MODELS"), reason="set FENNEC_TEST_MODELS=1")]


@pytest.fixture(scope="module")
def interrupt():
    return FennecInterrupt(device="cpu")


@pytest.fixture(scope="module")
def handoff():
    return FennecHandoff(device="cpu")


@pytest.mark.parametrize("said,unsaid,caller,label", [
    ("Your flight from Mumbai to Delhi is", "on the seventeenth of October.", "mm hmm", "BACKCHANNEL"),
    ("Business class for two adults. Is that", "correct?", "yes that's right", "ACK_AFTER_QUESTION"),
    ("What date would you", "like to travel?", "wait hold on", "HOLD_REQUEST"),
    ("The fare is nine thousand", "rupees per person.", "no business class", "CORRECTION"),
    ("I can book that", "for you right now.", "stop stop", "STOP_REQUEST"),
])
def test_interrupt(interrupt, said, unsaid, caller, label):
    r = interrupt.predict(said, unsaid, caller)
    assert r.label == label
    assert r.keep_talking == (label == "BACKCHANNEL")


def test_interrupt_no_words(interrupt):
    r = interrupt.predict("Let me check the", "flights for you.", "")
    assert r.label is None and r.keep_talking


def test_handoff(handoff):
    a, b, c = handoff.predict_batch(["I want to fly from Mumbai to um", "Delhi, one way for two people.", ""])
    assert a.finished is False and b.finished is True and c.finished is None
