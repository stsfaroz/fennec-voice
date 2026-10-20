"""Fast checks, no model download."""
import pytest

import fennec_voice
from fennec_voice.registry import resolve
from fennec_voice.text import build_input, normalise


def test_version():
    assert fennec_voice.__version__.count(".") == 2


def test_registry():
    assert resolve("latest", "interrupt")[0] == "SalmanFaroz/fennec-interrupt-150m-v1"
    assert resolve(None, "handoff")[0] == "SalmanFaroz/fennec-handoff-150m-v1"
    assert resolve("SalmanFaroz/fennec-handoff-150m-v1", "handoff")[1].name == "fennec-handoff-150m-v1"
    assert resolve("someone/other-model", "interrupt") == ("someone/other-model", None)
    with pytest.raises(ValueError):
        resolve("fennec-handoff-150m-v1", "interrupt")
    with pytest.raises(ValueError):
        resolve("no-such-model", "interrupt")


def test_formatter():
    text = build_input("your flight is on the", "seventeenth of october", "mm hmm").text
    assert text == "[BOT SAID] your flight is on the\n[BOT NOT YET SAID] seventeenth of october\n[CALLER] mm hmm"
    assert build_input("", "", "").text.endswith("[CALLER] (empty)")
    assert normalise("OK, 2 tickets!") == "okay two tickets"
    assert normalise("Уга") == ""
