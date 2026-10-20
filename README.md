<div align="center">

<h1>🦊 fennec-voice</h1>

<p><b>Small turn-taking models for voice agents: is the caller nodding, interrupting, or done?</b></p>

<a href="https://pypi.org/project/fennec-voice/"><img src="https://img.shields.io/pypi/v/fennec-voice" alt="PyPI"></a>
<a href="https://huggingface.co/SalmanFaroz/fennec-interrupt-150m-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-fennec--interrupt--150m--v1-yellow" alt="fennec-interrupt"></a>
<a href="https://huggingface.co/SalmanFaroz/fennec-handoff-150m-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-fennec--handoff--150m--v1-yellow" alt="fennec-handoff"></a>
<a href="https://github.com/stsfaroz/fennec-voice-demo"><img src="https://img.shields.io/badge/%E2%96%B6%20Demo-fennec--voice--demo-red" alt="Demo"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>

</div>

---

## ✨ What it does

| Model | Moment | Question |
|---|---|---|
| **FennecInterrupt** | The agent is talking and the caller makes a sound | Nod (keep talking) or interruption (stop and listen)? |
| **FennecHandoff** | The agent is quiet and the caller paused | Finished (reply) or not (wait)? |

Two 150M models, ~12 ms per decision on a laptop GPU (~30–60 ms on CPU). English, text in, text out: plug them in after any speech-to-text.

---

## 🚀 Install

```bash
pip install fennec-voice
```

The GPU is used when available, otherwise the CPU. For a smaller CPU-only install, get PyTorch's CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fennec-voice
```

---

## 🧩 Use

```python
from fennec_voice import FennecInterrupt, FennecHandoff

interrupt = FennecInterrupt()
r = interrupt.predict(said="Your flight from Mumbai to Delhi is",   # what the caller had heard
                      unsaid="on the seventeenth of October.",      # the rest of the agent's sentence
                      caller="mm hmm")                               # what the caller said
r.label, r.keep_talking        # ('BACKCHANNEL', True)

handoff = FennecHandoff()
handoff.predict("I want to fly from Mumbai to um").finished   # False: wait for more
handoff.predict("Delhi, one way for two people.").finished    # True: reply
```

- `said` / `unsaid`: split the agent's sentence at the word playing when the caller started speaking (TTS word timestamps + playback position).
- `predict_batch([...])` runs many inputs in one pass.
- Results have `.to_dict()`.

**Labels** (`interrupt.labels`): `BACKCHANNEL` (a nod: keep talking) and, for stop and listen: `ACK_AFTER_QUESTION`, `ANSWER`, `QUESTION`, `CORRECTION`, `STOP_REQUEST`, `HOLD_REQUEST`, `CANT_HEAR`, `PICKUP_GREETING`, `TAKES_TURN`.

---

## ⚙️ Options

```python
FennecInterrupt(
    model="latest",      # or "fennec-interrupt-150m-v1", a Hugging Face repo id, or a local folder
    device="auto",       # "cuda" or "cpu"
    fp16=None,           # default: on for GPU
    threads=None,        # CPU threads
    revision=None,       # pin exact model files (Hugging Face commit or tag)
)
FennecHandoff(..., threshold=0.5)   # p_unfinished at or above it means "not finished"
```

```python
import fennec_voice
fennec_voice.list_models()   # all known models
```

The package and the models have separate versions: new models arrive as new names, and `"latest"` only moves to a better default. Pin a name for stable behaviour.

---

## ▶️ Demo

[**fennec-voice-demo**](https://github.com/stsfaroz/fennec-voice-demo): a full local voice agent built around these two models, one `./run.sh`.

---

## 💡 Good to know

- English only.
- "okay" can mean "I'm listening" or "yes": right before the agent's yes/no question it counts as "yes".
- v1, a hobby project: expect rough edges.

## 📄 License

Apache-2.0. The input formatter comes from [ozonetg/bargein-classifier-ettin-150m](https://huggingface.co/ozonetg/bargein-classifier-ettin-150m) (MIT), see [NOTICE](NOTICE).
