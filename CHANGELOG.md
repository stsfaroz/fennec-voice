# Changelog

## 0.1.0

- First release: `FennecInterrupt` (fennec-interrupt-150m-v1) and `FennecHandoff` (fennec-handoff-150m-v1).
- `device="auto"`: GPU when available, otherwise CPU.
- `predict_batch()` for several inputs in one pass.
- `list_models()`; pin a model by name, Hugging Face repo, local folder or `revision`.
