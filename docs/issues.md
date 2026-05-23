# Issues — Work in Progress

## Features

### Multi documents support

See [the implementation plan](features/multi-document-pipeline.md).

### Feature 3 — Multi-model support (GLM-OCR)

Add GLM-OCR as an optional VLM alongside PaddleOCR-VL-1.5. GLM-OCR reuses `ppdoclayout` and `paddlex`, so the layout and table sub-pipelines stay identical — only the vision backbone changes.

**Prerequisites:**
- Define a model-registry abstraction (`ModelConfig` or similar) so prompt templates, API schemas, and sampling defaults are model-specific rather than hard-coded.
- Keep the door open for additional models without refactoring the pipeline each time.

**Next step:** prototype in `draft/` with a head-to-head quality comparison on the reference pages.

### Feature 4 — vLLM backend

Add vLLM as an optional inference backend alongside llama-server. OCR on hundreds of pages is embarrassingly parallel; vLLM's continuous batching and PagedAttention should deliver much higher throughput on capable hardware.

**Platform notes:**
- vLLM + ROCm keeps AMD GPU support viable, but ROCm is effectively Linux-only.
- On Windows, vLLM is practically CUDA-only today.
- Therefore, llama-server remains the default for Windows/APU users; vLLM is the recommended path for Linux + NVIDIA/AMD datacenter GPUs.

**Prerequisites:**
- Abstract the server lifecycle in `pipeline.py` (llama-server spawn/restart vs vLLM serve).
- `ocr_client.py` already speaks OpenAI-compatible HTTP, so the API surface is small.


## Bugs

### Fragility 1 (pending) — Resource Release

Resources are not always released when closing the server: "Background thread did not terminate in time. Some resources may not be properly released."

## Stabilisation (for release)

### Release readiness — CHANGELOG

Before tagging a release (e.g., `v0.1.0`), write a `CHANGELOG.md` starting from the first usable state. Requires: stable patch detection and resolved bugs above.
