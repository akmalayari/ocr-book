# Issues — Work in Progress

## Features

### Multi documents support

See [the implementation plan](features/multi-document-pipeline.md).

### Feature 2 — LaTeX Postprocess

Make OCR of academic documents with many LaTeX equations more robust.

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

### Friction 2 — GPU memory contention from other apps

llama-server may hang or crash on startup if other applications (e.g., embedder/reranker, local LLM frontends, games) are already occupying GPU memory. On APUs with shared memory this is especially common.

**Symptom:** llama-server process starts but never prints `model loaded`, or the first VLM request returns HTTP 500.
**Fix:** Close GPU-heavy applications before starting the pipeline. Use Task Manager → Performance → GPU to verify available VRAM.

### Robustness 1 — OTSL patch detection

French comments in the paddlex source cause the patch status check to return "unknown". The detection logic exists but needs to be more robust (e.g., hash-based or match the actual patched code signature).

### Robustness 2 — Output file overwrite

The pipeline always opens the output file (`book.md`) with `"w"` (truncate), overwriting any existing file. There is no automatic backup, numbering suffix, or prompt to prevent data loss.

**Impact:** Re-running the pipeline or doing an EPUB extraction will silently erase a previously generated `book.md`.
**Workaround:** Manually rename or copy `book.md` before re-running.
**Fix options:**
- Auto-increment suffix: `book.md` → `book_1.md` → `book_2.md` if the target exists.
- Or backup the existing file with a timestamp before writing.
- Or add a `--force` flag and prompt/abort by default when the target exists.

### Ops 1 — Destructive setup.py

`setup.py` unconditionally runs `conda env remove -n ocr-livre --yes` before creating the environment. Any user-installed packages, manual patches, or environment customizations are lost without confirmation or backup.

**Fix options:**
- Add a `--skip-env-remove` flag or prompt for confirmation.
- Or check if the env exists and only remove it when `--force` is passed.

### Robustness 3 — Non-interactive hang (header patterns)

`main.py` calls `input()` to prompt for header-detection regexes when `cfg.header_patterns is None`. In non-interactive shells, CI pipelines, or remote SSH sessions, the process hangs forever with no timeout.

**Fix:** Add a CLI flag (e.g., `--header-pattern REGEX LEVEL` or `--no-header-detection`) so the pipeline can run fully headless.

### Robustness 4 — EPUB Pandoc failure has no fallback

`epub.py` falls back to EbookLib only when Pandoc is **missing**. If Pandoc is installed but fails on a specific EPUB (corrupted, unsupported encoding, etc.), the pipeline aborts with `RuntimeError` instead of trying EbookLib.

**Fix:** Catch `RuntimeError` from `_extract_with_pandoc` and fall back to `_extract_with_ebooklib`, logging a warning.

### Cleanup 2 — Stale `.part` files accumulate

`output/parts/*.part` files are never cleaned up. Over multiple runs with different books, page IDs can collide (e.g., `page_001.part` from a previous run). On resume, the pipeline skips those pages, producing a mixed or truncated output.

**Fix options:**
- Clear `output/parts/` at the start of each run when `--no-resume` is used.
- Or hash the input folder/file name into a run-specific subfolder (`output/parts/<run_id>/`).


## Stabilisation (for release)

### Dependency update — PaddleOCR 3.5 released

PaddleOCR 3.5 is now available. Evaluate whether it improves layout detection accuracy or VLM output quality compared to the current pinned commit. Check if the OTSL and parallel patches still apply cleanly.

### Testing 1 — Smoke test

Add at least one automated smoke test that runs the pipeline on a single reference image. Currently there is no test suite — validation is entirely manual.

### Release readiness — CHANGELOG

Before tagging a release (e.g., `v0.1.0`), write a `CHANGELOG.md` starting from the first usable state. Requires: tests, stable patch detection, and resolved cleanup items above.
