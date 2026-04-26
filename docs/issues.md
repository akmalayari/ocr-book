# Issues — Work in Progress

## Multi documents version (current)

Multi documents support. See [the implementation plan](docs/features/multi-document-pipeline.md).


---

## Next versions

### Feature 2 - LaTeX Postprocess

Make OCR of academic documents with many LaTeX equations more robust.

### Fragility 1 (pending) - Resource Release

Resources are not always released when closing the server: "Background thread did not terminate in time. Some resources may not be properly released."

### Friction 2 — GPU memory contention from other apps

llama-server may hang or crash on startup if other applications (e.g., embedder/reranker, local LLM frontends, games) are already occupying GPU memory. On APUs with shared memory this is especially common.

**Symptom:** llama-server process starts but never prints `model loaded`, or the first VLM request returns HTTP 500.
**Fix:** Close GPU-heavy applications before starting the pipeline. Use Task Manager → Performance → GPU to verify available VRAM.
