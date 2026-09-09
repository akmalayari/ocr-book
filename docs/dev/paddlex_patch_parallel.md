# VLM Intra-page Parallelism Patch

## Context

PaddleOCR-VL processes blocks of a page sequentially: for each block detected by PP-DocLayoutV3, an HTTP call is sent to llama-server. With 5-8 blocks per page, these calls run one after another while the GPU is often idle between two.

## Solution

`docs/dev/apply_paddlex_patch_parallel.py` patches `paddlex/inference/pipelines/paddleocr_vl/pipeline.py` to replace the sequential loop with a single global `ThreadPoolExecutor`.

**Must be applied after `apply_paddlex_patch_otsl.py` (OTSL patch).**

## How It Works

Original loop:
```python
for pixel_key in batch_dict_by_pixel:
    for image, query in zip(images, queries):
        result = self.vl_rec_model.predict(...)  # sequential
```

The patch collects all blocks (all pixel_keys combined) into a single list, submits them to the pool, then redistributes results:
```python
_all_tasks = [(img, qry, kwargs), ...]   # all blocks
_all_results = pool.map(_infer_block, _all_tasks)  # parallel
# redistribute by pixel_key afterwards
```

**Why thread-safe**: PaddleX uses `asyncio.run_coroutine_threadsafe()` on a single global event loop (background thread). Multiple threads can call `predict()` simultaneously — their coroutines stack on the same loop, which executes them concurrently in I/O towards llama-server.

## Consistent Parameters

The patch reads `OCR_N_PARALLEL` at runtime. The pipeline publishes the resolved
configuration value before prediction, so the PaddleX worker pool and
llama-server's `-np` argument stay synchronized. CLI precedence remains
`--n-parallel` > `.env` > default (`1`). Changing the value does not require
reapplying the patch.

Context (`-c`) must still be sized for the selected concurrency:

| OCR_N_PARALLEL / -np | Recommended -c | Tokens/slot |
|---|---|---|
| 2 | 4096 | 2048 |
| 3 | 6144 | 2048 — hardware-dependent |
| 4 | 8192 | 2048 — experimental, hardware-dependent |

## Limits

- **Hardware-dependent limit**: higher concurrency can saturate or crash the
  Vulkan vision encoder. Start at 2, validate representative pages, then test 3
  or higher only if the hardware remains stable.
- **Diminishing returns on the tested machine**: 60s → 49s → 43.4s text /
  37s graph (2.4s gain between np=2 and np=3). These measurements and the
  failure threshold are not portable to other GPUs.
- **Pages with few blocks**: a page with 2 blocks only uses 2 workers even with np=3. Gain is proportional to the number of blocks.

## Usage

```bash
# Apply (OTSL patch must already be active)
python docs/dev/apply_paddlex_patch_parallel.py

# Default remains sequential. Test two workers first.
python main.py --n-parallel 2

# Check
python docs/dev/apply_paddlex_patch_parallel.py --check

# Remove (back to OTSL patch only)
python docs/dev/apply_paddlex_patch_parallel.py --revert
```

Applying the script upgrades older patches with a hardcoded worker count.
Reverting is also independent of the current `OCR_N_PARALLEL` value and leaves
`.env` unchanged. Set the value back to `1` before running without the patch.

## Associated src/ Config

`src/pipeline.py`: `-np` receives the resolved `n_parallel` value.
`src/config.py`: `n_parallel = 1`; automatic `n_ctx = n_parallel * 2048`.
