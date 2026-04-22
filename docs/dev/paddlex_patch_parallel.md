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

`_VLM_PARALLEL` in the patch must match `-np` in llama-server, and `-c` must be sized accordingly:

| VLM_PARALLEL / -np | Recommended -c | Tokens/slot |
|---|---|---|
| 2 | 4096 | 2048 |
| 3 | 6144 | 2048 — **retained** |
| 4 | 8192 | 2048 — vision encoder crash |

## Limits

- **Vision encoder Vulkan saturated at 4 workers**: 4 simultaneous image encodings crash the driver. Stable floor at 3 workers.
- **Diminishing returns**: 60s → 49s → 43.4s text / 37s graph (2.4s gain between np=2 and np=3). Beyond 3, crash.
- **Pages with few blocks**: a page with 2 blocks only uses 2 workers even with np=3. Gain is proportional to the number of blocks.

## Usage

```bash
# Apply (OTSL patch must already be active)
python docs/dev/apply_paddlex_patch_parallel.py

# Check
python docs/dev/apply_paddlex_patch_parallel.py --check

# Remove (back to OTSL patch only)
python docs/dev/apply_paddlex_patch_parallel.py --revert
```

## Associated src/ Config

`src/pipeline.py` : `-np 3`
`src/config.py` : `n_ctx = 6144`
