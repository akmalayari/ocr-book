# PaddleOCR-VL — Performance and Optimizations

## Reference Measurements

Hardware: Windows 11, GPU AMD Radeon 890M (Vulkan, iGPU), llama-server Vulkan backend.
Model: PaddleOCR-VL-1.5-0.9B, GGUF BF16, 890 MiB. Total VRAM: ~23 GiB.
Images: 4080×3072 px (12.5 MP), book page photos.

| Config | Speed/page | Notes |
|---|---|---|
| -np 1, sequential (baseline) | ~60s | initial state |
| -np 2, 2 workers (global pool) | ~49s | -11s |
| -np 3, 3 workers (global pool) | ~46s | -14s, **retained** |
| -np 4, 4 workers | crash | vision encoder Vulkan saturated |
| -np 6, 6 workers | hang | total GPU contention |

Time per page is proportional to the number of blocks detected by PP-DocLayoutV3 (each block = separate VLM call). Real gain on 150 pages: ~35 min.

## Bottlenecks

Two distinct bottlenecks:
1. **Vision encoder** (image encoding to base64 → tokens): ~350-4500 ms/block depending on size. Saturated from 4 simultaneous encodings under Vulkan → crash.
2. **LLM generation**: ~36 tok/s per slot, but multiple slots can generate in parallel if the vision encoder is not saturated.

## Tested Optimizations

### Intra-page Parallelism (global pool) — **retained**

PaddleOCR processes blocks of a page sequentially. A patch (`docs/dev/apply_paddlex_patch_parallel.py`) replaces the loop with a global `ThreadPoolExecutor` that submits all blocks from all `pixel_key`s simultaneously.

**Why "global pool" rather than "per pixel_key pool"**: the initial version (per pixel_key pool) recreated a pool for each block group — no overlap between groups. The global pool collects all blocks into a single list, workers pick continuously, results are redistributed by pixel_key afterwards.

Limit: Vulkan vision encoder crashes from 4 simultaneous encodings. Floor at 3 workers.

### `-np N` Alone (without intra-page parallelism)
**Result: counter-productive. Abandoned.**

Tested with `n_parallel=2` on 2 full pages simultaneously: 213s vs 110s. Requests fight for the entire GPU. Context divided between slots → each slot only has 2048 tokens → long blocks truncated (HTTP 400).

### PIL Resize Before predict (`--max-image-size 1500`)
**Result: strongly degraded quality. Abandoned.**

Tested on 4080×3072 → 1500×1129 images. Slightly faster but OCR strongly degraded.
Cause: resize was applied **before** layout detection which received a degraded image, compromising block detection.

### `max_pixels` Parameter
**Not applicable for `llama-cpp-server`. Ignored.**

`max_pixels` controls the number of pixels sent to the vision encoder. Default: `28×28×3600 = 2,822,400` px. Only supported by the `vllm-server` backend. For `llama-cpp-server`, a `warnings.warn` is emitted and the parameter is ignored. The image is sent as-is in base64 to llama-server.

### `--flash-attn` (llama-server)
Automatically enabled in llama-server.

### Increasing `n_ubatch`
**Tested (512 → 1024), no gain.**

## Retained llama-server Parameters

```
-c 6144      # context window (2048 tokens/slot × 3 slots)
-np 3         # parallel slots (consistent with VLM_PARALLEL=3 in patch)
-ngl 99       # all layers on GPU
-b 512        # batch size
-ub 512       # ubatch size (tested 1024: 0 gain)
-t 4          # CPU threads
--prio 2      # process priority
--temp 0.0    # deterministic
-kvo          # KV cache offload
```

KV cache with np=3: ~297 MiB × 3 / 2 ≈ 445 MiB. Total VRAM used: ~4.5 GiB out of 23 GiB.

## Unexplored Paths

- **vllm-server backend** instead of llama-cpp-server: would support `max_pixels` and potentially be more performant, but requires a different setup (Linux-friendly, not tested on Windows/Vulkan)
