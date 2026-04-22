# Architecture — ocr-book (PaddleOCR version)

Python CLI pipeline that OCRs a book (page photos) into Markdown via PaddleOCR-VL-1.5 served locally by llama-server.

---

## Overview

```
photos/          →  pipeline  →  output/book.md  (or vault_root/vault_path/book.md in obsidian mode)
                                  output/figures/<page>/
                                  output/parts/<page>.part
                                  output/ocr_report.md
```

Execution flow:

```
main.py
  └── pipeline.run_pipeline(cfg)
        ├── images.collect_images(cfg)                 # sorted image list
        ├── _start_server(cfg, port) × n_servers       # n subprocesses llama-server
        ├── _wait_for_server(url, timeout)             # /health polling for each server
        ├── PaddleOCRVL(...) × n_servers               # one pipeline per server
        │
        └── ThreadPoolExecutor(max_workers=n_servers) — for each image:
              ├── ocr_client.ocr_image(img, pipeline, cfg)
              │     ├── pipeline.predict(image)            # layout + OCR
              │     ├── save_to_markdown(save_path)        # writes figures/page/page.md + imgs/
              │     └── returns (markdown_text, {total_latency})
              │
              ├── postprocess.extract_page_number(text)
              ├── postprocess.clean_page(text, cfg)
              ├── postprocess.strip_table_styles(text)
              ├── fix_image_paths / fix_image_paths_obsidian
              ├── postprocess.format_page_block(page_id, text, page_number)
              └── writes to output/parts/<page_id>.part
        │
        ├── Combines parts in input order → output/book.md
        ├── postprocess.apply_header_detection(text, cfg.header_patterns)  # if configured
        ├── obsidian.migrate_figures(cfg)               # if obsidian mode
        └── stats.write_report(...)

  On page timeout:
        ├── restart_servers()                           # kill + relaunch all servers
        └── retry with fallback pipeline (use_layout_detection=False)
```

---

## Modules

### `main.py`
CLI entry point (argparse). Parses arguments, builds `Config`, calls `run_pipeline`.

Main options:
- `--images`, `--out` — paths
- `--no-layout`, `--no-resume`, `--no-postprocess` — OCR behavior
- `--mode [base|obsidian]` — output mode
- `--postprocess-only` — with `--mode obsidian`: postprocess on existing `.md` without re-running OCR
- `--migrate` — copy figures to vault without running OCR
- `--rename`, `--rename-only [START]`, `--rename-prefix` — image renaming
- `--chapters NAME…` — subfolders to process in given order
- `--dir-level` — folder-level order (alpha folders > alpha subfolders > images by date)
- `--dry-run`, `--verbose`

### `config.py`
`Config` dataclass — all default values in one place.

Parameter groups:

- **llama-server paths**: `llama_server_path`, `model_path`, `mmproj_path`, `server_base_port`, `server_timeout`
- **llama-server tuning**: `n_ctx`, `n_gpu_layers`, `n_batch`, `n_ubatch`, `n_threads`, `prio`, `kv_offload`, `temperature`, `max_tokens`
- **Parallelism**: `n_servers` (parallel servers), `n_parallel` (intra-page slots, requires patch), `page_timeout` (max seconds per page, 0 = disabled)
- **PaddleOCR**: `use_layout_detection`
- **Images**: `images_dir`, `extensions`, `rename_prefix`, `image_files` (explicit list, bypasses `images_dir`)
- **Output**: `output_file`, `figures_dir`, `resume`
- **Mode**: `mode` (`"base"` | `"obsidian"`)
- **Obsidian**: `vault_root`, `vault_path`, `vault_figures_dir`
- **Post-processing**: `postprocess`, `remove_isolated_page_numbers`, `rejoin_hyphenated_words`, `collapse_blank_lines`, `header_patterns`
- **Logging**: `log_file`, `report_file`, `verbose`

### `pipeline.py`
Full orchestration. Responsibilities:

1. Start `n_servers` llama-server as subprocesses (ports `server_base_port`, `server_base_port+1`, …)
2. Wait for each server to respond on `/health`
3. Instantiate `n_servers` PaddleOCRVL pipelines in a queue
4. Process pages in parallel via `ThreadPoolExecutor(max_workers=n_servers)`
5. Write each page to `output/parts/<page_id>.part` (atomic)
6. On timeout (`OCRTimeout`): restart all servers, retry with fallback pipeline (without layout)
7. Combine parts in input order at end of run
8. Apply `apply_header_detection` if `cfg.header_patterns`
9. Call `obsidian.migrate_figures` if obsidian mode
10. Stop all servers in `finally` block

### `ocr_client.py`
OCR of a single image. Interface: `ocr_image(image_path, pipeline, cfg) → (text, metrics)`.

Internal steps:
1. `pipeline.predict(image_path)` — layout detection (ppdoclayout) + VLM OCR per region
2. `save_to_markdown(save_path)` — writes `figures/<page>/<page>.md` + crops in `figures/<page>/imgs/`
3. `read_text()` — loads generated Markdown
4. Returns `(text, {"total_latency": float})` — latency measured on predict+save+read

Raises `OCRTimeout` if `cfg.page_timeout` is exceeded (caught in `pipeline.py`).

### `postprocess.py`
Cleanup of Markdown generated by PaddleOCR.

- `clean_page(text, cfg, no_layout)` — isolated page number removal, hyphenated word rejoining, blank line reduction; in no_layout mode, also removes generation loops
- `strip_table_styles(text)` — removes inline CSS styles from `<table>` and `<td>/<th>` generated by PaddleOCR, centers tables
- `extract_page_number(text)` — extracts printed page number from first/last N lines; returns `(label, cleaned_text)` where label is `None`, `"42"`, or `"42-43"`
- `apply_header_detection(text, header_patterns)` — adds markdown headers according to configured regex patterns, with anti-false-positive heuristics
- `fix_image_paths(text, page_id, figures_rel)` — fixes `src="imgs/..."` paths to be relative from the output file's folder (base mode)
- `format_page_block(page_id, text, page_number)` — wraps each page with `<!-- Page xxx (p. NN) -->`
- `format_error_block(page_id, error)` — error block for a failed page
- `extract_done_pages(output_text)` — reads `<!-- Page ... -->` markers for resume (old method, replaced by parts)

### `obsidian.py`
Utilities for Obsidian export.

- `prompt_if_needed(cfg)` — interactive prompt for `vault_root`, `vault_path`, `vault_figures_dir` if not configured
- `fix_image_paths_obsidian(text, vault_figures_dir)` — converts `<img src="imgs/...">` to Obsidian wikilinks `![[vault_figures_dir/...]]`; removes `<div>` wrappers
- `migrate_figures(cfg, page_ids, dry_run)` — copies crops from `output/figures/*/imgs/*` to `vault_root/vault_figures_dir/` (flat structure, skip if already present)
- `postprocess_file(cfg)` — applies full postprocess (`clean_page`, `strip_table_styles`, img → wikilinks conversion, `apply_header_detection`) on an existing `.md` without re-running OCR

### `images.py`
- `collect_images(cfg)` — lists and sorts images in `cfg.images_dir`; if `cfg.image_files` is provided, uses it directly; detects duplicate names
- `rename_images(folder, extensions, prefix, dry_run, start)` — renames images to `page_001.jpg`, `page_002.jpg`…
- `has_image_subdirs(folder, extensions)` — returns True if the folder contains subfolders with images
- `copy_from_subdirs(folder, extensions, chapters, prefix, start, dry_run, dir_level)` — copies images from subfolders to parent folder with sequential numbering; `chapters` allows choosing subfolders in order; `dir_level` sorts by folder then date

### `progress.py`
`Stats` dataclass — accumulates run metrics and generates the final report.

Collected metrics: OCR time per page, post-processing time, total time, characters, errors, fallback (no_layout) pages, skipped pages.

Markdown report written to `output/ocr_report.md`.

---

## Inference Stack

| Component | Role |
|-----------|------|
| **llama-server** (llama.cpp, Vulkan) | VLM inference (PaddleOCR-VL-1.5 GGUF F16) |
| **paddleocr** (from git repo) | Orchestration: layout detection → prompt routing → VLM calls |
| **paddlepaddle CPU** | Layout detection (ppdoclayout) |
| **paddlex[ocr]** | Table sub-pipeline (OTSL → HTML via `convert_otsl_to_html`) |
| **openai** | HTTP client for paddleocr's `llama-cpp-server` backend |

Required paddlex patches:
- `docs/dev/apply_paddlex_patch_otsl.py` — per-region VLM error handling (complex tables). See `docs/dev/paddlex_patch_otsl.md`.
- `docs/dev/apply_paddlex_patch_parallel.py` — intra-page parallelism (`n_parallel`). Required if `n_parallel > 1`. See `docs/dev/paddlex_patch_parallel.md`.

---

## Output Format

PaddleOCR generates embedded HTML in Markdown (Obsidian-compatible):

```markdown
<!-- Page page_001 (p. 42) -->

Regular paragraph text...

<table align="center" border=1>...</table>

![[Files/OCR/page_001_fig_0.png]]  (obsidian mode)
<img src="figures/page_001/imgs/page_001_fig_0.png" />  (base mode)
```

Figure crops are saved in `output/figures/<page>/imgs/`.

---

## Resume (`--resume`)

Each processed page is written to `output/parts/<page_id>.part`. On startup, the pipeline lists existing `.part` files — already present pages are skipped. At end of run, parts are combined in input order. Disable with `--no-resume` (deletes existing parts).

---

## Obsidian Mode (`--mode obsidian`)

When `--mode obsidian` is active:
1. `vault_root`, `vault_path`, `vault_figures_dir` are prompted if not configured in `Config`
2. Output file is written to `vault_root/vault_path/book.md`
3. `<img src="imgs/...">` are converted to wikilinks `![[vault_figures_dir/...]]`
4. At end of run, crops are copied to `vault_root/vault_figures_dir/` (`migrate_figures`)

Derived options:
- `--postprocess-only` — applies postprocess on existing `.md` without re-running OCR
- `--migrate` — copies only figures to the vault

---

## Performance Tuning

Configurable llama-server parameters in `Config`:

| Parameter | Default | Effect |
|-----------|--------|--------|
| `n_servers` | 1 | Number of parallel llama-servers (one page per server simultaneously) |
| `n_gpu_layers` | 99 | Layers offloaded to GPU (Vulkan) |
| `n_batch` / `n_ubatch` | 512 | Batch sizes |
| `kv_offload` | True | KV cache offload to CPU |
| `n_ctx` | 6144 | Max context (2048 tokens/slot × n_parallel=3) |
| `n_parallel` | 3 | Intra-page parallel slots (requires paddlex patch) |
| `page_timeout` | 120 | Max seconds per page before giving up and restarting server (0 = disabled) |
