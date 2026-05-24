# AGENTS.md — ocr-book

Reference file for AI coding agents. This project is a Python CLI pipeline that OCRs an entire book (page photos) into Markdown, using **PaddleOCR-VL-1.5** served locally by **llama-server** (llama.cpp, Vulkan backend).

---

## Project Overview

| | |
|---|---|
| **Name** | ocr-book |
| **Language** | Python 3.10 |
| **Type** | Local CLI tool (no web server, no deployment) |
| **Goal** | Convert a folder of book page photos into a single Markdown file, with HTML tables, extracted figures, and optional Obsidian export. |
| **Primary doc language** | English (docstrings, comments, README, docs). Commit messages in English. |

### Tech Stack

| Component | Role |
|-----------|------|
| **llama-server** (llama.cpp, Vulkan) | Local VLM inference of the PaddleOCR-VL-1.5 GGUF model |
| **paddleocr** (installed from git repo, not PyPI) | Orchestration: layout detection → prompt routing → VLM calls |
| **paddlepaddle 3.3.1** | CPU layout detection (ppdoclayout) |
| **paddlex[ocr] 3.4.3** | Table sub-pipeline (OTSL → HTML) |
| **openai** | HTTP client for paddleocr's `llama-cpp-server` backend |
| **requests** | llama-server health polling |
| **natsort** | Natural sorting of files and folders |

---

## Project Structure

```
ocr-livre/
├── main.py                       # CLI entry point (argparse)
├── src/                          # Main source code (9 modules)
│   ├── config.py                 # Config dataclass (all default values)
│   ├── pipeline.py               # Full orchestration (servers, parallelism, parts, fallback)
│   ├── ocr_client.py             # OCR of an image via PaddleOCRVL + retry/timeout
│   ├── postprocess.py            # Text cleanup, page number extraction, headers
│   ├── images.py                 # Collection, renaming, copying from subfolders
│   ├── obsidian.py               # Obsidian export (wikilinks, figure migration)
│   ├── progress.py               # Configured logging + statistics (Stats dataclass)
│   ├── pdf.py                    # PDF processing (text extraction or render → OCR)
│   └── epub.py                   # EPUB extraction (Pandoc-based)
│
├── docs/                         # Project documentation
│   ├── architecture/overview.md  # Detailed pipeline architecture
│   ├── SETUP.md                  # Installation instructions
│   ├── tested.md                 # Results of all experiments
│   ├── issues.md                 # Work in progress / planned features
│   ├── dev/                      # Patches and development scripts
│   │   ├── apply_paddlex_patch_otsl.py      # Required patch: per-region VLM error handling
│   │   ├── apply_paddlex_patch_parallel.py  # Optional patch: intra-page parallelism
│   │   ├── paddlex_patch_otsl.md            # OTSL patch doc
│   │   └── paddlex_patch_parallel.md        # Parallel patch doc
│   └── paddleocr/                # Internal PaddleOCR docs (config, output, performance)
│
├── draft/                        # Explorations, informal tests, throwaway scripts (gitignored)
├── photos/                       # Source images (gitignored)
├── output/                       # Generated Markdown, logs, figures, reports (gitignored)
├── memory/                       # Inter-session memory notes (gitignored)
│
├── environment.yml               # Conda dependencies (Python 3.10 + pip packages)
├── setup.py                      # Automated installation script (conda env + patches)
├── README.md                     # User documentation
└── .gitignore                    # Python + data folders + draft/ + memory/
```

---

## Installation and Build

### System Prerequisites
- Windows (developed and tested on Windows)
- [miniforge](https://github.com/conda-forge/miniforge) or Anaconda
- [llama-server](https://github.com/ggerganov/llama.cpp) compiled with Vulkan (or another GPU backend)
- GGUF model: [PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) (`.gguf` + `.mmproj.gguf`)

### Installation Commands

```bash
# Full setup (creates conda env, installs deps, applies patch)
python setup.py

# Activate environment
conda activate ocr-livre

# Verify everything works
python -c "from paddleocr import PaddleOCRVL; print('OK')"
```

The `setup.py` script:
1. Removes the old conda env `ocr-livre` (if it exists)
2. Creates the env from `environment.yml`
3. Installs `paddleocr` from the git repo (the PyPI version does not contain the `llama-cpp-server` backend)
4. Applies the required patch `docs/dev/apply_paddlex_patch_otsl.py`

### Required paddlex patches

| Patch | Status | Command |
|-------|--------|----------|
| **OTSL** (`apply_paddlex_patch_otsl.py`) | **Required** | `python docs/dev/apply_paddlex_patch_otsl.py` |
| **Parallel** (`apply_paddlex_patch_parallel.py`) | Optional (~30% gain) | `python docs/dev/apply_paddlex_patch_parallel.py` |

These patches modify the installed file at `sys.prefix/Lib/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py`. They accept `--check` and `--revert` arguments.

---

## Execution

All commands are run from the **project root** with the `ocr-livre` environment activated.

```bash
# Default pipeline (images)
python main.py
python main.py --images ./photos --out output/book.md

# PDF input
python main.py --images ./book.pdf --out output/book.md

# EPUB input
python main.py --images ./book.epub --out output/book.md

# Main options
python main.py --no-layout          # Disable layout detection
python main.py --no-resume          # Restart from the beginning
python main.py --no-postprocess     # Raw output without cleanup
python main.py --verbose            # DEBUG logs

# Obsidian mode
python main.py --mode obsidian
python main.py --mode obsidian --postprocess-only   # Without re-running OCR

# Image renaming
python main.py --rename --dry-run   # Preview
python main.py --rename             # Rename to page_001.jpg, page_002.jpg...
python main.py --rename-only        # Rename without running OCR
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Full success |
| 1 | Fatal error |
| 2 | Finished with errors on some pages |

---

## Execution Architecture

```
photos/  →  main.py  →  pipeline.run_pipeline(cfg)
                              │
                              ├── images._collect_sources(cfg)       # images + PDFs + EPUBs
                              ├── pdf.process_pdf(...)  (if needed)  # text extract or render
                              ├── _start_server(cfg, port) × n_servers  (if OCR needed)
                              ├── PaddleOCRVL(...) × n_servers
                              │
                              └── ThreadPoolExecutor(max_workers=n_servers)
                                    for each image/PDF page:
                                    ├── ocr_client.ocr_image(img, pl, cfg)
                                    │     ├── pipeline.predict(image)       # layout + OCR VLM
                                    │     ├── save_to_markdown(save_path)   # figures/page/page.md
                                    │     └── returns (text, {latency})
                                    ├── postprocess.extract_page_number(text)
                                    ├── postprocess.clean_page(text, cfg)
                                    ├── postprocess.strip_table_styles(text)
                                    ├── fix_image_paths (base) or fix_image_paths_obsidian
                                    └── writes to output/parts/<page_id>.part
                              │
                              ├── Combines parts → output/book.md
                              ├── apply_header_detection (if configured)
                              ├── migrate_figures (if obsidian mode)
                              └── stats.write_report(output/ocr_report.md)
```

### Key Architecture Points

- **Automatic resume** (`--resume`): each processed page is written to `output/parts/<page_id>.part`. On restart, existing parts are skipped. Disable with `--no-resume`.
- **Timeout + fallback**: if `pipeline.predict()` exceeds `cfg.page_timeout` (120s default), servers are restarted and the page is reprocessed without layout detection (`use_layout_detection=False`).
- **Parallelism**: `n_servers` llama-server instances on distinct ports (8080, 8081…). One page per server in parallel. The intra-page parallel patch (`-np 3`) allows processing multiple blocks of the same page simultaneously via a thread pool in paddlex.
- **Figures**: crops of `image` regions are saved in `output/figures/<page_id>/imgs/`.

---

## Code Organization

### Main Modules (`src/`)

| Module | Responsibility |
|--------|----------------|
| `main.py` | CLI argparse, obsidian argument validation, pipeline launch |
| `config.py` | `Config` dataclass with all defaults (paths, tuning, post-processing, obsidian) |
| `pipeline.py` | Full orchestration: server startup, inter-page parallelism, parts management, timeout fallback, final assembly |
| `ocr_client.py` | Interface `ocr_image(image, pipeline, cfg) → (text, metrics)`. Handles ×1 retry, threaded timeout, raises `OCRError` / `OCRTimeout` |
| `postprocess.py` | Markdown cleanup (page numbers, broken words, blank lines), table styles, header detection, page block formatting |
| `images.py` | Image discovery and sorting (`collect_images`), sequential renaming (`rename_images`), copying from subfolders (`copy_from_subdirs`) |
| `obsidian.py` | `<img>` → wikilinks `![[...]]`, figure migration to vault, postprocess on existing `.md` |
| `progress.py` | Logging configuration (console INFO + file DEBUG), `Stats` dataclass with final Markdown report |
| `pdf.py` | PDF classification (text vs image-based), text extraction with `pymupdf`, figure detection, page rendering for OCR |
| `epub.py` | EPUB extraction to Markdown via Pandoc, with figure extraction and path rewriting |

### Key Configuration (`config.py`)

Paths to llama-server and models are read from **environment variables** by default (`LLAMA_SERVER_PATH`, `MODEL_PATH`, `MMPROJ_PATH`). They can also be passed via CLI (`--llama-server`, `--model`, `--mmproj`) or edited directly in `src/config.py`.

Important tuning parameters:
- `n_servers`: number of parallel llama-servers (1 default, useless on single APU/GPU)
- `n_parallel`: intra-page slots (3 default, must match parallel patch)
- `n_ctx`: 6144 (2048 tokens/slot × np=3)
- `page_timeout`: 120s before giving up and restarting
- `use_layout_detection`: True by default

---

## Code Style and Conventions

### Language
- **Code**: English (variable, function, class names, commits)
- **Docstrings and comments**: English
- **User messages / logs**: English
- **Commits**: brief English message, format `fix(module): description` or `feat(scope): description`

### Development Conventions
- Do not modify `README.md` unless explicitly requested.
- Do not add formal tests unless explicitly requested. Explorations and informal tests go in `draft/`.
- `draft/` is gitignored — never commit its contents.
- `output/` and `photos/` are gitignored — they are data folders.
- After resolving an issue, update `docs/issues.md`:
  - Delete resolved subsections (`###`) and items.
  - Avoid leaving an empty section (`##`): write "OK".
- Group modified files in a single commit when relevant.
- Avoid adding "Co-Authored-By: AI..." attribution tags in commits unless the project conventions require them.
- Do not generate docstrings or comments on unmodified code.
- Do not explore `output/`, `photos/`, `__pycache__`, `.pytest_cache` (large and irrelevant content).
- Use git worktrees for parallel branches (two features, two fixes, etc.): `git worktree add -b <branch> ../<name> master`. Since `--images` accepts absolute paths, no symlinks are needed. Remove with `git worktree remove ../<name>` when done.

### Interaction Rules
When the user asks for an opinion, proposal, or point of view using phrases such as (in any language):
- "What do you think?"
- "What do you propose?"
- "What is your opinion?"
- or any similar phrasing,

**respond with text only. Do not write or modify code.** Wait for explicit user approval (e.g., "Go ahead", "Do it", "Yes", "OK", etc.) before making any code changes.

### Dependency Management
- No `pyproject.toml`, `requirements.txt`, or `poetry.lock`.
- Dependencies are declared in `environment.yml` (conda + pip).
- `setup.py` is an **installation script** (not a setuptools package).

---

## Tests

A minimal automated smoke-test suite lives in `tests/test_smoke.py`. It validates imports, config defaults, and CLI argument parsing without requiring llama-server or the model stack.

```bash
python -m pytest tests/ -v
```

The primary test strategy remains **experimental and manual**:
- Test and comparison scripts are in `draft/` (gitignored).
- Experiment results are documented in `docs/tested.md`.
- Reference pages used: `page_1` to `page_9` (described in `docs/tested.md`).

### Useful development scripts in `docs/dev/`

```bash
# Check patch status
python docs/dev/apply_paddlex_patch_otsl.py --check
python docs/dev/apply_paddlex_patch_parallel.py --check

# Quick diagnostic
python main.py --images photos/page_1.jpg --no-resume
# Then check output/ocr_run.log
```

---

## Security and Operational Considerations

- **Path configuration**: llama-server and model paths are read from environment variables (`LLAMA_SERVER_PATH`, `MODEL_PATH`, `MMPROJ_PATH`) or CLI arguments. The user must set them before the first run.
- **Patches on installed libraries**: patches directly modify files in `sys.prefix/Lib/site-packages/paddlex/`. They must be reapplied after each environment reinstallation.
- **GPU resources**: the pipeline launches `n_servers` llama-server processes. Each process loads the model into GPU memory. On an APU (shared CPU/GPU memory), `n_servers > 1` generally brings no gain due to Vulkan command queue serialization.
- **Timeout and resume**: the parts mechanism makes the pipeline robust to crashes. However, a hard kill may leave orphan llama-server processes or unreleased resources (see `docs/issues.md`).
- **No secrets / API keys**: everything is local (llama-server). No data is sent over the network.

---

## Quick Troubleshooting

| Symptom | Likely Cause | Solution |
|----------|---------------|----------|
| `paddlex file not found` | Conda env not activated | `conda activate ocr-livre` |
| VLM 500 error on tables | OTSL patch not applied | `python docs/dev/apply_paddlex_patch_otsl.py` |
| Dense tables cut off (first rows only) | Context slot too small (`n_ctx / n_parallel`) | Increase `--n-ctx` or decrease `--n-parallel` (e.g., `--n-ctx 12288 --n-parallel 3`) |
| Vision encoder crash | `-np` too large (≥4) | Lower to `-np 3` |
| No pages processed | `photos/` empty or wrong path | Check `--images` and extensions |
| Timeout on all pages | llama-server not started or model not found | Check `--llama-server`, `--model`, `--mmproj` or the corresponding env vars |
| llama-server hangs on startup | GPU memory held by other apps (Obsidian embedder, games, etc.) | Close GPU-heavy apps, verify free VRAM in Task Manager |

---

## External Resources

- llama-server docs: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- PaddleOCR docs: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- HuggingFace model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
