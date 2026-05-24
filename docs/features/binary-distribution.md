# Binary Distribution — Planned Feature

> Status: **Planned** (post-v1.0)  
> Goal: Provide a Windows installer so end users can run the pipeline without manually managing Python, conda, or dependencies.

---

## Problem

The current installation requires:
- miniforge/conda
- `conda activate ocr-livre`
- `python setup.py`
- Manual patch application
- Manual configuration of `LLAMA_SERVER_PATH`, `MODEL_PATH`, `MMPROJ_PATH`

This is acceptable for development and technical users, but it is a barrier for a broader audience and for an "official release."

---

## Non-Goals

- **Single-file executable.** Freezing the entire Python ML stack (PaddlePaddle, PaddleX, llama.cpp bindings) into one `.exe` via PyInstaller/Nuitka is fragile and produces enormous binaries (~2–5 GB). The complexity of debugging native-library loading at runtime outweighs the UX benefit.
- **Bundling model weights.** The `.gguf` / `.mmproj.gguf` files are multi-gigabyte and evolve independently. Like llama.cpp and Ollama, models will remain a separate download step.

---

## Proposed Approach: Lightweight Installer

Instead of a frozen binary, ship a **portable, pre-configured Python environment** wrapped in a standard Windows installer (Inno Setup or NSIS). The user experience is:

1. Download `OCR-Book-Setup-x.x.x.exe`
2. Run installer (Next → Next → Finish)
3. Download or point to a PaddleOCR-VL-1.5 GGUF model
4. Launch from Start Menu or desktop shortcut

Under the hood, the shortcut simply runs `python src/main.py` from an isolated, private Python copy.

---

## Option 1: Embeddable Python + pip

Use the official Windows **embeddable Python** distribution (`python-3.10.x-embed-amd64.zip`).

### How it works
- Ship the ~8 MB embeddable Python base.
- Pre-install all dependencies as wheels into the embeddable environment.
- Bundle `src/`, `llama-server.exe`, and a launcher batch script.
- Wrap in Inno Setup / NSIS.

### Launcher (`ocr-book.bat`)
```batch
@echo off
set PATH=%~dp0python;%PATH%
set LLAMA_SERVER_PATH=%~dp0bin\llama-server.exe
python\python.exe src\main.py %*
```

### Pros
- Minimal installer overhead.
- No registry or system Python required.
- Clean, self-contained folder.

### Cons
- Converting the full conda dependency tree (`paddlepaddle`, `paddlex[ocr]`, etc.) to pure pip wheels may be difficult.
- Some packages may lack Windows wheels or require manual DLL placement.
- The OTSL and parallel patches would need to be applied to the pre-installed wheels before bundling.

---

## Option 2: Portable Miniforge / Micromamba (Recommended)

Ship a **portable conda prefix** created with `micromamba` or `miniforge`.

### How it works
- Create a fully resolved conda environment in a local prefix:
  ```bash
  micromamba create -p ./ocr-book-env -f environment.yml
  ```
- The installer unpacks this prefix (~1–2 GB) along with `src/` and `bin/`.
- Activation is done via environment variables in a launcher script (no system conda needed).

### Launcher (`ocr-book.bat`)
```batch
@echo off
set CONDA_PREFIX=%~dp0env
set PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%PATH%
set LLAMA_SERVER_PATH=%~dp0bin\llama-server.exe
python src\main.py %*
```

### Pros
- Existing `environment.yml` works unchanged.
- `paddlepaddle`, `paddlex`, `pymupdf`, and all native libraries install exactly as in development.
- Patches can be applied during the build step before packaging.
- Behavior is identical to the dev environment, making debugging straightforward.

### Cons
- Larger installer size (~1.5 GB). Acceptable given the target audience already downloads multi-gigabyte AI models.

---

## Option 3: `uv` + Python Standalone Builds

Use [`uv`](https://github.com/astral-sh/uv) from Astral with `python-build-standalone` distributions.

### How it works
- Ship `uv.exe` + a pinned Python 3.10 standalone build.
- Use `uv pip install` with a lockfile or pre-downloaded wheel directory.
- `uv` creates the virtual environment on the user's machine during install (very fast).

### Pros
- Extremely fast install and update process.
- Modern tooling with excellent wheel caching.
- Clean update story: replace wheel directory and re-install.

### Cons
- Requires all dependencies to be available as wheels (no conda packages).
- `paddlepaddle` wheels exist on PyPI, but verifying that `paddlex[ocr]` and its transitive dependencies install cleanly outside conda requires extra validation.
- The project currently targets conda; a pip-only migration is extra work.

---

## Recommended Option

**Option 2: Portable Miniforge / Micromamba.**

Rationale:
- The project is already battle-tested on conda. Porting the dependency tree to pure pip is unnecessary work.
- The size difference versus a "true binary" is negligible once PaddlePaddle is included.
- Debugging user issues is trivial because their runtime matches the developer's runtime exactly.
- `llama-server.exe` can be bundled in `./bin/` and pointed to via the launcher script.

---

## Installer Flow (Inno Setup)

Regardless of the Python backend, the installer should:

1. **Welcome / License** (MIT)
2. **Select install folder** (`C:\Program Files\ocr-book` or a portable path)
3. **Optional:** Ask for `llama-server.exe` path if not bundled; otherwise use bundled binary
4. **Optional:** Ask for model folder or skip — user configures later via CLI or settings file
5. **Extract** the pre-built environment + `src/` + `bin/`
6. **Create shortcuts:**
   - `OCR-Book` (default pipeline)
   - `OCR-Book Obsidian Mode`
   - `OCR-Book (Command Prompt)` (opens terminal in isolated env)
7. **Add to PATH** (optional checkbox, off by default)

---

## Update Story

Ship patch updates that replace only `src/` and `docs/` without touching the heavy environment folder. A simple `update.bat` can download the latest release ZIP from GitHub and overwrite the code directory, or the installer can support in-place upgrades via Inno Setup's standard versioning.

---

## Open Questions

- Should the installer bundle `llama-server.exe` directly, or prompt the user to select their own build (to support different GPU backends: Vulkan, CUDA, ROCm)?
- Should we provide a "portable mode" (install to USB drive) in addition to a system-wide installer?
- How should first-run model configuration work — wizard, CLI prompt, or a simple settings JSON file?
