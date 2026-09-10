# Setup — ocr-book (PaddleOCR version)

## Environment Installation

### Prerequisites

- Miniforge or Anaconda, with `conda` initialized in the current shell.
- A 64-bit Python-compatible system. The project environment uses Python 3.10.
- Enough disk space for the PaddleOCR-VL GGUF and mmproj files (about 1.82 GB).

For an automated Vulkan llama-server build on Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake git libvulkan-dev glslc spirv-headers vulkan-tools
vulkaninfo --summary
```

The setup script reports missing build tools but never invokes `sudo` itself. PaddlePaddle's published Linux support should also be checked for the Ubuntu release in use; a newer Ubuntu release may work through the Linux wheel before it is officially listed.

### Option 1: Automatic Script (recommended)

```bash
python setup.py                    # Windows or an activated conda base shell
# or
python3 setup.py                   # Linux when only python3 is available
conda activate ocr-livre
```

The normal setup applies the required OTSL recovery patch and the
runtime-configurable parallel patch. Parallelism remains disabled by default
(`OCR_N_PARALLEL=1`); test `2` on representative pages before trying higher
hardware-dependent values.

On Linux, the normal setup downloads the pinned official PaddleOCR-VL-1.5 GGUF
model and mmproj (about 1.82 GB total) when valid files are not already
configured. Interrupted downloads can be resumed by rerunning setup.

On Windows, the normal setup never downloads model files automatically. It
checks configured paths and the default cache, records valid files in `.env`, or
prints browser download links and optional default destination paths. Files may
instead be stored anywhere by setting `OCR_MODEL_PATH` and `OCR_MMPROJ_PATH` in
`.env`. After downloading both files, rerun the local check:

```bash
conda run -n ocr-livre python docs/dev/download_models.py --local-only
```

To keep browser downloads in another directory, place both files there and run:

```bash
conda run -n ocr-livre python docs/dev/download_models.py --local-only --model-dir "C:/path/to/models" --force-location
```

The local check verifies the pinned file sizes and GGUF headers before recording
their paths. Automatic download remains available as an explicit Windows opt-in.
Use these options when needed:

```bash
python setup.py --model-only       # explicitly download or verify model files
python setup.py --model-dir PATH   # check this local directory during normal Windows setup
python setup.py --skip-model       # keep model setup fully manual
```

The default cache is
`%LOCALAPPDATA%/ocr-book/models/PaddleOCR-VL-1.5-GGUF/` on Windows and
`$XDG_CACHE_HOME/ocr-book/models/PaddleOCR-VL-1.5-GGUF/` on Linux when
configured. Without `XDG_CACHE_HOME`, Linux uses
`~/.cache/ocr-book/models/PaddleOCR-VL-1.5-GGUF/`. Existing valid
`OCR_MODEL_PATH` and `OCR_MMPROJ_PATH` values are preserved.

On Linux, setup can also fetch the latest llama.cpp `master`, build a Vulkan server in the user cache (`$XDG_CACHE_HOME/ocr-book` or `~/.cache/ocr-book`), verify its devices, and record its path in `.env`:

```bash
python3 setup.py --build-llama     # Python environment + llama-server
python3 setup.py --llama-only      # llama-server only
```

The cached `master` checkout is refreshed on every build. Use `--llama-ref REF` to pin a specific branch, tag, or commit (for example, `--llama-ref b8683` for the previously tested build).
Use `--llama-build-dir PATH` when the default cache is unavailable. Prefer a native Linux filesystem: CMake and compiler toolchains can fail or become extremely slow on some NTFS, network, or shared mounts. Build concurrency defaults to eight jobs and can be changed with `--llama-jobs N`.

### Option 2: Manual

```bash
# Create conda env from environment.yml
conda env create -f environment.yml

# Activate env
conda activate ocr-livre

# Install PaddleOCR from git repo (dev version with llama-server compatibility)
pip install "git+https://github.com/PaddlePaddle/PaddleOCR.git@740a04dc4"

# Apply the required paddlex patches, in this order
python docs/dev/apply_paddlex_patch_otsl.py
python docs/dev/apply_paddlex_patch_parallel.py

# Linux: download the pinned GGUF model and write its paths to .env
python docs/dev/download_models.py

# Windows: validate browser-downloaded files and write their paths to .env
python docs/dev/download_models.py --local-only --model-dir "C:/path/to/models" --force-location
```

---

## Configuration

Before running OCR, you must tell the pipeline where `llama-server` is located.
The normal setup configures model files automatically on Linux and registers
browser-downloaded files on Windows; custom model paths remain supported.

### Option A: `.env` file (recommended)

Copy the example file and edit it.

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Set the llama-server path in `.env`. Model paths are optional overrides:

```bash
# Linux
OCR_LLAMA_SERVER_PATH=/absolute/path/to/llama-server
OCR_MODEL_PATH=       # optional: automatic user-cache fallback
OCR_MMPROJ_PATH=      # optional: automatic user-cache fallback

# Windows uses C:/... paths and llama-server.exe instead.
```

The server value may also be a command such as `llama-server` when it is available on `PATH`. The pipeline resolves the executable and validates all three files before starting OCR. A Windows `.exe` cannot be reused on Linux.

If `.env` is removed, the standard model cache is still discovered
automatically. Explicit CLI or `.env` paths take precedence over cached files.

### Option B: Environment Variables

```bash
# Windows (PowerShell)
$env:OCR_LLAMA_SERVER_PATH = "C:\path\to\llama-server.exe"
$env:OCR_MODEL_PATH        = "C:\path\to\PaddleOCR-VL-1.5.gguf"
$env:OCR_MMPROJ_PATH       = "C:\path\to\PaddleOCR-VL-1.5-mmproj.gguf"

# Linux / macOS
export OCR_LLAMA_SERVER_PATH=/path/to/llama-server
export OCR_MODEL_PATH=/path/to/PaddleOCR-VL-1.5.gguf
export OCR_MMPROJ_PATH=/path/to/PaddleOCR-VL-1.5-mmproj.gguf
```

### Option C: CLI Arguments

```bash
python main.py \
  --llama-server /path/to/llama-server \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --images ./photos
```

### Option D: Edit config.py

Edit `src/config.py` directly and set the absolute paths in the `Config` dataclass.

---

## Run the Pipeline

```bash
# Default (photos in ./photos)
python main.py

# With explicit paths
python main.py --images ./photos --out output/book.md

# PDF input
python main.py --images ./book.pdf --out output/book.md

# EPUB input
python main.py --images ./book.epub --out output/book.md
```

### Installation verification

```bash
conda activate ocr-livre
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddleocr import PaddleOCRVL; print('OK')"
python docs/dev/apply_paddlex_patch_otsl.py --check
python docs/dev/apply_paddlex_patch_parallel.py --check
# Linux: automatic download/verification
python docs/dev/download_models.py

# Windows: local verification without network
python docs/dev/download_models.py --local-only
llama-server --list-devices  # or the absolute path recorded in .env
python -m pytest tests/ -v
python main.py --help
```

Before processing a full book, run one representative page with `--no-resume` and inspect `output/ocr_run.log`.

---

## Troubleshooting

- **paddlex file not found**: Verify the env is activated (`conda activate ocr-livre`)
- **Patch fails**: paddlex state may be "unknown" if the version differs. See [apply_paddlex_patch_otsl.py](dev/apply_paddlex_patch_otsl.py) for details
- **Missing required configuration**: Configure `llama-server`. Model paths are
  discovered from the user cache unless explicit overrides point to missing files.
  See the Configuration section above.
- **llama-server is not executable**: Run `chmod +x /path/to/llama-server`.
- **No Vulkan device appears**: Run `vulkaninfo --summary`, verify the GPU driver, and rebuild with `-DGGML_VULKAN=ON`.
