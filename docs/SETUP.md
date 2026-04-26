# Setup — ocr-book (PaddleOCR version)

## Environment Installation

### Option 1: Automatic Script (recommended)

```bash
python setup.py
conda activate ocr-livre
# Optional: intra-page parallelism patch
python docs/dev/apply_paddlex_patch_parallel.py
```

### Option 2: Manual

```bash
# Create conda env from environment.yml
conda env create -f environment.yml

# Activate env
conda activate ocr-livre

# Install PaddleOCR from git repo (dev version with llama-server compatibility)
pip install "git+https://github.com/PaddlePaddle/PaddleOCR.git@740a04dc4"

# Apply required paddlex patch
python docs/dev/apply_paddlex_patch_otsl.py

# Apply optional intra-page parallelism patch (gain ~30%, hardware dependent)
python docs/dev/apply_paddlex_patch_parallel.py
```

---

## Configuration

Before running OCR, you must tell the pipeline where `llama-server` and the models are located.

### Option A: Environment Variables (recommended)

```bash
# Windows (PowerShell)
$env:LLAMA_SERVER_PATH = "C:\path\to\llama-server.exe"
$env:MODEL_PATH        = "C:\path\to\PaddleOCR-VL-1.5.gguf"
$env:MMPROJ_PATH       = "C:\path\to\PaddleOCR-VL-1.5-mmproj.gguf"

# Linux / macOS
export LLAMA_SERVER_PATH=/path/to/llama-server
export MODEL_PATH=/path/to/PaddleOCR-VL-1.5.gguf
export MMPROJ_PATH=/path/to/PaddleOCR-VL-1.5-mmproj.gguf
```

### Option B: CLI Arguments

```bash
python main.py \
  --llama-server /path/to/llama-server \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --images ./photos
```

### Option C: Edit config.py

Edit `src/config.py` directly and set the absolute paths in the `Config` dataclass.

---

## Run the Pipeline

```bash
# Default (photos in ./photos)
python src/main.py

# With explicit paths
python src/main.py --images ./photos --out output/book.md

# PDF input
python src/main.py --images ./book.pdf --out output/book.md

# EPUB input
python src/main.py --images ./book.epub --out output/book.md
```

---

## Troubleshooting

- **paddlex file not found**: Verify the env is activated (`conda activate ocr-livre`)
- **Patch fails**: paddlex state may be "unknown" if the version differs. See [apply_paddlex_patch_otsl.py](dev/apply_paddlex_patch_otsl.py) for details
- **Missing required configuration**: You haven't set the llama-server or model paths. See the Configuration section above.
