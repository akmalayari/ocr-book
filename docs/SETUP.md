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
pip install git+https://github.com/PaddlePaddle/PaddleOCR.git

# Apply required paddlex patch
python docs/dev/apply_paddlex_patch_otsl.py

# Apply optional intra-page parallelism patch (gain ~30%, hardware dependent)
python docs/dev/apply_paddlex_patch_parallel.py
```

## Run the Pipeline

```bash
python src/main.py --help
python src/main.py <photos_dir>
```

## Troubleshooting

- **paddlex file not found**: Verify the env is activated (`conda activate ocr-livre`)
- **Patch fails**: paddlex state may be "unknown" if the version differs. See [apply_paddlex_patch_otsl.py](dev/apply_paddlex_patch_otsl.py) for details
