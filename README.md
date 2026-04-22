# ocr-book — Book OCR Pipeline → Markdown

Digitizes an entire book into Markdown from page photos,
using **PaddleOCR-VL-1.5** via **llama-server** (local inference).

---

## Prerequisites

- [miniforge](https://github.com/conda-forge/miniforge) or Anaconda
- [llama-server](https://github.com/ggerganov/llama.cpp) (Vulkan recommended on Windows)
- GGUF model: [PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

---

## Installation

```bash
python setup.py
conda activate ocr-livre
```

See [docs/SETUP.md](docs/SETUP.md) for details.

---

## Project Structure

```
ocr-livre/
├── src/
│   ├── main.py          # CLI entry point
│   ├── config.py        # Central configuration (dataclass)
│   ├── ocr_client.py    # OCR of an image via PaddleOCRVL
│   ├── postprocess.py   # OCR text cleanup
│   ├── obsidian.py      # Obsidian export (wikilinks, migration)
│   ├── images.py        # Image collection and renaming
│   ├── pipeline.py      # Full orchestration
│   └── progress.py      # Logging and statistics
├── docs/
│   ├── architecture/    # Architecture documentation
│   ├── dev/             # Patches and development notes
│   ├── SETUP.md         # Installation instructions
│   ├── tested.md        # Experiment results
│   └── issues.md        # Work in progress
├── photos/              # Source images (one per page)
├── output/              # Generated Markdown + logs + figures
├── environment.yml      # Conda dependencies
└── setup.py             # Automated installation script
```

---

## Usage

Run from `src/`:

```bash
# Default pipeline (photos in ./photos, output output/book.md)
python main.py

# Specify folders
python main.py --images ./my_photos --out output/my_book.md

# Without layout detection
python main.py --no-layout

# Restart from the beginning
python main.py --no-resume

# Detailed logs
python main.py --verbose
```

---

## Obsidian Export

In `obsidian` mode, the pipeline:
- converts figures to wikilinks `![[Files/image.jpg]]`
- saves the `.md` directly into the vault
- copies figures to `vault_path/vault_figures_dir/`

Configure `vault_path` and `vault_figures_dir` in `config.py`, then:

```bash
# Full OCR + obsidian export
python main.py --mode obsidian

# Re-apply obsidian postprocess without re-running OCR
python main.py --mode obsidian --postprocess-only

# Migrate figures to the vault only
python main.py --migrate
```

---

## Image Renaming

```bash
# Preview without modifying
python main.py --rename --dry-run

# Rename for real (→ page_001.jpg, page_002.jpg, …)
python main.py --rename

# Rename without running OCR
python main.py --rename-only

# Process subfolders by chapter
python main.py --rename-only --chapters "Chapter 1" "Chapter 2"
```

---

## Automatic Resume

If the pipeline is interrupted, simply re-run:

```bash
python main.py
```

Already processed pages are automatically skipped.

---

## Full Options

```
--images PATH              Photo folder                (default: ./photos)
--out FILE                 Output Markdown file        (default: output/book.md)
--mode {base,obsidian}     Output mode                 (default: base)
--no-layout                Disable layout detection
--no-resume                Restart from the beginning
--no-postprocess           Raw output without cleanup
--postprocess-only         Obsidian postprocess without OCR  (requires --mode obsidian)
--migrate                  Copy figures to the vault  (requires vault_path configured)
--dry-run                  Simulate without modifying
--verbose                  DEBUG logs
--rename                   Rename images before OCR
--rename-only [N]          Rename without running OCR (N = starting number)
--rename-prefix P          Rename prefix                 (default: page)
--chapters NAME…           Subfolders to process (in order)
--dir-level                Folder-level order for --rename
```

---

## Exit Codes

| Code | Meaning                                      |
|------|----------------------------------------------|
| 0    | Full success                                 |
| 1    | Fatal error                                  |
| 2    | Finished with errors on some pages           |
