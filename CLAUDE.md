# CLAUDE.md — ocr-book

Python CLI pipeline that OCRs a book (page photos) into Markdown via PaddleOCR-VL-1.5 served locally by llama-server.

## Architecture

```
src/
  main.py        — CLI argparse, entry point
  config.py      — Config dataclass (all defaults here)
  ocr_client.py  — OCR of an image via PaddleOCRVL
  postprocess.py — Text cleanup + page block management in .md
  images.py      — Collection, renaming and copying from subfolders
  pipeline.py    — Full orchestration (multi-servers, parts, fallback)
  obsidian.py    — Obsidian export (wikilinks, migrate_figures, postprocess_file)
  progress.py    — Logging + statistics (Stats dataclass)
```

Dependencies: `environment.yaml`. Conda env: `ocr-livre`. Run from `src/`: `python main.py`.

Project docs: `docs/`.

Explorations and informal tests: `draft/`. Experiment results: `docs/tested.md`.

Work in progress: `docs/issues.md`.

Check your memory at session start: `memory/`.

## Conventions

- **Commits**: brief English message (`fix(module): description`)
- **Language**: code and commits in English
- **Do not modify README** unless explicitly requested
- **Do not add tests** unless explicitly requested

## Work Preferences

### General
- Read files directly (Glob/Grep/Read) without using a sub-agent unless the search is really open-ended
- Do not suggest corrections beyond the requested scope
- If the user says "I fixed X", check the actual file state before assuming anything
- No recap at the end of the message unless the change is complex
- Answer honestly: disagree if necessary and explain your point of view
- Always ask for clarifications before coding unless instructions are clear or obvious

### Git
- add and commit for each resolved issue: group modified files when relevant unless the change is isolated
- `draft/` is gitignored
- do not add the message "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

### Issues
- after implementation, update issues.md if relevant
- after implementation, delete resolved subsections (`###`) and items (`issues.md`)
- avoid leaving an empty section (`##`) in issues.md: write "OK"

### Draft
- All explorations and initial tests go in `draft/`
- All outputs from `draft/` must land in `output/`
- Before writing a test script, always check if modules or functions from `src/` can be reused

## Limit token consumption

- Do not re-read a file already read in the conversation unless it has changed or explicitly requested
- Use targeted Grep rather than a broad Glob across the whole repo
- Do not explore `output/` nor `photos/` nor `__pycache__` nor `.pytest_cache` (irrelevant, very large content)
- Do not generate docstrings or comments on unmodified code

## Resources
Documentation on the specific stack used in the project.

### llama-server

- Docs GitHub: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

### PaddleOCR

- General documentation: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html

- HuggingFace page: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

- GitHub page: https://github.com/PaddlePaddle/PaddleOCR

- Internal docs: `docs/paddleocr/`

## Troubleshooting
- Run `python src/main.py --images photos/page_1.jpg --no-resume`.
- Check `output/ocr_run.log`.
- Find what's wrong.

As a last resort only: run tests `python -m pytest tests/ -v`.
