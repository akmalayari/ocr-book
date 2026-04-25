# Multi-Document Pipeline — Implementation Plan

**Status**: Planned  
**Goal**: Allow a single input folder to contain multiple files (images, PDFs, EPUBs) and produce one Markdown file per source document, while loading the VLM only once.

---

## 1. Overview

Today the pipeline assumes a **single output** per run:
- One folder of images → one `book.md`
- One folder with images + PDFs → everything concatenated into one `book.md`
- One EPUB → one `book.md`, but rejected if mixed with other formats

The new behaviour:
- Each **source file** becomes its own output document.
- Images in the input folder are grouped into a single virtual document.
- The VLM (llama-server + PaddleOCRVL instances) is started once, processes all pages from all image/PDF documents, then shuts down.
- EPUBs are extracted without touching the VLM.
- `--unify` forces concatenation of all outputs into a single file (legacy behaviour).

---

## 2. Document Definition

A "document" is a unit that produces its own `.md` file.

| Source | Document type | Output stem example |
|--------|---------------|---------------------|
| `page_001.jpg`, `page_002.jpg` … | `images` (batch) | `book-images` |
| `memoire.pdf` | `pdf` | `book-memoire` |
| `livre.epub` | `epub` | `book-livre` |

**Rules**:
- One PDF = one document.
- One EPUB = one document.
- All loose images in the folder = one document.
- Order: alphabetical by source filename (EPUBs, PDFs, images sorted together).

---

## 3. Output File Naming

Given `--out output/book.md`:

| Scenario | Output files |
|----------|-------------|
| Single document (any type) | `output/book.md` |
| Images + `memoire.pdf` | `output/book-images.md`, `output/book-memoire.md` |
| `a.pdf` + `b.epub` + images | `output/book-a.md`, `output/book-b.md`, `output/book-images.md` |
| `--unify` passed | `output/book.md` (everything concatenated) |

The stem is taken from `--out`. The suffix is:
- `-images` for the loose-image batch
- `-<source_stem>` for PDFs and EPUBs

If `--out` is a directory path (ends with `/` or `\`), each document produces `<source_stem>.md` inside that directory.

---

## 4. Figure Directory Layout

Each document gets its own isolated figure tree to avoid collisions.

```
output/
├── figures/
│   ├── book-images/
│   │   ├── page_001/
│   │   │   └── imgs/
│   │   │       └── fig.jpg
│   │   └── page_002/
│   │       └── imgs/
│   │           └── fig.jpg
│   ├── book-memoire/
│   │   └── page_001/
│   │       └── imgs/
│   │           └── fig.jpg
│   └── book-livre/
│       ├── CH01_F01.png
│       └── CH01_F02.png
```

**OCR documents** (images + PDFs): keep the existing `figures/<doc_stem>/<page_id>/imgs/` structure.  
**EPUB documents**: flat inside `figures/<doc_stem>/`.

---

## 5. Option `--unify`

New CLI flag. When set, the pipeline reverts to **single-output mode**:
- All EPUB extracts, all PDF renderings, all image OCRs are concatenated into the single `--out` file.
- Figures still go to `figures/<doc_stem>/` (to avoid collisions), but the Markdown references them all.
- Useful when the user explicitly wants one big file.

Default: `False`.

---

## 6. VLM Lifecycle (Single Load)

### 6.1. Problem

`run_pipeline()` today does:
1. Collect sources
2. Start servers
3. Process pages
4. Assemble parts
5. Stop servers

If we call it N times, the VLM is loaded N times.

### 6.2. Solution

Split `run_pipeline` into three public phases:

```python
# pipeline.py

class ServerPool:
    """Context manager that starts and stops llama-server instances."""
    def __init__(self, cfg: Config): ...
    def __enter__(self) -> "ServerPool": ...
    def __exit__(self, *exc): ...
    def get_pipeline(self) -> PaddleOCRVL: ...
    def put_pipeline(self, pl: PaddleOCRVL): ...
    def restart(self): ...
```

And a processing function that takes the pool:

```python
def process_document(
    cfg: Config,
    doc: Document,
    server_pool: ServerPool,
    parts_parent: Path,
) -> None:
    """
    Processes all pages of a single document (images or PDF).
    Writes .part files to parts_parent / <doc_stem> / <page_id>.part.
    """
```

### 6.3. Sequence

```
main.py
│
├── Collect sources
├── Build Document list
│
├── For each EPUB doc:
│   └── extract_epub(...)      # no VLM
│
├── Pre-render all PDF docs:
│   └── pdf.process_pdf(...) for each PDF
│
├── With ServerPool(cfg) as pool:
│   │
│   └── For each image/PDF doc:
│       └── process_document(doc, pool)
│
├── For each doc (including EPUBs):
│   └── assemble_parts(doc)
│
└── Obsidian post-process & migrate per doc
```

---

## 7. Resume Mechanism

Today: `output/parts/<page_id>.part`

New: `output/parts/<doc_stem>/<page_id>.part`

- Resume checks per document: if a document has **all** its `.part` files present, skip it entirely.
- If `--no-resume`, delete `output/parts/<doc_stem>/*` for every document.
- EPUBs are idempotent (fast), so they are always re-extracted unless the output `.md` already exists and `--resume` is on.

---

## 8. Stats & Logging

`Stats` is currently designed for a single run. Options:

1. **One global Stats + per-doc sub-stats** (preferred)
   - `Stats` gets a `documents: dict[str, Stats]` field.
   - Each `process_document` populates its own sub-stat.
   - Final report shows a summary table per document.

2. **List of Stats**
   - Simpler but harder to print a unified report.

---

## 9. Modifications by File

### `src/config.py`

```python
@dataclass
class Config:
    ...
    # NEW
    unify: bool = False
```

### `main.py`

**New data class** (local to main):
```python
@dataclass
class Document:
    stem: str                 # suffix for naming
    doc_type: str             # "images" | "pdf" | "epub"
    sources: list[Path]       # original source files
    output_md: Path           # final markdown path
    figures_dir: Path         # root figure dir for this doc
    temp_images: list[Path]   # rendered PDF pages (if pdf)
```

**New function** `build_documents(sources, cfg) -> list[Document]`:
- Groups loose images into one `Document(stem="images", doc_type="images")`
- Creates one `Document` per PDF and per EPUB
- Computes `output_md` and `figures_dir` for each

**Refactored `main()` flow**:
1. Parse args
2. Build `cfg`
3. Obsidian setup
4. `sources = _collect_sources(cfg)`
5. `docs = build_documents(sources, cfg)`
6. If `--unify`:
   - Override every `doc.output_md` to `cfg.output_path`
7. Extract EPUB docs (no VLM)
8. Pre-render PDF docs (`pdf.process_pdf`)
9. If image/PDF docs exist:
   - `with ServerPool(cfg) as pool:`
   - For each doc: `process_document(cfg, doc, pool)`
10. For each doc: `assemble_document(doc)`
11. If obsidian: `postprocess_and_migrate(doc)` per doc

### `src/pipeline.py`

**New / refactored:**

1. `class ServerPool`
   - Extract server startup from `_start_server` + `_wait_for_server`
   - Extract queue management (`pipeline_queue`, `restart_servers`, `get_fallback_pipeline`)
   - Keep the same retry/fallback/timeout logic

2. `process_document(cfg, doc, pool, parts_parent)`
   - Like the inner `process_page` of today, but scoped to one document
   - Writes `.part` to `parts_parent / doc.stem / <page_id>.part`
   - Calls `fix_image_paths` with the document-specific `figures_rel`

3. `assemble_document(doc, parts_parent)`
   - Reads `parts_parent / doc.stem / *.part`
   - Writes to `doc.output_md`

4. Keep `run_pipeline(cfg)` as a **thin compatibility wrapper**:
   - Builds a single `Document` from `cfg`
   - Calls the new multi-doc flow
   - This preserves any external callers or tests

### `src/postprocess.py`

`fix_image_paths(text, page_id, figures_rel)` already accepts `figures_rel` — no signature change needed.  
Just ensure `figures_rel` is computed per-document in `process_document`.

### `src/epub.py`

No change needed — `extract_epub(epub_path, output_md, figures_dir)` is already document-scoped.

### `src/obsidian.py`

`migrate_figures` already supports `flat=True`.  
For multi-document, we need to call it once per document with the correct `figures_dir`:

```python
migrate_figures(cfg, page_ids=doc.page_ids, flat=(doc.doc_type == "epub"))
```

Or refactor to accept a source directory directly:
```python
migrate_figures_to_vault(src_dir: Path, vault_root: Path, vault_figures_dir: str)
```

### `src/pdf.py`

`process_pdf` currently renders to `cfg.temp_dir` and returns `(page_id, temp_img_path)`.  
It should accept an optional `temp_dir` override per document to avoid collisions when multiple PDFs are rendered concurrently.

---

## 10. Edge Cases & Risks

| Risk | Mitigation |
|------|-----------|
| Huge folder (1000+ images + several PDFs) | The single `ThreadPoolExecutor` handles all pages. Memory pressure comes from temp images; render PDFs sequentially. |
| Name collision: two PDFs with same stem | `build_documents` appends a counter: `memoire.pdf` + `memoire.epub` → `book-memoire.pdf`, `book-memoire.epub` |
| Resume with changed `--out` | Resume keys on `parts/<doc_stem>/`. Changing `--out` does not affect resume. |
| EPUB extracted twice (resume) | Check if `output_md` exists and is newer than the EPUB file. |
| `--rename` with multi-doc | Renaming applies only to loose images, not to PDF/EPUB files. Behaviour unchanged. |

---

## 11. Acceptance Criteria

- [ ] `python main.py --images mixed_folder/` with `a.pdf`, `b.epub`, `page_1.jpg` produces `book-a.md`, `book-b.md`, `book-images.md`
- [ ] VLM is loaded exactly once (one block of server logs at start)
- [ ] `--unify` produces a single `book.md`
- [ ] Figures from different documents do not collide
- [ ] `--mode obsidian` migrates figures per document
- [ ] `--resume` skips documents whose `parts/<stem>/` are complete
- [ ] Backward compatibility: a folder with only images behaves exactly as before

---

## 12. Future Enhancements (out of scope)

- Parallel EPUB extraction (currently sequential, but fast)
- Document-specific `--method` (`--method pdf:text` vs `--method pdf:paddleocrvl`)
- Nested folders as separate documents
