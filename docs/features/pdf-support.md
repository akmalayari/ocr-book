# Implementation Plan — PDF Support

## Goal

Accept `.pdf` files as input alongside images. Automatically detect whether the PDF is **text-based** (native text layer) or **image-based** (scanned), then process each page via the appropriate path.

- **Text-based**: extract text natively with `pymupdf`, detect figures with `PP-DocLayoutV3`, crop figures from rendered pages. No VLM OCR.
- **Image-based**: render pages to images, feed to existing `PaddleOCRVL` pipeline.

The end result is still a single Markdown file, with a unified page sequence regardless of source.

---

## Architecture

### Core principle: unified page list

The pipeline must treat every page — whether from a photo, a text-based PDF, or an image-based PDF — as an entry in a single ordered list. The final assembly iterates over this list, not over the image file list.

```
run_pipeline(cfg)
  ├── _collect_sources(cfg)               # images + PDFs, naturally sorted
  │     └── returns list[Path]
  │
  ├── pdf.process_pdf(src, ...) for each PDF
  │     ├── classify_pdf(doc) -> "text" | "image"
  │     ├── if text-based:
  │     │     └── for each page:
  │     │           ├── if done: skip
  │     │           ├── extract text + detect figures
  │     │           ├── write output/parts/{page_id}.part
  │     │           └── return (page_id, None)
  │     └── if image-based:
  │           └── for each page:
  │                 ├── if done: skip rendering
  │                 ├── render -> output/temp/{page_id}.png
  │                 └── return (page_id, temp_path)
  │
  ├── Build unified lists:
  │     all_page_ids  = [page_id, ...]    # every page in read order
  │     ocr_queue     = [(page_id, image_path), ...]  # pages needing VLM
  │
  ├── if ocr_queue:
  │     ├── start llama-servers
  │     ├── ThreadPoolExecutor -> ocr_image()
  │     └── write .part files
  │
  ├── Combine parts in all_page_ids order
  ├── postprocess, header detection, obsidian export
  └── cleanup temp images
```

### New module: `src/pdf.py`

Responsibilities:
1. PDF classification (text-based vs image-based)
2. Text extraction and figure detection for text-based PDFs
3. Page rendering for image-based PDFs
4. Return per-page metadata compatible with the unified page list

---

## Detailed Design

### 1. Source discovery (`images._collect_sources()`)

```python
def _collect_sources(cfg: Config) -> list[Path]:
    """
    Returns all processable files in cfg.images_path:
    images (by extension) + .pdf files, naturally sorted.
    """
    path = cfg.images_path
    if not path.exists():
        raise ImageCollectionError(f"Path not found: {path}")

    extensions = cfg.extensions + (".pdf",)
    files = [
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return natsorted(files, key=lambda p: p.name)
```

`collect_images()` is kept unchanged for `--rename` and utility modes. `pipeline.py` calls `_collect_sources()` instead.

### 2. PDF entry point (`pdf.process_pdf()`)

```python
def process_pdf(
    pdf_path: Path,
    cfg: Config,
    done_pages: set[str],
    parts_dir: Path,
) -> list[tuple[str, Path | None]]:
    """
    Processes a single PDF.

    Returns a list of (page_id, temp_image_path) in page order.
    - text-based page  -> (page_id, None)
    - image-based page -> (page_id, temp_image_path)

    Already-done pages are skipped (no rendering, no extraction).
    """
```

**Page ID convention:** `f"{pdf_stem}_p{page_num:03d}"` (e.g. `book_p001`).
- Deterministic, sort-safe, and won't collide with manually-named images unless the user deliberately names an image `book_p001.jpg`.
- No attempt to share a `page_NNN` namespace with photos; the natural sort order of `_collect_sources()` determines placement.

### 3. PDF Classification (`pdf.classify_pdf()`)

Heuristic on first 3 pages:

```python
def classify_pdf(doc: fitz.Document, threshold: float = 0.001) -> Literal["text", "image"]:
    """Classify PDF as text-based or image-based."""
    text_pages = 0
    for page in doc[:3]:
        text = page.get_text()
        density = len(text.strip()) / (page.rect.width * page.rect.height)
        if density > threshold:
            text_pages += 1
    return "text" if text_pages >= 2 else "image"
```

**Edge cases:**
- Mixed PDF (cover images, then text): conservative — classified as image-based if fewer than 2 text-dense pages.
- Fake text layer (scanned PDFs with garbage text): future improvement — check average word length.
- Password-protected PDF: catch exception, log error, skip file.
- `cfg.pdf_force_ocr`: if True, bypass classification and treat as image-based.

### 4. Text-Based PDF Path

#### 4a. Text extraction

```python
def extract_page_text(page: fitz.Page) -> str:
    """Extract text preserving some structure."""
    return page.get_text("text")
```

**Decision:** start with `"text"` + manual paragraph breaks from block bboxes. If `pymupdf`'s `"markdown"` output proves sufficient on test files, switch to it.

#### 4b. Page rendering for layout detection

```python
def render_page(page: fitz.Page, dpi: int = 200) -> Path:
    pix = page.get_pixmap(dpi=dpi)
    path = cfg.temp_dir / f"{page_id}_layout.png"
    pix.save(path)
    return path
```

Rendered once per page for `PP-DocLayoutV3`. The render file may be deleted after figure detection.

#### 4c. Layout detection for figures

```python
from paddlex import create_model

FIGURE_LABELS = {"image", "chart", "header_image", "footer_image", "table"}
_layout_model = None  # lazy singleton — loaded on first text-based page

def _get_layout_model():
    global _layout_model
    if _layout_model is None:
        _layout_model = create_model(model_name="PP-DocLayoutV3")
    return _layout_model

def detect_figures(image_path: Path) -> list[dict]:
    model = _get_layout_model()
    figures = []
    for result in model.predict(str(image_path)):
        for box in result.json["res"]["boxes"]:
            if box["label"] in FIGURE_LABELS and box["score"] > 0.5:
                figures.append(box)
    return figures
```

Tables are included in `FIGURE_LABELS` and rasterized as images (Option B).

#### 4d. Figure cropping

PP-DocLayoutV3 returns bboxes in **pixel coordinates** at the DPI used for rendering (default 200). `fitz.Rect` expects **PDF points** (72 DPI). Convert before cropping:

```python
def crop_figure(page: fitz.Page, bbox: list[float], render_dpi: int, crop_dpi: int) -> fitz.Pixmap:
    """Crop figure from PDF page. bbox is in pixels at render_dpi."""
    scale = 72.0 / render_dpi
    rect = fitz.Rect(
        bbox[0] * scale, bbox[1] * scale,
        bbox[2] * scale, bbox[3] * scale,
    )
    return page.get_pixmap(clip=rect, dpi=crop_dpi)
```

Cropped figures are saved to `output/figures/{page_id}/imgs/figure_{n}.png`.

#### 4e. Figure insertion into text stream

**Decision (v1):** append figures at the end of the page text.

The text-based path emits the **same relative image format** that PaddleOCR produces:

```html
<img src="imgs/figure_0.png" />
```

This ensures `fix_image_paths()` and `fix_image_paths_obsidian()` work without changes. After `fix_image_paths` runs, the path becomes `figures/{page_id}/imgs/figure_0.png` in base mode, or `![[vault_figures_dir/figure_0.png]]` in Obsidian mode.

#### 4f. Part file writing and page number detection

```python
def write_text_based_part(page_id: str, text: str, cfg: Config, parts_dir: Path):
    page_number, cleaned_text = extract_page_number(text)
    if not page_number:
        # Fall back to the PDF internal page number
        page_number = str(int(page_id.rsplit("_p", 1)[-1]))
    formatted = format_page_block(page_id, cleaned_text, page_number)
    part_path = parts_dir / f"{page_id}.part"
    part_path.write_text(formatted, encoding="utf-8")
```

The printed page number is detected from the extracted text using the same `extract_page_number()` logic used for OCR output. If no printed number is found, the fallback is the PDF's sequential page number (`p001` -> `1`).

**Risk:** `extract_page_number()` was written for VLM OCR output and may not match pymupdf's text formatting (different whitespace, line endings). Verify on a real text-based PDF before relying on this; the sequential fallback covers the failure case.

### 5. Image-Based PDF Path

Render each page to a temporary image, then let the existing pipeline process it:

```python
def convert_pdf_page(page: fitz.Page, page_id: str, cfg: Config) -> Path:
    pix = page.get_pixmap(dpi=cfg.pdf_dpi)
    path = cfg.temp_dir / f"{page_id}.png"
    pix.save(path)
    return path
```

These images are added to `ocr_queue` and processed through `ocr_client.ocr_image()` normally.

### 6. Pipeline integration (`src/pipeline.py`)

```python
def run_pipeline(cfg: Config) -> Stats:
    parts_dir = Path(cfg.log_file).parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # -- Resume state --
    done_pages: set[str] = set()
    if cfg.resume:
        done_pages = {p.stem for p in parts_dir.glob("*.part")}
        if done_pages:
            logger.info("Resume: %d page(s) already processed.", len(done_pages))
    else:
        for p in parts_dir.glob("*.part"):
            p.unlink()

    # -- Discover all sources --
    sources = _collect_sources(cfg)

    # -- PDF preprocessing (before servers) --
    all_page_ids: list[str] = []
    ocr_queue: list[tuple[str, Path]] = []

    for src in sources:
        if src.suffix.lower() == ".pdf":
            pdf_pages = pdf.process_pdf(src, cfg, done_pages, parts_dir)
            for page_id, temp_img in pdf_pages:
                all_page_ids.append(page_id)
                if temp_img:  # None for text-based or already-done image-based pages
                    ocr_queue.append((page_id, temp_img))
        else:
            page_id = src.stem
            all_page_ids.append(page_id)
            if page_id not in done_pages:
                ocr_queue.append((page_id, src))

    stats = Stats(total=len(all_page_ids))
    stats.skipped = sum(1 for pid in all_page_ids if pid in done_pages)

    # -- Server startup (conditional) --
    # NOTE: this requires extracting the server startup block (currently inline)
    # into a helper or moving it inside this branch. Non-trivial refactor —
    # the thread pool, server list, and process_page closure all reference
    # each other. Plan this carefully before touching pipeline.py.
    if ocr_queue:
        # ... existing server startup code ...
        # process_page now receives (idx, page_id, img_path)
        pass
    else:
        logger.info("No pages require OCR — skipping server startup.")

    # -- Combine in all_page_ids order --
    with cfg.output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# OCR Book\n\n")
        out.write("<!-- Generated with PaddleOCR-VL-1.5 via llama-server -->\n")
        for page_id in all_page_ids:
            part = parts_dir / f"{page_id}.part"
            if part.exists():
                out.write(part.read_text(encoding="utf-8"))

    # -- Header detection, stats, obsidian --
    if cfg.header_patterns:
        # ... existing header detection ...
        pass

    stats.log_summary()
    stats.write_report(Path(cfg.report_file), cfg)

    if cfg.mode == "obsidian":
        from obsidian import migrate_figures
        migrate_figures(cfg, page_ids=all_page_ids)

    # -- Cleanup --
    if not cfg.verbose:
        for p in cfg.temp_dir.glob("*.png"):
            p.unlink(missing_ok=True)

    return stats
```

### 7. `process_page()` signature change

```python
def process_page(idx: int, page_id: str, img_path: Path) -> dict:
    # ... existing logic ...
    part_path = parts_dir / f"{page_id}.part"
    # ...
```

This is a minimal change that makes the pipeline source-agnostic.

---

## CLI / Config Changes

No new CLI arguments required for basic support. `.pdf` files are detected automatically in `--images`.

**Optional future arguments:**
- `--pdf-force-ocr` — treat all PDFs as image-based
- `--pdf-dpi` — rendering DPI (default 200)
- `--pdf-min-density` — text density threshold for classification

**Config additions (`src/config.py`):**

```python
@dataclass
class Config:
    # ... existing ...
    pdf_dpi: int = 200
    pdf_text_density_threshold: float = 0.001
    pdf_force_ocr: bool = False
    temp_dir: Path = field(default_factory=lambda: Path("output/temp"))
```

---

## Resume Compatibility

The existing resume system works with minor changes:
- `done_pages` is computed **before** PDF preprocessing and passed into `pdf.process_pdf()`.
- Text-based PDF pages already done are skipped entirely (no extraction, no layout detection).
- Image-based PDF pages already done are skipped for rendering and OCR, but their `page_id` is still added to `all_page_ids` so the final assembly includes them.
- On restart, existing `.part` files are skipped regardless of source.

---

## Obsidian / Figure Compatibility

Figures from text-based PDFs are saved to `output/figures/{page_id}/imgs/`, exactly the same structure as the OCR pipeline. Because the text-based path emits `src="imgs/figure_0.png"`:
- `fix_image_paths()` works without changes
- `fix_image_paths_obsidian()` works without changes
- `postprocess.strip_table_styles()` works without changes (tables are images, not HTML tables)
- `obsidian.migrate_figures(cfg, page_ids=all_page_ids)` copies figures for every page

---

## File Changes

| File | Change |
|---|---|
| `src/pdf.py` | **New module** — classification, extraction, rendering, layout detection, figure cropping, part writing |
| `src/config.py` | Add `pdf_dpi`, `pdf_text_density_threshold`, `pdf_force_ocr`, `temp_dir` |
| `src/images.py` | Add `_collect_sources()`; no changes to `collect_images()` |
| `src/pipeline.py` | Use `_collect_sources()`; call `pdf.process_pdf()` before server startup; build `all_page_ids` + `ocr_queue`; conditional server startup; assembly over `all_page_ids`; pass `page_ids=all_page_ids` to `migrate_figures`; temp cleanup |
| `src/obsidian.py` | Add `page_ids` parameter to `migrate_figures()` so it can iterate `all_page_ids` instead of re-deriving from image files |
| `environment.yml` | Add `pymupdf` explicitly |
| `docs/features/pdf-support.md` | **This document** |

---

## Dependencies

- `pymupdf` (fitz) — already available via `paddleocr`/`paddlex` transitive deps, but must be explicit in `environment.yml`
- `PP-DocLayoutV3` — downloaded automatically by `paddlex.create_model()` on first use

---

## Testing Plan

| Test | Input | Expected Result |
|---|---|---|
| Text-based magazine PDF | The Economist issue | Fast text extraction + figure crops, no VLM calls |
| Image-based scanned book | 10-page scanned PDF | Rendered images -> normal OCR pipeline, quality comparable to photos |
| Mixed-quality PDF | First 2 pages scanned, rest text | Classified as image-based (conservative) or text-based depending on density |
| Resume | Run 5 pages, kill, restart | Skips existing `.part` files, processes remaining, output is complete |
| Text-only input | 100% text-based PDF | No llama-server startup, finishes quickly |
| Figure quality comparison | Same page as photo vs PDF render | Vector figures from PDF should be sharper than photo OCR crops |
| Obsidian export | Text-based PDF with figures | Figures migrated correctly, wikilinks resolve |
| No figures | Text-based PDF with 0 detected figures | Clean `.part` file, no broken `<img>` tags in output |

---

## Open Questions / Decisions

### 1. Table handling in text-based PDFs

**Decision:** rasterize tables as images (Option B). Tables are treated the same as figures: detected by `PP-DocLayoutV3`, cropped from the PDF page, saved as PNG, appended at the end of the page text.

**Rationale:** simple, reuses existing infrastructure. Acceptable for magazines and simple documents. If table structure is critical, the user can use `--pdf-force-ocr` to get full VLM pipeline behavior.

**Future:** try `pymupdf` table extraction (`page.find_tables()`) as a hybrid fallback.

### 2. Figure insertion position

**Decision:** append figures at the end of the page text. No reading-order logic in v1.

**Rationale:** simple, robust, avoids misplacement. For magazine layouts where figures are often full-bleed and captions are separate text blocks, exact reading order is ambiguous anyway.

### 3. Page numbering for text-based PDFs

**Decision:** detect printed page numbers from extracted text using the existing `extract_page_number()` function. If no printed number is found, fall back to the PDF's internal sequential page number (`p001` -> `1`).

**Rationale:** consistent with the OCR path, preserves the existing `format_page_block()` behavior, and produces meaningful page references in the Markdown comments.
