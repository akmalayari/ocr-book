"""
pdf.py — PDF processing: classification, text extraction, rendering, figure detection
"""

import logging
import time
from pathlib import Path

import fitz

from config import Config
from postprocess import extract_page_number, format_page_block, format_error_block
from progress import Stats

logger = logging.getLogger(__name__)

FIGURE_LABELS = {"image", "chart", "header_image", "footer_image", "table"}
_layout_model = None


def _get_layout_model():
    global _layout_model
    if _layout_model is None:
        from paddlex import create_model
        logger.info("Loading PP-DocLayoutV3 layout model...")
        _layout_model = create_model(model_name="PP-DocLayoutV3")
    return _layout_model


def classify_pdf(doc: fitz.Document, threshold: float = 0.001) -> str:
    """Classify PDF as text-based or image-based."""
    text_pages = 0
    for page in doc[:3]:
        text = page.get_text()
        area = page.rect.width * page.rect.height
        if area == 0:
            continue
        density = len(text.strip()) / area
        if density > threshold:
            text_pages += 1
    return "text" if text_pages >= 2 else "image"


def _render_page(page: fitz.Page, page_id: str, temp_dir: Path, dpi: int = 200) -> Path:
    pix = page.get_pixmap(dpi=dpi)
    path = temp_dir / f"{page_id}_layout.png"
    pix.save(str(path))
    return path


def _detect_figures(image_path: Path) -> list[dict]:
    model = _get_layout_model()
    figures = []
    result = model.predict(str(image_path))
    if not isinstance(result, (list, tuple)):
        result = [result]
    for r in result:
        json_data = getattr(r, "json", None)
        if json_data is None:
            continue
        res = json_data.get("res", {})
        boxes = res.get("boxes", [])
        for box in boxes:
            if box.get("label") in FIGURE_LABELS and box.get("score", 0) > 0.5:
                figures.append(box)
    return figures


def _crop_figure(page: fitz.Page, bbox: list[float], render_dpi: int, crop_dpi: int) -> fitz.Pixmap:
    """Crop figure from PDF page. bbox is in pixels at render_dpi."""
    scale = 72.0 / render_dpi
    rect = fitz.Rect(
        bbox[0] * scale, bbox[1] * scale,
        bbox[2] * scale, bbox[3] * scale,
    )
    return page.get_pixmap(clip=rect, dpi=crop_dpi)


def process_pdf(
    pdf_path: Path,
    cfg: Config,
    done_pages: set[str],
    parts_dir: Path,
    stats: Stats | None = None,
) -> list[tuple[str, Path | None]]:
    """
    Processes a single PDF.

    Returns a list of (page_id, temp_image_path) in page order.
    - text-based page  -> (page_id, None)
    - image-based page -> (page_id, temp_image_path)

    Already-done pages are skipped.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error("Failed to open PDF '%s': %s", pdf_path.name, e)
        return []

    pdf_type = "image" if cfg.pdf_force_ocr else classify_pdf(doc, threshold=cfg.pdf_text_density_threshold)
    logger.info("PDF '%s' classified as %s-based (%d pages).", pdf_path.name, pdf_type, len(doc))

    results: list[tuple[str, Path | None]] = []
    render_dpi = cfg.pdf_dpi

    for page_num in range(len(doc)):
        page_id = f"{pdf_path.stem}_p{page_num + 1:03d}"

        if page_id in done_pages:
            results.append((page_id, None))
            continue

        page = doc[page_num]
        t0 = time.time()

        try:
            if pdf_type == "text":
                # Text extraction
                text = page.get_text("text")

                # Render for layout detection
                cfg.temp_dir.mkdir(parents=True, exist_ok=True)
                layout_image = _render_page(page, page_id, cfg.temp_dir, dpi=render_dpi)

                # Detect figures
                try:
                    figures = _detect_figures(layout_image)
                except Exception as e:
                    logger.warning("Layout detection failed for %s: %s", page_id, e)
                    figures = []

                # Clean up layout render
                layout_image.unlink(missing_ok=True)

                # Crop and save figures
                if figures:
                    figure_dir = cfg.figures_path / page_id / "imgs"
                    figure_dir.mkdir(parents=True, exist_ok=True)

                    for i, fig in enumerate(figures):
                        try:
                            bbox = fig.get("bbox")
                            if bbox is None:
                                continue
                            pix = _crop_figure(page, bbox, render_dpi, render_dpi)
                            fig_path = figure_dir / f"figure_{i}.png"
                            pix.save(str(fig_path))
                        except Exception as e:
                            logger.warning("Failed to crop figure %d for %s: %s", i, page_id, e)

                # Build figure tags
                figure_tags = "\n".join(
                    f'<img src="imgs/figure_{i}.png" />'
                    for i in range(len(figures))
                )

                if figure_tags:
                    text = text + "\n\n" + figure_tags

                # Detect page number and write part
                page_number, cleaned_text = extract_page_number(text)
                if not page_number:
                    page_number = str(page_num + 1)
                formatted = format_page_block(page_id, cleaned_text, page_number)
                part_path = parts_dir / f"{page_id}.part"
                part_path.write_text(formatted, encoding="utf-8")

                if stats is not None:
                    elapsed = time.time() - t0
                    stats.record_success(
                        elapsed=elapsed,
                        chars=len(cleaned_text),
                        t_ocr=0.0,
                        t_post=elapsed,
                        page_name=page_id,
                    )

                results.append((page_id, None))

            else:
                # Image-based: render to temp image
                cfg.temp_dir.mkdir(parents=True, exist_ok=True)
                pix = page.get_pixmap(dpi=cfg.pdf_dpi)
                temp_path = cfg.temp_dir / f"{page_id}.png"
                pix.save(str(temp_path))
                results.append((page_id, temp_path))

        except Exception as e:
            logger.error("Failed to process %s: %s", page_id, e)
            part_path = parts_dir / f"{page_id}.part"
            with part_path.open("w", encoding="utf-8") as f:
                f.write(format_error_block(page_id, str(e)))
            if stats is not None:
                stats.record_error(page_name=page_id)
            results.append((page_id, None))

    doc.close()
    return results
