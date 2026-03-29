"""
pipeline.py — Orchestration du pipeline OCR complet
"""

import logging
import time
from pathlib import Path

from config import Config
from server import nexa_server
from ocr_client import ocr_image, OCRError
from postprocess import clean_page, format_page_block, format_error_block, extract_done_pages
from images import collect_images
from progress import Stats

logger = logging.getLogger(__name__)


def run_pipeline(cfg: Config) -> Stats:
    """
    Lance le pipeline complet :
      1. Collecte les images
      2. Démarre le serveur Nexa
      3. Traite chaque image (avec reprise si cfg.resume)
      4. Écrit le Markdown au fur et à mesure
      5. Retourne les statistiques

    Le fichier de sortie est écrit incrémentalement :
    chaque page est flushée immédiatement → aucune perte en cas de crash.
    """
    images = collect_images(cfg)

    # ── Reprise ──────────────────────────────────────────────────────────────
    done_pages: set[str] = set()
    if cfg.resume and cfg.output_path.exists():
        existing = cfg.output_path.read_text(encoding="utf-8")
        done_pages = extract_done_pages(existing)
        if done_pages:
            logger.info("Reprise : %d page(s) déjà traitée(s).", len(done_pages))

    stats = Stats(total=len(images))

    # ── Pipeline ─────────────────────────────────────────────────────────────
    with nexa_server(cfg):
        mode = "a" if (cfg.resume and cfg.output_path.exists()) else "w"
        with cfg.output_path.open(mode, encoding="utf-8") as out:

            if mode == "w":
                # En-tête Markdown minimal
                out.write(f"# Livre OCR\n\n")
                out.write(f"<!-- Généré avec DeepSeek-OCR via Nexa SDK -->\n")
                out.flush()

            for idx, img_path in enumerate(images, 1):
                page_id = img_path.stem

                # ── Déjà traitée ? ───────────────────────────────────────────
                if page_id in done_pages:
                    stats.record_skip()
                    logger.debug("[%d/%d] %s — skip", idx, len(images), img_path.name)
                    continue

                # ── OCR ──────────────────────────────────────────────────────
                t0 = time.time()
                try:
                    raw_text = ocr_image(img_path, cfg)
                    clean_text = clean_page(raw_text, cfg)
                    elapsed = time.time() - t0

                    out.write(format_page_block(page_id, clean_text))
                    out.flush()

                    stats.record_success(elapsed, len(clean_text))
                    stats.log_page(idx, img_path.name, elapsed, len(clean_text))

                except OCRError as e:
                    elapsed = time.time() - t0
                    logger.error("[%d/%d] %s — ERREUR (%.1fs) : %s",
                                 idx, len(images), img_path.name, elapsed, e)
                    out.write(format_error_block(page_id, str(e)))
                    out.flush()
                    stats.record_error()

    stats.log_summary()
    logger.info("Fichier de sortie : %s", cfg.output_path.resolve())
    return stats
