"""
pipeline.py — Orchestration du pipeline OCR complet
"""

import logging
import time
from pathlib import Path

from nexaai import VLM

from config import Config
from ocr_client import ocr_image, OCRError
from preprocess import preprocess_image, sauvola_binarize
from figure import process_figures
from postprocess import clean_page, format_page_block, format_error_block, extract_done_pages
from images import collect_images
from progress import Stats

logger = logging.getLogger(__name__)


def run_pipeline(cfg: Config) -> Stats:
    """
    Lance le pipeline complet :
      1. Collecte les images
      2. Charge le VLM (une seule fois)
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

    # ── Chargement du VLM ────────────────────────────────────────────────────
    logger.info("Chargement du modèle %s ...", cfg.model)
    vlm = VLM.from_(model=cfg.model, quant=cfg.quant, config=cfg.to_model_config())
    logger.info("Modèle chargé.")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    output_is_new = not (cfg.resume and cfg.output_path.exists())
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if not output_is_new else "w"
    with cfg.output_path.open(mode, encoding="utf-8") as out:

        if mode == "w":
            out.write("# Livre OCR\n\n")
            out.write("<!-- Généré avec DeepSeek-OCR via Nexa SDK -->\n")
            out.flush()

        for idx, img_path in enumerate(images, 1):
            page_id = img_path.stem

            # ── Déjà traitée ? ───────────────────────────────────────────
            if page_id in done_pages:
                stats.record_skip()
                logger.debug("[%d/%d] %s — skip", idx, len(images), img_path.name)
                continue

            # ── OCR ──────────────────────────────────────────────────────
            # Passe 1 : preprocess(page) → ocr → [passe 2] → postprocess(page)
            # Passe 2 : pour chaque figure détectée dans le résultat layout :
            #           crop(original) → preprocess(crop) → ocr → postprocess(crop)
            #           (orchestré dans figure.process_figures)
            t0 = time.time()
            try:
                if cfg.preprocess_mode == "binarize":
                    preprocessed_path = preprocess_image(
                        img_path, cfg.binarize_block_size, cfg.binarize_c,
                        cfg.blur_ksize, cfg.blur_sigma,
                    )
                elif cfg.preprocess_mode == "sauvola":
                    preprocessed_path = sauvola_binarize(
                        img_path, cfg.sauvola_window_size, cfg.sauvola_k,
                    )
                else:
                    preprocessed_path = img_path
                raw_text, metrics = ocr_image(preprocessed_path, vlm, cfg)

                if cfg.prompt_mode == "layout" and cfg.two_pass:
                    raw_text, fig_metrics = process_figures(
                        raw_text, img_path, vlm, cfg, page_id
                    )
                    metrics["total_latency"] += fig_metrics["total_latency"]

                clean_text = clean_page(raw_text, cfg)
                elapsed = time.time() - t0

                out.write(format_page_block(page_id, clean_text))
                out.flush()

                stats.record_success(elapsed, len(clean_text), metrics["total_latency"])
                stats.log_page(idx, img_path.name, elapsed, len(clean_text))

            except OCRError as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s — ERREUR (%.1fs) : %s",
                             idx, len(images), img_path.name, elapsed, e)
                out.write(format_error_block(page_id, str(e)))
                out.flush()
                stats.record_error()

    stats.log_summary()

    if output_is_new and stats.done == 0 and cfg.output_path.exists():
        cfg.output_path.unlink()
        logger.warning("Aucune page traitée avec succès — fichier de sortie supprimé.")
    else:
        logger.info("Fichier de sortie : %s", cfg.output_path.resolve())

    return stats
