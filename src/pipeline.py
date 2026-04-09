"""
pipeline.py — Orchestration du pipeline OCR complet
"""

import logging
import os
import subprocess
import time
from pathlib import Path

import requests
from paddleocr import PaddleOCRVL

from config import Config
from ocr_client import ocr_image, OCRError
from postprocess import clean_page, format_page_block, format_error_block, extract_done_pages, fix_image_paths
from images import collect_images
from progress import Stats

logger = logging.getLogger(__name__)


def _start_server(cfg: Config) -> subprocess.Popen:
    cmd = [
        cfg.llama_server_path,
        "-m",       cfg.model_path,
        "--mmproj", cfg.mmproj_path,
        "--port",   str(cfg.server_port),
        "--host",   "127.0.0.1",
        "-c",       str(cfg.n_ctx),
        "-ngl",     str(cfg.n_gpu_layers),
        "-b",       str(cfg.n_batch),
        "-ub",      str(cfg.n_ubatch),
        "-t",       str(cfg.n_threads),
        "--prio",   str(cfg.prio),
        "--temp",   str(cfg.temperature),
        "-np",      "1",
    ]
    if cfg.kv_offload:
        cmd += ["-kvo"]
    logger.info("Démarrage llama-server...")
    return subprocess.Popen(cmd)  # DEBUG: logs visibles


def _wait_for_server(cfg: Config) -> bool:
    deadline = time.time() + cfg.server_timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{cfg.server_url}/health", timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def run_pipeline(cfg: Config) -> Stats:
    """
    Lance le pipeline complet :
      1. Collecte les images
      2. Démarre llama-server
      3. Instancie PaddleOCRVL
      4. Traite chaque image (avec reprise si cfg.resume)
      5. Écrit le Markdown au fur et à mesure
      6. Retourne les statistiques
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

    # ── Démarrage llama-server ────────────────────────────────────────────────
    logger.info("Démarrage llama-server...")
    t_load0 = time.time()
    proc = _start_server(cfg)
    if not _wait_for_server(cfg):
        proc.kill()
        raise RuntimeError(f"llama-server n'a pas démarré dans les délais ({cfg.server_timeout}s).")
    stats.model_load_time = time.time() - t_load0
    logger.info("Serveur prêt en %.1fs.", stats.model_load_time)

    _vlm_kwargs = dict(
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=f"{cfg.server_url}/v1",
        vl_rec_api_model_name="paddleocr",
    )
    pipeline          = PaddleOCRVL(**_vlm_kwargs)
    pipeline_fallback = PaddleOCRVL(use_layout_detection=False, **_vlm_kwargs)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    output_is_new = not (cfg.resume and cfg.output_path.exists())
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_rel = os.path.relpath(cfg.figures_path, cfg.output_path.parent)

    mode = "a" if not output_is_new else "w"
    try:
        with cfg.output_path.open(mode, encoding="utf-8") as out:

            if mode == "w":
                out.write("# Livre OCR\n\n")
                out.write("<!-- Généré avec PaddleOCR-VL-1.5 via llama-server -->\n")
                out.flush()

            for idx, img_path in enumerate(images, 1):
                page_id = img_path.stem

                # ── Déjà traitée ? ───────────────────────────────────────
                if page_id in done_pages:
                    stats.record_skip()
                    logger.debug("[%d/%d] %s — skip", idx, len(images), img_path.name)
                    continue

                # ── OCR ──────────────────────────────────────────────────
                t0 = time.time()
                try:
                    raw_text, metrics = ocr_image(img_path, pipeline, cfg)
                    t_ocr = metrics["total_latency"]
                except OCRError:
                    logger.warning("%s — layout failed, tentative fallback...", img_path.name)
                    try:
                        raw_text, metrics = ocr_image(img_path, pipeline_fallback, cfg)
                        t_ocr = metrics["total_latency"]
                    except OCRError as e:
                        elapsed = time.time() - t0
                        logger.error("[%d/%d] %s — ERREUR (%.1fs) : %s",
                                     idx, len(images), img_path.name, elapsed, e)
                        out.write(format_error_block(page_id, str(e)))
                        out.flush()
                        stats.record_error(page_name=img_path.name)
                        continue

                # ── Post-traitement + écriture ────────────────────────────
                t_post0 = time.time()
                clean_text = clean_page(raw_text, cfg) if cfg.postprocess else raw_text
                clean_text = fix_image_paths(clean_text, page_id, figures_rel)
                out.write(format_page_block(page_id, clean_text))
                out.flush()
                t_post = time.time() - t_post0

                elapsed = time.time() - t0
                stats.record_success(
                    elapsed, len(clean_text), t_ocr,
                    t_pre=0.0, t_ocr=t_ocr, t_post=t_post,
                    looped=False,
                    page_name=img_path.name,
                )
                stats.log_page(idx, img_path.name, elapsed, len(clean_text))

    finally:
        proc.kill()
        proc.wait()
        logger.info("llama-server arrêté.")

    stats.log_summary()
    stats.write_report(Path(cfg.report_file), cfg)

    if output_is_new and stats.done == 0 and cfg.output_path.exists():
        cfg.output_path.unlink()
        logger.warning("Aucune page traitée avec succès — fichier de sortie supprimé.")
    else:
        logger.info("Fichier de sortie : %s", cfg.output_path.resolve())

    return stats
