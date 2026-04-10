"""
pipeline.py — Orchestration du pipeline OCR complet
"""

import logging
import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from paddleocr import PaddleOCRVL

from config import Config
from ocr_client import ocr_image, OCRError
from postprocess import clean_page, strip_table_styles, format_page_block, format_error_block, fix_image_paths, extract_page_number
from obsidian import fix_image_paths_obsidian
from images import collect_images
from progress import Stats

logger = logging.getLogger(__name__)


def _start_server(cfg: Config, port: int) -> subprocess.Popen:
    cmd = [
        cfg.llama_server_path,
        "-m",       cfg.model_path,
        "--mmproj", cfg.mmproj_path,
        "--port",   str(port),
        "--host",   "127.0.0.1",
        "-c",       str(cfg.n_ctx),
        "-ngl",     str(cfg.n_gpu_layers),
        "-b",       str(cfg.n_batch),
        "-ub",      str(cfg.n_ubatch),
        "-t",       str(cfg.n_threads),
        "--prio",   str(cfg.prio),
        "--temp",   str(cfg.temperature),
        "-np",      str(cfg.n_parallel),
    ]
    if cfg.kv_offload:
        cmd += ["-kvo"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_server(url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def run_pipeline(cfg: Config) -> Stats:
    """
    Lance le pipeline complet :
      1. Collecte les images
      2. Démarre n_servers llama-server en parallèle
      3. Instancie n_servers PaddleOCRVL
      4. Traite les pages en parallèle (une page par serveur)
      5. Écrit chaque page dans output/parts/<page_id>.part
      6. Combine les parts dans l'ordre en fin de run
      7. Retourne les statistiques
    """
    images = collect_images(cfg)

    # ── Parts dir ────────────────────────────────────────────────────────────
    parts_dir = cfg.output_path.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # ── Reprise ──────────────────────────────────────────────────────────────
    done_pages: set[str] = set()
    if cfg.resume:
        done_pages = {p.stem for p in parts_dir.glob("*.part")}
        if done_pages:
            logger.info("Reprise : %d page(s) déjà traitée(s).", len(done_pages))
    else:
        for p in parts_dir.glob("*.part"):
            p.unlink()

    stats = Stats(total=len(images))
    stats.skipped = sum(1 for img in images if img.stem in done_pages)

    # ── Démarrage des serveurs ────────────────────────────────────────────────
    ports = [cfg.server_base_port + i for i in range(cfg.n_servers)]
    urls  = [f"http://127.0.0.1:{port}" for port in ports]

    logger.info("Démarrage de %d llama-server...", cfg.n_servers)
    t_load0 = time.time()
    procs = [_start_server(cfg, port) for port in ports]

    for url in urls:
        if not _wait_for_server(url, cfg.server_timeout):
            for proc in procs:
                proc.kill()
            raise RuntimeError(
                f"llama-server ({url}) n'a pas démarré dans les délais ({cfg.server_timeout}s)."
            )

    stats.model_load_time = time.time() - t_load0
    logger.info("Serveurs prêts en %.1fs.", stats.model_load_time)

    # ── Instanciation des pipelines ───────────────────────────────────────────
    _vlm_kwargs_base = dict(
        vl_rec_backend="llama-cpp-server",
        vl_rec_api_model_name="paddleocr",
        use_layout_detection=cfg.use_layout_detection,
        markdown_ignore_labels=["header_image", "footer", "footer_image"],
    )
    pipelines = [
        PaddleOCRVL(vl_rec_server_url=f"{url}/v1", **_vlm_kwargs_base)
        for url in urls
    ]

    pipeline_queue: queue.Queue = queue.Queue()
    for pl in pipelines:
        pipeline_queue.put(pl)

    # ── Traitement parallèle ──────────────────────────────────────────────────
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_rel = os.path.relpath(cfg.figures_path, cfg.output_path.parent)
    to_process = [(idx, img) for idx, img in enumerate(images, 1)
                  if img.stem not in done_pages]

    def process_page(idx: int, img_path: Path) -> dict:
        page_id = img_path.stem
        pl = pipeline_queue.get()
        t0 = time.time()
        try:
            raw_text, metrics = ocr_image(img_path, pl, cfg)
            t_ocr = metrics["total_latency"]

            t_post0 = time.time()
            page_number, raw_text = extract_page_number(raw_text)
            clean_text = clean_page(raw_text, cfg) if cfg.postprocess else raw_text
            clean_text = strip_table_styles(clean_text)
            if cfg.mode == "obsidian":
                clean_text = fix_image_paths_obsidian(clean_text, cfg.vault_figures_dir)
            else:
                clean_text = fix_image_paths(clean_text, page_id, figures_rel)
            t_post = time.time() - t_post0

            elapsed = time.time() - t0
            part_path = parts_dir / f"{page_id}.part"
            with part_path.open("a", encoding="utf-8") as f:
                f.write(format_page_block(page_id, clean_text, page_number))
            return {
                "idx": idx, "page_name": img_path.name,
                "elapsed": elapsed, "chars": len(clean_text),
                "t_ocr": t_ocr, "t_post": t_post, "error": False,
            }
        except OCRError as e:
            elapsed = time.time() - t0
            logger.error("[%d/%d] %s — ERREUR (%.1fs) : %s",
                         idx, len(images), img_path.name, elapsed, e)
            part_path = parts_dir / f"{page_id}.part"
            with part_path.open("a", encoding="utf-8") as f:
                f.write(format_error_block(page_id, str(e)))
            return {"idx": idx, "page_name": img_path.name, "error": True}
        finally:
            pipeline_queue.put(pl)

    try:
        with ThreadPoolExecutor(max_workers=cfg.n_servers) as executor:
            futures = [executor.submit(process_page, idx, img) for idx, img in to_process]
            for future in as_completed(futures):
                result = future.result()
                if result["error"]:
                    stats.record_error(page_name=result["page_name"])
                else:
                    stats.record_success(
                        result["elapsed"], result["chars"],
                        latency=result["t_ocr"],
                        t_ocr=result["t_ocr"], t_post=result["t_post"],
                        page_name=result["page_name"],
                    )
                    stats.log_page(
                        result["idx"], result["page_name"],
                        result["elapsed"], result["chars"],
                    )
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
        logger.info("%d serveur(s) arrêté(s).", cfg.n_servers)

    # ── Combinaison dans l'ordre d'entrée ─────────────────────────────────────
    with cfg.output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Livre OCR\n\n")
        out.write("<!-- Généré avec PaddleOCR-VL-1.5 via llama-server -->\n")
        for img_path in images:
            part = parts_dir / f"{img_path.stem}.part"
            if part.exists():
                out.write(part.read_text(encoding="utf-8"))

    stats.log_summary()
    stats.write_report(Path(cfg.report_file), cfg)

    if stats.done == 0 and stats.skipped == 0 and cfg.output_path.exists():
        cfg.output_path.unlink()
        logger.warning("Aucune page traitée avec succès — fichier de sortie supprimé.")
    else:
        logger.info("Fichier de sortie : %s", cfg.output_path.resolve())

    return stats
