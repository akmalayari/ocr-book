"""
pipeline.py — Orchestration of the complete OCR pipeline
"""

import logging
import os
import queue
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Bypass paddlex online model-source check early (before any paddlex import)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import requests
from paddleocr import PaddleOCRVL

from config import Config
from ocr_client import ocr_image, OCRError, OCRTimeout
from postprocess import clean_page, strip_table_styles, strip_math_spacing, format_page_block, format_error_block, fix_image_paths, extract_page_number, apply_header_detection, html_to_markdown
from obsidian import fix_image_paths_obsidian
from images import _collect_sources
from progress import Stats
import pdf

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


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_pipeline(cfg: Config) -> Stats:
    """
    Runs the complete pipeline:
      1. Collects images and PDFs
      2. Processes PDFs (text extraction or rendering)
      3. Starts n_servers llama-server in parallel (if needed)
      4. Instantiates n_servers PaddleOCRVL
      5. Processes pages in parallel (one page per server)
      6. Writes each page to output/parts/<page_id>.part
      7. Combines parts in order at the end of the run
      8. Returns statistics
    """
    # ── Parts dir ────────────────────────────────────────────────────────────
    # Always in output/parts, even in obsidian mode where output_path points to the vault
    parts_dir = Path(cfg.log_file).parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ───────────────────────────────────────────────────────────────
    done_pages: set[str] = set()
    if cfg.resume:
        done_pages = {p.stem for p in parts_dir.glob("*.part")}
        if done_pages:
            logger.info("Resume: %d page(s) already processed.", len(done_pages))
    else:
        for p in parts_dir.glob("*.part"):
            p.unlink()

    # ── Discover all sources ─────────────────────────────────────────────────
    sources = _collect_sources(cfg)

    # ── PDF preprocessing (before servers) ───────────────────────────────────
    all_page_ids: list[str] = []
    ocr_queue: list[tuple[int, str, Path]] = []

    stats = Stats()

    for src in sources:
        if src.suffix.lower() == ".pdf":
            pdf_pages = pdf.process_pdf(src, cfg, done_pages, parts_dir, stats=stats)
            for page_id, temp_img in pdf_pages:
                all_page_ids.append(page_id)
                if temp_img:
                    ocr_queue.append((len(all_page_ids), page_id, temp_img))
        else:
            page_id = src.stem
            all_page_ids.append(page_id)
            if page_id not in done_pages:
                ocr_queue.append((len(all_page_ids), page_id, src))

    stats.total = len(all_page_ids)
    stats.skipped = sum(1 for pid in all_page_ids if pid in done_pages)

    # ── Server startup (conditional) ─────────────────────────────────────────
    procs: list[subprocess.Popen] = []

    if ocr_queue:
        missing = []
        for name, val in [
            ("llama_server_path", cfg.llama_server_path),
            ("model_path", cfg.model_path),
            ("mmproj_path", cfg.mmproj_path),
        ]:
            if not val:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}.\n"
                "Set them via:\n"
                "  CLI: --llama-server PATH --model PATH --mmproj PATH\n"
                "  Env: LLAMA_SERVER_PATH, MODEL_PATH, MMPROJ_PATH\n"
                "  Or edit src/config.py directly."
            )

        ports = []
        for i in range(cfg.n_servers):
            port = cfg.server_base_port + i
            while _is_port_in_use(port):
                logger.warning("Port %d is already in use, trying next...", port)
                port += 1
            ports.append(port)
        urls = [f"http://127.0.0.1:{port}" for port in ports]

        logger.info("Starting %d llama-server...", cfg.n_servers)
        t_load0 = time.time()
        procs = [_start_server(cfg, port) for port in ports]

        try:
            for proc, url in zip(procs, urls):
                if not _wait_for_server(url, cfg.server_timeout):
                    raise RuntimeError(
                        f"llama-server ({url}) did not start within the time limit ({cfg.server_timeout}s)."
                    )
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server ({url}) exited unexpectedly — port may already be in use."
                    )

            stats.model_load_time = time.time() - t_load0
            logger.info("Servers ready in %.1fs.", stats.model_load_time)

            # ── Pipeline instantiation ────────────────────────────────────────────
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

            # ── Server restart ────────────────────────────────────────────────────
            restart_lock = threading.Lock()
            _fallback_pipeline: list = [None]
            fallback_init_lock = threading.Lock()

            def get_fallback_pipeline():
                if _fallback_pipeline[0] is None:
                    with fallback_init_lock:
                        if _fallback_pipeline[0] is None:
                            logger.info("Instantiating fallback pipeline (without layout)...")
                            _fallback_pipeline[0] = PaddleOCRVL(
                                vl_rec_server_url=f"{urls[0]}/v1",
                                **{**_vlm_kwargs_base, "use_layout_detection": False},
                            )
                return _fallback_pipeline[0]

            def restart_servers() -> None:
                logger.warning("Restarting llama-server instances...")
                for proc in procs:
                    proc.kill()
                    proc.wait()
                new_procs = [_start_server(cfg, port) for port in ports]
                procs[:] = new_procs
                for url in urls:
                    if not _wait_for_server(url, cfg.server_timeout):
                        raise RuntimeError(f"Restart failed: {url} did not respond.")
                while not pipeline_queue.empty():
                    try:
                        pipeline_queue.get_nowait()
                    except queue.Empty:
                        break
                for pl in [PaddleOCRVL(vl_rec_server_url=f"{url}/v1", **_vlm_kwargs_base) for url in urls]:
                    pipeline_queue.put(pl)
                _fallback_pipeline[0] = None
                logger.info("Servers restarted successfully.")

            # ── Parallel processing ───────────────────────────────────────────────
            cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            figures_rel = os.path.relpath(cfg.figures_path, cfg.output_path.parent)

            def process_page(idx: int, page_id: str, img_path: Path) -> dict:
                def _postprocess_and_write(raw_text, metrics, t0, no_layout: bool = False) -> dict:
                    t_ocr = metrics["total_latency"]
                    t_post0 = time.time()
                    page_number, text = extract_page_number(raw_text)
                    clean_text = clean_page(text, cfg, no_layout=no_layout) if cfg.postprocess else text
                    clean_text = strip_table_styles(clean_text)
                    clean_text = strip_math_spacing(clean_text)
                    if cfg.mode == "obsidian":
                        clean_text = fix_image_paths_obsidian(clean_text, cfg.vault_figures_dir)
                    else:
                        clean_text = fix_image_paths(clean_text, page_id, figures_rel)
                    if not cfg.keep_html:
                        clean_text = html_to_markdown(clean_text)
                        if cfg.postprocess:
                            clean_text = clean_page(clean_text, cfg, no_layout=no_layout)
                            clean_text = strip_math_spacing(clean_text)
                    t_post = time.time() - t_post0
                    elapsed = time.time() - t0
                    part_path = parts_dir / f"{page_id}.part"
                    with part_path.open("a", encoding="utf-8") as f:
                        f.write(format_page_block(page_id, clean_text, page_number))
                    return {
                        "idx": idx, "page_name": img_path.name,
                        "elapsed": elapsed, "chars": len(clean_text),
                        "t_ocr": t_ocr, "t_post": t_post, "error": False,
                        "no_layout": no_layout,
                    }

                def _write_error(e, elapsed) -> dict:
                    logger.error("[%d/%d] %s — ERROR (%.1fs): %s",
                                 idx, stats.total, img_path.name, elapsed, e)
                    part_path = parts_dir / f"{page_id}.part"
                    with part_path.open("a", encoding="utf-8") as f:
                        f.write(format_error_block(page_id, str(e)))
                    return {"idx": idx, "page_name": img_path.name, "error": True}

                pl = pipeline_queue.get()
                t0 = time.time()
                try:
                    raw_text, metrics = ocr_image(img_path, pl, cfg)
                    return _postprocess_and_write(raw_text, metrics, t0)
                except OCRTimeout as e:
                    elapsed = time.time() - t0
                    logger.warning("[%d/%d] %s — timeout (%.1fs). Restarting server...",
                                   idx, stats.total, img_path.name, elapsed)
                    pipeline_queue.put(pl)
                    pl = None
                    if restart_lock.acquire(blocking=False):
                        try:
                            restart_servers()
                        finally:
                            restart_lock.release()
                    else:
                        restart_lock.acquire()
                        restart_lock.release()
                    pl_fallback = get_fallback_pipeline()
                    logger.info("[%d/%d] %s — retry without layout detection.",
                                idx, stats.total, img_path.name)
                    t0 = time.time()
                    try:
                        raw_text, metrics = ocr_image(img_path, pl_fallback, cfg)
                        return _postprocess_and_write(raw_text, metrics, t0, no_layout=True)
                    except OCRError as e2:
                        return _write_error(e2, time.time() - t0)
                except OCRError as e:
                    return _write_error(e, time.time() - t0)
                finally:
                    if pl is not None:
                        pipeline_queue.put(pl)

            executor = ThreadPoolExecutor(max_workers=cfg.n_servers)
            try:
                futures = [executor.submit(process_page, idx, page_id, img_path) for idx, page_id, img_path in ocr_queue]
                for future in as_completed(futures):
                    result = future.result()
                    if result["error"]:
                        stats.record_error(page_name=result["page_name"])
                    else:
                        stats.record_success(
                            result["elapsed"], result["chars"],
                            t_ocr=result["t_ocr"], t_post=result["t_post"],
                            page_name=result["page_name"],
                            no_layout=result.get("no_layout", False),
                        )
                        stats.log_page(
                            result["idx"], result["page_name"],
                            result["elapsed"], result["chars"],
                        )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            logger.info("%d server(s) stopped.", cfg.n_servers)

    else:
        logger.info("No pages require OCR — skipping server startup.")

    # ── Combine in all_page_ids order ────────────────────────────────────────
    method_label = cfg.extraction_method
    with cfg.output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# OCR Book\n\n")
        out.write(f"<!-- Generated with {method_label} -->\n")
        for page_id in all_page_ids:
            part = parts_dir / f"{page_id}.part"
            if part.exists():
                out.write(part.read_text(encoding="utf-8"))

    if cfg.header_patterns:
        text = cfg.output_path.read_text(encoding="utf-8")
        cfg.output_path.write_text(
            apply_header_detection(text, cfg.header_patterns),
            encoding="utf-8", newline="\n",
        )
        logger.info("Header detection applied.")

    stats.log_summary()
    stats.write_report(Path(cfg.report_file), cfg)

    if cfg.mode == "obsidian":
        from obsidian import migrate_figures
        migrate_figures(cfg, page_ids=all_page_ids)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if not cfg.verbose:
        for p in cfg.temp_dir.glob("*.png"):
            p.unlink(missing_ok=True)

    if stats.done == 0 and stats.skipped == 0 and cfg.output_path.exists():
        cfg.output_path.unlink()
        logger.warning("No pages processed successfully — output file deleted.")
    else:
        logger.info("Output file: %s", cfg.output_path.resolve())

    return stats
