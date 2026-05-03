"""
ocr_client.py — OCR of an image via PaddleOCRVL + llama-server
"""

import logging
import threading
import time
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    pass


class OCRTimeout(OCRError):
    pass


def _predict_with_timeout(pipeline, image_path_str: str, timeout: int, name: str, **kwargs) -> list:
    result: list = []
    exc: list = []

    def _run():
        try:
            result.extend(list(pipeline.predict(image_path_str, **kwargs)))
        except Exception as e:
            exc.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout if timeout > 0 else None)

    if t.is_alive():
        raise OCRTimeout(f"Timeout ({timeout}s) exceeded for {name}.")
    if exc:
        raise exc[0]
    return result


def ocr_image(image_path: Path | str, pipeline, cfg: Config) -> tuple[str, dict]:
    """
    Runs OCR on an image with the PaddleOCRVL pipeline.

    Args:
        image_path : path to the image
        pipeline   : PaddleOCRVL instance (loaded once in pipeline.py)
        cfg        : configuration

    Returns:
        (markdown_text, metrics) where metrics = {"total_latency": float}

    Raises:
        OCRError if the image is not found or generation fails after retry
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise OCRError(f"Image not found: {image_path}")

    save_path = str(cfg.figures_path / image_path.stem)
    md_path = Path(save_path) / f"{image_path.stem}.md"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            output = _predict_with_timeout(
                pipeline, str(image_path), cfg.page_timeout, image_path.name,
                max_new_tokens=cfg.max_tokens,
            )
        except OCRError:
            raise
        except Exception as e:
            raise OCRError(f"Generation failed for {image_path.name}: {e}") from e

        total_latency = time.perf_counter() - t0

        if not output:
            if attempt == 0:
                logger.warning("%s — empty output, retry...", image_path.name)
                continue
            raise OCRError(f"Empty content for {image_path.name}.")

        for res in output:
            res.save_to_markdown(save_path=save_path, pretty=True)

        if not md_path.exists():
            if attempt == 0:
                logger.warning("%s — MD not generated, retry...", image_path.name)
                continue
            raise OCRError(f"Markdown file not generated for {image_path.name}.")

        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            if attempt == 0:
                logger.warning("%s — empty MD, retry...", image_path.name)
                continue
            raise OCRError(f"Empty content for {image_path.name}.")

        logger.debug("%s → %d characters (%.1fs)", image_path.name, len(text), total_latency)
        return text, {"total_latency": total_latency}

    raise OCRError(f"Failed after retry for {image_path.name}.")
