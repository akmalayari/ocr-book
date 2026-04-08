"""
ocr_client.py — OCR d'une image via PaddleOCRVL + llama-server
"""

import logging
import time
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    pass


def ocr_image(image_path: Path | str, pipeline, cfg: Config) -> tuple[str, dict]:
    """
    OCRise une image avec le pipeline PaddleOCRVL.

    Args:
        image_path : chemin vers l'image
        pipeline   : instance PaddleOCRVL (chargée une fois dans pipeline.py)
        cfg        : configuration

    Returns:
        (markdown_text, métriques) où métriques = {"total_latency": float}

    Raises:
        OCRError si l'image est introuvable ou si la génération échoue
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise OCRError(f"Image introuvable : {image_path}")

    save_path = str(cfg.figures_path / image_path.stem)

    t0 = time.perf_counter()
    try:
        output = list(pipeline.predict(str(image_path)))
    except Exception as e:
        raise OCRError(f"Échec génération pour {image_path.name} : {e}") from e
    total_latency = time.perf_counter() - t0

    if not output:
        raise OCRError(f"Contenu vide pour {image_path.name}.")

    md_path = Path(save_path) / f"{image_path.stem}.md"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for res in output:
        res.save_to_markdown(save_path=save_path)

    if not md_path.exists():
        raise OCRError(f"Fichier markdown non généré pour {image_path.name}.")

    text = md_path.read_text(encoding="utf-8").strip()
    if not text:
        raise OCRError(f"Contenu vide pour {image_path.name}.")

    logger.debug("%s → %d caractères (%.1fs)", image_path.name, len(text), total_latency)

    return text, {"total_latency": total_latency}
