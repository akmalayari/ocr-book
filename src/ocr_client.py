"""
ocr_client.py — OCR d'une image via nexaai.VLM
"""

import logging
import time
from pathlib import Path

from nexaai.nexa_sdk.types import GenerationConfig, VlmChatMessage, VlmContent

from config import Config
from preprocess import preprocess_image

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    pass


def ocr_image(image_path: Path | str, vlm, cfg: Config) -> tuple[str, dict]:
    """
    OCRise une image avec le VLM chargé.

    Args:
        image_path : chemin vers l'image source
        vlm        : instance nexaai.VLM (chargée une fois dans pipeline.py)
        cfg        : configuration

    Returns:
        (texte_ocr, métriques) où métriques = {"total_latency": float}

    Raises:
        OCRError si l'image est introuvable ou si la génération échoue
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise OCRError(f"Image introuvable : {image_path}")

    if cfg.preprocess_mode == "binarize":
        input_path = preprocess_image(image_path, cfg.binarize_block_size, cfg.binarize_c)
    else:
        input_path = image_path

    msg = VlmChatMessage(
        role="user",
        contents=[
            VlmContent(type="image", text=str(input_path.resolve())),
            VlmContent(type="text",  text=cfg.prompt),
        ],
    )
    formatted = vlm.apply_chat_template([msg])
    gen_config = GenerationConfig(
        image_paths=[str(input_path.resolve())],
        max_tokens=cfg.max_tokens,
    )

    t0 = time.perf_counter()
    try:
        result = vlm.generate(formatted, config=gen_config)
    except Exception as e:
        raise OCRError(f"Échec génération pour {image_path.name} : {e}") from e
    total_latency = time.perf_counter() - t0

    text = result.full_text.strip() if result.full_text else ""
    if not text:
        raise OCRError(f"Contenu vide pour {image_path.name}.")

    logger.debug("%s → %d caractères (%.1fs)", image_path.name, len(text), total_latency)

    return text, {"total_latency": total_latency}
