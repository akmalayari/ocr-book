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


def _is_looping(text: str, window_words: int, threshold: float) -> bool:
    """Détecte une boucle de génération.

    Si le ratio de mots apparaissant 2+ fois dans la fenêtre dépasse `threshold` → boucle.
    """
    words = text.split()[-window_words:]
    if len(words) < window_words:
        return False
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    n_unique = len(counts)
    repeated = sum(1 for c in counts.values() if c >= 2)
    return repeated / n_unique >= threshold


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
        input_path = preprocess_image(
            image_path,
            cfg.binarize_block_size,
            cfg.binarize_c,
            cfg.blur_ksize,
            cfg.blur_sigma,
        )
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
        sampler_config=cfg.to_sampler_config(),
    )

    accumulated = []
    token_count = [0]

    def on_token(token: str) -> bool:
        accumulated.append(token)
        token_count[0] += 1
        if token_count[0] % cfg.loop_check_every == 0:
            if _is_looping("".join(accumulated), cfg.loop_window_words, cfg.loop_divisor_threshold):
                logger.warning("%s — boucle détectée à %d tokens, génération interrompue", image_path.name, token_count[0])
                return False
        return True

    t0 = time.perf_counter()
    try:
        vlm.generate(formatted, config=gen_config, on_token=on_token)
    except Exception as e:
        raise OCRError(f"Échec génération pour {image_path.name} : {e}") from e
    total_latency = time.perf_counter() - t0

    text = "".join(accumulated).strip()
    if not text:
        raise OCRError(f"Contenu vide pour {image_path.name}.")

    logger.debug("%s → %d caractères (%.1fs)", image_path.name, len(text), total_latency)

    return text, {"total_latency": total_latency}
