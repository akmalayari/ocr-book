"""
ocr_client.py — OCR d'une image via nexaai.VLM
"""

import logging
import re
import time
from pathlib import Path

from nexaai.nexa_sdk.types import GenerationConfig, VlmChatMessage, VlmContent

from config import Config

logger = logging.getLogger(__name__)


def _has_char_repeat(text: str, min_period: int = 10, max_period: int = 200, min_repeats: int = 4) -> bool:
    """Détecte un motif de caractères qui se répète (ex : boucle HTML sans espaces).

    Vérifie si les derniers `period × min_repeats` caractères forment un motif
    de longueur `period` répété `min_repeats` fois, pour tout period dans [min_period, max_period].
    """
    tail = text[-(max_period * min_repeats):]
    n = len(tail)
    for period in range(min_period, max_period + 1):
        required = period * min_repeats
        if n < required:
            continue
        chunk = tail[-required:]
        unit = chunk[:period]
        if all(chunk[i:i + period] == unit for i in range(0, required, period)):
            return True
    return False


_RE_DET = re.compile(r"<\|det\|>\[\[[^\]]*\]\]<\|/det\|>")


def _is_looping(text: str, window_words: int, threshold: float) -> bool:
    """Détecte une boucle de génération.

    Deux stratégies complémentaires :
    1. Fréquence de mots dans une fenêtre glissante (texte normal).
    2. Motif de caractères répété (boucles HTML/structurées sans espaces).
    """
    # Stratégie 1 : fréquence de mots (coordonnées <|det|> retirées pour ne
    # pas diluer le ratio avec des valeurs numériques changeantes)
    words = [w.lower().strip(".,;:!?\"'()[]{}") for w in _RE_DET.sub("", text).split()]
    window = words[-window_words:]
    if len(window) >= window_words:
        counts = {}
        for w in window:
            counts[w] = counts.get(w, 0) + 1
        n_unique = len(counts)
        repeated = sum(1 for c in counts.values() if c >= 2)
        if repeated / n_unique >= threshold:
            return True

    # Stratégie 2 : motif caractère répété (catches HTML sans espaces)
    return _has_char_repeat(text)


class OCRError(RuntimeError):
    pass


def ocr_image(image_path: Path | str, vlm, cfg: Config) -> tuple[str, dict]:
    """
    OCRise une image avec le VLM chargé.

    Args:
        image_path : chemin vers l'image (déjà préprocessée si nécessaire)
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
    did_loop = [False]

    def on_token(token: str) -> bool:
        accumulated.append(token)
        token_count[0] += 1
        if token_count[0] % cfg.loop_check_every == 0:
            if _is_looping("".join(accumulated), cfg.loop_window_words, cfg.loop_divisor_threshold):
                logger.warning("%s — boucle détectée à %d tokens, génération interrompue", image_path.name, token_count[0])
                did_loop[0] = True
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

    return text, {"total_latency": total_latency, "looped": did_loop[0]}
