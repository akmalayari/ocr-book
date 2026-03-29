"""
ocr_client.py — Envoi d'images au serveur Nexa et récupération du texte OCR
"""

import base64
import logging
from pathlib import Path

import requests

from config import Config

logger = logging.getLogger(__name__)

# Types MIME acceptés
_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}


class OCRError(RuntimeError):
    pass


def _encode_image(image_path: Path) -> tuple[str, str]:
    """
    Retourne (base64_data, mime_type) pour une image.
    """
    mime = _MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


def ocr_image(image_path: Path | str, cfg: Config) -> str:
    """
    Envoie une image au serveur Nexa et retourne le texte OCR.

    Args:
        image_path : chemin vers l'image
        cfg        : configuration

    Returns:
        Texte OCR (str)

    Raises:
        OCRError si la requête échoue ou si la réponse est vide
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise OCRError(f"Image introuvable : {image_path}")

    b64, mime = _encode_image(image_path)

    payload = {
        "model": cfg.model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": cfg.prompt
                },
            ],
        }],
        "max_completion_tokens": cfg.max_tokens,
        "temperature":           cfg.temperature,
        "enable_think":          False,
        "stream":                False,
    }

    url = f"http://127.0.0.1:{cfg.port}/v1/chat/completions"

    try:
        resp = requests.post(url, json=payload, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
    except requests.Timeout:
        raise OCRError(
            f"Timeout ({cfg.request_timeout_s}s) sur {image_path.name}. "
            "Augmentez request_timeout_s dans la config."
        )
    except requests.HTTPError as e:
        raise OCRError(f"Erreur HTTP {resp.status_code} : {e}")
    except requests.ConnectionError:
        raise OCRError(
            "Impossible de joindre le serveur Nexa. "
            f"Est-il bien démarré sur le port {cfg.port} ?"
        )

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise OCRError(f"Réponse vide pour {image_path.name} : {data}")

    text = choices[0].get("message", {}).get("content", "").strip()
    if not text:
        raise OCRError(f"Contenu vide pour {image_path.name}.")

    logger.debug(
        "%s → %d caractères générés",
        image_path.name,
        len(text),
    )
    return text
