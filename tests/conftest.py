"""
conftest.py — Fixtures partagées entre tous les modules de tests.

Conventions :
  - Toutes les fixtures qui créent des fichiers utilisent tmp_path (pytest built-in).
  - cfg_default() produit un Config minimal sans effets de bord (pas de vrai port, 
    pas de vrai dossier).
  - Les images factices sont de vrais JPEG minimaux (2 octets valides suffisent
    pour tester l'encodage sans PIL).
"""

import sys
from pathlib import Path

import pytest

# Ajouter le dossier parent au PYTHONPATH pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


# ── JPEG minimal (2×2 pixels, valide) ────────────────────────────────────────
# Séquence JFIF minimale : SOI + APP0 + EOI — suffisant pour un test d'encodage.
MINIMAL_JPEG = bytes([
    0xFF, 0xD8,              # SOI  (Start Of Image)
    0xFF, 0xE0,              # APP0 marker
    0x00, 0x10,              # APP0 length = 16
    0x4A, 0x46, 0x49, 0x46, 0x00,  # "JFIF\0"
    0x01, 0x01,              # version 1.1
    0x00,                    # aspect ratio units = 0
    0x00, 0x01, 0x00, 0x01,  # Xdensity, Ydensity
    0x00, 0x00,              # Xthumbnail, Ythumbnail
    0xFF, 0xD9,              # EOI  (End Of Image)
])

# PNG minimal (1×1 pixel transparent)
MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
    0x00, 0x00, 0x00, 0x0D,                             # IHDR chunk length
    0x49, 0x48, 0x44, 0x52,                             # "IHDR"
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,    # 1x1
    0x08, 0x02,                                         # bit depth=8, color type=RGB
    0x00, 0x00, 0x00,                                   # compression, filter, interlace
    0x90, 0x77, 0x53, 0xDE,                             # CRC
    0x00, 0x00, 0x00, 0x0C,                             # IDAT chunk length
    0x49, 0x44, 0x41, 0x54,                             # "IDAT"
    0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x01,                             # compressed data + CRC
    0xE2, 0x21, 0xBC, 0x33,
    0x00, 0x00, 0x00, 0x00,                             # IEND chunk length
    0x49, 0x45, 0x4E, 0x44,                             # "IEND"
    0xAE, 0x42, 0x60, 0x82,                             # CRC
])


# ── Fixtures de configuration ─────────────────────────────────────────────────

@pytest.fixture
def cfg_default(tmp_path):
    """Config avec des chemins dans tmp_path, port fictif, timeout court."""
    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    return Config(
        model="NexaAI/DeepSeek-OCR-GGUF",
        port=19999,                   # port non utilisé en prod
        server_timeout_s=2,           # timeout court pour les tests
        request_timeout_s=5,
        images_dir=str(images_dir),
        output_file=str(tmp_path / "output.md"),
        log_file="",                  # pas de fichier log pendant les tests
        resume=False,
    )


@pytest.fixture
def cfg_all_cleanups(cfg_default):
    """Config avec tous les post-traitements activés."""
    cfg_default.remove_isolated_page_numbers = True
    cfg_default.rejoin_hyphenated_words = True
    cfg_default.collapse_blank_lines = True
    return cfg_default


@pytest.fixture
def cfg_no_cleanups(cfg_default):
    """Config avec tous les post-traitements désactivés."""
    cfg_default.remove_isolated_page_numbers = False
    cfg_default.rejoin_hyphenated_words = False
    cfg_default.collapse_blank_lines = False
    return cfg_default


# ── Fixtures de fichiers image ────────────────────────────────────────────────

@pytest.fixture
def jpeg_file(tmp_path):
    """Crée un vrai fichier JPEG minimal sur disque, retourne son Path."""
    p = tmp_path / "page_001.jpg"
    p.write_bytes(MINIMAL_JPEG)
    return p


@pytest.fixture
def png_file(tmp_path):
    """Crée un vrai fichier PNG minimal sur disque, retourne son Path."""
    p = tmp_path / "page_001.png"
    p.write_bytes(MINIMAL_PNG)
    return p


@pytest.fixture
def image_folder(tmp_path):
    """
    Crée un dossier avec 3 images JPEG nommées page_001/002/003.
    Retourne (dossier: Path, fichiers: list[Path]).
    """
    folder = tmp_path / "photos"
    folder.mkdir()
    files = []
    for i in range(1, 4):
        p = folder / f"page_{i:03d}.jpg"
        p.write_bytes(MINIMAL_JPEG)
        files.append(p)
    return folder, files


# ── Réponses HTTP simulées ────────────────────────────────────────────────────

def make_ocr_response(text: str) -> dict:
    """Construit un dict JSON simulant une réponse OpenAI chat/completions."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


GOOD_OCR_RESPONSE = make_ocr_response("## Chapitre 1\n\nTexte extrait.")
EMPTY_CHOICES_RESPONSE = {"id": "x", "choices": []}
EMPTY_CONTENT_RESPONSE = make_ocr_response("   ")  # whitespace only → vide après strip
