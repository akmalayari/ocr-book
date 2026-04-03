"""
conftest.py — Fixtures partagées entre tous les modules de tests.
"""

import sys
from pathlib import Path

import pytest

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401 — doit être importé avant nexaai (fix Windows UnicodeDecodeError)

from config import Config


# ── Images minimales ──────────────────────────────────────────────────────────

MINIMAL_JPEG = bytes([
    0xFF, 0xD8,
    0xFF, 0xE0,
    0x00, 0x10,
    0x4A, 0x46, 0x49, 0x46, 0x00,
    0x01, 0x01,
    0x00,
    0x00, 0x01, 0x00, 0x01,
    0x00, 0x00,
    0xFF, 0xD9,
])

MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D,
    0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02,
    0x00, 0x00, 0x00,
    0x90, 0x77, 0x53, 0xDE,
    0x00, 0x00, 0x00, 0x0C,
    0x49, 0x44, 0x41, 0x54,
    0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x01,
    0xE2, 0x21, 0xBC, 0x33,
    0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4E, 0x44,
    0xAE, 0x42, 0x60, 0x82,
])


# ── Fixtures de configuration ─────────────────────────────────────────────────

@pytest.fixture
def cfg_default(tmp_path):
    """Config avec des chemins dans tmp_path."""
    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    return Config(
        model="NexaAI/DeepSeek-OCR-GGUF",
        images_dir=str(images_dir),
        output_file=str(tmp_path / "output.md"),
        log_file="",
        resume=False,
    )


@pytest.fixture
def cfg_all_cleanups(cfg_default):
    cfg_default.remove_isolated_page_numbers = True
    cfg_default.rejoin_hyphenated_words = True
    cfg_default.collapse_blank_lines = True
    return cfg_default


@pytest.fixture
def cfg_no_cleanups(cfg_default):
    cfg_default.remove_isolated_page_numbers = False
    cfg_default.rejoin_hyphenated_words = False
    cfg_default.collapse_blank_lines = False
    return cfg_default


# ── Fixtures de fichiers image ────────────────────────────────────────────────

@pytest.fixture
def jpeg_file(tmp_path):
    p = tmp_path / "page_001.jpg"
    p.write_bytes(MINIMAL_JPEG)
    return p


@pytest.fixture
def png_file(tmp_path):
    p = tmp_path / "page_001.png"
    p.write_bytes(MINIMAL_PNG)
    return p


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    files = []
    for i in range(1, 4):
        p = folder / f"page_{i:03d}.jpg"
        p.write_bytes(MINIMAL_JPEG)
        files.append(p)
    return folder, files
