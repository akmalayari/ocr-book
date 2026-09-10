"""Shared locations and identifiers for the PaddleOCR-VL GGUF assets."""

import os
import platform
from pathlib import Path


MODEL_REPO_ID = "PaddlePaddle/PaddleOCR-VL-1.5-GGUF"
MODEL_REVISION = "cc977c16989848c264d813ec1705cb181b7a21ee"
MODEL_FILENAME = "PaddleOCR-VL-1.5.gguf"
MMPROJ_FILENAME = "PaddleOCR-VL-1.5-mmproj.gguf"
MODEL_FILE_SIZE = 935_768_992
MMPROJ_FILE_SIZE = 881_770_496


def default_model_dir() -> Path:
    """Return a platform-native per-user cache directory for model files."""
    override = os.environ.get("OCR_MODEL_CACHE_DIR", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()

    system = platform.system()
    if system == "Windows":
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif system == "Darwin":
        cache_root = Path.home() / "Library/Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
        cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "ocr-book/models/PaddleOCR-VL-1.5-GGUF"


def default_model_paths(model_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Return the expected model and multimodal projector paths."""
    directory = (
        Path(os.path.expandvars(str(model_dir))).expanduser()
        if model_dir is not None
        else default_model_dir()
    )
    return directory / MODEL_FILENAME, directory / MMPROJ_FILENAME
