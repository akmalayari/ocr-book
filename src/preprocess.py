"""
preprocess.py — Pré-traitement des images avant OCR
"""

import tempfile
from pathlib import Path

import cv2


def preprocess_image(
    image_path: Path,
    block_size: int,
    c: int,
    blur_ksize: int = 5,
    blur_sigma: float = 0.0,
    save_path: Path | None = None,
) -> Path:
    """
    Applique GaussianBlur puis une binarisation adaptative GAUSSIAN_C à l'image.
    Si save_path est fourni, sauvegarde l'image prétraitée à cet emplacement et le retourne.
    Sinon, retourne le chemin vers un fichier JPEG temporaire (delete=False).
    """
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    bw = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), bw)
        return save_path
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, bw)
    return Path(tmp.name)
