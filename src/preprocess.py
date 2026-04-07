"""
preprocess.py — Pré-traitement des images avant OCR
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np


def _save(img: np.ndarray, save_path: Path | None) -> Path:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
        return save_path
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return Path(tmp.name)


def _blur_and_adaptive(
    gray: np.ndarray,
    block_size: int,
    c: int,
    blur_ksize: int,
    blur_sigma: float,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )

def preprocess_image(image_path: Path, cfg, save_path: Path | None = None) -> Path:
    img  = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw   = _blur_and_adaptive(
        gray,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.blur_ksize, cfg.blur_sigma,
    )
    return _save(bw, save_path)

def nlmeans(image_path: Path, cfg, save_path: Path | None = None) -> Path:
    """
    fastNlMeansDenoising — débruitage non-local sans binarisation.
    """
    img      = cv2.imread(str(image_path))
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise    = estimate_noise_level(image_path)
    denoised = cv2.fastNlMeansDenoising(gray, h=cfg.nlmeans_k * noise)
    return _save(denoised, save_path)


def estimate_noise_level(image_path: Path) -> float:
    from skimage.restoration import estimate_sigma

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return float(estimate_sigma(img, average_sigmas=True))