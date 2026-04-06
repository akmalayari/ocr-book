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

def _sauvola_and_adaptive(
    gray: np.ndarray,
    block_size: int,
    c: int,
    window_size: int,
    k: float,
) -> np.ndarray:
    """AND(Sauvola, adaptiveThreshold) sur une image déjà en niveaux de gris."""
    from skimage.filters import threshold_sauvola
    thresh   = threshold_sauvola(gray, window_size=window_size, k=k)
    sauvola  = ((gray > thresh).astype(np.uint8)) * 255
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )
    return cv2.bitwise_and(sauvola, adaptive)


def sauvola_binarize(image_path: Path, cfg, save_path: Path | None = None) -> Path:
    """
    AND(Sauvola(gray), adaptive(blurred)) — conserve les pixels texte détectés par l'un ou l'autre.
    Corrige la perte de texte dans les zones à faible variance (pliure, ombre).
    """
    from skimage.filters import threshold_sauvola
    img     = cv2.imread(str(image_path))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh  = threshold_sauvola(gray, window_size=cfg.sauvola_window_size, k=cfg.sauvola_k)
    sauvola = ((gray > thresh).astype(np.uint8)) * 255
    adaptive = _blur_and_adaptive(
        gray,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.blur_ksize, cfg.blur_sigma,
    )
    return _save(cv2.bitwise_and(sauvola, adaptive), save_path)


def median_and(image_path: Path, cfg, save_path: Path | None = None) -> Path:
    """
    medianBlur(3) + AND(Sauvola, adaptive) — variante légère de sauvola_and.
    Le medianBlur supprime les granules isolés sans étaler les bords comme le blur gaussien.
    """
    img    = cv2.imread(str(image_path))
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    bw = _sauvola_and_adaptive(
        median,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.sauvola_window_size, cfg.sauvola_k,
    )
    return _save(bw, save_path)


def nlmeans_and(image_path: Path, cfg, save_path: Path | None = None) -> Path:
    """
    fastNlMeansDenoising + AND(Sauvola, adaptive) — débruitage non-local + double seuillage.
    Équivalent de sauvola_and sur image débruitée ; préserve mieux les bords fins.
    """
    img      = cv2.imread(str(image_path))
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise    = estimate_noise_level(image_path)
    denoised = cv2.fastNlMeansDenoising(gray, h=cfg.nlmeans_k*noise)
    bw = _sauvola_and_adaptive(
        denoised,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.sauvola_window_size, cfg.sauvola_k,
    )
    return _save(bw, save_path)

def estimate_noise_level(image_path: Path) -> float:
    from skimage.restoration import estimate_sigma

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return float(estimate_sigma(img, average_sigmas=True))