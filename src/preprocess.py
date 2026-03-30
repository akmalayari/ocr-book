"""
preprocess.py — Pré-traitement des images avant OCR
"""

import tempfile
from pathlib import Path

import cv2


def preprocess_image(image_path: Path) -> Path:
    """
    Applique une binarisation adaptative à l'image.

    Paramètres validés sur DeepSeek-OCR-GGUF :
    - ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY
    - blockSize=31, C=10

    Retourne le chemin vers un fichier JPEG temporaire.
    Le fichier temporaire persiste jusqu'à la fin du processus
    (delete=False) ; le pipeline n'a pas à le gérer explicitement.
    """
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, bw)
    return Path(tmp.name)
