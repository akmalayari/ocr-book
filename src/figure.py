"""
figure.py — Détection et traitement des figures (two-pass OCR)
"""

import logging
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from config import Config
from ocr_client import ocr_image
from postprocess import extract_table_footnotes
from preprocess import preprocess_image

logger = logging.getLogger(__name__)

DET_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
)


def parse_image_bboxes(layout_text: str) -> list[tuple[int, int, int, int]]:
    """Extrait les bbox des régions 'image' du résultat layout."""
    bboxes = []
    for m in DET_RE.finditer(layout_text):
        if m.group(1) == "image":
            bboxes.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return bboxes


def crop_image(image_path: Path, bbox: tuple, out_path: Path) -> Path:
    """Crop en coordonnées normalisées (0–1000) et sauvegarde dans out_path."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w / 1000), int(y1 * h / 1000)
    px2, py2 = int(x2 * w / 1000), int(y2 * h / 1000)
    crop = img.crop((px1, py1, px2, py2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def inject_parse_results(layout_text: str, parse_results: list[str]) -> str:
    """Réinjecte les résultats parse sous chaque balise image dans le texte layout."""
    lines = layout_text.splitlines()
    result = []
    bbox_idx = 0
    for line in lines:
        result.append(line)
        m = DET_RE.search(line)
        if m and m.group(1) == "image" and bbox_idx < len(parse_results):
            result.append(parse_results[bbox_idx])
            bbox_idx += 1
    return "\n".join(result)


def process_figures(
    layout_text: str,
    image_path: Path,
    vlm,
    cfg: Config,
    page_id: str,
) -> tuple[str, dict]:
    """
    Orchestre la passe 2 :
      - détecte les bboxes image dans layout_text
      - crop image_path → figures_dir/page_id/figure_N.jpg (sauvegarde)
      - binarise le crop → fichier temporaire pour OCR
      - lance ocr_image (prompt_mode="parse") sur chaque crop binarisé
      - réinjecte les résultats dans layout_text
      - retourne le texte enrichi + métriques cumulées
    """
    bboxes = parse_image_bboxes(layout_text)
    if not bboxes:
        return layout_text, {"total_latency": 0.0}

    cfg_parse = replace(cfg, prompt_mode="parse")
    figures_path = Path(cfg.figures_dir) / page_id
    parse_results = []
    total_latency = 0.0

    for i, bbox in enumerate(bboxes):
        crop_orig = figures_path / f"figure_{i}.jpg"
        crop_image(image_path, bbox, crop_orig)
        logger.debug("%s — figure %d sauvegardée : %s", page_id, i, crop_orig)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if cfg.preprocess_mode == "binarize":
                input_path = preprocess_image(crop_orig, cfg, save_path=tmp_path)
            else:
                input_path = crop_orig
            parse_text, metrics = ocr_image(input_path, vlm, cfg_parse)
            parse_results.append(extract_table_footnotes(parse_text))
            total_latency += metrics["total_latency"]
            logger.debug("%s — figure %d parsée (%.1fs)", page_id, i, metrics["total_latency"])
        finally:
            tmp_path.unlink(missing_ok=True)

    enriched = inject_parse_results(layout_text, parse_results)
    return enriched, {"total_latency": total_latency}
