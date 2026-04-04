"""
viz_boxes.py — Visualise les bounding boxes DeepSeek-OCR sur les images originales.

Parse le format <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> depuis un .md
et dessine les boîtes sur les images sources (coordonnées normalisées 0-1000).

Usage:
    python draft/viz_boxes.py                                    # défaut
    python draft/viz_boxes.py --md output/livre.md --photos photos/ --out draft/viz
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MD_FILE   = "output/prompt_results/page_10__rec.md"
PHOTOS    = "photos/page_10.jpg"
OUT_DIR   = "output/draft/viz"

COLORS = {
    "text":            "#2196F3",  # bleu
    "title":           "#F44336",  # rouge
    "sub_title":       "#E91E63",  # rose
    "image":           "#4CAF50",  # vert
    "image_caption":   "#8BC34A",  # vert clair
    "table":           "#FF9800",  # orange
    "table_caption":   "#FFC107",  # ambre
    "table_footnote":  "#FF5722",  # orange foncé
    "figure":          "#4CAF50",  # vert (alias)
}
DEFAULT_COLOR = "#9C27B0"  # violet pour labels inconnus

PAGE_RE  = re.compile(r"<!-- Page (\S+) -->")
DET_RE   = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")


def parse_md(md_path: Path) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Retourne {page_stem: [(label, x1, y1, x2, y2), ...]}."""
    pages: dict[str, list] = {}
    # Stem du fichier comme page par défaut (si pas de marqueur <!-- Page ... -->)
    default_stem = re.sub(r"__.*$", "", md_path.stem)
    current = default_stem
    pages[current] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = PAGE_RE.search(line)
        if m:
            current = m.group(1)
            pages.setdefault(current, [])
            continue
        for det in DET_RE.finditer(line):
            label = det.group(1)
            coords = tuple(int(det.group(i)) for i in range(2, 6))
            pages[current].append((label, *coords))
    # Supprimer les pages sans boxes
    return {k: v for k, v in pages.items() if v}


def draw_boxes(image_path: Path, boxes: list, out_path: Path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, h // 80))
    except OSError:
        font = ImageFont.load_default()

    for label, x1, y1, x2, y2 in boxes:
        px1 = int(x1 * w / 1000)
        py1 = int(y1 * h / 1000)
        px2 = int(x2 * w / 1000)
        py2 = int(y2 * h / 1000)
        color = COLORS.get(label, DEFAULT_COLOR)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=max(3, h // 500))
        draw.text((px1 + 4, py1 + 2), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  → {out_path}  ({len(boxes)} box(es))")


def find_image(photos_dir: Path, stem: str) -> Path | None:
    if photos_dir.is_file():
        return photos_dir if photos_dir.exists() else None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = photos_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md",     default=MD_FILE)
    ap.add_argument("--photos", default=PHOTOS)
    ap.add_argument("--out",    default=OUT_DIR)
    args = ap.parse_args()

    md_path     = Path(args.md)
    photos_dir  = Path(args.photos)
    out_dir     = Path(args.out)

    if not md_path.exists():
        sys.exit(f"Fichier introuvable : {md_path}")

    pages = parse_md(md_path)
    if not pages:
        sys.exit("Aucune bounding box trouvée dans le fichier.")

    print(f"{len(pages)} page(s) détectée(s) dans {md_path}\n")
    for stem, boxes in pages.items():
        img_path = find_image(photos_dir, stem)
        if img_path is None:
            print(f"  ✗ {stem}: image introuvable dans {photos_dir}")
            continue
        out_path = out_dir / f"{stem}_boxes.jpg"
        draw_boxes(img_path, boxes, out_path)


if __name__ == "__main__":
    main()
