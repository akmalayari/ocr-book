"""
test_two_pass.py — Comparaison parse vs describe × raw vs binarize sur crop.

Passe 1 : layout → bboxes image
Passe 2 : 4 combinaisons sur chaque crop
  - parse   + raw
  - parse   + binarize
  - describe + raw
  - describe + binarize

Usage (depuis la racine du projet) :
    python draft/test_two_pass.py
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import patch  # noqa: F401 — doit précéder tout import nexaai
from nexaai import VLM
from nexaai.nexa_sdk.types import GenerationConfig, VlmChatMessage, VlmContent

from config import Config
from preprocess import preprocess_image

IMAGE_PATH = Path("photos/page_6.jpg")
OUT_DIR    = Path("output/draft/test_two_pass")

DET_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
)

PASS2_VARIANTS = [
    ("parse",    "raw"),
    ("parse",    "binarize"),
    ("describe", "raw"),
    ("describe", "binarize"),
]


def run_vlm(vlm, cfg: Config, image_path: Path) -> tuple[str, float]:
    if cfg.preprocess_mode == "binarize":
        input_path = preprocess_image(image_path, cfg)
    else:
        input_path = image_path

    msg = VlmChatMessage(
        role="user",
        contents=[
            VlmContent(type="image", text=str(input_path.resolve())),
            VlmContent(type="text",  text=cfg.prompt),
        ],
    )
    formatted = vlm.apply_chat_template([msg])
    gen_config = GenerationConfig(
        image_paths=[str(input_path.resolve())],
        max_tokens=cfg.max_tokens,
        sampler_config=cfg.to_sampler_config(),
    )
    accumulated = []

    def on_token(token: str) -> bool:
        accumulated.append(token)
        return True

    t0 = time.perf_counter()
    vlm.generate(formatted, config=gen_config, on_token=on_token)
    latency = time.perf_counter() - t0
    return "".join(accumulated).strip(), latency


def parse_image_bboxes(layout_text: str) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for m in DET_RE.finditer(layout_text):
        if m.group(1) == "image":
            bboxes.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return bboxes


def crop_image(image_path: Path, bbox: tuple, out_path: Path) -> Path:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w / 1000), int(y1 * h / 1000)
    px2, py2 = int(x2 * w / 1000), int(y2 * h / 1000)
    crop = img.crop((px1, py1, px2, py2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    print(f"  crop → {out_path}  ({px2-px1}×{py2-py1}px)")
    return out_path


def main():
    cfg_layout = Config(prompt_mode="layout")

    print("Chargement VLM...")
    vlm = VLM.from_(model=cfg_layout.model, quant=cfg_layout.quant, config=cfg_layout.to_model_config())

    # Passe 1 : layout
    print(f"\nPasse 1 — layout sur {IMAGE_PATH}")
    layout_text, lat1 = run_vlm(vlm, cfg_layout, IMAGE_PATH)
    print(f"  latency: {lat1:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "page_6__layout.md").write_text(layout_text, encoding="utf-8")

    bboxes = parse_image_bboxes(layout_text)
    print(f"  {len(bboxes)} bbox(es) image détectée(s) : {bboxes}")
    if not bboxes:
        print("Aucune figure détectée — fin.")
        return

    # Passe 2 : 4 variantes par crop
    for i, bbox in enumerate(bboxes):
        crop_path = OUT_DIR / f"page_6__crop_{i}.jpg"
        crop_image(IMAGE_PATH, bbox, crop_path)

        print(f"\n--- Crop {i} ---")
        for prompt_mode, preprocess_mode in PASS2_VARIANTS:
            label = f"{prompt_mode}_{preprocess_mode}"
            cfg = Config(prompt_mode=prompt_mode, preprocess_mode=preprocess_mode)

            print(f"  [{label}]")
            text, lat = run_vlm(vlm, cfg, crop_path)
            print(f"    latency: {lat:.1f}s")

            out_path = OUT_DIR / f"page_6__crop_{i}__{label}.md"
            out_path.write_text(text, encoding="utf-8")
            print(f"    → {out_path}")


if __name__ == "__main__":
    main()
