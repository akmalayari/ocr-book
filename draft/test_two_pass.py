"""
test_two_pass.py — Test pipeline deux passes layout+parse sur page_6.

Passe 1 : layout → bboxes image
Passe 2 : parse sur chaque crop
Résultat : layout avec résultats parse réinjectés sous chaque bbox image

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

IMAGE_PATH = Path("photos/page_6.jpg")
OUT_DIR    = Path("output/draft/test_two_pass")

DET_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
)


def run_vlm(vlm, cfg: Config, image_path: Path) -> tuple[str, float]:
    from preprocess import preprocess_image
    if cfg.preprocess_mode == "binarize":
        input_path = preprocess_image(
            image_path,
            cfg.binarize_block_size,
            cfg.binarize_c,
            cfg.blur_ksize,
            cfg.blur_sigma,
        )
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


def inject_parse_results(layout_text: str, parse_results: list[str]) -> str:
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


def main():
    PASS2_MODE = "describe"

    cfg_layout = Config(prompt_mode="layout")
    cfg_pass2  = Config(prompt_mode=PASS2_MODE)

    print("Chargement VLM...")
    vlm = VLM.from_(model=cfg_layout.model, quant=cfg_layout.quant, config=cfg_layout.to_model_config())

    # Passe 1 : layout
    print(f"\nPasse 1 — layout sur {IMAGE_PATH}")
    layout_text, lat1 = run_vlm(vlm, cfg_layout, IMAGE_PATH)
    print(f"  latency: {lat1:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "page_6__layout.md").write_text(layout_text, encoding="utf-8")

    # Bboxes image
    bboxes = parse_image_bboxes(layout_text)
    print(f"  {len(bboxes)} bbox(es) image détectée(s) : {bboxes}")
    if not bboxes:
        print("Aucune figure détectée — fin.")
        return

    # Crop + passe 2
    pass2_results = []
    for i, bbox in enumerate(bboxes):
        crop_path = OUT_DIR / f"page_6__crop_{i}.jpg"
        crop_image(IMAGE_PATH, bbox, crop_path)

        print(f"\nPasse 2 — {PASS2_MODE} sur crop {i}")
        pass2_text, lat2 = run_vlm(vlm, cfg_pass2, crop_path)
        print(f"  latency: {lat2:.1f}s")

        pass2_out = OUT_DIR / f"page_6__crop_{i}__{PASS2_MODE}.md"
        pass2_out.write_text(pass2_text, encoding="utf-8")
        print(f"  → {pass2_out}")

        pass2_results.append(pass2_text)

    # Réinjection
    final = inject_parse_results(layout_text, pass2_results)
    out_path = OUT_DIR / f"page_6__two_pass_{PASS2_MODE}.md"
    out_path.write_text(final, encoding="utf-8")
    print(f"\nRésultat final → {out_path}")


if __name__ == "__main__":
    main()
