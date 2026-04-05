"""
Preprocess test: compare exposure boost vs adaptive binarization.
For each image, saves preprocessed variants to draft/preprocess_out/
then runs OCR on each variant and prints results side by side.

Run from repo root: python draft/preprocess_test.py
"""

import time
from pathlib import Path

import cv2
from PIL import Image, ImageEnhance

from nexaai import VLM
from nexaai.nexa_sdk import types as _nexa_types
from nexaai.nexa_sdk.types import GenerationConfig, VlmChatMessage, VlmContent

# ── Monkey-patch ──────────────────────────────────────────────────────────────
_orig_from_c_struct = _nexa_types.ProfileData.from_c_struct.__func__

@classmethod
def _safe_from_c_struct(cls, c_struct):
    try:
        return _orig_from_c_struct(cls, c_struct)
    except (UnicodeDecodeError, AttributeError):
        return cls(stop_reason="unknown")

_nexa_types.ProfileData.from_c_struct = _safe_from_c_struct
# ─────────────────────────────────────────────────────────────────────────────

IMAGES = [Path("photos/page_1.jpg"), Path("photos/page_2.jpg")]
PROMPT = "Free OCR."
MODEL = "NexaAI/DeepSeek-OCR-GGUF"
OUT_DIR = Path("draft/preprocess_out")
OUT_DIR.mkdir(exist_ok=True)


# ── Preprocessing variants ────────────────────────────────────────────────────

def exposure_pillow(path: Path) -> Path:
    """Pillow: contrast + brightness boost."""
    img = Image.open(path)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    out = OUT_DIR / f"{path.stem}_exposure_pillow.jpg"
    img.save(out)
    return out

'''
def exposure_clahe(path: Path) -> Path:
    """OpenCV: CLAHE on L channel (LAB colorspace)."""
    img = cv2.imread(str(path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = OUT_DIR / f"{path.stem}_exposure_clahe.jpg"
    cv2.imwrite(str(out), result)
    return out
'''

def binarize_adaptive(path: Path) -> Path:
    """OpenCV: adaptive threshold (Gaussian, blockSize=31)."""
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    out = OUT_DIR / f"{path.stem}_binarize_adaptive.jpg"
    cv2.imwrite(str(out), bw)
    return out

'''
def binarize_otsu(path: Path) -> Path:
    """OpenCV: Gaussian blur + Otsu threshold."""
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = OUT_DIR / f"{path.stem}_binarize_otsu.jpg"
    cv2.imwrite(str(out), bw)
    return out
'''

VARIANTS = [
    #("original",          lambda p: p),
    ("exposure_pillow",   exposure_pillow),
    #("exposure_clahe",    exposure_clahe),
    ("binarize_adaptive", binarize_adaptive)
    #("binarize_otsu",     binarize_otsu),
]


# ── OCR helper ────────────────────────────────────────────────────────────────

def run_ocr(vlm, image_path: Path) -> tuple[str, float]:
    msg = VlmChatMessage(
        role="user",
        contents=[
            VlmContent(type="image", text=str(image_path.resolve())),
            VlmContent(type="text", text=PROMPT),
        ],
    )
    formatted = vlm.apply_chat_template([msg])
    config = GenerationConfig(image_paths=[str(image_path.resolve())], max_tokens=2048)

    t0 = time.perf_counter()
    result = vlm.generate(formatted, config=config)
    elapsed = time.perf_counter() - t0

    return result.full_text.strip(), elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Loading {MODEL} ...")
vlm = VLM.from_(model=MODEL)
print("Model loaded.\n")

for image_path in IMAGES:
    print(f"{'='*60}")
    print(f"IMAGE: {image_path}")
    print(f"{'='*60}")

    for name, preprocess_fn in VARIANTS:
        processed = preprocess_fn(image_path)
        text, elapsed = run_ocr(vlm, processed)
        tokens_approx = len(text.split())
        print(f"\n── {name} ({elapsed:.1f}s, ~{tokens_approx} words) ──")
        # Show first 400 chars and last 100 chars to detect looping
        print(text)

    print()
