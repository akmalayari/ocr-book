"""
realesrgan_sesr.py — Génération d'images super-résolues via AMD NPU.

Modèles :
  esrgan  — RealESRGAN x4, tiles 128×128  (~14 FPS NPU)
  sesr    — SESR-M7 x2,   tiles 256×256  (~91 FPS NPU)

Stratégie : upscale → resize à la taille originale (sharpening sans changer
la résolution finale transmise au VLM).

Expose generate_sr() pour import depuis test_preprocess.py.

Usage standalone :
    python draft/realesrgan_sesr.py
    python draft/realesrgan_sesr.py --models sesr
    python draft/realesrgan_sesr.py --pages page_5 page_6
    python draft/realesrgan_sesr.py --device cpu
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

CONDA_PYTHON = Path(r"C:\path\to\miniforge3\envs\ryzen-ai-1.7.1\python.exe")

_ROOT = Path(__file__).parent.parent

MODELS: dict[str, dict] = {
    "esrgan": {
        "repo_dir":  _ROOT / "realesrgan-128x128-tiles-amdnpu",
        "onnx_path": _ROOT / "realesrgan-128x128-tiles-amdnpu" / "onnx-models" / "realesrgan_nchw_128x128_u8s8.onnx",
        "scale":     4,
    },
    "sesr": {
        "repo_dir":  _ROOT / "sesr-m7-256x256-tiles-amdnpu",
        "onnx_path": _ROOT / "sesr-m7-256x256-tiles-amdnpu" / "onnx-models" / "sesr_nchw_int8.onnx",
        "scale":     2,
    },
}

PHOTOS_DIR     = _ROOT / "photos"
OUT_DIR        = _ROOT / "output" / "sr"
DEFAULT_PAGES  = ["page_5", "page_6"]
DEFAULT_DEVICE = "npu"


def _check_setup(model: str) -> None:
    if not CONDA_PYTHON.exists():
        print(f"[ERREUR] Python conda introuvable : {CONDA_PYTHON}")
        sys.exit(1)
    onnx = MODELS[model]["onnx_path"]
    if not onnx.exists():
        print(f"[ERREUR] Modèle ONNX introuvable : {onnx}")
        sys.exit(1)


def generate_sr(img_path: Path, model: str, device: str, save_path: Path) -> Path:
    """
    Upscale via AMD NPU (esrgan ×4 ou sesr ×2), resize à la taille originale.
    Retourne save_path.
    """
    _check_setup(model)
    info    = MODELS[model]
    orig    = cv2.imread(str(img_path))
    h, w    = orig.shape[:2]

    tmp_dir = Path(tempfile.mkdtemp())
    result  = subprocess.run(
        [
            str(CONDA_PYTHON),
            str(info["repo_dir"] / "onnx_inference.py"),
            "--onnx",    str(info["onnx_path"]),
            "--input",   str(img_path),
            "--out-dir", str(tmp_dir),
            "--device",  device,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        raise RuntimeError(f"onnx_inference [{model}] : {stderr or stdout or f'exit {result.returncode}'}")

    upscaled_path = tmp_dir / f"{img_path.stem}.png"
    try:
        upscaled = cv2.imread(str(upscaled_path))
        resized  = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), resized)
        return save_path
    finally:
        upscaled_path.unlink(missing_ok=True)
        tmp_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages",   nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--models",  nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--device",  default=DEFAULT_DEVICE, choices=["npu", "cpu"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    for name in args.pages:
        candidates = list(PHOTOS_DIR.glob(f"{name}.*"))
        if not candidates:
            print(f"[WARN] Aucune image trouvée pour '{name}'")
            continue
        images.append(candidates[0])
    if not images:
        print("Aucune image.")
        sys.exit(1)

    for img_path in images:
        for model in args.models:
            save_path = OUT_DIR / f"{img_path.stem}_{model}.jpg"
            print(f"  [{img_path.stem} {model}] ...", end=" ", flush=True)
            try:
                generate_sr(img_path, model, args.device, save_path)
                print(f"→ {save_path.name}")
            except RuntimeError as e:
                print(f"ERREUR: {e}")

    print(f"\nImages dans {OUT_DIR}")


if __name__ == "__main__":
    main()
