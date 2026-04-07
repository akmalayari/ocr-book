"""
sesr.py — Super-résolution SESR-M7 x2 via AMD NPU.

Upscale ×2 puis resize à la taille originale (sharpening sans changer
la résolution transmise au VLM).
"""

import subprocess
import tempfile
from pathlib import Path

import cv2

_ROOT      = Path(__file__).parent.parent
_REPO_DIR  = _ROOT / "sesr-m7-256x256-tiles-amdnpu"
_ONNX_PATH = _REPO_DIR / "onnx-models" / "sesr_nchw_int8.onnx"
_CONDA_PY  = Path(r"C:\path\to\miniforge3\envs\ryzen-ai-1.7.1\python.exe")


def sesr(img_path: Path, cfg, save_path: Path | None = None) -> Path:
    """Upscale SESR-M7 ×2 via NPU, resize à la taille originale."""
    if not _CONDA_PY.exists():
        raise RuntimeError(f"Python conda introuvable : {_CONDA_PY}")
    if not _ONNX_PATH.exists():
        raise RuntimeError(f"Modèle SESR introuvable : {_ONNX_PATH}")

    orig = cv2.imread(str(img_path))
    h, w = orig.shape[:2]

    tmp_dir = Path(tempfile.mkdtemp())
    result = subprocess.run(
        [
            str(_CONDA_PY),
            str(_REPO_DIR / "onnx_inference.py"),
            "--onnx",    str(_ONNX_PATH),
            "--input",   str(img_path),
            "--out-dir", str(tmp_dir),
            "--device",  cfg.sesr_device,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        raise RuntimeError(f"sesr: {stderr or stdout or f'exit {result.returncode}'}")

    upscaled_path = tmp_dir / f"{img_path.stem}.png"
    try:
        upscaled = cv2.imread(str(upscaled_path))
        resized  = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if save_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            save_path = Path(tmp.name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), resized)
        return save_path
    finally:
        upscaled_path.unlink(missing_ok=True)
        tmp_dir.rmdir()
