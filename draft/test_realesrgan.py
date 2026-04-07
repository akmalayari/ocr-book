"""
test_realesrgan.py — Super-résolution via AMD NPU (realesrgan-128x128-tiles-amdnpu) avant OCR.

Stratégie : upscale ×4 (NPU AMD, modèle INT8 128×128 tiles) → redimensionner à taille originale
(sharpening sans changer la résolution finale transmise au VLM).

Phase 1 (défaut)  : génère les images super-résolues.
Phase 2 (--ocr)   : lance l'OCR sur les images SR + configs baseline pour comparaison.

Sorties : output/realesrgan/

Usage :
    python draft/test_realesrgan.py
    python draft/test_realesrgan.py --ocr
    python draft/test_realesrgan.py --ocr esrgan esrgan_blur
    python draft/test_realesrgan.py --device cpu
    python draft/test_realesrgan.py --list
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image
from preprocess import preprocess_image, median_and, sauvola_binarize
from figure import process_figures

PHOTOS_DIR   = Path(__file__).parent.parent / "photos"
OUT_DIR      = Path(__file__).parent.parent / "output" / "realesrgan"

REPO_DIR     = Path(__file__).parent.parent / "realesrgan-128x128-tiles-amdnpu"
ONNX_PATH    = REPO_DIR / "onnx-models" / "realesrgan_nchw_128x128_u8s8.onnx"
CONDA_PYTHON = Path(r"C:\path\to\miniforge3\envs\ryzen-ai-1.7.1\python.exe")

DEFAULT_PAGES  = ["page_5", "page_6"]
DEFAULT_DEVICE = "npu"

# Configs SR : label → postprocess_key | None
SR_CONFIGS: dict[str, str | None] = {
    "esrgan":         None,
    "esrgan_blur":    "blur_adaptive",
    "esrgan_sauvola": "sauvola_and",
    "esrgan_median":  "median_and",
}

# Configs baseline (sans SR) pour point de comparaison
BASELINE_CONFIGS: dict[str, str] = {
    "none":          "image originale",
    "blur_adaptive": "GaussianBlur(5,5) + adaptive C=15",
    "sauvola_and":   "AND(Sauvola w=51, adaptive)",
    "median_and":    "medianBlur(3) + AND(Sauvola, adaptive)",
}

_POSTPROCESS_FNS = {
    "blur_adaptive": preprocess_image,
    "sauvola_and":   sauvola_binarize,
    "median_and":    median_and,
}


# ── SR helpers ────────────────────────────────────────────────────────────────

def _check_npu_setup() -> None:
    if not CONDA_PYTHON.exists():
        print(f"[ERREUR] Python conda introuvable : {CONDA_PYTHON}")
        sys.exit(1)
    if not ONNX_PATH.exists():
        print(f"[ERREUR] Modèle ONNX introuvable : {ONNX_PATH}")
        sys.exit(1)


def _upscale_sr(img_path: Path, device: str) -> Path:
    """Upscale ×4 via AMD NPU (onnx_inference.py). Retourne le PNG upscalé."""
    tmp_dir = Path(tempfile.mkdtemp())
    result = subprocess.run(
        [
            str(CONDA_PYTHON),
            str(REPO_DIR / "onnx_inference.py"),
            "--onnx",    str(ONNX_PATH),
            "--input",   str(img_path),
            "--out-dir", str(tmp_dir),
            "--device",  device,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        msg = stderr or stdout or f"exit {result.returncode}"
        raise RuntimeError(f"onnx_inference.py échoué : {msg}")
    return tmp_dir / f"{img_path.stem}.png"


def _apply_sr(img_path: Path, device: str, cfg: Config,
              postprocess_key: str | None, save_path: Path) -> Path:
    """
    Upscale ×4 via AMD NPU, puis redimensionne à la taille originale.
    Si postprocess_key est fourni, applique la binarisation correspondante ensuite.
    """
    orig = cv2.imread(str(img_path))
    h_orig, w_orig = orig.shape[:2]

    upscaled_path = _upscale_sr(img_path, device)
    try:
        upscaled = cv2.imread(str(upscaled_path))
        resized  = cv2.resize(upscaled, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

        if postprocess_key is None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), resized)
            return save_path

        tmp_resized = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        cv2.imwrite(str(tmp_resized), resized)
        try:
            return _POSTPROCESS_FNS[postprocess_key](tmp_resized, cfg, save_path)
        finally:
            tmp_resized.unlink(missing_ok=True)
    finally:
        upscaled_path.unlink(missing_ok=True)
        upscaled_path.parent.rmdir()


def _apply_baseline(img_path: Path, config_name: str, cfg: Config, save_path: Path) -> Path:
    if config_name == "none":
        return img_path
    return _POSTPROCESS_FNS[config_name](img_path, cfg, save_path)


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path], device: str) -> None:
    _check_npu_setup()
    print(f"── Génération des images super-résolues (device={device}) ──────────")
    cfg = Config()
    for img_path in images:
        for label, postprocess_key in SR_CONFIGS.items():
            save_path = OUT_DIR / f"{img_path.stem}_{label}.jpg"
            _apply_sr(img_path, device, cfg, postprocess_key, save_path)
            print(f"  {img_path.name} [{label}] → {save_path.name}")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr esrgan esrgan_blur ...)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str], device: str) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    results = []
    all_known = {**{k: None for k in SR_CONFIGS}, **BASELINE_CONFIGS}
    print(f"\n── OCR sur {len(configs_to_run)} config(s) × {len(images)} image(s) ──")

    for img_path in images:
        for config_name in configs_to_run:
            if config_name not in all_known:
                print(f"  [SKIP] '{config_name}' inconnu. Configs : {', '.join(all_known)}")
                continue

            cfg = Config(prompt_mode="layout")
            out_md = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            print(f"  [{img_path.stem} {config_name}] ...", end=" ", flush=True)

            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "latency": 0.0, "error": ""}
            try:
                if config_name in SR_CONFIGS:
                    _check_npu_setup()
                    postprocess_key = SR_CONFIGS[config_name]
                    img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
                    preprocessed_path = _apply_sr(img_path, device, cfg, postprocess_key, img_file)
                else:
                    img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
                    preprocessed_path = _apply_baseline(img_path, config_name, cfg, img_file)

                text, metrics = ocr_image(preprocessed_path, vlm, cfg)

                if cfg.prompt_mode == "layout" and cfg.two_pass:
                    text, fig_metrics = process_figures(text, img_path, vlm, cfg, img_path.stem)
                    metrics["total_latency"] += fig_metrics["total_latency"]

                out_md.write_text(text, encoding="utf-8")
                row["words"]   = len(text.split())
                row["latency"] = metrics["total_latency"]
                row["looped"]  = metrics.get("looped", False)
                flag = " [BOUCLE]" if row["looped"] else ""
                print(f"{row['words']} mots ({row['latency']:.1f}s){flag}")
            except Exception as e:
                row["error"] = str(e)
                print(f"ERREUR: {e}")
            results.append(row)

    _write_ocr_report(results)


def _write_ocr_report(results: list[dict]) -> None:
    lines = [
        "# Rapport OCR — realesrgan AMD NPU\n",
        "| Page | Config | Boucle | Mots | Durée (s) | Note |",
        "|------|--------|--------|------|-----------|------|",
    ]
    for r in results:
        boucle = "**oui**" if r["looped"] else "non"
        lines.append(
            f"| {r['page']} | {r['config']} | {boucle} "
            f"| {r['words']} | {r['latency']:.1f} | {r.get('error', '')} |"
        )
    out = OUT_DIR / "ocr_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["npu", "cpu"],
                        help=f"Device SR (défaut: {DEFAULT_DEVICE}).")
    parser.add_argument("--ocr", nargs="*", metavar="CONFIG",
                        help="Sans valeur = toutes les configs SR + baselines.")
    parser.add_argument("--list", action="store_true",
                        help="Afficher les configs disponibles et quitter.")
    args = parser.parse_args()

    if args.list:
        print("Configs SR (AMD NPU realesrgan-128x128) :")
        for name, post in SR_CONFIGS.items():
            print(f"  {name:20s}" + (f"  + {post}" if post else ""))
        print("\nConfigs baseline :")
        for name, desc in BASELINE_CONFIGS.items():
            print(f"  {name:20s}  {desc}")
        return

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

    phase1_visualize(images, args.device)

    if args.ocr is None:
        return

    all_configs = list(SR_CONFIGS) + list(BASELINE_CONFIGS)
    configs_to_run = all_configs if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run, args.device)


if __name__ == "__main__":
    main()
