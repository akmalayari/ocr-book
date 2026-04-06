"""
report_laplacian.py — Rapport de netteté (variance du laplacien) pour toutes
les images de photos/.

Interprétation : plus la variance est élevée, plus l'image est nette.
Une valeur faible (< 100) indique généralement une image floue.

Sortie : output/laplacian_report.md

Usage :
    python draft/report_laplacian.py
"""

import sys
from pathlib import Path
from datetime import date

import cv2
from skimage.restoration import estimate_sigma

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from images import collect_images

OUT_FILE = Path(__file__).parent.parent / "output" / "laplacian_report.md"


def laplacian_variance(image_path: Path) -> float:
    img    = cv2.imread(str(image_path))
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(cv2.Laplacian(smooth, cv2.CV_64F).var())


def estimate_noise_level(image_path: Path) -> float:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return float(estimate_sigma(img, average_sigmas=True))


def main() -> None:
    cfg    = Config()
    images = collect_images(cfg)

    results = []
    for img_path in images:
        lap   = laplacian_variance(img_path)
        noise = estimate_noise_level(img_path)
        results.append((img_path.name, lap, noise))
        print(f"  {img_path.name:<30}  lap={lap:8.2f}  noise={noise:6.2f}")

    scores      = [lap   for _, lap, _     in results]
    noises      = [noise for _, _,   noise in results]
    threshold   = 100.0
    blurry      = [(n, l, s) for n, l, s in results if l < threshold]

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        return sum(vals) / len(vals), min(vals), max(vals)

    lap_mean, lap_min, lap_max       = _stats(scores)
    noise_mean, noise_min, noise_max = _stats(noises)

    lines = [
        f"# Laplacian variance report",
        f"",
        f"Date : {date.today()}  |  Images : {len(results)}  |  "
        f"Threshold (blurry) : `< {threshold:.0f}`",
        f"",
        f"| Stat | Laplacian variance | Noise sigma |",
        f"|------|--------------------|-------------|",
        f"| Mean | {lap_mean:.2f} | {noise_mean:.2f} |",
        f"| Min  | {lap_min:.2f} | {noise_min:.2f} |",
        f"| Max  | {lap_max:.2f} | {noise_max:.2f} |",
        f"| Blurry (< {threshold:.0f}) | {len(blurry)} / {len(results)} | — |",
        f"",
        f"## Per-image scores",
        f"",
        f"| Image | Laplacian variance | Noise sigma | Status |",
        f"|-------|--------------------|-------------|--------|",
    ]

    for name, lap, noise in results:
        status = "blurry" if lap < threshold else "ok"
        lines.append(f"| {name} | {lap:.2f} | {noise:.2f} | {status} |")

    if blurry:
        lines += [
            f"",
            f"## Blurry images",
            f"",
        ]
        for name, lap, noise in sorted(blurry, key=lambda x: x[1]):
            lines.append(f"- `{name}` — lap={lap:.2f}, noise={noise:.2f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {OUT_FILE}")


if __name__ == "__main__":
    main()
