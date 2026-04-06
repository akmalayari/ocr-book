"""
test_nlmeans.py — Test de fastNlMeansDenoising avant binarisation adaptative.

Compare plusieurs valeurs de h (force du filtre) contre baseline et sauvola existants.
Pages cibles : page_4, page_9 (boucles persistantes), page_5 pour comparaison croisée.

Phase 1 (défaut)    : génère les images prétraitées (baseline, sauvola, nlmeans_*).
Phase 2 (--ocr)     : lance l'OCR uniquement sur les configs nlmeans_*.
Phase 3 (--compare) : compare nlmeans vs fichiers de référence existants (REFS).

Sorties : output/nlmeans/

Usage :
    python draft/test_nlmeans.py
    python draft/test_nlmeans.py --ocr
    python draft/test_nlmeans.py --ocr nlmeans_10 nlmeans_15
    python draft/test_nlmeans.py --compare
    python draft/test_nlmeans.py --list
"""

import argparse
import re
import sys
import difflib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image

PHOTOS_DIR = Path(__file__).parent.parent / "photos"
OUT_DIR    = Path(__file__).parent.parent / "output" / "nlmeans"
REF_DIR    = Path(__file__).parent.parent / "output" / "sauvola_patch"

DEFAULT_PAGES = ["page_4", "page_9", "page_5"]

# Configs OCR-éligibles (phase 2)
OCR_CONFIGS: dict[str, str] = {
    "median_and":  "medianBlur(3) + AND(Sauvola w=51, adaptive)",
    "nlm5_median": "nlmeans(h=5) + medianBlur(3) + adaptive",
    "nlmeans_10":  "nlmeans(h=10) + adaptive",
    "nlm10_and":   "nlmeans(h=10) + AND(Sauvola w=51, adaptive)",
    "nlm5_and":    "nlmeans(h=5) + AND(Sauvola w=51, adaptive)",
}

# Toutes les configs pour la visualisation phase 1 (OCR + viz-only)
VIZ_CONFIGS: dict[str, str] = {
    "baseline":           "GaussianBlur(5,5) + adaptive C=15",
    "sauvola_and":            "AND(Sauvola w=51 k=0.3, baseline)",
    **OCR_CONFIGS,
}

# Références existantes pour la comparaison (label → chemin du .md)
# Format : "{page_stem}_{label}" → chemin
# Remplir avec les fichiers déjà générés.
REFS: dict[str, Path] = {
    # ex: "page_4_baseline": REF_DIR / "page_4_baseline.md",
    #     "page_4_and_51_03": REF_DIR / "page_4_and_51_03.md",
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _baseline(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _sauvola(gray: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_sauvola
    thresh   = threshold_sauvola(gray, window_size=51, k=0.3)
    sauvola  = ((gray > thresh).astype(np.uint8)) * 255
    baseline = _baseline(gray)
    return cv2.bitwise_and(sauvola, baseline)


def _nlmeans(gray: np.ndarray, h: int) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _nlmeans_sauvola(gray: np.ndarray, h: int) -> np.ndarray:
    """fastNlMeans + Sauvola seul (sans AND avec adaptive)."""
    from skimage.filters import threshold_sauvola
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    thresh   = threshold_sauvola(denoised, window_size=51, k=0.3)
    return ((denoised > thresh).astype(np.uint8)) * 255


def _nlmeans_and(gray: np.ndarray, h: int) -> np.ndarray:
    """fastNlMeans + AND(Sauvola, adaptive) — équivalent sauvola_binarize mais sur image débruitée."""
    from skimage.filters import threshold_sauvola
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    thresh   = threshold_sauvola(denoised, window_size=51, k=0.3)
    sauvola  = ((denoised > thresh).astype(np.uint8)) * 255
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    return cv2.bitwise_and(sauvola, adaptive)


def _nlmeans_median(gray: np.ndarray, h: int) -> np.ndarray:
    """nlmeans + medianBlur(3) + adaptive — supprime les granules résiduels."""
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    median   = cv2.medianBlur(denoised, 3)
    return cv2.adaptiveThreshold(
        median, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _nlmeans_open(gray: np.ndarray, h: int) -> np.ndarray:
    """nlmeans + adaptive + MORPH_OPEN(2×2) — élimine les pixels isolés post-binarisation."""
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    bw = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


def _bg_divide(gray: np.ndarray, blur_ksize: int = 101) -> np.ndarray:
    bg   = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0).astype(np.float32)
    bg   = np.where(bg < 1, 1, bg)
    return (gray.astype(np.float32) / bg * 128).clip(0, 255).astype(np.uint8)


def _nlmeans_bgdiv(gray: np.ndarray, h: int) -> np.ndarray:
    """nlmeans + bg_divide + adaptive."""
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    normed   = _bg_divide(denoised)
    return cv2.adaptiveThreshold(
        normed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _nlmeans_bgdiv_and(gray: np.ndarray, h: int) -> np.ndarray:
    """nlmeans + bg_divide + AND(Sauvola, adaptive)."""
    from skimage.filters import threshold_sauvola
    denoised = cv2.fastNlMeansDenoising(gray, h=h)
    normed   = _bg_divide(denoised)
    thresh   = threshold_sauvola(normed, window_size=51, k=0.3)
    sauvola  = ((normed > thresh).astype(np.uint8)) * 255
    adaptive = cv2.adaptiveThreshold(
        normed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    return cv2.bitwise_and(sauvola, adaptive)


def _median_adaptive(gray: np.ndarray) -> np.ndarray:
    """medianBlur(3) + adaptive."""
    median = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        median, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _median_and(gray: np.ndarray) -> np.ndarray:
    """medianBlur(3) + AND(Sauvola, adaptive)."""
    from skimage.filters import threshold_sauvola
    median   = cv2.medianBlur(gray, 3)
    thresh   = threshold_sauvola(median, window_size=51, k=0.3)
    sauvola  = ((median > thresh).astype(np.uint8)) * 255
    adaptive = cv2.adaptiveThreshold(
        median, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    return cv2.bitwise_and(sauvola, adaptive)


def _get_image(img_path: Path, config_name: str) -> np.ndarray:
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if config_name == "baseline":
        return _baseline(gray)
    if config_name == "sauvola":
        return _sauvola(gray)
    if config_name == "nlmeans_10_raw":
        return cv2.fastNlMeansDenoising(gray, h=10)
    if config_name == "nlmeans_10_sauvola":
        return _nlmeans_sauvola(gray, h=10)
    if config_name == "nlmeans_5":
        return _nlmeans(gray, 5)
    if config_name == "nlmeans_10":
        return _nlmeans(gray, 10)
    if config_name == "nlm5_median":
        return _nlmeans_median(gray, 5)
    if config_name == "nlm5_open":
        return _nlmeans_open(gray, 5)
    if config_name == "nlm5_and":
        return _nlmeans_and(gray, 5)
    if config_name == "nlm10_and":
        return _nlmeans_and(gray, 10)
    if config_name == "nlm10_bgdiv":
        return _nlmeans_bgdiv(gray, 10)
    if config_name == "nlm10_bgdiv_and":
        return _nlmeans_bgdiv_and(gray, 10)
    if config_name == "median_adaptive":
        return _median_adaptive(gray)
    if config_name == "median_and":
        return _median_and(gray)

    raise ValueError(f"Config inconnue : {config_name}")


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path]) -> None:
    print("── Génération des images prétraitées ───────────────────────")
    for img_path in images:
        n = 0
        for config_name in VIZ_CONFIGS:
            result = _get_image(img_path, config_name)
            cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_{config_name}.jpg"), result)
            n += 1
        print(f"  {img_path.name} → {n} fichiers")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr nlm5_and nlm10_and ...)")


# ── Phase 2 : OCR (nlmeans uniquement) ───────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str]) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout", preprocess_mode="none")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    results = []
    print(f"\n── OCR sur {len(configs_to_run)} config(s) × {len(images)} image(s) ──")

    for img_path in images:
        for config_name in configs_to_run:
            if config_name not in OCR_CONFIGS:
                print(f"  [SKIP] '{config_name}' n'est pas une config OCR.")
                print(f"  Configs OCR disponibles : {', '.join(OCR_CONFIGS)}")
                continue

            preprocessed = _get_image(img_path, config_name)
            img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            cv2.imwrite(str(img_file), preprocessed)

            out_md = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            print(f"  [{img_path.stem} {config_name}] ...", end=" ", flush=True)
            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "latency": 0.0, "error": ""}
            try:
                text, metrics = ocr_image(img_file, vlm, cfg_base)
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
    print("\nPour comparer : --compare")


def _write_ocr_report(results: list[dict]) -> None:
    lines = [
        "# Rapport OCR — nlmeans\n",
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


# ── Phase 3 : comparaison ─────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    from postprocess import _clean_layout
    _HTML_TAG = re.compile(r'<[^>]+>')
    text = _clean_layout(text)
    text = _HTML_TAG.sub(' ', text)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _sim(path_a: Path, path_b: Path) -> float:
    from draft.obsolete.compare import tokenize_words
    ta = tokenize_words(path_a.read_text(encoding="utf-8"))
    tb = tokenize_words(path_b.read_text(encoding="utf-8"))
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def phase3_compare(images: list[Path]) -> None:
    from draft.obsolete.compare import compare

    cmp_dir = OUT_DIR / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cmp_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Charger les sorties nlmeans générées
    normed: dict[str, Path] = {}
    for md in OUT_DIR.glob("*.md"):
        if md.stem == "ocr_report":
            continue
        n = tmp_dir / md.name
        n.write_text(_normalize(md.read_text(encoding="utf-8")), encoding="utf-8")
        normed[md.stem] = n

    # Charger les références existantes (baseline, sauvola, ...)
    missing_refs: list[str] = []
    for label, ref_path in REFS.items():
        if not ref_path.exists():
            missing_refs.append(f"{label} → {ref_path}")
            continue
        n = tmp_dir / f"ref_{label}.md"
        n.write_text(_normalize(ref_path.read_text(encoding="utf-8")), encoding="utf-8")
        normed[label] = n

    if missing_refs:
        print("[WARN] Références introuvables (remplir REFS dans le script) :")
        for m in missing_refs:
            print(f"  {m}")

    summary: list[dict] = []
    print("\n── Comparaisons nlmeans vs références ────────────────────")

    for img_path in images:
        # Refs disponibles pour cette page
        page_refs = {k: v for k, v in normed.items()
                     if k.startswith(img_path.stem + "_") and
                     not any(k == f"{img_path.stem}_{c}" for c in OCR_CONFIGS)}

        for config_name in OCR_CONFIGS:
            cand_key = f"{img_path.stem}_{config_name}"
            if cand_key not in normed:
                print(f"  [SKIP] {cand_key}.md manquant — lancer --ocr d'abord")
                continue

            for ref_key, ref_path in page_refs.items():
                ref_label = ref_key.replace(img_path.stem + "_", "")
                out_path  = cmp_dir / f"diff_{img_path.stem}_{ref_label}_vs_{config_name}.md"
                print(f"  {img_path.stem} {ref_label} vs {config_name} ...", end=" ", flush=True)
                compare(normed[ref_key], normed[cand_key], mode="sentence", out_path=out_path)
                sim = _sim(normed[ref_key], normed[cand_key])
                summary.append({"page": img_path.stem, "ref": ref_label,
                                 "config": config_name, "sim": sim, "diff": out_path})
                print(f"{sim:.1%}")

    if not summary:
        print("\nAucune comparaison effectuée.")
        print("Vérifier que --ocr a été lancé et que REFS contient les chemins des sorties existantes.")
        return

    summary_sorted = sorted(summary, key=lambda r: (r["page"], r["ref"], -r["sim"]))
    lines = [
        "# Rapport global — nlmeans\n",
        "| Page | Référence | Config nlmeans | Similarité | Diff |",
        "|------|-----------|---------------|-----------|------|",
    ]
    for r in summary_sorted:
        diff_link = f"[diff]({r['diff'].name})"
        lines.append(
            f"| {r['page']} | {r['ref']} | {r['config']} | {r['sim']:.1%} | {diff_link} |"
        )

    out = cmp_dir / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → rapport global : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--ocr", nargs="*", metavar="CONFIG",
                        help="Sans valeur = toutes les configs nlmeans.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Afficher les configs disponibles et quitter.")
    args = parser.parse_args()

    if args.list:
        print("Configs visualisation :")
        for name, desc in VIZ_CONFIGS.items():
            print(f"  {name:20s}  {desc}")
        print("\nConfigs OCR :")
        for name, desc in OCR_CONFIGS.items():
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

    if args.compare:
        phase3_compare(images)
        return

    phase1_visualize(images)

    if args.ocr is None:
        return

    configs_to_run = list(OCR_CONFIGS.keys()) if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run)


if __name__ == "__main__":
    main()
