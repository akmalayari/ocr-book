"""
compare_grid.py — Compare les sorties OCR de binarize_grid/ contre une référence.

Exclut les configs qui ont bouclé (C=10, suffixe _10), sauf la référence.
Pour chaque candidat : un fichier de diff en md.
En fin : un rapport global de similarité.

Usage :
    python draft/compare_grid.py
    python draft/compare_grid.py --ref output/binarize_grid/page_6_31_10.md
    python draft/compare_grid.py --mode sentence
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from compare import compare, tokenize_sentences, tokenize_words
from postprocess import _clean_layout
import difflib


_HTML_TAG_RE = re.compile(r'<[^>]+>')


def _normalize(text: str) -> str:
    """Retire les balises grounding et HTML pour une comparaison sur le contenu textuel."""
    text = _clean_layout(text)
    text = _HTML_TAG_RE.sub(' ', text)
    # Réassembler les mots coupés par un tiret + espace ("structu- relle" → "structurelle")
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    # Fusionner les retours à la ligne simples (wrapping) en espaces,
    # en préservant les doubles sauts de ligne (séparateurs de paragraphes)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

GRID_DIR = Path(__file__).parent.parent / "output" / "binarize_grid"
OUT_DIR  = GRID_DIR / "comparisons"

# ── Configuration ────────────────────────────────────────────────────────────
REFERENCES = [
    GRID_DIR / "page_6_31_10.md",
    GRID_DIR / "page_6_31_15.md",
    GRID_DIR / "page_6_21_15.md",
    GRID_DIR / "page_5_21_15.md",
    GRID_DIR / "page_5_31_15.md"
]

# Configs ayant bouclé — on les exclut sauf si elles sont dans REFERENCES
LOOPED_SUFFIXES = {"_31_10", "_21_10"}

# ─────────────────────────────────────────────────────────────────────────────


def is_looped(path: Path) -> bool:
    if path in REFERENCES:
        return False
    return any(path.stem.endswith(s) for s in LOOPED_SUFFIXES)


def similarity_ratio(path_a: Path, path_b: Path, mode: str) -> float:
    text_a = path_a.read_text(encoding="utf-8")
    text_b = path_b.read_text(encoding="utf-8")
    tok_fn = tokenize_sentences if mode == "sentence" else tokenize_words
    return difflib.SequenceMatcher(None, tok_fn(text_a), tok_fn(text_b), autojunk=False).ratio()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ref in REFERENCES:
        if not ref.exists():
            print(f"[ERREUR] Référence introuvable : {ref}")
            continue
        _run_for_ref(ref)


def _run_for_ref(ref: Path) -> None:
    candidates = sorted(
        p for p in GRID_DIR.glob("*.md")
        if p != ref
        and not is_looped(p)
        and not p.stem.startswith(("laplacian", "ocr_loop"))
    )

    if not candidates:
        print("Aucun candidat à comparer.")
        sys.exit(0)

    print(f"Référence : {ref.name}")
    print(f"Candidats : {len(candidates)}\n")

    # Fichiers normalisés (temporaires)
    tmp_dir = OUT_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ref_norm = tmp_dir / ref.name
    ref_norm.write_text(_normalize(ref.read_text(encoding="utf-8")), encoding="utf-8")

    results = []
    for cand in candidates:
        out_path = OUT_DIR / f"diff_{ref.stem}_vs_{cand.stem}.md"
        cand_norm = tmp_dir / cand.name
        cand_norm.write_text(_normalize(cand.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"  {cand.name} ...", end=" ", flush=True)
        compare(ref_norm, cand_norm, mode="sentence", out_path=out_path)
        sim = similarity_ratio(ref_norm, cand_norm, "word")
        results.append({"name": cand.stem, "similarity": sim, "out": out_path})

    _write_global_report(ref, results)


def _write_global_report(ref: Path, results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r["similarity"], reverse=True)
    lines = [
        "# Rapport global — Comparaison binarize_grid\n",
        f"Référence : `{ref.name}`  ",
        f"Similarité : word-level | Diff : sentence-level\n",
        "| Config | Similarité | Diff |",
        "|--------|-----------|------|",
    ]
    for r in results_sorted:
        diff_link = f"[diff]({r['out'].name})"
        lines.append(f"| {r['name']} | {r['similarity']:.1%} | {diff_link} |")

    out = OUT_DIR / f"global_report_{ref.stem}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → rapport global : {out}")


if __name__ == "__main__":
    main()
