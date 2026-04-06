"""
compare_ocr.py — Comparaison générale de sorties OCR.

Modifier la section CONFIG ci-dessous, puis lancer :
    python draft/compare_ocr.py

Modes de comparaison :
  - "all_vs_ref"  : chaque fichier comparé contre REFERENCE
  - "all_pairs"   : toutes les paires (NxN/2)
  - "sequential"  : chaque fichier comparé au suivant dans FILES

Sorties : OUTPUT_DIR/
  - diff_{a}_vs_{b}.md   pour chaque paire
  - global_report.md     tableau de similarité trié
"""

import re
import sys
import difflib
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare import compare, tokenize_words, tokenize_sentences
from postprocess import _clean_layout


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — seule section à modifier
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent.parent / "output"
PHOTOS = Path(__file__).parent.parent / "photos"

# Fichiers à comparer : liste de (chemin, label court)
FILES: list[tuple[Path, str]] = [
    (PHOTOS / "md"      / "page_6.md",              "authentic"),
    # page_6 (nette) — même contenu que la référence
    (ROOT / "pipeline" / "page_6_baseline.md",    "p6_baseline"),
    (ROOT / "pipeline" / "page_6_sauvola.md",     "p6_sauvola"),
    (ROOT / "nlmeans"  / "page_6_median_and.md",  "p6_median_and"),
    (ROOT / "nlmeans"  / "page_6_nlm5_median.md", "p6_nlm5_median"),
    (ROOT / "nlmeans"  / "page_6_nlmeans_10.md",  "p6_nlmeans_10"),
    (ROOT / "nlmeans"  / "page_6_nlm10_and.md",   "p6_nlm10_and"),
    (ROOT / "nlmeans"  / "page_6_nlm5_and.md",    "p6_nlm5_and"),
    # page_5 (bruitée) — nlm5_and exclu (4 mots, échec)
    (ROOT / "pipeline" / "page_5_baseline.md",    "p5_baseline"),
    (ROOT / "pipeline" / "page_5_sauvola.md",     "p5_sauvola"),
    (ROOT / "nlmeans"  / "page_5_median_and.md",  "p5_median_and"),
    (ROOT / "nlmeans"  / "page_5_nlm5_median.md", "p5_nlm5_median"),
    (ROOT / "nlmeans"  / "page_5_nlmeans_10.md",  "p5_nlmeans_10"),
    (ROOT / "nlmeans"  / "page_5_nlm10_and.md",   "p5_nlm10_and"),
]

# Référence pour le mode "all_vs_ref" et pour la similarité relative dans le rapport.
# Peut être None (premier fichier de FILES utilisé par défaut) ou un chemin explicite.
REFERENCE: Path | None = PHOTOS / "md" / "page_6.md"

# "all_vs_ref" | "all_pairs" | "sequential"
MODE_COMPARE: str = "all_vs_ref"

# "sentence" | "word"  — contenu des fichiers diff_{a}_vs_{b}.md
MODE_DIFF_FILE:   str = "sentence"
# "sentence" | "word"  — similarité affichée dans global_report.md
MODE_DIFF_REPORT: str = "word"

# True : regrouper les résultats par page (préfixe p5_, p6_, …) dans le rapport
GROUP_BY_PAGE: bool = True

OUTPUT_DIR: Path = ROOT / "compare_ocr"

# ══════════════════════════════════════════════════════════════════════════════


_HTML_TAG_RE = re.compile(r'<[^>]+>')


def _normalize(text: str) -> str:
    text = _clean_layout(text)
    text = _HTML_TAG_RE.sub(' ', text)
    text = re.sub(r'(\w)- (\w)',   r'\1\2',  text)
    text = re.sub(r'(\w)-\n(\w)',  r'\1\2',  text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ',   text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _sim(path_a: Path, path_b: Path) -> float:
    tok = tokenize_sentences if MODE_DIFF_REPORT == "sentence" else tokenize_words
    ta = tok(path_a.read_text(encoding="utf-8"))
    tb = tok(path_b.read_text(encoding="utf-8"))
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def _norm_file(src: Path, tmp_dir: Path, label: str) -> Path:
    safe = re.sub(r'[^\w]', '_', label)
    out = tmp_dir / f"{safe}.md"
    out.write_text(_normalize(src.read_text(encoding="utf-8")), encoding="utf-8")
    return out


def _pairs(files: list[tuple[Path, str]], ref: tuple[Path, str]) -> list[tuple[tuple, tuple]]:
    if MODE_COMPARE == "all_vs_ref":
        return [(ref, f) for f in files if f != ref]
    if MODE_COMPARE == "all_pairs":
        return list(combinations(files, 2))
    if MODE_COMPARE == "sequential":
        return list(zip(files, files[1:]))
    raise ValueError(f"MODE_COMPARE inconnu : {MODE_COMPARE}")


def main() -> None:
    # Validation
    missing = [str(p) for p, _ in FILES if not p.exists()]
    if missing:
        print("[ERREUR] Fichiers introuvables :")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    if len(FILES) < 2:
        print("[ERREUR] Au moins 2 fichiers requis dans FILES.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUTPUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Résoudre la référence
    if REFERENCE is not None:
        ref_entry = next(((p, l) for p, l in FILES if p == REFERENCE), None)
        if ref_entry is None:
            ref_entry = (REFERENCE, REFERENCE.stem)
    else:
        ref_entry = FILES[0]

    # Normaliser tous les fichiers
    normed: dict[str, Path] = {}
    for src_path, label in FILES:
        normed[label] = _norm_file(src_path, tmp_dir, label)
    if REFERENCE is not None and ref_entry[1] not in normed:
        normed[ref_entry[1]] = _norm_file(ref_entry[0], tmp_dir, ref_entry[1])

    pairs = _pairs(FILES, ref_entry)
    if not pairs:
        print("Aucune paire à comparer.")
        sys.exit(0)

    results: list[dict] = []
    print(f"Mode : {MODE_COMPARE}  |  diff : {MODE_DIFF_FILE}  |  report : {MODE_DIFF_REPORT}  |  {len(pairs)} paire(s)\n")

    for (pa, la), (pb, lb) in pairs:
        out_path = OUTPUT_DIR / f"diff_{re.sub(r'[^\w]', '_', la)}_vs_{re.sub(r'[^\w]', '_', lb)}.md"
        print(f"  {la}  vs  {lb} ...", end=" ", flush=True)
        compare(normed[la], normed[lb], mode=MODE_DIFF_FILE, out_path=out_path)
        sim = _sim(normed[la], normed[lb])
        results.append({"a": la, "b": lb, "similarity": sim, "diff": out_path})

    _write_global_report(ref_entry[1], results, normed)


_RE_PAGE = re.compile(r'^(p\d+)_')


def _page_key(label: str) -> str:
    m = _RE_PAGE.match(label)
    return m.group(1) if m else label


def _write_global_report(ref_label: str, results: list[dict], normed: dict[str, Path]) -> None:
    lines = [
        "# Rapport global — compare_ocr\n",
        f"Mode : `{MODE_COMPARE}` | Diff : `{MODE_DIFF_FILE}` | Report : `{MODE_DIFF_REPORT}` | Référence : `{ref_label}`\n",
    ]

    if GROUP_BY_PAGE:
        # Regrouper par page (clé extraite du label B en mode all_vs_ref, sinon B)
        from itertools import groupby
        keyed = sorted(results, key=lambda r: (_page_key(r["b"]), -r["similarity"]))
        for page, group in groupby(keyed, key=lambda r: _page_key(r["b"])):
            rows = list(group)
            lines += [
                f"## {page}\n",
                "| Config | Similarité | Diff |",
                "|--------|-----------|------|",
            ]
            for r in rows:
                diff_link = f"[diff]({r['diff'].name})"
                lines.append(f"| {r['b']} | {r['similarity']:.1%} | {diff_link} |")
            lines.append("")
    else:
        sorted_results = sorted(results, key=lambda r: -r["similarity"])
        lines += [
            "| A | B | Similarité | Diff |",
            "|---|---|-----------|------|",
        ]
        for r in sorted_results:
            diff_link = f"[diff]({r['diff'].name})"
            lines.append(f"| {r['a']} | {r['b']} | {r['similarity']:.1%} | {diff_link} |")

    # Mots par fichier
    lines += ["", "## Nombre de mots par fichier (après normalisation)\n",
              "| Fichier | Mots |", "|---------|------|"]
    for label, path in normed.items():
        if path.name.startswith("_"):
            continue
        words = len(path.read_text(encoding="utf-8").split())
        lines.append(f"| {label} | {words} |")

    out = OUTPUT_DIR / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
