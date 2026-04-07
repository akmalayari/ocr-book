"""
compare_ocr.py — Comparaison générale de sorties OCR.

Modifier la section CONFIG ci-dessous, puis lancer :
    python draft/compare_ocr.py

Modes de comparaison (MODE_COMPARE) :
  - "all_vs_ref"  : chaque fichier comparé contre REFERENCE
  - "all_pairs"   : toutes les paires (NxN/2)
  - "sequential"  : chaque fichier comparé au suivant dans FILES

Mode composant (SCORE_BY_COMPONENT = True) :
  Calcule séparément texte / figure / global pour chaque fichier vs ses
  références dédiées. Remplace le rapport pairwise par un tableau par page.

Sorties : OUTPUT_DIR/
  - diff_{a}_vs_{b}.md   pour chaque paire / composant
  - global_report.md     tableau de similarité trié
"""

import html
import re
import sys
import difflib
from itertools import combinations, groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare import compare, tokenize_words, tokenize_sentences, weighted_ratio
from postprocess import _clean_layout



# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — seule section à modifier
# ══════════════════════════════════════════════════════════════════════════════

ROOT   = Path(__file__).parent.parent / "output"
PHOTOS = Path(__file__).parent.parent / "photos"

# Fichiers à comparer : liste de (chemin, label court)
FILES: list[tuple[Path, str]] = [
    (ROOT / "paddle_ocr" / "page_4.md",        "p4_paddle"),
    (ROOT / "paddle_ocr" / "page_4_clean.md","p4c_paddle"),
]

# Référence globale (mode all_vs_ref et composant global)
REFERENCE: Path | None = PHOTOS / "md" / "page_6.md"

# Références par composant (utilisées si SCORE_BY_COMPONENT = True)
TEXT_REFERENCE: Path | None = PHOTOS / "md" / "page_6_text.md"
FIG_REFERENCE:  Path | None = None

# True : score séparé texte / figure / global par fichier (remplace MODE_COMPARE)
SCORE_BY_COMPONENT: bool = True

# "all_vs_ref" | "all_pairs" | "sequential"  (ignoré si SCORE_BY_COMPONENT)
MODE_COMPARE: str = "all_pairs"

# "sentence" | "word"  — contenu des fichiers diff_{a}_vs_{b}.md
MODE_DIFF_FILE:   str = "sentence"
# "sentence" | "word"  — similarité affichée dans global_report.md
MODE_DIFF_REPORT: str = "word"

# True : pondérer les différences par Levenshtein (partial credit pour near-misses)
USE_LEVENSHTEIN: bool = True

# True : regrouper les résultats par page (préfixe p5_, p6_, …) dans le rapport
GROUP_BY_PAGE: bool = True

OUTPUT_DIR: Path = ROOT / "compare_paddle"

# ══════════════════════════════════════════════════════════════════════════════


_HTML_TAG_RE    = re.compile(r'<[^>]+>')
_TABLE_BLOCK_RE = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
_DIV_BLOCK_RE   = re.compile(r'<div[^>]*>[\s\S]*?</div>', re.IGNORECASE)
_DET_LINE_RE    = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>")
_NON_TEXT_LABELS = {"image", "table"}
_RE_PAGE        = re.compile(r'^(p[^_]+)_')
_TABLE_NOTE_RE = re.compile(
    r'(?m)^(Champ|Lecture|Sources?|Note|Tableau\b|Document\b)[^\n]*$', re.IGNORECASE
)


def _normalize(text: str, strip_tables: bool = False, strip_notes: bool = False) -> str:
    text = html.unescape(text)
    text = _clean_layout(text)
    if strip_tables:
        text = _TABLE_BLOCK_RE.sub(' ', text)
        text = _DIV_BLOCK_RE.sub(' ', text)
    text = _HTML_TAG_RE.sub(' ', text)
    text = re.sub(r'[–—]',             '-',     text)
    text = re.sub(r'(\w)-\s*(\w)',     r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ',     text)
    text = re.sub(r' ([;:!?»])',       r'\1',   text)  # espace avant ponctuation FR
    text = re.sub(r"[\u2018\u2019\u201b\u02bc\u02bb\u00b4`]", "'", text)  # apostrophes → droite
    text = re.sub(r' {2,}', ' ', text)
    if strip_notes:
        text = _TABLE_NOTE_RE.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_layout(text: str) -> tuple[str, str, bool]:
    """Split raw layout OCR en (text_content, non_text_content, non_text_detected).

    non_text_detected = True si au moins un bloc label='image' ou 'table' est présent.
    """
    text_parts:     list[str] = []
    non_text_parts: list[str] = []
    current_label = "text"
    current_lines: list[str] = []
    non_text_detected = False

    for line in text.splitlines():
        m = _DET_LINE_RE.search(line)
        if m:
            block = "\n".join(current_lines).strip()
            if block:
                (non_text_parts if current_label in _NON_TEXT_LABELS else text_parts).append(block)
            current_label = m.group(1)
            if current_label in _NON_TEXT_LABELS:
                non_text_detected = True
            current_lines = []
        else:
            current_lines.append(line)

    block = "\n".join(current_lines).strip()
    if block:
        (non_text_parts if current_label in _NON_TEXT_LABELS else text_parts).append(block)

    return "\n\n".join(text_parts), "\n\n".join(non_text_parts), non_text_detected


def _sim(path_a: Path, path_b: Path) -> float:
    tok = tokenize_sentences if MODE_DIFF_REPORT == "sentence" else tokenize_words
    ta = tok(path_a.read_text(encoding="utf-8"))
    tb = tok(path_b.read_text(encoding="utf-8"))
    if USE_LEVENSHTEIN:
        return weighted_ratio(ta, tb)
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def _write_norm(content: str, tmp_dir: Path, name: str,
                strip_tables: bool = False, strip_notes: bool = False) -> Path:
    out = tmp_dir / f"{re.sub(r'[^\w]', '_', name)}.md"
    out.write_text(_normalize(content, strip_tables=strip_tables, strip_notes=strip_notes), encoding="utf-8")
    return out


def _norm_file(src: Path, tmp_dir: Path, label: str,
               strip_tables: bool = False, strip_notes: bool = False) -> Path:
    return _write_norm(src.read_text(encoding="utf-8"), tmp_dir, label,
                       strip_tables=strip_tables, strip_notes=strip_notes)


def _page_key(label: str) -> str:
    m = _RE_PAGE.match(label)
    return m.group(1) if m else label


def _pairs(files: list[tuple[Path, str]], ref: tuple[Path, str]) -> list[tuple[tuple, tuple]]:
    if MODE_COMPARE == "all_vs_ref":
        return [(ref, f) for f in files if f != ref]
    if MODE_COMPARE == "all_pairs":
        return list(combinations(files, 2))
    if MODE_COMPARE == "sequential":
        return list(zip(files, files[1:]))
    raise ValueError(f"MODE_COMPARE inconnu : {MODE_COMPARE}")


# ── Mode composant ─────────────────────────────────────────────────────────────

def _run_component_mode(files: list[tuple[Path, str]], ref_path: Path,
                         text_ref: Path | None, fig_ref: Path | None,
                         tmp_dir: Path) -> None:
    ref_norm      = _norm_file(ref_path,  tmp_dir, "ref_global")
    text_ref_norm = _norm_file(text_ref,  tmp_dir, "ref_text",  strip_tables=True, strip_notes=True) if text_ref else None
    fig_ref_norm  = _norm_file(fig_ref,   tmp_dir, "ref_fig")   if fig_ref   else None

    rows: list[dict] = []
    non_ref = [(p, l) for p, l in files if p != ref_path]

    print(f"Mode composant | {len(non_ref)} fichier(s)\n")

    for src_path, label in non_ref:
        raw = src_path.read_text(encoding="utf-8")
        text_raw, non_text_raw, non_text_detected = _split_layout(raw)

        # Normaliser les composants
        global_norm   = _write_norm(raw,          tmp_dir, f"{label}_global")
        text_norm     = _write_norm(text_raw,     tmp_dir, f"{label}_text", strip_tables=True)
        non_text_norm = _write_norm(non_text_raw, tmp_dir, f"{label}_fig")

        # Scores
        global_sim = _sim(ref_norm, global_norm)
        text_sim   = _sim(text_ref_norm, text_norm)       if text_ref_norm else None
        fig_sim    = _sim(fig_ref_norm,  non_text_norm)   if (fig_ref_norm and non_text_detected) else None

        # Diffs
        def _diff(norm_a: Path, norm_b: Path, suffix: str) -> Path:
            out = OUTPUT_DIR / f"diff_{re.sub(r'[^\w]', '_', label)}_{suffix}.md"
            compare(norm_a, norm_b, mode=MODE_DIFF_FILE, out_path=out)
            return out

        diff_global = _diff(ref_norm,      global_norm,   "global")
        diff_text   = _diff(text_ref_norm, text_norm,     "text")  if text_ref_norm else None
        diff_fig    = _diff(fig_ref_norm,  non_text_norm, "fig")   if fig_ref_norm  else None

        rows.append({
            "label":            label,
            "non_text_detected": non_text_detected,
            "global_sim":       global_sim,
            "text_sim":         text_sim,
            "fig_sim":          fig_sim,
            "diff_global":      diff_global,
            "diff_text":        diff_text,
            "diff_fig":         diff_fig,
        })

        parts = [f"global={global_sim:.1%}"]
        if text_sim is not None: parts.append(f"texte={text_sim:.1%}")
        non_text_str = f"non-texte={'oui' if non_text_detected else 'non'}"
        if fig_sim is not None: non_text_str += f" {fig_sim:.1%}"
        parts.append(non_text_str)
        print(f"  {label:25s}  {' | '.join(parts)}")

    _write_component_report(rows)


def _write_component_report(rows: list[dict]) -> None:
    lines = [
        "# Rapport composant — compare_ocr\n",
        f"Diff : `{MODE_DIFF_FILE}` | Score : `{MODE_DIFF_REPORT}`\n",
    ]

    keyed = sorted(rows, key=lambda r: (_page_key(r["label"]), -r["global_sim"]))
    for page, group in groupby(keyed, key=lambda r: _page_key(r["label"])):
        page_rows = list(group)
        lines += [
            f"## {page}\n",
            "| Config | Texte % | Non-texte détecté | Non-texte % | Global % | Diffs |",
            "|--------|---------|:-----------------:|-------------|----------|-------|",
        ]
        for r in page_rows:
            text_s        = f"{r['text_sim']:.1%}" if r["text_sim"] is not None else "—"
            non_text_icon = "oui" if r["non_text_detected"] else "non"
            non_text_s    = f"{r['fig_sim']:.1%}"  if r["fig_sim"]  is not None else "—"
            diffs = []
            if r["diff_global"]: diffs.append(f"[G]({r['diff_global'].name})")
            if r["diff_text"]:   diffs.append(f"[T]({r['diff_text'].name})")
            if r["diff_fig"]:    diffs.append(f"[F]({r['diff_fig'].name})")
            lines.append(
                f"| {r['label']} | {text_s} | {non_text_icon} | {non_text_s} | {r['global_sim']:.1%} | {' '.join(diffs)} |"
            )
        lines.append("")

    out = OUTPUT_DIR / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → {out}")


# ── Mode pairwise ──────────────────────────────────────────────────────────────

def _run_pairwise_mode(files: list[tuple[Path, str]], ref_entry: tuple[Path, str],
                        normed: dict[str, Path]) -> None:
    pairs = _pairs(files, ref_entry)
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
        print(f"{sim:.1%}")

    _write_pairwise_report(ref_entry[1], results, normed)


def _write_pairwise_report(ref_label: str, results: list[dict], normed: dict[str, Path]) -> None:
    lines = [
        "# Rapport global — compare_ocr\n",
        f"Mode : `{MODE_COMPARE}` | Diff : `{MODE_DIFF_FILE}` | Report : `{MODE_DIFF_REPORT}` | Référence : `{ref_label}`\n",
    ]

    if GROUP_BY_PAGE:
        keyed = sorted(results, key=lambda r: (_page_key(r["b"]), -r["similarity"]))
        for page, group in groupby(keyed, key=lambda r: _page_key(r["b"])):
            rows = list(group)
            if MODE_COMPARE == "all_pairs":
                lines += [f"## {page}\n", "| Config | Vs | Similarité | Diff |", "|--------|---|-----------|------|"]
                for r in rows:
                    lines.append(f"| {r['b']} | {r['a']} | {r['similarity']:.1%} | [diff]({r['diff'].name}) |")
            else:
                lines += [f"## {page}\n", "| Config | Similarité | Diff |", "|--------|-----------|------|"]
                for r in rows:
                    lines.append(f"| {r['b']} | {r['similarity']:.1%} | [diff]({r['diff'].name}) |")
            lines.append("")
    else:
        sorted_results = sorted(results, key=lambda r: -r["similarity"])
        lines += ["| A | B | Similarité | Diff |", "|---|---|-----------|------|"]
        for r in sorted_results:
            lines.append(f"| {r['a']} | {r['b']} | {r['similarity']:.1%} | [diff]({r['diff'].name}) |")

    lines += ["", "## Mots par fichier (après normalisation)\n", "| Fichier | Mots |", "|---------|------|"]
    for label, path in normed.items():
        words = len(path.read_text(encoding="utf-8").split())
        lines.append(f"| {label} | {words} |")

    out = OUTPUT_DIR / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    missing = [str(p) for p, _ in FILES if not p.exists()]
    if missing:
        print("[ERREUR] Fichiers introuvables :")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUTPUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    ref_path = REFERENCE if REFERENCE is not None else FILES[0][0]

    if SCORE_BY_COMPONENT and MODE_COMPARE != "all_pairs":
        _run_component_mode(FILES, ref_path, TEXT_REFERENCE, FIG_REFERENCE, tmp_dir)
        return

    # Mode pairwise — compare le texte pur (strip_notes=True)
    ref_entry = next(((p, l) for p, l in FILES if p == ref_path), (ref_path, ref_path.stem))
    normed: dict[str, Path] = {}
    for src_path, label in FILES:
        raw = src_path.read_text(encoding="utf-8")
        text_raw, _, _ = _split_layout(raw)
        normed[label] = _write_norm(text_raw, tmp_dir, label, strip_notes=True)
    if ref_entry[1] not in normed:
        raw = ref_entry[0].read_text(encoding="utf-8")
        text_raw, _, _ = _split_layout(raw)
        normed[ref_entry[1]] = _write_norm(text_raw, tmp_dir, ref_entry[1], strip_notes=True)
    _run_pairwise_mode(FILES, ref_entry, normed)


if __name__ == "__main__":
    main()
