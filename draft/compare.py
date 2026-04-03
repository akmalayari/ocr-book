"""Compare two markdown files word by word or sentence by sentence."""
import sys
import re
import difflib
import argparse
from pathlib import Path
from datetime import datetime


def tokenize_words(text):
    return re.findall(r'\S+|\s+', text)


def tokenize_sentences(text):
    """Split text into sentences.

    - Titles/subtitles: short lines or lines without terminal punct → kept as-is
    - Paragraphs: split on the whitespace between `. Capital` boundaries
      Lookbehind on 2 chars (lowercase/digit + punctuation) avoids splitting
      abbreviations like 'I.', 'M.', 'Fig.' where the char before '.' is uppercase.
    """
    sentences = []
    for line in re.split(r'\n+', text.strip()):
        line = line.strip()
        if not line:
            continue
        # Split only on whitespace that follows [lowercase/digit/closing][.!?]
        # and precedes an uppercase letter — punctuation stays with its sentence
        parts = re.split(r'(?<=[a-zà-ÿ0-9\)\»][.!?])\s+(?=[A-ZÀ-Ÿ])', line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def interpret(similarity, n_diffs):
    if similarity == 1.0:
        return "Identical — no differences found."
    if similarity >= 0.95:
        return f"Nearly identical — {n_diffs} minor difference(s)."
    if similarity >= 0.80:
        return f"Similar — {n_diffs} noticeable difference(s)."
    if similarity >= 0.50:
        return f"Partially similar — {n_diffs} significant difference(s)."
    return f"Very different — {n_diffs} difference(s), low overlap."


def build_diff_blocks_words(tokens_a, tokens_b, opcodes, context):
    blocks = []
    for tag, a0, a1, b0, b1 in opcodes:
        if tag == "equal":
            continue
        ctx_start_a = max(0, a0 - context)
        ctx_end_a   = min(len(tokens_a), a1 + context)
        before  = "".join(tokens_a[ctx_start_a:a0]).strip()
        after   = "".join(tokens_a[a1:ctx_end_a]).strip()
        removed = "".join(tokens_a[a0:a1]).strip()
        added   = "".join(tokens_b[b0:b1]).strip()
        blocks.append((before, removed, added, after))
    return blocks


def pair_by_similarity(sents_a, sents_b):
    """Pair sentences from two lists by word-level similarity (greedy best-match)."""
    if not sents_a:
        return [("", b) for b in sents_b]
    if not sents_b:
        return [(a, "") for a in sents_a]

    scores = [
        [difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
         for b in sents_b]
        for a in sents_a
    ]
    all_pairs = sorted(
        ((scores[i][j], i, j) for i in range(len(sents_a)) for j in range(len(sents_b))),
        reverse=True
    )
    used_a, used_b = set(), set()
    pairs = []
    for _, i, j in all_pairs:
        if i not in used_a and j not in used_b:
            pairs.append((sents_a[i], sents_b[j]))
            used_a.add(i)
            used_b.add(j)
    for i, a in enumerate(sents_a):
        if i not in used_a:
            pairs.append((a, ""))
    for j, b in enumerate(sents_b):
        if j not in used_b:
            pairs.append(("", b))
    return pairs


def build_diff_blocks_sentences(sents_a, sents_b, opcodes):
    blocks = []
    for tag, a0, a1, b0, b1 in opcodes:
        if tag == "equal":
            continue
        for removed, added in pair_by_similarity(sents_a[a0:a1], sents_b[b0:b1]):
            blocks.append(("", removed, added, ""))
    return blocks


def format_block_word(idx, before, removed, added, after):
    before_str = f"...{before} " if before else ""
    after_str  = f" {after}..." if after else ""
    return f"[{idx}] {before_str}[-{removed}-]{{+{added}+}}{after_str}"


def highlight_diff(sent_a, sent_b):
    """Return (a_highlighted, b_highlighted) with changed words marked."""
    words_a = re.findall(r'\S+|\s+', sent_a)
    words_b = re.findall(r'\S+|\s+', sent_b)
    sm = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    out_a, out_b = [], []
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal":
            out_a.extend(words_a[a0:a1])
            out_b.extend(words_b[b0:b1])
        elif tag == "replace":
            out_a.append(f"[{''.join(words_a[a0:a1]).strip()}]")
            out_b.append(f"[{''.join(words_b[b0:b1]).strip()}]")
        elif tag == "delete":
            out_a.append(f"[{''.join(words_a[a0:a1]).strip()}]")
        elif tag == "insert":
            out_b.append(f"[{''.join(words_b[b0:b1]).strip()}]")
    return "".join(out_a).strip(), "".join(out_b).strip()


def format_block_sentence(idx, before, removed, added, after):
    lines = [f"[{idx}]"]
    if removed and added:
        a_hl, b_hl = highlight_diff(removed, added)
        lines.append(f"  A: {a_hl}")
        lines.append("")
        lines.append(f"  B: {b_hl}")
    elif removed:
        lines.append(f"  A: {removed}")
        lines.append("")
        lines.append(f"  B: (nothing)")
    else:
        lines.append(f"  A: (nothing)")
        lines.append("")
        lines.append(f"  B: {added}")
    return "\n".join(lines)


def compare(path_a, path_b, mode="word", context=6, out_path=None):
    text_a = open(path_a, encoding="utf-8").read()
    text_b = open(path_b, encoding="utf-8").read()

    if mode == "sentence":
        tokens_a = tokenize_sentences(text_a)
        tokens_b = tokenize_sentences(text_b)
    else:
        tokens_a = tokenize_words(text_a)
        tokens_b = tokenize_words(text_b)

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    similarity = matcher.ratio()
    opcodes = matcher.get_opcodes()

    if similarity < 1.0:
        if mode == "sentence":
            diff_blocks = build_diff_blocks_sentences(tokens_a, tokens_b, opcodes)
        else:
            diff_blocks = build_diff_blocks_words(tokens_a, tokens_b, opcodes, context)
    else:
        diff_blocks = []

    verdict = interpret(similarity, len(diff_blocks))

    lines = []
    lines.append(f"date:        {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"mode:        {mode}")
    lines.append(f"A:           {path_a}")
    lines.append(f"B:           {path_b}")
    lines.append(f"similarity:  {similarity:.1%}")
    lines.append(f"differences: {len(diff_blocks)}")
    lines.append(f"verdict:     {verdict}")
    lines.append("")
    if mode == "word":
        lines.append("legend:      [-removed from A-]  {+added in B+}")
    else:
        lines.append("legend:      [word] = differs at that position")
    lines.append("")

    if diff_blocks:
        for idx, (before, removed, added, after) in enumerate(diff_blocks, 1):
            if mode == "sentence":
                lines.append(format_block_sentence(idx, before, removed, added, after))
            else:
                lines.append(format_block_word(idx, before, removed, added, after))
            lines.append("")

    result = "\n".join(lines)

    if out_path is None:
        a_stem = Path(path_a).stem
        b_stem = Path(path_b).stem
        out_dir = Path(__file__).parent.parent / "output" / "compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"compare_{mode}_{a_stem}_vs_{b_stem}.md"

    Path(out_path).write_text(result, encoding="utf-8")
    print(f"Result written to {out_path}")
    print(f"Similarity: {similarity:.1%} — {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two markdown files.")
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--mode", choices=["word", "sentence"], default="word")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    compare(args.file_a, args.file_b, mode=args.mode, out_path=args.out)
