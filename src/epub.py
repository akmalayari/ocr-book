"""
epub.py — EPUB extraction to Markdown via Pandoc (EbookLib fallback)

Images are extracted to a figures/ subdirectory and referenced with
relative paths (figures/<name>). Transparent PNGs are flattened to white.
Inline SVGs that wrap raster images are dropped; pure vector SVGs are
saved and referenced.
"""

import logging
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────


def _pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def _flatten_on_white(img_path: Path) -> None:
    """Convert transparent images to white background (in-place)."""
    img = Image.open(img_path)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        background.convert("RGB").save(img_path)
    elif img.mode == "P":
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        background.convert("RGB").save(img_path)


def _extract_images(epub_path: Path, figures_dir: Path) -> dict[str, Path]:
    """Extract raster + svg images from EPUB zip to *figures_dir*.

    Returns a mapping  original_zip_path -> saved_path.
    """
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, Path] = {}
    with zipfile.ZipFile(epub_path, "r") as zf:
        for name in zf.namelist():
            if not any(
                name.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg")
            ):
                continue
            data = zf.read(name)
            dest = figures_dir / Path(name).name
            # Handle collisions
            counter = 1
            original = dest
            while dest.exists():
                dest = figures_dir / f"{original.stem}_{counter}{original.suffix}"
                counter += 1
            dest.write_bytes(data)
            mapping[name] = dest
            mapping[Path(name).as_posix()] = dest

            # Flatten transparency for raster images
            if dest.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
                _flatten_on_white(dest)

    logger.info("Extracted %d image(s) to %s", len(mapping), figures_dir)
    return mapping


def _extract_with_pandoc(epub_path: Path, out_md: Path) -> dict:
    cmd = [
        "pandoc",
        str(epub_path),
        "-f", "epub",
        "-t", "markdown",
        "--wrap=none",
        "-o", str(out_md),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Pandoc failed: {result.stderr}")
    text = out_md.read_text(encoding="utf-8")
    return {
        "method": "pandoc",
        "chars": len(text),
        "lines": text.count("\n"),
    }


def _extract_with_ebooklib(epub_path: Path, out_md: Path) -> dict:
    import ebooklib
    from ebooklib import epub
    from xml.etree import ElementTree as ET

    book = epub.read_epub(str(epub_path))
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    md_parts = []
    for item in items:
        content = item.get_content().decode("utf-8", errors="ignore")
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            continue

        def _elem_text(elem):
            chunks = []
            tag = elem.tag.split("}")[-1] if isinstance(elem.tag, str) else ""
            block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "br"}
            is_block = tag in block_tags
            if elem.text and elem.text.strip():
                chunks.append(elem.text.strip())
            for child in elem:
                chunks.extend(_elem_text(child))
                if child.tail and child.tail.strip():
                    chunks.append(child.tail.strip())
            if is_block and chunks:
                chunks.append("\n")
            return chunks

        text = " ".join(_elem_text(root)).replace("  ", " ").strip()
        if text:
            md_parts.append(text)

    full_text = "\n\n".join(md_parts)
    out_md.write_text(full_text, encoding="utf-8")
    return {
        "method": "ebooklib",
        "chars": len(full_text),
        "lines": full_text.count("\n"),
        "items_processed": len(items),
    }


def _clean_markdown(
    text: str,
    image_mapping: dict[str, Path],
    figures_dir: Path,
) -> str:
    """Clean Pandoc Markdown for RAG: fix image paths, remove inline styles/classes/divs."""
    svg_counter = [0]

    def _extract_inline_svg(m: re.Match) -> str:
        svg_content = m.group(0)
        # If the SVG wraps a raster image, drop it (image already extracted separately)
        if re.search(r'<image\b[^>]*\bxlink:href="[^"]+"', svg_content):
            return ""
        # Pure vector SVG: save as file and reference it
        svg_counter[0] += 1
        svg_name = f"inline_svg_{svg_counter[0]:03d}.svg"
        svg_path = figures_dir / svg_name
        svg_path.write_text(svg_content, encoding="utf-8")
        return f"\n\n![](figures/{svg_name})\n\n"

    # Step 0: extract inline SVGs (multiline blocks) before line-by-line processing
    text = re.sub(r'<svg\b.*?</svg>', _extract_inline_svg, text, flags=re.DOTALL)

    lines = text.splitlines()
    out_lines = []

    for line in lines:
        # Skip fenced div markers (Pandoc ::: ...), including inside blockquotes (> :::)
        if re.match(r"\s*(?:>\s*)?:{3,}", line):
            continue

        # Fix image paths: replace EPUB internal paths with local figures/ paths
        def _fix_img_ref(m: re.Match) -> str:
            alt = m.group(1)
            src = m.group(2)
            src_norm = src.replace("\\", "/")
            if src in image_mapping:
                new_src = image_mapping[src].name
            elif src_norm in image_mapping:
                new_src = image_mapping[src_norm].name
            else:
                new_src = Path(src).name
            return f"![{alt}](figures/{new_src})"

        line = re.sub(r"!\[(.*?)\]\((.+?)\)", _fix_img_ref, line)

        # Pandoc sometimes encodes slashes as [/] inside links (EPUB artefact)
        line = line.replace("[/]", "/")

        # Remove Pandoc inline attributes {style="..."}, {.class}, {#id}, {any="thing"}
        line = re.sub(r"\s*\{[^{}]+\}", "", line)

        # Remove raw HTML spans like `...`{=html}
        line = re.sub(r"`[^`]*`\{=html\}", "", line)

        # Remove empty bracket spans []{#id} or []{.class}
        line = re.sub(r"\[\]\{[^{}]*\}", "", line)

        # Remove stray empty brackets, but keep those inside image/link syntax ![]() or []()
        line = re.sub(r"(?<!!)\[\](?!\()", "", line)

        # Fix headings with bracketed numbers:
        #   ## [1.3] Popular...  →  ## 1.3 Popular...
        #   # [1 ]LLMs...        →  # 1 LLMs...
        #   # [[1] ][LLMs... ]   →  # 1 LLMs...
        if line.lstrip().startswith("#"):
            line = re.sub(
                r"^(#{1,6}\s*)\[\[?\s*(\d+(?:\.\d+)*)\s*\]?\s*\](?:\[\s*([^]]*?)\s*\])?",
                r"\1\2 \3",
                line,
            )

        # Convert Pandoc internal links [[text]] (no URL) to plain text
        line = re.sub(r"(?<!!)\[\[([^\]]+)\]\](?!\()", r"\1", line)

        # Fix broken Pandoc URL links: [[https://...]](url) → [https://...](url)
        def _fix_broken_url(m: re.Match) -> str:
            text = m.group(1)
            url = m.group(2)
            return f"[{text}]({url})"

        line = re.sub(r"\[\[([^\]]+)\]\]\(([^)]+)\)", _fix_broken_url, line)

        # Remove brackets around plain text that is NOT a markdown link/image
        line = re.sub(r"(?<!!)\[([^\]]+)\](?!\[|\()", r"\1", line)

        # Remove backslash escapes before punctuation or at end of line
        line = re.sub(r"\\(?=[,;.!?])", "", line)
        line = re.sub(r"\\$", "", line)

        # Strip HTML/XML comment lines (including backtick-wrapped)
        if re.match(r"\s*<!--.*?-->", line):
            continue
        if re.match(r"\s*`<!--.*?-->`", line):
            continue

        # Clean up multiple spaces
        line = re.sub(r"  +", " ", line).rstrip()

        if line or (out_lines and out_lines[-1] != ""):
            out_lines.append(line)

    result = "\n".join(out_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── Public API ─────────────────────────────────────────────────────────────


def extract_epub(epub_path: Path, output_md: Path, figures_dir: Path) -> dict:
    """
    Extract an EPUB to a clean Markdown file.

    Returns a dict with keys: method, raw_chars, clean_chars, lines, images.
    """
    if not epub_path.exists():
        raise FileNotFoundError(epub_path)

    output_md.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract images
    logger.info("Extracting images from EPUB...")
    image_mapping = _extract_images(epub_path, figures_dir)

    # 2. Convert with Pandoc or EbookLib
    raw_md = output_md.with_suffix(".raw.md")
    if _pandoc_available():
        logger.info("Using Pandoc for EPUB → Markdown")
        try:
            stats = _extract_with_pandoc(epub_path, raw_md)
        except RuntimeError as e:
            logger.warning("Pandoc failed (%s); falling back to EbookLib", e)
            stats = _extract_with_ebooklib(epub_path, raw_md)
    else:
        logger.info("Pandoc not found; falling back to EbookLib")
        stats = _extract_with_ebooklib(epub_path, raw_md)

    # 3. Clean Markdown
    raw_text = raw_md.read_text(encoding="utf-8")
    clean_text = _clean_markdown(raw_text, image_mapping, figures_dir)
    output_md.write_text(clean_text, encoding="utf-8")

    # Remove raw intermediate file unless debugging
    raw_md.unlink(missing_ok=True)

    stats["raw_chars"] = stats.pop("chars")
    stats["clean_chars"] = len(clean_text)
    stats["images"] = len(image_mapping)
    logger.info(
        "EPUB extracted — method=%s raw_chars=%d clean_chars=%d images=%d",
        stats["method"],
        stats["raw_chars"],
        stats["clean_chars"],
        stats["images"],
    )
    return stats
