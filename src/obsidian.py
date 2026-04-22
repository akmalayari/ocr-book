"""
obsidian.py — Utilities for Obsidian export
"""

import logging
import re
import shutil
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def prompt_if_needed(cfg: Config) -> None:
    """Interactive prompt for vault_root, vault_path and vault_figures_dir if not configured."""
    if cfg.vault_root is None:
        cfg.vault_root = input("Absolute path to the Obsidian vault root: ").strip().strip('"').strip("'")
    if cfg.vault_path is None:
        cfg.vault_path = input("Output subfolder relative to the vault root (ex: Documents/OCR): ").strip().strip('"').strip("'")
    if cfg.vault_figures_dir is None:
        cfg.vault_figures_dir = input("Figures path relative to the vault ROOT (ex: Documents/OCR/Files): ").strip().strip('"').strip("'")


def fix_image_paths_obsidian(text: str, vault_figures_dir: str) -> str:
    """
    Converts <img src="imgs/..."> tags into Obsidian wikilinks ![[...]].
    Used during OCR (raw PaddleOCR paths).
    The path is built from the vault root.
    """
    if not vault_figures_dir:
        raise ValueError("vault_figures_dir not configured")
    prefix = vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    text = re.sub(r'<img\b[^>]*\bsrc="imgs/([^"]+)"[^>]*/?\s*>', _replace, text)
    # Strip <div> wrappers that solely contain the image wikilink
    text = re.sub(r'<div\b[^>]*>\s*(!\[\[[^\]]+\]\])\s*</div>', r'\1', text)
    return text


def migrate_figures(cfg: Config, page_ids: list[str] | None = None, dry_run: bool = False) -> int:
    """
    Copies figures from output/figures/*/imgs/* to vault_root/vault_figures_dir/.
    Flat structure: page_001/imgs/fig.jpg → vault_figures_dir/fig.jpg.
    Skips files already present. Returns the number of files copied.

    page_ids : if provided, limits copying to the corresponding subfolders (pages of the current run).
               If None, copies all content from figures_path.
    """
    if not cfg.vault_root or not cfg.vault_figures_dir:
        raise ValueError("vault_root and vault_figures_dir must be configured")

    dest = Path(cfg.vault_root) / cfg.vault_figures_dir
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    if page_ids is not None:
        sources = sorted(
            src
            for page_id in page_ids
            for src in (cfg.figures_path / page_id / "imgs").glob("*")
            if (cfg.figures_path / page_id / "imgs").exists()
        )
    else:
        sources = sorted(cfg.figures_path.glob("*/imgs/*"))

    copied = 0
    for src in sources:
        if not src.is_file():
            continue
        target = dest / src.name
        if target.exists():
            continue
        if dry_run:
            logger.info("[dry-run] %s → %s", src.name, target)
        else:
            shutil.copy2(src, target)
            logger.info("Copied: %s", src.name)
        copied += 1

    logger.info("%d file(s) copied to %s", copied, dest)
    return copied


def postprocess_file(cfg: Config) -> None:
    """
    Applies full postprocess on an already generated .md file (without re-running OCR):
      - text cleanup (clean_page, strip_table_styles)
      - conversion of <img> tags to wikilinks ![[ ]]
      - header detection (if cfg.header_patterns)
    """
    from postprocess import clean_page, strip_table_styles, apply_header_detection

    if not cfg.vault_figures_dir:
        raise ValueError("vault_figures_dir not configured")
    prefix = cfg.vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    md_path = cfg.output_path
    text = md_path.read_text(encoding="utf-8")

    if cfg.postprocess:
        text = clean_page(text, cfg)
        text = strip_table_styles(text)

    text = re.sub(r'<img\b[^>]*\bsrc="[^"]*/imgs/([^"]+)"[^>]*/?\s*>', _replace, text)
    text = re.sub(r'<div\b[^>]*>\s*(!\[\[[^\]]+\]\])\s*</div>', r'\1', text)

    if cfg.header_patterns:
        text = apply_header_detection(text, cfg.header_patterns)

    md_path.write_text(text, encoding="utf-8", newline="\n")
