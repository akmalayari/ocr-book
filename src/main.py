"""
main.py — CLI entry point for the book OCR pipeline

Usage :
    python src/main.py                                # default config (base mode)
    python src/main.py --images ./photos --out output/book.md
    python src/main.py --no-resume                    # restart from the beginning
    python src/main.py --no-layout                    # disable layout detection
    python src/main.py --no-postprocess               # raw output
    python src/main.py --mode obsidian                # OCR + Obsidian postprocess (prompts vault if not configured)
    python src/main.py --mode obsidian --postprocess-only  # Obsidian postprocess without re-running OCR
    python src/main.py --rename                       # rename images then OCR
    python src/main.py --rename-only                  # rename images without running OCR
    python src/main.py --rename-only 15               # rename starting at page_015
    python src/main.py --rename --dry-run             # print renames without doing them or OCR
    python src/main.py --rename-only --chapters "Lesson 1" "Lesson 3"  # selected subfolders
    python src/main.py --rename-only --dir-level      # order: folders alpha > subfolders alpha > images by date
    python src/main.py --verbose                      # detailed logs
"""

import argparse
import logging
import sys

from config import Config
from images import rename_images, copy_from_subdirs, has_image_subdirs
from progress import setup_logging


def build_parser() -> argparse.ArgumentParser:
    _cfg = Config()
    p = argparse.ArgumentParser(
        description="Book OCR pipeline → Markdown via PaddleOCR-VL-1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Paths
    p.add_argument("--images", default=_cfg.images_dir,
                   help="Folder containing page photos")
    p.add_argument("--out", default=_cfg.output_file,
                   help="Output Markdown file")

    # PaddleOCR
    p.add_argument("--no-layout", action="store_true",
                   help="Disable layout detection")

    # Behavior
    p.add_argument("--no-resume", action="store_true",
                   help="Restart from the beginning (ignore existing file)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="Disable post-processing")
    p.add_argument("--mode", choices=["base", "obsidian"], default=_cfg.mode,
                   help="Output mode: base (HTML img) or obsidian (wikilinks ![[]])")
    p.add_argument("--postprocess-only", action="store_true",
                   help="With --mode obsidian: apply postprocess on existing .md without re-running OCR")
    p.add_argument("--verbose", action="store_true",
                   help="Detailed logs (DEBUG)")

    # Utilities
    p.add_argument("--rename", action="store_true",
                   help="Rename images to page_001.jpg, page_002.jpg… then run OCR")
    p.add_argument("--rename-prefix", default=_cfg.rename_prefix,
                   help="Prefix for --rename (default: page)")
    p.add_argument("--rename-only", nargs="?", type=int, const=1, default=None, metavar="START",
                   help="Rename images without running OCR (optional: starting number, default: 1)")
    p.add_argument("--chapters", nargs="+", metavar="NAME",
                   help="Subfolders to process (in the given order). Implies copying to the parent folder.")
    p.add_argument("--dir-level", action="store_true",
                   help="With --rename/--rename-only: order by folder (folders alpha, subfolders alpha, images by date)")
    p.add_argument("--dry-run", action="store_true",
                   help="With --rename/--rename-only: print renames without doing them or running OCR")
    p.add_argument("--migrate", action="store_true",
                   help="Copy figures to the Obsidian vault (vault_path/vault_figures_dir) without running OCR")

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    cfg = Config(
        images_dir=args.images,
        output_file=args.out,
        use_layout_detection=not args.no_layout,
        resume=not args.no_resume,
        postprocess=not args.no_postprocess,
        mode=args.mode,
        verbose=args.verbose,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # ── Obsidian setup (common to all obsidian modes) ────────────────────────
    if cfg.mode == "obsidian" or args.migrate:
        from pathlib import Path as _Path
        from obsidian import prompt_if_needed
        prompt_if_needed(cfg)
        if not cfg.vault_root or not cfg.vault_figures_dir or not cfg.vault_path:
            logger.error("vault_root, vault_path and vault_figures_dir must be configured.")
            return 1
        if not args.migrate:
            cfg.output_file = str(_Path(cfg.vault_root) / cfg.vault_path / _Path(cfg.output_file).name)

    # ── Figure migration to vault ────────────────────────────────────────────
    if args.migrate:
        from obsidian import migrate_figures
        migrate_figures(cfg, dry_run=args.dry_run)
        return 0

    # ── Obsidian --postprocess-only mode ─────────────────────────────────────
    if args.postprocess_only:
        if cfg.mode != "obsidian":
            logger.error("--postprocess-only requires --mode obsidian")
            return 1
        from obsidian import postprocess_file, migrate_figures
        postprocess_file(cfg)
        migrate_figures(cfg)
        logger.info("Obsidian postprocess applied: %s", cfg.output_path.resolve())
        return 0

    # ── Rename / copy from subfolders (optional, before OCR) ─────────────────
    if args.rename or args.rename_only is not None:
        start = args.rename_only if args.rename_only is not None else 1
        images_path = cfg.images_path
        if args.chapters or (images_path.is_dir() and has_image_subdirs(images_path, cfg.extensions)):
            logger.info("Subfolder mode: copying to %s", cfg.images_dir)
            copy_from_subdirs(
                images_path, cfg.extensions,
                chapters=args.chapters,
                prefix=args.rename_prefix,
                start=start,
                dry_run=args.dry_run,
                dir_level=args.dir_level,
            )
        else:
            logger.info("Renaming images in: %s", cfg.images_dir)
            rename_images(cfg.images_dir, cfg.extensions, prefix=args.rename_prefix, dry_run=args.dry_run, start=start)
        if args.dry_run or args.rename_only is not None:
            return 0

    # ── Header detection (prompt if not configured) ──────────────────────────
    if cfg.header_patterns is None:
        print("Header detection on the final file (leave empty to disable):")
        patterns = []
        for level, label, example in [
            (2,  "sections    (##) ", r"^[IVX]+\."),
            (3,  "sub-sections (###)", r"^[A-Z]\."),
        ]:
            val = input(f"  Pattern {label} [ex: {example}] : ").strip()
            if val:
                patterns.append((val, level))
        cfg.header_patterns = patterns or []

    # ── OCR Pipeline ─────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("OCR Pipeline — PaddleOCR-VL-1.5")
    logger.info("  Images  : %s", cfg.images_path.resolve())
    logger.info("  Output  : %s", cfg.output_path.resolve())
    logger.info("  Layout  : %s", cfg.use_layout_detection)
    logger.info("  Resume  : %s", cfg.resume)
    logger.info("═" * 60)

    try:
        from pipeline import run_pipeline
        stats = run_pipeline(cfg)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        return 1

    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
