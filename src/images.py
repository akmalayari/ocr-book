"""
images.py — Discovery and sorting of images to process
"""

import logging
import shutil
from pathlib import Path

from natsort import natsorted

from config import Config

logger = logging.getLogger(__name__)


class ImageCollectionError(FileNotFoundError):
    pass


def collect_images(cfg: Config) -> list[Path]:
    """
    Returns the sorted list of images in cfg.images_dir,
    filtered by cfg.extensions.

    If cfg.image_files is provided, uses that list directly.

    Sorting is alphanumeric on the filename, which assumes
    your photos are named with consistent numeric padding:
      page_001.jpg, page_002.jpg, …
    If not, rename them first with rename_images().

    Raises:
        ImageCollectionError if the folder is empty or does not exist.
    """
    if cfg.image_files is not None:
        images = [Path(p) for p in cfg.image_files]
        logger.info("%d explicit image(s).", len(images))
        return images

    path = cfg.images_path
    if not path.exists():
        raise ImageCollectionError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in cfg.extensions:
            raise ImageCollectionError(f"Unsupported extension: {path.suffix}")
        logger.info("1 image: %s", path)
        return [path]

    images = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.extensions
    )

    if not images:
        raise ImageCollectionError(
            f"No image ({', '.join(cfg.extensions)}) in: {path}"
        )

    stems = [p.stem for p in images]
    duplicates = {s for s in stems if stems.count(s) > 1}
    for dup in sorted(duplicates):
        logger.warning("Duplicate name '%s': pages will overwrite each other in parts.", dup)

    logger.info("%d image(s) found in %s", len(images), path)
    return images


def _collect_sources(cfg: Config) -> list[Path]:
    """
    Returns all processable files in cfg.images_path:
    images (by extension) + .pdf files, naturally sorted.
    """
    if cfg.image_files is not None:
        files = [Path(p) for p in cfg.image_files]
        logger.info("%d explicit file(s).", len(files))
        return files

    path = cfg.images_path
    if not path.exists():
        raise ImageCollectionError(f"Path not found: {path}")

    if path.is_file():
        return [path]

    extensions = cfg.extensions + (".pdf", ".epub")
    files = [
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return natsorted(files, key=lambda p: p.name)


def rename_images(
        folder: str | Path,
        extensions: tuple,
        prefix: str = "page",
        dry_run: bool = False,
        start: int = 1,
        ) -> list[Path]:
    """
    Renames images in a folder with uniform numeric padding:
      DSC_0042.jpg → page_001.jpg
      IMG_2024.jpg → page_002.jpg
      …

    Args:
        folder  : folder containing the images
        prefix  : prefix for new names (default: "page")
        dry_run : if True, prints renames without performing them
        start   : starting number (default: 1)

    Returns:
        List of new paths.
    """
    folder = Path(folder)
    images = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: getattr(p.stat(), "st_birthtime", p.stat().st_mtime)
    )

    renamed = []
    width = max(3, len(str(start + len(images) - 1)))

    for i, img in enumerate(images, start):
        new_name = f"{prefix}_{str(i).zfill(width)}{img.suffix.lower()}"
        new_path = folder / new_name
        if dry_run:
            print(f"  {img.name}  →  {new_name}")
        else:
            img.rename(new_path)
            logger.debug("Renamed: %s → %s", img.name, new_name)
        renamed.append(new_path)

    if dry_run:
        print(f"[dry-run] {len(images)} file(s) would be renamed.")
    else:
        logger.info("%d image(s) renamed.", len(images))

    return renamed


def has_image_subdirs(folder: str | Path, extensions: tuple) -> bool:
    """Returns True if the folder contains subfolders with images (recursive)."""
    folder = Path(folder)
    return any(
        d.is_dir() and any(p.suffix.lower() in extensions for p in d.rglob("*") if p.is_file())
        for d in folder.iterdir()
    )


def _collect_per_dir(folder: Path, extensions: tuple) -> list[Path]:
    """
    Collects images recursively, preserving directory hierarchy in the order.

    Within each directory:
      1. Local images first, sorted by creation date (birthtime, fallback mtime)
      2. Then subdirectories in alphabetical order, processed recursively

    This means all images from one folder come before any image from its
    subfolders, and subfolders are visited alphabetically.
    """
    result = []
    local = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: getattr(p.stat(), "st_birthtime", p.stat().st_mtime),
    )
    result.extend(local)
    for subdir in natsorted(folder.iterdir(), key=lambda d: d.name):
        if subdir.is_dir():
            result.extend(_collect_per_dir(subdir, extensions))
    return result


def copy_from_subdirs(
        folder: str | Path,
        extensions: tuple,
        chapters: list[str] | None = None,
        prefix: str = "page",
        start: int = 1,
        dry_run: bool = False,
        dir_level: bool = False,
        ) -> list[Path]:
    """
    Copies images from subfolders of folder into folder, with sequential numbering.

    Args:
        folder   : parent folder (destination and source of subfolders)
        chapters : ordered list of subfolder names to process (None = all, alpha sort)
        prefix   : prefix for copied file names
        start    : starting number
        dry_run  : if True, prints operations without performing them
        dir_level: If False, all images under each subdir are collected
                   recursively and flattened into a single date-sorted list.
                   If True, images are grouped by directory depth:
                   current folder first (by date), then subfolders
                   alphabetically, recursively.

    Returns:
        List of copied paths.
    """
    folder = Path(folder)

    subdirs = natsorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name)

    if chapters is not None:
        subdir_by_name = {d.name: d for d in subdirs}
        missing = [c for c in chapters if c not in subdir_by_name]
        for m in missing:
            logger.warning("Subfolder not found: '%s'", m)
        subdirs = [subdir_by_name[c] for c in chapters if c in subdir_by_name]

    all_images: list[Path] = []
    for subdir in subdirs:
        if dir_level:
            imgs = _collect_per_dir(subdir, extensions)
        else:
            imgs = sorted(
                (p for p in subdir.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
                key=lambda p: getattr(p.stat(), "st_birthtime", p.stat().st_mtime),
            )
        all_images.extend(imgs)

    if not all_images:
        logger.warning("No images found in subfolders.")
        return []

    width = max(3, len(str(start + len(all_images) - 1)))
    copied: list[Path] = []

    for i, img in enumerate(all_images, start):
        new_name = f"{prefix}_{str(i).zfill(width)}{img.suffix.lower()}"
        new_path = folder / new_name
        if dry_run:
            print(f"  [{img.parent.name}/{img.name}]  →  {new_name}")
        else:
            shutil.copy2(img, new_path)
            logger.debug("Copied: %s/%s → %s", img.parent.name, img.name, new_name)
        copied.append(new_path)

    if dry_run:
        print(f"[dry-run] {len(all_images)} file(s) would be copied from {len(subdirs)} subfolder(s).")
    else:
        logger.info("%d image(s) copied from %d subfolder(s).", len(all_images), len(subdirs))

    return copied
