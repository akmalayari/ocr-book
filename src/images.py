"""
images.py — Découverte et tri des images à traiter
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
    Retourne la liste triée des images dans cfg.images_dir,
    filtrées par cfg.extensions.

    Si cfg.image_files est fourni, utilise directement cette liste.

    Le tri est alphanumérique sur le nom de fichier, ce qui suppose
    que vos photos sont nommées avec un padding numérique cohérent :
      page_001.jpg, page_002.jpg, …
    Si ce n'est pas le cas, renommez-les d'abord avec rename_images().

    Raises:
        ImageCollectionError si le dossier est vide ou inexistant.
    """
    if cfg.image_files is not None:
        images = [Path(p) for p in cfg.image_files]
        logger.info("%d image(s) explicite(s).", len(images))
        return images

    path = cfg.images_path
    if not path.exists():
        raise ImageCollectionError(f"Chemin introuvable : {path}")

    if path.is_file():
        if path.suffix.lower() not in cfg.extensions:
            raise ImageCollectionError(f"Extension non supportée : {path.suffix}")
        logger.info("1 image : %s", path)
        return [path]

    images = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.extensions
    )

    if not images:
        raise ImageCollectionError(
            f"Aucune image ({', '.join(cfg.extensions)}) dans : {path}"
        )

    stems = [p.stem for p in images]
    duplicates = {s for s in stems if stems.count(s) > 1}
    for dup in sorted(duplicates):
        logger.warning("Nom dupliqué '%s' : les pages s'écraseront mutuellement dans les parts.", dup)

    logger.info("%d image(s) trouvée(s) dans %s", len(images), path)
    return images


def rename_images(
        folder: str | Path,
        extensions: tuple,
        prefix: str = "page",
        dry_run: bool = False,
        start: int = 1,
        ) -> list[Path]:
    """
    Renomme les images d'un dossier avec un padding numérique uniforme :
      DSC_0042.jpg → page_001.jpg
      IMG_2024.jpg → page_002.jpg
      …

    Args:
        folder  : dossier contenant les images
        prefix  : préfixe des nouveaux noms (défaut : "page")
        dry_run : si True, affiche les renommages sans les effectuer
        start   : numéro de départ (défaut : 1)

    Returns:
        Liste des nouveaux chemins.
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
            logger.debug("Renommé : %s → %s", img.name, new_name)
        renamed.append(new_path)

    if dry_run:
        print(f"[dry-run] {len(images)} fichier(s) seraient renommés.")
    else:
        logger.info("%d image(s) renommée(s).", len(images))

    return renamed


def has_image_subdirs(folder: str | Path, extensions: tuple) -> bool:
    """Retourne True si le dossier contient des sous-dossiers avec des images (récursif)."""
    folder = Path(folder)
    return any(
        d.is_dir() and any(p.suffix.lower() in extensions for p in d.rglob("*") if p.is_file())
        for d in folder.iterdir()
    )


def _collect_per_dir(folder: Path, extensions: tuple) -> list[Path]:
    """
    Collecte les images récursivement :
    - images du dossier courant d'abord, triées par date de création
    - puis sous-dossiers en ordre alphabétique, récursivement
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
    Copie les images des sous-dossiers de folder vers folder, avec numérotation séquentielle.

    Args:
        folder   : dossier parent (destination et source des sous-dossiers)
        chapters : liste ordonnée de noms de sous-dossiers à traiter (None = tous, tri alpha)
        prefix   : préfixe des noms de fichiers copiés
        start    : numéro de départ
        dry_run  : si True, affiche les opérations sans les effectuer

    Returns:
        Liste des chemins copiés.
    """
    folder = Path(folder)

    subdirs = natsorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name)

    if chapters is not None:
        subdir_by_name = {d.name: d for d in subdirs}
        missing = [c for c in chapters if c not in subdir_by_name]
        for m in missing:
            logger.warning("Sous-dossier introuvable : '%s'", m)
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
        logger.warning("Aucune image trouvée dans les sous-dossiers.")
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
            logger.debug("Copié : %s/%s → %s", img.parent.name, img.name, new_name)
        copied.append(new_path)

    if dry_run:
        print(f"[dry-run] {len(all_images)} fichier(s) seraient copiés depuis {len(subdirs)} sous-dossier(s).")
    else:
        logger.info("%d image(s) copiées depuis %d sous-dossier(s).", len(all_images), len(subdirs))

    return copied
