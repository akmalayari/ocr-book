"""
images.py — Découverte et tri des images à traiter
"""

import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class ImageCollectionError(FileNotFoundError):
    pass


def collect_images(cfg: Config) -> list[Path]:
    """
    Retourne la liste triée des images dans cfg.images_dir,
    filtrées par cfg.extensions.

    Le tri est alphanumérique sur le nom de fichier, ce qui suppose
    que vos photos sont nommées avec un padding numérique cohérent :
      page_001.jpg, page_002.jpg, …
    Si ce n'est pas le cas, renommez-les d'abord avec rename_images().

    Raises:
        ImageCollectionError si le dossier est vide ou inexistant.
    """
    folder = cfg.images_path
    if not folder.exists():
        raise ImageCollectionError(f"Dossier images introuvable : {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.extensions
    )

    if not images:
        raise ImageCollectionError(
            f"Aucune image ({', '.join(cfg.extensions)}) dans : {folder}"
        )

    logger.info("%d image(s) trouvée(s) dans %s", len(images), folder)
    return images


def rename_images(
        folder: str | Path, 
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"), 
        prefix: str = "page", 
        dry_run: bool = False
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

    Returns:
        Liste des nouveaux chemins.
    """
    folder = Path(folder)
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    renamed = []
    width = len(str(len(images)))  # padding : 3 chiffres pour 100-999 images

    for i, img in enumerate(images, 1):
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
