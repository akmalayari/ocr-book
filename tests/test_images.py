"""
test_images.py — Tests unitaires pour images.py

Couvre :
  - collect_images : dossier inexistant, dossier vide, filtrage extensions,
    tri alphanumérique, fichiers non-image ignorés
  - rename_images : renommage réel, dry_run, padding adaptatif, extensions
    conservées, retour de la liste des nouveaux chemins
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from images import ImageCollectionError, collect_images, rename_images

JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xD9])
PNG_BYTES  = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44])

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def make_cfg(folder: Path) -> Config:
    return Config(images_dir=str(folder), log_file="")


# ── collect_images ────────────────────────────────────────────────────────────

class TestCollectImages:

    def test_raises_if_folder_does_not_exist(self, tmp_path):
        cfg = make_cfg(tmp_path / "inexistant")
        with pytest.raises(ImageCollectionError, match="introuvable"):
            collect_images(cfg)

    def test_raises_if_folder_is_empty(self, tmp_path):
        folder = tmp_path / "vide"
        folder.mkdir()
        with pytest.raises(ImageCollectionError):
            collect_images(make_cfg(folder))

    def test_raises_if_no_images_only_other_files(self, tmp_path):
        folder = tmp_path / "noimg"
        folder.mkdir()
        (folder / "readme.txt").write_text("hello")
        (folder / "data.csv").write_text("a,b,c")
        with pytest.raises(ImageCollectionError):
            collect_images(make_cfg(folder))

    def test_returns_list_of_paths(self, tmp_path):
        folder = tmp_path / "imgs"
        folder.mkdir()
        (folder / "page_001.jpg").write_bytes(JPEG_BYTES)
        result = collect_images(make_cfg(folder))
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)

    def test_returns_correct_count(self, tmp_path):
        folder = tmp_path / "imgs"
        folder.mkdir()
        for i in range(5):
            (folder / f"page_{i:03d}.jpg").write_bytes(JPEG_BYTES)
        assert len(collect_images(make_cfg(folder))) == 5

    def test_filters_non_image_files(self, tmp_path):
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "page_001.jpg").write_bytes(JPEG_BYTES)
        (folder / "readme.txt").write_text("ignore me")
        result = collect_images(make_cfg(folder))
        assert len(result) == 1
        assert result[0].name == "page_001.jpg"

    def test_accepts_jpg_jpeg_png_webp(self, tmp_path):
        folder = tmp_path / "exts"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(JPEG_BYTES)
        (folder / "b.jpeg").write_bytes(JPEG_BYTES)
        (folder / "c.png").write_bytes(PNG_BYTES)
        (folder / "d.webp").write_bytes(JPEG_BYTES)
        assert len(collect_images(make_cfg(folder))) == 4

    def test_case_insensitive_extension(self, tmp_path):
        folder = tmp_path / "case"
        folder.mkdir()
        (folder / "photo.JPG").write_bytes(JPEG_BYTES)
        (folder / "scan.PNG").write_bytes(PNG_BYTES)
        assert len(collect_images(make_cfg(folder))) == 2

    def test_sorted_alphabetically(self, tmp_path):
        folder = tmp_path / "sorted"
        folder.mkdir()
        for name in ["page_003.jpg", "page_001.jpg", "page_002.jpg"]:
            (folder / name).write_bytes(JPEG_BYTES)
        result = collect_images(make_cfg(folder))
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_padding_matters_for_sort(self, tmp_path):
        folder = tmp_path / "padding"
        folder.mkdir()
        for i in [1, 2, 9, 10, 11]:
            (folder / f"page_{i:03d}.jpg").write_bytes(JPEG_BYTES)
        result = collect_images(make_cfg(folder))
        names = [p.name for p in result]
        assert names == sorted(names)
        assert names[0] == "page_001.jpg"
        assert names[-1] == "page_011.jpg"

    def test_ignores_subdirectories(self, tmp_path):
        folder = tmp_path / "withsub"
        folder.mkdir()
        (folder / "page_001.jpg").write_bytes(JPEG_BYTES)
        subdir = folder / "subdir"
        subdir.mkdir()
        (subdir / "page_002.jpg").write_bytes(JPEG_BYTES)
        assert len(collect_images(make_cfg(folder))) == 1

    def test_imagecollectionerror_is_filenotfounderror_subclass(self):
        assert issubclass(ImageCollectionError, FileNotFoundError)


# ── rename_images ─────────────────────────────────────────────────────────────

class TestRenameImages:

    def test_renames_files_on_disk(self, tmp_path):
        (tmp_path / "IMG_001.jpg").write_bytes(JPEG_BYTES)
        (tmp_path / "IMG_002.jpg").write_bytes(JPEG_BYTES)
        rename_images(tmp_path, EXTENSIONS, prefix="page")
        renamed = list(tmp_path.glob("page_*.jpg"))
        assert len(renamed) == 2

    def test_returns_list_of_new_paths(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(JPEG_BYTES)
        (tmp_path / "b.jpg").write_bytes(JPEG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="page")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)

    def test_dry_run_does_not_rename(self, tmp_path):
        original = tmp_path / "IMG_001.jpg"
        original.write_bytes(JPEG_BYTES)
        rename_images(tmp_path, EXTENSIONS, prefix="page", dry_run=True)
        assert original.exists()

    def test_dry_run_returns_expected_paths(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(JPEG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="scan", dry_run=True)
        assert len(result) == 1
        assert "scan" in result[0].name

    def test_custom_prefix(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(JPEG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="scan", dry_run=True)
        assert result[0].name.startswith("scan_")

    def test_preserves_extension(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(JPEG_BYTES)
        (tmp_path / "image.png").write_bytes(PNG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="p", dry_run=True)
        extensions = {p.suffix.lower() for p in result}
        assert ".jpg" in extensions or ".png" in extensions

    def test_padding_adapts_to_count(self, tmp_path):
        for i in range(10):
            (tmp_path / f"img_{i}.jpg").write_bytes(JPEG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="p", dry_run=True)
        for p in result:
            num_part = p.stem.split("_")[1]
            assert len(num_part) >= 2, f"Padding insuffisant : {p.name}"

    def test_order_preserved_alphabetically(self, tmp_path):
        for name in ["aaa.jpg", "bbb.jpg", "ccc.jpg"]:
            (tmp_path / name).write_bytes(JPEG_BYTES)
        result = rename_images(tmp_path, EXTENSIONS, prefix="page")
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert rename_images(tmp_path, EXTENSIONS, prefix="page") == []

    def test_non_image_files_ignored(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(JPEG_BYTES)
        (tmp_path / "readme.txt").write_text("ignore")
        result = rename_images(tmp_path, EXTENSIONS, prefix="page", dry_run=True)
        assert len(result) == 1
        assert result[0].suffix == ".jpg"
