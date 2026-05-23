"""
Unit tests for src/images.py — collection, sorting, renaming, and copy logic.
Uses tempfile.TemporaryDirectory; no llama-server or model stack required.
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import Config
from images import (
    ImageCollectionError,
    _collect_sources,
    collect_images,
    copy_from_subdirs,
    has_image_subdirs,
    rename_images,
)


def _cfg_for(path: str | Path, extensions=(".jpg", ".jpeg", ".png")) -> Config:
    cfg = Config()
    cfg.images_dir = str(path)
    cfg.image_files = None
    cfg.extensions = extensions
    return cfg


def _touch(path: Path) -> Path:
    """Create an empty file at path."""
    path.touch()
    return path


# ---------------------------------------------------------------------------
# collect_images — image_files override (no filesystem)
# ---------------------------------------------------------------------------

class TestCollectImagesExplicit(unittest.TestCase):
    def test_image_files_returned_directly(self):
        cfg = Config()
        cfg.image_files = ["/fake/page_001.jpg", "/fake/page_002.jpg"]
        result = collect_images(cfg)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Path)

    def test_image_files_empty_list_returned(self):
        cfg = Config()
        cfg.image_files = []
        result = collect_images(cfg)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# collect_images — filesystem paths
# ---------------------------------------------------------------------------

class TestCollectImagesFilesystem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_folder_raises(self):
        cfg = _cfg_for(self.folder / "nonexistent")
        with self.assertRaises(ImageCollectionError):
            collect_images(cfg)

    def test_empty_folder_raises(self):
        cfg = _cfg_for(self.folder)
        with self.assertRaises(ImageCollectionError):
            collect_images(cfg)

    def test_single_supported_file_returned(self):
        img = _touch(self.folder / "page.jpg")
        cfg = _cfg_for(img)
        result = collect_images(cfg)
        self.assertEqual(result, [img])

    def test_single_unsupported_extension_raises(self):
        f = _touch(self.folder / "doc.pdf")
        cfg = _cfg_for(f)
        with self.assertRaises(ImageCollectionError):
            collect_images(cfg)

    def test_filters_by_extension(self):
        _touch(self.folder / "page_001.jpg")
        _touch(self.folder / "page_002.jpg")
        _touch(self.folder / "readme.txt")
        cfg = _cfg_for(self.folder)
        result = collect_images(cfg)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(p.suffix == ".jpg" for p in result))

    def test_returns_sorted_images(self):
        _touch(self.folder / "page_003.jpg")
        _touch(self.folder / "page_001.jpg")
        _touch(self.folder / "page_002.jpg")
        cfg = _cfg_for(self.folder)
        result = collect_images(cfg)
        names = [p.name for p in result]
        self.assertEqual(names, sorted(names))


# ---------------------------------------------------------------------------
# _collect_sources — natural sort order
# ---------------------------------------------------------------------------

class TestCollectSourcesNaturalSort(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_natural_sort_beats_lexicographic(self):
        # Lexicographic order: page_1, page_10, page_2
        # Natural order:       page_1, page_2, page_10
        for name in ("page_10.jpg", "page_2.jpg", "page_1.jpg"):
            _touch(self.folder / name)
        cfg = _cfg_for(self.folder)
        result = _collect_sources(cfg)
        names = [p.name for p in result]
        self.assertEqual(names, ["page_1.jpg", "page_2.jpg", "page_10.jpg"])

    def test_includes_pdf_files(self):
        _touch(self.folder / "page_001.jpg")
        _touch(self.folder / "document.pdf")
        cfg = _cfg_for(self.folder)
        result = _collect_sources(cfg)
        suffixes = {p.suffix.lower() for p in result}
        self.assertIn(".pdf", suffixes)
        self.assertIn(".jpg", suffixes)

    def test_includes_epub_files(self):
        _touch(self.folder / "book.epub")
        cfg = _cfg_for(self.folder)
        result = _collect_sources(cfg)
        self.assertTrue(any(p.suffix.lower() == ".epub" for p in result))


# ---------------------------------------------------------------------------
# has_image_subdirs
# ---------------------------------------------------------------------------

class TestHasImageSubdirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_true_when_subdir_has_images(self):
        subdir = self.folder / "chapter1"
        subdir.mkdir()
        _touch(subdir / "page.jpg")
        self.assertTrue(has_image_subdirs(self.folder, (".jpg",)))

    def test_false_when_subdir_has_no_images(self):
        subdir = self.folder / "chapter1"
        subdir.mkdir()
        _touch(subdir / "notes.txt")
        self.assertFalse(has_image_subdirs(self.folder, (".jpg",)))

    def test_false_when_no_subdirs(self):
        _touch(self.folder / "page.jpg")
        self.assertFalse(has_image_subdirs(self.folder, (".jpg",)))


# ---------------------------------------------------------------------------
# rename_images
# ---------------------------------------------------------------------------

class TestRenameImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_images(self, names: list[str]) -> list[Path]:
        paths = []
        for name in names:
            p = self.folder / name
            p.touch()
            time.sleep(0.01)  # ensure distinct mtimes for ordering
            paths.append(p)
        return paths

    def test_dry_run_returns_expected_names(self):
        self._make_images(["DSC001.jpg", "DSC002.jpg", "DSC003.jpg"])
        result = rename_images(self.folder, (".jpg",), dry_run=True)
        stems = [p.stem for p in result]
        self.assertIn("page_001", stems)
        self.assertIn("page_002", stems)
        self.assertIn("page_003", stems)

    def test_dry_run_does_not_rename_files(self):
        self._make_images(["DSC001.jpg"])
        rename_images(self.folder, (".jpg",), dry_run=True)
        self.assertTrue((self.folder / "DSC001.jpg").exists())

    def test_actual_rename_renames_files(self):
        self._make_images(["DSC001.jpg"])
        rename_images(self.folder, (".jpg",), dry_run=False)
        self.assertFalse((self.folder / "DSC001.jpg").exists())
        self.assertTrue((self.folder / "page_001.jpg").exists())

    def test_custom_prefix(self):
        self._make_images(["a.jpg", "b.jpg"])
        result = rename_images(self.folder, (".jpg",), prefix="img", dry_run=True)
        self.assertTrue(all(p.stem.startswith("img_") for p in result))

    def test_start_parameter(self):
        self._make_images(["a.jpg"])
        result = rename_images(self.folder, (".jpg",), start=5, dry_run=True)
        self.assertIn("page_005", result[0].stem)

    def test_zero_padding_adjusts_to_width(self):
        # 10 images: width should be at least 3 (010, 011, …)
        self._make_images([f"img{i:02d}.jpg" for i in range(10)])
        result = rename_images(self.folder, (".jpg",), dry_run=True)
        self.assertTrue(all(len(p.stem.split("_")[1]) >= 3 for p in result))


# ---------------------------------------------------------------------------
# copy_from_subdirs
# ---------------------------------------------------------------------------

class TestCopyFromSubdirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        ch1 = self.folder / "chapter1"
        ch2 = self.folder / "chapter2"
        ch1.mkdir()
        ch2.mkdir()
        _touch(ch1 / "img1.jpg")
        _touch(ch1 / "img2.jpg")
        _touch(ch2 / "img3.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_returns_all_images(self):
        result = copy_from_subdirs(self.folder, (".jpg",), dry_run=True)
        self.assertEqual(len(result), 3)

    def test_dry_run_does_not_copy_files(self):
        copy_from_subdirs(self.folder, (".jpg",), dry_run=True)
        self.assertFalse((self.folder / "page_001.jpg").exists())

    def test_actual_copy_creates_files(self):
        copy_from_subdirs(self.folder, (".jpg",), dry_run=False)
        self.assertTrue((self.folder / "page_001.jpg").exists())

    def test_chapters_filter_limits_sources(self):
        result = copy_from_subdirs(self.folder, (".jpg",), chapters=["chapter1"], dry_run=True)
        self.assertEqual(len(result), 2)

    def test_naming_format(self):
        result = copy_from_subdirs(self.folder, (".jpg",), dry_run=True)
        for p in result:
            self.assertTrue(p.name.startswith("page_"))
            self.assertTrue(p.suffix == ".jpg")

    def test_empty_subdirs_returns_empty(self):
        empty = Path(self.tmp.name + "_empty")
        empty.mkdir()
        (empty / "sub").mkdir()
        result = copy_from_subdirs(empty, (".jpg",), dry_run=True)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
