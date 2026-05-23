"""
Unit tests for src/obsidian.py — string transformations and figure migration.
No llama-server or model stack required.
"""

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import Config
from obsidian import fix_image_paths_obsidian, fix_markdown_image_paths_obsidian, migrate_figures


# ---------------------------------------------------------------------------
# fix_image_paths_obsidian
# ---------------------------------------------------------------------------

class TestFixImagePathsObsidian(unittest.TestCase):
    def test_converts_img_tag_to_wikilink(self):
        text = '<img src="imgs/fig1.png" />'
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, "![[Vault/Figures/fig1.png]]")

    def test_backslash_in_prefix_normalised(self):
        text = '<img src="imgs/fig1.png" />'
        result = fix_image_paths_obsidian(text, "Vault\\Figures")
        self.assertIn("![[Vault/Figures/fig1.png]]", result)

    def test_trailing_slash_on_prefix_removed(self):
        text = '<img src="imgs/fig1.png" />'
        result = fix_image_paths_obsidian(text, "Vault/Figures/")
        self.assertIn("![[Vault/Figures/fig1.png]]", result)
        self.assertNotIn("//", result)

    def test_div_wrapper_stripped(self):
        text = '<div class="figure"><img src="imgs/fig1.png" /></div>'
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertNotIn("<div", result)
        self.assertEqual(result.strip(), "![[Vault/Figures/fig1.png]]")

    def test_div_wrapper_with_other_content_kept(self):
        text = '<div class="figure"><img src="imgs/fig1.png" /> caption text</div>'
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertIn("<div", result)

    def test_multiple_img_tags_converted(self):
        text = '<img src="imgs/a.png" />\n<img src="imgs/b.png" />'
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertIn("![[Vault/Figures/a.png]]", result)
        self.assertIn("![[Vault/Figures/b.png]]", result)

    def test_empty_vault_figures_dir_raises(self):
        with self.assertRaises(ValueError):
            fix_image_paths_obsidian('<img src="imgs/fig1.png" />', "")

    def test_no_img_tag_unchanged(self):
        text = "Plain text with no images."
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, text)

    def test_img_without_imgs_prefix_unchanged(self):
        # Only replaces src="imgs/..." — a different src path should be untouched
        text = '<img src="other/fig1.png" />'
        result = fix_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, text)


# ---------------------------------------------------------------------------
# fix_markdown_image_paths_obsidian
# ---------------------------------------------------------------------------

class TestFixMarkdownImagePathsObsidian(unittest.TestCase):
    def test_converts_markdown_image_link(self):
        text = "![alt text](figures/page_001/imgs/fig1.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertIn("![[Vault/Figures/fig1.png]]", result)
        self.assertNotIn("![alt", result)

    def test_empty_alt_text(self):
        text = "![](figures/fig1.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertIn("![[Vault/Figures/fig1.png]]", result)

    def test_external_url_skipped(self):
        text = "![logo](https://example.com/logo.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, text)

    def test_http_url_also_skipped(self):
        text = "![logo](http://example.com/logo.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, text)

    def test_backslash_in_prefix_normalised(self):
        text = "![fig](figures/fig1.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault\\Figures")
        self.assertIn("![[Vault/Figures/fig1.png]]", result)

    def test_empty_vault_figures_dir_raises(self):
        with self.assertRaises(ValueError):
            fix_markdown_image_paths_obsidian("![fig](figures/fig1.png)", "")

    def test_no_image_link_unchanged(self):
        text = "No images here, just [a link](page.md)."
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertEqual(result, text)

    def test_multiple_links_converted(self):
        text = "![a](figures/a.png) and ![b](figures/b.png)"
        result = fix_markdown_image_paths_obsidian(text, "Vault/Figures")
        self.assertIn("![[Vault/Figures/a.png]]", result)
        self.assertIn("![[Vault/Figures/b.png]]", result)


# ---------------------------------------------------------------------------
# migrate_figures
# ---------------------------------------------------------------------------

def _cfg_migrate(figures_dir: str, vault_root: str, vault_figures_dir: str = "Figures") -> Config:
    cfg = Config()
    cfg.figures_dir = figures_dir
    cfg.vault_root = vault_root
    cfg.vault_figures_dir = vault_figures_dir
    return cfg


class TestMigrateFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.figures = self.root / "figures"
        self.vault = self.root / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ocr_figure(self, page_id: str, filename: str) -> Path:
        """Create a figure in the standard OCR nested structure."""
        imgs = self.figures / page_id / "imgs"
        imgs.mkdir(parents=True, exist_ok=True)
        p = imgs / filename
        p.write_bytes(b"img")
        return p

    def _make_flat_figure(self, filename: str) -> Path:
        """Create a figure in the flat EPUB structure."""
        self.figures.mkdir(parents=True, exist_ok=True)
        p = self.figures / filename
        p.write_bytes(b"img")
        return p

    def test_missing_vault_root_raises(self):
        cfg = Config()
        cfg.vault_root = None
        cfg.vault_figures_dir = "Figures"
        with self.assertRaises(ValueError):
            migrate_figures(cfg)

    def test_missing_vault_figures_dir_raises(self):
        cfg = Config()
        cfg.vault_root = str(self.vault)
        cfg.vault_figures_dir = None
        with self.assertRaises(ValueError):
            migrate_figures(cfg)

    def test_copies_nested_figures(self):
        self._make_ocr_figure("page_001", "fig1.png")
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        count = migrate_figures(cfg)
        self.assertEqual(count, 1)
        self.assertTrue((self.vault / "Figures" / "fig1.png").exists())

    def test_dry_run_does_not_copy(self):
        self._make_ocr_figure("page_001", "fig1.png")
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        count = migrate_figures(cfg, dry_run=True)
        self.assertEqual(count, 1)
        self.assertFalse((self.vault / "Figures" / "fig1.png").exists())

    def test_flat_mode_copies_from_flat_structure(self):
        self._make_flat_figure("fig1.png")
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        count = migrate_figures(cfg, flat=True)
        self.assertEqual(count, 1)
        self.assertTrue((self.vault / "Figures" / "fig1.png").exists())

    def test_skips_already_existing_files(self):
        self._make_ocr_figure("page_001", "fig1.png")
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        dest = self.vault / "Figures"
        dest.mkdir(parents=True)
        (dest / "fig1.png").write_bytes(b"existing")
        count = migrate_figures(cfg)
        self.assertEqual(count, 0)

    def test_page_ids_filter_limits_copy(self):
        self._make_ocr_figure("page_001", "fig1.png")
        self._make_ocr_figure("page_002", "fig2.png")
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        count = migrate_figures(cfg, page_ids=["page_001"])
        self.assertEqual(count, 1)
        self.assertTrue((self.vault / "Figures" / "fig1.png").exists())
        self.assertFalse((self.vault / "Figures" / "fig2.png").exists())

    def test_returns_zero_when_no_figures(self):
        self.figures.mkdir(parents=True)
        cfg = _cfg_migrate(str(self.figures), str(self.vault))
        count = migrate_figures(cfg)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
