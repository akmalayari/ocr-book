"""
Unit tests for src/obsidian.py — pure string-transformation functions only.
No filesystem, vault, or llama-server required.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from obsidian import fix_image_paths_obsidian, fix_markdown_image_paths_obsidian


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


if __name__ == "__main__":
    unittest.main()
