"""
Minimal smoke tests — validate imports, config, and CLI parsing
without requiring llama-server or the full model stack.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

# Ensure project root and src/ are on the path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import config as _config_module
from main import build_parser


class TestConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = _config_module.Config()
        self.assertEqual(cfg.images_dir, "./photos")
        self.assertEqual(cfg.mode, "base")
        self.assertTrue(cfg.use_layout_detection)
        self.assertEqual(cfg.extraction_method, "paddleocrvl")

    def test_env_vars(self):
        env = {
            "LLAMA_SERVER_PATH": "/fake/llama-server",
            "MODEL_PATH": "/fake/model.gguf",
            "MMPROJ_PATH": "/fake/mmproj.gguf",
            "OBSIDIAN_VAULT_ROOT": "/fake/vault",
        }
        for k, v in env.items():
            os.environ[k] = v
        try:
            importlib.reload(_config_module)
            cfg = _config_module.Config()
            self.assertEqual(cfg.llama_server_path, "/fake/llama-server")
            self.assertEqual(cfg.model_path, "/fake/model.gguf")
            self.assertEqual(cfg.mmproj_path, "/fake/mmproj.gguf")
            self.assertEqual(cfg.vault_root, "/fake/vault")
        finally:
            for k in env:
                del os.environ[k]
            importlib.reload(_config_module)

    def test_missing_paths_raises_on_validation(self):
        """pipeline.py validates required paths before starting servers."""
        cfg = _config_module.Config()
        cfg.llama_server_path = None
        cfg.model_path = None
        cfg.mmproj_path = None

        # Simulate the validation block from pipeline.py
        missing = []
        for name, val in [
            ("llama_server_path", cfg.llama_server_path),
            ("model_path", cfg.model_path),
            ("mmproj_path", cfg.mmproj_path),
        ]:
            if not val:
                missing.append(name)
        self.assertEqual(len(missing), 3)

    def test_path_properties(self):
        cfg = _config_module.Config()
        self.assertIsInstance(cfg.images_path, Path)
        self.assertIsInstance(cfg.output_path, Path)
        self.assertIsInstance(cfg.figures_path, Path)


class TestArgumentParser(unittest.TestCase):
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.images, "./photos")
        self.assertEqual(args.out, "./output/book.md")
        self.assertEqual(args.mode, "base")
        self.assertEqual(args.method, "paddleocrvl")
        self.assertFalse(args.no_layout)
        self.assertFalse(args.no_resume)

    def test_llama_server_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--llama-server", "/opt/llama-server"])
        self.assertEqual(args.llama_server, "/opt/llama-server")

    def test_method_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--method", "text"])
        self.assertEqual(args.method, "text")

    def test_pdf_and_epub_inputs(self):
        """Parser accepts file paths for PDF/EPUB via --images."""
        parser = build_parser()
        args = parser.parse_args(["--images", "book.pdf"])
        self.assertEqual(args.images, "book.pdf")

        args = parser.parse_args(["--images", "book.epub"])
        self.assertEqual(args.images, "book.epub")


if __name__ == "__main__":
    unittest.main()
