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
        cfg = _config_module.Config()
        cfg.llama_server_path = None
        cfg.model_path = None
        cfg.mmproj_path = None
        with self.assertRaises(ValueError) as ctx:
            cfg.validate_ocr_paths()
        msg = str(ctx.exception)
        self.assertIn("llama_server_path", msg)
        self.assertIn("model_path", msg)
        self.assertIn("mmproj_path", msg)

    def test_partial_missing_paths_raises_on_validation(self):
        cfg = _config_module.Config()
        cfg.llama_server_path = "/fake/llama-server"
        cfg.model_path = None
        cfg.mmproj_path = "/fake/mmproj.gguf"
        with self.assertRaises(ValueError) as ctx:
            cfg.validate_ocr_paths()
        self.assertIn("model_path", str(ctx.exception))
        self.assertNotIn("llama_server_path", str(ctx.exception))

    def test_all_paths_set_does_not_raise(self):
        cfg = _config_module.Config()
        cfg.llama_server_path = "/fake/llama-server"
        cfg.model_path = "/fake/model.gguf"
        cfg.mmproj_path = "/fake/mmproj.gguf"
        cfg.validate_ocr_paths()  # must not raise

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

    def test_header_pattern_single(self):
        parser = build_parser()
        args = parser.parse_args(["--header-pattern", "^[IVX]+\\.", "2"])
        self.assertEqual(args.header_pattern, [["^[IVX]+\\.", "2"]])

    def test_header_pattern_multiple(self):
        parser = build_parser()
        args = parser.parse_args([
            "--header-pattern", "^Chapter \\d+", "1",
            "--header-pattern", "^Section \\d+", "2",
        ])
        self.assertEqual(len(args.header_pattern), 2)
        self.assertEqual(args.header_pattern[0], ["^Chapter \\d+", "1"])
        self.assertEqual(args.header_pattern[1], ["^Section \\d+", "2"])

    def test_header_pattern_absent_is_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.header_pattern)


if __name__ == "__main__":
    unittest.main()
