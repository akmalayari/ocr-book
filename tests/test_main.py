"""
test_main.py — Tests unitaires pour main.py

Couvre :
  - build_parser : valeurs par défaut, surcharges, types, mode rec
  - main()       : construction du Config, codes de retour,
                   branche --rename, propagation des erreurs fatales
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from main import main, build_parser
from progress import Stats


# ── build_parser : valeurs par défaut ────────────────────────────────────────

class TestBuildParserDefaults:

    def _parse(self, args=None):
        return build_parser().parse_args(args or [])

    def test_images_default(self):
        assert self._parse().images == "./photos"

    def test_out_default(self):
        assert self._parse().out == "./output/livre.md"

    def test_model_default(self):
        assert self._parse().model == "NexaAI/DeepSeek-OCR-GGUF"

    def test_mode_default(self):
        assert self._parse().mode == "plain"

    def test_max_tokens_default(self):
        assert self._parse().max_tokens == 4096

    def test_quant_default(self):
        assert self._parse().quant == "bf16"

    def test_no_resume_default_false(self):
        assert self._parse().no_resume is False

    def test_verbose_default_false(self):
        assert self._parse().verbose is False

    def test_rename_default_false(self):
        assert self._parse().rename is False

    def test_rename_prefix_default(self):
        assert self._parse().rename_prefix == "page"

    def test_dry_run_default_false(self):
        assert self._parse().dry_run is False


# ── build_parser : surcharges ─────────────────────────────────────────────────

class TestBuildParserOverrides:

    def _parse(self, args):
        return build_parser().parse_args(args)

    def test_images_override(self):
        assert self._parse(["--images", "/mes/photos"]).images == "/mes/photos"

    def test_out_override(self):
        assert self._parse(["--out", "mon_livre.md"]).out == "mon_livre.md"

    def test_mode_plain(self):
        assert self._parse(["--mode", "plain"]).mode == "plain"

    def test_mode_layout(self):
        assert self._parse(["--mode", "layout"]).mode == "layout"

    def test_mode_describe(self):
        assert self._parse(["--mode", "describe"]).mode == "describe"

    def test_mode_rec_with_target(self):
        assert self._parse(["--mode", "rec:titre"]).mode == "rec:titre"

    def test_max_tokens_override(self):
        assert self._parse(["--max-tokens", "1024"]).max_tokens == 1024

    def test_quant_q8_0(self):
        assert self._parse(["--quant", "q8_0"]).quant == "q8_0"

    def test_no_resume_flag(self):
        assert self._parse(["--no-resume"]).no_resume is True

    def test_verbose_flag(self):
        assert self._parse(["--verbose"]).verbose is True

    def test_rename_flag(self):
        assert self._parse(["--rename"]).rename is True

    def test_rename_prefix_override(self):
        assert self._parse(["--rename", "--rename-prefix", "scan"]).rename_prefix == "scan"

    def test_dry_run_flag(self):
        assert self._parse(["--rename", "--dry-run"]).dry_run is True

    def test_invalid_mode_raises_system_exit(self):
        # La validation du mode est dans main(), pas dans build_parser()
        with patch("sys.argv", ["main.py", "--mode", "mode_invalide"]):
            with patch("main.setup_logging"):
                with pytest.raises(SystemExit):
                    main()

    def test_invalid_quant_raises_system_exit(self):
        with pytest.raises(SystemExit):
            self._parse(["--quant", "f16"])


# ── main() : codes de retour ──────────────────────────────────────────────────

class TestMainReturnCodes:

    def _run_main(self, cli_args, stats=None, pipeline_raises=None):
        if stats is None:
            stats = Stats(total=1)
            stats.record_success(10.0, 100)

        with patch("sys.argv", ["main.py"] + cli_args):
            with patch("main.setup_logging"):
                with patch("main.run_pipeline", return_value=stats) as mock_pipeline:
                    if pipeline_raises:
                        mock_pipeline.side_effect = pipeline_raises
                    return main()

    def test_returns_0_on_full_success(self):
        stats = Stats(total=2)
        stats.record_success(10.0, 100)
        stats.record_success(8.0, 200)
        assert self._run_main([], stats=stats) == 0

    def test_returns_2_when_some_errors(self):
        stats = Stats(total=3)
        stats.record_success(10.0, 100)
        stats.record_error()
        assert self._run_main([], stats=stats) == 2

    def test_returns_1_on_fatal_exception(self):
        assert self._run_main([], pipeline_raises=RuntimeError("Erreur fatale")) == 1

    def test_returns_1_on_image_collection_error(self):
        from images import ImageCollectionError
        assert self._run_main([], pipeline_raises=ImageCollectionError("Dossier vide")) == 1


# ── main() : construction du Config ──────────────────────────────────────────

class TestMainConfigConstruction:

    def _run_and_capture_cfg(self, cli_args):
        captured = {}
        stats = Stats(total=0)

        def fake_pipeline(cfg):
            captured["cfg"] = cfg
            return stats

        with patch("sys.argv", ["main.py"] + cli_args):
            with patch("main.setup_logging"):
                with patch("main.run_pipeline", side_effect=fake_pipeline):
                    main()
        return captured.get("cfg")

    def test_config_images_dir_from_args(self):
        cfg = self._run_and_capture_cfg(["--images", "/custom/photos"])
        assert cfg.images_dir == "/custom/photos"

    def test_config_output_file_from_args(self):
        cfg = self._run_and_capture_cfg(["--out", "custom.md"])
        assert cfg.output_file == "custom.md"

    def test_config_prompt_mode_plain(self):
        cfg = self._run_and_capture_cfg(["--mode", "plain"])
        assert cfg.prompt_mode == "plain"

    def test_config_prompt_mode_layout(self):
        cfg = self._run_and_capture_cfg(["--mode", "layout"])
        assert cfg.prompt_mode == "layout"

    def test_config_prompt_mode_rec(self):
        cfg = self._run_and_capture_cfg(["--mode", "rec:un graphique"])
        assert cfg.prompt_mode == "rec"
        assert cfg.locate_target == "un graphique"

    def test_config_resume_false_when_no_resume_flag(self):
        cfg = self._run_and_capture_cfg(["--no-resume"])
        assert cfg.resume is False

    def test_config_resume_true_by_default(self):
        cfg = self._run_and_capture_cfg([])
        assert cfg.resume is True

    def test_config_verbose_from_args(self):
        cfg = self._run_and_capture_cfg(["--verbose"])
        assert cfg.verbose is True

    def test_config_max_tokens_from_args(self):
        cfg = self._run_and_capture_cfg(["--max-tokens", "512"])
        assert cfg.max_tokens == 512

    def test_config_quant_from_args(self):
        cfg = self._run_and_capture_cfg(["--quant", "q8_0"])
        assert cfg.quant == "q8_0"


# ── main() : branche --rename ─────────────────────────────────────────────────

class TestMainRename:

    def test_rename_calls_rename_images(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename", "--dry-run",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    main()
        mock_rename.assert_called_once()

    def test_rename_dry_run_returns_0(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename", "--dry-run",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images"):
                    result = main()
        assert result == 0

    def test_rename_dry_run_does_not_call_pipeline(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename", "--dry-run",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images"):
                    with patch("main.run_pipeline") as mock_pipeline:
                        main()
        mock_pipeline.assert_not_called()

    def test_rename_passes_prefix(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename", "--dry-run",
                                "--rename-prefix", "scan",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    main()
        _, kwargs = mock_rename.call_args
        assert kwargs.get("prefix") == "scan"

    def test_rename_passes_dry_run(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename", "--dry-run",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    main()
        _, kwargs = mock_rename.call_args
        assert kwargs.get("dry_run") is True
