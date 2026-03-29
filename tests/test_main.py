"""
test_main.py — Tests unitaires pour main.py

Couvre :
  - parse_args : valeurs par défaut, surcharges, types
  - main()     : construction du Config depuis les args, codes de retour,
                 branche rename-only, propagation des erreurs fatales
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from main import main, parse_args
from progress import Stats


# ── parse_args : valeurs par défaut ──────────────────────────────────────────

class TestParseArgsDefaults:

    def _parse(self, args=None):
        with patch("sys.argv", ["main.py"] + (args or [])):
            return parse_args()

    def test_images_default(self):
        args = self._parse()
        assert args.images == "./photos"

    def test_out_default(self):
        args = self._parse()
        assert args.out == "livre.md"

    def test_model_default(self):
        args = self._parse()
        assert args.model == "NexaAI/DeepSeek-OCR-GGUF"

    def test_port_default(self):
        args = self._parse()
        assert args.port == 18181

    def test_mode_default(self):
        args = self._parse()
        assert args.mode == "markdown"

    def test_max_tokens_default(self):
        args = self._parse()
        assert args.max_tokens == 4096

    def test_timeout_default(self):
        args = self._parse()
        assert args.timeout == 180

    def test_no_resume_default_false(self):
        args = self._parse()
        assert args.no_resume is False

    def test_verbose_default_false(self):
        args = self._parse()
        assert args.verbose is False

    def test_rename_only_default_false(self):
        args = self._parse()
        assert args.rename_only is False

    def test_rename_prefix_default(self):
        args = self._parse()
        assert args.rename_prefix == "page"

    def test_dry_run_default_false(self):
        args = self._parse()
        assert args.dry_run is False


# ── parse_args : surcharges ───────────────────────────────────────────────────

class TestParseArgsOverrides:

    def _parse(self, args):
        with patch("sys.argv", ["main.py"] + args):
            return parse_args()

    def test_images_override(self):
        args = self._parse(["--images", "/mes/photos"])
        assert args.images == "/mes/photos"

    def test_out_override(self):
        args = self._parse(["--out", "mon_livre.md"])
        assert args.out == "mon_livre.md"

    def test_model_override(self):
        args = self._parse(["--model", "NexaAI/Autre-GGUF"])
        assert args.model == "NexaAI/Autre-GGUF"

    def test_port_override(self):
        args = self._parse(["--port", "9090"])
        assert args.port == 9090

    def test_port_is_int(self):
        args = self._parse(["--port", "8080"])
        assert isinstance(args.port, int)

    def test_mode_plain(self):
        args = self._parse(["--mode", "plain"])
        assert args.mode == "plain"

    def test_mode_figure(self):
        args = self._parse(["--mode", "figure"])
        assert args.mode == "figure"

    def test_max_tokens_override(self):
        args = self._parse(["--max-tokens", "1024"])
        assert args.max_tokens == 1024

    def test_timeout_override(self):
        args = self._parse(["--timeout", "60"])
        assert args.timeout == 60

    def test_no_resume_flag(self):
        args = self._parse(["--no-resume"])
        assert args.no_resume is True

    def test_verbose_flag(self):
        args = self._parse(["--verbose"])
        assert args.verbose is True

    def test_rename_only_flag(self):
        args = self._parse(["--rename-only"])
        assert args.rename_only is True

    def test_rename_prefix_override(self):
        args = self._parse(["--rename-only", "--rename-prefix", "scan"])
        assert args.rename_prefix == "scan"

    def test_dry_run_flag(self):
        args = self._parse(["--rename-only", "--dry-run"])
        assert args.dry_run is True

    def test_invalid_mode_raises_system_exit(self):
        with pytest.raises(SystemExit):
            self._parse(["--mode", "invalid_mode"])


# ── main() : codes de retour ──────────────────────────────────────────────────

class TestMainReturnCodes:

    def _run_main(self, cli_args, stats=None, pipeline_raises=None):
        """Helper : lance main() avec des args et des mocks."""
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
        code = self._run_main([], stats=stats)
        assert code == 0

    def test_returns_2_when_some_errors(self):
        stats = Stats(total=3)
        stats.record_success(10.0, 100)
        stats.record_error()
        code = self._run_main([], stats=stats)
        assert code == 2

    def test_returns_1_on_fatal_exception(self):
        code = self._run_main([], pipeline_raises=RuntimeError("Erreur fatale"))
        assert code == 1

    def test_returns_1_on_image_collection_error(self):
        from images import ImageCollectionError
        code = self._run_main([], pipeline_raises=ImageCollectionError("Dossier vide"))
        assert code == 1


# ── main() : construction du Config ──────────────────────────────────────────

class TestMainConfigConstruction:

    def _run_and_capture_cfg(self, cli_args):
        """Lance main() et capture le Config passé à run_pipeline."""
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

    def test_config_model_from_args(self):
        cfg = self._run_and_capture_cfg(["--model", "NexaAI/Custom"])
        assert cfg.model == "NexaAI/Custom"

    def test_config_port_from_args(self):
        cfg = self._run_and_capture_cfg(["--port", "9999"])
        assert cfg.port == 9999

    def test_config_prompt_mode_from_args(self):
        cfg = self._run_and_capture_cfg(["--mode", "plain"])
        assert cfg.prompt_mode == "plain"

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

    def test_config_timeout_from_args(self):
        cfg = self._run_and_capture_cfg(["--timeout", "30"])
        assert cfg.request_timeout_s == 30


# ── main() : branche rename-only ─────────────────────────────────────────────

class TestMainRenameOnly:

    def test_rename_only_calls_rename_images(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename-only",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    result = main()
        mock_rename.assert_called_once()

    def test_rename_only_returns_0(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename-only",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images"):
                    result = main()
        assert result == 0

    def test_rename_only_does_not_call_pipeline(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename-only",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images"):
                    with patch("main.run_pipeline") as mock_pipeline:
                        main()
        mock_pipeline.assert_not_called()

    def test_rename_only_passes_prefix(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename-only",
                                "--rename-prefix", "scan",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    main()
        _, kwargs = mock_rename.call_args
        assert kwargs.get("prefix") == "scan"

    def test_rename_only_passes_dry_run(self, tmp_path):
        with patch("sys.argv", ["main.py", "--rename-only", "--dry-run",
                                "--images", str(tmp_path)]):
            with patch("main.setup_logging"):
                with patch("main.rename_images") as mock_rename:
                    main()
        _, kwargs = mock_rename.call_args
        assert kwargs.get("dry_run") is True
