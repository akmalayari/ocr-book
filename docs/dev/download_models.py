#!/usr/bin/env python3
"""Download the pinned PaddleOCR-VL GGUF assets and record their paths."""

import argparse
from io import StringIO
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_assets import (  # noqa: E402
    MMPROJ_FILENAME,
    MMPROJ_FILE_SIZE,
    MODEL_FILENAME,
    MODEL_FILE_SIZE,
    MODEL_REPO_ID,
    MODEL_REVISION,
    default_model_paths,
)


def _asset_is_valid(path: Path | None, expected_size: int | None, label: str) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        actual_size = path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            print(
                f"Ignoring incomplete {label}: {path} "
                f"({actual_size} bytes, expected {expected_size})"
            )
            return False
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                print(f"Ignoring invalid {label} (missing GGUF header): {path}")
                return False
    except OSError as exc:
        print(f"Ignoring unreadable {label}: {path} ({exc})")
        return False
    return True


def _download_url(filename: str, revision: str) -> str:
    revision_path = quote(revision, safe="")
    filename_path = quote(filename, safe="")
    return (
        f"https://huggingface.co/{MODEL_REPO_ID}/resolve/"
        f"{revision_path}/{filename_path}?download=true"
    )


def _print_browser_instructions(
    model_path: Path,
    mmproj_path: Path,
    revision: str,
) -> None:
    print("Windows automatic model download is disabled by default.")
    print("Download both files in a browser:")
    print(f"  model : {_download_url(MODEL_FILENAME, revision)}")
    print(f"  mmproj: {_download_url(MMPROJ_FILENAME, revision)}")
    print("Option 1 - Save them in the default locations (automatically detected):")
    print(f"  model : {model_path.resolve()}")
    print(f"  mmproj: {mmproj_path.resolve()}")
    print("Option 2 - Save them anywhere and set their paths in .env:")
    print('  OCR_MODEL_PATH="C:/path/to/PaddleOCR-VL-1.5.gguf"')
    print('  OCR_MMPROJ_PATH="C:/path/to/PaddleOCR-VL-1.5-mmproj.gguf"')
    print("Then rerun the model check to validate and record the paths.")


def _read_env_text(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _env_file_value(text: str, name: str) -> str | None:
    value = dotenv_values(stream=StringIO(text)).get(name)
    return value.strip() if value and value.strip() else None


def _resolve_configured_path(text: str, name: str) -> Path | None:
    value = os.environ.get(name, "").strip() or _env_file_value(text, name)
    if not value:
        return None
    return Path(os.path.expandvars(value)).expanduser()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.as_posix() if os.name == "nt" else str(resolved)


def _set_env_values(path: Path, values: dict[str, Path]) -> None:
    text = _read_env_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    for name, value in values.items():
        assignment = f'{name}="{_portable_path(value)}"'
        pattern = re.compile(
            rf"^\s*(?:export\s+)?{re.escape(name)}\s*=.*$", re.MULTILINE
        )
        if pattern.search(text):
            text = pattern.sub(assignment, text)
        else:
            if text and not text.endswith(("\n", "\r")):
                text += newline
            text += assignment + newline
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PaddleOCR-VL-1.5 GGUF assets")
    parser.add_argument("--model-dir", help="Destination directory (default: user cache)")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--force-location",
        action="store_true",
        help="Use --model-dir even when valid model paths are already configured",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Validate/register local files without accessing Hugging Face",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser()
    env_text = _read_env_text(env_file)
    configured_model = _resolve_configured_path(env_text, "OCR_MODEL_PATH")
    configured_mmproj = _resolve_configured_path(env_text, "OCR_MMPROJ_PATH")
    exact_sizes = args.revision == MODEL_REVISION
    keep_model = (
        not args.force_location
        and _asset_is_valid(
            configured_model,
            MODEL_FILE_SIZE if exact_sizes else None,
            "model",
        )
    )
    keep_mmproj = (
        not args.force_location
        and _asset_is_valid(
            configured_mmproj,
            MMPROJ_FILE_SIZE if exact_sizes else None,
            "mmproj",
        )
    )
    if keep_model and keep_mmproj:
        print("Keeping existing model files:")
        print(f"  model : {configured_model.resolve()}")
        print(f"  mmproj: {configured_mmproj.resolve()}")
        return 0

    configured_cache = (
        os.environ.get("OCR_MODEL_CACHE_DIR", "").strip()
        or _env_file_value(env_text, "OCR_MODEL_CACHE_DIR")
    )
    cached_model, cached_mmproj = default_model_paths(args.model_dir or configured_cache)
    model_path = configured_model if keep_model else cached_model
    mmproj_path = configured_mmproj if keep_mmproj else cached_mmproj

    if configured_model and not keep_model and not args.force_location:
        print(f"Ignoring missing configured model: {configured_model}")
    if configured_mmproj and not keep_mmproj and not args.force_location:
        print(f"Ignoring missing configured mmproj: {configured_mmproj}")

    cached_model_valid = keep_model or _asset_is_valid(
        model_path,
        MODEL_FILE_SIZE if exact_sizes else None,
        "model",
    )
    cached_mmproj_valid = keep_mmproj or _asset_is_valid(
        mmproj_path,
        MMPROJ_FILE_SIZE if exact_sizes else None,
        "mmproj",
    )
    if cached_model_valid and cached_mmproj_valid:
        _set_env_values(env_file, {
            "OCR_MODEL_PATH": model_path,
            "OCR_MMPROJ_PATH": mmproj_path,
        })
        print(f"Recorded model paths in {env_file}")
        print(f"  model : {model_path.resolve()}")
        print(f"  mmproj: {mmproj_path.resolve()}")
        return 0

    if args.local_only:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        mmproj_path.parent.mkdir(parents=True, exist_ok=True)
        _print_browser_instructions(model_path, mmproj_path, args.revision)
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed in the active environment.")
        return 1

    print(f"Downloading missing assets from pinned revision {args.revision}")
    try:
        if keep_model:
            downloaded_model = configured_model
            print(f"Keeping existing model: {downloaded_model.resolve()}")
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_model = Path(hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                revision=args.revision,
                local_dir=model_path.parent,
            ))

        if keep_mmproj:
            downloaded_mmproj = configured_mmproj
            print(f"Keeping existing mmproj: {downloaded_mmproj.resolve()}")
        else:
            mmproj_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_mmproj = Path(hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MMPROJ_FILENAME,
                revision=args.revision,
                local_dir=mmproj_path.parent,
            ))
    except Exception as exc:
        print(f"[ERROR] Model download failed: {exc}")
        print("Rerun the command to resume the download.")
        return 1

    if not _asset_is_valid(
        downloaded_model,
        MODEL_FILE_SIZE if exact_sizes else None,
        "model",
    ) or not _asset_is_valid(
        downloaded_mmproj,
        MMPROJ_FILE_SIZE if exact_sizes else None,
        "mmproj",
    ):
        print("[ERROR] Download completed without both valid model files.")
        return 1

    _set_env_values(env_file, {
        "OCR_MODEL_PATH": downloaded_model,
        "OCR_MMPROJ_PATH": downloaded_mmproj,
    })
    print(f"Recorded model paths in {env_file}")
    print(f"  model : {downloaded_model.resolve()}")
    print(f"  mmproj: {downloaded_mmproj.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
