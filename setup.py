#!/usr/bin/env python3
"""
setup.py — Setup the ocr-livre environment and optional llama-server build.

Usage:
    python setup.py                    # Full Python environment setup
    python setup.py --build-llama      # Full setup + latest Vulkan llama-server
    python setup.py --llama-only       # Only build latest Vulkan llama-server
    python setup.py --env-only         # Only recreate the conda environment
    python setup.py --patch-only       # Only install/patch/verify PaddleOCR
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ENV_NAME = "ocr-livre"
LLAMA_CPP_REF = "master"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"


def run(cmd: list[str], description: str) -> bool:
    """Run a command and report status without hiding its output."""
    print(f"\n{'=' * 60}")
    print(f"→ {description}")
    print(f"{'=' * 60}")
    print(f"$ {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ {description} failed (exit code {exc.returncode})")
    except OSError as exc:
        print(f"❌ {description} could not start: {exc}")
    return False


def _missing_commands(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if shutil.which(name) is None]


def _record_llama_path(root: Path, executable: Path) -> None:
    """Record the server path, replacing only an obvious stale placeholder."""
    env_path = root / ".env"
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    escaped = str(executable.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    assignment = f'OCR_LLAMA_SERVER_PATH="{escaped}"'
    existing = re.search(
        r"^\s*OCR_LLAMA_SERVER_PATH\s*=.*$",
        text,
        flags=re.MULTILINE,
    )
    if existing:
        current = existing.group(0).split("=", 1)[1].strip().strip("'\"")
        stale_windows_path = platform.system() == "Linux" and current.lower().endswith(".exe")
        placeholder = "/path/to/" in current or "/absolute/path/to/" in current
        if not stale_windows_path and not placeholder:
            print(f"Keeping the existing OCR_LLAMA_SERVER_PATH in {env_path}")
            print(f"Built executable (set it manually if needed): {executable.resolve()}")
            return
        text = text[:existing.start()] + assignment + text[existing.end():]
        print(f"Replaced the stale OCR_LLAMA_SERVER_PATH in {env_path}")
    else:
        if text and not text.endswith(("\n", "\r")):
            text += "\n"
        text += assignment + "\n"
    env_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Recorded OCR_LLAMA_SERVER_PATH in {env_path}")


def build_llama_server(
    root: Path,
    ref: str = LLAMA_CPP_REF,
    cache_dir: str | Path | None = None,
    jobs: int = 8,
) -> Path | None:
    """Build llama-server with Vulkan, refreshing the requested upstream ref."""
    if platform.system() != "Linux":
        print("❌ Automated llama-server builds currently support Linux only.")
        print("   On Windows, configure an existing llama-server Vulkan executable.")
        return None

    missing = _missing_commands(("git", "cmake", "c++", "glslc"))
    if missing:
        print(f"❌ Missing llama.cpp build tools: {', '.join(missing)}")
        print("   Ubuntu: sudo apt install build-essential cmake git libvulkan-dev glslc spirv-headers")
        return None
    if shutil.which("vulkaninfo") is None:
        print("⚠ vulkaninfo was not found; install vulkan-tools to diagnose the GPU runtime.")

    if cache_dir is None:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        cache_root /= "ocr-book"
    else:
        cache_root = Path(os.path.expandvars(str(cache_dir))).expanduser()

    safe_ref = re.sub(r"[^A-Za-z0-9._-]", "_", ref)
    source_dir = cache_root / f"llama.cpp-{safe_ref}"
    build_dir = source_dir / "build-vulkan"

    if source_dir.exists() and not (source_dir / ".git").is_dir():
        print(f"❌ Build cache exists but is not a llama.cpp checkout: {source_dir}")
        print("   Move it aside and retry.")
        return None
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        if not run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_CPP_REPO, str(source_dir)],
            "Clone llama.cpp build cache",
        ):
            return None
    else:
        print(f"Reusing cached llama.cpp checkout: {source_dir}")

    if not run(
        ["git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", ref],
        f"Fetch current llama.cpp {ref}",
    ):
        return None
    if not run(
        ["git", "-C", str(source_dir), "checkout", "--detach", "FETCH_HEAD"],
        f"Check out current llama.cpp {ref}",
    ):
        return None

    if not run(
        [
            "cmake", "-S", str(source_dir), "-B", str(build_dir),
            "-DGGML_VULKAN=ON", "-DLLAMA_BUILD_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "Configure llama-server with Vulkan",
    ):
        return None
    if not run(
        [
            "cmake", "--build", str(build_dir), "--config", "Release",
            "--target", "llama-server", "--parallel", str(max(1, jobs)),
        ],
        "Build llama-server",
    ):
        return None

    executable = build_dir / "bin" / "llama-server"
    if not executable.is_file():
        print(f"❌ Build completed but llama-server was not found at {executable}")
        return None
    executable.chmod(executable.stat().st_mode | 0o111)
    if not run([str(executable), "--version"], "Verify llama-server executable"):
        return None
    if not run([str(executable), "--list-devices"], "List llama-server compute devices"):
        return None

    _record_llama_path(root, executable)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup ocr-livre environment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--env-only", action="store_true", help="Only create the conda environment")
    mode.add_argument("--patch-only", action="store_true", help="Only install and patch PaddleOCR")
    mode.add_argument("--llama-only", action="store_true", help="Only build llama-server (Linux)")
    parser.add_argument(
        "--build-llama", action="store_true",
        help="Build the latest Vulkan llama-server after the normal setup (Linux)",
    )
    parser.add_argument(
        "--llama-ref", default=LLAMA_CPP_REF, metavar="REF",
        help=f"llama.cpp branch, tag, or commit to build (default: {LLAMA_CPP_REF}, refreshed)",
    )
    parser.add_argument(
        "--llama-build-dir", metavar="PATH",
        help="Build cache root (default: XDG_CACHE_HOME/ocr-book or ~/.cache/ocr-book)",
    )
    parser.add_argument(
        "--llama-jobs", type=int, default=8, metavar="N",
        help="Maximum parallel llama.cpp build jobs (default: 8)",
    )
    args = parser.parse_args()
    if args.llama_jobs < 1:
        parser.error("--llama-jobs must be at least 1")

    root = Path(__file__).resolve().parent
    llama_requested = args.build_llama or args.llama_only
    steps: list[tuple[list[str], str, bool]] = []

    if not args.patch_only and not args.llama_only:
        steps.extend([
            (["conda", "env", "remove", "-n", ENV_NAME, "--yes"],
             f"Remove old {ENV_NAME} env (if it exists)", True),
            (["conda", "env", "create", "-f", str(root / "environment.yml")],
             "Create conda environment from environment.yml", False),
        ])

    if not args.env_only and not args.llama_only:
        conda_run = ["conda", "run", "-n", ENV_NAME, "--no-capture-output"]
        steps.extend([
            (conda_run + [
                "python", "-m", "pip", "install",
                "git+https://github.com/PaddlePaddle/PaddleOCR.git@740a04dc4",
            ], "Install PaddleOCR from repo (with llama-server compatibility)", False),
            (conda_run + ["python", str(root / "docs/dev/apply_paddlex_patch_otsl.py")],
             "Apply paddlex patch (per-region VLM error recovery)", False),
            (conda_run + [
                "python", "-c", "from paddleocr import PaddleOCRVL; print('PaddleOCR loaded')",
            ], "Verify PaddleOCR import", False),
        ])

    print("\n" + "=" * 60)
    print("ocr-livre setup — PaddleOCR version")
    print("=" * 60)

    failed = []
    if steps and shutil.which("conda") is None:
        failed.append("Conda is not installed or is not available on PATH")
        print("❌ Conda was not found.")
        print("   Install Miniforge, reopen the shell, then rerun this command.")
    else:
        for cmd, desc, optional in steps:
            if not run(cmd, desc) and not optional:
                failed.append(desc)

    llama_path = None
    if llama_requested:
        llama_path = build_llama_server(
            root,
            ref=args.llama_ref,
            cache_dir=args.llama_build_dir,
            jobs=args.llama_jobs,
        )
        if llama_path is None:
            failed.append("Build llama-server")

    print("\n" + "=" * 60)
    if failed:
        print("❌ Setup incomplete. Failed steps:")
        for desc in failed:
            print(f"  - {desc}")
        print("\nResolve the reported prerequisite or command error, then rerun setup.")
        sys.exit(1)

    print("✅ Setup complete!")
    if llama_path:
        print(f"  llama-server: {llama_path}")
    if not args.llama_only:
        print("\nNext steps:")
        print(f"  1. conda activate {ENV_NAME} (if not already active)")
        print("  2. Configure the model and mmproj paths in .env or through CLI flags")
        if not llama_path:
            print("  3. Configure OCR_LLAMA_SERVER_PATH or run: python setup.py --llama-only")
        print("  4. python main.py --help")
    print("=" * 60)


if __name__ == "__main__":
    main()
