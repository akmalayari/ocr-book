#!/usr/bin/env python3
"""
setup.py — Setup ocr-livre environment (PaddleOCR version)

Usage:
    python setup.py                    # Full setup
    python setup.py --env-only         # Only create conda env
    python setup.py --patch-only       # Only apply paddlex patch OTSL
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], description: str) -> bool:
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"→ {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed (exit code {e.returncode})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup ocr-livre environment")
    parser.add_argument("--env-only", action="store_true", help="Only create conda env")
    parser.add_argument("--patch-only", action="store_true", help="Only apply paddlex patch")
    args = parser.parse_args()

    root = Path(__file__).parent

    steps = []

    if not args.patch_only:
        steps.extend([
            (["conda", "env", "remove", "-n", "ocr-livre", "--yes"],
             "Remove old ocr-livre env (if exists)", True),
            (["conda", "env", "create", "-f", str(root / "environment.yml")],
             "Create conda environment from environment.yml", False),
        ])

    if not args.env_only:
        conda_run = ["conda", "run", "-n", "ocr-livre", "--no-capture-output"]
        steps.extend([
            (conda_run + ["pip", "install", "git+https://github.com/PaddlePaddle/PaddleOCR.git@740a04dc4"],
             "Install PaddleOCR from repo (with llama-server compatibility)", False),
            (conda_run + ["python", str(root / "docs" / "dev" / "apply_paddlex_patch_otsl.py")],
             "Apply paddlex patch (per-region VLM error recovery)", False),
            (conda_run + ["python", "-c", "from paddleocr import PaddleOCRVL; print('✅ PaddleOCR loaded')"],
             "Verify PaddleOCR import", False),
        ])

    print("\n" + "="*60)
    print("ocr-livre setup — PaddleOCR version")
    print("="*60)

    failed = []
    for cmd, desc, optional in steps:
        if not run(cmd, desc) and not optional:
            failed.append(desc)

    print("\n" + "="*60)
    if failed:
        print(f"❌ Setup incomplete. Failed steps:")
        for desc in failed:
            print(f"  - {desc}")
        print("\nNote: If commands fail, you may need to:")
        print("  1. conda activate ocr-livre")
        print("  2. Re-run: python setup.py --patch-only")
        sys.exit(1)
    else:
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("  1. conda activate ocr-livre (if not already active)")
        print("  2. Configure llama-server and model paths (pick one):")
        print("       a) cp .env.example .env  → edit the file")
        print("       b) Export env vars OCR_LLAMA_SERVER_PATH, OCR_MODEL_PATH, OCR_MMPROJ_PATH")
        print("       c) Pass --llama-server, --model, --mmproj to main.py")
        print("  3. python main.py --help")
        print("="*60)


if __name__ == "__main__":
    main()
