"""
apply_paddlex_patch.py — Applique (ou vérifie) le patch paddlex per-region VLM.

Voir paddlex_patch.md pour la description complète.

Usage :
    python docs/dev/apply_paddlex_patch.py           # applique le patch
    python docs/dev/apply_paddlex_patch.py --check   # vérifie sans modifier
    python docs/dev/apply_paddlex_patch.py --revert  # restaure l'original
"""

import argparse
import sys
from pathlib import Path

TARGET = (
    Path(sys.prefix)
    / "Lib/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py"
)

ORIGINAL = """\
            images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]
            batch_results = list(
                self.vl_rec_model.predict(
                    [
                        {
                            "image": image,
                            "query": query,
                        }
                        for image, query in zip(images, queries)
                    ],
                    skip_special_tokens=False if has_spotting else True,
                    **kwargs,
                )
            )
            del images, queries
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results"""

PATCHED = """\
            images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]
            batch_results = []
            for image, query in zip(images, queries):
                try:
                    result = list(
                        self.vl_rec_model.predict(
                            [{"image": image, "query": query}],
                            skip_special_tokens=False if has_spotting else True,
                            **kwargs,
                        )
                    )[0]
                except Exception as _vlm_err:
                    err_msg = str(_vlm_err)
                    otsl_start = err_msg.find("<fcel>")
                    if otsl_start != -1:
                        # OTSL content echoed back by llama-server; convert directly
                        result = {"result": err_msg[otsl_start:]}
                    else:
                        result = {"result": ""}
                batch_results.append(result)
            del images, queries
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results"""


def status(text: str) -> str:
    if PATCHED in text:
        return "patched"
    if ORIGINAL in text:
        return "original"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true", help="Vérifie sans modifier.")
    parser.add_argument("--revert", action="store_true", help="Restaure l'original.")
    args = parser.parse_args()

    if not TARGET.exists():
        print(f"[ERREUR] Fichier introuvable : {TARGET}")
        sys.exit(1)

    text = TARGET.read_text(encoding="utf-8")
    state = status(text)
    print(f"Fichier  : {TARGET}")
    print(f"État     : {state}")

    if args.check:
        sys.exit(0 if state == "patched" else 1)

    if args.revert:
        if state == "original":
            print("Déjà à l'état original, rien à faire.")
            return
        if state != "patched":
            print("[ERREUR] État inconnu, modification manuelle requise.")
            sys.exit(1)
        TARGET.write_text(text.replace(PATCHED, ORIGINAL), encoding="utf-8")
        print("Patch retiré.")
        return

    # Appliquer le patch
    if state == "patched":
        print("Déjà patché, rien à faire.")
        return
    if state != "original":
        print("[ERREUR] État inconnu, modification manuelle requise.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Patch appliqué.")


if __name__ == "__main__":
    main()
