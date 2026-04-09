"""
apply_paddlex_patch_parallel.py — Parallélise les appels VLM intra-page.

Prérequis : apply_paddlex_patch.py doit avoir été appliqué en premier (patch OTSL).

Principe :
    PaddleOCR traite les blocs d'une page séquentiellement (un appel HTTP par bloc).
    Ce patch remplace la boucle séquentielle par un ThreadPoolExecutor, permettant
    à N blocs d'être soumis simultanément à llama-server.

    L'architecture de PaddleX utilise asyncio.run_coroutine_threadsafe() sur un event
    loop global unique (background thread). Les appels depuis plusieurs threads sont
    thread-safe par conception. llama-server doit être lancé avec -np N pour ouvrir
    N slots GPU correspondant à VLM_PARALLEL.

    VLM_PARALLEL = 2 ici, ce qui correspond à -np 2 dans llama-server (config.py).

Usage :
    python docs/dev/apply_paddlex_patch_parallel.py           # applique le patch
    python docs/dev/apply_paddlex_patch_parallel.py --check   # vérifie sans modifier
    python docs/dev/apply_paddlex_patch_parallel.py --revert  # restaure le patch OTSL
"""

import argparse
import sys
from pathlib import Path

TARGET = (
    Path(sys.prefix)
    / "Lib/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py"
)

# État attendu avant ce patch : résultat de apply_paddlex_patch.py
ORIGINAL = """\
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

# Remplacement : boucle parallèle via ThreadPoolExecutor
# VLM_PARALLEL doit correspondre à -np dans llama-server (config.py : n_parallel)
PATCHED = """\
            images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]

            _VLM_PARALLEL = 2  # doit correspondre à -np dans llama-server

            def _infer_block(args):
                _img, _qry = args
                try:
                    return list(
                        self.vl_rec_model.predict(
                            [{"image": _img, "query": _qry}],
                            skip_special_tokens=False if has_spotting else True,
                            **kwargs,
                        )
                    )[0]
                except Exception as _vlm_err:
                    _err_msg = str(_vlm_err)
                    _otsl = _err_msg.find("<fcel>")
                    if _otsl != -1:
                        # OTSL content echoed back by llama-server; convert directly
                        return {"result": _err_msg[_otsl:]}
                    return {"result": ""}

            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=_VLM_PARALLEL) as _pool:
                batch_results = list(_pool.map(_infer_block, zip(images, queries)))
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
    parser.add_argument("--revert", action="store_true", help="Restaure le patch OTSL (retire le parallélisme).")
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
            print("Déjà à l'état original (patch OTSL), rien à faire.")
            return
        if state != "patched":
            print("[ERREUR] État inconnu, modification manuelle requise.")
            sys.exit(1)
        TARGET.write_text(text.replace(PATCHED, ORIGINAL), encoding="utf-8")
        print("Patch parallèle retiré — retour au patch OTSL seul.")
        return

    # Appliquer le patch
    if state == "patched":
        print("Déjà patché, rien à faire.")
        return
    if state != "original":
        print("[ERREUR] État inconnu. Vérifier que apply_paddlex_patch.py a été appliqué en premier.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Patch parallèle appliqué.")
    print("N'oublie pas de lancer llama-server avec -np 2.")


if __name__ == "__main__":
    main()
