"""
apply_paddlex_patch_streaming.py — Débloque -np 4 via streaming HTTP.

Prérequis : apply_paddlex_patch_parallel.py doit avoir été appliqué en premier.

Principe :
    Le crash à -np 4 vient de 4 encodages vision simultanés sur Vulkan.
    Ce patch remplace l'appel HTTP bloquant par du streaming SSE (stream=True).
    Le sémaphore asyncio est libéré dès le premier token reçu (= fin du prefill).
    Résultat : max 2 prefills simultanés, jusqu'à 4 slots en génération.

    Modifie deux fichiers :
      1. paddlex/inference/models/common/genai.py
           GenAIClient.__init__  : sémaphore réduit à 2 pour llama-cpp-server
           create_chat_completion : streaming + release après premier token
      2. paddlex/inference/pipelines/paddleocr_vl/pipeline.py
           _VLM_PARALLEL : 3 → 4

Usage :
    python docs/dev/apply_paddlex_patch_streaming.py           # applique
    python docs/dev/apply_paddlex_patch_streaming.py --check   # vérifie
    python docs/dev/apply_paddlex_patch_streaming.py --revert  # retire (retour parallel)
"""

import argparse
import sys
from pathlib import Path

GENAI = (
    Path(sys.prefix)
    / "Lib/site-packages/paddlex/inference/models/common/genai.py"
)

PIPELINE = (
    Path(sys.prefix)
    / "Lib/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py"
)

# ── genai.py — sémaphore ──────────────────────────────────────────────────────

GENAI_SEM_ORIGINAL = """\
        self._semaphore = asyncio.Semaphore(self._max_concurrency)"""

GENAI_SEM_PATCHED = """\
        # Limit concurrent prefills for llama-cpp-server (vision encoder Vulkan crash).
        # 3 simultaneous prefills are safe (tested); 4 crash. The streaming patch releases
        # this semaphore after the first token (prefill done), so decode phases overlap
        # with the next prefill — enabling 4 active slots with only 3 simultaneous prefills.
        _prefill_limit = 3 if backend == "llama-cpp-server" else self._max_concurrency
        self._semaphore = asyncio.Semaphore(_prefill_limit)"""

# ── genai.py — create_chat_completion ────────────────────────────────────────

GENAI_METHOD_ORIGINAL = """\
    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        async def _create_chat_completion_with_semaphore(*args, **kwargs):
            async with self._semaphore:
                return await self._client.chat.completions.create(
                    *args,
                    **kwargs,
                )

        return run_async(
            _create_chat_completion_with_semaphore(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ),
            return_future=return_future,
        )"""

GENAI_METHOD_PATCHED = """\
    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        if self.backend == "llama-cpp-server":
            async def _stream_with_prefill_sem(*args, **kwargs):
                from types import SimpleNamespace
                async with self._semaphore:
                    _stream = await self._client.chat.completions.create(
                        *args, stream=True, **kwargs
                    )
                    # Skip role/empty chunks until first real content token.
                    # The role chunk {"delta": {"role": "assistant", "content": ""}} may
                    # arrive before prefill completes — releasing the semaphore on it
                    # would let all workers bypass the prefill guard. We wait for a
                    # non-empty content token, which is emitted only after prefill.
                    _content = ""
                    async for _chunk in _stream:
                        _piece = _chunk.choices[0].delta.content if _chunk.choices else None
                        if _piece:
                            _content = _piece
                            break
                # semaphore released — prefill confirmed done, decode continues
                async for _chunk in _stream:
                    if _chunk.choices and _chunk.choices[0].delta.content:
                        _content += _chunk.choices[0].delta.content
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=_content))]
                )

            return run_async(
                _stream_with_prefill_sem(
                    model=self._model_name,
                    messages=messages,
                    **kwargs,
                ),
                return_future=return_future,
            )

        async def _create_chat_completion_with_semaphore(*args, **kwargs):
            async with self._semaphore:
                return await self._client.chat.completions.create(
                    *args,
                    **kwargs,
                )

        return run_async(
            _create_chat_completion_with_semaphore(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ),
            return_future=return_future,
        )"""

# ── pipeline.py — _VLM_PARALLEL ──────────────────────────────────────────────

PIPELINE_ORIGINAL = """\
        _VLM_PARALLEL = 3  # doit correspondre à -np dans llama-server"""

PIPELINE_PATCHED = """\
        _VLM_PARALLEL = 4  # doit correspondre à -np dans llama-server"""


# ── helpers ───────────────────────────────────────────────────────────────────

def status_genai(text: str) -> str:
    if GENAI_SEM_PATCHED in text and GENAI_METHOD_PATCHED in text:
        return "patched"
    if GENAI_SEM_ORIGINAL in text and GENAI_METHOD_ORIGINAL in text:
        return "original"
    return "unknown"


def status_pipeline(text: str) -> str:
    if PIPELINE_PATCHED in text:
        return "patched"
    if PIPELINE_ORIGINAL in text:
        return "original"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    for path in (GENAI, PIPELINE):
        if not path.exists():
            print(f"[ERREUR] Fichier introuvable : {path}")
            sys.exit(1)

    genai_text    = GENAI.read_text(encoding="utf-8")
    pipeline_text = PIPELINE.read_text(encoding="utf-8")

    sg = status_genai(genai_text)
    sp = status_pipeline(pipeline_text)

    print(f"genai.py    : {sg}")
    print(f"pipeline.py : {sp}")

    if args.check:
        ok = sg == "patched" and sp == "patched"
        sys.exit(0 if ok else 1)

    if args.revert:
        if sg == "original" and sp == "original":
            print("Déjà à l'état original (parallel seul), rien à faire.")
            return
        if sg not in ("patched", "original") or sp not in ("patched", "original"):
            print("[ERREUR] État inconnu, modification manuelle requise.")
            sys.exit(1)
        if sg == "patched":
            genai_text = (
                genai_text
                .replace(GENAI_SEM_PATCHED, GENAI_SEM_ORIGINAL)
                .replace(GENAI_METHOD_PATCHED, GENAI_METHOD_ORIGINAL)
            )
            GENAI.write_text(genai_text, encoding="utf-8")
            print("genai.py    : reverted")
        if sp == "patched":
            PIPELINE.write_text(
                pipeline_text.replace(PIPELINE_PATCHED, PIPELINE_ORIGINAL),
                encoding="utf-8",
            )
            print("pipeline.py : reverted (_VLM_PARALLEL = 3)")
        return

    # Apply
    errors = []
    if sg == "unknown":
        errors.append("genai.py : état inconnu — vérifie que le fichier n'a pas été modifié manuellement")
    if sp == "unknown":
        errors.append("pipeline.py : état inconnu — vérifie que apply_paddlex_patch_parallel.py a été appliqué")
    if errors:
        for e in errors:
            print(f"[ERREUR] {e}")
        sys.exit(1)

    if sg != "patched":
        genai_text = (
            genai_text
            .replace(GENAI_SEM_ORIGINAL, GENAI_SEM_PATCHED)
            .replace(GENAI_METHOD_ORIGINAL, GENAI_METHOD_PATCHED)
        )
        GENAI.write_text(genai_text, encoding="utf-8")
        print("genai.py    : patched (streaming + prefill semaphore=3)")
    else:
        print("genai.py    : déjà patché")

    if sp != "patched":
        PIPELINE.write_text(
            pipeline_text.replace(PIPELINE_ORIGINAL, PIPELINE_PATCHED),
            encoding="utf-8",
        )
        print("pipeline.py : patched (_VLM_PARALLEL = 4)")
    else:
        print("pipeline.py : déjà patché")

    print("\nAssure-toi que llama-server tourne avec -np 4 -c 8192.")


if __name__ == "__main__":
    main()
