# PaddleOCR-VL — Performance et optimisations

## Mesures de référence

Hardware : Windows 11, GPU AMD Radeon 890M (Vulkan, iGPU), llama-server Vulkan backend.
Modèle : PaddleOCR-VL-1.5-0.9B, GGUF BF16, 890 MiB. VRAM totale : ~23 GiB.
Images : 4080×3072 px (12.5 MP), photos de pages de livre.

| Config | Vitesse/page | Notes |
|---|---|---|
| -np 1, séquentiel (baseline) | ~60s | état initial |
| -np 2, 2 workers (pool global) | ~49s | -11s |
| -np 3, 3 workers (pool global) | ~46s | -14s, **retenu** |
| -np 4, 4 workers | crash | vision encoder Vulkan saturé |
| -np 6, 6 workers | hang | contention GPU totale |

Le temps par page est proportionnel au nombre de blocs détectés par PP-DocLayoutV3 (chaque bloc = un appel VLM séparé). Gain réel sur 150 pages : ~35 min.

## Goulot d'étranglement

Deux goulots distincts :
1. **Vision encoder** (encodage image en base64 → tokens) : ~350-4500 ms/bloc selon la taille. Saturé à partir de 4 encodages simultanés sous Vulkan → crash.
2. **Génération LLM** : ~36 tok/s par slot, mais plusieurs slots peuvent générer en parallèle si le vision encoder n'est pas saturé.

## Optimisations testées

### Parallélisation intra-page (pool global) — **retenu**

PaddleOCR traite les blocs d'une page séquentiellement. Un patch (`docs/dev/apply_paddlex_patch_parallel.py`) remplace la boucle par un `ThreadPoolExecutor` global qui soumet tous les blocs de toutes les `pixel_key` simultanément.

**Pourquoi "pool global" plutôt que "pool par pixel_key"** : la version initiale (pool par pixel_key) recréait un pool à chaque groupe de blocs — pas de chevauchement entre groupes. Le pool global collecte tous les blocs en une seule liste, les workers pickent en continu, les résultats sont redistribués par pixel_key après.

Limite : le vision encoder Vulkan crashe à partir de 4 encodages simultanés. Plancher à 3 workers.

### `-np N` seul (sans parallélisation intra-page)
**Résultat : contre-productif. Abandonné.**

Testé avec `n_parallel=2` sur 2 pages entières simultanées : 213s vs 110s. Les requêtes se battent pour le GPU entier. Contexte divisé entre slots → chaque slot n'a que 2048 tokens → blocs longs tronqués (HTTP 400).

### Resize PIL avant predict (`--max-image-size 1500`)
**Résultat : qualité fortement dégradée. Abandonné.**

Testé sur images 4080×3072 → 1500×1129. Légèrement plus rapide mais OCR fortement dégradé.
Cause : le resize s'appliquait **avant** la layout detection qui recevait une image dégradée, compromettant la détection des blocs.

### Paramètre `max_pixels`
**Non applicable pour `llama-cpp-server`. Ignoré.**

`max_pixels` contrôle le nombre de pixels envoyés au vision encoder. Défaut : `28×28×3600 = 2 822 400` px. Supporté uniquement par le backend `vllm-server`. Pour `llama-cpp-server`, un `warnings.warn` est émis et le paramètre est ignoré. L'image est envoyée telle quelle en base64 à llama-server.

### `--flash-attn` (llama-server)
**Non testé** — potentiellement non supporté par Vulkan.

### Augmenter `n_ubatch`
**Testé (512 → 1024), aucun gain.**

## Paramètres llama-server retenus

```
-c 6144      # context window (2048 tokens/slot × 3 slots)
-np 3         # slots parallèles (cohérent avec VLM_PARALLEL=3 dans le patch)
-ngl 99       # toutes les couches sur GPU
-b 512        # batch size
-ub 512       # ubatch size (testé 1024 : 0 gain)
-t 4          # threads CPU
--prio 2      # priorité process
--temp 0.0    # déterministe
-kvo          # KV cache offload
```

KV cache avec np=3 : ~297 MiB × 3 / 2 ≈ 445 MiB. VRAM totale occupée : ~4.5 GiB sur 23 GiB.

## Pistes non explorées

- **Backend vllm-server** à la place de llama-cpp-server : supporterait `max_pixels` et potentiellement plus performant, mais nécessite un setup différent (Linux-friendly, pas testé sous Windows/Vulkan)
