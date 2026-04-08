# PaddleOCR-VL — Performance et optimisations

## Mesures de référence

Hardware : Windows 11, GPU non-NVIDIA (Vulkan), llama-server Vulkan backend.
Modèle : PaddleOCR-VL-1.5-0.9B, GGUF F16.
Images : 4080×3072 px (12.5 MP), photos de pages de livre.

| Condition | Vitesse |
|---|---|
| Page texte simple | ~47s |
| Page avec tableau + graphique | ~55s |
| Vitesse de génération estimée | ~36 tok/s |

Le temps par page est proportionnel au nombre de blocs détectés par PP-DocLayoutV3 (chaque bloc = un appel VLM séparé).

## Goulot d'étranglement

**Vitesse de génération brute du LLM.** ~2000 tokens par page à 36 tok/s ≈ 55s. C'est la limite du hardware sous Vulkan — difficile à dépasser sans changer de GPU ou de backend.

## Optimisations testées

### `n_parallel` (llama-server `-np N`)
**Résultat : contre-productif. Abandonné.**

Testé avec `n_parallel=2` : temps total passé de 110s à 213s pour 2 pages. Les requêtes parallèles se battent pour les ressources GPU Vulkan. Aucun gain de débit, latence fortement augmentée.

Vulkan n'est pas optimisé pour le parallélisme multi-slots contrairement à CUDA.

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
-c 4096       # context window
-ngl 99       # toutes les couches sur GPU
-b 512        # batch size
-ub 512       # ubatch size (testé 1024 : 0 gain)
-t 4          # threads CPU
--prio 2      # priorité process
--temp 0.0    # déterministe
-kvo          # KV cache offload
```

## Pistes non explorées

- **Backend vllm-server** à la place de llama-cpp-server : supporterait `max_pixels` et potentiellement plus performant, mais nécessite un setup différent (Linux-friendly, pas testé sous Windows/Vulkan)
- **Modèle quantifié plus agressif** (Q4, Q5) : gain de vitesse au prix de la précision
- **GPU NVIDIA + CUDA** : backend CUDA de llama-server, flash-attn natif, parallélisme réel
