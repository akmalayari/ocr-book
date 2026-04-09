# Issues — travaux en cours

## Version paddle (actuelle)

### Validation du patch parallèle sur images de test

Le patch `apply_paddlex_patch_parallel.py` (pool global, np=3) doit être testé sur l'ensemble des images de référence (page_1 à page_9) pour vérifier l'absence de crash et la qualité de l'output.

### Optimisations vitesse à tester

**Réduire `-c` (facile)** : la plupart des blocs font 150-1300 tokens, `-c 12288` / 3 slots = 4096/slot est surdimensionné. Tester `-c 6144` (2048/slot). Moins de KV cache → prefill potentiellement plus rapide. Risque : blocs très longs (grands tableaux) pourraient être tronqués.

**Sémaphore vision encoder (invasif)** : le crash à -np 4 vient du vision encoder Vulkan saturé par 4 encodages simultanés. Un `threading.Semaphore(2)` dans `_infer_block` du patch limiterait les encodages simultanés à 2, tout en gardant 4 slots de génération actifs. Permettrait de passer -np 4 sans crash. Gain estimé ~5s supplémentaires.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
