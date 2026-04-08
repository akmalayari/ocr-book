# Issues — travaux en cours

## Version paddle (actuelle)

### Intégration PaddleOCR dans la pipeline principale

**Points à traiter :**
1. **Gestion des crops figures** — PaddleOCR sauvegarde les crops dans `imgs/` relatif au `save_path`. À intégrer dans la pipeline de sortie.

### Amélioration 3 — Tuning llama-server

Pistes pour réduire le temps de traitement (actuellement 35–40s/image) :
- `--n-gpu-layers` : décharger des couches sur GPU via Vulkan
- `--ctx-size` : réduire si 4096 tokens non nécessaires pour les pages courtes
- `--batch-size` : ajuster pour le throughput

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
