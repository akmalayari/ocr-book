# Issues — travaux en cours

## Version paddle (actuelle)

### Optimisations vitesse à tester

**Sémaphore vision encoder (invasif)** : le crash à -np 4 vient du vision encoder Vulkan saturé par 4 encodages simultanés. Un `threading.Semaphore(2)` dans `_infer_block` du patch limiterait les encodages simultanés à 2, tout en gardant 4 slots de génération actifs. Permettrait de passer -np 4 sans crash. Gain estimé ~5s supplémentaires.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
