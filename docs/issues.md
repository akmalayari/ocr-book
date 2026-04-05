# Issues — travaux en cours

## Version robuste (actuelle)

### Bug 4 — Robustesse du préprocessing sur images bruitées
État actuel : mode `sauvola` (AND Sauvola w=51 k=0.3 + binarize adaptive) ~93 % de précision estimée sur page bruitée, vs ~90 % pour baseline. Qualité insuffisante. Boucles persistantes sur pages 4, 9 avec les deux modes.

**Pistes à tester (par ordre de priorité) :**
1. MedianBlur(3) → gaussianblur → binarise avec un C plus petit pour nettoyer le bruit en amont et garder les formes des lettres intactes.
2. **fastNlMeansDenoising avant binarisation** — débruitage non-local, préserve mieux les bords. Lent (~×10 vs Gaussian).
3. **Gris sans binarisation + denoising fort** — le modèle reçoit peut-être plus d'information depuis une image grise bien débruitée.
4. **Real-ESRGAN** — super-résolution x2/x4, récupère des caractères illisibles. ~2–5s/image sur GPU. Dépendance lourde.


---

## Prochaines versions

### Feature 2 — Support PDF multi-pages
Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

### Bug 3 — Courbure de page
Courbure due à la reliure — déforme les lignes de texte géométriquement. Provoque des erreurs sur les mots coupés en fin de ligne.

**Pistes à tester (par ordre de priorité) :**
1. **Lignes de texte + fit polynomial + `cv2.remap`** — pure OpenCV, aucune dépendance. À appliquer sur chaque moitié de l'image séparément (double page). Approche retenue en priorité.
2. **DocTr** — transformer léger de déwarpage de documents, tourne sur CPU. Dépendance PyTorch/ONNX (~200 MB).
3. **DewarpNet / DocUNet** — modèle neural, résultats solides sur courbures prononcées. GPU requis, dépendance lourde. Dernier recours.

### Amélioration 1 — Performance BF16 (~50s/image)
BF16 est 2.5× plus lent que Q8_0 (~20s). Temps couteux sur des centaines de pages.

**Pistes à tester (par ordre de priorité) :**
1. **Réduire la résolution des images** — redimensionner à ~1200px de large avant VLM. Hypothèse : gain 30–50%.
2. **Changer de wrapper Python (llama-cpp-python)** — secondaire, peu probable d'accélérer.

## Architecture

OK

## Style

OK
