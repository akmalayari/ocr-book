# Issues — travaux en cours

## Version robuste (actuelle)

### Bug 4 — Robustesse du préprocessing sur images bruitées
État actuel : précision page_5/page_6 évaluée (2026-04-06). Voir `output/rapports/preprocess_p5_p6.md`.

- **page_4** : boucle persistante sur toutes les configs. Aucune config ne transcrit le tableau sans boucler.
- **page_9** : `median_and` et `nlm5_and` ne bouclent pas mais précision subpar. `nlm5_median`/`nlm10_and` bouclent sur suite de chiffres (faux positif détection).
- **page_5/6** : `none` meilleur texte sur image floue (94.9%) ; `blur_adaptive` meilleur équilibre sur image nette (96.3% texte, 94.9% figure). Figure intraitable sur image floue (< 40% toutes configs). Precision toujours insatisfaisante.

**Prochaines étapes (prétraitements non destructifs) :**

Hypothèse directrice : DeepSeek-OCR est optimisé pour des photos naturelles. La binarisation transforme trop radicalement la distribution d'entrée → boucles. Les filtres légers qui préservent le look photo sont préférables.

Scripts : `draft/test_preprocess.py` (OCR), `draft/realesrgan_sesr.py` (génération SR).

1. **nlmeans seul** — `fastNlMeansDenoising(h=noise_level)`, dans `preprocess.py` + `test_preprocess.py`. À tester sur pages 4, 5, 6, 9.
2. **bilateralFilter** — `bilateralFilter(d=9, σ=75)`, dans `preprocess.py` + `test_preprocess.py`.
3. **SESR-M7 x2** — `sesr-m7-256x256-tiles-amdnpu` (2026-04-07), ~91 FPS NPU. Intégré dans `realesrgan_sesr.py` + `test_preprocess.py`. À tester sur pages 5, 6.
4. **RealESRGAN x4** — intégré dans `realesrgan_sesr.py` + `test_preprocess.py`. Comparaison avec SESR-M7.
5. **bg_divide seul** (sans binarisation) — uniquement pour page_1 (mauvais éclairage). À ajouter si nlmeans/SR insuffisants.
6. **Déconvolution de flou** — si SR insuffisant sur figure floue. Wiener filter ou déconvolution aveugle (`skimage.restoration`).
7. **Changer de VLM** (si tout insuffisant) — voir Amélioration 2.


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

### Amélioration 2 - Performance et précision
Tester d'autres modèles OCR : paddleocr, lightonocr, olmocr, glmocr, nanonetsocr. Critères : 1. vitesse d'exécution 2. précision de retranscription.

**Stratégie envisagée : modèle adaptatif selon la balise grounding.** Utiliser un VLM généraliste (ex. DeepSeek-OCR) pour le texte courant, et basculer sur un modèle spécialisé selon la balise détectée — modèle tableau pour `table`, modèle figure pour `image`. Permettrait de contourner les boucles sur tableaux numériques (page_4) sans dégrader le texte.

## Architecture

OK

## Style

OK
