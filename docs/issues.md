# Issues — travaux en cours

## Version robuste (actuelle)

### Bug 4 — Robustesse du préprocessing sur images bruitées
État actuel : phase OCR sur 5 configs × pages 4/5/6/9 terminée (2026-04-06).

- **page_4** : boucle persistante sur toutes les configs. Aucune config ne transcrit le tableau sans boucler.
- **page_9** : `median_and` et `nlm5_and` ne bouclent pas mais précision subpar. `nlm5_median`/`nlm10_and` bouclent sur suite de chiffres (faux positif détection).
- **page_5/6** : `nlm5_and` échoue sur page_5 (4 mots). Les autres configs sont sans boucle — précision à évaluer.

**Prochaines étapes :**
1. **Évaluer la précision page_5 et page_6** — comparer les sorties `output/nlmeans/page_5_*.md` et `page_6_*.md` contre `photos/md/page_6.md` (référence 100%) via `--compare`. Détermine quelle config préserve le mieux le texte sur image bruitée vs nette.
2. **estimate_noise_level pour fine-tuning de h** — utiliser `estimate_sigma` (skimage) ou Laplacien pour adapter dynamiquement `nlmeans_h` selon le niveau de bruit de chaque image. Voir `draft/report_laplacian.py::estimate_noise_level()`.
3. **Real-ESRGAN** — super-résolution x2/x4, récupère des caractères illisibles. ~2–5s/image sur GPU. Dépendance lourde. Dernier recours.


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
Tester d'autres modèles OCR via llama-cpp-python: paddleocr, lightonocr, olmocr, glmocr, nanonetsocr. Il faut tester 1. la vitesse d'execution 2. la precision de la retranscription. 

## Architecture

OK

## Style

OK
