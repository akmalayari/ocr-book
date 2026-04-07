# Issues — travaux en cours

## Version robuste (actuelle)

### Bug 4 — Robustesse du préprocessing et précision texte

**Résultats des tests prétraitements légers (2026-04-07, `draft/test_preprocess.py`, pages 4/5/6/9) :**
- `bilateral` : boucle sur page_6 (nette) et page_4 — à abandonner.
- `nlmeans` : dégrade légèrement le texte sur images floues, boucle sur page_4 — pas d'avantage vs `none`.
- `esrgan` : 137s/image, gain marginal — trop lent pour la pipeline.
- `sesr` : +0.7% global vs `none` sur page_5 (floue), ~7s, pas de boucle sur pages 5/6/9, boucle sur page_4.
- `none` : seule config sans boucle sur toutes les pages testées.
- Aucune config ne détecte la figure sur page_6 avec preprocess léger (régression vs `blur_adaptive`).

Voir `output/rapports/preprocess_legers_analyse.md`.

**Pivot architectural (2026-04-07) :**
Tables et figures traitées comme des crops — référencées par chemin image, pas retranscrites.
Le score de précision est désormais calculé sur le **texte pur uniquement** (labels `text`, `title`, `sub_title`, `table_caption`, `table_footnote`, `image_caption`).
Objectif : >99% texte sur image clean, dégradation minimale sur image bruitée.

**Images clean disponibles : page_4, page_5, page_6, page_9, page_10** (nettes, éclairage uniforme).

**Configs retenues pour la suite : `none`, `sesr`, `nlmeans`**

Résultats texte pur (2026-04-07, `compare_ocr.py` avec score texte seul, `output/compare_preprocess/global_report.md`) :

| Config | p5 texte % | p6 texte % |
|--------|:----------:|:----------:|
| none | 92.1% | 92.1% |
| sesr | 92.0% | 92.6% |
| nlmeans | 92.1% | 95.4% |
| bilateral | 91.8% | 1.7% *(boucle)* |
| esrgan | 92.2% | 95.2% |

`nlmeans` meilleur texte sur p6 (95.4%), quasi égal sur p5. `esrgan` proche mais 137s — éliminé. `bilateral` abandonné (boucle p6). Sur p5, tous les preprocess sont à ±0.2% — le bruit est le facteur limitant, pas le preprocess.

**Prochaines étapes :**

1. **Baseline texte pur sur images clean** — tester `none`, `sesr`, `nlmeans` sur les images clean (page_4, page_5, page_6, page_9, page_10). Valider l'hypothèse >99% texte sur image propre.
2. **Tester sur originaux bruités** — mesurer la dégradation vs baseline clean pour les 3 configs.
3. **bg_divide** (sans binarisation) — pour les images à éclairage inégal (pliure, lumière rasante). À tester sur page_1 et les originaux bruités si dégradation significative.
4. **Déconvolution de flou** (`skimage.restoration.unsupervised_wiener`) — pour les très floues (Laplacian < 40). Après bg_divide.
5. **Loop recovery via `rec`** — dernier recours si boucles persistantes sur images clean.

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
