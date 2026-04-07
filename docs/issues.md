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

**Config retenue : `nlmeans` (défaut pipeline). `sesr` disponible en option.**

Résultats texte pur avec normalisation (2026-04-07, `compare_ocr.py`, référence `page_6_text.md`, rapport `output/rapports/global_report_5-6.md`) :

| Config | p5 texte % | p6 texte % | p56c texte % | boucles |
|--------|:----------:|:----------:|:------------:|---------|
| none | 98.8% | 98.7% | 97.9% | p4_clean |
| nlmeans | 98.9% | 98.8% | 99.3% | p4 bruité |
| sesr | 98.7% | 99.2% | 98.8% | p4 bruité + p4_clean (symboles) |

Hypothèse >99% validée. `nlmeans` retenu pour robustesse (zéro boucle sur images clean). Erreurs résiduelles (~1%) : mots hors-dictionnaire proches d'un mot existant.

**Prochaines étapes :**

1. **Run complet sur toutes les photos disponibles** — `none`, `sesr`, `nlmeans` pour comparer sur l'ensemble du corpus.
2. **bg_divide** (sans binarisation) — pour les images à éclairage inégal (pliure, lumière rasante).
3. **Déconvolution de flou** (`skimage.restoration.unsupervised_wiener`) — pour les très floues (Laplacian < 40).
4. **Loop recovery via `rec`** — dernier recours si boucles persistantes sur images difficiles.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages
Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.


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
