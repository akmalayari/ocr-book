# Issues — travaux en cours

## Version multimodale (actuelle)
OK

---

## Prochaines versions

### Bug 4 — Robustesse du préprocessing (bruit, flou, éclairage inégal)
Objectif : rendre l'OCR plus tolérant aux images imparfaites. Urgent: le modele boucle dès que la qualité n'est pas parfaite, meme en bf16. Cela rend le modèle concrètement inutilisable.

**Hypothèse principale :** le bouclage sur images floues est en partie dû aux paramètres `block_size`/`C` inadaptés — sur flou, la moyenne locale sur 31×31px converge vers les valeurs de pixels et `C=10` efface des traits mous.

**Pistes à tester (par ordre de priorité) :**
1. **Grid test block_size / C** — tester block_size ∈ {11,21,31,41,51} × C ∈ {5,10,15,20} sur page_5 (floue) et page_6 (nette). Script : `draft/test_binarize_grid.py`. Phase 1 visuelle, phase 2 OCR sur configs retenues.
2. **Détection de flou par variance Laplacien + paramètres adaptatifs** — `cv2.Laplacian(gray, CV_64F).var()` rapide (~1ms). En dessous d'un seuil → appliquer des paramètres `block_size`/`C` différents (plus petits). Seuil à calibrer via `--pages page_5 page_6` du script ci-dessus.
3. **preprocess_mode="none" sur images floues** — le preprocess actuel avait été mis en place pour réduire les boucles, mais sur flou prononcé l'image originale peut être plus lisible qu'une binarisation ratée.
4. **bg_divide + binarize adaptive** — normalise l'illumination en divisant par un fond estimé (GaussianBlur 101×101). Résultat visuel prometteur (déjà dans `draft/viz_preprocess2.py`). Non encore testé en OCR.
5. **Opérations morphologiques après binarisation** — `opening` supprime les artefacts de bruit, `closing` rebouche les trous dans les lettres.
6. **fastNlMeansDenoising avant binarisation** — débruitage non-local, préserve mieux les bords. Lent (~×10 vs Gaussian).
7. **Gris sans binarisation + denoising fort** — le modèle reçoit peut-être plus d'information depuis une image grise bien débruitée.
8. **Sauvola binarization** (`scikit-image`) — conçue pour les documents dégradés, tient compte de la variance locale.
9. **Real-ESRGAN** — super-résolution x2/x4, récupère des caractères illisibles. ~2–5s/image sur GPU. Dépendance lourde.


### Feature 2 — Support PDF multi-pages
Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

### Bug 3 — Courbure de page
Courbure due à la reliure — déforme les lignes de texte géométriquement. Provoque des erreurs sur les mots coupés en fin de ligne.

**Pistes à tester (par ordre de priorité) :**
1. **Lignes de texte + fit polynomial + `cv2.remap`** — pure OpenCV, aucune dépendance. À appliquer sur chaque moitié de l'image séparément (double page). Approche retenue en priorité.
2. **DocTr** — transformer léger de déwarpage de documents, tourne sur CPU. Dépendance PyTorch/ONNX (~200 MB).
3. **DewarpNet / DocUNet** — modèle neural, résultats solides sur courbures prononcées. GPU requis, dépendance lourde. Dernier recours.

### Amélioration 1 — Performance BF16 (~50s/image)
BF16 est 2.5× plus lent que Q8_0 (~20s). Coût purement inference.

**Pistes à tester (par ordre de priorité) :**
1. **Réduire la résolution des images** — redimensionner à ~1200px de large avant VLM. Hypothèse : gain 30–50%.
2. **Profiler le vrai bottleneck** — chronométrer séparément l'encodage visuel et le décodage texte.
3. **Changer de wrapper Python (llama-cpp-python)** — secondaire, peu probable d'accélérer.

## Architecture

OK

## Style

OK
