# Tested — ce qui a été expérimenté dans le projet

Pages de référence utilisées pour les tests : `page_1` à `page_9`.
- page_1 : texte simple, mauvais éclairage
- page_2 : texte dense, éclairage normal
- page_3 : texte + tableau textuel
- page_4 : texte + tableau numérique
- page_5 : texte + tableau + graphique, image floue
- page_6 : même contenu que page_5, image nette (version de référence pour tests graphe)
- page_7 : début de chapitre, texte seul, image nette
- page_8 : texte seul, image nette
- page_9 : fin de chapitre, texte seul, image nette

---

## Résumé des explorations

| Sujet | Résultat |
|---|---|
| **Stack DeepSeek-OCR + nexaai** | Abandonné — migration vers PaddleOCR VL 1.5 |
| **Prompts DeepSeek-OCR** (layout, plain, rec…) | Abandonné avec le modèle |
| **Prétraitement images** (binarize, sauvola, nlmeans, sesr…) | Abandonné — PaddleOCR fonctionne sur image brute |
| **Parallélisation inter-pages** (`n_servers > 1`) | Abandonné sur APU — sérialisation GPU Vulkan |
| **Streaming HTTP pour -np 4** | Abandonné — texte incomplet, aucun gain |
| **Patch OTSL** (`apply_paddlex_patch_otsl.py`) | Retenu — erreurs VLM par région sur tableaux complexes |
| **Patch parallélisme intra-page** (`apply_paddlex_patch_parallel.py`, -np 3) | Retenu — gain ~30% (60s → 43s/page) |
| **PaddleOCR VL 1.5 + llama-server** | Retenu — stack actuel |
| **`page_timeout` + fallback no-layout** | Retenu — protection contre boucles llama-server |

---

## DeepSeek-OCR (abandonné — migration vers PaddleOCR VL 1.5)

> Les statuts "retenu" dans cette section sont relatifs à l'ère DeepSeek-OCR uniquement.
> Ce modèle a été entièrement remplacé par PaddleOCR VL 1.5 le 2026-04-08.

### Stack d'inférence

#### REST API `nexa serve`
**Statut : abandonné.**
Retournait systématiquement HTTP 500 sur les requêtes multimodales sous Windows. Le serveur REST de Nexa ne sait pas gérer le format GGUF + mmproj de DeepSeek-OCR.

#### `nexaai.VLM` Python direct
**Statut : retenu (ère DeepSeek-OCR).**
Charge les deux fichiers du modèle (GGUF + mmproj) via `nexa_bridge.dll`. Contourne le serveur REST entièrement. Interface : `VlmChatMessage` + `apply_chat_template` + `vlm.generate(GenerationConfig(image_paths=[...]))`.

Nexaai utilise llama-cpp sous le capot — DeepSeek-OCR peut tourner sur GPU via Vulkan.

**Limite :** sur Windows, `nexa_bridge.dll` retourne un `stop_reason` corrompu (byte `0xc0` invalide UTF-8) dans les métadonnées de profiling. Contourné par un monkey-patch dans `src/patch.py` — le texte OCR lui-même est intact.

---

### Quantization du modèle

| Quantization | Vitesse | Qualité |
|---|---|---|
| Q8_0 | ~20s/page | boucle immédiatement sur pages difficiles |
| BF16 | ~50s/page | plus fidèle que F16 sur les passages difficiles |
| F16  | ~50s/page | moins fidèle que BF16 — hallucinations et reformulations sur passages difficiles |

**Statut : BF16 retenu (ère DeepSeek-OCR).** Comparaison BF16 vs F16 (2026-04-02, `compare.py` mode sentence, page_1) : BF16 retranscrit fidèlement plusieurs passages là où F16 hallucine ou reformule. BF16 reste imparfait (voir Bug #1 et #5).

---

### Prompts

Testés sur pages 1–6 avec `preprocess=binarize` (`draft/test_prompts.py`).

| Mode | Prompt | Résultat |
|---|---|---|
| `plain` | `"Free OCR."` | texte brut propre sur pages simples, boucle sur pages denses |
| `layout` | `"<\|grounding\|>Convert the document to markdown."` | ajoute grounding boxes, règle la boucle de page_2 mais boucle sur les tableaux (`<tr>`/`<td>`) |
| `describe` | `"Describe this image in detail."` | description en anglais, indépendamment de la langue du document |
| `parse` | `"Parse the figure."` | analyse fine des éléments visuels en anglais |
| `rec` | `"Locate <\|ref\|>{target}<\|/ref\|> in the image."` | retourne la/les bbox correspondant au target. Voir section dédiée ci-dessous. |
| `classic` | (supprimé) | une grounding box par phrase, dépasse systématiquement `n_ctx` |

**Statut :** `layout` retenu (ère DeepSeek-OCR). Précision légèrement supérieure à `plain`. Permet de récupérer les bbox pour le traitement des images dans la deuxième passe.

**`repetition_penalty`** testé — apparemment sans effet sur les boucles de génération.

#### Comportement du mode `rec` (2026-04-03, page_6, BF16, binarize)

`Locate <|ref|>{target}<|/ref|> in the image.`

| Prompt | Résultat |
|---|---|
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image.` | 1 bbox exacte pour le graphique, s'arrête proprement (5.8s) |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Describe it.` | identique — instruction post-bbox ignorée |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Parse it.` | identique — instruction post-bbox ignorée |
| `Locate <\|ref\|>every element<\|/ref\|> in the image.` | toutes les bboxes de la page (même résultat que `layout`) |
| `Locate <\|ref\|>everything<\|/ref\|> in the image.` | toutes les bboxes de la page |
| `Locate <\|ref\|>pliure du livre<\|/ref\|> in the image.` | toutes les bboxes de la page |
| `Locate <\|ref\|>image<\|/ref\|> in the image.` | toutes les bboxes de la page + boucle `<td>` sur le tableau |

**Comportement observé :** quand le modèle ne trouve pas l'élément demandé (ou que le target est trop générique), il retourne toutes les bboxes de la page. Quand il trouve un élément spécifique, il retourne uniquement sa bbox et s'arrête. Toute instruction ajoutée après le `<|det|>` est ignorée — la deux passes est obligatoire pour obtenir le contenu d'une région.

**Usage retenu :** `Locate <|ref|>A figure or graph<|/ref|> in the image.` pour test rapide, mais non nécessaire dans la pipeline puisque `layout` repère déjà les bbox avec le texte en plus.

#### Vocabulaire de labels grounding (mode `layout` et `rec`)

Observé sur pages 1–6 (BF16, binarize) :

| Label | Contenu |
|---|---|
| `text` | bloc de texte courant |
| `title` | titre |
| `sub_title` | sous-titre / intertitre |
| `table` | tableau (y compris figures mal classées sur image floue) |
| `table_caption` | légende de tableau |
| `table_footnote` | note de bas de tableau |
| `image` | figure / graphique (label correct sur image nette) |
| `image_caption` | légende de figure |

**Effet de la qualité d'image sur la classification :** page_5 (floue) → graphique classé `table`, hallucination `<td>30</td>` en boucle. Page_6 (nette, même contenu) → graphique correctement classé `image`, contenu vide.

**Contenu des régions `image` :** toujours vide — le modèle détecte et délimite la région mais ne génère aucune description. Voir section ci-dessous pour les résultats des deux passes sur le crop.

#### Deuxième passe sur crop de graphique (2026-04-05, page_6, BF16, `draft/test_two_pass.py`)

| Combinaison | Résultat |
|---|---|
| `parse` + raw | tableau structuré, notes de bas de tableau absentes |
| `parse` + binarize | tableau structuré, notes de bas de tableau présentes mais traitées comme des lignes du tableau |
| `describe` + raw | description générale + bruit d'interprétation |
| `describe` + binarize | description générale + bruit d'interprétation |

**Verdict : `parse` + `binarize` retenu (ère DeepSeek-OCR)** pour la deuxième passe sur les régions `image`. Limitation connue : les notes de bas de tableau sont incluses dans le tableau au lieu d'être séparées.

#### Prompts custom testés (2026-04-03, BF16, binarize)

| Prompt | Résultat |
|---|---|
| `"Describe this image in detail in french."` | description en anglais (instruction de langue ignorée) |
| `"Describe this image in detail in the language of the document."` | description en anglais |
| `"Décrit cette image en détail."` | description en anglais |
| `"Décrit cette image en détail en français."` | description en anglais |
| `"What is the language of the document?"` | répond `"pt"` puis fait l'OCR de la page |
| `"Figure or text?"` | boucle sur le titre |
| `"Does this document contain a figure?"` | décrit le document en anglais |
| `"Does this document contain a figure? Yes \| No"` | écrit les titres puis un court paragraphe incohérent en français |
| `"If this is a figure or illustration, describe it. Otherwise, Free OCR."` | décrit l'image en anglais même sur page de texte |
| `"Transcribe the text exactly as it appears."` | OCR instable et désordonné, qualité inférieure à `"Free OCR."` |
| `"Is there a figure in this document?"` | pages 1–3 : description générale ; page_4 : description détaillée du tableau ; page_5 : boucle |
| `"OCR only the text, ignore any figures."` | ignore figures et tableaux, mais skip aussi certaines colonnes de texte (comportement partiel — voir issues Feature 1) |
| `"<\|grounding\|>Describe this image in detail."` | équivalent exact de `layout` (grounding boxes + classification régions) — testé page_6 (43.5s) |
| `"<\|grounding\|>Parse the figure."` | **freeze terminal** — à éviter |

**Conclusion :** le modèle ignore systématiquement les instructions de langue. Les prompts de classification (yes/no, figure?) ne produisent pas de réponse structurée utilisable. `"OCR only the text, ignore any figures."` est le seul prompt qui filtre réellement les figures, mais de façon incomplète.

---

### Prétraitement des images

> Explorations menées avec DeepSeek-OCR. PaddleOCR fonctionne directement sur image brute —
> aucun prétraitement n'est appliqué dans le pipeline actuel.

#### Image originale (aucun prétraitement) — mode `"none"`
**Statut : retenu comme référence (ère DeepSeek-OCR).** Initialement abandonné (boucles en Q8_0 + prompt `plain` sur page_1). Réévalué en BF16 + prompt `layout` (2026-04-06/07) : aucune boucle sur pages 4, 5, 6, 9. Meilleur score texte sur image floue (94.9% sur page_5). Seule config sans boucle sur toutes les pages testées avec preprocess léger.

**Limite :** ne détecte pas la figure sur page_6 nette (graphique ignoré ou absorbé dans le texte).

#### Exposure boost Pillow (contrast ×1.8 + brightness ×1.2)
**Statut : abandonné.** Évite les boucles mais produit moins de mots (~830 vs ~1000 pour binarize_adaptive), plus lent (~25s), davantage d'hallucinations.

#### CLAHE (égalisation adaptative du contraste, OpenCV LAB)
**Statut : abandonné.** Testé visuellement (`draft/viz_preprocess2.py`) et en OCR (`draft/preprocess_test.py`). Produit ~1519 mots mais avec des boucles en fin de génération. Pas d'amélioration nette sur la précision.

#### Binarisation Otsu (GaussianBlur(5,5) + seuil global Otsu)
**Statut : abandonné.** Boucle immédiatement, hallucinations massives (page_1 : répétitions de "Ils sont des mots utilisés dans la société").

#### EqualizeHist seul / EqualizeHist + binarize adaptive
**Statut : abandonné.** Testé visuellement (`draft/viz_preprocess2.py`). EqualizeHist amplifie le bruit de fond et les gradients d'éclairage — contre-productif avant binarisation adaptative.

#### bg_divide + binarize adaptive
**Statut : non retenu.** Testé visuellement (`draft/viz_preprocess2.py`). Normalise l'illumination en divisant par un fond estimé (GaussianBlur 101×101). Résultat visuel intéressant sur les pages avec éclairage très inégal, mais non testé en OCR.

#### Binarisation adaptive (GAUSSIAN_C, blockSize=31, C=10)
**Statut : remplacé par blockSize=31, C=15.** Provoque des boucles de génération sur pages floues ou bruitées.

**Avantages :** rapide, supprime le fond, robuste aux variations d'éclairage locales.
**Limites :** C=10 efface les traits mous sur images floues → boucles `<td>` HTML.

#### GaussianBlur(5,5) + binarize adaptive (blockSize=31, C=15)
**Statut : retenu (ère DeepSeek-OCR).** Grid test blockSize ∈ {21,31} × C ∈ {10,15} sur page_5 (bruitée, Laplacien=58) et page_6 (nette, Laplacien=134) via `draft/test_binarize_grid.py` + `draft/compare_grid.py`.

- C=10 (ancien défaut) : boucles HTML `<tr><td>` sur pages floues/bruitées. Détection de boucle par fréquence de mots insuffisante pour ce type de boucle — ajout de `_has_char_repeat` dans `ocr_client.py`.
- C=15 : aucune boucle sur page_5 ni page_6. blockSize=31 >≈ blockSize=21 sur page bruitée (91 % vs 85 % de similarité word-level vs référence page_6).

**Limites :** précision dégradée sur pages très bruitées (page_5).

#### Opérations morphologiques après binarisation (opening/closing)
**Statut : abandonné.** Testé sur pages 1, 2, 5, 6 via `draft/test_morpho.py` (kernels 2×2 et 3×3, opening, closing, open+close, close+open). Le texte reste haché sur les pages bruitées — les opérations morphologiques ne récupèrent pas les traits effacés par la binarisation adaptative.

#### Sauvola binarization seul (scikit-image, `threshold_sauvola`)
**Statut : abandonné.** Rend bien le texte en général, mais efface le texte dans les zones à faible variance locale (pliure, ombre de reliure) → la variante `AND` ci-dessous est retenue à la place.

#### fastNlMeansDenoising + binarize adaptive — mode `"nlmeans"`
**Statut : phase OCR terminée sur pages 4, 5, 6, 9.** Testé visuellement sur pages 4, 5, 9 via `draft/test_nlmeans.py`. Configs évaluées : `nlmeans_5`, `nlmeans_10`, `nlmeans_15`, `nlm5_median`, `nlm5_open`, `nlm5_and`, `nlm10_and`, `nlm10_bgdiv`, `nlm10_bgdiv_and`, `median_adaptive`, `median_and`.

**Observations visuelles :**
- `nlmeans_5` : granulés résiduels sur certaines pages, texte plus net qu'avec baseline.
- `nlmeans_10` : bon équilibre débruitage/préservation des traits.
- `nlm5_median` (nlmeans h=5 + medianBlur(3) + adaptive) : granulés supprimés, prometteur.
- `nlm5_and` : très bon sur page_9, granulés sur les autres pages.
- `nlm10_and` : bons résultats, régulier entre pages.
- `median_and` (medianBlur(3) + AND(Sauvola, adaptive)) : prometteur, rapide.
- `nlm10_bgdiv`, `nlm10_bgdiv_and` : non retenus pour l'OCR après visualisation.
- `nlm5_open` (MORPH_OPEN post-binarisation) : non retenu.

**Configs retenues pour phase OCR :** `median_and`, `nlm5_median`, `nlmeans_10`, `nlm10_and`, `nlm5_and`.

**Résultats OCR (2026-04-06, `draft/test_nlmeans.py --ocr`, pages 4/5/6/9, BF16, layout) :**

- **page_4** : toutes les configs bouclent. `median_and`, `nlm5_median`, `nlm10_and` reproduisent approximativement le tableau avant de boucler ; `nlmeans_10` et `nlm5_and` bouclent sans résultat utile.
- **page_9** : `nlm5_median` et `nlm10_and` bouclent dès le début. `nlm10` pseudo-boucle (suite de chiffres — faux négatif de détection de boucle). `median_and` et `nlm5_and` ne bouclent pas, précision apparemment subpar.
- **page_5** (bruitée) : `nlm5_and` → 4 mots (bbox image seule, échec). `median_and`, `nlm5_median`, `nlmeans_10`, `nlm10_and` → 820–958 mots, pas de boucle. Précision évaluée — voir section ci-dessous.
- **page_6** (nette, même contenu que page_5) : toutes configs sans boucle, 944–1005 mots. Référence authentique : `photos/md/page_6.md` (précision 100%).

**Détection de boucle :** `page_4_median_and` a bouclé sans être détecté car les coordonnées `<|det|>[[x,y,…]]<|/det|>` changent à chaque bloc, diluant le ratio `repeated/n_unique`. Fix : retirer les blocs `<|det|>` avant l'analyse de fréquence dans `_is_looping` (`src/ocr_client.py`).

**Intégration :** `src/preprocess.py::nlmeans_binarize()`, `nlmeans_h: int = 15` dans `Config`.

#### medianBlur(3) seul + adaptive
Testé visuellement dans `draft/test_nlmeans.py`. La variante `median_and` (avec AND Sauvola) est retenue pour l'OCR.

#### AND(Sauvola w=51 k=0.3, binarize adaptive) — mode `"sauvola"`
**Statut : retenu (ère DeepSeek-OCR).** Testé sur pages 1, 2, 5, 6 via `draft/test_sauvola_patch.py` + pipeline complète (`--preprocess sauvola`).

`bitwise_and(sauvola(gray, w=51, k=0.3), adaptive_binarize(gray))` — conserve les pixels texte détectés par l'un ou l'autre → corrige la perte de texte de Sauvola dans la pliure, améliore la précision sur les pages bruitées.

**Précision estimée (page_5) :** `0.98 × 0.95 = 93 %` (vs `0.98 × 0.92 = 90 %` pour baseline). Calcul : `sim(sauvola_page5, baseline_page6) = 95 %` et `sim(baseline_page5, baseline_page6) = 92 %`, avec `sim(page6_31_15, page6_31_10) ≈ 98 %` comme proxy de précision de la référence.

**Pipeline complète (2026-04-05, `--preprocess sauvola`) :**
- Boucle sur pages 4, 9, 10.
- Page 4 : retranscrit approximativement le grand tableau (meilleure tentative à ce jour), boucle sur les notes de bas de tableau.
- Page 9 : retranscrit les repères sans boucler, boucle sur le début de la bibliographie.
- Page 10 : boucle au milieu du texte.
- Page 5 : échec complet sur le graphique (très flou, 2e passe).

**Pipeline complète (2026-04-05, `--preprocess binarize`) :**
- Boucle sur pages 4, 5 (2e passe), 9.
- Page 4 : boucle sur le grand tableau sans retranscrire les informations.
- Page 5 : boucle uniquement sur la 2e passe (figure), texte principal OK.
- Page 9 : boucle dès le début ("un, " en boucle).
- Page 10 : première retranscription de la courbe de Lorenz, qualité très médiocre.

**Intégration :** `src/preprocess.py::sauvola_binarize()`, `preprocess_mode="sauvola"` dans `Config`, `--preprocess sauvola` en CLI.

#### Évaluation comparative des prétraitements — page_5 / page_6 (2026-04-06)

5 configs × 2 pages via `compare_ocr.py` (diff=sentence, score=word). Référence : `photos/md/page_6.md`.
Rapport complet : `output/rapports/preprocess_p5_p6.md`.

**page_5 — floue (Laplacian=57.97) :**

| Config | Texte % | Fig % | Global % |
|--------|---------|-------|----------|
| none | **94.9%** | 16.9% | **92.3%** |
| sauvola_and | 93.9% | 16.7% | 92.0% |
| nlmeans_and | 93.1% | 17.0% | 90.8% |
| median_and | 92.2% | **38.0%** | 91.2% |
| blur_adaptive | 92.0% | 16.8% | 90.1% |

**page_6 — nette (Laplacian=134.19) :**

| Config | Texte % | Fig détectée | Fig % | Global % |
|--------|---------|:------------:|-------|----------|
| blur_adaptive | **96.3%** | oui | 94.9% | **96.3%** |
| none | 95.7% | **non** | — | 96.1% |
| nlmeans_and | 96.1% | oui | 92.8% | 95.8% |
| median_and | 96.0% | oui | 92.8% | 95.6% |
| sauvola_and | 94.6% | oui | **96.6%** | 94.4% |

**Conclusions :**
- Sur image floue, `none` donne le meilleur score texte — le prétraitement dégrade les traits déjà mous.
- Sur image nette, `blur_adaptive` est la meilleure config équilibrée. `none` rate systématiquement la figure.
- La figure reste intraitable sur image floue quelle que soit la config (max 38%).
- **Piste :** prétraitement conditionnel selon Laplacian — `blur_adaptive` si > seuil, `none` sinon.

#### Prétraitements légers sans binarisation (2026-04-07, `draft/test_preprocess.py`, pages 4/5/6/9, BF16, layout)

Hypothèse testée : DeepSeek-OCR étant entraîné sur photos naturelles, les filtres légers préservant le look photo sont préférables à la binarisation. Scripts : `draft/test_preprocess.py` (OCR), `draft/realesrgan_sesr.py` (génération SR).

Rapport complet : `output/rapports/preprocess_legers_analyse.md`.

**fastNlMeansDenoising seul — mode `"nlmeans"`**
**Statut : retenu, défaut pipeline (ère DeepSeek-OCR).** `h = nlmeans_k × noise_level`. Boucle sur page_4 bruitée (noise=5.5) — seule page problématique. Aucune boucle sur pages 5, 6, 9, 10 et variantes clean. Précision texte pur ~99% sur images clean (p56c_nlmeans=99.3%). Coût preprocess ~2-3s.

**bilateralFilter(d=9, σ=75) — mode `"bilateral"`**
**Statut : abandonné.** Boucle sur page_6 (nette, 31 mots) et page_4. L'effet "cartoon" (zones très lisses + bords très nets) perturbe le modèle sur images nettes. Comportement inverse de l'attendu.

**SESR-M7 x2 (AMD NPU, 256×256 tiles) → resize original — mode `"sesr"`**
**Statut : retenu, disponible en option (ère DeepSeek-OCR).** ~7s/image sur NPU. Boucle sur page_4 bruitée (noise=5.5) et sur page_4_clean sur symboles de bas de page (texte principal complet). Précision texte pur légèrement supérieure à nlmeans sur images clean (p56c_sesr=98.8%, p6_sesr=99.2%). Intégré dans `src/sesr.py`.

| Config | p5 texte % | p6 texte % | p56c texte % |
|--------|:---:|:---:|:---:|
| none | 98.8% | 98.7% | 97.9% |
| nlmeans | 98.9% | 98.8% | 99.3% |
| sesr | 98.7% | 99.2% | 98.8% |

Résultats 2026-04-07, `compare_ocr.py` mode composant texte pur, référence `photos/md/page_6_text.md`, rapport `output/rapports/global_report_5-6.md`.

**RealESRGAN x4 (AMD NPU, 128×128 tiles) → resize original — mode `"esrgan"`**
**Statut : abandonné.** ~137s/image. Gain nul ou marginal vs `none`. Ratio coût/bénéfice rédhibitoire.

**Pivot architectural (2026-04-07)**
Tables et figures traitées comme des crops (référencées par chemin image, non retranscrites). Score de précision calculé sur **texte pur** uniquement. Objectif : >99% texte sur image clean. Images clean disponibles : page_4, page_5, page_6, page_9, page_10.

#### Unsharp Mask (standard : `img + alpha*(img - blurred)`)
**Statut : abandonné.** Testé dans `draft/test_unsharp.py`. Amplifie les hautes fréquences — ajoute des granulés sombres, dégrade la binarisation sur la plupart des configs. Inefficace sur du flou de mise au point (l'information haute fréquence est physiquement perdue).

#### Unsharp Mask inversé (`blurred - alpha*(img - blurred)`)
**Statut : abandonné.** Aucune amélioration sur la binarisation, introduce des artefacts sur certaines configs.

#### page-dewarp (lmmx, `pip install page-dewarp[jax]`)
**Statut : abandonné.**

Nos images sont des doubles pages (deux pages par photo). page-dewarp est conçu pour une seule page — il ne détecte pas correctement les contours ni les lignes de texte sur des images deux pages et produit un résultat dégradé.

Tentative de contournement : couper l'image en deux moitiés (mi-largeur), dewarp chaque moitié séparément, recombinaison. Problème : page-dewarp recadre la sortie selon sa détection de contours de page, ce qui coupe les bords des demi-images artificielles et rend la recombinaison incohérente.

---

## PaddleOCR VL 1.5 (stack actuel, 2026-04-07)

Modèle : PaddleOCR-VL-1.5, 0.9B paramètres, format GGUF (F16).
Scripts : `draft/test_paddle.py`, `draft/compare_ocr.py`.

### Stack retenu

- **llama-server** (llama-b8683-bin-win-vulkan-x64, Vulkan) — inférence VLM
- **paddleocr** depuis le dépôt main (pas PyPI 3.4.0 — `llama-cpp-server` backend absent de la release) — orchestration layout + routing des prompts
- **paddlepaddle CPU** — layout detection (ppdoclayout)
- **Python 3.10** (conda env `ocr-livre`) — incompatibilité paddlepaddle avec Python 3.11+
- Patch paddlex requis : `docs/dev/apply_paddlex_patch_otsl.py` (voir `docs/dev/paddlex_patch_otsl.md`)

### Résultats de précision (texte pur, référence `page_6_text.md`)

| Page | Condition | Texte % | Notes |
|------|-----------|---------|-------|
| page_6 | clean | 100% | seule diff : niveau de header (`##` vs `###`) |
| page_5-6 | clean | 100% | idem |
| page_5 | bruitée | 99.9% | seule erreur réelle : "croisance" au lieu de "croissance" |
| page_4 | bruitée | 97% vs version clean | tableau complexe : chiffres avec quelques erreurs, notes de bas de page ignorées |
| page_9 | bruité vs clean | 99.5% | erreurs mineures sur version bruitée |

Global (~98%) : différences sur accents, formatage ou erreurs facilement interprétables.

### Vitesse

35–40s/image vs 45–55s pour DeepSeek-OCR BF16. Gain ~25%.

### Comportement sur les tableaux complexes

PaddleOCR utilise un pipeline spécialisé pour les tableaux :
1. ppdoclayout extrait les cellules via OCR
2. Le contenu est encodé en OTSL (`<fcel>col<fcel>col<nl>...`)
3. Le VLM reçoit l'OTSL pour reconstruction en HTML

Avec le backend `llama-cpp-server`, llama-server ne sait pas parser l'OTSL comme image → erreur 500. Le patch `paddlex_patch_otsl.md` intercepte cette erreur par région, extrait l'OTSL depuis le message d'erreur, et le convertit directement via `convert_otsl_to_html()`.

### Format de sortie

HTML embarqué dans Markdown :
- Tableaux : `<table><tr><td>...</td></tr></table>` (HTML brut sans styles — `pretty=False` retenu, voir ci-dessous)
- Figures : `<img src="imgs/..." />` (crop sauvegardé localement)
- Exposants : `<sup>er</sup>`

Pas de tokens DeepSeek (`<|ref|>`, `<|det|>`).

### Vitesse et optimisations (2026-04-08/09)

Vitesse mesurée baseline : ~60s/page. Bottleneck = vitesse de génération brute (~36 tok/s sous Vulkan) + idle entre blocs (GPU alternance layout detection → appel HTTP → idle).

**`n_parallel=2` seul (llama-server `-np 2`, pages entières en parallèle)** : testé, **abandonné**. Augmente le temps total (128s + 85s vs 55s + 55s). Contention GPU Vulkan — les deux requêtes se battent pour le GPU entier. De plus, le contexte est divisé entre slots : -np 2 avec -c 4096 → 2048 tokens/slot → blocs longs tronqués (HTTP 400).

**Resize PIL avant predict (`--max-image-size 1500`)** : testé, **abandonné**. Accélère légèrement mais dégrade fortement la qualité OCR. Cause : le resize s'appliquait avant la layout detection. Image source : 4080×3072.

**`max_pixels` (param PaddleOCR)** : non applicable pour `llama-cpp-server` (seulement `vllm-server`). Ignoré silencieusement. Défaut interne : `28 × 28 × 3600 = 2 822 400` pixels.

**`save_to_markdown(pretty=True)`** : **retenu**. Les styles inline sont ajoutés par PaddleOCR en post-processing, pas générés par le VLM. Strip des styles `<td>`/`<th>` dans `postprocess.py`.

**Parallélisation intra-page (pool global ThreadPoolExecutor, 2026-04-09)** : **retenu**, `docs/dev/apply_paddlex_patch_parallel.py`.

Principe : PaddleOCR traite les blocs séquentiellement. Le patch soumet tous les blocs de toutes les pixel_keys simultanément à un pool global, les workers pickent en continu sans restart.

| Config | Temps/page |
|---|---|
| séquentiel (baseline) | ~60s |
| -np 2, 2 workers | ~49s |
| -np 3, 3 workers | ~46s |
| -np 4, 4 workers | crash (vision encoder Vulkan saturé) |
| -np 6, 6 workers | hang |
| -np 3, 3 workers, **-c 6144** (2048/slot) | **~43.6s** — **retenu** |

Gain : ~35 min sur 150 pages. Config retenue : `-np 3 -c 6144` (2048/slot). Réduction de `-c 12288` → `-c 6144` : gain supplémentaire ~2.5s/page sans troncature observée.

**Parallélisation inter-pages — 2 serveurs llama-server (2026-04-09)** : **abandonné** (encore disponible via `n_servers > 1` dans `Config`, mais inutile sur APU). `src/pipeline.py` modifié pour lancer N serveurs sur des ports distincts (8080, 8081…) et traiter les pages en parallèle via `ThreadPoolExecutor`.

Résultats sur 3 pages test (pages 1–3) :
- Pages 1 et 2 (simultanées) : ~102s et ~105s chacune
- Page 3 (serveur libéré) : ~55s
- Débit moyen : ~53s/page — gain nul vs séquentiel (56s/page baseline)

Cause : sur APU (Ryzen AI 9 HX370), le GPU et la RAM sont physiquement le même LPDDR5X. Sous Vulkan/Windows, les command queues de deux processus distincts sont **sérialisées par le driver** — pas de vrai parallélisme inter-processus. Chaque serveur obtient ~50% du GPU, devenant ~2× plus lent. Le débit total est identique.

Variante testée conceptuellement (1 GPU + 1 CPU) : abandonnée sans implémentation. CPU inference ~5–10× plus lente que GPU, même bus mémoire → débit limité par le serveur CPU.

**Conclusion :** sur GPU unique, le continuous batching intra-page (1 serveur, -np 3) reste la seule optimisation efficace. L'approche multi-serveur n'apporte rien sur APU.

**Config finale retenue (2026-04-09)** : 1 serveur, -np 3, -c 6144. Vitesse mesurée sur `photos/test/` : **43.4s/page texte, 37s/page avec graphe**.

Architecture pipeline mise à jour dans cette version :
- Pages écrites dans `output/parts/<page_id>.part` (pas de lock, reprise robuste aux crashs)
- Combinaison dans l'ordre d'entrée en fin de run
- Retry ×1 dans `ocr_client.py` sur sortie vide, MD non généré ou MD vide
- Fallback `PaddleOCRVL` (layout → no-layout) supprimé à ce stade (couvert par le patch OTSL)

### Timeout page + fallback no-layout (post 2026-04-09)

`page_timeout = 120s` : `pipeline.predict()` s'exécute dans un thread de surveillance. Si la timeout est dépassée, `OCRTimeout` est levé.

Sur `OCRTimeout` dans `pipeline.py` :
1. Tous les serveurs sont killés et relancés (`restart_servers()`)
2. La page est retraitée avec un pipeline fallback `use_layout_detection=False`
3. Si le fallback échoue aussi (`OCRError`) : bloc `<!-- Page page_xxx — ERREUR -->` + poursuite

**Motivation :** certaines pages déclenchent des boucles de génération interne à llama-server non détectées par `ocr_client.py` (l'appel HTTP ne revient jamais). Ni le retry, ni le patch OTSL ne couvrent ce cas. Le timeout + restart est le seul moyen de récupérer proprement sans bloquer le run.

### Streaming HTTP pour débloquer -np 4 (2026-04-09)
**Statut : abandonné.** `docs/obsolete/apply_paddlex_patch_streaming.py`.

Principe : patcher `GenAIClient.create_chat_completion` (genai.py) pour utiliser `stream=True` et libérer un sémaphore asyncio après le premier token de contenu (= fin du prefill). Permettrait d'avoir 3 prefills simultanés max tout en gardant 4 slots en génération.

Résultats :
- `-np 3` + streaming : aucun gain (attendu — 3 slots disponibles, le 4ème worker queue côté serveur)
- `-np 4` + streaming : texte incomplet, aucun gain de temps

Cause probable du texte incomplet : llama-server envoie un chunk `role: assistant` (contenu vide) avant la fin du prefill. La version initiale libérait le sémaphore sur `stream.__anext__()`, soit sur ce chunk vide, laissant passer les 4 prefills simultanément et corrompant les générations. Version corrigée (attente du premier token non-vide) : même résultat — texte incomplet, aucun gain. Cause racine non identifiée, probablement une limitation de llama-server à 4 slots simultanés en streaming sous Vulkan.

### Verdict

**PaddleOCR VL 1.5 retenu et intégré** — supérieur à DeepSeek-OCR BF16 sur tous les critères :
précision, vitesse, absence de boucles. Résout Bug 4 (boucles), Amélioration 1 (vitesse) et Amélioration 2 (modèle alternatif).

Pipeline principale (`src/`) migrée vers PaddleOCR (2026-04-08) : `ocr_client.py` réécrit, `pipeline.py` simplifié (preprocess/figure/nexaai supprimés), `config.py` nettoyé.

---

### `markdown_ignore_labels`

Paramètre du constructeur `PaddleOCRVL(...)` — liste des labels de blocs exclus de la sortie markdown.

**Config retenue :**
```python
markdown_ignore_labels=["header_image", "footer", "footer_image"]
```

| Label retiré de la liste (inclus) | Effet observé |
|---|---|
| `number` | numéro de page imprimé récupéré → extrait dans `<!-- Page page_2 (p. 42-43) -->` via `extract_page_number()` |
| `header` | en-tête courant récupéré (ex : "Leçon 5") — testé pages 2 et 9, sans bruit particulier |
| `footnote` | notes de bas de page récupérées — testées, ignorées car bruit sans valeur |
| `aside_text` | texte marginal récupéré — retenu (contenu potentiellement utile) |

Labels toujours ignorés : `header_image`, `footer`, `footer_image`.

---

### Paramètres de génération

| Paramètre | Valeur testée | Effet |
|---|---|---|
| `max_tokens` | 4096 | valeur courante — 2048 coupait certaines pages longues |
| `temperature` | 0.0 | déterministe, retenu |
