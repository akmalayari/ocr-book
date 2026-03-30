# Issues — à traiter plus tard

## Features
rename_images par date de création: du plus vieux au plus récent. Permet de reconstruire le livre dans l'ordre.

## Style / architecture

### `PROMPTS` comme constante de module
`config.py:26` — `PROMPTS` est un dict statique dans un `dataclass` avec `field(default_factory=...)`.
Chaque instance crée son propre dict alors que la valeur ne change jamais.
Mieux placé comme constante au niveau module ou comme `ClassVar`.


### Padding du renommage non documenté
`images.py:71` — `width = len(str(len(images)))` donne un padding minimal basé sur
le nombre d'images au moment du renommage. Si on ajoute des images plus tard et qu'on
re-renomme, les anciens fichiers (`page_01.jpg`) et les nouveaux (`page_001.jpg`)
seront incohérents. À documenter ou à fixer avec un minimum de 3 chiffres.

## Bug bloquant

### HTTP 500 — `SDKError(Multimodal generation failed)` sur `nexa serve`
`ocr_client.py:79` — Les requêtes image+texte vers `/v1/chat/completions` retournent
systématiquement HTTP 500 (`{"code":-201201,"error":"SDKError(Multimodal generation failed)"}`).
Les requêtes texte seul fonctionnent. Le modèle lui-même est fonctionnel via `nexa infer`.

**Environnement :** Windows 11, NexaSDK Bridge v1.0.45-rc1, Python package nexaai 1.0.44.

**Cause probable :** Bug dans le pathway multimodal du serveur REST Nexa sur Windows.

**Pistes explorées :**

`/v1/cv` — endpoint existant, retourne 400 si `model` absent, 500
`SDKError(Operation not supported)` avec le modèle. Écarté : la classe `CV` du SDK
attend `det_model_path` + `rec_model_path` + `char_dict_path` (pipeline PaddleOCR),
incompatible avec DeepSeek-OCR-GGUF qui est un VLM GGUF. L'endpoint `/v1/cv` ne
sait pas gérer ce type de modèle.

`nexaai.CV` (Python direct) — tentée, bugs rencontrés (cohérent avec le diagnostic
ci-dessus : mauvais type de modèle).

`nexaai.VLM` (Python direct) — tentée, semblait inadaptée à DeepSeek-OCR (probablement
en raison des chemins mmproj/tokenizer spécifiques ou du format de prompt).

**Autres pistes à explorer :**
- Tester `nexa serve --verbose` pour voir les logs internes du SDK au moment de l'erreur.
- Vérifier si le problème est lié à la taille de l'image (4.4 MB, ~6 MB en base64).
- Tester avec une image PNG ou WebP au lieu de JPEG.
- Tester avec une version antérieure de nexaai.

## À creuser

### Redimensionnement des images avant envoi OCR
`ocr_client.py:56` — Les photos haute résolution sont encodées en base64 entièrement
en mémoire avant envoi. Aucune compression/resize. À vérifier si l'API Nexa impose
une taille maximale, et si oui, ajouter un resize automatique avec opencv.
