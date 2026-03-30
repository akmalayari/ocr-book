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

**Piste explorée : SDK Python direct (`VLM` class)**
Contourner le serveur HTTP en chargeant le modèle en-process via `nexaai.VLM.from_()` +
`apply_chat_template()` + `generate()`. API disponible et cohérente, mais cette route
a déjà été tentée et a rencontré des problèmes (à documenter ici si reproductibles).

**Autres pistes à explorer :**
- Tester avec une version antérieure de nexaai (ex. 1.0.43 ou avant).
- Tester `nexa serve` en mode verbose pour voir les logs internes du SDK.
- Vérifier si le problème est lié à la taille de l'image (4.4 MB, ~6 MB en base64).
- Tester avec une image PNG ou WebP au lieu de JPEG.

## À creuser

### Redimensionnement des images avant envoi OCR
`ocr_client.py:56` — Les photos haute résolution sont encodées en base64 entièrement
en mémoire avant envoi. Aucune compression/resize. À vérifier si l'API Nexa impose
une taille maximale, et si oui, ajouter un resize automatique avec opencv.
