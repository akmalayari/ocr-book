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

## À creuser

### Redimensionnement des images avant envoi OCR
`ocr_client.py:56` — Les photos haute résolution sont encodées en base64 entièrement
en mémoire avant envoi. Aucune compression/resize. À vérifier si l'API Nexa impose
une taille maximale, et si oui, ajouter un resize automatique avec opencv.
