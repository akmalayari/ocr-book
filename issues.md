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

## ~~Bug bloquant~~ — RÉSOLU

### HTTP 500 — `SDKError(Multimodal generation failed)` sur `nexa serve` ✓ contourné

**Cause :** DeepSeek-OCR-GGUF est un VLM (GGUF + mmproj). Le serveur REST `nexa serve`
ne sait pas gérer ce type de modèle sur Windows — bug dans son pathway multimodal.

**Solution adoptée :** Appel Python direct via `nexaai.VLM`, sans serveur REST.
Voir `docs/nexa_vlm.md` pour le détail complet.

Pattern fonctionnel (`draft/nexa_vlm.py`) :

```python
vlm = VLM.from_("NexaAI/DeepSeek-OCR-GGUF")
msg = VlmChatMessage(role="user", contents=[
    VlmContent(type="image", text=str(image_path)),
    VlmContent(type="text", text=prompt),
])
formatted = vlm.apply_chat_template([msg])
config = GenerationConfig(image_paths=[str(image_path)])
result = vlm.generate(formatted, config=config)
```

**Bug résiduel :** Sur Windows, la C lib (`nexa_bridge.dll`) retourne des données de
profiling corrompues dans `stop_reason` après génération — `UnicodeDecodeError` dans
`ProfileData.from_c_struct`. La génération elle-même réussit. Contourné par
monkey-patch de `ProfileData.from_c_struct` (voir `draft/nexa_vlm.py`).

**À faire :** Intégrer ce pattern dans `ocr_client.py` en remplacement des appels HTTP.


