# Utilisation de `nexaai.VLM` pour l'OCR — explication

## Pourquoi l'approche fonctionne

### Le modèle DeepSeek-OCR-GGUF est un VLM

Le cache Nexa contient deux fichiers distincts :

```
DeepSeek-OCR.Q8_0.gguf   — le LLM texte (poids quantifiés)
mmproj.F16.nexa           — le projecteur multimodal (vision encoder → espace LLM)
```

Ce format GGUF + mmproj est celui des VLMs llama.cpp (LLaVA, etc.). La classe
`nexaai.VLM` est conçue pour ce cas précis : elle charge les deux fichiers et câble
le projecteur. `nexaai.CV` et le serveur REST (`nexa serve`) ne savent pas gérer ce
type de modèle, d'où les erreurs HTTP 500.

### Chargement via `VLM.from_()`

```python
vlm = VLM.from_("NexaAI/DeepSeek-OCR-GGUF")
```

`ModelLoaderMixin.from_()` lit le `nexa.manifest` déjà présent dans le cache local,
en extrait les chemins vers le GGUF et le mmproj, puis instancie `VLM` avec les bons
paramètres. Pas besoin de spécifier les chemins manuellement.

### Format du prompt

`apply_chat_template()` avec un `VlmChatMessage` contenant un `VlmContent(type="image")`
et un `VlmContent(type="text")` produit :

```
<image>\n<texte du prompt>
```

Le token `<image>` est le placeholder attendu par le modèle (format LLaVA). Il indique
à la C lib où insérer les embeddings visuels dans la séquence.

### Passage du chemin image

Le chemin de l'image ne doit **pas** être dans la string de prompt. Il se passe via
`GenerationConfig` :

```python
config = GenerationConfig(image_paths=[str(image_path)])
result = vlm.generate(formatted_prompt, config=config)
```

La C lib (`nexa_bridge.dll`) lit le fichier image, l'encode via le projecteur mmproj,
et injecte les embeddings visuels à l'emplacement du token `<image>` dans le prompt.

### Monkey-patch de `ProfileData.from_c_struct`

Sur Windows, la C lib retourne des données de profiling corrompues dans le champ
`stop_reason` (byte `0xc0`, invalide en UTF-8). Le crash se produit **après** que le
texte OCR a été généré et extrait — la génération elle-même réussit.

Le patch intercepte uniquement la désérialisation des métadonnées de profiling :

```python
@classmethod
def _safe_from_c_struct(cls, c_struct):
    try:
        return _orig_from_c_struct(cls, c_struct)
    except (UnicodeDecodeError, AttributeError):
        return cls(stop_reason="unknown")
```

Le texte OCR (`result.full_text`) est intact.

## Récapitulatif du flux

```
VLM.from_("NexaAI/DeepSeek-OCR-GGUF")
  └── lit nexa.manifest → résout chemins GGUF + mmproj

apply_chat_template([VlmChatMessage(image + text)])
  └── produit "<image>\n<prompt>"

vlm.generate("<image>\n<prompt>", config=GenerationConfig(image_paths=[...]))
  └── C lib charge l'image → encode via mmproj → génère le texte
  └── _extract_result : récupère full_text, monkey-patch absorbe le stop_reason corrompu
```
