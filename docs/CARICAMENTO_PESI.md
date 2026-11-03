# Come portare i pesi nella sessione cloud

Situazione verificata il 3/11/2026 nella sessione cloud:

- Repository vuoto (nessun peso caricato).
- `huggingface.co` **bloccato** dalla policy di rete dell'ambiente (proxy 403).
- PyPI raggiungibile (si possono installare `torch`, `transformers`, `safetensors`, `gguf`).
- Spazio disco libero ~31 GB, RAM 15 GB, 4 core (valori di questa sessione, non garantiti in altre).

## Opzioni, dalla più semplice

### A. Scaricare da Hugging Face direttamente nella sessione (consigliata)
Se il modello che hai in locale è pubblico su Hugging Face, è la via più pulita: niente LFS, niente quote.
Serve aggiungere `huggingface.co` (e probabilmente il suo CDN, es. `cdn-lfs.huggingface.co` /
`cas-bridge.xethub.hf.co` – **da verificare** in base agli errori) agli *Allowed domains* dell'ambiente:
menu dell'ambiente cloud nella barra del titolo della sessione → Edit → Network access.
Guida: https://code.claude.com/docs/en/cloud-environments#network-access

### B. Git LFS sul repository
```bash
git lfs install
```
```bash
git lfs track "*.safetensors" "*.gguf"
```
```bash
git add .gitattributes models/ && git commit -m "Aggiunti pesi" && git push
```
Il file `.gitattributes` del repo traccia già questi formati con LFS.
Limite: quota LFS gratuita di GitHub (spazio e banda) – **da verificare** sul tuo account
(Settings → Billing → Git LFS) prima di caricare GB.

### C. File spezzati < 100 MB, senza LFS
```bash
split -b 95M -d models/model.safetensors models/model.safetensors.part-
```
In sessione si ricompone con `cat models/model.safetensors.part-* > models/model.safetensors`
e si verifica con `sha256sum`. Funziona, ma appesantisce la storia git: va bene per un solo modello piccolo.

## Quale modello

Per E0–E4 basta un modello piccolo: **0,5–1,5 B parametri, safetensors BF16** (~1–3 GB).
Più grande = più lento su CPU e più difficile far girare la baseline, senza vantaggi per la lettura.
Se hai solo un GGUF quantizzato Q4_K_M: la lettura sarà parziale (K-quants non decodificati) e la
scrittura non è supportata dai tool → meglio procurarsi la versione safetensors dello stesso modello.

Metti i file in `models/<nome-modello>/` (con `config.json`, tokenizer e `*.safetensors`).
