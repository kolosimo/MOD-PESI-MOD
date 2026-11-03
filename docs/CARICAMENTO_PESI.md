# Come portare i pesi nella sessione cloud

Situazione verificata il 3/11/2026 nella sessione cloud:

- Repository vuoto (nessun peso caricato).
- `huggingface.co` e `*.hf.co` **bloccati** dalla policy di rete dell'ambiente (proxy 403). Il connettore Hugging Face di claude.ai è attivo ma legge solo file di testo, non i pesi.
- PyPI raggiungibile (si possono installare `torch`, `transformers`, `safetensors`, `gguf`).
- Spazio disco libero ~31 GB, RAM 15 GB, 4 core (valori di questa sessione, non garantiti in altre).

## Opzioni, dalla più semplice

### A. Scaricare da Hugging Face direttamente nella sessione (consigliata)
Se il modello che hai in locale è pubblico su Hugging Face, è la via più pulita: niente LFS, niente quote.
Serve aggiungere `huggingface.co` (e probabilmente il suo CDN, es. `cdn-lfs.huggingface.co` /
`cas-bridge.xethub.hf.co` – **da verificare** in base agli errori) agli *Allowed domains* dell'ambiente:
menu dell'ambiente cloud nella barra del titolo della sessione → Edit → Network access.
Guida: https://code.claude.com/docs/en/cloud-environments#network-access

### B. Git LFS sul repository (scelta del 3/11/2026)

Verifiche fatte dalla sessione cloud:

| Passo | Esito |
|---|---|
| `git-lfs` installato nel container | sì (3.4.1) |
| API LFS del repo (`/info/lfs/objects/batch`) | raggiungibile e autenticata |
| Upload dal container | **bloccato**: il file arriva allo storage, ma la conferma finale su `lfs.github.com` viene rifiutata dal proxy |
| Download nel container | probabile (l'host di download risponde), **non ancora provato** su un file reale |

→ **L'upload si fa dal PC di Delio**; la sessione cloud scarica con `git lfs pull`.

Procedura sul PC (Git for Windows include già Git LFS):
```bash
git clone https://github.com/kolosimo/MOD-PESI-MOD.git
```
```bash
cd MOD-PESI-MOD && git checkout claude/peaceful-knuth-nhw87k && git lfs install
```
Copia la cartella del modello in `models/<nome-modello>/` (config.json, tokenizer, *.safetensors), poi:
```bash
git add models/ && git lfs ls-files
```
`git lfs ls-files` deve elencare i file dei pesi: se non compaiono, NON fare commit (finirebbero in git normale e il push fallirebbe sopra i 100 MB).
```bash
git commit -m "Aggiunti pesi <nome-modello>" && git push origin claude/peaceful-knuth-nhw87k
```
Il file `.gitattributes` del repo traccia già `*.safetensors`, `*.gguf`, `*.bin`, `*.pt`, `*.pth` con LFS.

Limiti **da verificare** prima del push (GitHub → Settings → Billing and licensing):
quota di spazio e banda LFS del piano gratuito e dimensione massima per singolo file LFS.
Ogni download nella sessione cloud consuma banda LFS: scaricare il modello una volta per sessione.

Nella sessione cloud:
```bash
git pull origin claude/peaceful-knuth-nhw87k && git lfs pull
```

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
