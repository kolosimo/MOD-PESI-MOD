# Registro risultati

## Accesso ai pesi (3/11/2026)
- Via scelta: download diretto da Hugging Face nella sessione cloud.
- Domini consentiti necessari: `huggingface.co` (API e file piccoli) + `us.aws.cdn.hf.co` (file dei pesi, redirect Xet).
  `hf.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.hf.co` risultavano ancora bloccati ma non sono serviti.
- Git LFS: upload dal container bloccato (verify su `lfs.github.com`); Remote Control: nessuna cartella visibile.

## E0 – Inventario e baseline: Qwen/Qwen2.5-0.5B-Instruct
- `model.safetensors` 988.097.824 byte, SHA-256 `fdf756fa…88fb7fe` = valore pubblicato su HF.
- 494,03 M parametri, tutti BF16, 290 tensori, 24 blocchi, hidden 896, MLP 4864,
  14 teste di query / 2 KV (GQA), vocabolario 151.936.
- **Embedding legati** (`tie_word_embeddings: true`, nessun `lm_head` separato):
  la riga di un token è insieme vettore di input e vettore di output.
- Bias presenti solo su q/k/v.
- Baseline (`logs/baseline/qwen05_originale.json`): 10 prompt greedy + perplexity 4,8375 su `prompts/ppl.txt`.
  Tempo ~38 s su CPU (4 core). Due run identici → 10/10 output uguali, stessa perplexity (deterministico).
- Osservazione: a "Quanto fa 12 per 7?" il modello risponde in portoghese e calcola una divisione. Utile come caso di controllo.

## E3-01 – Soppressione del token "Paris"
- Previsione scritta prima: `logs/e3/01_paris_previsione.md`.
- Lettura preliminare: norma riga "Paris" 0,413, " Paris" 0,376 (mediana righe 0,451). Token più vicini a " Paris"
  per coseno: "Paris" 0,79, "巴黎" 0,65, " paris" 0,53, " France" 0,46, " French" 0,40, " London" 0,38.
  (lettura strumentale: calcolata con codice, non "a vista").
- Modifica: righe 59604 e 12095 di `model.embed_tokens.weight` × 0 (1.792 valori su 136 M).
- Misura (`logs/e3/01_paris_misura.json`):
  | Previsione | Esito |
  |---|---|
  | 1. "Paris" non più generato; il modello sbaglia perché non "vede" la parola nel prompt | **confermata**: scrive `the word "HAH"` e poi `HAHAHA…` |
  | 2. Capitale della Francia invariata ("Parigi" usa altri token) | **confermata** |
  | 3. Altri 8 prompt identici | **confermata** (9/10 identici in totale) |
  | 4. Perplexity varia < 0,5% | **confermata** (4,8375 → 4,8375) |
- Limite: è una modifica "da manuale", non una prova di lettura nativa. Dimostra che la catena
  lettura → modifica → misura funziona su un modello reale.
- Bug trovato e corretto durante la prova: `edit_weights.py --inplace-out` sovrascriveva la modifica precedente
  (ripartiva dall'originale). Aggiunto test di regressione.
