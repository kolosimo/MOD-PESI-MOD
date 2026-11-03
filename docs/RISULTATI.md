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

## E1 run1 – Lettura alla cieca (33 porzioni da 64 valori, 11 classi)
Protocollo: chiave generata con seme da `os.urandom`, sigillata con SHA-256 (`32e3b1a8…`);
risposte date leggendo solo `probes.md`, senza codice sui valori, e committate (`7c8aa6a`) prima di aprire la chiave.

| Metodo | Classe (11) | Profondità (3) |
|---|---|---|
| **Claude, a vista** | **27,3%** (9/33) | **30,0%** |
| k-NN su 10 statistiche (addestrato su altre righe dello stesso modello) | 21,2% | 46,7% |
| Centroide su statistiche | 36,4% | 16,7% |
| Caso | 9,1% | 33,3% |

Lettura dei risultati:
- I 9 corretti sono quasi tutti "facili": 3/3 bias (valori O(1) con outlier fino a −14) e i vettori di norma
  (6/6 riconosciuti come norma, ma solo 2/6 col tipo giusto in/post = caso).
- Matrici + embedding (24 porzioni): 4/24 corretti ≈ 17%, caso ≈ 12,5% → **indistinguibile dal caso** con n così piccolo.
- Embedding: 0/3. Le tre righe di embedding erano le porzioni con valori più piccoli, che ho attribuito a `mlp_down`:
  il segnale di intensità c'era, la mappatura era sbagliata.
- Profondità: 30% contro 33% del caso → **nessuna capacità**.
- Un classificatore banale su statistiche (centroide) fa meglio di me sulla classe.
- Intervallo di confidenza al 95% su 9/33: circa 13–45%. Un solo run, un solo modello.

Conclusione di E1 run1: **nessuna evidenza di lettura "nativa"** dei pesi oltre a ciò che spiegano
segno e intensità dei valori. Non confuta lo studio (che riguardava reti piccole e un altro LLM), ma per questo
compito e questo modello la capacità non c'è.

## E1 run2 – Lettura alla cieca, 4 righe × 32 valori (pre-registrata)
| Metodo | Classe | Profondità |
|---|---|---|
| **Claude, a vista** | **36,4%** | **43,3%** |
| k-NN statistiche | 27,3% | 56,7% |
| Centroide statistiche | 27,3% | 36,7% |
| Caso | 9,1% | 33,3% |
- Criterio pre-registrato (≥ miglior classificatore +15 punti): **NON superato** (+9 punti sulla classe; −13 sulla profondità).
- Norme 6/6 con il tipo giusto (incluso P12 = norm_in layer 0, dedotto dal vincolo di conteggio); bias 3/3.
- Matrici: 2/21 + embedding 1/3 → al livello del caso.

## E4 – Ablazione: previsione dai pesi, poi misura (pre-registrata)
Base: perplexity 11,07 su 2.861 token di wikitext-2. Previsioni committate in `242ad11` prima della misura.

| Test | Claude (statistiche pesi + conoscenza) | Predittore ingenuo | Soglia | Esito |
|---|---|---|---|---|
| E4a MLP per layer (Spearman) | **+0,887** | +0,104 (norma down_proj) | ≥ 0,5 | **superata** |
| E4a attenzione per layer (Spearman) | +0,167 | +0,007 (norma o_proj) | ≥ 0,5 | non superata |
| E4b gruppi di neuroni layer 12 (top5/casuali) | 0,47× | – | ≥ 2× | non superata (**fallimento previsto**) |

- MLP: misurati più dannosi L0 (ppl 175.104, modello distrutto), L2 (73), L23 (20,7). Previsti: L2, L0, L21, L23.
  I due layer con il neurone anomalo (L2) e la norma d'ingresso minuscola (L0) erano leggibili nelle statistiche dei pesi.
  Errore: L21 (neurone con punteggio 18) ha Δ solo +2,7 → il "super neurone" di L21 non è critico.
- Attenzione: L1 (ppl 19.912) e L0 (123) catastrofici, previsti 1° e 2° (invertiti). Ma L8 (+6,5) e L15-17 non previsti,
  e L2-L3 previsti alti sono bassi → ranking complessivo debole.
- E4b: nel layer 12 i gruppi da 64 neuroni sono quasi equivalenti (Δ tutti < 0,05 su 11,07): i pesi non li distinguono, come previsto.
- Verifica scrittura su file: MLP L0 azzerata via `edit_weights.py` → ppl 175104,0024 = misura in memoria. ✔

## Confronto E1 vs E4
| | E1 – lettura "nativa" a vista | E4 – lettura strumentale (statistiche) + conoscenza |
|---|---|---|
| Cosa si chiede | dire cos'è un frammento di pesi | prevedere l'effetto di una modifica |
| Risultato | non batte un classificatore banale (2 run) | MLP ρ = 0,89 (soglia superata); attenzione ρ = 0,17 |
| Dove funziona | segnali grossolani: norme, bias | anomalie forti: layer iniziali, neuroni anomali, ultimo layer |
| Dove fallisce | tipo e profondità delle matrici | ordine fine dei layer intermedi, attenzione, gruppi di neuroni |

Conclusione: la lettura utile dei pesi passa da **strumenti di calcolo + conoscenza dell'architettura**, non dalla lettura
dei numeri a vista. Prevede bene i **casi estremi** (cosa rompe il modello), male le differenze fini.
Per specializzare un modello su un tema serve quindi l'addestramento (gradiente), non la scrittura manuale dei pesi.
