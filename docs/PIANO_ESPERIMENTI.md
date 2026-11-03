# Piano esperimenti

Obiettivo: rispondere con dati a due domande, separate.

1. **Lettura** – Claude riesce a capire qualcosa di un modello reale guardando i valori dei pesi?
2. **Scrittura** – Claude e Delio riescono a modificare i pesi ottenendo un effetto *previsto prima* e *misurato dopo*?

## Distinzione da tenere ferma

| Modalità | Cosa significa | Cosa dimostra |
|---|---|---|
| **Lettura strumentale** | Claude scrive codice che calcola statistiche/norme/similarità, poi interpreta i risultati | Che Claude sa fare analisi dei pesi (lo fa già qualunque buon ingegnere ML) |
| **Lettura nativa** | Claude riceve numeri grezzi *senza nome del tensore* e deve dire cosa sono | È la tesi dello studio: la parte nuova e incerta |

Ordine di grandezza: un modello da 0,5 B ha ~500 milioni di valori; in un turno Claude può leggere
realisticamente qualche migliaio–decina di migliaia di numeri. La lettura nativa riguarda quindi
**porzioni** (vettori di norma, bias, righe di embedding, blocchi di matrici), mai il modello intero.

## Prerequisiti

- Pesi accessibili nella sessione cloud (vedi `CARICAMENTO_PESI.md`).
- Formato consigliato: **safetensors BF16/F16** (leggibile e modificabile con `tools/`).
  GGUF va bene per la lettura se F16/BF16/Q8_0/Q4_0; i K-quants (Q4_K_M ecc.) non sono decodificati dai tool.
- Per misurare gli effetti serve far girare il modello: `torch` (CPU) + `transformers` per safetensors,
  oppure `llama.cpp` per GGUF. Da installare al momento (PyPI raggiungibile dalla sessione cloud).

## Esperimenti (in ordine di difficoltà)

### E0 – Inventario e baseline
- `inspect_weights.py summary` → architettura, n. layer, dtype, tensori legati (embedding = lm_head?).
- Baseline di generazione: 10–20 prompt fissi, decodifica greedy, output salvati in `logs/baseline/`.
- Criterio di uscita: modello caricato, output baseline riproducibili.

### E1 – Lettura nativa "alla cieca" (test della tesi)
- Uno script estrae N porzioni (es. 30) da tensori diversi: vettori RMSNorm di layer iniziali/finali,
  righe di q/k/v/o, gate/up/down, bias (se presenti), righe di embedding.
- Claude riceve solo i numeri (stessa lunghezza, nome nascosto) e deve classificare:
  tipo di tensore, posizione nel modello (inizio/metà/fine).
- Si confronta con la verità → accuratezza vs. caso (baseline casuale nota).
- Variante di controllo: stessa prova data a un semplice classificatore su statistiche (media, std,
  curtosi). Se Claude non fa meglio del classificatore, non c'è "lettura" oltre la statistica.

### E2 – Embedding dei token
- Righe di embedding di token scelti (es. numeri, nomi di città, parole italiane/inglesi).
- Claude stima a vista quali token sono simili; il tool calcola la similarità coseno reale.

### E3 – Prima modifica con effetto prevedibile
- Ridurre la riga di `lm_head` (unembedding) di un token preciso → il modello smette di generarlo.
- Effetto atteso dichiarato *prima*, misurato con i prompt della baseline.
- Attenzione: nei modelli piccoli con **embedding legati** la stessa matrice è anche l'input;
  la modifica cambia anche come il token viene "letto". Verificare in E0.

### E4 – Ablazione mirata
- Azzerare/scalare gruppi di neuroni MLP (colonne di `down_proj`) o una testa di attenzione
  (blocco di colonne di `o_proj`) in un layer → misurare la perplexity su un testo fisso.
- Domanda di lettura: Claude, guardando i pesi, sa prevedere *quale* ablazione farà più danno?

### E5 – Modifica comportamentale (stile "ridurre i loop di ragionamento")
- Richiede attivazioni (far girare il modello), non solo pesi: steering vector o modifica a rango 1.
- Da affrontare solo se E1–E4 danno risultati solidi.

## Regole di lavoro

- Mai modificare l'originale: `edit_weights.py` scrive sempre una copia + log JSON in `logs/`.
- Una modifica alla volta, effetto atteso scritto prima di misurarlo.
- Risultati negativi vanno registrati quanto quelli positivi.
