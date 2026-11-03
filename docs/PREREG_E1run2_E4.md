# Pre-registrazione: E1 run2 ed E4 (scritta prima di vedere i dati)

Data: 3/11/2026. Modello: Qwen/Qwen2.5-0.5B-Instruct (BF16, SHA-256 `fdf756fa…`).
Regole fissate prima di generare le porzioni di E1 run2 e prima di qualsiasi ablazione di E4.

## E1 run2 – Lettura alla cieca, porzioni più ricche
- 33 porzioni (3 per ciascuna delle 11 classi), ogni porzione = **4 righe diverse × 32 valori** dello stesso
  tensore (vettori 1D: 4 segmenti diversi), 3 cifre significative.
  Le righe complete (896–4864 valori × 5) non sono leggibili in un turno: 33×5×896 ≈ 150.000 numeri.
- Stesso protocollo di run1: seme da `os.urandom`, chiave sigillata con SHA-256, risposte committate prima della rivelazione,
  nessun codice eseguito sui valori delle porzioni.
- Nota di equità: rispetto a run1 ho visto la chiave del run1 (feedback). Anche i classificatori sono addestrati
  sullo stesso modello, quindi il confronto resta equo.
- **Criterio di successo**: accuratezza sulla classe ≥ miglior classificatore su statistiche **+ 15 punti**.
  Profondità: successo se ≥ miglior classificatore + 15 punti.
- Un solo tentativo: niente run3 per "riprovare" se fallisce.

## E4 – Ablazione: prevedere l'effetto leggendo i pesi
Testo di valutazione: `prompts/eval_wikitext2.txt` (primi ~12.000 caratteri del test set wikitext-2-raw-v1, 2.861 token).

### E4a – Ablazione per layer
- Per ciascuno dei 24 layer: (1) MLP azzerata (`down_proj = 0`), (2) attenzione azzerata (`o_proj = 0`).
  Misura: Δ perplexity rispetto alla base. 48 misure.
- Procedura: calcolo le statistiche dei pesi (`ablation.py stats`), scrivo e committo **una classifica prevista**
  del danno per MLP e per attenzione (con le motivazioni), *poi* misuro.
- **Criterio di successo**: correlazione di Spearman tra classifica prevista e Δ misurato **ρ ≥ 0,5**,
  separatamente per MLP e attenzione.
- Confronto obbligatorio con un predittore ingenuo: solo norma di Frobenius di `down_proj` (MLP) e di `o_proj` (attenzione).

### E4b – Ablazione di gruppi di neuroni nel layer 12
- Layer scelto ora: **12** (metà modello). Gruppi da 64 neuroni MLP (76 gruppi).
- Punteggio per gruppo da `stats` (‖gate_i‖·‖up_i‖·‖down_:,i‖ sommato sul gruppo).
  Ablo: i 5 gruppi con punteggio più alto, i 5 più bassi, 5 casuali (seme 12345 fissato qui).
- **Criterio di successo**: Δ medio dei "top-5" ≥ **2×** Δ medio dei "casuali-5", e Δ medio dei "bottom-5" ≤ "casuali-5".

### Verifica della catena di scrittura
- L'ablazione MLP del layer con Δ maggiore viene rifatta scrivendo il file con `edit_weights.py`
  e misurata con `ablation.py ppl --weights`: deve dare la stessa perplexity (±0,1%).

## Confronto finale
Lettura nativa (E1) vs lettura strumentale + conoscenza (E4): quale dei due produce previsioni utili e verificabili.
