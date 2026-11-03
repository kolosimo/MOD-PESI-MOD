# Piano: specializzare un piccolo LLM open su prompt / loop / context engineering

Obiettivo (Delio, 3/11/2026): prendere il modello open più piccolo possibile e **addestrarlo** (modificandone i pesi)
perché diventi specialista di **prompt engineering, loop engineering e context engineering**, seguendo le best practice.

Lezione dagli esperimenti E1–E4 (`docs/RISULTATI.md`): la scrittura manuale dei pesi serve per interventi stretti
(spegnere un token, rompere/proteggere un layer); per insegnare una competenza serve l'**addestramento** (gradiente).
Il fine-tuning *è* una modifica dei pesi: alla fine si ottiene un nuovo `model.safetensors`, confrontabile con
`inspect_weights.py diff` (si vedrà quali layer ha cambiato l'addestramento).

## Definizioni di lavoro (da confermare)
- **Prompt engineering**: scrivere, criticare e migliorare istruzioni per LLM (ruolo, obiettivo, vincoli, esempi, formato d'uscita).
- **Loop engineering**: progettare cicli agentici: pianifica → agisci (tool) → osserva → verifica, con criteri di stop,
  gestione errori/retry, auto-verifica, limiti di costo. *Termine non standardizzato: definizione da concordare.*
- **Context engineering**: decidere cosa mettere nel contesto e come: selezione e ordinamento delle informazioni, budget di token,
  riassunti/compattazione, memoria, recupero (RAG), istruzioni di sistema vs dati.

## Modelli candidati (verificati su Hugging Face il 3/11/2026)
| Modello | Parametri | Licenza | Lingue | Note |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M-Instruct | 134,5 M | Apache-2.0 | inglese | il più piccolo; qualità di ragionamento molto limitata |
| google/gemma-3-270m-it | 268 M | Gemma (gated) | multilingue | accesso da richiedere; licenza con restrizioni |
| HuggingFaceTB/SmolLM2-360M-Instruct | 361,8 M | Apache-2.0 | inglese | |
| **Qwen/Qwen2.5-0.5B-Instruct** | 494 M | Apache-2.0 | multilingue (risponde in italiano) | già scaricato, baseline e mappa dei layer già misurate (E0, E4) |
| Qwen/Qwen3-0.6B | 751,6 M | Apache-2.0 | multilingue | modalità "thinking"; più grande |

Raccomandazione: **pilota su Qwen2.5-0.5B-Instruct**; SmolLM2-135M come confronto "limite inferiore" solo se l'uso è in inglese.

## Fasi (best practice)
1. **Valutazione prima dei dati**
   - 60–100 compiti di prova *tenuti fuori dall'addestramento*, bilanciati sui 3 temi (es. "migliora questo prompt",
     "progetta un loop con criteri di stop", "riduci questo contesto a 2.000 token mantenendo X").
   - Rubrica di valutazione per tema; giudizio a coppie (modello A vs B) con un modello più forte come giudice + controllo umano su un campione.
   - **Baseline obbligatorie**: modello base; modello base + buon prompt di sistema da esperto (baseline forte: se il fine-tuning
     non la batte, non serve); regressione generale (perplexity wikitext + prompt generali già esistenti).
   - Criteri di successo fissati prima: es. ≥ 60% di vittorie contro "base + prompt di sistema"; perplexity generale ≤ +10%.
2. **Dati (la parte che decide il risultato)**
   - 1.500–5.000 esempi istruzione → risposta, di alta qualità, nel formato chat del modello.
   - Tassonomia: tema × tipo di compito × difficoltà × dominio applicativo, per garantire varietà.
   - Fonti: generazione sintetica con un modello forte + filtri (deduplica, giudice automatico, revisione umano del 10%);
     fonti scritte da persone solo con licenza compatibile.
   - **Verificare i termini d'uso del modello usato per generare i dati**: alcuni fornitori limitano l'uso degli output
     per addestrare altri modelli (non verificato qui).
   - 10–20% di dati generali in mix per limitare l'"oblio" delle capacità generali; decontaminazione rispetto al set di prova.
3. **Addestramento**
   - SFT con LoRA (TRL `SFTTrainer`), loss solo sulle risposte dell'assistente, template chat del modello base.
   - Punti di partenza tipici (da tarare): rank 16–32, alpha 2×rank, dropout 0,05, lr 1e-4–2e-4 cosine, warmup 3–5%,
     2–3 epoche, sequenze 1.024–2.048 token. Validazione a ogni epoca, fermarsi se la loss di validazione risale.
   - Fusione dell'adattatore nei pesi → nuovo `model.safetensors`; `inspect_weights.py diff` per vedere cosa è cambiato.
   - Opzionale dopo SFT: DPO su coppie "risposta buona / risposta peggiore".
4. **Valutazione e iterazione**: stesse prove, stesse soglie; analisi degli errori → correggere i dati, non solo gli iperparametri.
5. **Rilascio**: model card (dati, limiti, valutazioni), versione, conversione GGUF per uso locale.

## Calcolo (stima, da verificare con una prova)
- Misurato in questa sessione (CPU 4 core): un passaggio in avanti di Qwen2.5-0.5B su 2.861 token ≈ 15 s.
- Addestramento ≈ 3× il passaggio in avanti → ~45 s ogni ~2.900 token → **~4–5 ore per epoca con 1 M di token** su CPU.
  SmolLM2-135M ≈ 3–4× più veloce (ipotesi). Con una GPU (anche gratuita, es. Colab/Kaggle) si scende a minuti.

## Rischi principali
- Un modello da 0,1–0,5 B impara bene formato e lessico, molto meno il giudizio: aspettarsi risposte "ben impaginate" ma superficiali.
- Il campo cambia in fretta: la conoscenza nei pesi invecchia → valutare anche "modello piccolo + documenti recuperati".
- Dati sintetici omogenei → modello ripetitivo: la varietà della tassonomia conta più del numero di esempi.
