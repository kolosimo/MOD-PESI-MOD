# Contesto – Lettura e modifica dei pesi di un LLM

Data: 3 novembre 2026. Origine: conversazione vocale tra Delio e Claude.
Questo file serve come contesto stabile per le sessioni di lavoro successive su questo repository.

## Come siamo arrivati al tema

Delio ha chiesto quale fosse una scoperta scientifica recente e importante sulle IA.
Notizie proposte e scartate prima di arrivare a quella giusta:

- Annuncio OpenAI (26 ottobre 2026) su una presunta dimostrazione matematica dell'esistenza di alcune teorie quantistiche dei campi (QCD).
- Studio ICTP di Trieste (team di Jean Barbier, settembre 2026) su come l'IA impara dai dati.
- ScientistTwo, un'IA che prova a fare ricerca scientifica senza ricercatori.
- Notizie Anthropic su Claude (Claude al 26% della R&S di Anthropic, agosto 2026; enzimi simili a CRISPR; limite legato alla funzione zeta di Riemann).
- Modello di IA che capisce come leggono gli esseri umani (eye tracking): scartato, ma giudicato interessante.

Indizi di Delio: capacità emergente; leggere e capire qualcosa che prima non si pensava possibile, cioè i **pesi di altri LLM**.

## La notizia di riferimento: Weight-Space Programming

Come riportata nella conversazione:

- Studio MIT CSAIL: *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*.
- Tesi: gli LLM di frontiera leggono e scrivono direttamente i pesi di reti neurali, senza discesa del gradiente e senza modelli ausiliari addestrati.
- Modello testato soprattutto: GPT-6 Astra; in otto esperimenti ha capito per cosa erano state addestrate piccole reti.
- Applicazione: modifiche mirate ai pesi di Qwen per ridurre i loop di ragionamento.
- Simon Willison (3 novembre 2026): anche Qwen 3.8 27B, in locale su laptop, riuscirebbe a interpretare i pesi di un modello.

> **Stato di verifica (3/11/2026, sessione cloud):** una ricerca web sul titolo esatto **non ha trovato** il paper.
> Esiste letteratura vicina (weight-space learning, interpretabilità di singoli parametri in transformer sparsi,
> "weight oracles"), ma nessuna fonte trovata conferma lo studio, GPT-6 Astra, Qwen 3.8 27B o il post di Willison.
> Trattare tutti questi punti come **riportati, non verificati**. Per verificarli: link arXiv, pagina MIT CSAIL,
> post originale di Simon Willison (simonwillison.net) e di Sean Goedecke.

## Limiti e cautele

- Studio (se esiste) non ancora sottoposto a revisione, basato su un solo LLM.
- Critica di Sean Goedecke: la capacità sembra superumana, ma secondo lui non lo è.
- Gli esperimenti riguardano soprattutto reti piccole, non LLM interi.
- Non è dimostrato che Claude abbia questa capacità: va misurato, non assunto.

## Obiettivo di Delio

- Non replicare l'esperimento su reti minuscole.
- Capire concretamente se Claude riesce a **leggere i pesi di un modello vero** e se insieme si riesce a **modificarli** con effetto misurabile.
- Modello open già scaricato in una cartella locale di Delio (nome/formato da definire).

## Piano operativo deciso

- Uso dei crediti cloud (limite di utilizzo locale fino al giorno dopo, ore 9).
- Repository GitHub `MOD-PESI-MOD` con dentro i pesi del modello.
- Questo documento = contesto per le sessioni successive.

## Punti tecnici aperti

- GitHub rifiuta file > 100 MB senza Git LFS; i pesi pesano GB. → vedi `docs/CARICAMENTO_PESI.md`.
- Miliardi di parametri: leggerli tutti direttamente non è realistico → lavorare su layer, tensori, porzioni.
- Da definire: quale modello, quale formato (safetensors / GGUF), quale modifica si vuole ottenere.
