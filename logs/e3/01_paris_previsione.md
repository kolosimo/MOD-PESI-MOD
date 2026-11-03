# E3-01 – Soppressione del token "Paris" (previsione scritta PRIMA della misura)

Modifica: righe 59604 ("Paris") e 12095 (" Paris") di model.embed_tokens.weight moltiplicate per 0.
Il modello ha embedding legati: la stessa riga è sia l'input del token sia il vettore di output.

Previsioni:
1. Prompt 'Write the word "Paris" five times.': il token intero "Paris" non viene più generato
   (logit = h·0 = 0, battuto da token con logit positivo). Il modello o lo spezza in sotto-token
   (es. "Par"+"is"), o sostituisce la parola, o sbaglia il compito perché nel prompt "Paris" arriva
   come vettore nullo (non "vede" quale parola ripetere).
2. Prompt sulla capitale della Francia: risposta invariata o quasi ("Parigi" usa altri token).
3. Gli altri 8 prompt: output identico.
4. Perplexity sul testo di riferimento (che non contiene "Paris"): variazione < 0,5%.
