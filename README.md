# MOD-PESI-MOD

Esperimento: verificare se Claude riesce a **leggere** i pesi di un LLM open reale e, insieme a Delio,
a **modificarli** con effetti prevedibili e misurati.

- Contesto e origine: [`CONTEXT.md`](CONTEXT.md)
- Piano esperimenti: [`docs/PIANO_ESPERIMENTI.md`](docs/PIANO_ESPERIMENTI.md)
- Come caricare i pesi: [`docs/CARICAMENTO_PESI.md`](docs/CARICAMENTO_PESI.md)
- **Risultati**: [`docs/RISULTATI.md`](docs/RISULTATI.md)
- **Piano di specializzazione** (obiettivo attuale): [`docs/PIANO_SPECIALIZZAZIONE.md`](docs/PIANO_SPECIALIZZAZIONE.md)

## Struttura

```
models/   pesi scaricati da Hugging Face (ignorati da git; si riscaricano)
tools/    wio.py (lettura safetensors/GGUF senza torch), inspect_weights.py, edit_weights.py,
          baseline.py (esegue il modello: prompt fissi + perplexity, confronto run)
prompts/  prompt fissi e testo per la perplexity
tests/    test dei tool contro le librerie ufficiali safetensors / gguf
logs/     log JSON di ogni modifica ai pesi
```

## Uso rapido

Dipendenze dei tool: solo `numpy`.

```bash
python tools/inspect_weights.py summary models/<modello>/
```
```bash
python tools/inspect_weights.py list models/<modello>/ --filter "layers\.0\."
```
```bash
python tools/inspect_weights.py stats models/<modello>/ model.layers.10.mlp.down_proj.weight
```
```bash
python tools/inspect_weights.py dump models/<modello>/ model.norm.weight --cols 0:64
```
```bash
python tools/edit_weights.py models/<modello>/model.safetensors --out models/mod/model.safetensors --tensor lm_head.weight --rows 1234:1235 --op scale=0 --note "sopprime token 1234"
```
```bash
python tools/inspect_weights.py diff models/<modello>/model.safetensors models/mod/model.safetensors
```

`edit_weights.py` non tocca mai il file originale, mantiene identico l'header (il modello resta
caricabile da transformers / llama.cpp) e scrive un log JSON con statistiche prima/dopo.

## Supporto formati

| Formato | Lettura | Modifica |
|---|---|---|
| safetensors F32/F16/BF16 (file, shard, index.json) | sì | sì (per shard) |
| GGUF F32/F16/BF16 | sì | sì |
| GGUF Q8_0 / Q4_0 | sì (dequantizzati) | no |
| GGUF K-quants / IQ | solo elenco/metadati | no |

## Far girare il modello

```bash
pip install -r requirements-run.txt
```
```bash
python tools/baseline.py run models/qwen2.5-0.5b-instruct --out logs/baseline/originale.json
```
```bash
python tools/baseline.py run models/qwen2.5-0.5b-instruct --weights models/mod/x.safetensors --out logs/x.json
```
```bash
python tools/baseline.py compare logs/baseline/originale.json logs/x.json
```

## Test

```bash
pip install numpy safetensors gguf ml_dtypes && python tests/test_tools.py
```
