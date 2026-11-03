#!/usr/bin/env python3
"""edit_weights.py - modifica mirata e tracciata di un tensore in un file di pesi.

Principi:
  - Il file originale NON viene mai toccato: si lavora su una copia (--out).
  - Si modifica un solo tensore per invocazione, eventualmente su un sotto-blocco
    (--rows / --cols). L'header del file resta identico, quindi il modello resta
    caricabile da transformers / llama.cpp senza conversioni.
  - Ogni modifica produce un log JSON (logs/) con statistiche prima/dopo, così
    l'intervento è ripetibile e reversibile.

Formati scrivibili: safetensors e GGUF con tensori F32 / F16 / BF16.
(I tensori quantizzati Q4/Q5/K-quants non sono modificabili con questo tool.)

Operazioni (--op):
  scale=K        moltiplica per K
  add=K          somma K
  zero           azzera
  noise=S        somma rumore gaussiano con deviazione standard S (seed --seed)
  clamp=A,B      limita i valori in [A, B]
  copy_from=NAME sostituisce con i valori dello stesso blocco di un altro tensore
                 (stessa shape del blocco selezionato)

Esempio:
  python tools/edit_weights.py models/m.safetensors --out models/m_mod.safetensors \\
      --tensor model.layers.20.mlp.down_proj.weight --cols 0:128 --op scale=0.5 \\
      --note "ridotto contributo neuroni 0-127 layer 20"
Per applicare più modifiche in sequenza, usa come input l'output precedente e
passa --inplace-out (modifica direttamente il file --out già esistente).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wio import dtype_size, encode_like, open_weights, read_tensor  # noqa: E402
from inspect_weights import parse_slice, tensor_stats  # noqa: E402


def apply_op(x: np.ndarray, op: str, seed: int, wf, block_sel) -> np.ndarray:
    name, _, arg = op.partition("=")
    if name == "scale":
        return x * float(arg)
    if name == "add":
        return x + float(arg)
    if name == "zero":
        return np.zeros_like(x)
    if name == "noise":
        rng = np.random.default_rng(seed)
        return x + rng.normal(0.0, float(arg), size=x.shape).astype(np.float32)
    if name == "clamp":
        a, b = (float(v) for v in arg.split(","))
        return np.clip(x, a, b)
    if name == "copy_from":
        src = read_tensor(wf.get(arg))[block_sel]
        if src.shape != x.shape:
            raise ValueError(f"shape diverse: {src.shape} vs {x.shape}")
        return src.astype(np.float32)
    raise ValueError(f"Operazione sconosciuta: {op}")


def sha(b: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()[:16]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="file di pesi sorgente (singolo .safetensors o .gguf)")
    p.add_argument("--out", required=True, help="file di destinazione (copia modificata)")
    p.add_argument("--inplace-out", action="store_true",
                   help="non ricopiare: modifica direttamente --out (che deve già esistere)")
    p.add_argument("--tensor", required=True)
    p.add_argument("--rows"); p.add_argument("--cols")
    p.add_argument("--op", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent.parent / "logs"))
    p.add_argument("--dry-run", action="store_true", help="calcola e mostra, senza scrivere")
    args = p.parse_args(argv)

    src, out = Path(args.file), Path(args.out)
    if src.is_dir():
        sys.exit("Passa il singolo shard che contiene il tensore, non la cartella.")
    if src.resolve() == out.resolve():
        sys.exit("--out deve essere diverso dal file sorgente.")

    if args.inplace_out and not out.exists():
        sys.exit("--inplace-out richiede che --out esista già.")
    # Con --inplace-out la base è il file già modificato, altrimenti le modifiche
    # precedenti verrebbero sovrascritte dai valori originali.
    wf_src = open_weights(out if args.inplace_out else src)
    t_src = wf_src.get(args.tensor)
    if t_src.dtype not in ("F32", "F16", "BF16", "F64"):
        sys.exit(f"dtype {t_src.dtype} non modificabile (solo F32/F16/BF16).")

    full = read_tensor(t_src)
    if full.ndim == 1:
        sel = (parse_slice(args.cols) or parse_slice(args.rows) or slice(None),)
    else:
        sel = (parse_slice(args.rows) or slice(None), parse_slice(args.cols) or slice(None))
    before = full[sel].copy()
    after = apply_op(before, args.op, args.seed, wf_src, sel).astype(np.float32)
    new_full = full.copy()
    new_full[sel] = after
    encoded = encode_like(t_src, new_full).reshape(-1)

    st_before, st_after = tensor_stats(before), tensor_stats(after)
    print(f"Tensore {t_src.name} {t_src.dtype}{list(t_src.shape)}  blocco {list(before.shape)}")
    print(f"  op={args.op}")
    for k in ("mean", "std", "rms", "min", "max"):
        print(f"  {k:<5} prima={st_before[k]:.5g}  dopo={st_after[k]:.5g}")
    if args.dry_run:
        print("dry-run: nessun file scritto.")
        return

    if not args.inplace_out:
        print(f"Copio {src} -> {out} …")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    elif not out.exists():
        sys.exit("--inplace-out richiede che --out esista già.")

    wf_out = open_weights(out)
    t_out = wf_out.get(args.tensor)
    if (t_out.dtype, t_out.shape, t_out.nbytes) != (t_src.dtype, t_src.shape, t_src.nbytes):
        sys.exit("Il tensore in --out non coincide con quello sorgente: interrompo.")
    mm = np.memmap(out, dtype=encoded.dtype, mode="r+", offset=t_out.offset,
                   shape=(t_out.nbytes // dtype_size(t_out.dtype),))
    old_hash = sha(mm)
    mm[:] = encoded
    mm.flush()
    del mm

    read_back = read_tensor(open_weights(out).get(args.tensor))
    stored_after = read_back[sel]
    max_err = float(np.abs(stored_after - after).max())
    print(f"Scritto. Errore max di arrotondamento al dtype {t_src.dtype}: {max_err:.3g}")

    log = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(src), "output": str(out), "tensor": args.tensor,
        "dtype": t_src.dtype, "shape": list(t_src.shape),
        "rows": args.rows, "cols": args.cols, "op": args.op, "seed": args.seed,
        "note": args.note, "hash_before": old_hash,
        "hash_after": sha(encoded), "rounding_max_err": max_err,
        "stats_before": st_before, "stats_after": st_after,
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    lp = log_dir / f"edit_{stamp}_{args.tensor.replace('/', '_')}.json"
    lp.write_text(json.dumps(log, indent=2, default=float))
    print(f"Log: {lp}")


if __name__ == "__main__":
    main()
