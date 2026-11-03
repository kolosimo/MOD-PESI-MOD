#!/usr/bin/env python3
"""blind_probe.py - E1: test di lettura "alla cieca" dei pesi.

  make   MODEL --out DIR [--per-class 3] [--len 64]
         Estrae porzioni anonime di pesi. Scrive:
           DIR/probes.md     solo i numeri (è l'unico file da leggere per rispondere)
           DIR/key.json      chiave delle risposte (NON aprire prima di aver risposto)
           DIR/key.sha256    impronta della chiave (si committa insieme alle risposte)
           DIR/train.npz     campioni etichettati per il classificatore di controllo
         Il seme casuale viene da os.urandom e finisce solo dentro key.json.

  score  DIR --answers DIR/answers.json
         Verifica l'impronta della chiave, calcola l'accuratezza delle risposte e la confronta
         con caso, classe più frequente e classificatori su statistiche (k-NN, centroide).

Classi: norm_in, norm_post, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_up, mlp_down, embed,
        bias (q/k/v). Profondità: early (layer 0-7), mid (8-15), late (16-23), n/a (embed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wio import open_weights, read_tensor  # noqa: E402

CLASSES = {
    "norm_in": r"layers\.(\d+)\.input_layernorm\.weight$",
    "norm_post": r"layers\.(\d+)\.post_attention_layernorm\.weight$",
    "attn_q": r"layers\.(\d+)\.self_attn\.q_proj\.weight$",
    "attn_k": r"layers\.(\d+)\.self_attn\.k_proj\.weight$",
    "attn_v": r"layers\.(\d+)\.self_attn\.v_proj\.weight$",
    "attn_o": r"layers\.(\d+)\.self_attn\.o_proj\.weight$",
    "mlp_gate": r"layers\.(\d+)\.mlp\.gate_proj\.weight$",
    "mlp_up": r"layers\.(\d+)\.mlp\.up_proj\.weight$",
    "mlp_down": r"layers\.(\d+)\.mlp\.down_proj\.weight$",
    "embed": r"embed_tokens\.weight$",
    "bias": r"layers\.(\d+)\.self_attn\.[qkv]_proj\.bias$",
}


def depth_bucket(layer, n_layers):
    if layer is None:
        return "n/a"
    f = layer / n_layers
    return "early" if f < 1 / 3 else ("mid" if f < 2 / 3 else "late")


def candidates(wf):
    out = {c: [] for c in CLASSES}
    for name in wf.tensors:
        for c, rx in CLASSES.items():
            m = re.search(rx, name)
            if m:
                layer = int(m.group(1)) if m.groups() else None
                out[c].append((name, layer))
    return out


def sample(wf, name, rng, length):
    t = wf.get(name)
    if len(t.shape) == 1:
        start = int(rng.integers(0, t.shape[0] - length + 1))
        return read_tensor(t)[start:start + length], {"row": None, "col": start}
    row = int(rng.integers(0, t.shape[0]))
    start = int(rng.integers(0, t.shape[1] - length + 1))
    return read_tensor(t, slice(row, row + 1))[0, start:start + length], {"row": row, "col": start}


def features(x):
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std() + 1e-12
    z = (x - m) / s
    return np.array([m, s, np.abs(x).mean(), x.min(), x.max(), np.median(x),
                     (z ** 3).mean(), (z ** 4).mean(), (x > 0).mean(), np.abs(x).max() / s])


def cmd_make(a):
    wf = open_weights(a.model)
    seed = int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)
    cands = candidates(wf)
    n_layers = 1 + max(l for v in cands.values() for _, l in v if l is not None)
    items = []
    for c, lst in cands.items():
        picks = rng.choice(len(lst), size=a.per_class, replace=len(lst) < a.per_class)
        for i in picks:
            name, layer = lst[int(i)]
            vals, pos = sample(wf, name, rng, a.len)
            items.append({"cls": c, "depth": depth_bucket(layer, n_layers), "tensor": name,
                          "layer": layer, **pos, "values": vals})
    order = rng.permutation(len(items))
    items = [items[i] for i in order]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E1 – Porzioni anonime di pesi", "",
             f"Modello: {Path(a.model).name}. {len(items)} porzioni, {a.len} valori consecutivi ciascuna",
             "(un segmento di una riga di una matrice, o di un vettore 1D). Valori in float32, 4 cifre significative.",
             "", "Classi possibili: " + ", ".join(CLASSES) + ".",
             "Profondità: early (layer 0-7), mid (8-15), late (16-23), n/a (embed).", ""]
    for i, it in enumerate(items):
        v = it["values"]
        lines.append(f"## P{i:02d}")
        lines.append("```")
        for r in range(0, len(v), 8):
            lines.append(" ".join(f"{x:+.4g}".rjust(11) for x in v[r:r + 8]))
        lines.append("```")
    (out / "probes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    key = {"seed": seed, "model": str(a.model), "len": a.len,
           "items": [{k: (v if k != "values" else None) for k, v in it.items() if k != "values"}
                     for it in items]}
    kb = json.dumps(key, indent=1).encode()
    (out / "key.json").write_bytes(kb)
    (out / "key.sha256").write_text(hashlib.sha256(kb).hexdigest() + "\n")

    # campioni di addestramento per il classificatore di controllo (righe diverse, stesso modello)
    X, yc, yd = [], [], []
    for c, lst in cands.items():
        for _ in range(a.train_per_class):
            name, layer = lst[int(rng.integers(0, len(lst)))]
            vals, _ = sample(wf, name, rng, a.len)
            X.append(features(vals)); yc.append(c); yd.append(depth_bucket(layer, n_layers))
    np.savez(out / "train.npz", X=np.array(X), yc=np.array(yc), yd=np.array(yd),
             Xp=np.array([features(it["values"]) for it in items]))
    print(f"Scritti {len(items)} probe in {out}/probes.md")
    print(f"Impronta chiave: {(out / 'key.sha256').read_text().strip()}")


def knn(Xtr, ytr, Xte, k=5):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, 1)[:, :k]
    return np.array([Counter(ytr[r]).most_common(1)[0][0] for r in nn])


def centroid(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    labs = sorted(set(ytr))
    C = np.array([A[ytr == l].mean(0) for l in labs])
    return np.array([labs[i] for i in ((B[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)])


def cmd_score(a):
    d = Path(a.dir)
    kb = (d / "key.json").read_bytes()
    ok = hashlib.sha256(kb).hexdigest() == (d / "key.sha256").read_text().strip()
    print(f"Impronta chiave verificata: {'sì' if ok else 'NO – chiave alterata'}")
    key = json.loads(kb)["items"]
    ans = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    ans = {x["id"]: x for x in ans["answers"]}
    n = len(key)
    yc = np.array([k["cls"] for k in key]); yd = np.array([k["depth"] for k in key])
    pc = np.array([ans.get(f"P{i:02d}", {}).get("cls", "") for i in range(n)])
    pd = np.array([ans.get(f"P{i:02d}", {}).get("depth", "") for i in range(n)])
    layered = yd != "n/a"

    tr = np.load(d / "train.npz")
    res = {
        "Claude": ((pc == yc).mean(), (pd[layered] == yd[layered]).mean()),
        "k-NN statistiche": ((knn(tr["X"], tr["yc"], tr["Xp"]) == yc).mean(),
                             (knn(tr["X"][tr["yd"] != "n/a"], tr["yd"][tr["yd"] != "n/a"],
                                  tr["Xp"][layered]) == yd[layered]).mean()),
        "centroide statistiche": ((centroid(tr["X"], tr["yc"], tr["Xp"]) == yc).mean(),
                                  (centroid(tr["X"][tr["yd"] != "n/a"], tr["yd"][tr["yd"] != "n/a"],
                                            tr["Xp"][layered]) == yd[layered]).mean()),
        "caso": (1 / len(CLASSES), 1 / 3),
    }
    print(f"\n{'metodo':<24} {'classe':>8} {'profondità':>11}")
    for k, (c, dd) in res.items():
        print(f"{k:<24} {c:>8.1%} {dd:>11.1%}")

    # classi "grossolane": norm / attn / mlp / embed / bias
    coarse = lambda c: c.split("_")[0]
    print(f"\nClaude, famiglia (norm/attn/mlp/embed/bias): "
          f"{np.mean([coarse(p) == coarse(t) for p, t in zip(pc, yc)]):.1%}  (caso ≈ 20%)")

    print("\nDettaglio:")
    for i in range(n):
        k = key[i]
        mark = "✓" if pc[i] == yc[i] else "✗"
        print(f"  P{i:02d} {mark} risposta={pc[i]:<9} {pd[i]:<6} | vero={yc[i]:<9} {yd[i]:<6} "
              f"{k['tensor']}")
    out = {m: {"classe": float(c), "profondita": float(dd)} for m, (c, dd) in res.items()}
    (d / "score.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("model"); m.add_argument("--out", required=True)
    m.add_argument("--per-class", type=int, default=3); m.add_argument("--len", type=int, default=64)
    m.add_argument("--train-per-class", type=int, default=200)
    m.set_defaults(fn=cmd_make)
    s = sub.add_parser("score"); s.add_argument("dir"); s.add_argument("--answers", required=True)
    s.set_defaults(fn=cmd_score)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
