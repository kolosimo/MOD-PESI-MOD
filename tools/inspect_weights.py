#!/usr/bin/env python3
"""inspect_weights.py - esplorazione di file di pesi LLM (safetensors / GGUF).

Comandi:
  summary  FILE                      formato, metadati, n. parametri, dtype, struttura a layer
  list     FILE [--filter REGEX]     elenco tensori (nome, dtype, shape, n. elementi)
  stats    FILE TENSOR [--rows a:b]  statistiche (min/max/media/std, sparsità, percentili,
                                     norme di riga, outlier)
  dump     FILE TENSOR [--rows a:b] [--cols c:d] [--precision N]
                                     stampa una porzione di valori (per leggerli "a vista")
  layers   FILE [--pattern REGEX]    statistiche compatte per tutti i tensori che matchano
                                     (es. confronto norme tra layer)
  diff     FILE_A FILE_B [--filter REGEX]
                                     confronto tensore per tensore (verifica delle modifiche)

FILE può essere un .safetensors, una cartella di shard, un *.index.json o un .gguf.
Esempi:
  python tools/inspect_weights.py summary models/qwen/
  python tools/inspect_weights.py stats models/qwen/ model.layers.10.mlp.down_proj.weight
  python tools/inspect_weights.py dump models/qwen/ model.norm.weight --cols 0:32
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wio import open_weights, read_tensor  # noqa: E402


def parse_slice(s: str | None) -> slice | None:
    if not s:
        return None
    a, _, b = s.partition(":")
    return slice(int(a) if a else None, int(b) if b else None)


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.2f}P"


def layer_key(name: str) -> str:
    """Sostituisce l'indice di layer con '*' per raggruppare tensori omologhi."""
    return re.sub(r"(\.|^)(\d+)(\.)", r"\1*\3", name, count=1)


# --------------------------------------------------------------------------- #

def cmd_summary(args):
    wf = open_weights(args.file)
    print(f"Formato: {wf.fmt}   file: {len(wf.paths)}")
    if wf.metadata:
        print("\nMetadati:")
        for k, v in wf.metadata.items():
            s = str(v)
            print(f"  {k}: {s[:160]}{'…' if len(s) > 160 else ''}")
    total = sum(t.numel for t in wf.tensors.values())
    by_dtype = Counter()
    for t in wf.tensors.values():
        by_dtype[t.dtype] += t.numel
    print(f"\nTensori: {len(wf.tensors)}   parametri: {human(total)} ({total:,})")
    print("Per dtype: " + ", ".join(f"{d}={human(n)}" for d, n in by_dtype.most_common()))
    nondec = [t.name for t in wf.tensors.values() if not t.decodable]
    if nondec:
        print(f"ATTENZIONE: {len(nondec)} tensori in formato non decodificabile da questi tool.")

    idx = set()
    for name in wf.tensors:
        m = re.search(r"(?:layers|blk|h|block|blocks)\.(\d+)\.", name)
        if m:
            idx.add(int(m.group(1)))
    if idx:
        print(f"Blocchi transformer rilevati: {len(idx)} (indici {min(idx)}..{max(idx)})")

    groups = defaultdict(list)
    for t in wf.tensors.values():
        groups[layer_key(t.name)].append(t)
    print("\nStruttura (tensori raggruppati per layer):")
    for key in sorted(groups, key=lambda k: (k.count("*") == 0, k)):
        ts = groups[key]
        t0 = ts[0]
        print(f"  {key:<60} x{len(ts):<4} {t0.dtype:<6} {str(list(t0.shape)):<20} "
              f"{human(t0.numel * len(ts))}")


def cmd_list(args):
    wf = open_weights(args.file)
    rx = re.compile(args.filter) if args.filter else None
    for t in wf.tensors.values():
        if rx and not rx.search(t.name):
            continue
        flag = "" if t.decodable else "  [non decodificabile]"
        print(f"{t.name:<70} {t.dtype:<6} {str(list(t.shape)):<22} {t.numel:>14,}{flag}")


def tensor_stats(x: np.ndarray) -> dict:
    f = x.astype(np.float32).reshape(-1)
    a = np.abs(f)
    p = np.percentile(f, [0.1, 1, 25, 50, 75, 99, 99.9])
    st = {
        "numel": f.size, "min": float(f.min()), "max": float(f.max()),
        "mean": float(f.mean()), "std": float(f.std()), "abs_mean": float(a.mean()),
        "rms": float(np.sqrt((f * f).mean())),
        "zeros_pct": float((f == 0).mean() * 100),
        "nan": int(np.isnan(f).sum()), "inf": int(np.isinf(f).sum()),
        "p0.1": p[0], "p1": p[1], "p25": p[2], "p50": p[3], "p75": p[4],
        "p99": p[5], "p99.9": p[6],
    }
    std = st["std"] or 1.0
    st["outliers_6sigma"] = int((np.abs(f - st["mean"]) > 6 * std).sum())
    if f.size > 3:
        st["kurtosis"] = float((((f - st["mean"]) / std) ** 4).mean() - 3)
    return st


def cmd_stats(args):
    wf = open_weights(args.file)
    t = wf.get(args.tensor)
    x = read_tensor(t, parse_slice(args.rows))
    print(f"{t.name}  dtype={t.dtype}  shape={list(t.shape)}  letto={list(x.shape)}")
    for k, v in tensor_stats(x).items():
        print(f"  {k:<16} {v:.6g}" if isinstance(v, float) else f"  {k:<16} {v}")
    if x.ndim == 2:
        rn = np.linalg.norm(x, axis=1)
        cn = np.linalg.norm(x, axis=0)
        print(f"\n  norme di riga    min={rn.min():.4g} med={np.median(rn):.4g} max={rn.max():.4g}"
              f"  (riga max: {int(rn.argmax())})")
        print(f"  norme di colonna min={cn.min():.4g} med={np.median(cn):.4g} max={cn.max():.4g}"
              f"  (colonna max: {int(cn.argmax())})")
        top = np.argsort(cn)[::-1][: args.top]
        print(f"  colonne con norma più alta: {top.tolist()}")
    flat = np.abs(x.reshape(-1))
    k = min(args.top, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    print(f"\n  top-{k} |valori|:")
    for i in idx:
        pos = np.unravel_index(i, x.shape)
        print(f"    {tuple(int(p) for p in pos)} = {x.reshape(-1)[i]:.6g}")


def cmd_dump(args):
    wf = open_weights(args.file)
    t = wf.get(args.tensor)
    x = read_tensor(t, parse_slice(args.rows) or slice(0, min(8, t.shape[0] if t.shape else 1)))
    cs = parse_slice(args.cols)
    if cs is not None and x.ndim >= 2:
        x = x[:, cs]
    elif cs is not None and x.ndim == 1:
        x = x[cs]
    np.set_printoptions(precision=args.precision, suppress=False, linewidth=160,
                        threshold=args.max_values, edgeitems=8)
    print(f"{t.name}  dtype={t.dtype}  shape={list(t.shape)}  porzione={list(x.shape)}")
    print(x)


def cmd_layers(args):
    wf = open_weights(args.file)
    rx = re.compile(args.pattern)
    print(f"{'tensore':<60} {'mean':>10} {'std':>10} {'rms':>10} {'absmax':>10} {'kurt':>8}")
    for t in wf.tensors.values():
        if not rx.search(t.name) or not t.decodable:
            continue
        s = tensor_stats(read_tensor(t))
        print(f"{t.name:<60} {s['mean']:>10.4g} {s['std']:>10.4g} {s['rms']:>10.4g} "
              f"{max(abs(s['min']), abs(s['max'])):>10.4g} {s.get('kurtosis', 0):>8.3g}")


def cmd_diff(args):
    a, b = open_weights(args.file_a), open_weights(args.file_b)
    rx = re.compile(args.filter) if args.filter else None
    only_a = set(a.tensors) - set(b.tensors)
    only_b = set(b.tensors) - set(a.tensors)
    if only_a:
        print(f"Solo in A: {sorted(only_a)}")
    if only_b:
        print(f"Solo in B: {sorted(only_b)}")
    changed = 0
    for name in a.tensors:
        if name not in b.tensors or (rx and not rx.search(name)):
            continue
        ta, tb = a.tensors[name], b.tensors[name]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            print(f"{name}: shape/dtype diversi {ta.dtype}{list(ta.shape)} vs {tb.dtype}{list(tb.shape)}")
            changed += 1
            continue
        if not ta.decodable:
            continue
        xa, xb = read_tensor(ta), read_tensor(tb)
        d = xb - xa
        n = int((d != 0).sum())
        if n:
            changed += 1
            rel = np.linalg.norm(d) / (np.linalg.norm(xa) or 1)
            print(f"{name}: {n:,} valori cambiati ({n / d.size:.2%}), "
                  f"max|Δ|={np.abs(d).max():.4g}, ‖Δ‖/‖A‖={rel:.4g}")
    print(f"\nTensori modificati: {changed}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary"); s.add_argument("file"); s.set_defaults(fn=cmd_summary)
    s = sub.add_parser("list"); s.add_argument("file"); s.add_argument("--filter")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("stats"); s.add_argument("file"); s.add_argument("tensor")
    s.add_argument("--rows"); s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("dump"); s.add_argument("file"); s.add_argument("tensor")
    s.add_argument("--rows"); s.add_argument("--cols")
    s.add_argument("--precision", type=int, default=4)
    s.add_argument("--max-values", type=int, default=4096)
    s.set_defaults(fn=cmd_dump)
    s = sub.add_parser("layers"); s.add_argument("file"); s.add_argument("--pattern", default=".")
    s.set_defaults(fn=cmd_layers)
    s = sub.add_parser("diff"); s.add_argument("file_a"); s.add_argument("file_b")
    s.add_argument("--filter"); s.set_defaults(fn=cmd_diff)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
