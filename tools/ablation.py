#!/usr/bin/env python3
"""ablation.py - E4: ablazioni mirate e misura dell'effetto sulla perplexity.

  stats   MODEL --out FILE.json
          Statistiche dei pesi per layer (lettura strumentale, solo numpy): norme di
          gate/up/down, o_proj, v_proj, pesi di normalizzazione; punteggi per gruppi di neuroni MLP.
  layers  MODEL --text TXT --out FILE.json
          Perplexity di base + per ogni layer: MLP azzerata (down_proj = 0) e
          attenzione azzerata (o_proj = 0). Misura Δ rispetto alla base.
  groups  MODEL --layer L --groups 3,17,... --text TXT --out FILE.json [--size 64]
          Azzera singolarmente gruppi di neuroni MLP (righe di gate/up + colonne di down) nel layer L.
  ppl     MODEL --text TXT [--weights FILE.safetensors]
          Sola perplexity (per verificare che un'ablazione scritta su file dia lo stesso numero).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- statistiche

def cmd_stats(a):
    from wio import open_weights, read_tensor
    wf = open_weights(a.model)
    n = 1 + max(int(k.split(".")[2]) for k in wf.tensors if k.startswith("model.layers."))
    rows = []
    groups = {}
    for L in range(n):
        p = f"model.layers.{L}."
        g = read_tensor(wf.get(p + "mlp.gate_proj.weight"))
        u = read_tensor(wf.get(p + "mlp.up_proj.weight"))
        d = read_tensor(wf.get(p + "mlp.down_proj.weight"))
        o = read_tensor(wf.get(p + "self_attn.o_proj.weight"))
        v = read_tensor(wf.get(p + "self_attn.v_proj.weight"))
        ln_in = read_tensor(wf.get(p + "input_layernorm.weight"))
        ln_post = read_tensor(wf.get(p + "post_attention_layernorm.weight"))
        # punteggio per neurone: |gate_i| * |up_i| * |down_:,i|  (quanto può "passare" dal neurone i)
        neuron = np.linalg.norm(g, axis=1) * np.linalg.norm(u, axis=1) * np.linalg.norm(d, axis=0)
        gs = neuron[: len(neuron) // a.size * a.size].reshape(-1, a.size).sum(1)
        groups[L] = gs.tolist()
        rows.append({
            "layer": L,
            "down_fro": float(np.linalg.norm(d)), "gate_fro": float(np.linalg.norm(g)),
            "up_fro": float(np.linalg.norm(u)), "o_fro": float(np.linalg.norm(o)),
            "v_fro": float(np.linalg.norm(v)),
            "ln_in_mean": float(ln_in.mean()), "ln_post_mean": float(ln_post.mean()),
            "ln_post_max": float(ln_post.max()),
            "neuron_score_sum": float(neuron.sum()), "neuron_score_max": float(neuron.max()),
            "neuron_score_top1pct_share": float(np.sort(neuron)[::-1][: len(neuron) // 100].sum() / neuron.sum()),
        })
    out = {"rows": rows, "group_size": a.size, "group_scores": groups}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    keys = ["down_fro", "gate_fro", "up_fro", "o_fro", "v_fro", "ln_in_mean", "ln_post_mean",
            "ln_post_max", "neuron_score_max", "neuron_score_top1pct_share"]
    print("layer " + " ".join(f"{k:>10.10}" for k in keys))
    for r in rows:
        print(f"{r['layer']:>5} " + " ".join(f"{r[k]:>10.4g}" for k in keys))


# --------------------------------------------------------------------------- modello

def load_model(model_dir, weights=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    if weights:
        from safetensors.torch import load_file
        missing, unexpected = model.load_state_dict(load_file(weights), strict=False)
        assert not [m for m in missing if m != "lm_head.weight"] and not unexpected
        model.tie_weights()
    model.eval()
    return tok, model


def make_ppl(tok, model, text_path, max_tokens):
    import torch
    ids = tok(Path(text_path).read_text(encoding="utf-8"), return_tensors="pt")["input_ids"][:, :max_tokens]

    def ppl():
        with torch.no_grad():
            return math.exp(float(model(input_ids=ids, labels=ids).loss))
    return ppl, ids.shape[1]


def cmd_ppl(a):
    tok, model = load_model(a.model, a.weights)
    ppl, n = make_ppl(tok, model, a.text, a.max_tokens)
    print(f"perplexity={ppl():.5f}  token={n}  weights={a.weights}")


def cmd_layers(a):
    import torch
    tok, model = load_model(a.model)
    ppl, n = make_ppl(tok, model, a.text, a.max_tokens)
    t0 = time.time()
    base = ppl()
    print(f"base perplexity={base:.4f} ({n} token, {time.time() - t0:.1f}s)")
    res = {"base": base, "tokens": n, "mlp": [], "attn": []}
    for L, layer in enumerate(model.model.layers):
        for kind, mod in (("mlp", layer.mlp.down_proj), ("attn", layer.self_attn.o_proj)):
            saved = mod.weight.data.clone()
            mod.weight.data.zero_()
            if getattr(mod, "bias", None) is not None:
                raise RuntimeError("bias inatteso")
            p = ppl()
            mod.weight.data.copy_(saved)
            res[kind].append(p)
            print(f"layer {L:>2} {kind:<4} ppl={p:>10.4f}  Δ={p - base:+.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_groups(a):
    tok, model = load_model(a.model)
    ppl, n = make_ppl(tok, model, a.text, a.max_tokens)
    base = ppl()
    mlp = model.model.layers[a.layer].mlp
    res = {"base": base, "layer": a.layer, "size": a.size, "groups": {}}
    for gi in [int(x) for x in a.groups.split(",")]:
        sl = slice(gi * a.size, (gi + 1) * a.size)
        saved = mlp.down_proj.weight.data[:, sl].clone()
        mlp.down_proj.weight.data[:, sl] = 0  # azzerare l'uscita del neurone equivale a spegnerlo
        p = ppl()
        mlp.down_proj.weight.data[:, sl] = saved
        res["groups"][gi] = p
        print(f"gruppo {gi:>3}  ppl={p:.5f}  Δ={p - base:+.5f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("model"); s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=64); s.set_defaults(fn=cmd_stats)
    for name, fn in (("layers", cmd_layers), ("groups", cmd_groups), ("ppl", cmd_ppl)):
        s = sub.add_parser(name); s.add_argument("model"); s.add_argument("--text", required=True)
        s.add_argument("--max-tokens", type=int, default=4096)
        if name != "ppl":
            s.add_argument("--out", required=True)
        if name == "groups":
            s.add_argument("--layer", type=int, required=True); s.add_argument("--groups", required=True)
            s.add_argument("--size", type=int, default=64)
        if name == "ppl":
            s.add_argument("--weights")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
