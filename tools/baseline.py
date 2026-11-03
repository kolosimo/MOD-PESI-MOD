#!/usr/bin/env python3
"""baseline.py - esegue un modello su prompt fissi e misura perplexity (E0 / verifica modifiche).

Richiede torch + transformers. Decodifica greedy (deterministica): due run sullo stesso
modello danno lo stesso output, quindi le differenze dopo una modifica sono dovute ai pesi.

  run      MODEL_DIR --out logs/baseline/NOME.json [--weights FILE.safetensors]
           --weights sostituisce model.safetensors con un file modificato (stessa architettura)
  compare  A.json B.json      differenze di testo, perplexity e probabilità del primo token

File di input (default nella cartella prompts/):
  prompts/baseline.txt   un prompt per riga (righe vuote e # ignorate)
  prompts/ppl.txt        testo su cui calcolare la perplexity
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(model_dir: str, weights: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    if weights:
        from safetensors.torch import load_file
        sd = load_file(weights)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if m != "lm_head.weight"]  # embedding legati
        if missing or unexpected:
            raise SystemExit(f"Pesi incompatibili: missing={missing} unexpected={unexpected}")
        model.tie_weights()
    model.eval()
    return tok, model


def read_lines(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def cmd_run(a):
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok, model = load(a.model, a.weights)
    prompts = read_lines(Path(a.prompts))
    results = []
    t0 = time.time()
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) \
            if a.chat and tok.chat_template else p
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
            probs = torch.softmax(logits, -1)
            top = torch.topk(probs, 5)
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "prompt": p, "output": gen,
            "top5_first_token": [[tok.decode([int(i)]), round(float(v), 5)]
                                 for v, i in zip(top.values, top.indices)],
        })
        print(f"> {p}\n  {gen!r}\n")

    ppl = None
    ppl_path = Path(a.ppl_text)
    if ppl_path.exists():
        enc = tok(ppl_path.read_text(encoding="utf-8"), return_tensors="pt")
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        ppl = math.exp(float(loss))
        print(f"Perplexity su {ppl_path.name}: {ppl:.4f}  ({enc['input_ids'].shape[1]} token)")

    rec = {"model": a.model, "weights": a.weights, "chat": a.chat,
           "max_new_tokens": a.max_new_tokens, "perplexity": ppl,
           "seconds": round(time.time() - t0, 1), "results": results}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Salvato: {out}")


def cmd_compare(a):
    A = json.loads(Path(a.a).read_text(encoding="utf-8"))
    B = json.loads(Path(a.b).read_text(encoding="utf-8"))
    if A.get("perplexity") and B.get("perplexity"):
        d = B["perplexity"] - A["perplexity"]
        print(f"Perplexity: {A['perplexity']:.4f} -> {B['perplexity']:.4f} ({d:+.4f}, {d / A['perplexity']:+.2%})")
    same = 0
    for ra, rb in zip(A["results"], B["results"]):
        if ra["output"] == rb["output"]:
            same += 1
            continue
        print(f"\n> {ra['prompt']}\n  A: {ra['output']!r}\n  B: {rb['output']!r}")
        print(f"  top5 A: {ra['top5_first_token']}\n  top5 B: {rb['top5_first_token']}")
    print(f"\nOutput identici: {same}/{len(A['results'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model"); r.add_argument("--out", required=True); r.add_argument("--weights")
    r.add_argument("--prompts", default=str(ROOT / "prompts" / "baseline.txt"))
    r.add_argument("--ppl-text", default=str(ROOT / "prompts" / "ppl.txt"))
    r.add_argument("--max-new-tokens", type=int, default=40)
    r.add_argument("--no-chat", dest="chat", action="store_false")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    c.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
