"""Test dei lettori wio.py contro le librerie ufficiali safetensors e gguf (file sintetici)."""
import sys, json, tempfile, subprocess, numpy as np, ml_dtypes
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO/"tools"))
from wio import open_weights, read_tensor
from safetensors.numpy import save_file, load_file
import gguf
from gguf.quants import quantize, dequantize
T = Path(tempfile.mkdtemp()); rng = np.random.default_rng(1)

# --- safetensors con BF16 / F16 / F32, 2 shard + index
a = rng.normal(0, .02, (64, 96)).astype(np.float32)
bf = a.astype(ml_dtypes.bfloat16)
save_file({"model.layers.0.mlp.down_proj.weight": bf, "model.norm.weight": np.ones(96, np.float16)}, T/"s-1.safetensors", metadata={"format":"pt"})
save_file({"model.layers.1.mlp.down_proj.weight": a}, T/"s-2.safetensors")
(T/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {
  "model.layers.0.mlp.down_proj.weight":"s-1.safetensors","model.norm.weight":"s-1.safetensors",
  "model.layers.1.mlp.down_proj.weight":"s-2.safetensors"}}))
wf = open_weights(T/"model.safetensors.index.json")
x = read_tensor(wf.get("model.layers.0.mlp.down_proj.weight"))
assert np.array_equal(x, bf.astype(np.float32)), "BF16 mismatch"
assert np.array_equal(read_tensor(wf.get("model.layers.0.mlp.down_proj.weight"), slice(10,20)), bf.astype(np.float32)[10:20])
assert np.array_equal(read_tensor(wf.get("model.layers.1.mlp.down_proj.weight")), a)
print("safetensors OK")

# --- GGUF con F32, F16, Q8_0, Q4_0, Q4_K (non decodificabile)
w = gguf.GGUFWriter(str(T/"m.gguf"), "llama")
w.add_block_count(1); w.add_string("general.name", "test")
w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(100)])
m = rng.normal(0, .05, (32, 256)).astype(np.float32)
w.add_tensor("blk.0.f32", m)
w.add_tensor("blk.0.f16", m.astype(np.float16))
Q = gguf.GGMLQuantizationType
for nm, qt in (("blk.0.q8", Q.Q8_0), ("blk.0.q4", Q.Q4_0)):
    q = quantize(m, qt); w.add_tensor(nm, q, raw_shape=q.shape, raw_dtype=qt)
qk = rng.integers(0,255,(32,144),dtype=np.uint8); w.add_tensor("blk.0.q4k", qk, raw_shape=qk.shape, raw_dtype=Q.Q4_K)
w.add_tensor("output_norm.weight", np.ones(256, np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
g = open_weights(T/"m.gguf")
assert g.get("blk.0.f32").shape == (32, 256)
assert np.array_equal(read_tensor(g.get("blk.0.f32")), m)
assert np.array_equal(read_tensor(g.get("blk.0.f16")), m.astype(np.float16).astype(np.float32))
for nm, qt in (("blk.0.q8", Q.Q8_0), ("blk.0.q4", Q.Q4_0)):
    ref = dequantize(quantize(m, qt), qt)
    got = read_tensor(g.get(nm))
    assert np.allclose(got, ref, atol=0, rtol=0), nm
    assert np.array_equal(read_tensor(g.get(nm), slice(5, 9)), ref[5:9])
assert not g.get("blk.0.q4k").decodable
assert g.metadata["general.name"] == "test" and g.metadata["tokenizer.ggml.tokens"]["_array_len"] == 100
# offset coerenti con la libreria ufficiale
r = gguf.GGUFReader(str(T/"m.gguf"))
for t in r.tensors:
    assert g.get(t.name).offset == t.data_offset, t.name
print("gguf OK")

# --- edit_weights: due modifiche in sequenza con --inplace-out devono sommarsi
EDIT = [sys.executable, str(REPO/"tools"/"edit_weights.py")]
src, out = T/"s-1.safetensors", T/"s-1_seq.safetensors"
nm = "model.layers.0.mlp.down_proj.weight"
subprocess.run(EDIT + [str(src), "--out", str(out), "--tensor", nm, "--rows", "0:1", "--op", "zero", "--log-dir", str(T/"logs")], check=True, capture_output=True)
subprocess.run(EDIT + [str(src), "--out", str(out), "--inplace-out", "--tensor", nm, "--rows", "5:6", "--op", "zero", "--log-dir", str(T/"logs")], check=True, capture_output=True)
y = load_file(out)[nm].astype(np.float32); x0 = bf.astype(np.float32)
assert (y[0] == 0).all() and (y[5] == 0).all(), "modifiche sequenziali non cumulate"
assert np.array_equal(np.delete(y, [0, 5], 0), np.delete(x0, [0, 5], 0))
print("edit sequenziale OK")
print("TUTTI I TEST OK")
