"""wio.py - lettura a basso livello di file di pesi (safetensors, GGUF) senza torch.

Solo numpy + libreria standard. I tensori vengono letti via memmap: non si carica
mai l'intero modello in RAM, si leggono solo le porzioni richieste.

Formati supportati:
  - safetensors: file singolo, cartella con shard, o *.safetensors.index.json
  - GGUF (v2/v3): metadati, elenco tensori; decodifica di F32, F16, BF16, Q8_0, Q4_0.
    Gli altri tipi quantizzati (K-quants, IQ*) sono elencati ma non decodificati.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Strutture comuni
# --------------------------------------------------------------------------- #


@dataclass
class TensorInfo:
    name: str
    dtype: str            # es. "BF16", "F16", "F32", "Q8_0", "Q4_K"
    shape: tuple          # ordine "numpy/PyTorch" (riga-maggiore)
    file: Path
    offset: int           # byte assoluto di inizio nel file
    nbytes: int
    decodable: bool = True

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class WeightFile:
    fmt: str                                  # "safetensors" | "gguf"
    paths: list
    metadata: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)   # name -> TensorInfo

    def get(self, name: str) -> TensorInfo:
        if name not in self.tensors:
            raise KeyError(f"Tensore non trovato: {name!r}")
        return self.tensors[name]


# --------------------------------------------------------------------------- #
# Conversioni dtype
# --------------------------------------------------------------------------- #

def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f: np.ndarray) -> np.ndarray:
    """float32 -> bf16 (come uint16) con arrotondamento round-to-nearest-even."""
    f = np.ascontiguousarray(f, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    rounding = 0x7FFF + ((u >> 16) & 1)
    out = ((u + rounding) >> 16).astype(np.uint16)
    nan = np.isnan(f)
    if nan.any():
        out[nan] = 0x7FC0
    return out


# --------------------------------------------------------------------------- #
# safetensors
# --------------------------------------------------------------------------- #

_ST_DTYPES = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "BF16": (np.uint16, 2), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


def _read_safetensors_file(path: Path, wf: WeightFile) -> None:
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    base = 8 + hlen
    meta = header.pop("__metadata__", None) or {}
    for k, v in meta.items():
        wf.metadata.setdefault(k, v)
    for name, info in header.items():
        a, b = info["data_offsets"]
        dt = info["dtype"]
        wf.tensors[name] = TensorInfo(
            name=name, dtype=dt, shape=tuple(info["shape"]), file=path,
            offset=base + a, nbytes=b - a, decodable=dt in _ST_DTYPES,
        )


def open_safetensors(path: Path) -> WeightFile:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"Nessun .safetensors in {path}")
    elif path.name.endswith(".index.json"):
        idx = json.loads(path.read_text())
        files = sorted({path.parent / f for f in idx["weight_map"].values()})
    else:
        files = [path]
    wf = WeightFile(fmt="safetensors", paths=files)
    for f in files:
        _read_safetensors_file(f, wf)
    return wf


# --------------------------------------------------------------------------- #
# GGUF
# --------------------------------------------------------------------------- #

# GGML tensor types: id -> (nome, elementi per blocco, byte per blocco)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}
GGUF_DECODABLE = {"F32", "F16", "BF16", "Q8_0", "Q4_0", "F64", "I8", "I16", "I32"}

_GGUF_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.fh.read(size))[0]

    def string(self):
        n = self.unpack("<Q")
        return self.fh.read(n).decode("utf-8", errors="replace")

    def value(self, vtype, max_array=None):
        if vtype in _GGUF_SCALARS:
            return self.unpack(_GGUF_SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.unpack("<I")
            count = self.unpack("<Q")
            if etype in _GGUF_SCALARS and max_array is not None and count > max_array:
                # salta i valori senza tenerli (es. vocabolari enormi)
                size = struct.calcsize(_GGUF_SCALARS[etype])
                first = [self.value(etype) for _ in range(max_array)]
                self.fh.seek((count - max_array) * size, os.SEEK_CUR)
                return {"_array_len": count, "_first": first}
            vals = []
            for i in range(count):
                v = self.value(etype, max_array)
                if max_array is None or i < max_array:
                    vals.append(v)
            if max_array is not None and count > max_array:
                return {"_array_len": count, "_first": vals}
            return vals
        raise ValueError(f"Tipo GGUF sconosciuto: {vtype}")


def open_gguf(path: Path, max_array: int = 16) -> WeightFile:
    path = Path(path)
    wf = WeightFile(fmt="gguf", paths=[path])
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError("Non è un file GGUF (magic errato)")
        version = r.unpack("<I")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        wf.metadata["gguf.version"] = version
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            wf.metadata[key] = r.value(vtype, max_array)
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(nd)]
            ttype = r.unpack("<I")
            off = r.unpack("<Q")
            infos.append((name, dims, ttype, off))
        align = int(wf.metadata.get("general.alignment", 32))
        pos = fh.tell()
        data_start = (pos + align - 1) // align * align
    for name, dims, ttype, off in infos:
        tname, blk, bsz = GGML_TYPES.get(ttype, (f"TYPE_{ttype}", 1, 0))
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel // blk * bsz
        wf.tensors[name] = TensorInfo(
            name=name, dtype=tname, shape=tuple(reversed(dims)), file=path,
            offset=data_start + off, nbytes=nbytes, decodable=tname in GGUF_DECODABLE,
        )
    return wf


# --------------------------------------------------------------------------- #
# Apertura generica e lettura dei valori
# --------------------------------------------------------------------------- #

def open_weights(path) -> WeightFile:
    p = Path(path)
    if p.is_dir() or p.suffix == ".safetensors" or p.name.endswith(".index.json"):
        return open_safetensors(p)
    with open(p, "rb") as fh:
        magic = fh.read(4)
    if magic == b"GGUF":
        return open_gguf(p)
    return open_safetensors(p)


def raw_view(t: TensorInfo, mode: str = "r") -> np.memmap:
    """Vista grezza (byte) del tensore sul file, senza decodifica."""
    return np.memmap(t.file, dtype=np.uint8, mode=mode, offset=t.offset, shape=(t.nbytes,))


def _dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def _dequant_q4_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (q * d).reshape(-1)


def read_tensor(t: TensorInfo, rows: slice | None = None) -> np.ndarray:
    """Legge il tensore come float32 (o int per tipi interi).

    rows: se dato, legge solo quelle righe del primo asse (utile per matrici enormi).
    """
    if not t.decodable:
        raise NotImplementedError(
            f"{t.name}: dtype {t.dtype} non decodificabile da questo tool "
            "(converti il modello in F16/BF16/Q8_0 o usa safetensors)."
        )
    shape = t.shape if t.shape else (1,)
    row_elems = 1
    for d in shape[1:]:
        row_elems *= int(d)
    start_row, stop_row = 0, int(shape[0])
    if rows is not None:
        start_row, stop_row, step = rows.indices(int(shape[0]))
        if step != 1:
            raise ValueError("step di riga non supportato")
    n_rows = max(0, stop_row - start_row)
    out_shape = (n_rows,) + tuple(int(d) for d in shape[1:])
    count = n_rows * row_elems

    dt = t.dtype
    if dt in ("Q8_0", "Q4_0"):
        blk, bsz = (32, 34) if dt == "Q8_0" else (32, 18)
        if (start_row * row_elems) % blk or count % blk:
            raise ValueError("Righe non allineate ai blocchi di quantizzazione")
        b0 = start_row * row_elems // blk * bsz
        b1 = b0 + count // blk * bsz
        raw = np.asarray(raw_view(t)[b0:b1])
        vals = _dequant_q8_0(raw) if dt == "Q8_0" else _dequant_q4_0(raw)
        return vals.reshape(out_shape)

    np_dt, size = _ST_DTYPES.get(dt, (None, None))
    if np_dt is None:  # tipi GGUF interi/float non in tabella safetensors
        np_dt, size = {"I8": (np.int8, 1), "I16": (np.int16, 2), "I32": (np.int32, 4)}[dt]
    mm = np.memmap(t.file, dtype=np_dt, mode="r",
                   offset=t.offset + start_row * row_elems * size, shape=(count,))
    arr = np.asarray(mm)
    if dt == "BF16":
        arr = bf16_to_f32(arr)
    elif arr.dtype in (np.float16, np.float64):
        arr = arr.astype(np.float32)
    return arr.reshape(out_shape)


def encode_like(t: TensorInfo, values: np.ndarray) -> np.ndarray:
    """Converte valori float32 nel formato binario del tensore (solo tipi non quantizzati)."""
    dt = t.dtype
    if dt == "BF16":
        return f32_to_bf16(values)
    if dt == "F16":
        return values.astype(np.float16)
    if dt == "F32":
        return values.astype(np.float32)
    if dt == "F64":
        return values.astype(np.float64)
    raise NotImplementedError(f"Scrittura non supportata per dtype {dt}")


def dtype_size(dt: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}[dt]
