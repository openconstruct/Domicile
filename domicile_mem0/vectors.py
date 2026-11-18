# Vector storage and our own quantized scan (spec D4: no sqlite-vector).
import numpy as np


def to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)


def normalize(vec):
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def quantize_int8(vec):
    # Symmetric per-vector INT8 (cosine-safe; never UINT8 — it destroys angles).
    v = np.asarray(vec, dtype=np.float32)
    peak = float(np.max(np.abs(v))) if v.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    codes = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
    return codes.tobytes(), scale


# --- the scan --------------------------------------------------------------
# Two tiers, picked by store size when the cache is (re)built:
#
#   exact  (< EXACT_MAX facts): one float32 matmul. 2 ms at 14k facts; nothing
#          approximate about it. This is the path almost every store is on.
#   tiered (>= EXACT_MAX): RAM holds sign bits (48 B/fact) and the stored int8
#          codes (384 B/fact) instead of float32 (1536 B/fact). A query is a
#          Hamming pass over the bits, an int8 re-rank of the best PREFILTER,
#          then exact float32 (read back from the store) for the few that can
#          actually be returned -- so scores stay on the calibrated scale.
#
# Measured on 13,789 real fact vectors tiled out (2026-11-12, 4 cores):
#   facts     float32 exact    int8 matmul    bits -> int8
#    14k          2.1 ms          8.5 ms         0.9 ms
#   100k         20   ms         84   ms         5.6 ms
#   500k         84   ms        420   ms        26   ms
# int8 alone is SLOWER than float32 (numpy has no SIMD integer matmul); its
# value is the 4x smaller RAM and an accurate re-rank. Recall@20 of the tiered
# path against exact: 0.979 at PREFILTER=400 on perturbed real vectors.
EXACT_MAX = 50_000
PREFILTER = 400

_HAS_BITCOUNT = hasattr(np, "bitwise_count")
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def sign_bits(vecs):
    """Rows of sign bits, padded to whole uint64 words. Accepts float or int8."""
    v = np.atleast_2d(np.asarray(vecs))
    packed = np.packbits(v > 0, axis=1)
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad)))
    return np.ascontiguousarray(packed).view(np.uint64)


def hamming(bits, qbits):
    x = bits ^ qbits
    if _HAS_BITCOUNT:
        return np.bitwise_count(x).sum(axis=1, dtype=np.uint16)
    return _POP8[x.view(np.uint8)].sum(axis=1, dtype=np.uint16)


class Scanner:
    """Immutable similarity index over the active facts, in row order.

    Every change returns a NEW Scanner (copy-on-write), so a reader that took
    one before a write keeps scanning a consistent set.
    """

    def __init__(self, matrix=None, codes=None, scales=None, bits=None):
        self.matrix, self.codes, self.scales, self.bits = matrix, codes, scales, bits
        self.tiered = matrix is None and codes is not None

    def __len__(self):
        m = self.codes if self.tiered else self.matrix
        return 0 if m is None else len(m)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.matrix, self.codes, self.scales, self.bits)
                   if a is not None)

    @classmethod
    def build(cls, entries, tiered):
        """entries: [(float32 unit vec or None, int8 codes or None, scale)]."""
        if not entries:
            return cls()
        if not tiered:
            return cls(matrix=np.vstack([e[0] if e[0] is not None
                                         else e[1].astype(np.float32) * np.float32(e[2])
                                         for e in entries]).astype(np.float32))
        codes, scales = [], []
        for vec, code, scale in entries:
            if code is None:
                blob, scale = quantize_int8(vec)
                code = np.frombuffer(blob, dtype=np.int8)
            codes.append(code)
            scales.append(scale)
        codes = np.ascontiguousarray(np.vstack(codes))
        return cls(codes=codes, scales=np.asarray(scales, dtype=np.float32),
                   bits=sign_bits(codes))

    def _entry(self, entry):
        vec, code, scale = entry
        if not self.tiered:
            if vec is None:
                vec = code.astype(np.float32) * np.float32(scale)
            return {"matrix": np.asarray(vec, dtype=np.float32)[None, :]}
        if code is None:
            blob, scale = quantize_int8(vec)
            code = np.frombuffer(blob, dtype=np.int8)
        code = np.asarray(code, dtype=np.int8)[None, :]
        return {"codes": code, "scales": np.asarray([scale], dtype=np.float32),
                "bits": sign_bits(code)}

    def _arrays(self):
        names = ("codes", "scales", "bits") if self.tiered else ("matrix",)
        return {n: getattr(self, n) for n in names}

    def appended(self, entry):
        if len(self) == 0:
            return Scanner.build([entry], self.tiered)
        new = self._entry(entry)
        return Scanner(**{n: np.concatenate([a, new[n]]) for n, a in self._arrays().items()})

    def replaced(self, i, entry):
        new, out = self._entry(entry), {}
        for n, a in self._arrays().items():
            a = a.copy()
            a[i] = new[n][0]
            out[n] = a
        return Scanner(**out)

    def deleted(self, i):
        return Scanner(**{n: np.delete(a, i, axis=0) for n, a in self._arrays().items()})

    def vector(self, i):
        """Unit float32 vector for row i (int8-accurate in tiered mode)."""
        if self.tiered:
            return normalize(self.codes[i].astype(np.float32) * self.scales[i])
        return normalize(self.matrix[i])

    def sims(self, query, mask=None, k=20, exact=None):
        """Cosine of `query` against every row, as a full-length array.

        Exact mode: every entry is the true cosine. Tiered mode: -inf except
        for the best candidates; `exact(row_indices) -> float32 matrix` (when
        given) replaces the int8 estimate with the true cosine for the top of
        them. Rows where mask is False are -inf either way.
        """
        n = len(self)
        if n == 0:
            return np.empty((0,), dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        if not self.tiered:
            s = self.matrix @ q
            return s if mask is None else np.where(mask, s, -np.inf).astype(np.float32)
        h = hamming(self.bits, sign_bits(q)[0])
        if mask is not None:
            h = np.where(mask, h, np.uint16(65535))
        pre = min(n, max(PREFILTER, 20 * int(k)))
        cand = np.argpartition(h, pre - 1)[:pre] if pre < n else np.arange(n)
        if mask is not None:
            cand = cand[mask[cand]]
        out = np.full(n, -np.inf, dtype=np.float32)
        if not len(cand):
            return out
        est = (self.codes[cand].astype(np.float32) @ q) * self.scales[cand]
        out[cand] = est
        if exact is not None:
            top = cand[np.argsort(-est)[:max(4 * int(k), 64)]]
            out[top] = exact(top) @ q
        return out
