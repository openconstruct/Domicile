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


def dequantize_int8(blob, scale):
    return np.frombuffer(blob, dtype=np.int8).astype(np.float32) * np.float32(scale)


def stack_matrix(blobs):
    if not blobs:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack([from_blob(b) for b in blobs])


def cosine_scores(matrix, query):
    # Rows and query are unit-normalized at write/read time, so cosine = dot.
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    return matrix @ np.asarray(query, dtype=np.float32)
