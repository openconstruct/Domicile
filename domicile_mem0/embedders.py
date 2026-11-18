# The embedder. ONE model for every store, loaded ONCE per process: the fact
# store (mem0_v2) and full memory (mem2_core) both call embed_one/embed_many
# here, so they share one ONNX session instead of each loading its own copy.
#
# Swapping the model means re-embedding BOTH stores (each stamps the model name
# and refuses to open on a mismatch). Set DOMICILE_EMBED_MODEL only for that.
import hashlib
import os
import threading

import numpy as np

from .vectors import normalize

EMBED_MODEL = os.environ.get("DOMICILE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMS = 384
EMBED_MAX_TOKENS = 256      # MiniLM's window: text past this is not embedded

_model = None
_model_lock = threading.Lock()


def _get_model():
    """The process-wide fastembed model (ONNX/CPU), built on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding
                import paths  # noqa: F401 -- sets FASTEMBED_CACHE_PATH to data/fastembed_cache
                _model = TextEmbedding(EMBED_MODEL,
                                       cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
    return _model


def embed_many(texts):
    """Unit-normalised float32 matrix, one row per text."""
    m = np.asarray(list(_get_model().embed(list(texts))), dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def embed_one(text):
    return embed_many([text])[0]


class SharedEmbedder:
    """MemoryCore's embedder interface over the shared model."""
    name = EMBED_MODEL
    dim = EMBED_DIMS

    def embed(self, text):
        return embed_one(text)


class HashEmbedder:
    # Deterministic token-projection embedder for tests and offline dev.
    # Structurally valid (unit vectors, stable per text), semantically weak.
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, text):
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in text.lower().split():
            seed = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            vec += rng.standard_normal(self.dim)
        return normalize(vec)
