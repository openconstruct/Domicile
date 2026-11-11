# Embedder surface. Spec D2: NOT pluggable in deployment — one fixed
# on-device embedder (ONNX; Qualcomm native path, shim elsewhere). The
# classes here are the seams for that one embedder, plus test doubles.
import hashlib

import numpy as np

from .vectors import normalize


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


class CallableEmbedder:
    # Adapts any text->vector callable (e.g. Domicile's own inference path).
    def __init__(self, fn, dim):
        self.fn = fn
        self.dim = dim

    def embed(self, text):
        return normalize(self.fn(text))


class OnnxEmbedder:
    # The deployment embedder: an ONNX embedding model, mean-pooled.
    # Execution provider selection (QNN on Qualcomm, CPU/XNNPACK via the
    # shim elsewhere) is onnxruntime session config, set by the host app.
    def __init__(self, model_path, tokenizer, dim, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for OnnxEmbedder") from e
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = tokenizer
        self.dim = dim

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="np")
        out = self.session.run(None, dict(inputs))[0]
        mask = inputs["attention_mask"].astype(np.float32)
        pooled = (out * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return normalize(pooled[0])
