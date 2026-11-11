# domicile_mem0 — drop-in mem0 replacement for Domicile (spec v3).
# One SQLite base: facts + vectors + tracking payload. No Qdrant, no
# sqlite-vector, no third-party memory service.
from .core import MemoryCore

__all__ = ["MemoryCore"]
