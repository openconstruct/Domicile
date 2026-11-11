"""Backend switch for the memory store: flash memory (SQLite, mem0_v2) or
the legacy mem0/qdrant stack (mem0_core).

Flag file data/mem0/flash.on present => flash memory. Absent => legacy.
Rollback is: delete the flag, restart cchat. Call sites do
`import mem0_switch as mem0_core` and use the same function names; anything
not defined here is looked up on the chosen backend.
"""
from __future__ import annotations

import importlib

from paths import MEM0_DIR, QDRANT_PATH

FLAG = MEM0_DIR / "flash.on"


def flash() -> bool:
    return FLAG.exists()


def enable_for_new_install() -> bool:
    """Make flash memory the backend on a fresh install (the setup wizard calls
    this). A legacy store already on disk is left alone: flipping the flag over
    it would hide every memory it holds. Returns True when flash is on after."""
    if FLAG.exists():
        return True
    if QDRANT_PATH.exists():
        return False
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text("flash memory (set by the setup wizard)\n")
    return True


def backend():
    return importlib.import_module("mem0_v2" if flash() else "mem0_core")


def store():
    """Open (warm) the store. Flash: the MemoryCore; legacy: the mem0 Memory."""
    b = backend()
    return b.core() if flash() else b.store()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(backend(), name)
