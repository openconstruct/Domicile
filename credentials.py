"""Domicile credentials: read only the repository .env, never legacy files."""
from __future__ import annotations

import os
import re
from io import StringIO
import tempfile
from pathlib import Path
from dotenv import dotenv_values, set_key
import file_cache

ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load(text) -> dict[str, str]:
    # No interpolation: dollar signs in tokens must remain literal.
    return {k: v for k, v in dotenv_values(stream=StringIO(text), interpolate=False).items()
            if v is not None}


def values() -> dict[str, str]:
    try:
        return file_cache.read(ENV_PATH, _load)
    except FileNotFoundError:
        return {}


def get(name: str, default: str = "") -> str:
    return values().get(name, default).strip()


def key_name(key_file: str | None = None, key_env: str | None = None,
             provider: str | None = None) -> str:
    # Old config paths are identifiers only; their contents are never read.
    name = key_env or (re.sub(r"[^A-Za-z0-9_]", "_",
                            Path(key_file).stem if key_file else provider or "").upper()
                       + "_API_KEY")
    if name == "_API_KEY" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Provider needs a valid key_env credential name")
    return name


def read_key(key_file: str | None = None, key_env: str | None = None) -> str:
    name = key_name(key_file, key_env)
    value = get(name)
    if not value:
        raise RuntimeError(f"No API key: set {name} in {ENV_PATH}.")
    return value


def save(updates: dict[str, str]) -> None:
    """Atomically update .env, preserving other settings and private permissions."""
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".env-", dir=ENV_PATH.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            if ENV_PATH.exists():
                handle.write(ENV_PATH.read_text())
        for name, value in updates.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("Invalid credential name")
            set_key(temporary, name, str(value), quote_mode="always")
        os.replace(temporary, ENV_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
