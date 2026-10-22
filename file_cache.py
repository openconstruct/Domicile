"""Bounded parsing caches for small settings files, with immediate edit detection."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path


def signature(path):
    path = Path(path).absolute()
    stat = path.stat()
    return (path, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=16)
def _parse(text, loader):
    return loader(text)


def read(path, loader):
    """Return an independent value, reparsing only when the file changes."""
    # Content keys also detect same-size edits on coarse-timestamp filesystems.
    return deepcopy(_parse(Path(path).read_text(), loader))
