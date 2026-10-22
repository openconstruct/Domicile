"""Domicile local filesystem, process, web, and display tools."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import glob as _globmod
import shutil
import mimetypes
import math
import os
import re
import selectors
import signal
import shlex
import subprocess
import sys
import time
import threading
from pathlib import Path

HOME = Path.home()

# limits

MAX_READ_BYTES = 2_000_000
MAX_WRITE_BYTES = 5_000_000
MAX_LIST_ENTRIES = 500
SHELL_TIMEOUT_S = 120.0
SHELL_MAX_OUTPUT = 60_000
MAX_IMAGE_BYTES = 8_000_000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_PDF_BYTES = 25_000_000        # matches api_server.MAX_UPLOAD
PDF_OCR_PAGE_CAP = 20              # OCR is slow; cap pages rasterized+OCR'd per read
PDF_TEXT_LAYER_MIN_CHARS = 40      # below this per-page average, treat as scanned/image-only

TOOL_RESULT_CHARS = 12_000

# safety

_BLOCKED_EXE = {
    "bash", "sh", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
    "cmd", "powershell", "pwsh",
    "sudo", "doas", "su", "pkexec",
}

_INTERPRETERS = {
    "python", "python2", "python3", "perl", "ruby", "node", "nodejs",
    "deno", "bun", "php", "lua", "awk", "gawk", "mawk", "sed",
    "env", "xargs", "find", "git", "vim", "vi", "nvim", "emacs",
    "ssh", "make", "docker", "systemd-run", "nohup", "setsid", "timeout",
    "watch", "script", "expect",
}

_EXEC_FLAGS = {
    "-c", "-e", "-E", "--eval", "--exec", "-exec", "-execdir", "--command",
    "-I", "--interactive", "-p", "--program", "-s", "--source",
}

_DESTRUCTIVE = re.compile(
    r"\b(rm|rmdir|unlink|shred|srm|truncate|mv|dd|mkfs|"
    r"tee|cp|rsync|install|ln|"
    r"rmtree|remove|move|replace|overwrite|"
    r"chown|chmod|chgrp)\b|>\s*/|--force|-f\b|--hard|-{1,2}delete|-D\b|"
    r"-{1,2}(prune|purge|wipe|destroy)\b",
    re.IGNORECASE,
)

_GIT_DESTRUCTIVE = re.compile(
    r"\b(reset\s+--hard|clean\s+-[a-z]*[fd]|push\s+.*--force|push\s+.*-f\b|"
    r"branch\s+-D|checkout\s+--\s|restore\s|filter-branch|reflog\s+expire|"
    r"gc\s+--prune)",
    re.IGNORECASE,
)


class ToolDenied(PermissionError):
    """A guard refused. Distinct from a tool that ran and failed."""


def guard_argv(argv: list[str]) -> None:
    """Existing command checks; not process isolation."""
    if not argv:
        raise ToolDenied("empty command")

    exe = os.path.basename(str(argv[0])).strip()
    joined = " ".join(str(a) for a in argv)

    if exe in _BLOCKED_EXE:
        raise ToolDenied(
            f"'{exe}' is not available: commands run as an argv list with no "
            f"shell, so shells and privilege tools are blocked. Run the "
            f"program directly instead of via a shell."
        )

    if exe in _INTERPRETERS:
        flags = {a for a in argv[1:] if str(a).startswith("-")}
        hit = flags & _EXEC_FLAGS
        if hit:
            if _DESTRUCTIVE.search(joined) or _GIT_DESTRUCTIVE.search(joined):
                raise ToolDenied(
                    f"refused: inline code ({sorted(hit)[0]}) that looks "
                    f"destructive. Write it to a file and run that, so it can "
                    f"be read before it executes."
                )

    if exe == "git" and _GIT_DESTRUCTIVE.search(joined):
        raise ToolDenied(
            f"refused: '{joined[:80]}' rewrites or discards git history. "
            f"Ask first -- practices.md makes this a human decision."
        )


def _clean_env() -> dict[str, str]:
    """Child environment with credentials removed."""
    env = dict(os.environ)
    for k in list(env):
        up = k.upper()
        if any(s in up for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD",
                                 "OPENROUTER", "ANTHROPIC", "OPENAI",
                                 "DISCORD", "GITHUB", "GITLAB")):
            env.pop(k, None)
    return env


# filesystem

def _have(exe: str) -> bool:
    """Is this a real executable on PATH? Not 'does the shell resolve it'."""
    from shutil import which
    return which(exe) is not None


ALLOWED_ROOTS = [HOME, Path("/tmp")]


def _resolve(path: str, *, roots: list[Path] | None = None) -> Path:
    """Resolve under an allowed root, refusing escapes. From freeclaw fs.py:169."""
    if not path:
        raise ValueError("path is required")
    roots = roots or ALLOWED_ROOTS
    p = Path(str(path)).expanduser()
    resolved = p.resolve() if p.is_absolute() else (roots[0] / p).resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ToolDenied(
        f"path escapes the allowed roots "
        f"({', '.join(str(r) for r in roots)}): {path!r}"
    )


_READ_FILES: set[str] = set()
_READ_STAMP: dict[str, int] = {}          # path -> mtime_ns when we last read/wrote it

# Set by execute(trusted=True) (the Claude SDK bridge only). Never read from
# model-supplied args, so a model cannot grant itself shell strings or
# whole-filesystem reads.
_TRUSTED: contextvars.ContextVar[bool] = contextvars.ContextVar("lt_trusted", default=False)
READ_ROOTS = [Path("/")]

# Write sandbox. None = unrestricted (build mode, every pre-existing caller).
# A list of resolved roots confines Write/Edit to them. Set per call by
# execute(write_roots=...), never from model-supplied args.
_WRITE_ROOTS: contextvars.ContextVar = contextvars.ContextVar("lt_write_roots", default=None)


def _enforce_write(p: Path) -> None:
    roots = _WRITE_ROOTS.get()
    if roots is None:
        return
    import sandbox
    if not sandbox.is_within(p, roots):
        raise ToolDenied(sandbox.refusal(p, roots))

READ_DEFAULT_LIMIT = 2000
READ_LINE_CHARS = 2000
READ_MAX_CHARS = 40_000        # keep under the Claude CLI's MCP output cap (~25k tokens)
MAX_TEXT_FILE_BYTES = 200_000_000
SHELL_TRUSTED_MAX_TIMEOUT_S = 600.0


def _read_roots():
    return READ_ROOTS if _TRUSTED.get() else None


def _mark_read(p: Path) -> None:
    _READ_FILES.add(str(p))
    try:
        _READ_STAMP[str(p)] = p.stat().st_mtime_ns
    except OSError:
        pass


def _check_fresh(p: Path) -> None:
    """Refuse to overwrite a file that changed on disk after we read it."""
    seen = _READ_STAMP.get(str(p))
    if seen is None:
        return
    try:
        now = p.stat().st_mtime_ns
    except OSError:
        return
    if now != seen:
        raise ToolDenied(
            f"refused: {p} changed on disk since you last read it. Read it "
            f"again before modifying it."
        )


def _atomic_write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        if p.exists():
            shutil.copymode(p, tmp)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def _guess_mime(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp"}[ext]
    return None


def _is_probably_image(path: Path) -> bool:
    return _guess_mime(path) is not None


def _is_pdf(path: Path) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    mime, _ = mimetypes.guess_type(str(path))
    return mime == "application/pdf"


def _pdftotext(p: Path):
    """Reading-order text via poppler, or None when unavailable / no text layer."""
    if not _have("pdftotext"):
        return None
    try:
        r = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    text = r.stdout.decode("utf-8", "replace")
    pages = max(1, text.count("\f"))
    if len(text.strip()) / pages < PDF_TEXT_LAYER_MIN_CHARS:
        return None
    return pages, text.replace("\f", "\n\n--- page break ---\n\n")


def _extract_pdf_text(p: Path) -> dict:
    _read_bounded(p, MAX_PDF_BYTES)
    got = _pdftotext(p)
    if got is not None:
        pages, text = got
        return {"pages": pages, "chars": len(text), "body": text}
    import pypdf

    reader = pypdf.PdfReader(p)
    n_pages = len(reader.pages)
    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            page_texts.append("")

    total_chars = sum(len(t) for t in page_texts)
    avg_chars = total_chars / n_pages if n_pages else 0
    ocr_note = ""

    if avg_chars < PDF_TEXT_LAYER_MIN_CHARS:
        try:
            import pytesseract
            from pdf2image import convert_from_path

            ocr_pages = n_pages if n_pages <= PDF_OCR_PAGE_CAP else PDF_OCR_PAGE_CAP
            images = convert_from_path(str(p), last_page=ocr_pages)
            ocr_texts = [pytesseract.image_to_string(img) for img in images]
            if any(t.strip() for t in ocr_texts):
                page_texts = ocr_texts
                total_chars = sum(len(t) for t in ocr_texts)
                if n_pages > PDF_OCR_PAGE_CAP:
                    ocr_note = (f"\n\n[OCR ran on the first {PDF_OCR_PAGE_CAP} of "
                                f"{n_pages} pages -- capped for time]")
                else:
                    ocr_note = "\n\n[extracted via OCR -- no text layer in the PDF]"
            else:
                ocr_note = "\n\n[OCR ran but found no text -- likely blank/graphic pages]"
        except ImportError:
            ocr_note = ("\n\n[no text layer, and pytesseract/pdf2image aren't "
                        "installed -- OCR fallback unavailable]")
        except Exception as exc:
            ocr_note = (f"\n\n[no text layer, and OCR fallback failed: "
                        f"{type(exc).__name__}: {exc} -- likely missing the "
                        f"tesseract-ocr / poppler-utils system packages]")

    body = "\n\n".join(f"--- page {i} ---\n{t}" for i, t in enumerate(page_texts, 1))
    return {"pages": n_pages, "chars": total_chars, "body": body + ocr_note}


def fs_read_image(path: str) -> dict:
    """Read one image file and return a data URL payload."""
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    if not _is_probably_image(p):
        raise ValueError(f"not an image file (need .png/.jpg/.webp/.gif/.bmp): {p}")
    data = _read_bounded(p, MAX_IMAGE_BYTES)
    if len(data) == 0:
        raise ValueError(f"empty image: {p}")
    mime = _guess_mime(p) or "image/png"
    b64 = base64.b64encode(data).decode()
    _READ_FILES.add(str(p))
    return {"ok": True, "tool": "ReadImage", "path": str(p),
            "media_type": mime, "bytes": len(data),
            "data_url": f"data:{mime};base64,{b64}",
            "base64": b64}


def fs_read_image_scaled(path: str, max_b64: int = 700_000, max_side: int = 1568) -> dict:
    """Image for the in-process Claude bridge: any readable path, shrunk until
    its base64 fits the (raised, but still finite) channel budget."""
    import io
    p = _resolve(path, roots=READ_ROOTS)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if not _is_probably_image(p):
        raise ValueError(f"not an image file: {p}")
    data = _read_bounded(p, 50_000_000)
    if not data:
        raise ValueError(f"empty image: {p}")
    mime = _guess_mime(p) or "image/png"
    _mark_read(p)
    if len(data) * 4 // 3 + 4 <= max_b64:
        return {"mime": mime, "data": base64.b64encode(data).decode(), "note": ""}
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    orig = img.size
    side = max_side
    while True:
        work = img.copy()
        work.thumbnail((side, side))
        buf = io.BytesIO()
        if work.mode in ("RGBA", "LA", "P") and side >= 800:
            work.convert("RGBA").save(buf, "PNG", optimize=True)
            out_mime = "image/png"
        else:
            work.convert("RGB").save(buf, "JPEG", quality=85 if side >= 800 else 70)
            out_mime = "image/jpeg"
        enc = base64.b64encode(buf.getvalue()).decode()
        if len(enc) <= max_b64 or side <= 200:
            return {"mime": out_mime, "data": enc,
                    "note": f"downscaled from {orig[0]}x{orig[1]} to fit the tool channel"}
        side = int(side * 0.75)


def _read_bounded(path: Path, cap: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"file too large (cap {cap} bytes): {path}")
    return data


def _format_window(picked: list[str], start: int, total: int, header: str = "") -> dict:
    if total == 0:
        return {"lines": "0-0 of 0", "content": header + "(empty file)"}
    if start > total:
        return {"lines": f"{start}-{start} of {total}",
                "content": header + f"(file has only {total} lines)"}
    rows, chars = [], 0
    for i, line in enumerate(picked, start):
        if len(line) > READ_LINE_CHARS:
            line = line[:READ_LINE_CHARS] + " ...[line truncated]"
        row = f"{i:6d}\t{line}"
        if rows and chars + len(row) > READ_MAX_CHARS:
            break
        rows.append(row)
        chars += len(row) + 1
    end = start + len(rows) - 1
    body = header + "\n".join(rows)
    if end < total:
        body += (f"\n[showing lines {start}-{end} of {total}; call Read again "
                 f"with offset={end + 1} to continue]")
    return {"lines": f"{start}-{end} of {total}", "content": body}


def fs_read(path: str, offset: int = 1, limit: int | None = None) -> dict:
    p = _resolve(path, roots=_read_roots())
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    if _is_probably_image(p):
        raise ValueError(f"{p} looks like an image -- use ReadImage instead of Read.")
    start = max(1, int(offset or 1))
    lim = READ_DEFAULT_LIMIT if limit is None else max(1, int(limit))
    if _is_pdf(p):
        extracted = _extract_pdf_text(p)
        _mark_read(p)
        lines = extracted["body"].split("\n")
        header = (f"[PDF: {extracted['pages']} page(s), {extracted['chars']} chars "
                  f"extracted; figures not included]\n")
        win = _format_window(lines[start - 1:start - 1 + lim], start, len(lines), header)
        return {"ok": True, "tool": "Read", "path": str(p), **win}
    if p.stat().st_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(f"file too large (cap {MAX_TEXT_FILE_BYTES} bytes): {p}")
    picked, total = [], 0
    with p.open("rb") as f:
        if b"\0" in f.read(8192):
            raise ValueError(f"{p} looks like a binary file; use Bash (xxd, strings, file) instead.")
        f.seek(0)
        for raw in f:
            total += 1
            if total >= start and len(picked) < lim:
                picked.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
    _mark_read(p)
    return {"ok": True, "tool": "Read", "path": str(p), **_format_window(picked, start, total)}


def fs_write(path: str, content: str) -> dict:
    p = _resolve(path)
    _enforce_write(p)
    b = str(content).encode("utf-8")
    if len(b) > MAX_WRITE_BYTES:
        raise ValueError(f"content too large ({len(b)} bytes)")
    existed = p.exists()
    if existed:
        if str(p) not in _READ_FILES:
            raise ToolDenied(
                f"refused: {p} exists but has not been read this session. Read it "
                f"first -- overwriting a file blind is how work gets lost."
            )
        _check_fresh(p)
    _atomic_write(p, b)
    _mark_read(p)
    return {"ok": True, "tool": "Write", "path": str(p), "bytes": len(b),
            "created": not existed}


def fs_edit(path: str, old: str, new: str, replace_all: bool = False) -> dict:
    """Exact string replacement, atomic, refused if the file changed since read."""
    p = _resolve(path)
    _enforce_write(p)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if str(p) not in _READ_FILES:
        raise ToolDenied(f"refused: read {p} before editing it.")
    _check_fresh(p)
    if not old:
        raise ValueError("old string must not be empty")
    if old == new:
        raise ValueError("old and new are identical; nothing to change")
    text = p.read_bytes().decode("utf-8")
    n = text.count(old)
    if n == 0:
        hint = ""
        if old.strip() and old.strip() in text:
            hint = " (the text exists but surrounding whitespace/indentation differs)"
        raise ValueError("old string not found (whitespace must match exactly)" + hint)
    if n > 1 and not replace_all:
        raise ValueError(
            f"old string appears {n} times; pass replace_all or include more "
            f"surrounding context to make it unique"
        )
    out = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    _atomic_write(p, out.encode("utf-8"))
    if p.read_bytes().decode("utf-8") != out:
        raise IOError(f"Edit verify failed: change did not persist at {p}")
    _mark_read(p)
    return {"ok": True, "tool": "Edit", "path": str(p), "replaced": n if replace_all else 1}


def _expand_braces(pat: str) -> list[str]:
    """'*.{py,md}' -> ['*.py', '*.md'] (nested ok); python's glob has no braces."""
    m = re.search(r"\{([^{}]*)\}", pat)
    if not m:
        return [pat]
    out: list[str] = []
    for alt in m.group(1).split(","):
        for x in _expand_braces(pat[:m.start()] + alt + pat[m.end():]):
            if x not in out:
                out.append(x)
    return out


def fs_glob(pattern: str, path: str | None = None, max_results: int = 200) -> dict:
    """Glob with real glob semantics; rg enumerates, newest first."""
    root = _resolve(path, roots=_read_roots()) if path else HOME
    pat = pattern.lstrip("/") if not os.path.isabs(pattern) else pattern
    matches: list[Path] = []
    if not os.path.isabs(pat):
        # walk only the literal directory prefix ("cchat/*.py" -> scan cchat/)
        parts = pat.split("/")
        pre = []
        while len(parts) > 1 and not any(c in parts[0] for c in "*?[]{"):
            pre.append(parts.pop(0))
        if pre:
            root = root.joinpath(*pre)
            pat = "/".join(parts)
            if not root.is_dir():
                return {"ok": True, "tool": "Glob", "count": 0, "files": []}
    if _have("rg") and hasattr(_globmod, "translate"):
        rxs = [re.compile(_globmod.translate(x, recursive=True, include_hidden=True))
               for x in _expand_braces(pat)]
        res = exec_argv(["rg", "--files", "--hidden", "-g", "!.git", "--no-messages",
                         str(root)], timeout_s=30, max_output_bytes=40_000_000)
        for line in res.get("stdout", "").splitlines():
            rel = line if os.path.isabs(pat) else os.path.relpath(line, root)
            if any(rx.match(rel) for rx in rxs):
                matches.append(Path(line))
    else:
        for m in root.glob(pat):
            if m.is_file():
                matches.append(m)
            if len(matches) >= max_results * 4:
                break
    def _mt(x):
        try:
            return x.stat().st_mtime
        except OSError:
            return 0
    matches.sort(key=_mt, reverse=True)
    hits = [str(m) for m in matches[:max_results]]
    out = {"ok": True, "tool": "Glob", "count": len(hits), "files": hits}
    if len(matches) > max_results:
        out["files"] = hits + [f"... {len(matches) - max_results} more not shown; narrow the pattern"]
    return out


def fs_grep(pattern: str, path: str | None = None, glob: str | None = None,
            max_results: int = 100, *, output_mode: str = "content",
            ignore_case: bool = False, line_numbers: bool = True,
            after: int | None = None, before: int | None = None,
            context: int | None = None, multiline: bool = False,
            type: str | None = None, head_limit: int | None = None,
            offset: int = 0) -> dict:
    """Search via ripgrep, falling back to GNU grep (reduced feature set)."""
    target = str(_resolve(path, roots=_read_roots()) if path else HOME)
    if output_mode not in ("content", "files_with_matches", "count"):
        raise ValueError("output_mode must be content, files_with_matches or count")
    limit = int(head_limit) if head_limit is not None else int(max_results)
    if _have("rg"):
        argv = ["rg", "--no-heading", "--color", "never", "--hidden",
                "-g", "!.git", "--with-filename", "--no-messages"]
        if output_mode == "files_with_matches":
            argv.append("-l")
        elif output_mode == "count":
            argv.append("-c")
        else:
            if line_numbers:
                argv.append("--line-number")
            for flag, val in (("-A", after), ("-B", before), ("-C", context)):
                if val:
                    argv += [flag, str(int(val))]
        if ignore_case:
            argv.append("-i")
        if multiline:
            argv += ["-U", "--multiline-dotall"]
        if type:
            argv += ["--type", type]
        if glob:
            argv += ["--glob", glob]
        argv += ["--", pattern, target]
    else:
        argv = ["grep", "-rn" if output_mode == "content" else "-rl"]
        if ignore_case:
            argv.append("-i")
        if glob:
            argv += [f"--include={glob}"]
        argv += ["-e", pattern, target]
    res = exec_argv(argv, timeout_s=30, max_output_bytes=4_000_000)
    if res.get("timed_out") or (res.get("exit_code") not in (0, 1) and not res.get("truncated")):
        raise RuntimeError(res.get("stderr") or "search did not complete")
    lines = res.get("stdout", "").splitlines()
    total = len(lines)
    lines = lines[int(offset or 0):][:limit]
    text = "\n".join(lines) or "(no matches)"
    if total > len(lines) + int(offset or 0):
        text += f"\n[{total - int(offset or 0) - len(lines)} more lines; raise head_limit or use offset]"
    return {"ok": True, "tool": "Grep", "count": len(lines), "matches": text}


# web


_HAS_DDGS = False
try:  # noqa: BLE001 - availability probe, not a code path
    import ddgs as _ddgs_probe  # type: ignore[import-not-found]
    _HAS_DDGS = True
except Exception:
    pass


def web_search(query: str, count: int = 5) -> dict:
    if not _HAS_DDGS:
        raise RuntimeError("ddgs not installed (pip install ddgs)")
    n = max(1, min(int(count or 5), 10))
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is empty")
    from ddgs import DDGS
    with DDGS() as ddgs:
        rows = list(ddgs.text(q, max_results=n))
    if not rows:
        return {"ok": True, "tool": "WebSearch", "query": q, "content": "(no results)"}
    lines = []
    for r in rows[:n]:
        title = (r.get("title") or "").strip()
        href = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        lines.append(f"- {title}\n  {href}\n  {body[:420]}")
    return {"ok": True, "tool": "WebSearch", "query": q, "content": "\n\n".join(lines)}


def web_fetch(url: str, max_chars: int = 12000) -> dict:
    u = str(url or "").strip()
    if not u:
        raise ValueError("url is empty")
    if not (u.startswith("http://") or u.startswith("https://")):
        raise ValueError("url must be http(s)")
    import httpx as _hx
    heads = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64) Domicile/1.0"}
    try:
        if "opencode.ai" in _hx.URL(u).host:
            heads["x-opencode-session"] = "cchat-webfetch"
            heads["x-opencode-client"] = "cchat"
    except Exception:
        pass
    cap = max(800, min(int(max_chars or 12000), 40000))
    r = _hx.get(u, headers=heads, timeout=15, follow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "html" not in ctype and "text" not in ctype and "xml" not in ctype:
        body = r.text[:2000]
        return {"ok": True, "tool": "WebFetch", "url": u,
                "content": f"[non-text content-type {ctype!r}]\n{body[:800]}"}
    html = r.text
    try:
        import lxml.html as _lh
        doc = _lh.fromstring(html)
        for bad in doc.xpath("//script | //style | //nav | //header | //footer | //noscript | //iframe"):
            try:
                bad.drop_tree()
            except Exception:
                pass
        text = doc.text_content() or ""
    except Exception:
        import re as _re
        t = _re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=_re.I | _re.S)
        t = _re.sub(r"<style[^>]*>.*?</style>", " ", t, flags=_re.I | _re.S)
        t = _re.sub(r"<[^>]+>", " ", t)
        text = t
    import re as _re2
    text = _re2.sub(r"[ \t]+", " ", text)
    text = _re2.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.strip() for ln in text.splitlines())
    text = _re2.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        text = html[:cap]
    if len(text) > cap:
        keep = cap // 2
        text = text[:keep] + f"\n\n... [{len(text) - cap} chars trimmed] ...\n\n" + text[-keep:]
    return {"ok": True, "tool": "WebFetch", "url": u, "content": text}


# shell

def CL_ENV_FILE() -> Path:
    """The install's .env (API keys). Masked unreadable inside the Bash sandbox."""
    import credentials
    return credentials.ENV_PATH


def exec_argv(argv: list[str], *, timeout_s: float = SHELL_TIMEOUT_S,
              max_output_bytes: int = SHELL_MAX_OUTPUT,
              cwd: str | None = None, stdin: str | None = None,
              _cancel: threading.Event | None = None,
              guard: bool = True, drain_overflow: bool = False,
              confine: list | None = None) -> dict:
    """Run a program directly. No shell. Ported from freeclaw shell.py:57.

    `confine` (a list of resolved write roots) runs it inside bubblewrap with
    only those roots writable. None = unconfined (build mode)."""
    if isinstance(argv, str):
        argv = shlex.split(argv)
    argv = [str(a) for a in argv]
    if guard:
        guard_argv(argv)

    to_s = float(timeout_s)
    max_b = int(max_output_bytes)
    if not math.isfinite(to_s) or to_s <= 0 or max_b <= 0:
        raise ValueError("timeout_s must be finite and positive; max_output_bytes must be > 0")
    work_dir = str(_resolve(cwd)) if cwd else str(HOME)
    if confine is not None:
        import sandbox
        if not cwd and confine:
            work_dir = str(confine[0])
        try:
            argv = sandbox.bwrap_argv(argv, confine, work_dir,
                                      mask=[CL_ENV_FILE()])
        except sandbox.SandboxUnavailable as exc:
            raise ToolDenied(str(exc))

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, cwd=work_dir, env=_clean_env(),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise ValueError(f"command not found: {argv[0]}")

    sel = selectors.DefaultSelector()
    out_b, err_b = bytearray(), bytearray()
    truncated = timed_out = False
    try:
        for stream, label in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, selectors.EVENT_READ, label)
        pending_input = memoryview((stdin or "").encode("utf-8"))
        if proc.stdin is not None:
            if pending_input:
                os.set_blocking(proc.stdin.fileno(), False)
                sel.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                proc.stdin.close()
        while True:
            if proc.poll() is not None and not sel.get_map():
                break
            if time.monotonic() - start >= to_s:
                timed_out = True
                break
            if _cancel is not None and _cancel.is_set():
                break
            for key, _ in sel.select(timeout=0.05):
                try:
                    if key.data == "stdin":
                        written = os.write(key.fd, pending_input[:4096])
                        pending_input = pending_input[written:]
                        if not pending_input:
                            sel.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    chunk = os.read(key.fd, 8192)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buf = out_b if key.data == "stdout" else err_b
                # drain mode: per-stream budgets so noisy stdout can't hide stderr
                remaining = (max_b - len(buf)) if drain_overflow else max_b - len(out_b) - len(err_b)
                buf.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated = True
                    if not drain_overflow:
                        break
            if truncated and not drain_overflow:
                break
    finally:
        if proc.poll() is None or sel.get_map():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sel.close()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        rc = proc.wait()

    return {
        "ok": True, "tool": "Bash", "argv": argv, "cwd": work_dir,
        "exit_code": rc,
        "stdout": out_b.decode("utf-8", errors="replace")[:max_b],
        "stderr": err_b.decode("utf-8", errors="replace")[:max_b],
        "truncated": truncated, "drained": bool(truncated and drain_overflow),
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - start, 2),
    }


# background tasks


from paths import BGTASK_DIR

BGTASK_LOG_TAIL_CHARS = 4000

_BG_PROCS: dict[str, "subprocess.Popen"] = {}


def _bgtask_meta_path(task_id: str) -> Path:
    return BGTASK_DIR / f"{task_id}.json"


def _bgtask_log_path(task_id: str) -> Path:
    return BGTASK_DIR / f"{task_id}.log"


def spawn_bg(argv: list[str], cwd: str | None = None,
             label: str | None = None, confine: list | None = None) -> dict:
    """Start a program detached and return immediately with a task id.

    `confine` (resolved write roots) runs it in bubblewrap: only those roots
    writable, .env masked, network ON (jobs fetch feeds and call APIs), and not
    tied to cchat's lifetime. None = unconfined (full profile)."""
    import json as _json

    if isinstance(argv, str):
        argv = shlex.split(argv)
    argv = [str(a) for a in argv]
    guard_argv(argv)

    work_dir = str(_resolve(cwd)) if cwd else str(HOME)
    shown_argv = list(argv)               # what the user sees, not the bwrap wrapper
    if confine is not None:
        import sandbox
        if not cwd and confine:
            work_dir = str(confine[0])
        try:
            argv = sandbox.bwrap_argv(argv, confine, work_dir, network=True,
                                      mask=[CL_ENV_FILE()], die_with_parent=False)
        except sandbox.SandboxUnavailable as exc:
            raise ToolDenied(str(exc))
    BGTASK_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"{int(time.time())}-{os.urandom(3).hex()}"
    log_path = _bgtask_log_path(task_id)
    exit_path = BGTASK_DIR / f"{task_id}.exit"
    runner = Path(__file__).with_name("bgtask_runner.py")

    lf = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(runner), str(exit_path), *argv],
            cwd=work_dir, env=_clean_env(),
            stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except FileNotFoundError:
        lf.close()
        raise ValueError(f"command not found: {argv[0]}")
    finally:
        lf.close()

    meta = {
        "id": task_id,
        "label": (label or " ".join(shown_argv))[:200],
        "argv": argv,
        "sandboxed": confine is not None,
        "cwd": work_dir,
        "pid": proc.pid,
        "started": time.time(),
        "started_iso": dt_now_iso(),
        "status": "running",
        "exit_code": None,
        "finished": None,
        "notified": False,
    }
    _BG_PROCS[task_id] = proc
    _bgtask_meta_path(task_id).write_text(_json.dumps(meta, indent=2) + "\n")
    return {
        "ok": True, "tool": "ToolWake", "id": task_id, "pid": proc.pid,
        "label": meta["label"], "log": str(log_path),
        "content": (
            f"Started [{task_id}] pid {proc.pid}: {meta['label']}\n"
            f"Running detached. Domicile will start a new turn with the result "
            f"when it exits -- do not poll it. Output streams to {log_path} "
            f"if you want to peek with Read."
        ),
    }


def dt_now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# dispatch

PHASE_READONLY = ["Read", "Glob", "Grep", "WebSearch", "WebFetch", "ReadImage",
                  "SetWakeup", "ExpressToPane", "CompactNow"]
PHASE_WRITE = PHASE_READONLY + ["Write", "Edit"]
PHASE_FULL = PHASE_WRITE + ["Bash", "ToolWake"]
# Write tier plus a shell that runs inside bubblewrap (see sandbox.py).
PHASE_SANDBOX = PHASE_WRITE + ["Bash"]
PHASE_WAKEUP = PHASE_SANDBOX + ["ToolWake"]

PHASE_ALIASES = {
    "none": [],
    "readonly": PHASE_READONLY,
    "write": PHASE_WRITE,
    "sandbox": PHASE_SANDBOX,
    "wakeup": PHASE_WAKEUP,
    "full": PHASE_FULL,
}


def resolve_phase(spec) -> list[str]:
    """Turn a per-mode tool spec into a concrete local-tool name list."""
    if spec is None:
        return list(PHASE_FULL)
    if isinstance(spec, str):
        return list(PHASE_ALIASES.get(spec, PHASE_READONLY))
    if isinstance(spec, (list, tuple)):
        known = set(PHASE_FULL) | set(_GT.ALL_TOOLS)
        return [t for t in spec if t in known]
    return list(PHASE_FULL)

_SCHEMAS: dict[str, dict] = {
    "Read": {
        "description": ("Read a file. Returns numbered lines. PDFs are handled "
                         "specially: text-layer extraction via pypdf, with an "
                         "OCR fallback (tesseract) for scanned/image-only PDFs."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path"},
            "offset": {"type": "integer", "description": "First line (1-based)"},
            "limit": {"type": "integer", "description": "Max lines"},
        }, "required": ["path"]},
    },
    "Glob": {
        "description": "Find files by glob pattern, newest first.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "e.g. **/*.py"},
            "path": {"type": "string", "description": "Directory to search"},
        }, "required": ["pattern"]},
    },
    "Grep": {
        "description": "Search file contents with ripgrep regex.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string", "description": "Filter, e.g. *.py"},
            "output_mode": {"type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                            "description": "Default content (matching lines)"},
            "ignore_case": {"type": "boolean", "description": "Case insensitive"},
            "line_numbers": {"type": "boolean", "description": "Show line numbers (content mode, default true)"},
            "after": {"type": "integer", "description": "Lines of context after each match"},
            "before": {"type": "integer", "description": "Lines of context before each match"},
            "context": {"type": "integer", "description": "Lines of context around each match"},
            "multiline": {"type": "boolean", "description": "Let patterns span lines"},
            "type": {"type": "string", "description": "ripgrep file type, e.g. py, js"},
            "head_limit": {"type": "integer", "description": "Max lines returned (default 250)"},
            "offset": {"type": "integer", "description": "Skip this many result lines first"},
        }, "required": ["pattern"]},
    },
    "Write": {
        "description": "Write a file. Must Read an existing file first.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
        }, "required": ["path", "content"]},
    },
    "Edit": {
        "description": "Replace an exact string in a file. Must be unique "
                       "unless replace_all is set.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        }, "required": ["path", "old", "new"]},
    },
    "Bash": {
        "description": (
            "Run a program. Takes an argv LIST, not a shell string: there is "
            "no shell, so pipes, redirects, globs and && do not work. Chain "
            "steps with separate calls."
        ),
        "parameters": {"type": "object", "properties": {
            "argv": {"type": "array", "items": {"type": "string"},
                     "description": 'e.g. ["ls","-la","/tmp"]'},
            "cwd": {"type": "string"},
            "timeout_s": {"type": "number"},
        }, "required": ["argv"]},
    },
    "ToolWake": {
        "description": (
            "Run a long program in the background and be woken when it exits "
            "(ssh install, training run, backfill). Takes an argv LIST, not a "
            "shell string: there is no shell and shell binaries are blocked, so "
            "pipes, redirects, globs and && are not interpreted and 'bash -c' "
            "is refused -- chain steps with separate calls, or use Bash, which "
            "does run through a shell. Returns immediately with a task id; "
            "Domicile starts a NEW turn carrying the exit code and output tail "
            "when it finishes. Use this instead of Bash + polling for anything "
            "over ~2 minutes. Output streams to a log you can Read while it "
            "runs."
        ),
        "parameters": {"type": "object", "properties": {
            "argv": {"type": "array", "items": {"type": "string"},
                     "description": 'e.g. ["ssh","box","bash /root/install.sh"]'},
            "cwd": {"type": "string"},
            "label": {"type": "string",
                      "description": "Short human label for the completion notice"},
        }, "required": ["argv"]},
    },
    "WebSearch": {
        "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Max results (default 5, max 10)"},
        }, "required": ["query"]},
    },
    "WebFetch": {
        "description": "Fetch a URL and return cleaned text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_chars": {"type": "integer", "description": "Max chars to return (default 12000)"},
        }, "required": ["url"]},
    },
    "ReadImage": {
        "description": "Read an image file from disk as base64. Use for png/jpg/webp/gif/bmp on the filesystem.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path to the image file"},
        }, "required": ["path"]},
    },
    "ExpressToPane": {
        "description": (
            "Replace the contents of the expression pane (bottom half of the "
            "collapsible side panel in the TUI). Persists across restarts and "
            "holds until you call this again -- a wall, not a scrollback. "
            "Content is either Markdown, or a small SVG document (starting with "
            "'<svg') which is rendered as terminal line art. Keep SVG simple: "
            "line, rect, circle, ellipse, polyline, polygon, path (M/L/H/V/Z), "
            "text. Use a viewBox."
        ),
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string",
                        "description": "Markdown text or an <svg> document"},
        }, "required": ["content"]},
    },
    "SetWakeup": {
        "description": (
            "Configure wakeup behavior. Set interval between wakeups (15-1440 minutes) "
            "and quiet hours (0-23). Quiet hours: start==end means disabled, "
            "start<end means same-day range, start>end wraps midnight."
        ),
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer",
                        "description": "Minutes between wakeups (15-1440, default 120)"},
            "quiet_start": {"type": "integer",
                            "description": "Quiet hours start hour (0-23, default 0=disabled)"},
            "quiet_end": {"type": "integer",
                          "description": "Quiet hours end hour (0-23, default 0=disabled)"},
        }},
    },
    "CompactNow": {
        "description": (
            "Ask for the conversation to be compacted (summarised and reseeded) "
            "as soon as this turn ends, instead of waiting for the history "
            "ceiling. Use it at a clean stopping point when context is heavy. "
            "Finish what you are saying first -- the compaction runs after your "
            "reply is delivered, and the next turn starts from the handover note."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

from paths import WAKEUP_CONFIG
WAKEUP_MIN_MINUTES = 15
WAKEUP_MAX_MINUTES = 1440
WAKEUP_DEFAULT_MINUTES = 120


def _set_wakeup(args: dict) -> str:
    """Set wakeup interval and quiet hours. Writes to data/wakeup.json."""
    import json as _json

    config = {"minutes": WAKEUP_DEFAULT_MINUTES, "quiet_start": 0, "quiet_end": 0}
    try:
        if WAKEUP_CONFIG.exists():
            config.update(_json.loads(WAKEUP_CONFIG.read_text()))
    except Exception:
        pass

    if "minutes" in args:
        m = int(args["minutes"])
        if m != 0 and m < WAKEUP_MIN_MINUTES:
            return f"ERROR: minimum interval is {WAKEUP_MIN_MINUTES} minutes (got {m})"
        if m > WAKEUP_MAX_MINUTES:
            return f"ERROR: maximum interval is {WAKEUP_MAX_MINUTES} minutes (got {m})"
        config["minutes"] = m

    if "quiet_start" in args:
        s = int(args["quiet_start"])
        if not 0 <= s <= 23:
            return "ERROR: quiet_start must be 0-23"
        config["quiet_start"] = s

    if "quiet_end" in args:
        e = int(args["quiet_end"])
        if not 0 <= e <= 23:
            return "ERROR: quiet_end must be 0-23"
        config["quiet_end"] = e

    WAKEUP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    WAKEUP_CONFIG.write_text(_json.dumps(config, indent=2) + "\n")

    parts = []
    m = config["minutes"]
    if m == 0:
        parts.append("wakeup disabled")
    else:
        parts.append(f"interval: {m}m")
    qs, qe = config["quiet_start"], config["quiet_end"]
    if qs == qe:
        parts.append("quiet hours: disabled")
    elif qs < qe:
        parts.append(f"quiet hours: {qs:02d}:00-{qe:02d}:00")
    else:
        parts.append(f"quiet hours: {qs:02d}:00-{qe:02d}:00 (wraps midnight)")
    return "wakeup configured: " + ", ".join(parts)


def _express_to_pane(args: dict) -> str:
    content = args.get("content") or ""
    if not content.strip():
        return "ERROR: content is empty"
    from expression import write_buffer
    write_buffer(content)
    kind = ("svg" if content.lstrip().startswith(("<svg", "<?xml"))
            else "markdown")
    return f"expression pane updated ({kind}, {len(content)} chars)"


def _grep_from_args(a: dict) -> dict:
    """Tolerates None for unset optionals (typed MCP wrappers pass them)."""
    def pick(k, default=None):
        v = a.get(k)
        return default if v is None else v
    return fs_grep(
        a["pattern"], a.get("path"), a.get("glob"),
        output_mode=pick("output_mode", "content"),
        ignore_case=bool(pick("ignore_case", False)),
        line_numbers=bool(pick("line_numbers", True)),
        after=a.get("after"), before=a.get("before"), context=a.get("context"),
        multiline=bool(pick("multiline", False)), type=a.get("type"),
        head_limit=int(pick("head_limit", 250)), offset=int(pick("offset", 0)))


def _compact_now(args: dict) -> str:
    """Drop the marker maybe_compact() looks for after each turn."""
    from paths import COMPACT_REQUEST
    COMPACT_REQUEST.parent.mkdir(parents=True, exist_ok=True)
    COMPACT_REQUEST.write_text("requested by tool\n")
    return ("Compaction requested. It runs when this turn ends, so finish your reply "
            "normally; the next turn starts from the handover note.")


_IMPL = {
    "Read": lambda a: fs_read(a["path"], a.get("offset", 1), a.get("limit")),
    "Glob": lambda a: fs_glob(a["pattern"], a.get("path")),
    "Grep": lambda a: _grep_from_args(a),
    "Write": lambda a: fs_write(a["path"], a["content"]),
    "Edit": lambda a: fs_edit(a["path"], a["old"], a["new"],
                              a.get("replace_all", False)),
    "Bash": lambda a: exec_argv(a["argv"], cwd=a.get("cwd"),
                                timeout_s=float(a.get("timeout_s")
                                                or SHELL_TIMEOUT_S)),
    "ToolWake": lambda a: spawn_bg(a["argv"], a.get("cwd"), a.get("label")),
    "WebSearch": lambda a: web_search(a["query"], int(a.get("count") or 5)),
    "WebFetch": lambda a: web_fetch(a["url"], int(a.get("max_chars") or 12000)),
    "ReadImage": lambda a: fs_read_image(a["path"]),
    "SetWakeup": _set_wakeup,
    "ExpressToPane": _express_to_pane,
    "CompactNow": _compact_now,
}


_TRUSTED_SCHEMAS = {
    "Read": {
        "description": (
            "Read a file from the local filesystem (any path). Text comes back "
            "as numbered lines, up to 2000 lines per call; pass offset/limit to "
            "page through longer files. PDFs are extracted to text (pdftotext, "
            "reading order; figures not included; OCR fallback for scans). "
            "Images (png/jpg/webp/gif/bmp) are returned as an image you can see."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path"},
            "offset": {"type": "integer", "description": "First line (1-based)"},
            "limit": {"type": "integer", "description": "Max lines (default 2000)"},
        }, "required": ["path"]},
    },
    "Bash": {
        "description": (
            "Run a shell command with bash -c, so pipes, redirects, heredocs, "
            "globs, && and $() all work. Credentials are stripped from the "
            "environment. Output beyond 60 KB is cut (the command still runs "
            "to completion). Default timeout 120 s, max 600; for anything "
            "longer use ToolWake. Each call starts fresh in the home directory "
            "(or `cwd`) -- no state carries over."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command line"},
            "cwd": {"type": "string"},
            "timeout_s": {"type": "number", "description": "Seconds (default 120, max 600)"},
        }, "required": ["command"]},
    },
}


def schemas(enabled: list[str], trusted: bool = False) -> list[dict]:
    """OpenAI function-tool definitions for the enabled subset."""
    out = []
    for n in enabled:
        if n not in _SCHEMAS:
            continue
        spec = dict(_SCHEMAS[n])
        if trusted and n in _TRUSTED_SCHEMAS:
            spec = dict(_TRUSTED_SCHEMAS[n])
        out.append({"type": "function", "function": {"name": n, **spec}})
    return out


_MUTATING_TOOLS = {"Write", "Edit", "SetWakeup", "ExpressToPane", "CompactNow"}

# Opt-in Google tools (google_tools.py). Not in any PHASE_* tier: a role gets
# them only when its tool spec lists them. Write actions are approval-gated there.
import google_tools as _GT
_SCHEMAS.update(_GT.SCHEMAS)
_IMPL.update(_GT.IMPL)
_MUTATING_TOOLS |= set(_GT.WRITE_TOOLS)
_MUTATION_LOCK = threading.Lock()


# Models trained on Claude Code's tools reach for these names; accept them.
_ARG_ALIASES = {"file_path": "path", "old_string": "old", "new_string": "new"}
_ALIASED_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep"}


def _normalize_args(name: str, args: dict) -> dict:
    if name not in _ALIASED_TOOLS or not isinstance(args, dict):
        return args
    out = dict(args)
    for alias, real in _ARG_ALIASES.items():
        if alias in out and real not in out:
            out[real] = out.pop(alias)
    return out


async def execute(name: str, args: dict, enabled: list[str], *, trusted: bool = False,
                  write_roots: list[Path] | None = None):
    """Local tool execution; returns the unrendered result.

    `trusted` is passed only by the in-process Claude bridge. It unlocks shell
    strings and whole-filesystem reads; writes stay confined to ALLOWED_ROOTS.
    """
    if name not in enabled:
        raise ToolDenied(f"tool {name!r} is not enabled in this phase.")
    if (name == "ToolWake" and write_roots is not None
            and not getattr(write_roots, "toolwake", False)):
        raise ToolDenied("ToolWake is not available under this permission profile. "
                         "Ask the user to run `python3 allow.py --profile wakeup` for "
                         "a role that needs it (the agent cannot do this itself).")
    args = _normalize_args(name, args)
    cancel = threading.Event()
    _tok = _TRUSTED.set(bool(trusted))
    _wtok = _WRITE_ROOTS.set(write_roots)

    def invoke():
        def call():
            if cancel.is_set():
                return "ERROR: cancelled before execution."
            if name == "Bash" and args.get("command") is not None:
                if not _TRUSTED.get():
                    raise ToolDenied("shell strings are not enabled for this provider; "
                                     "pass argv as a list instead.")
                return exec_argv(
                    ["bash", "-c", str(args["command"])], cwd=args.get("cwd"),
                    timeout_s=min(float(args.get("timeout_s") or SHELL_TIMEOUT_S),
                                  SHELL_TRUSTED_MAX_TIMEOUT_S),
                    guard=False, drain_overflow=True, _cancel=cancel,
                    confine=write_roots)
            if name == "Bash":
                return exec_argv(args["argv"], cwd=args.get("cwd"),
                                 timeout_s=float(args.get("timeout_s") or SHELL_TIMEOUT_S),
                                 _cancel=cancel, confine=write_roots)
            if name == "ToolWake":
                return spawn_bg(args["argv"], args.get("cwd"), args.get("label"),
                                confine=write_roots)
            return _IMPL[name](args)

        if name in _MUTATING_TOOLS:
            with _MUTATION_LOCK:
                return call()
        return call()

    worker = asyncio.create_task(asyncio.to_thread(invoke))
    _TRUSTED.reset(_tok)   # the task already holds its own copy of the context
    _WRITE_ROOTS.reset(_wtok)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        worker.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        raise


async def dispatch(name: str, args: dict, enabled: list[str], *,
                   write_roots: list[Path] | None = None) -> str:
    """Run one tool call and return a string for the `tool` message."""
    if name not in enabled:
        return f"ERROR: tool {name!r} is not enabled in this phase."
    try:
        result = await execute(name, args, enabled, write_roots=write_roots)
        text = result if isinstance(result, str) else _render(result)
    except ToolDenied as exc:
        return f"DENIED: {exc}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    if len(text) > TOOL_RESULT_CHARS:
        keep = TOOL_RESULT_CHARS // 2
        text = (text[:keep] + f"\n\n... [{len(text) - TOOL_RESULT_CHARS} chars "
                f"trimmed] ...\n\n" + text[-keep:])
    return text


def _render(result: dict) -> str:
    if result.get("tool") == "ReadImage":
        if "data_url" in result or "base64" in result:
            return (f"Image {result.get('path')} ({result.get('media_type')}, "
                    f"{result.get('bytes')} bytes) -- sent as image_url block "
                    f"after this tool result. Describe what you see.")
        return "; ".join(f"{k}={v}" for k, v in result.items() if k != "ok")
    if "content" in result:
        return result["content"]
    if "matches" in result:
        return result["matches"]
    if "files" in result:
        return "\n".join(result["files"]) or "(no matches)"
    if result.get("tool") == "Bash":
        bits = []
        if result.get("stdout"):
            bits.append(result["stdout"].rstrip())
        if result.get("stderr"):
            bits.append("[stderr]\n" + result["stderr"].rstrip())
        if result.get("timed_out"):
            bits.append(f"[timed out after {result['seconds']}s]")
        if result.get("drained"):
            bits.append("[output truncated at the limit; process ran to completion]")
        elif result.get("truncated"):
            bits.append("[output limit reached; process stopped]")
        if result.get("exit_code"):
            bits.append(f"[exit {result['exit_code']}]")
        return "\n".join(bits) or "(no output)"
    return "; ".join(f"{k}={v}" for k, v in result.items() if k != "ok")
