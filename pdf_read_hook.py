"""PreToolUse hook: make the Claude-path built-in Read work on PDFs.

The SDK's built-in Read ships PDFs as base64 page images through a stdio JSON
channel capped at 1 MiB ("JSON message exceeded maximum buffer size of
1048576 bytes"), so a modest PDF kills the turn with no result. This hook
rewrites a Read of any *.pdf to a cached pdftotext dump (reading order, no
-layout) so the model just calls Read on the PDF and it works.

Falls through untouched (original behaviour) when pdftotext is missing, fails,
or the PDF has no text layer (scans) -- the model can still try images then.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "data" / "pdf_text"
MIN_CHARS_PER_PAGE = 40  # below this average, treat as a scan and fall through
TIMEOUT_S = 60


def pdf_to_text_file(pdf: Path) -> tuple[Path | None, str]:
    """Return (txt_path, note). txt_path is None when we should fall through."""
    try:
        st = pdf.stat()
    except OSError as exc:
        return None, f"stat failed: {exc}"
    key = hashlib.sha1(f"{pdf.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    out = CACHE_DIR / f"{pdf.stem[:60]}-{key}.txt"
    if out.exists() and out.stat().st_size > 0:
        return out, "cached"
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                           timeout=TIMEOUT_S)
    except FileNotFoundError:
        return None, "pdftotext not installed"
    except subprocess.TimeoutExpired:
        return None, "pdftotext timed out"
    if r.returncode != 0:
        return None, f"pdftotext exit {r.returncode}"
    text = r.stdout.decode("utf-8", "replace")
    pages = max(1, text.count("\f"))
    if len(text.strip()) / pages < MIN_CHARS_PER_PAGE:
        return None, "no usable text layer (scan?)"
    header = (f"[Text extracted from {pdf.name} by pdftotext ({pages} page(s); "
              f"pages separated by form-feed; figures not included).]\n\n")
    out.write_text(header + text, encoding="utf-8")
    return out, "converted"


async def pdf_read_hook(input_data, tool_use_id, context):
    if input_data.get("tool_name") != "Read":
        return {}
    ti = dict(input_data.get("tool_input") or {})
    fp = ti.get("file_path") or ""
    if not fp.lower().endswith(".pdf"):
        return {}
    pdf = Path(fp).expanduser()
    if not pdf.is_file():
        return {}
    txt, note = pdf_to_text_file(pdf)
    if txt is None:
        return {}
    ti["file_path"] = str(txt)
    ti.pop("pages", None)  # page ranges mean nothing on a text file
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": ti,
        "additionalContext": (
            f"Read was redirected from {pdf.name} to its pdftotext dump "
            f"({txt}) because the PDF reader hits a 1 MiB cap. Use offset/limit "
            "on that path for long documents; figures are not included."),
    }}
