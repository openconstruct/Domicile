"""Quick text editor for the three core mem1 files, with a live token budget.

Logic (check/save) is plain Python so it tests without a terminal; EditorScreen is
the thin Textual shell. Saving is refused when the file or the mem1 total would go
over its cap -- the editor never truncates.
"""
from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, Static, TextArea

import mem1
import wizard_persona as persona


def check(name: str, text: str, caps: dict[str, int], others: dict[str, str]) -> dict:
    """Budget status for `text` as file `name`, given the other files' contents."""
    cost = mem1.text_tokens(text)
    total = cost + sum(mem1.text_tokens(t) for t in others.values())
    limit = sum(caps.values())
    cap = caps[name]
    problems = []
    if cost > cap:
        problems.append(f"{name} is {cost - cap:,} tokens over its {cap:,} cap")
    if total > limit:
        problems.append(f"mem1 total is {total - limit:,} tokens over its {limit:,} cap")
    return {"cost": cost, "cap": cap, "total": total, "limit": limit,
            "problems": problems, "ok": not problems}


def read_all(mem_dir: Path) -> dict[str, str]:
    out = {}
    for name in persona.CORE:
        p = mem_dir / name
        try:
            out[name] = p.read_text(encoding="utf-8") if p.exists() else ""
        except (OSError, UnicodeError) as exc:
            raise persona.PersonaError(f"Cannot read {p}: {exc}") from exc
    return out


def save(name: str, text: str, caps: dict[str, int], mem_dir: Path) -> dict:
    """Validate then write atomically. Raises PersonaError and writes nothing if over cap."""
    if name not in persona.CORE:
        raise persona.PersonaError(f"{name} is not an editable persona file.")
    others = {n: t for n, t in read_all(mem_dir).items() if n != name}
    status = check(name, text, caps, others)
    if not status["ok"]:
        raise persona.PersonaError("Not saved: " + "; ".join(status["problems"]) + ".")
    mem_dir.mkdir(parents=True, exist_ok=True)
    tmp = mem_dir / f".{name}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, mem_dir / name)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise persona.PersonaError(f"Could not save {name}: {exc}") from exc
    return status


class EditorScreen(Screen):
    BINDINGS = [("ctrl+s", "save", "Save"), ("escape", "back", "Back")]
    DEFAULT_CSS = "#budget { height: auto; padding: 0 1; } TextArea { height: 1fr; }"

    def __init__(self, name: str, mem_dir: Path | None = None, cfg: dict | None = None):
        super().__init__()
        self.fname = name
        self.mem_dir = mem_dir if mem_dir is not None else mem1.MEM1_DIR
        self.caps = persona.file_caps(cfg)
        self.disk = read_all(self.mem_dir)
        self.clean = self.disk[name]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="budget")
        yield TextArea(self.clean, id="buffer")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Edit {self.fname}"
        self.query_one("#buffer", TextArea).focus()
        self._refresh_budget()

    def _refresh_budget(self, note: str = "") -> None:
        text = self.query_one("#buffer", TextArea).text
        others = {n: t for n, t in self.disk.items() if n != self.fname}
        s = check(self.fname, text, self.caps, others)
        color = "green" if s["ok"] else "red"
        dirty = " [yellow]*unsaved[/yellow]" if text != self.clean else ""
        line = (f"[{color}]{self.fname}: ~{s['cost']:,} / {s['cap']:,} tokens[/{color}]"
                f"   mem1 total ~{s['total']:,} / {s['limit']:,}{dirty}")
        if s["problems"]:
            line += "\n[red]" + "; ".join(s["problems"]) + " - cut before saving.[/red]"
        if note:
            line += "\n" + note
        self.query_one("#budget", Static).update(line)

    def on_text_area_changed(self, _ev: TextArea.Changed) -> None:
        self._refresh_budget()

    def action_save(self) -> None:
        text = self.query_one("#buffer", TextArea).text
        try:
            save(self.fname, text, self.caps, self.mem_dir)
        except persona.PersonaError as e:
            return self._refresh_budget(f"[red]{e}[/red]")
        self.clean = text
        self.disk[self.fname] = text
        self._refresh_budget("[green]Saved.[/green]")

    def action_back(self) -> None:
        self.app.pop_screen()


class ConfigEditorScreen(Screen):
    """Raw editor for config.json: saves only if it parses as a JSON object."""
    BINDINGS = [("ctrl+s", "save", "Save"), ("escape", "back", "Back")]
    DEFAULT_CSS = "#cmsg { height: auto; padding: 0 1; } TextArea { height: 1fr; }"

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Restart cchat after saving for changes to apply.", id="cmsg")
        yield TextArea(self.path.read_text(encoding="utf-8") if self.path.exists() else "{}\n",
                       language="json", id="buffer")
        yield Footer()

    def action_save(self) -> None:
        import json
        text = self.query_one("#buffer", TextArea).text
        try:
            if not isinstance(json.loads(text), dict):
                raise ValueError("top level must be an object")
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except (ValueError, OSError) as e:
            return self.query_one("#cmsg", Static).update(f"[red]Not saved: {e}[/red]")
        self.query_one("#cmsg", Static).update("[green]Saved. Restart cchat to apply.[/green]")

    def action_back(self) -> None:
        self.app.pop_screen()
