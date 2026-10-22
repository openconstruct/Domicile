import asyncio, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wizard_editor as E, wizard_persona as P
from textual.app import App
from textual.widgets import TextArea, Static

def mk():
    d = Path(tempfile.mkdtemp())
    for n in P.CORE: (d/n).write_text(f"# {n}\nhello\n")
    return d

def test_logic():
    d = mk(); caps = {"identity.md": 50, "user.md": 50, "practices.md": 50}
    E.save("user.md", "# new\n", caps, d); assert (d/"user.md").read_text() == "# new\n"
    try: E.save("user.md", "word " * 500, caps, d); assert False
    except P.PersonaError: pass
    assert (d/"user.md").read_text() == "# new\n"
    assert not list(d.glob(".*.tmp"))
    try: E.save("goals.md", "x", caps, d); assert False
    except P.PersonaError: pass

def test_ui():
    d = mk()
    class A(App):
        def on_mount(self): self.push_screen(E.EditorScreen("identity.md", d, {}))
    async def run():
        app = A()
        async with app.run_test() as pilot:
            ta = app.screen.query_one("#buffer", TextArea)
            ta.text = "# Identity\nchanged\n"
            await pilot.pause(); await pilot.press("ctrl+s"); await pilot.pause()
            assert (d/"identity.md").read_text() == "# Identity\nchanged\n"
            ta.text = "word " * 100000
            await pilot.pause(); await pilot.press("ctrl+s"); await pilot.pause()
            assert (d/"identity.md").read_text() == "# Identity\nchanged\n"
            await pilot.press("escape")
    asyncio.run(run())

def test_config_editor():
    import json
    d = mk(); p = d/"config.json"; p.write_text('{"a": 1}')
    class A(App):
        def on_mount(self): self.push_screen(E.ConfigEditorScreen(p))
    async def run():
        app = A()
        async with app.run_test() as pilot:
            ta = app.screen.query_one("#buffer", TextArea)
            ta.text = '{"a": '; await pilot.pause(); await pilot.press("ctrl+s"); await pilot.pause()
            assert json.loads(p.read_text()) == {"a": 1}
            ta.text = '{"a": 2}'; await pilot.pause(); await pilot.press("ctrl+s"); await pilot.pause()
            assert json.loads(p.read_text()) == {"a": 2}
    asyncio.run(run())

if __name__ == "__main__":
    test_logic(); test_ui(); test_config_editor(); print("wizard_editor tests passed")
