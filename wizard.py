"""Domicile setup wizard: Textual shell over the wizard_* / import_* logic.

All validation and writes live in wizard_provider.py, wizard_persona.py and the import_* modules;
this file only collects answers and shows errors. Steps (local deployment only):
Welcome -> Provider (3.3) -> Context window (3.3a) -> Import memories (3.4)
-> Persona (3.5) -> Edit files -> Discord (optional) -> Google (optional) -> Service (3.6)
-> Private backup (optional) -> Smoke test (3.7) -> Done.

Edit later:  python3 wizard.py --edit
Run:  python3 wizard.py        (writes the REAL config -- use on a fresh install)
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Header, Input, Label, RadioButton, RadioSet, Static

import import_chatgpt, import_claude, import_common, import_mem0, import_sql, import_text
import wizard_discord as dd
import paths
import wizard_google as wg
import wizard_backup as wb
import wizard_provider as wp
import wizard_persona as persona
import wizard_service as service
import wizard_smoke as smoke
from wizard_editor import ConfigEditorScreen, EditorScreen

ERR = "error"


def _field(label: str, wid: str, *, placeholder: str = "", password: bool = False):
    yield Label(label)
    yield Input(placeholder=placeholder, id=wid, password=password)


class Step(Screen):
    """Base: a title, body widgets, an error line, a Continue button."""
    TITLE = ""
    BINDINGS = [("escape", "app.quit", "Quit")]
    DEFAULT_CSS = """
    #bar { dock: bottom; height: auto; max-height: 8; padding: 0 1;
           border-top: solid $primary; background: $surface; }
    #msg { height: auto; }
    """

    def body(self) -> ComposeResult:
        return iter(())

    def extra_buttons(self) -> ComposeResult:
        return iter(())

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield from self.body()      # the Header already shows TITLE
        with Vertical(id="bar"):      # pinned: always visible, even on 80x24
            yield Static("", id="msg")
            yield from self.extra_buttons()
            yield Button("Continue", id="next", variant="primary")

    def say(self, text: str, *, error: bool = False) -> None:
        self.query_one("#msg", Static).update(f"[red]{text}[/red]" if error else text)

    def val(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "next":
            self.advance()

    def advance(self) -> None:           # subclasses validate, apply, then push
        raise NotImplementedError

    def go(self, screen: "Step") -> None:
        self.app.push_screen(screen)


class Welcome(Step):
    TITLE = "Welcome to Domicile"

    def body(self):
        yield Static(
            "Domicile is a home for one continuous agent: one memory, one history,\n"
            "running on your own machine. This takes a few minutes. Nothing leaves\n"
            "your computer except calls to the model provider you choose, and\n"
            "backups to a private GitHub repo if you turn them on.\n\n"
            f"Installed in: {paths.ROOT}\n"
            f"Your data:    {paths.CCHAT_DATA}\n\n"
            "Esc quits at any time; finished steps stay saved.")

    def advance(self):
        self.go(ProviderStep())


class ProviderStep(Step):
    TITLE = "Step 1 - Model provider"
    KINDS = ("anthropic", "openai", "api")

    def body(self):
        yield Static("anthropic = Claude via the installed `claude` CLI (no key).\n"
                     "openai = ChatGPT login via the `codex` CLI (no key).\n"
                     "api = any OpenAI-compatible endpoint (needs URL, model, key).")
        with RadioSet(id="kind"):
            yield RadioButton("anthropic", value=True)
            yield RadioButton("openai")
            yield RadioButton("api")
        yield from _field("Name (optional)", "name", placeholder="e.g. deepseek")
        yield from _field("Base URL (api only)", "url", placeholder="https://host/v1")
        yield from _field("Model id (api; optional for openai)", "model")
        yield from _field("API key (api only)", "key", password=True)

    def advance(self):
        kind = self.KINDS[self.query_one("#kind", RadioSet).pressed_index]
        try:
            res = wp.build_provider(kind, self.val("name") or None,
                                    base_url=self.val("url"),
                                    models=self.val("model") or None,
                                    api_key=self.val("key"))
            wp.apply_provider(res, make_active=True)
        except wp.WizardError as e:
            return self.say(str(e), error=True)
        self.app.answers["provider"] = res["name"]
        self.go(ContextStep())


class ContextStep(Step):
    TITLE = "Step 2 - Context window"

    def __init__(self, *, revisit: bool = False):
        super().__init__()
        self.revisit = revisit

    def on_mount(self):
        if self.revisit:
            ctx = (wp.CL.load_config().get("context") or {})
            for field, key in (("ceiling", "compact_ceiling_tokens"),
                               ("floor", "compact_floor_tokens"),
                               ("mem1", "mem1_total")):
                value = ctx.get(key)
                if value is not None:
                    self.query_one(f"#{field}", Input).value = str(value)

    def body(self):
        yield Static("How many tokens can your model hold at once? The wizard sets\n"
                     "when to compact and how much memory to load from that.")
        yield from _field("Context window (tokens)", "window", placeholder="200000")
        yield Static("[dim]Advanced - fill ceiling to override the derived values:[/dim]")
        yield from _field("Ceiling", "ceiling")
        yield from _field("Floor (optional)", "floor")
        yield from _field("Memory budget (optional)", "mem1")

    def _num(self, wid):
        v = self.val(wid)
        return int(v.replace(",", "")) if v else None

    def advance(self):
        try:
            if self.val("ceiling"):
                out = wp.apply_context(ceiling=self._num("ceiling"),
                                       floor=self._num("floor"),
                                       mem1_total=self._num("mem1"))
            else:
                out = wp.apply_context(self._num("window"))
        except ValueError:
            return self.say("Numbers only, please.", error=True)
        except wp.WizardError as e:
            return self.say(str(e), error=True)
        if out.get("warning") and not getattr(self, "_warned", False):
            self._warned = True      # show once; second Continue accepts it
            return self.say(out["warning"] + "\nPress Continue again to keep it.", error=True)
        self.app.answers["context"] = out["context"]
        self.go(PersonaStep() if self.revisit else ImportStep())


class ImportStep(Step):
    TITLE = "Step 3 - Import memories (optional)"
    KINDS = ("skip", "claude", "chatgpt", "text", "sql", "mem0")

    def body(self):
        yield Static("Bring existing history in. Run as many imports as you like,\n"
                     "then press Continue. Imported logs are ingested overnight.")
        with RadioSet(id="src"):
            yield RadioButton("skip", value=True)
            yield RadioButton("Claude export (.zip / folder / conversations.json)")
            yield RadioButton("ChatGPT export (.zip / folder / conversations-*.json)")
            yield RadioButton("Text / markdown files or folder")
            yield RadioButton("SQLite database (map columns below)")
            yield RadioButton("mem0 memories (.json)")
        yield from _field("Path", "path")
        yield Static("[dim]SQLite only - table and column names:[/dim]")
        for wid, lab in (("table", "Table"), ("tcol", "Text column"),
                         ("ts", "Timestamp column"), ("who", "Sender column (optional)"),
                         ("convo", "Conversation column (optional)")):
            yield from _field(lab, wid)

    def extra_buttons(self):
        yield Button("Run this import", id="run")

    def on_button_pressed(self, ev):
        if ev.button.id == "run":
            self.run_import()
        else:
            super().on_button_pressed(ev)

    def run_import(self):
        kind = self.KINDS[self.query_one("#src", RadioSet).pressed_index]
        if kind == "skip":
            return self.say("Nothing selected.")
        path = self.val("path")
        try:
            if kind == "claude":
                r = import_claude.run(path)
            elif kind == "chatgpt":
                r = import_chatgpt.run(path)
            elif kind == "text":
                r = import_text.run(path)
            elif kind == "mem0":
                r = import_mem0.run(path, self.app.mem0_add)
            else:
                r = import_sql.run(path, self.val("table"), self.val("tcol"),
                                   self.val("ts"), self.val("who") or None,
                                   self.val("convo") or None)
        except import_common.ImportError_ as e:
            return self.say(str(e), error=True)
        self.app.answers.setdefault("imports", []).append((kind, r))
        self.say(f"Imported ({kind}): {r}")

    def advance(self):
        self.go(PersonaStep())


class PersonaStep(Step):
    TITLE = "Step 4 - Persona"
    KINDS = ("grow", "define")

    def body(self):
        caps = persona.file_caps()
        yield Static(
            "These three files are loaded on every turn. Keep them short and durable:\n"
            "identity.md - who the agent is, its values and voice.\n"
            "user.md - what the agent should know about you.\n"
            "practices.md - how you work together: rules and habits.\n"
            "The wizard also creates goals.md, open.md and reflections.md: the agent's\n"
            "own notes (what it wants, what is unfinished, what it has thought about).\n"
            "They start as empty skeletons and are loaded every turn too.\n\n"
            f"Current caps: identity {caps['identity.md']:,}, user {caps['user.md']:,}, "
            f"practices {caps['practices.md']:,} tokens.\n"
            "If an old persona file won't fit, go back to Context and give mem1 "
            "more room, set a per-file cap in config.json, or trim that file. "
            "The wizard will refuse an oversized file, never cut it silently.\n\n"
            "Grow: the agent chooses its own name in your first conversation. "
            "Define: you name it now."
        )
        with RadioSet(id="persona-kind"):
            yield RadioButton("Grow", value=True)
            yield RadioButton("Define")
        yield from _field("Your name", "user-name")
        yield from _field("Agent name (Define only)", "agent-name")
        yield Static("Optional existing markdown files to copy into the new persona:")
        yield from _field("Identity / soul.md path", "identity-source")
        yield from _field("User notes path", "user-source")
        yield from _field("Working practices path", "practices-source")
        yield Static("[dim]Existing persona files will not be overwritten.[/dim]")

    def extra_buttons(self):
        yield Button("Adjust mem1 room", id="adjust-memory")

    def on_button_pressed(self, ev):
        if ev.button.id == "adjust-memory":
            self.go(ContextStep(revisit=True))
        else:
            super().on_button_pressed(ev)

    def advance(self):
        kind = self.KINDS[self.query_one("#persona-kind", RadioSet).pressed_index]
        sources = {"identity.md": self.val("identity-source"),
                   "user.md": self.val("user-source"),
                   "practices.md": self.val("practices-source")}
        try:
            result = persona.apply(kind, self.val("user-name"),
                                   self.val("agent-name"), sources)
        except persona.PersonaError as e:
            return self.say(str(e), error=True)
        self.app.answers["persona"] = result
        self.go(EditStep())


class EditStep(Step):
    TITLE = "Step 5 - Review and edit your persona files"

    def body(self):
        yield Static(
            "Optional. Open a file to trim or rewrite it. A live counter shows tokens "
            "against the file's cap; saving is refused while it is over.\n"
            "Ctrl+S saves, Esc returns here.")
        for name in persona.CORE:
            yield Button(f"Edit {name}", id=f"edit-{name.removesuffix('.md')}")

    def on_button_pressed(self, ev):
        bid = ev.button.id or ""
        if bid.startswith("edit-"):
            self.app.push_screen(EditorScreen(bid[5:] + ".md"))
        else:
            super().on_button_pressed(ev)

    def advance(self):
        self.go(DiscordStep())


class DiscordStep(Step):
    TITLE = "Step 6 - Connect Discord (optional)"
    DEFAULT_CSS = """
    #dc-row { height: auto; }
    #dc-row Button { margin-right: 1; }
    """

    def __init__(self):
        super().__init__()
        self.connected = None        # DMResult once the handshake has succeeded and saved

    def body(self):
        yield Static(
            "Talk to your agent from Discord, on your phone or desktop. Optional, about "
            "five minutes. To skip, leave the token empty and press Continue.\n\n"
            f"1. Open {dd.DEVELOPER_PORTAL}\n"
            "   and click New Application. Name it after your agent if you like.\n"
            "2. Open Bot in the left menu. Under Privileged Gateway Intents turn ON\n"
            "   Message Content Intent, then Save Changes. Without it the bot is deaf.\n"
            "3. Press Reset Token, copy it (shown once) and paste it below.\n"
            "4. Press Check bot. It shows an invite link. Open it, choose a server you\n"
            "   are in (a private one you make just for this is fine) and Authorize.\n"
            "5. Press Wait for my DM. In Discord, open the bot's profile from that\n"
            "   server's member list and send it a direct message. Anything works.\n\n"
            "Your message is how the wizard learns your Discord user ID, so only you can\n"
            "talk to the agent. The token is stored in .env, readable only by you.")
        yield from _field("Bot token", "dc-token", password=True)
        yield Static("", id="dc-status")

    def extra_buttons(self):
        with Horizontal(id="dc-row"):
            yield Button("Check bot", id="dc-check")
            yield Button("Wait for my DM", id="dc-wait")

    def on_input_changed(self, ev: Input.Changed) -> None:
        if ev.input.id == "dc-token":
            self.connected = None    # a different token needs its own handshake

    def on_button_pressed(self, ev):
        if ev.button.id == "dc-check":
            self.run_worker(self._check(), exclusive=True, group="dc")
        elif ev.button.id == "dc-wait":
            self.run_worker(self._wait(), exclusive=True, group="dc")
        else:
            super().on_button_pressed(ev)

    async def _check(self):
        self.say("Connecting to Discord...")
        try:
            info = await dd.check_bot(self.val("dc-token"))
        except dd.DiscordSetupError as e:
            return self.say(str(e), error=True)
        servers = ", ".join(info.guilds) or "none yet"
        self.query_one("#dc-status", Static).update(
            f"Bot: {info.name}\nIn servers: {servers}\n\nInvite link:\n{info.invite_url}")
        self.say("Token works and Message Content Intent is on. Invite the bot, "
                 "then press Wait for my DM.")

    async def _wait(self):
        token = self.val("dc-token")
        self.say("Connecting to Discord...")
        try:
            got = await dd.wait_for_dm(token, on_waiting=self.say)
            dd.save(token, got.user_id)
        except dd.DiscordSetupError as e:
            return self.say(str(e), error=True)
        except (ValueError, OSError) as e:
            return self.say(f"Couldn't save the Discord settings: {e}", error=True)
        self.connected = got
        self.app.answers["discord"] = {"user": got.username, "user_id": got.user_id}
        greet = "" if got.reply_sent else " (the bot couldn't reply, but the connection is saved)"
        self.say(f"Connected: {got.username}. Saved{greet}. Press Continue.")

    def advance(self):
        if self.connected is None:
            if self.val("dc-token"):
                return self.say("A token is entered but not connected yet. Press Wait for my "
                                "DM, or clear the token to skip Discord.", error=True)
            self.app.answers["discord"] = "skipped"
        self.go(GoogleStep())


class GoogleStep(Step):
    TITLE = "Step 7 - Connect Google (optional)"
    DEFAULT_CSS = """
    #g-row { height: 1; }
    #g-row Button { margin-right: 1; }
    """

    def __init__(self):
        super().__init__()
        self.signin = None          # wg.SignIn for the current attempt
        self.client = None          # parsed client_secret.json
        self.creds = None           # credentials once Google has accepted the sign-in
        self.account = ""
        self.verified = None        # frozenset of services whose live check passed
        self.chat_code = ""
        self.chat_link = None       # wg.ChatLink once the Chat handshake has succeeded
        self._cancel = threading.Event()

    def body(self):
        yield Static(
            "Give your agent a Google account so it can read and send mail, keep a "
            "calendar, and (Workspace only) talk to you in Google Chat. Optional, about "
            "15 minutes, free. To skip, leave the path below empty and press Continue.\n\n"
            "The agent acts AS the account you sign in with: it can read and send that "
            "account's mail. Create a separate account for the agent; don't use your "
            "own. Changing that account's password later cancels the sign-in, and you "
            "would repeat this step.\n\n"
            "1. Signed in as the AGENT's account, open\n"
            f"   {wg.CONSOLE}/projectcreate  and create a project (any name).\n"
            f"2. Open {wg.CONSOLE}/apis/library and Enable the API for each box\n"
            "   you tick below: Gmail API, Google Calendar API, Blogger API, Google Chat API.\n"
            "3. Open Google Auth Platform (the menu, top left) and press Get started.\n"
            "   App name: anything. Audience: External. Contact email: the agent's.\n"
            "4. On the Audience page press Publish app, so the status reads In production.\n"
            "   You do not need Google's verification. If you leave it in Testing, Google\n"
            "   cancels the sign-in every 7 days.\n"
            "5. On the Clients page choose Create client, Application type: Desktop app,\n"
            "   then download the JSON. Copy it to this machine and enter its path below.\n"
            "6. Press Get sign-in link and open the link in any browser, a phone is fine.\n"
            "   Sign in as the agent's account. Google warns the app is unverified (it is\n"
            "   your own): Advanced, then Go to (your app). Approve every box. The last\n"
            "   page says it can't be reached: that is expected. Copy the whole address\n"
            "   from the address bar, paste it below, and press Finish sign-in.\n\n"
            "The sign-in is stored in data/google/token.json, readable only by you.")
        yield from _field("Path to the downloaded client JSON", "g-client",
                          placeholder=str(wg.GOOGLE_DIR / "client_secret.json"))
        yield Checkbox("Gmail", True, id="g-gmail")
        yield Checkbox("Calendar", True, id="g-calendar")
        yield Checkbox("Blogger", False, id="g-blogger")
        yield Checkbox("Google Chat (needs a Google Workspace account)", False, id="g-chat")
        yield Static("", id="g-link")
        yield from _field("Paste the address from the browser here", "g-paste")
        yield Static("", id="g-status")

    def extra_buttons(self):
        with Horizontal(id="g-row"):
            yield Button("Get sign-in link", id="g-start", compact=True)
            yield Button("Finish sign-in", id="g-finish", compact=True)
            yield Button("Re-check", id="g-recheck", compact=True)
            yield Button("Chat handshake", id="g-chat-go", compact=True)

    def selected(self) -> list:
        return [k for k in wg.SERVICES if self.query_one(f"#g-{k}", Checkbox).value]

    def on_checkbox_changed(self, ev: Checkbox.Changed) -> None:
        self.verified = None        # a different selection needs its own check

    def on_screen_suspend(self) -> None:
        self._cancel.set()

    def on_button_pressed(self, ev):
        bid = ev.button.id
        if bid == "g-start":
            self._start()
        elif bid == "g-finish":
            self.run_worker(self._finish(), exclusive=True, group="g")
        elif bid == "g-recheck":
            self.run_worker(self._recheck(), exclusive=True, group="g")
        elif bid == "g-chat-go":
            self.run_worker(self._chat(), exclusive=True, group="g")
        else:
            super().on_button_pressed(ev)

    def _start(self):
        try:
            client = wg.load_client_secret(self.val("g-client"))
            signin = wg.SignIn(client, self.selected())
            url = signin.start()
        except wg.GoogleSetupError as e:
            return self.say(str(e), error=True)
        self.client, self.signin = client, signin
        self.query_one("#g-link", Static).update(f"Sign-in link:\n{url}")
        try:
            self.app.copy_to_clipboard(url)
            copied = " It was also copied to your clipboard."
        except Exception:
            copied = ""
        self.say("Open the link, sign in as the agent's account and approve."
                 f"{copied} Then paste the final address below and press Finish sign-in.")

    def _show(self, checks) -> bool:
        lines = [f"{'OK  ' if c.ok else 'FAIL'} {c.label}: {c.detail}" for c in checks]
        self.query_one("#g-status", Static).update("\n".join(lines))
        return all(c.ok for c in checks)

    async def _verify(self) -> bool:
        wanted = self.selected()
        checks = await asyncio.to_thread(wg.verify, self.creds, wanted)
        self.account = wg.account_of(checks) or self.account
        if self.account:
            await asyncio.to_thread(wg.save_token, self.creds, None, self.account)
        ok = self._show(checks)
        self.verified = frozenset(wanted) if ok else None
        return ok

    async def _finish(self):
        if self.signin is None:
            return self.say("Press Get sign-in link first.", error=True)
        self.say("Finishing sign-in...")
        try:
            self.creds = await asyncio.to_thread(self.signin.finish, self.val("g-paste"))
            await asyncio.to_thread(wg.save_token, self.creds, self.client)
        except wg.GoogleSetupError as e:
            return self.say(str(e), error=True)
        except OSError as e:
            return self.say(f"Couldn't save the sign-in: {e}", error=True)
        self.say("Signed in. Checking each service...")
        if await self._verify():
            self.say("Everything works. " + ("Now press Chat handshake. " if "chat" in
                     self.selected() and self.chat_link is None else "") + "Press Continue.")
        else:
            self.say("Signed in, but some services failed (see above). Fix them, then press "
                     "Re-check, or untick a service you don't need.", error=True)

    async def _recheck(self):
        if self.creds is None:
            try:
                self.creds = await asyncio.to_thread(wg.load_token)
            except wg.GoogleSetupError:
                return self.say("Sign in first: Get sign-in link, then Finish sign-in.", error=True)
            except Exception as e:
                return self.say(f"Couldn't read the saved sign-in: {e}", error=True)
        self.say("Checking...")
        if await self._verify():
            self.say("Everything selected works. Press Continue.")
        else:
            self.say("Still failing (see above).", error=True)

    async def _chat(self):
        if "chat" not in self.selected():
            return self.say("Tick Google Chat first.", error=True)
        if self.creds is None or self.verified is None or "chat" not in self.verified:
            return self.say("Finish sign-in (or Re-check) with Google Chat ticked first.", error=True)
        self.chat_code = self.chat_code or wg.new_chat_code()
        self._cancel.clear()
        status = self.query_one("#g-status", Static)
        status.update(f"From YOUR Google account open chat.google.com, start a direct "
                      f"message with the agent's address, and send exactly:\n\n  {self.chat_code}")
        self.say("Waiting for your message (up to 3 minutes)...")
        try:
            link = await asyncio.to_thread(
                wg.wait_for_chat_message, self.creds, self.chat_code, cancel=self._cancel)
            greeted = await asyncio.to_thread(wg.complete_chat, self.creds, link)
        except wg.GoogleSetupError as e:
            return self.say(str(e), error=True)
        except OSError as e:
            return self.say(f"Couldn't save the Chat settings: {e}", error=True)
        self.chat_link = link
        who = link.sender_display or link.sender
        self.say(f"Chat connected: only {who} can message the agent. Saved"
                 f"{'' if greeted else ' (the greeting failed, the link is saved)'}. "
                 "Press Continue.")

    def advance(self):
        if self.signin is None and self.creds is None:
            if self.val("g-client"):
                return self.say("A client file is entered but you haven't signed in. Finish "
                                "the sign-in, or clear the path to skip Google.", error=True)
            self.app.answers["google"] = "skipped"
            return self.go(ServiceStep())
        wanted = frozenset(self.selected())
        if self.creds is None:
            return self.say("Press Finish sign-in first, or clear the path to skip Google.",
                            error=True)
        if self.verified != wanted:
            return self.say("The selected services haven't all passed their check. "
                            "Press Re-check.", error=True)
        if "chat" in wanted and self.chat_link is None:
            return self.say("Google Chat is ticked but not connected. Press Chat handshake, "
                            "or untick it and press Re-check.", error=True)
        self.app.answers["google"] = {"account": self.account, "services": sorted(wanted)}
        self.go(ServiceStep())


class ServiceStep(Step):
    TITLE = "Step 8 - Run in the background"

    def body(self):
        yield Static(
            "cchat can run as a systemd user service inside tmux, restarting itself "
            "if it crashes, with a watchdog every 5 minutes. If you connected Gmail, a "
            "timer also checks the inbox every 5 minutes. Existing unit files are "
            "never replaced.\n\nAttach later with: tmux -L cchat attach -t main\n"
            "Starting it here is off by default; enable now, start after the smoke test.")
        with RadioSet(id="svc-kind"):
            yield RadioButton("Install units and enable (don't start yet)", value=True)
            yield RadioButton("Install, enable and start now")
            yield RadioButton("Skip - I'll run cchat by hand")
        yield Checkbox("Also check a few RSS feeds hourly; the agent reads what's new "
                       "(edit data/rss/feeds.txt)", False, id="svc-rss")

    def advance(self):
        idx = self.query_one("#svc-kind", RadioSet).pressed_index
        if idx == 2:
            self.app.answers["service"] = "skipped"
            return self.go(BackupStep())
        try:
            gmail = "gmail" in (self.app.answers.get("google") or {}).get("services", []) \
                if isinstance(self.app.answers.get("google"), dict) else False
            rss = self.query_one("#svc-rss", Checkbox).value
            service.install(Path(__file__).resolve().parent, gmail=gmail, rss=rss)
            note = service.enable(start=(idx == 1), gmail=gmail, rss=rss)
        except service.ServiceError as e:
            return self.say(str(e), error=True)
        self.app.answers["service"] = note
        self.go(BackupStep())


class BackupStep(Step):
    TITLE = "Step 9 - Private backup (optional)"

    def __init__(self):
        super().__init__()
        self.checked: tuple[str, str] | None = None
        self.info: wb.GitHubInfo | None = None

    def body(self):
        yield Static(
            "Your agent's memory and chats live on this machine. Domicile can copy "
            "them to a PRIVATE GitHub repository so you can restore them if this "
            "machine fails. This is optional. A private repo is access-controlled, "
            "not end-to-end encrypted; skip this if you don't want your personal "
            "data on GitHub. API tokens and .env are NOT included in the backup.\n\n"
            "1. On GitHub, create an empty PRIVATE repository under your own account.\n"
            "2. Create a fine-grained personal access token for that repository with "
            "Contents: read and write. Paste the token and repo name below.\n"
            "3. Press Check private repo, then Continue to save the settings. "
            "No backup runs during setup.\n\n"
            "To skip, leave both fields empty and press Continue.")
        yield from _field("Private GitHub repo name", "backup-repo",
                          placeholder="domicile-backup")
        yield from _field("GitHub personal access token", "backup-token",
                          password=True)
        yield Static("", id="backup-status")

    def extra_buttons(self):
        yield Button("Check private repo", id="backup-check")

    def on_input_changed(self, ev: Input.Changed) -> None:
        if ev.input.id in {"backup-repo", "backup-token"}:
            self.checked = None
            self.info = None

    def on_button_pressed(self, ev):
        if ev.button.id == "backup-check":
            self.run_worker(self._check(), exclusive=True, group="backup")
        else:
            super().on_button_pressed(ev)

    async def _check(self):
        self.checked = None
        self.info = None
        failed = [(name, detail) for name, ok, detail in wb.check_local() if not ok]
        if failed:
            return self.say("; ".join(f"{name}: {detail}" for name, detail in failed),
                            error=True)
        token, repo = self.val("backup-token"), self.val("backup-repo")
        self.say("Checking GitHub...")
        try:
            info = await asyncio.to_thread(wb.check_github, token, repo)
        except wb.BackupSetupError as e:
            return self.say(str(e), error=True)
        if not info.exists:
            return self.say(
                "That repository isn't visible to the token. Create it PRIVATE "
                "under your account, grant the token access, then check again.",
                error=True)
        self.checked = (token, repo)
        self.info = info
        self.query_one("#backup-status", Static).update(
            f"Verified: {info.owner}/{info.repo} is private.")
        self.say("Private repository verified. Press Continue to save. "
                 "No backup will run yet.")

    def advance(self):
        token, repo = self.val("backup-token"), self.val("backup-repo")
        if not token and not repo:
            self.app.answers["backup"] = "skipped"
            return self.go(SmokeStep())
        if self.checked != (token, repo) or self.info is None:
            return self.say(
                "Enter both fields and press Check private repo first, or "
                "clear both fields to skip backup.", error=True)
        try:
            wb.save(token, repo)
        except (wb.BackupSetupError, OSError, ValueError) as e:
            return self.say(f"Could not save backup settings: {e}", error=True)
        self.app.answers["backup"] = f"private GitHub repo {self.info.owner}/{self.info.repo}"
        self.go(SmokeStep())


class SmokeStep(Step):
    TITLE = "Step 10 - Smoke test"

    def body(self):
        yield Static("Offline checks of what you just set up. No model calls, no cost.")
        yield Static("", id="report")

    def on_mount(self):
        res = smoke.run(service_skipped=self.app.answers.get("service") == "skipped")
        self.app.answers["smoke"] = all(ok for _, ok, _ in res)
        self.query_one("#report", Static).update(smoke.format_report(res))

    def advance(self):
        self.go(Done())


def _discord_line(d) -> str:
    if isinstance(d, dict):
        return f"connected to {d.get('user')}"
    return d or "-"


def _google_line(g) -> str:
    if isinstance(g, dict):
        names = ", ".join(wg.SERVICES[k].label for k in g.get("services", []) if k in wg.SERVICES)
        return f"{g.get('account') or 'signed in'} ({names})"
    return g or "-"


def _persona_line(p) -> str:
    if not p:
        return "Persona: -"
    who = f"for {p.get('user_name')}" if p.get("user_name") else ""
    if p.get("mode") == "define":
        return f"Persona: {p.get('agent_name')} {who}".rstrip()
    return f"Persona: unnamed - it will choose its own name {who}".rstrip()


class Done(Step):
    TITLE = "Done (so far)"

    def body(self):
        a = self.app.answers
        yield Static(
            f"Provider: {a.get('provider', '-')}\n"
            f"Context: {a.get('context', '-')}\n"
            f"Imports: {len(a.get('imports', []))}\n"
            f"{_persona_line(a.get('persona'))}\n"
            f"Discord: {_discord_line(a.get('discord'))}\n"
            f"Google: {_google_line(a.get('google'))}\n\n"
            f"Service: {a.get('service', '-')}\n"
            f"Backup: {a.get('backup', '-')}\n\n"
            f"Smoke test: {'passed' if a.get('smoke') else 'see failures'}\n\n"
            "Start cchat, then say hello: with Grow, your agent picks its name.")

    def advance(self):
        self.app.exit(self.app.answers)


class EditMenu(Step):
    TITLE = "Edit your setup"

    def body(self):
        yield Static("Pick a file. Ctrl+S saves, Esc returns here. Memory files show a live "
                     "token budget; config.json must stay valid JSON.")
        for name in persona.CORE:
            yield Button(f"Edit {name}", id=f"edit-{name.removesuffix('.md')}")
        yield Button("Edit config.json (provider, context, caps)", id="edit-config")

    def on_button_pressed(self, ev):
        bid = ev.button.id or ""
        if bid == "edit-config":
            import config_loader as CL
            self.app.push_screen(ConfigEditorScreen(Path(CL.CONFIG_PATH)))
        elif bid.startswith("edit-"):
            self.app.push_screen(EditorScreen(bid[5:] + ".md"))
        else:
            super().on_button_pressed(ev)

    def advance(self):
        self.app.exit()


class EditApp(App):
    TITLE = "Domicile - edit"

    def on_mount(self):
        self.push_screen(EditMenu())


class WizardApp(App):
    TITLE = "Domicile setup"

    def __init__(self, mem0_add=None):
        super().__init__()
        self.answers: dict = {}
        self.mem0_add = mem0_add      # None -> real mem0_core; tests inject a stub

    def on_mount(self):
        self.push_screen(Welcome())


if __name__ == "__main__":
    import sys
    (EditApp() if "--edit" in sys.argv[1:] else WizardApp()).run()
