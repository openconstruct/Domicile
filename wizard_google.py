"""Google connector setup: no UI here, so the TUI and a future web step share it.

The agent signs in to its own Google account so it can use Gmail, Calendar, Blogger
and (Workspace only) Google Chat. Three pieces:

  SignIn          OAuth that works on a machine with no browser: the user opens a link
                  anywhere, approves, and pastes back the address the browser ends on.
  verify()        one cheap, read-only call per selected service, with errors worded
                  for a person ("the Gmail API isn't switched on", not a stack trace).
  wait_for_chat_message()
                  the Google Chat handshake. The user DMs the agent account a one-time
                  code; whoever sent that code becomes the only allowed sender.

save_token() writes data/google/token.json, the file google_auth/google_client.py reads.

Later, when Google stops accepting the saved sign-in (password change, access removed,
six idle months), no need to rerun the wizard:
    python3 wizard_google.py check     is the saved sign-in still good?
    python3 wizard_google.py signin    sign in again, same services; the client comes from the
                                       saved token.json, so no other file is needed
google-auth libraries are imported lazily: a user who skips Google never needs them.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.parse import parse_qs, urlparse

import paths

GOOGLE_DIR = paths.GOOGLE_DIR
CONSOLE = "https://console.cloud.google.com"
# Desktop-app clients may use any loopback port. Nothing listens here: the browser
# shows "can't connect" and the address bar holds the code, which the user pastes.
REDIRECT_URI = "http://localhost:8765/"
POLL_INTERVAL = 35          # seconds; the value chat.json has always carried
GREETING = ("Connected. Only you can talk to this agent from Google Chat now. "
            "Finish setup in the wizard, then start Domicile and say hello here.")


class GoogleSetupError(Exception):
    """A setup problem, worded for the person running the wizard."""


class Service(NamedTuple):
    key: str
    label: str
    scopes: tuple
    api: str            # the name in Cloud Console's API library
    version: str
    needs_workspace: bool = False


_A = "https://www.googleapis.com/auth/"
SERVICES = {s.key: s for s in (
    Service("gmail", "Gmail", (_A + "gmail.modify",), "gmail.googleapis.com", "v1"),
    Service("calendar", "Calendar", (_A + "calendar",), "calendar-json.googleapis.com", "v3"),
    Service("blogger", "Blogger", (_A + "blogger",), "blogger.googleapis.com", "v3"),
    Service("chat", "Google Chat", (_A + "chat.messages", _A + "chat.spaces"),
            "chat.googleapis.com", "v1", needs_workspace=True),
)}


class Check(NamedTuple):
    key: str
    label: str
    ok: bool
    detail: str


class ChatLink(NamedTuple):
    space: str
    sender: str
    sender_display: str
    message_name: str
    create_time: str


# --- client_secret.json ----------------------------------------------------------

def load_client_secret(path: str) -> dict:
    """Read a downloaded client_secret JSON and insist on a Desktop-app client."""
    raw = (path or "").strip().strip('"').strip("'")
    if not raw:
        raise GoogleSetupError("Enter the path of the client_secret JSON you downloaded.")
    file = Path(raw).expanduser()
    if not file.is_file():
        raise GoogleSetupError(f"No file at {file}.")
    try:
        data = json.loads(file.read_text())
    except (OSError, ValueError) as e:
        raise GoogleSetupError(f"{file.name} is not readable JSON: {e}")
    if not isinstance(data, dict):
        raise GoogleSetupError(f"{file.name} is not a Google client file.")
    if "installed" not in data:
        kind = next(iter(data), "unknown")
        raise GoogleSetupError(
            f"This is a '{kind}' client; Domicile needs a Desktop app client. In Cloud "
            "Console open Google Auth Platform, Clients, Create client, and choose "
            "Application type: Desktop app.")
    inst = data["installed"]
    if not isinstance(inst, dict) or not inst.get("client_id") or not inst.get("client_secret"):
        raise GoogleSetupError(f"{file.name} is missing its client_id or client_secret. "
                               "Download the JSON again from the Clients page.")
    return data


# --- sign-in ---------------------------------------------------------------------

def scopes_for(keys) -> list:
    out = []
    for k in keys:
        if k not in SERVICES:
            raise GoogleSetupError(f"Unknown service: {k}")
        out += [s for s in SERVICES[k].scopes if s not in out]
    if not out:
        raise GoogleSetupError("Tick at least one service.")
    return out


def parse_redirect(pasted: str, state: str) -> str:
    """Pull the authorization code out of whatever the user pasted: the whole address
    (http://localhost:8765/?state=...&code=...), just its query string, or the bare code."""
    text = (pasted or "").strip().strip('"').strip("'").strip("<>")
    if not text:
        raise GoogleSetupError("Paste the address from the browser's address bar first.")
    if "://" in text or "code=" in text or "error=" in text:
        query = urlparse(text).query if "://" in text else text.lstrip("?")
        q = parse_qs(query)
        if q.get("error"):
            why = q["error"][0]
            raise GoogleSetupError(
                "You declined access on Google's page." if why == "access_denied"
                else f"Google reported an error: {why}. Press Get sign-in link and try again.")
        code = (q.get("code") or [""])[0]
        if not code:
            raise GoogleSetupError("That address has no code in it. Copy the whole address "
                                   "after approving, even though the page itself failed to load.")
        if (q.get("state") or [""])[0] != state:
            raise GoogleSetupError("That address came from an older sign-in link. Use the "
                                   "newest link, or press Get sign-in link again.")
        return code
    if " " in text or len(text) < 10:
        raise GoogleSetupError("That doesn't look like the address or the code.")
    return text


@contextlib.contextmanager
def _relaxed_scope():
    # Google lets people untick individual permissions. Without this oauthlib raises
    # "Scope has changed" instead of letting us report which ones were refused.
    old = os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE")
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OAUTHLIB_RELAX_TOKEN_SCOPE", None)
        else:
            os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = old


class SignIn:
    """One sign-in attempt. start() returns the link; finish() trades the pasted result
    for credentials. Each start() makes a fresh attempt, so older links are rejected."""

    def __init__(self, client_config: dict, services, *, flow_factory=None):
        self.client_config = client_config
        self.services = list(services)
        self.scopes = scopes_for(self.services)
        self._factory = flow_factory
        self.flow = None
        self.state = ""

    def _new_flow(self):
        if self._factory:
            return self._factory(self.client_config, self.scopes, REDIRECT_URI)
        try:
            from google_auth_oauthlib.flow import Flow
        except ImportError:
            raise GoogleSetupError(
                "The Google libraries aren't installed. Run: "
                "pip install google-auth-oauthlib google-api-python-client   then try again.")
        try:
            return Flow.from_client_config(self.client_config, scopes=self.scopes,
                                           redirect_uri=REDIRECT_URI)
        except ValueError:
            raise GoogleSetupError(
                "That client file isn't in the format Google downloads. Download it again "
                "from the Clients page (Google Auth Platform, Clients, your client, Download JSON).")

    def start(self) -> str:
        self.flow = self._new_flow()
        # prompt=consent makes Google issue a refresh token even on a repeat sign-in.
        url, self.state = self.flow.authorization_url(access_type="offline", prompt="consent")
        return url

    def finish(self, pasted: str):
        if self.flow is None:
            raise GoogleSetupError("Press Get sign-in link first.")
        code = parse_redirect(pasted, self.state)
        try:
            with _relaxed_scope():
                self.flow.fetch_token(code=code)
        except Exception as e:
            text = str(e)
            if "invalid_client" in text:
                raise GoogleSetupError(
                    "Google doesn't recognise this client. Download client_secret JSON again "
                    "from the Clients page of the project that has the consent screen.")
            if "invalid_grant" in text:
                raise GoogleSetupError(
                    "Google rejected the code. Codes work once and expire within minutes: "
                    "press Get sign-in link and go through it again.")
            raise GoogleSetupError(f"Couldn't finish sign-in: {type(e).__name__}: {text}")
        creds = self.flow.credentials
        if not getattr(creds, "refresh_token", None):
            raise GoogleSetupError(
                "Google signed you in but sent no refresh token, so the agent would lose "
                "access within the hour. Remove this app at myaccount.google.com/permissions "
                "and sign in again.")
        return creds


# --- saving ----------------------------------------------------------------------

def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def granted_scopes(creds) -> list:
    got = getattr(creds, "granted_scopes", None) or getattr(creds, "scopes", None) or []
    return sorted(got)


def token_json(creds, account: str = "") -> str:
    """creds.to_json(), but 'scopes' lists what Google actually granted rather than what
    was asked for (they differ when the user unticks a box), and google_client.py
    refreshes with exactly this list."""
    data = json.loads(creds.to_json())
    data["scopes"] = granted_scopes(creds)
    if account:
        data["account"] = account
    return json.dumps(data, indent=2) + "\n"


def save_token(creds, client_config: dict | None = None, account: str = "") -> Path:
    path = GOOGLE_DIR / paths.GOOGLE_TOKEN.name
    GOOGLE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(GOOGLE_DIR, 0o700)
    _write_private(path, token_json(creds, account))
    if client_config is not None:
        _write_private(GOOGLE_DIR / "client_secret.json", json.dumps(client_config, indent=2) + "\n")
    return path


def load_token():
    """Credentials from the saved token.json, using the scopes recorded in it."""
    from google.oauth2.credentials import Credentials
    path = GOOGLE_DIR / "token.json"
    if not path.is_file():
        raise GoogleSetupError("No saved Google sign-in yet.")
    return Credentials.from_authorized_user_file(str(path))


# --- verification ----------------------------------------------------------------

def _build(name: str, version: str, creds):
    from googleapiclient.discovery import build
    return build(name, version, credentials=creds, cache_discovery=False)


def _explain(exc: Exception, svc: Service) -> str:
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        HttpError = ()
    if type(exc).__name__ == "RefreshError":
        return (f"{svc.label}: Google no longer accepts the saved sign-in (the account's password "
                "changed, access was removed, or it sat unused for six months). "
                "Run: python3 wizard_google.py signin")
    if HttpError and isinstance(exc, HttpError):
        status = getattr(exc.resp, "status", 0)
        try:
            msg = json.loads(exc.content)["error"]["message"]
        except Exception:
            msg = str(exc)
        low = (msg + " " + str(getattr(exc, "content", b""))).lower()
        if status == 403 and ("has not been used" in low or "is disabled" in low
                              or "accessnotconfigured" in low):
            return (f"The {svc.label} API isn't switched on for your project. Open "
                    f"{CONSOLE}/apis/library/{svc.api} , press Enable, wait a minute, "
                    "then Re-check.")
        if status in (401, 403) and svc.needs_workspace:
            return (f"{svc.label} refused the request ({status}): {msg}. Google's docs say "
                    "the Chat API needs a Google Workspace account; a plain @gmail.com "
                    "account can't use it. Untick Google Chat to continue without it.")
        return f"{svc.label} refused the request ({status}): {msg}"
    return f"{svc.label}: {type(exc).__name__}: {exc}"


_PROBES = {
    "gmail": lambda s: s.users().getProfile(userId="me").execute().get("emailAddress", ""),
    "calendar": lambda s: s.calendarList().list(maxResults=1).execute() and "reachable",
    "blogger": lambda s: s.blogs().listByUser(userId="self").execute() and "reachable",
    "chat": lambda s: s.spaces().list(pageSize=1).execute() and "reachable",
}


def verify(creds, services, *, build: Callable | None = None) -> list:
    """One read-only call per service. Never raises for an API problem; that is a Check."""
    build = build or _build
    granted = set(granted_scopes(creds))
    out = []
    for key in services:
        svc = SERVICES[key]
        missing = [s for s in svc.scopes if s not in granted]
        if missing:
            out.append(Check(key, svc.label, False,
                             f"Permission for {svc.label} wasn't granted. Sign in again and "
                             "leave every box on Google's consent page ticked."))
            continue
        try:
            detail = _PROBES[key](build(svc.key, svc.version, creds))
            out.append(Check(key, svc.label, True, detail or "ok"))
        except Exception as e:
            out.append(Check(key, svc.label, False, _explain(e, svc)))
    return out


def account_of(checks) -> str:
    for c in checks:
        if c.key == "gmail" and c.ok and "@" in c.detail:
            return c.detail
    return ""


# --- Google Chat handshake -------------------------------------------------------

def new_chat_code() -> str:
    return "domicile-" + secrets.token_hex(3)


def _dm_spaces(chat) -> list:
    out, token = [], None
    while True:
        params = {"filter": 'spaceType = "DIRECT_MESSAGE"', "pageSize": 100}
        if token:
            params["pageToken"] = token
        page = chat.spaces().list(**params).execute()
        out += [s["name"] for s in page.get("spaces", []) if s.get("name")]
        token = page.get("nextPageToken")
        if not token:
            return out


def find_code(chat, code: str) -> ChatLink | None:
    want = code.lower()
    for space in _dm_spaces(chat):
        page = chat.spaces().messages().list(
            parent=space, pageSize=25, orderBy="createTime desc").execute()
        for m in page.get("messages", []):
            sender = m.get("sender") or {}
            if want in (m.get("text") or "").lower() and sender.get("type", "HUMAN") == "HUMAN":
                return ChatLink(space, sender.get("name", ""), sender.get("displayName", ""),
                                m.get("name", ""), m.get("createTime", ""))
    return None


def wait_for_chat_message(creds, code: str, *, timeout: float = 180.0, interval: float = 5.0,
                          on_waiting: Callable[[str], None] | None = None,
                          cancel=None, build: Callable | None = None,
                          clock=time.monotonic, sleep=time.sleep) -> ChatLink:
    """Poll until a direct message containing `code` shows up. Blocking: run in a thread.
    The code, not just 'any DM', is what proves the sender is the person at this keyboard."""
    chat = (build or _build)("chat", "v1", creds)
    if on_waiting:
        on_waiting(f"Waiting for a Chat message containing {code} ...")
    deadline = clock() + timeout
    while True:
        if cancel is not None and cancel.is_set():
            raise GoogleSetupError("Cancelled.")
        try:
            link = find_code(chat, code)
        except Exception as e:
            raise GoogleSetupError(_explain(e, SERVICES["chat"]))
        if link:
            if not link.sender.startswith("users/"):
                raise GoogleSetupError(f"Google gave an unexpected sender id: {link.sender!r}")
            return link
        if clock() >= deadline:
            raise GoogleSetupError(
                f"No message containing {code} arrived in {timeout:.0f}s. From YOUR Google "
                "account open chat.google.com, start a direct message with the agent's "
                "address, and send the code exactly as shown.")
        sleep(interval)


def complete_chat(creds, link: ChatLink, *, build: Callable | None = None) -> bool:
    """Save chat.json + the delivery state, then greet. Returns whether the greeting went
    out; a failed greeting does not undo the saved link."""
    chat = (build or _build)("chat", "v1", creds)
    self_id = ""
    sent = True
    try:
        reply = chat.spaces().messages().create(
            parent=link.space, body={"text": GREETING}).execute()
        self_id = (reply.get("sender") or {}).get("name", "")
        if not self_id and reply.get("name"):
            self_id = (chat.spaces().messages().get(name=reply["name"]).execute()
                       .get("sender") or {}).get("name", "")
    except Exception:
        sent = False
    save_chat(link, self_id)
    return sent


def save_chat(link: ChatLink, self_id: str = "") -> None:
    from config_loader import _write_json_atomic
    _write_json_atomic(paths.CHAT_CONFIG, {
        "space": link.space, "authorized_sender": link.sender, "poll_interval": POLL_INTERVAL})
    # Mark the handshake message as already seen, or the bridge's first poll would
    # hand "domicile-ab12cd" to the model as if it were a question.
    _write_json_atomic(paths.CHAT_STATE, {
        "space": link.space, "pending": None, "self_id": self_id or None,
        "last_create_time": link.create_time, "last_seen_names": [link.message_name]})


# --- command line: re-sign-in without the wizard ---------------------------------

def services_in_token() -> list:
    """Services the saved token covers (every scope of the service was granted)."""
    try:
        data = json.loads((GOOGLE_DIR / paths.GOOGLE_TOKEN.name).read_text())
    except (OSError, ValueError):
        return []
    got = set(data.get("scopes") or [])
    return [k for k, svc in SERVICES.items() if set(svc.scopes) <= got]


def client_from_token() -> dict | None:
    """The OAuth client recorded inside the saved token.json, in client_secret.json shape.
    None if there is no token or it lacks the client id/secret."""
    try:
        data = json.loads((GOOGLE_DIR / paths.GOOGLE_TOKEN.name).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("client_id") or not data.get("client_secret"):
        return None
    return {"installed": {
        "client_id": data["client_id"], "client_secret": data["client_secret"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": data.get("token_uri") or "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"]}}


def _saved_account() -> str:
    try:
        return json.loads((GOOGLE_DIR / paths.GOOGLE_TOKEN.name).read_text()).get("account") or ""
    except (OSError, ValueError):
        return ""


def _report(checks, say) -> bool:
    for c in checks:
        say(f"  {'OK  ' if c.ok else 'FAIL'} {c.label}: {c.detail}")
    return all(c.ok for c in checks)


def cmd_check(say=print, build=None) -> int:
    """Read-only: does the saved sign-in still work for every service it covers?"""
    services = services_in_token()
    if not services:
        say("No saved Google sign-in. Run: python3 wizard_google.py signin --services gmail,calendar")
        return 1
    try:
        creds = load_token()
    except Exception as e:
        say(f"Couldn't read the saved sign-in: {e}")
        return 1
    return 0 if _report(verify(creds, services, build=build), say) else 1


def cmd_signin(services=None, client_path="", *, ask=input, say=print,
               flow_factory=None, build=None) -> int:
    """Sign in again. Keeps the previous service list and account unless told otherwise;
    nothing is replaced until Google has accepted the new sign-in."""
    try:
        services = list(services or services_in_token())
        if not services:
            raise GoogleSetupError("No earlier sign-in to copy the service list from. Say which "
                                   "services: --services gmail,calendar,blogger,chat")
        saved = GOOGLE_DIR / paths.GOOGLE_CLIENT_SECRET.name
        client = None
        if not client_path and not saved.is_file():
            client = client_from_token()      # token.json carries its own client id + secret
        if client is None:
            client_path = client_path or (str(saved) if saved.is_file() else
                                          ask("Path to the client_secret JSON from Cloud Console: "))
            client = load_client_secret(client_path)
        signin = SignIn(client, services, flow_factory=flow_factory)
        say("Open this link in any browser and sign in as the AGENT's account.\n"
            "Approve every box. The last page will fail to load: that is expected.\n"
            "Copy the whole address from the address bar.\n")
        say(signin.start() + "\n")
        creds = signin.finish(ask("Paste the address here: "))
        old_account = _saved_account()
        save_token(creds, client, old_account)
        checks = verify(creds, services, build=build)
        account = account_of(checks) or old_account
        if account and account != old_account:
            save_token(creds, client, account)
    except GoogleSetupError as e:
        say(f"\n{e}")
        return 1
    except (EOFError, KeyboardInterrupt):
        say("\nCancelled. Nothing was changed.")
        return 1
    except OSError as e:
        say(f"\nCouldn't save the sign-in: {e}")
        return 1
    say("\nSigned in and saved. Checking each service:")
    if not _report(checks, say):
        say("Some services failed (see above). Fix them, then: python3 wizard_google.py check")
        return 1
    say("All good. No restart needed: the next Google call picks up the new sign-in.")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="wizard_google.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test the saved sign-in (changes nothing)")
    si = sub.add_parser("signin", help="sign in again")
    si.add_argument("--services", help="comma list, default: whatever the old sign-in covered "
                    f"({', '.join(SERVICES)})")
    si.add_argument("--client", default="", help="client_secret JSON (default: the client inside token.json)")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return cmd_check()
    wanted = [x.strip() for x in (args.services or "").split(",") if x.strip()]
    return cmd_signin(wanted, args.client)


if __name__ == "__main__":
    raise SystemExit(main())
