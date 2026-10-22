"""Expression pane rendering and telemetry."""

from __future__ import annotations

import math
import re
import socket
import time
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

try:
    import psutil
except Exception:  # pragma: no cover - optional dep
    psutil = None

try:
    from textual_plotext import PlotextPlot
except Exception:  # pragma: no cover - optional dep
    PlotextPlot = None


from paths import EXPRESSION_FILE as BUFFER_FILE, TOKEN_LOG

_WELCOME = (
    "# expression\n\n"
    "*Nothing sent yet. This pane holds whatever I last drew or wrote.*\n"
)


def write_buffer(content: str) -> None:
    BUFFER_FILE.parent.mkdir(parents=True, exist_ok=True)
    BUFFER_FILE.write_text(content)


def read_buffer() -> str:
    try:
        return BUFFER_FILE.read_text() or _WELCOME
    except OSError:
        return _WELCOME


def append_usage(provider: str, inp: float, out: float,
                 model: str | None = None, session: str | None = None,
                 prompt_chars: int | None = None) -> None:
    """Append one usage row. `session` groups calls that share a context
    lineage (changes at compaction / new SDK session); `prompt_chars` is the
    non-system history size sent on that call, blank when unknown. Together
    they let base load be fit per provider: input ~ base + ratio * chars.
    Older rows have 4 or 5 columns; readers must go by width, not header."""
    try:
        inp_i, out_i = int(inp or 0), int(out or 0)
        if inp_i <= 0 and out_i <= 0:
            return
        slug = (model or "?").replace(",", " ").strip() or "?"
        sess = str(session or "").replace(",", " ").strip()
        chars = "" if prompt_chars is None else str(int(prompt_chars))
        TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
        header = not TOKEN_LOG.exists()
        with TOKEN_LOG.open("a") as fh:
            if header:
                fh.write("ts_iso,provider,model,input,output,session,"
                         "prompt_chars\n")
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},"
                     f"{provider or '?'},{slug},{inp_i},{out_i},"
                     f"{sess},{chars}\n")
    except (OSError, ValueError, TypeError):
        pass


# SVG


_BRAILLE_BASE = 0x2800
_DOT_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


class _Bitmap:
    def __init__(self, w: int, h: int):
        self.w = max(2, w)
        self.h = max(4, h)
        self.px = bytearray(self.w * self.h)  # 0/1

    def set(self, x: int, y: int) -> None:
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y * self.w + x] = 1

    def line(self, x0, y0, x1, y1) -> None:
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def polyline(self, pts, close=False) -> None:
        for i in range(len(pts) - 1):
            self.line(*pts[i], *pts[i + 1])
        if close and len(pts) > 2:
            self.line(*pts[-1], *pts[0])

    def ellipse(self, cx, cy, rx, ry) -> None:
        if rx <= 0 or ry <= 0:
            return
        steps = max(12, int(2 * math.pi * max(rx, ry)))
        prev = None
        for i in range(steps + 1):
            a = 2 * math.pi * i / steps
            p = (cx + rx * math.cos(a), cy + ry * math.sin(a))
            if prev is not None:
                self.line(*prev, *p)
            prev = p

    def to_text(self, style: str = "cyan") -> Text:
        out = Text()
        rows = self.h // 4
        cols = self.w // 2
        for cy in range(rows):
            line = []
            for cx in range(cols):
                bits = 0
                for dx in range(2):
                    for dy in range(4):
                        if self.px[(cy * 4 + dy) * self.w + (cx * 2 + dx)]:
                            bits |= _DOT_BITS[dx][dy]
                line.append(chr(_BRAILLE_BASE + bits))
            out.append("".join(line) + "\n", style=style)
        return out


_NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")


def _nums(s: str):
    return [float(x) for x in _NUM.findall(s or "")]


def _attr(tag: str, name: str) -> str:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', tag)
    return m.group(1) if m else ""


def _viewbox(svg: str):
    vb = _nums(_attr(svg, "viewBox"))
    if len(vb) == 4:
        return vb[0], vb[1], vb[2], vb[3]
    w = _nums(_attr(svg, "width")) or [100.0]
    h = _nums(_attr(svg, "height")) or [100.0]
    return 0.0, 0.0, w[0], h[0]


def _path_points(d: str):
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:e-?\d+)?", d or "")
    polys, cur = [], []
    x = y = 0.0
    i = 0
    cmd = None
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz" and cur:
                cur.append(cur[0])
                polys.append(cur)
                cur = []
            continue
        rel = cmd.islower()
        c = cmd.upper()
        try:
            if c == "M":
                nx, ny = float(tokens[i]), float(tokens[i + 1]); i += 2
                x, y = (x + nx, y + ny) if rel else (nx, ny)
                if cur:
                    polys.append(cur)
                cur = [(x, y)]
                cmd = "l" if rel else "L"
            elif c == "L":
                nx, ny = float(tokens[i]), float(tokens[i + 1]); i += 2
                x, y = (x + nx, y + ny) if rel else (nx, ny)
                cur.append((x, y))
            elif c == "H":
                nx = float(tokens[i]); i += 1
                x = x + nx if rel else nx
                cur.append((x, y))
            elif c == "V":
                ny = float(tokens[i]); i += 1
                y = y + ny if rel else ny
                cur.append((x, y))
            elif c in ("C", "S", "Q", "T", "A"):
                take = {"C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}[c]
                vals = tokens[i:i + take]; i += take
                if len(vals) >= 2:
                    ex, ey = float(vals[-2]), float(vals[-1])
                    x, y = (x + ex, y + ey) if rel else (ex, ey)
                    cur.append((x, y))
            else:
                i += 1
        except (IndexError, ValueError):
            break
    if cur:
        polys.append(cur)
    return polys


def render_svg(svg: str, cols: int, rows: int) -> Text:
    cols = max(8, cols)
    rows = max(4, rows)
    vx, vy, vw, vh = _viewbox(svg)
    if vw <= 0 or vh <= 0:
        vw, vh = 100.0, 100.0
    bmp = _Bitmap(cols * 2, rows * 4)
    sx = (bmp.w - 1) / vw
    sy = (bmp.h - 1) / vh
    sc = min(sx, sy)  # uniform, preserve aspect

    def tx(px):
        return (px - vx) * sc

    def ty(py):
        return (py - vy) * sc

    labels = []  # (col, row, text, style)

    for tag in re.findall(r"<[^>]+>", svg):
        name = re.match(r"<\s*/?\s*([a-zA-Z]+)", tag)
        if not name:
            continue
        el = name.group(1).lower()
        stroke = _attr(tag, "stroke") or ""
        try:
            if el == "line":
                a = _nums(_attr(tag, "x1")) + _nums(_attr(tag, "y1"))
                b = _nums(_attr(tag, "x2")) + _nums(_attr(tag, "y2"))
                if len(a) == 2 and len(b) == 2:
                    bmp.line(tx(a[0]), ty(a[1]), tx(b[0]), ty(b[1]))
            elif el in ("polyline", "polygon"):
                raw = _nums(_attr(tag, "points"))
                pts = [(tx(raw[k]), ty(raw[k + 1])) for k in range(0, len(raw) - 1, 2)]
                bmp.polyline(pts, close=(el == "polygon"))
            elif el == "rect":
                x = (_nums(_attr(tag, "x")) or [0])[0]
                y = (_nums(_attr(tag, "y")) or [0])[0]
                w = (_nums(_attr(tag, "width")) or [0])[0]
                h = (_nums(_attr(tag, "height")) or [0])[0]
                bmp.polyline([(tx(x), ty(y)), (tx(x + w), ty(y)),
                              (tx(x + w), ty(y + h)), (tx(x), ty(y + h))], close=True)
            elif el == "circle":
                cx = (_nums(_attr(tag, "cx")) or [0])[0]
                cy = (_nums(_attr(tag, "cy")) or [0])[0]
                r = (_nums(_attr(tag, "r")) or [0])[0]
                bmp.ellipse(tx(cx), ty(cy), r * sc, r * sc)
            elif el == "ellipse":
                cx = (_nums(_attr(tag, "cx")) or [0])[0]
                cy = (_nums(_attr(tag, "cy")) or [0])[0]
                rx = (_nums(_attr(tag, "rx")) or [0])[0]
                ry = (_nums(_attr(tag, "ry")) or [0])[0]
                bmp.ellipse(tx(cx), ty(cy), rx * sc, ry * sc)
            elif el == "path":
                for poly in _path_points(_attr(tag, "d")):
                    bmp.polyline([(tx(p[0]), ty(p[1])) for p in poly])
        except Exception:
            continue

    for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg, re.S):
        attrs, body = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        xs = _nums(_attr("<t " + attrs + ">", "x")) or [0]
        ys = _nums(_attr("<t " + attrs + ">", "y")) or [0]
        col = int(tx(xs[0]) / 2)
        row = int(ty(ys[0]) / 4)
        if body:
            labels.append((col, row, body))

    art = bmp.to_text("cyan")
    if labels:
        lines = art.split("\n")
        grid = [list(str(l)) for l in lines]
        for col, row, body in labels:
            if 0 <= row < len(grid):
                for k, ch in enumerate(body):
                    if 0 <= col + k < len(grid[row]):
                        grid[row][col + k] = ch
        art = Text("\n".join("".join(r) for r in grid), style="cyan")
    return art


# system

def _bar(label: str, pct: float, width: int = 14) -> str:
    pct = max(0.0, min(100.0, pct))
    fill = int(round(pct / 100 * width))
    colour = "red" if pct >= 90 else "yellow" if pct >= 70 else "green"
    return (f"{label:<4}[{colour}]{'█' * fill}{'░' * (width - fill)}[/{colour}] "
            f"{pct:>3.0f}%")


def _health_hosts() -> dict:
    """{provider: hostname} for every provider in config.json, in provider_order."""
    from urllib.parse import urlparse
    try:
        import config_loader as _cl
        cfg = _cl.load_config()
    except Exception:
        return {"anthropic": "api.anthropic.com"}

    hosts = {}
    for name in _cl.provider_order(cfg):
        blk = (cfg.get("providers") or {}).get(name) or {}
        base = blk.get("base_url")
        if base:
            netloc = urlparse(base).netloc
            if netloc:
                hosts[name] = netloc.split("@")[-1].split(":")[0]
        elif _cl.is_cli_provider(name, cfg):
            hosts[name] = "api.anthropic.com"
    return hosts or {"anthropic": "api.anthropic.com"}


class _Health:

    def __init__(self):
        self.hosts = _health_hosts()
        self.state = {k: ("dim", "?") for k in self.hosts}

    @staticmethod
    def _one(host: str):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, 443), timeout=2.5):
                pass
            dt_ms = (time.monotonic() - t0) * 1000
            return ("yellow" if dt_ms > 1200 else "green", f"{dt_ms:.0f}ms")
        except Exception:
            return ("red", "down")

    def probe(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self.hosts = _health_hosts()          # pick up live config edits
        if not self.hosts:
            self.state = {}
            return
        with ThreadPoolExecutor(max_workers=len(self.hosts)) as ex:
            results = dict(zip(self.hosts,
                               ex.map(self._one, self.hosts.values())))
        self.state = {k: results.get(k, self.state.get(k, ("dim", "?")))
                      for k in self.hosts}

    def render(self) -> str:
        return "  ".join(
            f"[{c}]●[/{c}] {k}" for k, (c, _lat) in self.state.items())

    def snapshot(self) -> dict:
        """{provider: {color, latency, host}} -- for the webui telemetry feed."""
        return {k: {"color": c, "latency": lat, "host": self.hosts.get(k, "")}
                for k, (c, lat) in self.state.items()}


def _cpu_temp():
    if psutil is not None:
        try:
            temps = psutil.sensors_temperatures()
            for key in ("coretemp", "k10temp", "zenpower", "acpitz",
                        "cpu_thermal", "cpu-thermal"):
                if temps.get(key):
                    return max(t.current for t in temps[key] if t.current)
            for arr in temps.values():
                if arr and arr[0].current:
                    return arr[0].current
        except Exception:
            pass
    try:
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            raw = (zone / "temp").read_text().strip()
            if raw:
                return int(raw) / 1000.0
    except Exception:
        pass
    return None


_TOKEN_CACHE = (None, 0.0, None)


def _token_series():
    """24h totals; reuse unchanged logs until their oldest sample expires."""
    import csv
    from collections import defaultdict
    from copy import deepcopy
    from file_cache import signature

    global _TOKEN_CACHE
    now = time.time()
    try:
        stamp = signature(TOKEN_LOG)
    except OSError:
        return None
    old_stamp, expires, data = _TOKEN_CACHE
    if stamp == old_stamp and now <= expires:
        return deepcopy(data)

    buckets = defaultdict(lambda: defaultdict(float))
    provs = set()
    expires = now + 60  # Also notice same-size rewrites on coarse-timestamp filesystems.
    try:
        with TOKEN_LOG.open(newline="") as handle:
            for row in csv.reader(handle):
                if not row or row[0].lower().startswith("ts"):
                    continue
                try:
                    ts = time.mktime(time.strptime(row[0][:19], "%Y-%m-%dT%H:%M:%S"))
                except ValueError:
                    continue
                if now - ts > 24 * 3600:
                    continue
                expires = min(expires, ts + 24 * 3600)
                prov = row[1] if len(row) > 1 else "?"
                # row[2:5] is (in,out) on 4-col rows, (model,in,out) on
                # 5- and 7-col rows; later columns are session/prompt_chars,
                # which are not tokens.
                total = sum(float(x) for x in row[2:5] if x.replace(".", "", 1).isdigit())
                buckets[int(ts // 3600)][prov] += total
                provs.add(prov)
    except OSError:
        return None
    hours = sorted(buckets)
    data = None if not hours else (
        [time.strftime("%H", time.localtime(h * 3600)) for h in hours],
        {p: [buckets[h].get(p, 0) for h in hours] for p in sorted(provs)},
    )
    _TOKEN_CACHE = (stamp, expires, data)
    return deepcopy(data)


# telemetry feed for the webui

_TELEM_HEALTH = _Health()
_TELEM_HEALTH_TS = 0.0
_TELEM_HEALTH_TTL = 25.0


def telemetry_snapshot(probe: bool = True) -> dict:
    """One JSON-able dict: provider health, host vitals, 24h token series."""
    global _TELEM_HEALTH_TS
    if probe and (time.time() - _TELEM_HEALTH_TS) > _TELEM_HEALTH_TTL:
        try:
            _TELEM_HEALTH.probe()
            _TELEM_HEALTH_TS = time.time()
        except Exception:
            pass

    host = {}
    if psutil is not None:
        try:
            host = {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage(str(Path.home())).percent,
            }
        except Exception:
            host = {}
    temp = _cpu_temp()
    if temp is not None:
        host["temp_c"] = round(temp, 1)

    tokens = None
    series = _token_series()
    if series is not None:
        labels, by_prov = series
        tokens = {"labels": labels, "series": by_prov}

    return {
        "ts": time.time(),
        "health": _TELEM_HEALTH.snapshot(),
        "host": host,
        "tokens": tokens,
    }


class ExpressionPanel(Vertical):
    """The whole right-side panel. Add to the chat screen next to #messages."""

    DEFAULT_CSS = """
    ExpressionPanel { width: 40%; border-left: solid $primary; }
    ExpressionPanel.collapsed { width: 24; }
    ExpressionPanel.collapsed #expr-graph,
    ExpressionPanel.collapsed #expr-buffer { display: none; }
    ExpressionPanel #expr-graph { height: 40%; min-height: 8; }
    ExpressionPanel #expr-sys { height: auto; padding: 0 1; }
    ExpressionPanel #expr-buffer { padding: 0 1; }
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._health = _Health()
        self._buf_mtime = -1.0
        self._tick = 0
        self._proc = psutil.Process() if psutil is not None else None

    def compose(self):
        if PlotextPlot is not None:
            yield PlotextPlot(id="expr-graph")
        yield Static(id="expr-sys")
        yield Static(id="expr-buffer")

    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh)
        self.run_worker(self._probe_async, thread=True, group="health")
        self._refresh(force=True)

    def _refresh(self, force: bool = False) -> None:
        self._tick += 1
        if self._tick % 15 == 0:
            self.run_worker(self._probe_async, thread=True, exclusive=True,
                            group="health")
        self._refresh_sys()
        self._refresh_buffer(force=force)
        if force or self._tick % 5 == 0:
            self._refresh_graph()

    def _probe_async(self) -> None:
        self._health.probe()

    def _refresh_sys(self) -> None:
        try:
            sys_w = self.query_one("#expr-sys", Static)
        except Exception:
            return
        lines = []
        if psutil is not None:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage(str(Path.home())).percent
            bars = [_bar("cpu", cpu), _bar("ram", ram), _bar("dsk", disk)]
            badges = ["", "", ""]
            if not self.has_class("collapsed"):
                temp = _cpu_temp()
                if temp is not None:
                    badges[0] = f"[dim]temp[/dim] {temp:>3.0f}°C"
                if self._proc is not None:
                    try:
                        rss = self._proc.memory_info().rss / (1024 * 1024)
                        pcpu = self._proc.cpu_percent()
                        badges[1] = f"[dim]Domicile[/dim] {rss:>4.0f}MB"
                        badges[2] = f"[dim] cpu[/dim] {pcpu:>3.0f}%"
                    except Exception:
                        pass
            lines += [f"{b}   {x}" if x else b for b, x in zip(bars, badges)]
        lines.append(self._health.render())
        sys_w.update("\n".join(lines))

    def _refresh_buffer(self, force: bool = False) -> None:
        try:
            buf = self.query_one("#expr-buffer", Static)
        except Exception:
            return
        try:
            mtime = BUFFER_FILE.stat().st_mtime
        except OSError:
            mtime = 0.0
        if not force and mtime == self._buf_mtime:
            return
        self._buf_mtime = mtime
        content = read_buffer()
        stripped = content.lstrip()
        if stripped.startswith("<svg") or stripped.startswith("<?xml"):
            w = buf.size.width or 36
            h = buf.size.height or 16
            buf.update(render_svg(content, w, h))
        else:
            buf.update(RichMarkdown(content))

    def _refresh_graph(self) -> None:
        if PlotextPlot is None:
            return
        try:
            widget = self.query_one("#expr-graph", PlotextPlot)
        except Exception:
            return
        plt = widget.plt
        plt.clear_data()
        plt.clear_figure()
        data = _token_series()
        widget.border_title = "tokens / 24h"
        if data is None:
            plt.text("no token log yet", 0.5, 0.5)
            widget.refresh()
            return
        labels, series = data
        xs = list(range(len(labels)))
        for prov, ys in series.items():
            plt.plot(xs, ys, label=prov, marker="braille")
        plt.xticks(xs, labels)
        widget.refresh()
