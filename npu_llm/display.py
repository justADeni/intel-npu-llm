"""Live in-terminal usage dashboard (``--display``).

Renders NPU hardware utilization (from :mod:`npu_llm.monitor`) alongside the
inference metrics the engine records (from :class:`npu_llm.engine.GenStats`),
using a Rich ``Live`` panel that refreshes a few times a second.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .console import console
from .hardware import get_info

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values, width: int = 40) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    out = []
    for v in vals:
        idx = int(max(0.0, min(100.0, v)) / 100.0 * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def _bar(pct: float, width: int = 30) -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100.0 * width)
    color = "green" if pct < 50 else "yellow" if pct < 85 else "red"
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="muted")
    t.append(f" {pct:5.1f}%", style=color)
    return t


class Dashboard:
    def __init__(self, engine, monitor, subtitle: str = ""):
        self.engine = engine
        self.monitor = monitor
        self.subtitle = subtitle
        self.history: deque[float] = deque(maxlen=40)
        self.peak = 0.0
        self.hw_seen = False  # have we ever read a real hardware utilization > 1%?
        self._stop = threading.Event()
        self.info = get_info()

    # tok/s that maps to a "full" bar when inferring load from throughput
    _REF_TPS = 40.0

    def _render(self) -> Panel:
        s = self.engine.stats
        util = self.history[-1] if self.history else 0.0

        head = Table.grid(expand=True)
        head.add_column(justify="left")
        head.add_column(justify="right")
        tops = f"{self.info.tops:.1f} TOPS" if self.info.tops else "? TOPS"
        mem = (
            f"{self.info.total_mem_bytes / (1024 ** 3):.1f} GiB"
            if self.info.total_mem_bytes
            else "? mem"
        )
        head.add_row(
            Text(self.info.name, style="info"),
            Text(f"{tops} · {mem}", style="muted"),
        )

        util_tbl = Table.grid(padding=(0, 1))
        util_tbl.add_column(justify="right", style="muted", no_wrap=True)
        util_tbl.add_column()
        state = "[good]● generating[/good]" if s.generating else "[muted]○ idle[/muted]"
        load_label = "NPU load" if self.hw_seen else "NPU load*"
        util_tbl.add_row(load_label, _bar(util))
        util_tbl.add_row("history", Text(_sparkline(self.history), style="accent"))
        util_tbl.add_row("peak", Text(f"{self.peak:5.1f}%", style="muted"))
        util_tbl.add_row("state", Text.from_markup(state))
        if not self.hw_seen:
            util_tbl.add_row(
                "", Text("* inferred from token throughput (no NPU HW counter here)", style="muted")
            )

        stats_tbl = Table.grid(padding=(0, 2))
        stats_tbl.add_column(justify="right", style="muted", no_wrap=True)
        stats_tbl.add_column(style="bold")
        live = s.live_tok_per_s if s.generating else s.last_tok_per_s
        stats_tbl.add_row("model", f"{s.model}  (ctx {s.ctx}, {s.device})")
        stats_tbl.add_row("tok/s", f"{live:.1f}")
        stats_tbl.add_row("last TTFT", f"{s.last_ttft_ms:.0f} ms")
        stats_tbl.add_row("last gen", f"{s.last_output_tokens} tok  (in {s.last_input_tokens})")
        stats_tbl.add_row("totals", f"{s.total_generations} gens · {s.total_output_tokens} tok")
        stats_tbl.add_row("load time", f"{s.load_seconds:.1f} s")

        body = Group(head, Text(), util_tbl, Text(), stats_tbl)
        title = "npu-llm" + (f" · {self.subtitle}" if self.subtitle else "")
        return Panel(body, title=title, border_style="info", padding=(1, 2))

    def _loop(self, live: Live, refresh_hz: float):
        interval = 1.0 / refresh_hz
        while not self._stop.is_set():
            s = self.engine.stats
            hw = self.monitor.sample(generating=s.generating) or 0.0
            if hw > 1.0:
                self.hw_seen = True
            # Where no hardware counter is available (e.g. OpenVINO NPU on
            # Windows), infer activity from our own measured throughput.
            inferred = 0.0
            if s.generating:
                tps = s.live_tok_per_s or 0.0
                inferred = max(20.0, min(100.0, tps / self._REF_TPS * 100.0))
            util = max(hw, inferred)
            self.history.append(util)
            self.peak = max(self.peak, util)
            live.update(self._render())
            time.sleep(interval)

    def run(self, refresh_hz: float = 4.0):
        """Blocking full-screen dashboard until Ctrl+C."""
        with Live(self._render(), console=console, refresh_per_second=refresh_hz, screen=False) as live:
            try:
                self._loop(live, refresh_hz)
            except KeyboardInterrupt:
                pass

    def start_background(self, refresh_hz: float = 4.0) -> threading.Thread:
        """Run the dashboard loop in a daemon thread (returns it)."""
        live = Live(self._render(), console=console, refresh_per_second=refresh_hz, screen=False)
        live.start()

        def _run():
            try:
                self._loop(live, refresh_hz)
            finally:
                live.stop()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()
