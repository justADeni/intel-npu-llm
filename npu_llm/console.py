"""Shared Rich console + small helpers (replaces the old colorama usage)."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# Ensure UTF-8 output so box-drawing / sparkline glyphs render on Windows
# consoles whose code page is not UTF-8 (e.g. cp1250), which otherwise raises
# UnicodeEncodeError / prints mojibake.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

_theme = Theme(
    {
        "info": "cyan",
        "good": "green",
        "warn": "yellow",
        "err": "bold red",
        "muted": "dim",
        "accent": "magenta",
    }
)

console = Console(theme=_theme, highlight=False)


def info(msg: str) -> None:
    console.print(msg, style="info")


def good(msg: str) -> None:
    console.print(msg, style="good")


def warn(msg: str) -> None:
    console.print(msg, style="warn")


def error(msg: str) -> None:
    console.print(msg, style="err")


def muted(msg: str) -> None:
    console.print(msg, style="muted")
