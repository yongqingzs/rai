from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ClipboardResult:
    success: bool
    method: str
    error: str | None = None


def copy_text_to_clipboard(
    app: Any,
    text: str,
    *,
    use_pyperclip: bool = True,
    use_osc52: bool = True,
) -> ClipboardResult:
    """Copy text using system, Textual, and terminal clipboard backends."""
    methods: list[tuple[str, Callable[[str], object]]] = []
    if use_pyperclip:
        try:
            import pyperclip
        except ImportError:
            pass
        else:
            methods.append(("pyperclip", pyperclip.copy))

    methods.append(("textual", app.copy_to_clipboard))
    if use_osc52:
        methods.append(("osc52", _copy_osc52))

    last_error: str | None = None
    for name, copy_fn in methods:
        try:
            copy_fn(text)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            last_error = str(exc) or type(exc).__name__
            continue
        return ClipboardResult(success=True, method=name)

    return ClipboardResult(success=False, method="", error=last_error)


def _copy_osc52(text: str) -> None:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sequence = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        sequence = f"\033Ptmux;\033{sequence}\033\\"
    with Path("/dev/tty").open("w", encoding="utf-8") as tty:
        tty.write(sequence)
        tty.flush()
