from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Markdown, Static


class UserMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__(content, classes="message user")
        self.border_title = " User "


class AssistantMessage(Markdown):
    def __init__(self, content: str) -> None:
        super().__init__(content, classes="message assistant")
        self.border_title = " Assistant "


class SystemMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__(content, classes="message system")


class ActivityMessage(Static):
    def __init__(self, renderable: Any) -> None:
        super().__init__(renderable, classes="message activity")

    def set_renderable(self, renderable: Any) -> None:
        self.update(renderable)


class ToolCallMessage(Static):
    def __init__(self, name: str, detail: str = "", renderable: Any | None = None) -> None:
        super().__init__(
            renderable if renderable is not None else _tool_text("Tool call", name, detail),
            classes="message tool",
        )

    def set_running(self, name: str, detail: str = "", renderable: Any | None = None) -> None:
        self.update(renderable if renderable is not None else _tool_text("Running", name, detail))

    def set_result(
        self,
        action: str,
        name: str,
        detail: str = "",
        renderable: Any | None = None,
    ) -> None:
        self.update(renderable if renderable is not None else _tool_text(action, name, detail))


class ToolResultMessage(Static):
    def __init__(self, name: str, detail: str = "", renderable: Any | None = None) -> None:
        super().__init__(
            renderable
            if renderable is not None
            else _tool_text(_tool_result_action(name), name, detail),
            classes="message tool",
        )


def tool_result_action(name: str) -> str:
    return _tool_result_action(name)


def _tool_result_action(name: str) -> str:
    normalized = name.lower()
    if any(token in normalized for token in ("vision", "visual", "image", "artifact")):
        return "Vision result summary"
    return "Tool result"


def _tool_text(action: str, name: str, detail: str = "") -> Text:
    text = Text("• ", style="#6f8796")
    text.append(action, style="#c8b47e")
    if name:
        text.append(f" {name}", style="#d7dde2")
    if detail:
        text.append("\n  └ ", style="#6f8796")
        text.append(detail, style="#8f9ba4")
    return text
