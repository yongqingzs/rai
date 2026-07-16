from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Markdown, Static

LONG_MESSAGE_CHARS = 4_000
LONG_MESSAGE_LINES = 48
MESSAGE_PREVIEW_CHARS = 1_600
MESSAGE_PREVIEW_LINES = 18
TOOL_DETAIL_CHARS = 280
TOOL_DETAIL_LINES = 4


def _now_label() -> str:
    return datetime.now().astimezone().strftime("%H:%M:%S")


def _preview_text(text: str, *, chars: int, lines: int) -> tuple[str, bool]:
    source_lines = text.splitlines()
    preview = "\n".join(source_lines[:lines])
    truncated = len(source_lines) > lines
    if len(preview) > chars:
        preview = preview[: chars - 1].rstrip()
        truncated = True
    elif len(text) > len(preview):
        truncated = True
    return (f"{preview}\n…" if truncated else preview), truncated


class _ExpandableMessage(Vertical):
    BINDINGS = [Binding("enter", "toggle_details", "Details", show=False)]

    role = "Message"
    markdown = False

    def __init__(self, content: str, *, timestamp: str | None = None) -> None:
        super().__init__(classes="message")
        self._content = content
        self._timestamp = timestamp or _now_label()
        self._timestamps_visible = False
        self._expanded = False
        self._preview, self._is_long = _preview_text(
            content, chars=MESSAGE_PREVIEW_CHARS, lines=MESSAGE_PREVIEW_LINES
        )
        self.can_focus = self._is_long

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), classes="message-header")
        body = self._preview if self._is_long else self._content
        yield self._body_widget(body)
        yield Static(
            "› Show full message" if self._is_long else "",
            classes="message-hint" if self._is_long else "message-hint hidden",
        )

    def _body_widget(self, content: str) -> Static | Markdown:
        if self.markdown:
            return Markdown(content, classes="message-body")
        return Static(content, classes="message-body")

    def _header_text(self) -> Text:
        text = Text(self.role)
        if self._timestamps_visible:
            text.append(f"  {self._timestamp}", style="dim")
        return text

    def set_timestamp_visible(self, visible: bool) -> None:
        self._timestamps_visible = visible
        if self.children:
            self.query_one(".message-header", Static).update(self._header_text())

    def action_toggle_details(self) -> None:
        if not self._is_long:
            return
        self._expanded = not self._expanded
        body = self.query_one(".message-body", Markdown if self.markdown else Static)
        body.update(self._content if self._expanded else self._preview)
        self.query_one(".message-hint", Static).update(
            "⌄ Show less" if self._expanded else "› Show full message"
        )

    def on_click(self, event: Click) -> None:
        if self._is_long:
            self.action_toggle_details()
            event.stop()


class UserMessage(_ExpandableMessage):
    role = "User"

    def __init__(self, content: str, *, timestamp: str | None = None) -> None:
        super().__init__(content, timestamp=timestamp)
        self.add_class("user")


class AssistantMessage(_ExpandableMessage):
    role = "Assistant"
    markdown = True

    def __init__(self, content: str, *, timestamp: str | None = None) -> None:
        super().__init__(content, timestamp=timestamp)
        self.add_class("assistant")


class SystemMessage(_ExpandableMessage):
    role = "System"

    def __init__(self, content: str, *, timestamp: str | None = None) -> None:
        super().__init__(content, timestamp=timestamp)
        self.add_class("system")


class ActivityMessage(Static):
    def __init__(self, renderable: Any) -> None:
        super().__init__(renderable, classes="message activity")

    def set_renderable(self, renderable: Any) -> None:
        self.update(renderable)


@dataclass(frozen=True)
class ToolPresentation:
    action: str
    summary: str
    detail: str
    domain: str


class ToolCallMessage(Vertical):
    BINDINGS = [Binding("enter", "toggle_details", "Details", show=False)]

    def __init__(
        self,
        name: str,
        detail: str = "",
        renderable: Any | None = None,
        *,
        full_content: Any | None = None,
        action: str = "Tool call",
        domain: str = "tool",
        timestamp: str | None = None,
    ) -> None:
        super().__init__(classes=f"message tool {domain}")
        self._name = name
        self._action = action
        self._summary = detail
        self._full_content = (
            _stringify(full_content) if full_content is not None else ""
        )
        self._timestamp = timestamp or _now_label()
        self._timestamps_visible = False
        self._expanded = False
        self._renderable = renderable
        self._refresh_expandable()

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), classes="tool-header")
        yield Static(self._summary, classes="tool-summary")
        yield Static("", classes="tool-detail hidden")
        yield Static(self._hint_text(), classes=self._hint_classes())

    def _refresh_expandable(self) -> None:
        self._expandable = bool(
            self._full_content
            and (
                len(self._full_content) > TOOL_DETAIL_CHARS
                or self._full_content.count("\n") + 1 > TOOL_DETAIL_LINES
                or self._full_content.strip() != self._summary.strip()
            )
        )
        self.can_focus = self._expandable

    def _header_text(self) -> Text:
        text = Text("• ", style="dim")
        text.append(self._action, style="bold")
        if self._name:
            text.append(f"  {self._name}")
        if self._timestamps_visible:
            text.append(f"  {self._timestamp}", style="dim")
        return text

    def _hint_text(self) -> str:
        if not self._expandable:
            return ""
        return "⌄ Hide details" if self._expanded else "› Details"

    def _hint_classes(self) -> str:
        return "tool-hint" if self._expandable else "tool-hint hidden"

    def set_running(
        self, name: str, detail: str = "", renderable: Any | None = None
    ) -> None:
        self._set_content("Running", name, detail, None, renderable)

    def set_result(
        self,
        action: str,
        name: str,
        detail: str = "",
        renderable: Any | None = None,
        *,
        full_content: Any | None = None,
        domain: str | None = None,
    ) -> None:
        self._set_content(action, name, detail, full_content, renderable, domain)

    def _set_content(
        self,
        action: str,
        name: str,
        detail: str,
        full_content: Any | None,
        renderable: Any | None,
        domain: str | None = None,
    ) -> None:
        self._action = action
        self._name = name
        self._summary = detail
        self._renderable = renderable
        if full_content is not None:
            self._full_content = _stringify(full_content)
        if domain:
            for known in ("vision", "rag", "navigation", "ros", "tool"):
                self.remove_class(known)
            self.add_class(domain)
        self._refresh_expandable()
        if not self.children:
            return
        self.query_one(".tool-header", Static).update(self._header_text())
        self.query_one(".tool-summary", Static).update(self._summary)
        detail_widget = self.query_one(".tool-detail", Static)
        detail_widget.update(self._full_content if self._expanded else "")
        detail_widget.set_class(not self._expanded, "hidden")
        hint = self.query_one(".tool-hint", Static)
        hint.update(self._hint_text())
        hint.set_class(not self._expandable, "hidden")

    def set_timestamp_visible(self, visible: bool) -> None:
        self._timestamps_visible = visible
        if self.children:
            self.query_one(".tool-header", Static).update(self._header_text())

    def action_toggle_details(self) -> None:
        if not self._expandable:
            return
        self._expanded = not self._expanded
        detail = self.query_one(".tool-detail", Static)
        if self._expanded:
            detail.update(self._full_content)
            detail.remove_class("hidden")
        else:
            detail.update("")
            detail.add_class("hidden")
        self.query_one(".tool-hint", Static).update(self._hint_text())

    def on_click(self, event: Click) -> None:
        if self._expandable:
            self.action_toggle_details()
            event.stop()


class ToolResultMessage(ToolCallMessage):
    def __init__(
        self,
        name: str,
        detail: str = "",
        renderable: Any | None = None,
        *,
        full_content: Any | None = None,
        timestamp: str | None = None,
    ) -> None:
        presentation = present_tool_result(
            name, full_content if full_content is not None else detail
        )
        super().__init__(
            name,
            detail or presentation.summary,
            renderable,
            full_content=full_content,
            action=presentation.action,
            domain=presentation.domain,
            timestamp=timestamp,
        )


def present_tool_result(name: str, content: Any, limit: int = 200) -> ToolPresentation:
    detail = _stringify(content)
    domain = _tool_domain(name)
    compact = _compact_value(content)
    if domain == "vision":
        action = "Vision analysis"
        conclusion = _section_value(detail, ("结论", "conclusion", "summary"))
        summary = conclusion or compact
        summary = f"Analysis complete · {summary}" if summary else "Analysis complete"
    elif domain == "rag":
        action = "Knowledge retrieval"
        count = _result_count(content)
        prefix = (
            f"Retrieved {count} item{'s' if count != 1 else ''}"
            if count
            else "Retrieval complete"
        )
        summary = f"{prefix} · {compact}" if compact else prefix
    elif domain == "navigation":
        action = "Navigation"
        status = _status_value(content)
        summary = (
            f"{status} · {compact}"
            if status and status.lower() not in compact.lower()
            else compact
        )
        summary = summary or status or "Navigation result received"
    elif domain == "ros":
        action = "ROS operation"
        status = _status_value(content)
        summary = (
            f"{status} · {compact}"
            if status and status.lower() not in compact.lower()
            else compact
        )
        summary = summary or status or "ROS result received"
    else:
        action = "Tool result"
        summary = compact or "Result received"
    summary = " ".join(summary.split())
    if len(summary) > limit:
        summary = f"{summary[: limit - 1].rstrip()}…"
    return ToolPresentation(action, summary, detail, domain)


def tool_result_action(name: str) -> str:
    return present_tool_result(name, "").action


def _tool_domain(name: str) -> str:
    normalized = name.lower()
    if any(
        token in normalized
        for token in ("vision", "visual", "image", "artifact", "scene")
    ):
        return "vision"
    if any(
        token in normalized
        for token in (
            "rag",
            "retriev",
            "knowledge",
            "semantic",
            "document",
            "robot_docs",
        )
    ):
        return "rag"
    if any(
        token in normalized
        for token in (
            "navigate",
            "navigation",
            "waypoint",
            "move_base",
            "save_location",
        )
    ):
        return "navigation"
    if any(
        token in normalized
        for token in (
            "ros",
            "topic",
            "service",
            "publish",
            "subscribe",
            "gimbal",
            "speaker",
            "gas",
        )
    ):
        return "ros"
    return "tool"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _compact_value(value: Any) -> str:
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, dict):
        preferred = ("status", "result", "message", "summary", "detail", "content")
        parts = [f"{key}: {_one_line(value[key])}" for key in preferred if key in value]
        if parts:
            return " · ".join(parts[:3])
    if isinstance(value, list):
        return _one_line(value[0]) if value else ""
    return _one_line(value)


def _one_line(value: Any) -> str:
    text = _stringify(value)
    return " ".join(text.split())


def _status_value(value: Any) -> str:
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, dict):
        for key in ("status", "state", "result"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float, bool)):
                return str(candidate)
    text = _one_line(value).lower()
    for status in (
        "success",
        "stalled",
        "timeout",
        "aborted",
        "canceled",
        "cancelled",
        "failed",
        "error",
    ):
        if status in text:
            return status
    return ""


def _result_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("documents", "results", "chunks", "items"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return None


def _section_value(text: str, headings: tuple[str, ...]) -> str:
    lines = [line.strip(" #*-\t") for line in text.splitlines()]
    for index, line in enumerate(lines):
        lowered = line.lower().rstrip(":：")
        if any(lowered == heading.lower() for heading in headings):
            for candidate in lines[index + 1 :]:
                if candidate:
                    return candidate
    return ""
