from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.theme import Theme
from textual.widgets import Header, Markdown, Static, TextArea

from rai.frontend.cli import (
    CliAgentEvent,
    CliCommandResult,
    CliTurn,
    MemoryCliSession,
    _format_json,
    parse_cli_input,
)
from rai.memory.long_term import format_long_term_item
from rai.memory.session import SessionSummary

RAI_AGENT_THEME = Theme(
    name="rai_agent_dark",
    primary="#8fb3c8",
    secondary="#6f8796",
    accent="#7bb7a7",
    foreground="#d7dde2",
    background="#0d1114",
    surface="#141a1f",
    panel="#192127",
    boost="#202a31",
    success="#86c39a",
    warning="#d7b46a",
    error="#d06f6f",
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "input-cursor-background": "#c8d5dc",
        "input-cursor-foreground": "#0d1114",
        "input-selection-background": "#355263 70%",
        "scrollbar": "#29343c",
        "scrollbar-hover": "#35434d",
        "scrollbar-active": "#50616d",
        "scrollbar-background": "#101519",
    },
)


class ChatTextArea(TextArea):
    """Multiline chat input: Enter submits, Shift/Alt+Enter inserts a newline."""

    class Submitted(Message):
        def __init__(self, text: str, text_area: "ChatTextArea") -> None:
            super().__init__()
            self.text = text
            self.text_area = text_area

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+c":
            event.stop()
            event.prevent_default()
            if self.text:
                self.load_text("")
            else:
                self.app.exit()
            return
        if getattr(self.app, "_session_picker", None):
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                self.app._resume_selected_session()
                return
            if event.key == "up":
                event.stop()
                event.prevent_default()
                self.app._move_session_picker(-1)
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                self.app._move_session_picker(1)
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.app._clear_command_panel()
                return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text, self))
            return
        if event.key in {"shift+enter", "alt+enter"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class MemoryTuiApp(App):
    """Codex-style terminal UI for RAI memory agents."""

    CSS = """
    Screen {
        layout: vertical;
        background: #0d1114;
        color: #d7dde2;
    }

    #conversation {
        height: 1fr;
        border: tall #27323a;
        padding: 1 1;
        background: #0d1114;
    }

    #command_panel {
        height: auto;
        max-height: 12;
        padding: 0 1;
        border: round #44525c;
        background: #182027;
        color: #d7dde2;
    }

    #command_panel.hidden {
        display: none;
    }

    ChatTextArea {
        height: auto;
        min-height: 1;
        max-height: 6;
        border: tall #34424b;
        background: #11171b;
        color: #dde4e8;
    }

    ChatTextArea:focus {
        border: tall #6f9eb4;
        background: #131b20;
    }

    ChatTextArea .text-area--cursor {
        background: #c8d5dc;
        color: #0d1114;
        text-style: none;
    }

    ChatTextArea .text-area--selection {
        background: #355263 70%;
    }

    ChatTextArea .text-area--cursor-line {
        background: #182128;
    }

    ChatTextArea .text-area--placeholder {
        color: #70808a;
    }

    .message {
        width: 100%;
        margin: 1 0 1 0;
        padding: 1 2;
        border: round #2f3a42;
        background: #182027;
        color: #d7dde2;
    }

    .message.user {
        color: #e6edf1;
        border-left: solid #6f8796;
        border-top: solid #2f3a42;
        border-right: solid #2f3a42;
        border-bottom: solid #2f3a42;
        border-title-color: #9ed0ff;
        background: #171f26;
    }

    .message.assistant {
        color: #d7dde2;
        border-left: solid #7bb7a7;
        border-top: solid #334048;
        border-right: solid #334048;
        border-bottom: solid #334048;
        border-title-color: #8fe3c2;
        background: #1b242b;
    }

    .message.system {
        color: #8f9ba4;
        border-left: solid #34424b;
        border-top: solid #283138;
        border-right: solid #283138;
        border-bottom: solid #283138;
        background: #141a1f;
    }

    .message.tool {
        color: #c8b47e;
        border-left: solid #6b6041;
        border-top: solid #3b382a;
        border-right: solid #3b382a;
        border-bottom: solid #3b382a;
        background: #1c211d;
    }

    .message.activity {
        color: #7f8c95;
        margin: 0 0 1 0;
        padding: 0 1;
        border: none;
        background: #0d1114;
    }

    #agent_status {
        height: 1;
        padding: 0 1;
        background: #0d1114;
        color: #7f8c95;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_conversation", "Clear", show=False),
        Binding("ctrl+shift+c", "copy_transcript", "Copy transcript", show=False),
        Binding("escape", "dismiss_panel", "Dismiss", show=False),
    ]

    def __init__(
        self,
        session: MemoryCliSession,
        *,
        log_path: str | Path | None = None,
    ):
        super().__init__()
        self.session = session
        self._status = "idle"
        self._session_picker: list[SessionSummary] = []
        self._session_picker_index = 0
        self._last_assistant_text = ""
        self._last_activity_status = ""
        self._turn_started_at: float | None = None
        self._turn_timer: Any | None = None
        self._working_widget: Static | None = None
        self._working_transcript_index: int | None = None
        self._tool_activity_widgets: dict[str, tuple[Static, str]] = {}
        self._realtime_tool_result_ids: set[str] = set()
        self._transcript: list[str] = []
        self.log_path = Path(log_path).expanduser() if log_path is not None else None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="conversation")
        yield Static("", id="command_panel", classes="hidden")
        yield ChatTextArea(id="input")
        yield Static(self._status_text(), id="agent_status")

    def on_mount(self) -> None:
        self.register_theme(RAI_AGENT_THEME)
        self.theme = RAI_AGENT_THEME.name
        self._write_conversation_system(
            "RAI TUI started. Use /help for commands, /resume to choose a session."
        )
        self._refresh_status("idle")
        self.query_one("#input", ChatTextArea).focus()

    def on_key(self, event) -> None:
        if not self._session_picker:
            return
        if event.key == "up":
            self._move_session_picker(-1)
            event.stop()
        elif event.key == "down":
            self._move_session_picker(1)
            event.stop()
        elif event.key == "enter":
            self._resume_selected_session()
            event.stop()
        elif event.key == "escape":
            self._clear_command_panel()
            event.stop()

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        value = event.text.strip()
        event.text_area.load_text("")
        if not value:
            return
        self._clear_command_panel()
        if value in {"/resume", "/session"}:
            self._show_session_picker()
            return
        if value == "/copy-last":
            self._copy_last_assistant_message()
            return
        if value == "/log":
            self._show_log_status()
            return
        if value == "/copy-transcript":
            self.action_copy_transcript()
            return
        command = parse_cli_input(value)
        if command.should_exit:
            self.exit()
            return
        if command.handled:
            self._handle_command(command)
            return
        if command.turn is not None:
            self._submit_turn(command.turn)

    def action_clear_conversation(self) -> None:
        self._conversation().remove_children()
        self._transcript.clear()

    def action_dismiss_panel(self) -> None:
        self._clear_command_panel()

    def action_copy_transcript(self) -> None:
        transcript = "\n\n".join(self._transcript).strip()
        if not transcript:
            self._show_notice("No transcript to copy.")
            return
        copy_to_clipboard = getattr(self, "copy_to_clipboard", None)
        if callable(copy_to_clipboard):
            copy_to_clipboard(transcript)
            self._show_notice("Copied transcript to clipboard.")
        else:
            self._show_notice("Clipboard API is unavailable. Use /log.")

    def _handle_command(self, command: CliCommandResult) -> None:
        message = command.message
        if message == "help":
            self._show_help()
        elif message == "status":
            self._show_table("Status", self.session.status().items())
        elif message == "tools":
            self._show_table("Tools", self.session.tool_summaries())
        elif message == "clear":
            self.action_clear_conversation()
        elif message == "new":
            self._write_conversation_system(
                f"Started new session: {self.session.new_session()}"
            )
        elif message == "sessions":
            self._show_sessions()
        elif message and message.startswith("resume:"):
            self._start_resume_by_value(message.removeprefix("resume:"))
        elif message == "users":
            self._show_table(
                "Users",
                [
                    (user_id, "*" if user_id == self.session.user_id else "")
                    for user_id in self.session.list_users()
                ],
            )
        elif message and message.startswith("user:"):
            user_id = message.removeprefix("user:")
            self.session.set_user(user_id)
            self._write_conversation_system(f"Switched user to {user_id}")
        elif message == "memory":
            self._show_memory(self.session.list_long_term_memory())
        elif message and message.startswith("memory:"):
            self._show_memory(
                self.session.list_long_term_memory(message.removeprefix("memory:"))
            )
        elif message and message.startswith("delete-session:"):
            self._show_notice(
                self.session.delete_session(message.removeprefix("delete-session:"))
            )
        elif message and message.startswith("delete-memory:"):
            _prefix, kind, key = message.split(":", maxsplit=2)
            self._show_notice(self.session.delete_long_term_memory(kind, key))
        elif message and message.startswith("delete-user:"):
            self._show_notice(
                self.session.delete_user(message.removeprefix("delete-user:"))
            )
        elif message and message.startswith("export-session:"):
            self._show_notice(
                self.session.export_session(message.removeprefix("export-session:"))
            )
        elif message:
            self._show_notice(message)
        self._refresh_status()

    def _submit_turn(self, turn: CliTurn) -> None:
        self._write_user(turn.text)
        self._start_turn_timeline()
        self._run_agent(turn)

    @work(thread=True)
    def _run_agent(self, turn: CliTurn) -> None:
        try:
            for event in self.session.stream_events(turn):
                self.call_from_thread(self._handle_agent_event, event)
        except Exception as e:
            self.call_from_thread(
                self._write_conversation_system,
                f"Agent invocation failed: {type(e).__name__}: {e}",
            )
            self.call_from_thread(self._finish_turn_timeline, False)

    def _handle_agent_event(self, event: CliAgentEvent) -> None:
        if event.kind == "status":
            self._handle_status_event(event)
            self._refresh_status(self._agent_status_label(event.status))
            return
        if event.kind == "message" and event.message is not None:
            self._write_messages([event.message])
            return
        if event.kind == "done":
            self._finish_turn_timeline(True)
            return

    def _write_messages(self, messages: Iterable[Any]) -> None:
        for message in messages:
            if isinstance(message, AIMessage):
                if message.content:
                    self._write_assistant(str(message.content))
                for tool_call in message.tool_calls or []:
                    name = tool_call.get("name", "tool")
                    self._refresh_status(f"tool: {name}")
                    self._write_tool_call(name, tool_call.get("args", {}))
            elif isinstance(message, ToolMessage):
                self._refresh_status(f"tool result: {message.name or 'tool'}")
                if self._is_realtime_tool_result(message):
                    self._append_log(
                        "tool result", f"{message.name or 'tool'}\n{message.content}"
                    )
                    continue
                self._write_tool_result(message.name or "tool", message.content)
            elif isinstance(message, HumanMessage):
                self._write_user(str(message.content))

    def _show_help(self) -> None:
        self._show_notice(
            "Commands: /help, /status, /tools, /new, /sessions, /resume, "
            "/resume <thread_id> [--quiet], /session, /users, /user <id>, "
            "/memory, /memory facts, /memory locations, /delete-session <thread_id>, "
            "/delete-memory <facts|locations> <key>, /delete-user <user_id>, "
            "/clear, /export-session <path>, /copy-last, /copy-transcript, /log, /exit"
        )

    def _show_sessions(self) -> None:
        table = self._sessions_table(self.session.list_session_summaries())
        self._show_panel(table)

    def _show_session_picker(self) -> None:
        self._session_picker = self.session.list_session_summaries()
        self._session_picker_index = 0
        if not self._session_picker:
            self._show_notice("No sessions.")
            return
        self._render_session_picker()

    def _move_session_picker(self, delta: int) -> None:
        if not self._session_picker:
            return
        self._session_picker_index = (self._session_picker_index + delta) % len(
            self._session_picker
        )
        self._render_session_picker()

    def _resume_selected_session(self) -> None:
        if not self._session_picker:
            return
        summary = self._session_picker[self._session_picker_index]
        self._clear_command_panel()
        self._start_resume_by_value(summary.thread_id)

    def _start_resume_by_value(self, value: str) -> None:
        self.run_worker(
            self._resume_by_value(value),
            name="resume-session",
            group="tui",
            exclusive=True,
        )

    async def _resume_by_value(self, value: str) -> None:
        thread_id, messages, quiet = self.session.handle_resume_command(value)
        self._write_conversation_system(f"Resumed session: {thread_id}")
        if not quiet:
            await self._mount_resume_messages(messages)
            self.call_after_refresh(self._scroll_conversation_end)
        self._refresh_status("idle")

    def _render_session_picker(self) -> None:
        table = Table(title="Resume Session - Up/Down select, Enter resume, Esc cancel")
        table.add_column("")
        table.add_column("Created")
        table.add_column("First message")
        table.add_column("Thread ID")
        for index, summary in enumerate(self._session_picker):
            table.add_row(
                ">" if index == self._session_picker_index else "",
                summary.created_at_display,
                summary.first_user_message or "(empty)",
                summary.thread_id,
            )
        self._show_panel(table)

    def _sessions_table(self, sessions: Iterable[SessionSummary]) -> Table:
        table = Table(title="Sessions")
        table.add_column("Current")
        table.add_column("Created")
        table.add_column("First message")
        table.add_column("Thread ID")
        for summary in sessions:
            table.add_row(
                "*" if summary.thread_id == self.session.thread_id else "",
                summary.created_at_display,
                summary.first_user_message or "(empty)",
                summary.thread_id,
            )
        return table

    def _show_memory(self, items: list[tuple[str, Any, str, Any]]) -> None:
        table = Table(title=f"Memory: {self.session.namespace}/{self.session.user_id}")
        table.add_column("Type")
        table.add_column("Key")
        table.add_column("Value")
        for schema, _ns, key, value in items:
            table.add_row(schema, key, format_long_term_item(schema, key, value))
        self._show_panel(table)

    def _show_table(self, title: str, rows: Iterable[tuple[Any, Any]]) -> None:
        table = Table(title=title)
        table.add_column("Name")
        table.add_column("Value")
        for name, value in rows:
            table.add_row(str(name), str(value))
        self._show_panel(table)

    def _show_notice(self, message: str) -> None:
        self._show_panel(Panel(message, title="Command", border_style="cyan"))

    def _show_panel(self, renderable: Any) -> None:
        panel = self.query_one("#command_panel", Static)
        panel.update(renderable)
        panel.remove_class("hidden")

    def _clear_command_panel(self) -> None:
        self._session_picker = []
        panel = self.query_one("#command_panel", Static)
        panel.update("")
        panel.add_class("hidden")

    def _write_conversation_system(self, message: str) -> None:
        self._append_message("system", f"System\n{message}")
        self._append_log("system", message)

    def _write_user(self, message: str) -> None:
        self._append_message("user", f"User\n{message}")
        self._append_log("user", message)

    def _write_assistant(self, message: str) -> None:
        self._last_assistant_text = message
        self._append_message("assistant", f"Assistant\n{message}")
        self._append_log("assistant", message)

    def _write_tool_call(self, name: str, args: Any) -> None:
        args_summary = self._summarize_value(args)
        text = f"• Tool call {name}"
        if args_summary:
            text = f"{text}\n  └ {args_summary}"
        self._append_message(
            "tool", text, self._timeline_text("Tool call", name, args_summary)
        )
        self._append_log("tool call", f"{name}\n{_format_json(args)}")

    def _write_tool_result(self, name: str, content: Any) -> None:
        result_summary = self._summarize_value(content)
        text = f"• Tool result {name}"
        if result_summary:
            text = f"{text}\n  └ {result_summary}"
        self._append_message(
            "tool", text, self._timeline_text("Tool result", name, result_summary)
        )
        self._append_log("tool result", f"{name}\n{content}")

    def _copy_last_assistant_message(self) -> None:
        if not self._last_assistant_text:
            self._show_notice("No assistant message to copy.")
            return
        copy_to_clipboard = getattr(self, "copy_to_clipboard", None)
        if callable(copy_to_clipboard):
            copy_to_clipboard(self._last_assistant_text)
            self._show_notice("Copied last assistant message to clipboard.")
        else:
            self._show_notice(
                "Clipboard API is unavailable in this terminal. Use /log or "
                "/export-session <path>."
            )

    def _show_log_status(self) -> None:
        if self.log_path is None:
            self._show_notice("Log mirror is disabled.")
            return
        self._show_notice(f"Log mirror: {self.log_path}")

    def _append_log(self, role: str, content: str) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(f"\n\n## {role} - {timestamp}\n\n{content}\n")

    def _write_activity(self, message: str, renderable: Any | None = None) -> None:
        if message == self._last_activity_status:
            return
        self._last_activity_status = message
        self._append_message("activity", message, renderable)
        self._append_log("activity", message)

    def _append_message(
        self, role: str, text: str, renderable: Any | None = None
    ) -> Static | Markdown:
        self._transcript.append(text)
        widget = self._message_widget(role, text, renderable)
        self._conversation().mount(widget)
        self._move_working_activity_to_end(after=widget)
        self._scroll_conversation_end()
        return widget

    def _message_widget(
        self, role: str, text: str, renderable: Any | None = None
    ) -> Static | Markdown:
        if renderable is not None:
            return Static(renderable, classes=f"message {role}")
        widget: Static | Markdown
        title, _, body = text.partition("\n")
        if role == "assistant":
            widget = Markdown(body or text, classes=f"message {role}")
            widget.border_title = f" {title} " if body else ""
        elif role == "user":
            widget = Static(body or text, classes=f"message {role}")
            widget.border_title = f" {title} " if body else ""
        else:
            widget = Static(text, classes=f"message {role}")
        return widget

    async def _mount_resume_messages(self, messages: Iterable[Any]) -> None:
        widgets: list[Static | Markdown] = []
        for message in messages:
            widgets.extend(self._resume_message_widgets(message))
        if widgets:
            await self._conversation().mount(*widgets)
        self._scroll_conversation_end()

    def _resume_message_widgets(self, message: Any) -> list[Static | Markdown]:
        widgets: list[Static | Markdown] = []
        if isinstance(message, AIMessage):
            if message.content:
                content = str(message.content)
                self._last_assistant_text = content
                text = f"Assistant\n{content}"
                self._transcript.append(text)
                self._append_log("assistant", content)
                widgets.append(self._message_widget("assistant", text))
            for tool_call in message.tool_calls or []:
                name = tool_call.get("name", "tool")
                args = _format_json(tool_call.get("args", {}))
                text = f"Tool call: {name}\n{args}"
                self._transcript.append(text)
                self._append_log("tool call", f"{name}\n{args}")
                widgets.append(self._message_widget("tool", text))
        elif isinstance(message, ToolMessage):
            name = message.name or "tool"
            text = f"Tool result: {name}\n{message.content}"
            self._transcript.append(text)
            self._append_log("tool result", f"{name}\n{message.content}")
            widgets.append(self._message_widget("tool", text))
        elif isinstance(message, HumanMessage):
            content = str(message.content)
            text = f"User\n{content}"
            self._transcript.append(text)
            self._append_log("user", content)
            widgets.append(self._message_widget("user", text))
        return widgets

    def _scroll_conversation_end(self) -> None:
        conversation = self._conversation()
        conversation.scroll_end(animate=False, immediate=True, force=True)
        conversation.set_scroll(None, conversation.max_scroll_y)

    def _start_turn_timeline(self) -> None:
        self._turn_started_at = monotonic()
        self._stop_turn_timer()
        self._tool_activity_widgets.clear()
        self._realtime_tool_result_ids.clear()
        text = "• Working (0s)"
        self._working_transcript_index = len(self._transcript)
        self._working_widget = self._append_message(
            "activity",
            text,
            self._timeline_text("Working", "(0s)", style="#8fb3c8"),
        )
        self._append_log("activity", text)
        self._turn_timer = self.set_interval(
            1.0, self._refresh_working_status, name="turn-elapsed"
        )

    def _finish_turn_timeline(self, succeeded: bool) -> None:
        elapsed = self._turn_elapsed_text()
        self._stop_turn_timer()
        if succeeded:
            text = f"• Worked for {elapsed}"
            renderable = self._timeline_text("Worked for", elapsed, style="#86c39a")
            self._refresh_status("idle")
        else:
            text = f"• Failed after {elapsed}"
            renderable = self._timeline_text("Failed after", elapsed, style="#d06f6f")
            self._refresh_status("error")
        self._update_working_activity(text, renderable)
        self._append_log("activity", text)
        self._turn_started_at = None
        self._working_widget = None
        self._working_transcript_index = None

    def _refresh_working_status(self) -> None:
        if self._turn_started_at is not None:
            elapsed = self._turn_elapsed_text()
            text = f"• Working ({elapsed})"
            self._update_working_activity(
                text, self._timeline_text("Working", f"({elapsed})", style="#8fb3c8")
            )

    def _stop_turn_timer(self) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None

    def _turn_elapsed_text(self) -> str:
        if self._turn_started_at is None:
            return "0s"
        return _format_duration(monotonic() - self._turn_started_at)

    def _agent_status_label(self, status: str) -> str:
        if not status:
            return "Working"
        if status.startswith("model: error"):
            return "Model error"
        if status.startswith("model:"):
            return "Thinking"
        if status.startswith("tool: "):
            tool_name = status.removeprefix("tool: ").rsplit(" ", maxsplit=1)[0]
            if status.endswith("starting"):
                return f"Running {tool_name}"
            if status.endswith("error"):
                return f"Tool error: {tool_name}"
            return f"Tool complete: {tool_name}"
        if status.startswith("step:") or status in {"agent: starting", "starting"}:
            return "Working"
        return status

    def _handle_status_event(self, event: CliAgentEvent) -> None:
        status = event.status
        if not status.startswith("tool: "):
            return
        tool_name = status.removeprefix("tool: ").rsplit(" ", maxsplit=1)[0]
        event_data = event.data if isinstance(event.data, dict) else {}
        tool_key = str(event_data.get("run_id") or tool_name)
        payload = (
            event_data.get("data") if isinstance(event_data.get("data"), dict) else {}
        )
        if status.endswith("starting"):
            self._start_tool_activity(tool_key, tool_name, payload.get("input"))
        elif status.endswith("done"):
            self._finish_tool_activity(tool_key, tool_name, payload.get("output"), True)
        elif status.endswith("error"):
            self._finish_tool_activity(tool_key, tool_name, payload.get("error"), False)

    def _start_tool_activity(self, tool_key: str, name: str, args: Any) -> None:
        summary = self._summarize_value(args)
        text = f"• Running {name}"
        if summary:
            text = f"{text}\n  └ {summary}"
        widget = self._append_message(
            "activity", text, self._timeline_text("Running", name, summary)
        )
        if isinstance(widget, Static):
            self._tool_activity_widgets[tool_key] = (widget, text)
            self._tool_activity_widgets[name] = (widget, text)
        self._append_log("activity", text)

    def _finish_tool_activity(
        self, tool_key: str, name: str, result: Any, succeeded: bool
    ) -> None:
        summary = self._summarize_value(result)
        if isinstance(result, ToolMessage) and result.tool_call_id:
            self._realtime_tool_result_ids.add(str(result.tool_call_id))
        action = "Ran" if succeeded else "Tool failed"
        style = "#86c39a" if succeeded else "#d06f6f"
        text = f"• {action} {name}"
        if summary:
            text = f"{text}\n  └ {summary}"
        renderable = self._timeline_text(action, name, summary, style=style)
        existing = self._tool_activity_widgets.pop(
            tool_key, self._tool_activity_widgets.pop(name, None)
        )
        if existing is None:
            self._append_message("activity", text, renderable)
        else:
            widget, old_text = existing
            self._tool_activity_widgets.pop(tool_key, None)
            self._tool_activity_widgets.pop(name, None)
            widget.update(renderable)
            if old_text in self._transcript:
                self._transcript[self._transcript.index(old_text)] = text
            else:
                self._transcript.append(text)
        self._append_log("activity", text)

    def _update_working_activity(self, text: str, renderable: Text) -> None:
        if self._working_widget is None or self._working_transcript_index is None:
            self._working_transcript_index = len(self._transcript)
            self._working_widget = self._append_message("activity", text, renderable)
            return
        self._working_widget.update(renderable)
        self._transcript[self._working_transcript_index] = text
        self._move_working_activity_to_end()
        self._scroll_conversation_end()
        self.call_after_refresh(self._scroll_conversation_end)

    def _move_working_activity_to_end(
        self, after: Static | Markdown | None = None
    ) -> None:
        if self._working_widget is None or self._working_transcript_index is None:
            return
        if after is self._working_widget:
            return
        conversation = self._conversation()
        if conversation.children and conversation.children[-1] is self._working_widget:
            return
        try:
            conversation.move_child(
                self._working_widget,
                after=after if after is not None else len(conversation.children) - 1,
            )
        except ValueError:
            return
        working_text = self._transcript.pop(self._working_transcript_index)
        self._transcript.append(working_text)
        self._working_transcript_index = len(self._transcript) - 1

    def _timeline_text(
        self,
        action: str,
        target: str = "",
        detail: str = "",
        *,
        style: str = "#7f8c95",
    ) -> Text:
        text = Text("• ", style="#6f8796")
        text.append(action, style=style)
        if target:
            text.append(f" {target}", style="#d7dde2")
        if detail:
            text.append("\n  └ ", style="#6f8796")
            text.append(detail, style="#8f9ba4")
        return text

    def _summarize_value(self, value: Any, limit: int = 180) -> str:
        if value is None:
            return ""
        if isinstance(value, ToolMessage):
            value = value.content
        if isinstance(value, str):
            summary = value
        else:
            summary = _format_json(value)
        summary = " ".join(summary.split())
        if len(summary) > limit:
            return f"{summary[: limit - 1]}…"
        return summary

    def _is_realtime_tool_result(self, message: ToolMessage) -> bool:
        return bool(
            message.tool_call_id
            and str(message.tool_call_id) in self._realtime_tool_result_ids
        )

    def _as_markdown_message(self, text: str) -> str:
        title, _, body = text.partition("\n")
        if not body:
            return text
        return f"**{title}**\n\n{body}"

    def _refresh_status(self, status: str | None = None) -> None:
        if status is not None:
            self._status = status
        self.query_one("#agent_status", Static).update(self._status_text())

    def _status_text(self) -> str:
        return (
            f"user={self.session.user_id} | namespace={self.session.namespace} | "
            f"thread={self.session.thread_id} | "
            "Enter send | Shift+Enter newline | Ctrl+C clear/exit | /resume"
        )

    def _conversation(self) -> VerticalScroll:
        return self.query_one("#conversation", VerticalScroll)


def default_tui_log_path(session: MemoryCliSession) -> Path:
    return (
        Path.home()
        / ".rai"
        / "tui_logs"
        / f"{session.namespace}-{session.user_id}-{session.thread_id}.md"
    )


def run_memory_tui(
    session: MemoryCliSession,
    *,
    log_path: str | Path | None = None,
) -> None:
    MemoryTuiApp(
        session,
        log_path=default_tui_log_path(session) if log_path is None else log_path,
    ).run()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
