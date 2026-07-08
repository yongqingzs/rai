from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Header, Input, RichLog, Static

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


class MemoryTuiApp(App):
    """Codex-style terminal UI for RAI memory agents."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #conversation {
        height: 1fr;
        border: solid $primary;
    }

    #command_panel {
        height: auto;
        max-height: 12;
        padding: 0 1;
        border: round $accent;
    }

    #command_panel.hidden {
        display: none;
    }

    #agent_status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_conversation", "Clear", show=False),
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
        self.log_path = Path(log_path).expanduser() if log_path is not None else None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="conversation", markup=True, wrap=True, highlight=True)
        yield Static("", id="command_panel", classes="hidden")
        yield Input(placeholder="Message or /help", id="input")
        yield Static(self._status_text(), id="agent_status")

    def on_mount(self) -> None:
        self._write_conversation_system(
            "RAI TUI started. Use /help for commands, /resume to choose a session."
        )
        self._refresh_status("idle")
        self.query_one("#input", Input).focus()

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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
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
        self.query_one("#conversation", RichLog).clear()

    def action_dismiss_panel(self) -> None:
        self._clear_command_panel()

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
            self._resume_by_value(message.removeprefix("resume:"))
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
        self._refresh_status("running")
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
            self.call_from_thread(self._refresh_status, "error")

    def _handle_agent_event(self, event: CliAgentEvent) -> None:
        if event.kind == "status":
            if event.status:
                self._write_activity(event.status)
            self._refresh_status(event.status or "running")
            return
        if event.kind == "message" and event.message is not None:
            self._write_messages([event.message])
            return
        if event.kind == "done":
            self._write_activity("agent: done")
            self._refresh_status("idle")
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
                self._write_tool_result(message.name or "tool", message.content)
            elif isinstance(message, HumanMessage):
                self._write_user(str(message.content))

    def _show_help(self) -> None:
        self._show_notice(
            "Commands: /help, /status, /tools, /new, /sessions, /resume, "
            "/resume <thread_id> [--quiet], /session, /users, /user <id>, "
            "/memory, /memory facts, /memory locations, /delete-session <thread_id>, "
            "/delete-memory <facts|locations> <key>, /delete-user <user_id>, "
            "/clear, /export-session <path>, /copy-last, /log, /exit"
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
        self._resume_by_value(summary.thread_id)

    def _resume_by_value(self, value: str) -> None:
        thread_id, messages, quiet = self.session.handle_resume_command(value)
        self._write_conversation_system(f"Resumed session: {thread_id}")
        if not quiet:
            self._write_messages(messages)
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
        self._conversation().write(Panel(message, title="System", border_style="cyan"))
        self._append_log("system", message)

    def _write_user(self, message: str) -> None:
        self._conversation().write(Panel(message, title="User", border_style="blue"))
        self._append_log("user", message)

    def _write_assistant(self, message: str) -> None:
        self._last_assistant_text = message
        self._conversation().write(
            Panel(Markdown(message), title="Assistant", border_style="green")
        )
        self._append_log("assistant", message)

    def _write_tool_call(self, name: str, args: Any) -> None:
        self._conversation().write(
            Panel(
                Syntax(_format_json(args), "json", word_wrap=True),
                title=f"Tool call: {name}",
                border_style="yellow",
            )
        )
        self._append_log("tool call", f"{name}\n{_format_json(args)}")

    def _write_tool_result(self, name: str, content: Any) -> None:
        self._conversation().write(
            Panel(str(content), title=f"Tool result: {name}", border_style="magenta")
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

    def _write_activity(self, message: str) -> None:
        if message == self._last_activity_status:
            return
        self._last_activity_status = message
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._conversation().write(f"[dim]{timestamp}[/dim] {message}")
        self._append_log("activity", message)

    def _refresh_status(self, status: str | None = None) -> None:
        if status is not None:
            self._status = status
        self.query_one("#agent_status", Static).update(self._status_text())

    def _status_text(self) -> str:
        return (
            f"{self._status} | user={self.session.user_id} | "
            f"namespace={self.session.namespace} | thread={self.session.thread_id} | "
            "Enter send | /resume select | /copy-last | /log | Ctrl+C exit"
        )

    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)


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
