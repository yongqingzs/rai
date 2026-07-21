from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Header, Static
from textual.worker import WorkerState, get_current_worker

from rai.frontend.cli import (
    CliAgentEvent,
    CliCommandResult,
    CliTurn,
    MemoryCliSession,
    _format_json,
    parse_cli_input,
)
from rai.frontend.clipboard import copy_text_to_clipboard
from rai.frontend.tui_adapter import TuiEventAdapter
from rai.frontend.tui_input import (
    ChatTextArea,
    CommandSpec,
    command_hint,
    matching_commands,
)
from rai.frontend.tui_widgets import (
    ActivityMessage,
    AssistantMessage,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    present_tool_result,
)
from rai.memory.long_term import format_long_term_item
from rai.memory.session import SessionSummary

TUI_COLORS = {
    "background": "#151b23",
    "surface": "#1b222c",
    "panel": "#202936",
    "border": "#334155",
    "border_focus": "#5f7896",
    "text": "#d8dee9",
    "muted": "#8290a3",
    "primary": "#82aaff",
    "assistant": "#72c7b7",
    "tool": "#d6ad68",
    "success": "#8fbc8f",
    "warning": "#d7b46a",
    "error": "#d77a7a",
    "selection": "#315a78",
}

RAI_AGENT_THEME = Theme(
    name="rai_agent_dark",
    primary=TUI_COLORS["primary"],
    secondary=TUI_COLORS["muted"],
    accent=TUI_COLORS["assistant"],
    foreground=TUI_COLORS["text"],
    background=TUI_COLORS["background"],
    surface=TUI_COLORS["surface"],
    panel=TUI_COLORS["panel"],
    boost="#273241",
    success=TUI_COLORS["success"],
    warning=TUI_COLORS["warning"],
    error=TUI_COLORS["error"],
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "input-cursor-background": "#c7d3df",
        "input-cursor-foreground": TUI_COLORS["background"],
        "input-selection-background": f"{TUI_COLORS['selection']} 80%",
        "scrollbar": TUI_COLORS["border"],
        "scrollbar-hover": TUI_COLORS["border_focus"],
        "scrollbar-active": TUI_COLORS["primary"],
        "scrollbar-background": TUI_COLORS["surface"],
        "rai-border": TUI_COLORS["border"],
        "rai-border-focus": TUI_COLORS["border_focus"],
        "rai-muted": TUI_COLORS["muted"],
        "rai-user": TUI_COLORS["primary"],
        "rai-assistant": TUI_COLORS["assistant"],
        "rai-tool": TUI_COLORS["tool"],
        "rai-vision": "#c099d8",
        "rai-rag": "#85b7d9",
        "rai-navigation": "#79c6a3",
        "rai-ros": "#d6ad68",
    },
)


class MemoryTuiApp(App):
    """Codex-style terminal UI for RAI memory agents."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    Screen > .screen--selection {
        background: $input-selection-background;
        color: $foreground;
        text-style: bold;
    }

    #conversation {
        height: 1fr;
        border: none;
        padding: 1 1;
        background: $background;
        scrollbar-gutter: stable;
    }

    #command_panel {
        height: auto;
        max-height: 45vh;
        padding: 0 1;
        border-left: solid $rai-border-focus;
        background: $surface;
        color: $foreground;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    #command_panel.hidden {
        display: none;
    }

    #input_shell {
        height: auto;
        min-height: 1;
        max-height: 6;
        border: tall $rai-border;
        background: $surface;
    }

    #input_shell:focus-within {
        border: tall $rai-border-focus;
        background: $panel;
    }

    #input_prompt {
        width: 3;
        height: 100%;
        padding: 0 0 0 1;
        color: $primary;
        text-style: bold;
        background: transparent;
    }

    ChatTextArea {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 6;
        border: none;
        padding: 0;
        background: transparent;
        color: $foreground;
    }

    ChatTextArea:focus {
        border: none;
        background: transparent;
    }

    ChatTextArea .text-area--cursor {
        background: $input-cursor-background;
        color: $input-cursor-foreground;
        text-style: none;
    }

    ChatTextArea .text-area--selection {
        background: $input-selection-background;
        color: $foreground;
    }

    ChatTextArea .text-area--cursor-line {
        background: transparent;
    }

    ChatTextArea .text-area--placeholder {
        color: $rai-muted;
    }

    .message {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1 0 2;
        border-left: solid $rai-border;
        background: $surface 55%;
        color: $foreground;
    }

    .message.user {
        border-left: solid $rai-user;
        background: $surface 35%;
    }

    .message.assistant {
        border-left: solid $rai-assistant;
        background: $panel 45%;
    }

    .message.system {
        color: $rai-muted;
        border-left: solid $rai-border;
        background: transparent;
    }

    .message.tool {
        border-left: solid $rai-tool;
        background: $surface 45%;
    }

    .message.tool.vision { border-left: solid $rai-vision; }
    .message.tool.rag { border-left: solid $rai-rag; }
    .message.tool.navigation { border-left: solid $rai-navigation; }
    .message.tool.ros { border-left: solid $rai-ros; }

    .message-header {
        height: 1;
        text-style: bold;
        color: $rai-muted;
    }

    .user .message-header { color: $rai-user; }
    .assistant .message-header { color: $rai-assistant; }

    .message-body {
        height: auto;
        padding: 0;
        background: transparent;
        color: $foreground;
    }

    .message-hint, .tool-hint {
        height: 1;
        color: $rai-muted;
    }

    .message-hint:hover, .tool-hint:hover {
        color: $primary;
        text-style: underline;
    }

    .tool-header {
        height: 1;
        color: $rai-tool;
    }

    .vision .tool-header { color: $rai-vision; }
    .rag .tool-header { color: $rai-rag; }
    .navigation .tool-header { color: $rai-navigation; }
    .ros .tool-header { color: $rai-ros; }

    .tool-summary {
        height: auto;
        color: $foreground;
        padding-left: 2;
    }

    .tool-detail {
        height: auto;
        margin-top: 1;
        padding: 0 1;
        color: $foreground;
        background: $background 45%;
    }

    .hidden {
        display: none;
    }

    .message.activity {
        color: $rai-muted;
        margin: 0 0 1 0;
        padding: 0 1;
        border: none;
        background: transparent;
    }

    #agent_status {
        height: 1;
        padding: 0 1;
        background: $background;
        color: $rai-muted;
    }
    """

    BINDINGS = [
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
        self.register_theme(RAI_AGENT_THEME)
        self.theme = RAI_AGENT_THEME.name
        self.session = session
        self._status = "idle"
        self._picker_mode: str | None = None
        self._session_picker: list[SessionSummary] = []
        self._session_picker_index = 0
        self._memory_picker: list[tuple[str, Any, str, Any]] = []
        self._memory_picker_index = 0
        self._memory_picker_kind = ""
        self._command_matches: list[CommandSpec] = []
        self._command_match_index = 0
        self._command_panel_mode: str | None = None
        self._timestamps_visible = False
        self._last_assistant_text = ""
        self._last_activity_status = ""
        self._turn_started_at: float | None = None
        self._turn_timer: Any | None = None
        self._working_widget: ActivityMessage | None = None
        self._working_transcript_index: int | None = None
        self._tool_activity_widgets: dict[str, tuple[ToolCallMessage, str]] = {}
        self._context_summary_activity: tuple[ActivityMessage, str] | None = None
        self._context_summary_started_at: float | None = None
        self._event_adapter = TuiEventAdapter(self)
        self._agent_worker: Any | None = None
        self._turn_sequence = 0
        self._active_turn_id: int | None = None
        self._canceled_turn_ids: set[int] = set()
        self._interrupted_turn_ids: set[int] = set()
        self._input_history: list[str] = []
        self._input_history_index: int | None = None
        self._input_history_draft = ""
        self._last_copy_panel_title = ""
        self._last_copy_panel_status = ""
        self._last_copy_panel_text = ""
        self._last_copy_system_target = ""
        self._transcript: list[str] = []
        self.log_path = Path(log_path).expanduser() if log_path is not None else None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="conversation")
        with VerticalScroll(id="command_panel", classes="hidden"):
            yield Static("", id="command_panel_content")
        with Horizontal(id="input_shell"):
            yield Static("›", id="input_prompt")
            yield ChatTextArea(
                id="input",
                placeholder="Ask RAI or type / for commands",
                highlight_cursor_line=False,
            )
        yield Static(self._status_text(), id="agent_status")

    def on_mount(self) -> None:
        self._write_conversation_system(
            "RAI TUI started. Use /help for commands, /resume to choose a session."
        )
        self._refresh_status("idle")
        self.query_one("#input", ChatTextArea).focus()

    def on_key(self, event) -> None:
        if not self._has_active_picker():
            return
        if event.key == "up":
            self._move_picker_selection(-1)
            event.stop()
        elif event.key == "down":
            self._move_picker_selection(1)
            event.stop()
        elif event.key == "enter":
            self._confirm_picker_selection()
            event.stop()
        elif event.key == "escape":
            self._clear_command_panel()
            event.stop()

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        value = event.text.strip()
        event.text_area.clear_after_submit()
        if not value:
            return
        self._remember_input(value)
        self._clear_command_panel()
        if value in {"/resume", "/session"}:
            self._show_session_picker()
            return
        if value == "/delete-session":
            self._show_delete_session_picker()
            return
        if value == "/delete-memory":
            self._show_delete_memory_picker()
            return
        if value in {"/delete-memory facts", "/delete-memory locations"}:
            self._show_delete_memory_picker(value.rsplit(" ", maxsplit=1)[1])
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
        if value == "/timestamps":
            self._toggle_timestamps()
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

    def on_text_area_changed(self, event: ChatTextArea.Changed) -> None:
        if self.is_running and event.text_area.id == "input":
            self._refresh_command_assist(event.text_area.text)

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
        self._copy_text_with_fallback(transcript, title="Transcript")

    def _copy_selected_text(self) -> bool:
        selected_text = self.screen.get_selected_text()
        if selected_text is None:
            return False
        self._copy_text_with_fallback(selected_text, title="Selection")
        self.screen.clear_selection()
        return True

    def _copy_text_with_fallback(self, text: str, *, title: str = "Copy") -> None:
        osc52_requested = False
        use_pyperclip = (
            getattr(self, "_supports_pyperclip", None) is not False
            and "copy_to_clipboard" not in self.__dict__
        )
        osc52_requested = self._has_terminal_clipboard()
        result = copy_text_to_clipboard(
            self,
            text,
            use_pyperclip=use_pyperclip,
            use_osc52=osc52_requested,
        )
        self._last_copy_system_target = (
            result.method if result.method == "pyperclip" else ""
        )
        if result.method == "pyperclip":
            status = (
                "Copied to system clipboard via pyperclip. "
                "If paste fails, copy from the text below."
            )
        elif osc52_requested:
            status = (
                "Terminal clipboard copy requested. If external paste fails, your "
                "terminal or SSH session likely blocks OSC52; copy from the text below."
            )
        elif result.success:
            status = (
                "Copied to TUI local clipboard only. External apps cannot paste this; "
                "copy from the text below."
            )
        else:
            status = (
                "Automatic clipboard copy is unavailable. Copy from the text below."
            )
        self._show_copy_panel(title, status, text)

    def _has_terminal_clipboard(self) -> bool:
        return getattr(self, "_driver", None) is not None

    def _show_copy_panel(self, title: str, status: str, text: str) -> None:
        self._last_copy_panel_title = title
        self._last_copy_panel_status = status
        self._last_copy_panel_text = text
        self._show_panel(
            Panel(
                Text(f"{status}\n\n{text}", style=TUI_COLORS["text"]),
                title=title,
                border_style=TUI_COLORS["assistant"],
            )
        )

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
        self._turn_sequence += 1
        self._active_turn_id = self._turn_sequence
        self._agent_worker = self._run_agent(turn, self._active_turn_id)

    @work(thread=True)
    def _run_agent(self, turn: CliTurn, turn_id: int) -> None:
        worker = get_current_worker()
        try:
            for event in self.session.stream_events(turn):
                if worker.state == WorkerState.CANCELLED or self._is_turn_canceled(
                    turn_id
                ):
                    self.call_from_thread(self._finish_turn_interrupted, turn_id)
                    return
                self.call_from_thread(self._handle_agent_event, turn_id, event)
            if worker.state == WorkerState.CANCELLED or self._is_turn_canceled(turn_id):
                self.call_from_thread(self._finish_turn_interrupted, turn_id)
        except Exception as e:
            if self._is_turn_canceled(turn_id):
                self.call_from_thread(self._finish_turn_interrupted, turn_id)
                return
            self.call_from_thread(
                self._write_conversation_system,
                f"Agent invocation failed: {type(e).__name__}: {e}",
            )
            self.call_from_thread(self._finish_turn_timeline, False)

    def _handle_agent_event(self, turn_id: int, event: CliAgentEvent) -> None:
        if self._is_turn_canceled(turn_id) or turn_id != self._active_turn_id:
            return
        if self._event_adapter.handle_event(event):
            self._finish_turn_timeline(True)

    def _remember_input(self, value: str) -> None:
        if not self._input_history or self._input_history[-1] != value:
            self._input_history.append(value)
        self._input_history_index = None
        self._input_history_draft = ""

    def _recall_input_history(self, direction: int) -> None:
        if not self._input_history:
            return
        text_area = self.query_one("#input", ChatTextArea)
        if self._input_history_index is None:
            if direction > 0:
                return
            self._input_history_draft = text_area.text
            self._input_history_index = len(self._input_history) - 1
        else:
            self._input_history_index += direction
            if self._input_history_index < 0:
                self._input_history_index = 0
            elif self._input_history_index >= len(self._input_history):
                self._input_history_index = None
                self._load_input_text(self._input_history_draft)
                return
        self._load_input_text(self._input_history[self._input_history_index])

    def _load_input_text(self, value: str) -> None:
        text_area = self.query_one("#input", ChatTextArea)
        text_area.load_text(value)
        lines = value.splitlines() or [""]
        text_area.move_cursor((len(lines) - 1, len(lines[-1])))

    def _is_agent_running(self) -> bool:
        if self._turn_started_at is None:
            return False
        if self._agent_worker is None:
            return True
        return self._agent_worker.state in {WorkerState.PENDING, WorkerState.RUNNING}

    def _is_turn_canceled(self, turn_id: int) -> bool:
        return turn_id in self._canceled_turn_ids

    def _cancel_agent_turn(self) -> None:
        if not self._is_agent_running():
            return
        turn_id = self._active_turn_id
        if turn_id is None:
            return
        self._canceled_turn_ids.add(turn_id)
        worker = self._agent_worker
        if worker is not None:
            worker.cancel()
        self._finish_turn_interrupted(turn_id)

    def _write_messages(self, messages: Iterable[Any]) -> None:
        self._event_adapter.write_messages(messages)

    def _show_help(self) -> None:
        self._show_notice(
            "Commands: /help, /status, /tools, /new, /sessions, /resume, "
            "/resume <thread_id> [--quiet], /session, /users, /user <id>, "
            "/memory, /memory facts, /memory locations, /delete-session [thread_id], "
            "/delete-memory [facts|locations] [key], /delete-user <user_id>, "
            "/clear, /export-session <path>, /copy-last, /copy-transcript, /log, "
            "/timestamps, /exit"
        )

    def _show_sessions(self) -> None:
        table = self._sessions_table(self.session.list_session_summaries())
        self._show_panel(table)

    def _show_session_picker(self) -> None:
        self._picker_mode = "resume"
        self._session_picker = self.session.list_session_summaries()
        self._session_picker_index = 0
        if not self._session_picker:
            self._show_notice("No sessions.")
            return
        self._render_session_picker()

    def _show_delete_session_picker(self) -> None:
        self._picker_mode = "delete-session"
        self._session_picker = self.session.list_session_summaries()
        self._session_picker_index = 0
        if not self._session_picker:
            self._show_notice("No sessions.")
            return
        self._render_session_picker()

    def _show_delete_memory_picker(self, kind: str | None = None) -> None:
        self._picker_mode = "delete-memory"
        self._memory_picker_kind = kind or ""
        self._memory_picker = self.session.list_long_term_memory(kind)
        self._memory_picker_index = 0
        if not self._memory_picker:
            suffix = f" ({kind})" if kind else ""
            self._show_notice(f"No memory items{suffix}.")
            return
        self._render_memory_picker()

    def _refresh_command_assist(self, value: str) -> None:
        if self._has_active_picker():
            return
        matches = matching_commands(value)
        exact = next((item for item in matches if item.command == value), None)
        if exact is not None:
            matches = []
        if matches:
            self._command_matches = matches[:8]
            self._command_match_index = min(
                self._command_match_index, len(self._command_matches) - 1
            )
            self._command_panel_mode = "completion"
            self._render_command_completions()
            return
        self._command_matches = []
        self._command_match_index = 0
        hint = command_hint(value)
        if hint is not None:
            self._command_panel_mode = "hint"
            self._show_panel(
                Text.assemble(
                    (hint.usage, f"bold {TUI_COLORS['primary']}"),
                    (f"  {hint.description}", TUI_COLORS["muted"]),
                )
            )
        elif self._command_panel_mode in {"completion", "hint"}:
            self._clear_command_panel()

    def _render_command_completions(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("", width=1)
        table.add_column("Command")
        table.add_column("Description", style=TUI_COLORS["muted"])
        for index, spec in enumerate(self._command_matches):
            table.add_row(
                "›" if index == self._command_match_index else "",
                spec.usage,
                spec.description,
                style=TUI_COLORS["primary"]
                if index == self._command_match_index
                else None,
            )
        self._show_panel(table)

    def _has_command_completion(self) -> bool:
        return bool(self._command_panel_mode == "completion" and self._command_matches)

    def _move_command_completion(self, delta: int) -> None:
        if not self._command_matches:
            return
        self._command_match_index = (self._command_match_index + delta) % len(
            self._command_matches
        )
        self._render_command_completions()

    def _accept_command_completion(self) -> bool:
        if not self._command_matches:
            return False
        spec = self._command_matches[self._command_match_index]
        value = f"{spec.command} " if spec.arguments else spec.command
        self._command_matches = []
        self._command_match_index = 0
        self._load_input_text(value)
        self._refresh_command_assist(value)
        return True

    def _has_active_picker(self) -> bool:
        if self._picker_mode in {"resume", "delete-session"}:
            return bool(self._session_picker)
        if self._picker_mode == "delete-memory":
            return bool(self._memory_picker)
        return False

    def _move_picker_selection(self, delta: int) -> None:
        if self._picker_mode in {"resume", "delete-session"}:
            if not self._session_picker:
                return
            self._session_picker_index = (self._session_picker_index + delta) % len(
                self._session_picker
            )
            self._render_session_picker()
            return
        if self._picker_mode == "delete-memory":
            if not self._memory_picker:
                return
            self._memory_picker_index = (self._memory_picker_index + delta) % len(
                self._memory_picker
            )
            self._render_memory_picker()

    def _confirm_picker_selection(self) -> None:
        if self._picker_mode == "resume":
            self._resume_selected_session()
        elif self._picker_mode == "delete-session":
            self._delete_selected_session()
        elif self._picker_mode == "delete-memory":
            self._delete_selected_memory()

    def _resume_selected_session(self) -> None:
        if not self._session_picker:
            return
        summary = self._session_picker[self._session_picker_index]
        self._clear_command_panel()
        self._start_resume_by_value(summary.thread_id)

    def _delete_selected_session(self) -> None:
        if not self._session_picker:
            return
        summary = self._session_picker[self._session_picker_index]
        self._clear_command_panel()
        self._show_notice(self.session.delete_session(summary.thread_id))

    def _delete_selected_memory(self) -> None:
        if not self._memory_picker:
            return
        schema, _ns, key, _value = self._memory_picker[self._memory_picker_index]
        kind = self._memory_picker_kind or schema
        self._clear_command_panel()
        self._show_notice(self.session.delete_long_term_memory(kind, key))

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
        if self._picker_mode == "delete-session":
            title = "Delete Session - Up/Down select, Enter delete, Esc cancel"
        else:
            title = "Resume Session - Up/Down select, Enter resume, Esc cancel"
        table = Table(title=title)
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

    def _render_memory_picker(self) -> None:
        if self._memory_picker_kind:
            title = (
                f"Delete Memory: {self._memory_picker_kind} - "
                "Up/Down select, Enter delete, Esc cancel"
            )
        else:
            title = "Delete Memory - Up/Down select, Enter delete, Esc cancel"
        table = Table(title=title)
        table.add_column("")
        table.add_column("Type")
        table.add_column("Key")
        table.add_column("Value")
        for index, (schema, _ns, key, value) in enumerate(self._memory_picker):
            table.add_row(
                ">" if index == self._memory_picker_index else "",
                schema,
                key,
                format_long_term_item(schema, key, value),
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
        panel = self.query_one("#command_panel", VerticalScroll)
        content = self.query_one("#command_panel_content", Static)
        content.update(renderable)
        panel.remove_class("hidden")
        panel.scroll_home(animate=False, immediate=True)

    def _clear_command_panel(self) -> None:
        self._picker_mode = None
        self._session_picker = []
        self._session_picker_index = 0
        self._memory_picker = []
        self._memory_picker_index = 0
        self._memory_picker_kind = ""
        self._command_matches = []
        self._command_match_index = 0
        self._command_panel_mode = None
        panel = self.query_one("#command_panel", VerticalScroll)
        content = self.query_one("#command_panel_content", Static)
        content.update("")
        panel.add_class("hidden")

    def _write_conversation_system(self, message: str) -> None:
        self._append_message("system", f"System\n{message}")
        self._append_log("system", message)

    def refresh_status(self, status: str | None = None) -> None:
        self._refresh_status(status)

    def agent_status_label(self, status: str) -> str:
        return self._agent_status_label(status)

    def append_log(self, role: str, content: str) -> None:
        self._append_log(role, content)

    def write_user(self, message: str) -> None:
        self._write_user(message)

    def _write_user(self, message: str) -> None:
        self._append_message("user", f"User\n{message}")
        self._append_log("user", message)

    def write_assistant(self, message: str) -> None:
        self._write_assistant(message)

    def _write_assistant(self, message: str) -> None:
        self._last_assistant_text = message
        self._append_message("assistant", f"Assistant\n{message}")
        self._append_log("assistant", message)

    def write_tool_call(self, name: str, args: Any) -> None:
        self._write_tool_call(name, args)

    def _write_tool_call(self, name: str, args: Any) -> None:
        args_summary = self._summarize_value(args)
        text = f"• Tool call {name}"
        if args_summary:
            text = f"{text}\n  └ {args_summary}"
        self._transcript.append(text)
        self._mount_message_widget(
            ToolCallMessage(
                name,
                args_summary,
                self._timeline_text("Tool call", name, args_summary),
                full_content=args,
            )
        )
        self._append_log("tool call", f"{name}\n{_format_json(args)}")

    def write_tool_result(self, name: str, content: Any) -> None:
        self._write_tool_result(name, content)

    def _write_tool_result(self, name: str, content: Any) -> None:
        presentation = present_tool_result(name, content)
        result_summary = presentation.summary
        action = presentation.action
        text = f"• {action} {name}"
        if result_summary:
            text = f"{text}\n  └ {result_summary}"
        self._transcript.append(text)
        self._mount_message_widget(
            ToolResultMessage(
                name,
                result_summary,
                self._timeline_text(action, name, result_summary),
                full_content=content,
            )
        )
        self._append_log("tool result", f"{name}\n{content}")

    def _copy_last_assistant_message(self) -> None:
        if not self._last_assistant_text:
            self._show_notice("No assistant message to copy.")
            return
        self._copy_text_with_fallback(self._last_assistant_text, title="Assistant")

    def _toggle_timestamps(self) -> None:
        self._timestamps_visible = not self._timestamps_visible
        for widget in self.query(".message"):
            if hasattr(widget, "set_timestamp_visible"):
                widget.set_timestamp_visible(self._timestamps_visible)
        state = "shown" if self._timestamps_visible else "hidden"
        self._show_notice(f"Message timestamps are now {state}.")

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
    ) -> Widget:
        self._transcript.append(text)
        widget = self._message_widget(role, text, renderable)
        self._mount_message_widget(widget)
        return widget

    def _mount_message_widget(self, widget: Widget) -> None:
        if hasattr(widget, "set_timestamp_visible"):
            widget.set_timestamp_visible(self._timestamps_visible)
        self._conversation().mount(widget)
        self._move_working_activity_to_end(after=widget)
        self._scroll_conversation_end()

    def _message_widget(
        self, role: str, text: str, renderable: Any | None = None
    ) -> Widget:
        if renderable is not None:
            if role == "activity":
                return ActivityMessage(renderable)
            if role == "tool":
                title, _, body = text.partition("\n")
                action_name = title.removeprefix("• ").strip()
                action, _, name = action_name.partition(" ")
                return ToolResultMessage(
                    name or action,
                    body.removeprefix("  └ ").strip(),
                    renderable,
                    full_content=body.removeprefix("  └ ").strip(),
                )
            return Static(renderable, classes=f"message {role}")
        title, _, body = text.partition("\n")
        if role == "assistant":
            widget = AssistantMessage(body or text)
        elif role == "user":
            widget = UserMessage(body or text)
        elif role == "tool":
            detail = body.removeprefix("  └ ").strip()
            if title.startswith("Tool call:"):
                widget = ToolCallMessage(
                    title.removeprefix("Tool call:").strip(), detail
                )
            elif title.startswith("Tool result:"):
                widget = ToolResultMessage(
                    title.removeprefix("Tool result:").strip(),
                    full_content=detail,
                )
            else:
                first_line = title.removeprefix("• ").strip()
                action, _, name = first_line.partition(" ")
                if action == "Tool":
                    widget = ToolCallMessage(name.removeprefix("call "), detail)
                else:
                    widget = ToolResultMessage(name, full_content=detail)
        elif role == "system":
            widget = SystemMessage(body or text)
        elif role == "activity":
            widget = ActivityMessage(text)
        else:
            widget = Static(text, classes=f"message {role}")
        return widget

    async def _mount_resume_messages(self, messages: Iterable[Any]) -> None:
        widgets: list[Widget] = []
        for message in messages:
            widgets.extend(self._resume_message_widgets(message))
        if widgets:
            for widget in widgets:
                if hasattr(widget, "set_timestamp_visible"):
                    widget.set_timestamp_visible(self._timestamps_visible)
            await self._conversation().mount(*widgets)
        self._scroll_conversation_end()

    def _resume_message_widgets(self, message: Any) -> list[Widget]:
        widgets: list[Widget] = []
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
        self._context_summary_activity = None
        self._context_summary_started_at = None
        self._event_adapter.reset_turn()
        text = "• Working (0s)"
        self._working_transcript_index = len(self._transcript)
        self._working_widget = self._append_message(
            "activity",
            text,
            self._timeline_text("Working", "(0s)", style=TUI_COLORS["primary"]),
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
            renderable = self._timeline_text(
                "Worked for", elapsed, style=TUI_COLORS["success"]
            )
            self._refresh_status("idle")
        else:
            text = f"• Failed after {elapsed}"
            renderable = self._timeline_text(
                "Failed after", elapsed, style=TUI_COLORS["error"]
            )
            self._refresh_status("error")
        self._update_working_activity(text, renderable)
        self._append_log("activity", text)
        self._turn_started_at = None
        self._working_widget = None
        self._working_transcript_index = None
        self._agent_worker = None
        self._active_turn_id = None

    def _finish_turn_interrupted(self, turn_id: int) -> None:
        if turn_id in self._interrupted_turn_ids:
            return
        self._interrupted_turn_ids.add(turn_id)
        elapsed = self._turn_elapsed_text()
        self._stop_turn_timer()
        text = f"• Interrupted after {elapsed}"
        self._update_working_activity(
            text,
            self._timeline_text(
                "Interrupted after", elapsed, style=TUI_COLORS["warning"]
            ),
        )
        self._append_log("activity", text)
        if turn_id == self._active_turn_id:
            self._refresh_status("interrupted")
            self._turn_started_at = None
            self._working_widget = None
            self._working_transcript_index = None
            self._agent_worker = None
            self._active_turn_id = None
        self._scroll_conversation_end()

    def _refresh_working_status(self) -> None:
        if self._turn_started_at is not None:
            elapsed = self._turn_elapsed_text()
            text = f"• Working ({elapsed})"
            self._update_working_activity(
                text,
                self._timeline_text(
                    "Working", f"({elapsed})", style=TUI_COLORS["primary"]
                ),
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
        if status == "context: summarizing":
            return "Summarizing context"
        if status == "context: summarized":
            return "Context summarized"
        if status == "context: summary error":
            return "Context summary failed"
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

    def start_tool_activity(self, tool_key: str, name: str, args: Any) -> None:
        self._start_tool_activity(tool_key, name, args)

    def _start_tool_activity(self, tool_key: str, name: str, args: Any) -> None:
        summary = self._summarize_value(args)
        text = f"• Running {name}"
        if summary:
            text = f"{text}\n  └ {summary}"
        renderable = self._timeline_text("Running", name, summary)
        self._transcript.append(text)
        widget = ToolCallMessage(name, summary, renderable, full_content=args)
        self._conversation().mount(widget)
        self._move_working_activity_to_end(after=widget)
        self._scroll_conversation_end()
        self._tool_activity_widgets[tool_key] = (widget, text)
        self._tool_activity_widgets[name] = (widget, text)
        self._append_log("activity", text)

    def finish_tool_activity(
        self, tool_key: str, name: str, result: Any, succeeded: bool
    ) -> None:
        self._finish_tool_activity(tool_key, name, result, succeeded)

    def start_context_summary(self) -> None:
        self._start_context_summary()

    def _start_context_summary(self) -> None:
        self._context_summary_started_at = monotonic()
        text = "• Summarizing conversation context…"
        renderable = self._timeline_text(
            "Summarizing conversation context", "…", style=TUI_COLORS["primary"]
        )
        if self._context_summary_activity is None:
            widget = self._append_message("activity", text, renderable)
        else:
            widget, old_text = self._context_summary_activity
            widget.update(renderable)
            if old_text in self._transcript:
                self._transcript[self._transcript.index(old_text)] = text
        self._context_summary_activity = (widget, text)
        self._move_working_activity_to_end(after=widget)
        self._scroll_conversation_end()
        self._append_log("activity", text)

    def finish_context_summary(self, succeeded: bool) -> None:
        self._finish_context_summary(succeeded)

    def _finish_context_summary(self, succeeded: bool) -> None:
        elapsed = _format_duration(
            monotonic() - self._context_summary_started_at
            if self._context_summary_started_at is not None
            else 0.0
        )
        action = "Context summarized" if succeeded else "Context summarization failed"
        style = TUI_COLORS["success"] if succeeded else TUI_COLORS["error"]
        text = f"• {action} ({elapsed})"
        renderable = self._timeline_text(action, f"({elapsed})", style=style)
        if self._context_summary_activity is None:
            widget = self._append_message("activity", text, renderable)
        else:
            widget, old_text = self._context_summary_activity
            widget.update(renderable)
            if old_text in self._transcript:
                self._transcript[self._transcript.index(old_text)] = text
            else:
                self._transcript.append(text)
        self._context_summary_activity = (widget, text)
        self._context_summary_started_at = None
        self._move_working_activity_to_end(after=widget)
        self._scroll_conversation_end()
        self._append_log("activity", text)

    def _finish_tool_activity(
        self, tool_key: str, name: str, result: Any, succeeded: bool
    ) -> None:
        presentation = present_tool_result(name, result)
        summary = presentation.summary
        action = (
            presentation.action
            if succeeded and presentation.domain != "tool"
            else "Ran"
            if succeeded
            else "Tool failed"
        )
        style = TUI_COLORS["success"] if succeeded else TUI_COLORS["error"]
        text = f"• {action} {name}"
        if summary:
            text = f"{text}\n  └ {summary}"
        renderable = self._timeline_text(action, name, summary, style=style)
        existing = self._tool_activity_widgets.pop(
            tool_key, self._tool_activity_widgets.pop(name, None)
        )
        if existing is None:
            self._append_message("tool", text, renderable)
        else:
            widget, old_text = existing
            self._tool_activity_widgets.pop(tool_key, None)
            self._tool_activity_widgets.pop(name, None)
            widget.set_result(
                action,
                name,
                summary,
                renderable,
                full_content=result,
                domain=presentation.domain,
            )
            if old_text in self._transcript:
                self._transcript[self._transcript.index(old_text)] = text
            else:
                self._transcript.append(text)
            self._move_working_activity_to_end(after=widget)
            self._scroll_conversation_end()
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

    def _move_working_activity_to_end(self, after: Widget | None = None) -> None:
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
        style: str = TUI_COLORS["muted"],
    ) -> Text:
        text = Text("• ", style=TUI_COLORS["muted"])
        text.append(action, style=style)
        if target:
            text.append(f" {target}", style=TUI_COLORS["text"])
        if detail:
            text.append("\n  └ ", style=TUI_COLORS["muted"])
            text.append(detail, style=TUI_COLORS["muted"])
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
        if self._is_agent_running():
            state = (
                self._status if self._status not in {"idle", "running"} else "Working"
            )
            return f"{state} · Ctrl+C to interrupt"
        thread = self.session.thread_id
        short_thread = thread if len(thread) <= 16 else f"…{thread[-12:]}"
        return f"{self.session.user_id} · session {short_thread} · /help"

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
