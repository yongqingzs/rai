from __future__ import annotations

import re
from dataclasses import dataclass

from textual import events
from textual.message import Message
from textual.widgets import TextArea

PASTE_THRESHOLD_CHARS = 800
PASTE_THRESHOLD_LINES = 3
_PASTE_REFERENCE = re.compile(r"\[Pasted text #(\d+)(?: \+(\d+) lines)?\]")


@dataclass(frozen=True)
class CommandSpec:
    command: str
    arguments: str
    description: str

    @property
    def usage(self) -> str:
        return " ".join(part for part in (self.command, self.arguments) if part)


COMMAND_SPECS = (
    CommandSpec("/help", "", "Show available commands"),
    CommandSpec("/status", "", "Show agent and session status"),
    CommandSpec("/tools", "", "List available tools"),
    CommandSpec("/new", "", "Start a new session"),
    CommandSpec("/sessions", "", "List saved sessions"),
    CommandSpec("/resume", "[thread_id] [--quiet]", "Choose or resume a session"),
    CommandSpec("/users", "", "List memory users"),
    CommandSpec("/user", "<user_id>", "Switch memory user"),
    CommandSpec("/memory", "[facts|locations]", "Show long-term memory"),
    CommandSpec("/delete-session", "[thread_id]", "Choose a session to delete"),
    CommandSpec(
        "/delete-memory",
        "[facts|locations] [key]",
        "Choose a memory item to delete",
    ),
    CommandSpec("/delete-user", "<user_id>", "Delete a memory user"),
    CommandSpec("/export-session", "<path>", "Export the current session"),
    CommandSpec("/image", "<path> <message>", "Send an image with a message"),
    CommandSpec("/timestamps", "", "Show or hide message timestamps"),
    CommandSpec("/copy-last", "", "Copy the latest assistant response"),
    CommandSpec("/copy-transcript", "", "Copy the conversation transcript"),
    CommandSpec("/log", "", "Show the transcript log path"),
    CommandSpec("/clear", "", "Clear the visible conversation"),
    CommandSpec("/exit", "", "Exit the TUI"),
)


class ChatTextArea(TextArea):
    """Multiline agent input with command and large-paste support."""

    class Submitted(Message):
        def __init__(self, text: str, text_area: ChatTextArea) -> None:
            super().__init__()
            self.text = text
            self.text_area = text_area

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pasted_contents: dict[int, str] = {}
        self._next_paste_id = 1

    @property
    def submitted_value(self) -> str:
        def expand(match: re.Match[str]) -> str:
            return self._pasted_contents.get(int(match.group(1)), match.group(0))

        return _PASTE_REFERENCE.sub(expand, self.text)

    def clear_after_submit(self) -> None:
        self.load_text("")
        self._pasted_contents.clear()
        self._next_paste_id = 1

    async def _on_paste(self, event: events.Paste) -> None:
        event.stop()
        event.prevent_default()
        text = event.text
        line_count = text.count("\n") + 1
        if len(text) <= PASTE_THRESHOLD_CHARS and line_count <= PASTE_THRESHOLD_LINES:
            self.insert(text)
            return
        paste_id = self._next_paste_id
        self._next_paste_id += 1
        self._pasted_contents[paste_id] = text
        self.insert(f"[Pasted text #{paste_id} +{line_count} lines]")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+c":
            event.stop()
            event.prevent_default()
            if self.app._copy_selected_text():
                return
            if self.app._is_agent_running():
                self.app._cancel_agent_turn()
                return
            if self.text:
                self.clear_after_submit()
            else:
                self.app.exit()
            return
        if self.app._has_active_picker():
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                self.app._confirm_picker_selection()
                return
            if event.key == "up":
                event.stop()
                event.prevent_default()
                self.app._move_picker_selection(-1)
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                self.app._move_picker_selection(1)
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.app._clear_command_panel()
                return
        if self.app._has_command_completion():
            if event.key == "up":
                event.stop()
                event.prevent_default()
                self.app._move_command_completion(-1)
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                self.app._move_command_completion(1)
                return
            if event.key in {"tab", "enter"}:
                event.stop()
                event.prevent_default()
                self.app._accept_command_completion()
                return
        if event.key == "tab" and self.app._accept_command_completion():
            event.stop()
            event.prevent_default()
            return
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.app._recall_input_history(-1)
            return
        if event.key == "down":
            event.stop()
            event.prevent_default()
            self.app._recall_input_history(1)
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.submitted_value, self))
            return
        if event.key in {"shift+enter", "alt+enter"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


def matching_commands(value: str) -> list[CommandSpec]:
    if not value.startswith("/") or any(char.isspace() for char in value):
        return []
    return [spec for spec in COMMAND_SPECS if spec.command.startswith(value)]


def command_hint(value: str) -> CommandSpec | None:
    if not value.startswith("/"):
        return None
    command = value.split(maxsplit=1)[0]
    return next((spec for spec in COMMAND_SPECS if spec.command == command), None)
