from __future__ import annotations

from typing import Any, Iterable, Protocol

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rai.frontend.cli import CliAgentEvent


class TuiEventRenderer(Protocol):
    def refresh_status(self, status: str | None = None) -> None: ...

    def agent_status_label(self, status: str) -> str: ...

    def write_assistant(self, message: str) -> None: ...

    def write_user(self, message: str) -> None: ...

    def write_tool_call(self, name: str, args: Any) -> None: ...

    def write_tool_result(self, name: str, content: Any) -> None: ...

    def append_log(self, role: str, content: str) -> None: ...

    def start_tool_activity(self, tool_key: str, name: str, args: Any) -> None: ...

    def finish_tool_activity(
        self, tool_key: str, name: str, result: Any, succeeded: bool
    ) -> None: ...

    def start_context_summary(self) -> None: ...

    def finish_context_summary(self, succeeded: bool) -> None: ...


class TuiEventAdapter:
    """Translate agent stream events into TUI rendering operations."""

    def __init__(self, renderer: TuiEventRenderer) -> None:
        self._renderer = renderer
        self._realtime_tool_result_ids: set[str] = set()

    def reset_turn(self) -> None:
        self._realtime_tool_result_ids.clear()

    def handle_event(self, event: CliAgentEvent) -> bool:
        if event.kind == "status":
            self._handle_status_event(event)
            self._renderer.refresh_status(
                self._renderer.agent_status_label(event.status)
            )
            return False
        if event.kind == "message" and event.message is not None:
            self.write_messages([event.message])
            return False
        return event.kind == "done"

    def write_messages(self, messages: Iterable[Any]) -> None:
        for message in messages:
            if isinstance(message, AIMessage):
                if message.content:
                    self._renderer.write_assistant(str(message.content))
                for tool_call in message.tool_calls or []:
                    name = tool_call.get("name", "tool")
                    self._renderer.refresh_status(f"tool: {name}")
                    self._renderer.write_tool_call(name, tool_call.get("args", {}))
            elif isinstance(message, ToolMessage):
                self._renderer.refresh_status(f"tool result: {message.name or 'tool'}")
                if self._is_realtime_tool_result(message):
                    self._renderer.append_log(
                        "tool result", f"{message.name or 'tool'}\n{message.content}"
                    )
                    continue
                self._renderer.write_tool_result(
                    message.name or "tool", message.content
                )
            elif isinstance(message, HumanMessage):
                self._renderer.write_user(str(message.content))

    def _handle_status_event(self, event: CliAgentEvent) -> None:
        status = event.status
        if status == "context: summarizing":
            self._renderer.start_context_summary()
            return
        if status == "context: summarized":
            self._renderer.finish_context_summary(True)
            return
        if status == "context: summary error":
            self._renderer.finish_context_summary(False)
            return
        if not status.startswith("tool: "):
            return
        tool_name = status.removeprefix("tool: ").rsplit(" ", maxsplit=1)[0]
        event_data = event.data if isinstance(event.data, dict) else {}
        tool_key = str(event_data.get("run_id") or tool_name)
        payload = (
            event_data.get("data") if isinstance(event_data.get("data"), dict) else {}
        )
        if status.endswith("starting"):
            self._renderer.start_tool_activity(
                tool_key, tool_name, payload.get("input")
            )
        elif status.endswith("done"):
            output = payload.get("output")
            if isinstance(output, ToolMessage) and output.tool_call_id:
                self._realtime_tool_result_ids.add(str(output.tool_call_id))
            self._renderer.finish_tool_activity(tool_key, tool_name, output, True)
        elif status.endswith("error"):
            self._renderer.finish_tool_activity(
                tool_key, tool_name, payload.get("error"), False
            )

    def _is_realtime_tool_result(self, message: ToolMessage) -> bool:
        return bool(
            message.tool_call_id
            and str(message.tool_call_id) in self._realtime_tool_result_ids
        )
