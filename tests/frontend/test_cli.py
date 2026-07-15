import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from PIL import Image
from rai.frontend.cli import (
    CliAgentEvent,
    CliTurn,
    MemoryCliSession,
    parse_cli_input,
    shutdown_tool_connectors,
)
from rai.frontend.tui import RAI_AGENT_THEME, ChatTextArea, MemoryTuiApp
from rai.memory.agent_factory import create_memory_agent_with_tools
from rai.memory.config import MemoryConfig
from rai.memory.manager import MemoryManager
from textual import events
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.selection import SELECT_ALL
from textual.widgets import Markdown, RichLog, Static


async def _wait_for_transcript(
    app: MemoryTuiApp,
    text: str,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in "\n".join(app._transcript):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for transcript text: {text}")


class _FakeCheckpointer:
    def __init__(self):
        self.deleted_threads = []

    def list(self, _config, limit=200):
        return [
            SimpleNamespace(config={"configurable": {"thread_id": "session-a"}}),
            SimpleNamespace(config={"configurable": {"thread_id": "session-b"}}),
        ][:limit]

    def delete_thread(self, thread_id):
        self.deleted_threads.append(thread_id)


class _FakeMemoryManager:
    def __init__(self):
        self._checkpointer = _FakeCheckpointer()
        self._store = _FakeStore()
        self._config = SimpleNamespace(backend="sqlite")

    @property
    def checkpointer(self):
        return self._checkpointer

    @property
    def store(self):
        return self._store


class _FakeStore:
    def __init__(self):
        self.deleted = []
        self.puts = []

    def search(self, namespace, query="", limit=200):
        schema = namespace[-1]
        if schema == "profiles":
            return [
                SimpleNamespace(key="operator", value={"user_id": "operator"}),
            ]
        if schema == "metadata":
            return [
                SimpleNamespace(key=key, value=value)
                for namespace, key, value in self.puts
                if namespace[-1] == "metadata"
            ][:limit]
        if schema == "facts":
            return [
                SimpleNamespace(key="fact-1", value={"text": "inspect point1 first"})
            ]
        if schema == "spatial":
            return [
                SimpleNamespace(
                    key="point1",
                    value={"location": "point1", "pose": {"x": 1.0, "y": 2.0}},
                )
            ]
        return []

    def list_namespaces(self, prefix=None, limit=200):
        return [
            ("inspection", "operator", "facts"),
            ("inspection", "inspector_01", "spatial"),
        ][:limit]

    def delete(self, namespace, key):
        self.deleted.append((namespace, key))

    def put(self, namespace, key, value):
        self.puts.append((namespace, key, value))

    def get(self, namespace, key):
        for stored_namespace, stored_key, value in reversed(self.puts):
            if stored_namespace == namespace and stored_key == key:
                return SimpleNamespace(key=key, value=value)
        return None


class _FakeGraph:
    def __init__(self):
        self.calls = []
        self.messages = [AIMessage(content="welcome")]

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self.messages, "summary": ""})

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        self.messages = [*self.messages, AIMessage(content="answer")]
        return {"messages": self.messages, "summary": "summary"}


class _FakeStreamingGraph(_FakeGraph):
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.messages = [
            *self.messages,
            HumanMessage(content="hello"),
            AIMessage(content="streamed answer"),
        ]
        yield {"agent": {"messages": self.messages[-2:]}}


class _FakeDelayedStreamingGraph(_FakeGraph):
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        first = AIMessage(content="first streamed step", id="first-step")
        second = AIMessage(content="final streamed step", id="final-step")
        self.messages = [*self.messages, first]
        yield {"agent": {"messages": [first]}}
        time.sleep(0.3)
        self.messages = [*self.messages, second]
        yield {"agent": {"messages": [second]}}


class _FakeAsyncEventsGraph(_FakeGraph):
    async def astream_events(self, **kwargs):
        self.calls.append(kwargs)
        yield {"event": "on_chain_start", "name": "enrich_prompt", "data": {}}
        yield {"event": "on_chat_model_start", "name": "ChatOpenAI", "data": {}}
        yield {"event": "on_chat_model_stream", "name": "ChatOpenAI", "data": {}}
        message = AIMessage(content="event answer", id="ai-event")
        self.messages = [*self.messages, message]
        yield {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "data": {"output": message},
        }
        yield {"event": "on_chain_end", "name": "react", "data": {}}


class _FakeUnsupportedAsyncEventsGraph(_FakeStreamingGraph):
    async def astream_events(self, **kwargs):
        self.calls.append(kwargs)
        raise NotImplementedError("The SqliteSaver doesn't support async methods")
        yield  # pragma: no cover


class _FakeToolStreamingGraph(_FakeGraph):
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect_area",
                        "args": {"target": "point1", "mode": "thermal"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(
                content='{"status": "ok", "temperature": 36.5}',
                name="inspect_area",
                tool_call_id="call-1",
            ),
            AIMessage(content="inspection complete"),
        ]
        self.messages = [*self.messages, *messages]
        yield {"agent": {"messages": messages}}


class _FakeToolEventsGraph(_FakeGraph):
    async def astream_events(self, **kwargs):
        self.calls.append(kwargs)
        tool_message = ToolMessage(
            content='{"status": "ok", "temperature": 36.5}',
            name="inspect_area",
            tool_call_id="call-1",
        )
        self.messages = [*self.messages, tool_message, AIMessage(content="done")]
        yield {
            "event": "on_tool_start",
            "name": "inspect_area",
            "run_id": "tool-run-1",
            "data": {"input": {"target": "point1"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "inspect_area",
            "run_id": "tool-run-1",
            "data": {"output": tool_message},
        }
        yield {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "data": {"output": AIMessage(content="done", id="tool-events-ai")},
        }


class _ToolCallingChatModel(BaseChatModel):
    i: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "tool-calling-test"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ):
        if self.i == 0:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delayed_tool",
                        "args": {"query": "point1"},
                        "id": "call-1",
                    }
                ],
            )
        else:
            message = AIMessage(content="final answer")
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ):
        return self._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


@tool
def delayed_tool(query: str) -> str:
    """Delay and return a deterministic tool result."""
    time.sleep(0.3)
    return f"done:{query}"


def test_parse_cli_input_handles_commands_and_image_turn():
    assert parse_cli_input("/exit").should_exit is True
    assert parse_cli_input("/help").message == "help"
    assert parse_cli_input("/status").message == "status"
    assert parse_cli_input("/tools").message == "tools"
    assert parse_cli_input("/clear").message == "clear"
    assert parse_cli_input("/new").message == "new"
    assert parse_cli_input("/sessions").message == "sessions"
    assert (
        parse_cli_input("/delete-session session-a").message
        == "delete-session:session-a"
    )
    assert (
        parse_cli_input("/delete-session").message
        == "usage: /delete-session <thread_id>"
    )
    assert (
        parse_cli_input("/delete-memory facts fact-1").message
        == "delete-memory:facts:fact-1"
    )
    assert (
        parse_cli_input("/delete-memory locations point1").message
        == "delete-memory:locations:point1"
    )
    assert (
        parse_cli_input("/delete-memory bad key").message
        == "usage: /delete-memory <facts|locations> <key>"
    )
    assert parse_cli_input("/resume session-a").message == "resume:session-a"
    assert parse_cli_input("/session session-a").message == "resume:session-a"
    assert (
        parse_cli_input("/resume session-a --quiet").message == "resume:session-a:quiet"
    )
    assert (
        parse_cli_input("/session session-a --quiet").message
        == "resume:session-a:quiet"
    )
    assert (
        parse_cli_input("/resume session-a extra").message
        == "usage: /resume <thread_id> [--quiet]"
    )
    assert parse_cli_input("/resume").message == "usage: /resume <thread_id>"
    assert parse_cli_input("/session").message == "usage: /session <thread_id>"
    assert parse_cli_input("/users").message == "users"
    assert parse_cli_input("/user").message == "users"
    assert parse_cli_input("/memory").message == "memory"
    assert parse_cli_input("/memory facts").message == "memory:facts"
    assert parse_cli_input("/memory locations").message == "memory:locations"
    assert parse_cli_input("/memory spatial").message == "memory:spatial"
    assert parse_cli_input("/memory bad").message == "usage: /memory [facts|locations]"
    assert parse_cli_input("/delete-user operator").message == "delete-user:operator"
    assert parse_cli_input("/delete-user").message == "usage: /delete-user <user_id>"
    assert (
        parse_cli_input("/export-session /tmp/session.json").message
        == "export-session:/tmp/session.json"
    )
    assert parse_cli_input("/export-session").message == "usage: /export-session <path>"
    assert parse_cli_input("/user operator").message == "user:operator"

    image = parse_cli_input("/image /tmp/a.jpg inspect this")
    assert image.handled is False
    assert image.turn is not None
    assert image.turn.text == "inspect this"
    assert image.turn.images == ["/tmp/a.jpg"]


def test_memory_cli_session_invokes_graph_with_context_and_thread_id(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
    graph = _FakeGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    new_messages = session.invoke(CliTurn(text="hello", images=[str(image_path)]))

    assert len(new_messages) == 1
    assert new_messages[0].content == "answer"
    call = graph.calls[0]
    assert call["input"]["messages"][0].content == "hello"
    assert call["config"]["configurable"]["thread_id"] == "session-a"
    assert call["context"].user_id == "operator"
    assert call["context"].namespace == "inspection"
    assert len(call["context"].transient_images) == 1
    assert call["context"].transient_images[0] != str(image_path)
    assert session.summary == "summary"
    summaries = session.list_session_summaries()
    summaries_by_id = {summary.thread_id: summary for summary in summaries}
    assert summaries_by_id["session-a"].first_user_message == "hello"
    assert "session-b" not in summaries_by_id


def test_memory_cli_session_streams_graph_events():
    graph = _FakeStreamingGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    events = list(session.stream_events(CliTurn(text="hello")))

    assert any(
        isinstance(event, CliAgentEvent) and event.kind == "status" for event in events
    )
    assert [event.message.content for event in events if event.kind == "message"] == [
        "streamed answer"
    ]
    assert events[-1].kind == "done"


def test_memory_cli_session_streaming_filters_echoed_human_messages():
    graph = _FakeStreamingGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    messages = [
        event.message
        for event in session.stream_events(CliTurn(text="hello"))
        if event.kind == "message"
    ]

    assert all(not isinstance(message, HumanMessage) for message in messages)
    assert [message.content for message in messages] == ["streamed answer"]


def test_memory_cli_session_streams_langchain_events():
    graph = _FakeAsyncEventsGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    events = list(session.stream_events(CliTurn(text="hello")))
    statuses = [event.status for event in events if event.kind == "status"]

    assert "step: enrich_prompt start" in statuses
    assert "model: connecting (ChatOpenAI)" in statuses
    assert "model: receiving (ChatOpenAI)" in statuses
    assert "model: complete (ChatOpenAI)" in statuses
    assert all(
        event.data
        for event in events
        if event.kind == "status" and event.status != "agent: starting"
    )
    assert [event.message.content for event in events if event.kind == "message"] == [
        "event answer"
    ]


def test_memory_cli_session_streams_real_sqlite_tool_start_before_tool_end(tmp_path):
    memory_mgr = MemoryManager(
        config=MemoryConfig(
            enabled=True,
            backend="sqlite",
            short_term_path=str(Path(tmp_path) / "checkpoints.db"),
            long_term_path=str(Path(tmp_path) / "store.db"),
            namespace="inspection",
        )
    )
    memory_mgr.start()
    try:
        graph = create_memory_agent_with_tools(
            memory_mgr=memory_mgr,
            llm=_ToolCallingChatModel(),
            base_system_prompt_builder=lambda _context: "You are a test agent.",
            namespace="inspection",
            user_id="operator",
            base_tools=[delayed_tool],
        )
        session = MemoryCliSession(
            memory_mgr=memory_mgr,
            graph=graph,
            namespace="inspection",
            user_id="operator",
            thread_id="session-a",
        )

        start_time = time.monotonic()
        tool_status_times: list[tuple[str, float]] = []
        messages = []
        for event in session.stream_events(CliTurn(text="run delayed tool")):
            if event.kind == "status" and event.status.startswith("tool:"):
                tool_status_times.append((event.status, time.monotonic() - start_time))
            if event.kind == "message":
                messages.append(event.message)

        starts = [
            elapsed
            for status, elapsed in tool_status_times
            if status == "tool: delayed_tool starting"
        ]
        ends = [
            elapsed
            for status, elapsed in tool_status_times
            if status == "tool: delayed_tool done"
        ]
        assert starts and starts[0] < 0.2
        assert ends and ends[0] >= 0.3
        assert starts[0] < ends[0]
        assert [
            message.content for message in messages if isinstance(message, ToolMessage)
        ] == ["done:point1"]
    finally:
        memory_mgr.stop()


def test_memory_cli_session_falls_back_when_checkpointer_lacks_async_methods():
    graph = _FakeUnsupportedAsyncEventsGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    events = list(session.stream_events(CliTurn(text="hello")))

    assert [event.message.content for event in events if event.kind == "message"] == [
        "streamed answer"
    ]
    assert events[-1].kind == "done"


def test_memory_cli_session_user_switch_rebuilds_graph():
    graphs = []

    def factory(user_id):
        graph = _FakeGraph()
        graph.user_id = user_id
        graphs.append(graph)
        return graph

    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="default",
        graph_factory=factory,
    )

    session.set_user("operator")

    assert session.user_id == "operator"
    assert session.graph is graphs[0]
    assert session.graph.user_id == "operator"


def test_memory_cli_session_lists_long_term_memory_by_kind():
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
    )

    all_items = session.list_long_term_memory()
    facts = session.list_long_term_memory("facts")
    locations = session.list_long_term_memory("locations")

    assert [item[0] for item in all_items] == ["facts", "spatial"]
    assert [item[2] for item in facts] == ["fact-1"]
    assert [item[2] for item in locations] == ["point1"]


def test_memory_cli_session_lists_users():
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
    )

    assert session.list_users() == ["default", "inspector_01", "operator"]


def test_memory_cli_session_reports_status_and_tools():
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
        thread_id="thread-1",
        tools=[
            SimpleNamespace(name="tool_a", description="Tool A"),
            SimpleNamespace(name="tool_b", description="Tool B"),
        ],
    )

    assert session.status()["thread"] == "thread-1"
    assert session.status()["memory_backend"] == "sqlite"
    assert session.tool_summaries() == [("tool_a", "Tool A"), ("tool_b", "Tool B")]


def test_memory_cli_session_deletes_session_and_memory():
    memory_mgr = _FakeMemoryManager()
    session = MemoryCliSession(
        memory_mgr=memory_mgr,
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
        thread_id="session-a",
    )

    session_message = session.delete_session("session-b")
    memory_message = session.delete_long_term_memory("locations", "point1")

    assert session_message == "Deleted session session-b."
    assert memory_mgr.checkpointer.deleted_threads == ["session-b"]
    assert memory_message == "Deleted spatial memory key=point1 for user=operator."
    assert memory_mgr.store.deleted == [
        (("inspection", "__sessions__", "metadata"), "session-b"),
        (("inspection", "operator", "spatial"), "point1"),
    ]


def test_memory_cli_session_deletes_user_and_switches_current_user_to_default():
    memory_mgr = _FakeMemoryManager()
    session = MemoryCliSession(
        memory_mgr=memory_mgr,
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
    )

    message = session.delete_user("operator")

    assert message.startswith("Deleted user operator")
    assert "switched to default" in message
    assert session.user_id == "default"
    assert memory_mgr.store.puts[0][0] == ("inspection", "__users__", "profiles")
    assert memory_mgr.store.puts[0][1] == "operator"
    assert memory_mgr.store.puts[0][2]["deleted"] is True


def test_memory_cli_session_refuses_to_delete_default_user():
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="default",
    )

    assert session.delete_user("default") == "Refusing to delete default user."


def test_memory_cli_session_exports_current_session(tmp_path):
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
        thread_id="thread-1",
    )
    export_path = tmp_path / "session.json"

    message = session.export_session(export_path)

    assert message == f"Exported session thread-1 to {export_path}."
    exported = export_path.read_text()
    assert '"thread_id": "thread-1"' in exported
    assert '"user_id": "operator"' in exported
    assert '"messages"' in exported


def test_memory_cli_session_resumes_thread_and_loads_messages():
    graph = _FakeGraph()
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=graph,
        namespace="inspection",
        user_id="operator",
        thread_id="old-thread",
    )

    messages = session.resume_session("session-a")

    assert session.thread_id == "session-a"
    assert messages == graph.messages


def test_shutdown_tool_connectors_shuts_each_unique_connector_once():
    class Connector:
        def __init__(self):
            self.shutdown_count = 0

        def shutdown(self):
            self.shutdown_count += 1

    connector = Connector()
    tools = [
        SimpleNamespace(connector=connector),
        SimpleNamespace(connector=connector),
        SimpleNamespace(),
    ]

    shutdown_tool_connectors(tools)

    assert connector.shutdown_count == 1


def test_memory_tui_app_can_be_constructed_with_session():
    session = MemoryCliSession(
        memory_mgr=_FakeMemoryManager(),
        graph=_FakeGraph(),
        namespace="inspection",
        user_id="operator",
    )

    app = MemoryTuiApp(session)

    assert app.session is session


def test_memory_tui_app_has_inline_activity_layout():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            try:
                app.query_one("#activity")
            except NoMatches:
                pass
            else:
                raise AssertionError("TUI should not render a separate activity log")
            conversation = app.query_one("#conversation")
            assert isinstance(conversation, VerticalScroll)
            assert not isinstance(conversation, RichLog)
            child_ids = [getattr(child, "id", None) for child in app.screen.children]
            assert child_ids.index("input") < child_ids.index("agent_status")

    asyncio.run(run_test())


def test_memory_tui_app_uses_neutral_agent_theme():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            assert app.theme == RAI_AGENT_THEME.name
            assert RAI_AGENT_THEME.background == "#1e1e1e"
            assert "ChatTextArea {\n        height: auto;" in app.CSS
            assert (
                "ChatTextArea {\n        height: auto;\n        min-height: 1;\n        max-height: 6;\n        border: tall #34424b;"
                not in app.CSS
            )
            assert "border: tall #3a3d41;" in app.CSS
            assert "background: #1e1e1e;" in app.CSS
            assert "background: #252526;" in app.CSS
            assert "ChatTextArea:focus" in app.CSS
            chat_text_area_css = app.CSS.split("ChatTextArea {", maxsplit=1)[1].split(
                "}", maxsplit=1
            )[0]
            assert "$accent" not in chat_text_area_css
            assert "$warning" not in chat_text_area_css
            assert "$error" not in chat_text_area_css
            message_css = app.CSS.split(".message {", maxsplit=1)[1]
            assert "#11171b" not in message_css
            assert "#131a1f" not in message_css
            assert "#0d1114" not in app.CSS
            assert "border: round #3a3d41;" in app.CSS
            assert "border-right: solid #3a3d41;" in app.CSS

    asyncio.run(run_test())


def test_memory_tui_app_uses_scrollable_command_panel_for_memory():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        memory_items = [
            (
                "facts",
                ("inspection", "operator", "facts"),
                f"fact-{index}",
                {"text": "x" * 60},
            )
            for index in range(30)
        ]

        async with app.run_test():
            panel = app.query_one("#command_panel", VerticalScroll)
            content = app.query_one("#command_panel_content", Static)
            assert panel.has_class("hidden")
            app._show_memory(memory_items)
            assert not panel.has_class("hidden")
            assert content.render() is not None
            assert "max-height: 45vh;" in app.CSS
            assert "overflow-y: auto;" in app.CSS

    asyncio.run(run_test())


def test_memory_tui_app_keeps_ctrl_c_for_input_clear_not_quit():
    bindings = {binding.key: binding.action for binding in MemoryTuiApp.BINDINGS}

    assert "ctrl+c" not in bindings
    assert bindings["ctrl+q"] == "quit"


def test_memory_tui_app_ctrl_c_clears_input_text():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            text_area = app.query_one("#input", ChatTextArea)
            text_area.load_text("draft message")
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert text_area.text == ""
            assert app.is_running
            assert app.query_one("#command_panel").has_class("hidden")

    asyncio.run(run_test())


def test_memory_tui_app_ctrl_c_exits_when_input_is_empty():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        exited = False

        def mark_exit(*args, **kwargs):
            nonlocal exited
            exited = True

        async with app.run_test() as pilot:
            app.exit = mark_exit
            text_area = app.query_one("#input", ChatTextArea)
            text_area.load_text("")
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert exited

    asyncio.run(run_test())


def test_memory_tui_app_ctrl_c_copies_selected_conversation_text():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        copied: list[str] = []
        app.copy_to_clipboard = copied.append

        async with app.run_test() as pilot:
            text_area = app.query_one("#input", ChatTextArea)
            app._write_user("selected conversation text")
            message = app.query(".message.user").last()
            text_area.load_text("draft message")
            app.screen.selections = {message: SELECT_ALL}
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert copied == ["selected conversation text"]
            assert app.screen.get_selected_text() is None
            assert text_area.text == "draft message"
            assert app.is_running
            assert not app.query_one("#command_panel").has_class("hidden")
            assert app._last_copy_panel_title == "Selection"
            assert app._last_copy_panel_text == "selected conversation text"

    asyncio.run(run_test())


def test_memory_tui_app_copy_fallback_panel_and_selection_style():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        copied: list[str] = []
        app.copy_to_clipboard = copied.append

        async with app.run_test():
            app._copy_text_with_fallback("manual copy text", title="Copy Test")
            assert copied == ["manual copy text"]
            assert "Terminal clipboard copy requested" in app._last_copy_panel_status
            assert app._last_copy_panel_text == "manual copy text"
            assert "Screen > .screen--selection" in app.CSS
            assert "background: #0e639c;" in app.CSS
            assert "ChatTextArea .text-area--selection" in app.CSS

    asyncio.run(run_test())


def test_memory_tui_app_copy_status_reports_tui_local_only():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            app._has_terminal_clipboard = lambda: False
            app._supports_pyperclip = False
            app._copy_text_with_fallback("local only text", title="Copy Test")
            assert "TUI local clipboard only" in app._last_copy_panel_status
            assert app._last_copy_panel_text == "local only text"

    asyncio.run(run_test())


def test_memory_tui_app_copy_status_reports_pyperclip_system_clipboard(monkeypatch):
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        pyperclip_calls = []
        fake_pyperclip = SimpleNamespace(copy=pyperclip_calls.append)
        monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)

        async with app.run_test():
            app._copy_text_with_fallback("system text", title="Copy Test")
            assert pyperclip_calls == ["system text"]
            assert "Copied to system clipboard via pyperclip" in (
                app._last_copy_panel_status
            )
            assert app._last_copy_panel_text == "system text"

    asyncio.run(run_test())


def test_memory_tui_app_copy_status_falls_back_when_pyperclip_fails(monkeypatch):
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        def fail_copy(_text):
            raise RuntimeError("clipboard unavailable")

        monkeypatch.setitem(sys.modules, "pyperclip", SimpleNamespace(copy=fail_copy))

        async with app.run_test():
            app._copy_text_with_fallback("fallback text", title="Copy Test")
            assert "Terminal clipboard copy requested" in app._last_copy_panel_status
            assert app._last_copy_panel_text == "fallback text"

    asyncio.run(run_test())


def test_memory_tui_app_input_history_uses_up_down_keys():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            text_area = app.query_one("#input", ChatTextArea)
            await pilot.press("f", "i", "r", "s", "t", "enter")
            await pilot.pause()
            await pilot.press("s", "e", "c", "o", "n", "d", "enter")
            await pilot.pause()

            text_area.load_text("draft")
            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "second"
            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "first"
            await pilot.press("down")
            await pilot.pause()
            assert text_area.text == "second"
            await pilot.press("down")
            await pilot.pause()
            assert text_area.text == "draft"

    asyncio.run(run_test())


def test_memory_tui_app_ctrl_c_interrupts_running_turn():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeDelayedStreamingGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("s", "t", "r", "e", "a", "m", "enter")
            await _wait_for_transcript(app, "first streamed step")
            await pilot.press("ctrl+c")
            await _wait_for_transcript(app, "• Interrupted after")
            await asyncio.sleep(0.5)
            transcript = "\n".join(app._transcript)
            assert "first streamed step" in transcript
            assert "• Interrupted after" in transcript
            assert "final streamed step" not in transcript
            assert "• Worked for" not in transcript
            assert app.is_running

    asyncio.run(run_test())


def test_memory_tui_app_renders_assistant_as_markdown_message():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            app._write_assistant("answer")
            assistant_message = app.query(".message.assistant").last()
            assert isinstance(assistant_message, Markdown)
            assert assistant_message.border_title == " Assistant "
            assert "Assistant\nanswer" in app._transcript

    asyncio.run(run_test())


def test_memory_tui_app_renders_user_role_title():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            app._write_user("hello")
            user_message = app.query(".message.user").last()
            assert user_message.border_title == " User "
            assert "User\nhello" in app._transcript

    asyncio.run(run_test())


def test_memory_tui_app_accepts_multiline_paste_and_submits_full_text():
    async def run_test():
        graph = _FakeGraph()
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=graph,
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            text_area = app.query_one("#input", ChatTextArea)
            await text_area._on_paste(events.Paste("first line\nsecond line"))
            assert text_area.text == "first line\nsecond line"
            await pilot.press("enter")
            await pilot.pause()
            assert graph.calls[0]["input"]["messages"][0].content == (
                "first line\nsecond line"
            )

    asyncio.run(run_test())


def test_memory_tui_app_copy_transcript_copies_plain_text():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        copied: list[str] = []
        app.copy_to_clipboard = copied.append

        async with app.run_test() as pilot:
            await pilot.press("h", "e", "l", "l", "o", "enter")
            await pilot.pause()
            app.action_copy_transcript()
            assert copied
            assert "User\nhello" in copied[-1]
            assert "Assistant\nanswer" in copied[-1]
            assert not app.query_one("#command_panel").has_class("hidden")
            assert app._last_copy_panel_title == "Transcript"

    asyncio.run(run_test())


def test_memory_tui_app_copy_last_uses_copy_panel():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)
        copied: list[str] = []
        app.copy_to_clipboard = copied.append

        async with app.run_test():
            app._write_assistant("last assistant answer")
            app._copy_last_assistant_message()
            assert copied == ["last assistant answer"]
            assert app._last_copy_panel_title == "Assistant"
            assert app._last_copy_panel_text == "last assistant answer"

    asyncio.run(run_test())


def test_memory_tui_app_renders_codex_style_agent_timeline():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeAsyncEventsGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("h", "e", "l", "l", "o", "enter")
            for _ in range(5):
                await pilot.pause()
            transcript = "\n".join(app._transcript)
            assert "• Worked for" in transcript
            assert "enrich_prompt" not in transcript
            assert "model: receiving" not in transcript

    asyncio.run(run_test())


def test_memory_tui_app_renders_stream_chunks_before_turn_done():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeDelayedStreamingGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("s", "t", "r", "e", "a", "m", "enter")
            await _wait_for_transcript(app, "first streamed step")
            transcript = "\n".join(app._transcript)
            assert "first streamed step" in transcript
            assert "final streamed step" not in transcript
            assert "• Worked for" not in transcript
            await _wait_for_transcript(app, "final streamed step")
            await _wait_for_transcript(app, "• Worked for")
            transcript = "\n".join(app._transcript)
            assert "final streamed step" in transcript
            assert "• Worked for" in transcript

    asyncio.run(run_test())


def test_memory_tui_app_updates_working_timeline_in_place():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            app._start_turn_timeline()
            assert "• Working (0s)" in app._transcript
            assert "Working" not in app._status_text()
            app._write_assistant("answer while working")
            assert app._transcript[-1] == "• Working (0s)"
            app._turn_started_at = app._turn_started_at - 65
            app._refresh_working_status()
            assert app._transcript[-1] == "• Working (1m 05s)"
            assert "• Working (0s)" not in app._transcript
            assert "Working" not in app._status_text()
            app._finish_turn_timeline(True)
            assert app._transcript[-1] == "• Worked for 1m 05s"
            assert all("• Working" not in item for item in app._transcript)
            conversation = app.query_one("#conversation", VerticalScroll)
            assert conversation.is_vertical_scroll_end

    asyncio.run(run_test())


def test_memory_tui_app_renders_final_tool_messages_without_running_state():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeToolStreamingGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("i", "n", "s", "p", "e", "c", "t", "enter")
            for _ in range(5):
                await pilot.pause()
            transcript = "\n".join(app._transcript)
            assert "• Tool call inspect_area" in transcript
            assert '└ { "target": "point1", "mode": "thermal" }' in transcript
            assert "• Tool result inspect_area" in transcript
            assert "• Running inspect_area" not in transcript
            assert "• Ran inspect_area" not in transcript
            assert "temperature" in transcript

    asyncio.run(run_test())


def test_memory_tui_app_labels_visual_tool_result_as_summary():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test():
            app._write_tool_result(
                "analyze_artifact_image",
                "Detailed visual inspection report.",
            )
            transcript = "\n".join(app._transcript)
            assert "• Vision result summary analyze_artifact_image" in transcript
            assert "Detailed visual inspection report." in transcript

    asyncio.run(run_test())


def test_memory_tui_app_renders_realtime_tool_events_as_running_then_ran():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeToolEventsGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("i", "n", "s", "p", "e", "c", "t", "enter")
            for _ in range(5):
                await pilot.pause()
            transcript = "\n".join(app._transcript)
            assert "• Ran inspect_area" in transcript
            assert "• Running inspect_area" not in transcript
            assert "• Tool result inspect_area" not in transcript
            assert "temperature" in transcript

    asyncio.run(run_test())


def test_memory_tui_app_handles_status_command():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session)

        async with app.run_test() as pilot:
            await pilot.press("/", "s", "t", "a", "t", "u", "s", "enter")
            assert app.session.user_id == "operator"

    asyncio.run(run_test())


def test_memory_tui_app_uses_command_panel_for_help():
    async def run_test():
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press("/", "h", "e", "l", "p", "enter")
            assert not app.query_one("#command_panel").has_class("hidden")

    asyncio.run(run_test())


def test_memory_tui_app_resume_picker_resumes_selected_session():
    async def run_test():
        memory_mgr = _FakeMemoryManager()
        memory_mgr.store.put(
            ("inspection", "__sessions__", "metadata"),
            "session-a",
            {
                "thread_id": "session-a",
                "created_at": 1.0,
                "updated_at": 1.0,
                "first_user_message": "first",
            },
        )
        graph = _FakeGraph()
        graph.messages = [
            message
            for index in range(60)
            for message in (
                HumanMessage(content=f"user {index}\n" + "text " * 20),
                AIMessage(content=f"assistant {index}\n" + "markdown text\n" * 4),
            )
        ]
        session = MemoryCliSession(
            memory_mgr=memory_mgr,
            graph=graph,
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
            assert app._session_picker
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause()
            conversation = app.query_one("#conversation", VerticalScroll)
            assert app.session.thread_id == "session-a"
            assert conversation.scroll_y == conversation.max_scroll_y
            assert conversation.is_vertical_scroll_end

    asyncio.run(run_test())


def test_memory_tui_app_delete_session_picker_deletes_selected_session():
    async def run_test():
        memory_mgr = _FakeMemoryManager()
        memory_mgr.store.put(
            ("inspection", "__sessions__", "metadata"),
            "session-a",
            {
                "thread_id": "session-a",
                "created_at": 1.0,
                "updated_at": 1.0,
                "first_user_message": "first",
            },
        )
        memory_mgr.store.put(
            ("inspection", "__sessions__", "metadata"),
            "session-b",
            {
                "thread_id": "session-b",
                "created_at": 2.0,
                "updated_at": 2.0,
                "first_user_message": "second",
            },
        )
        session = MemoryCliSession(
            memory_mgr=memory_mgr,
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press(
                "/",
                "d",
                "e",
                "l",
                "e",
                "t",
                "e",
                "-",
                "s",
                "e",
                "s",
                "s",
                "i",
                "o",
                "n",
                "enter",
            )
            assert app._picker_mode == "delete-session"
            assert app._session_picker
            assert (
                app._session_picker[app._session_picker_index].thread_id == "session-b"
            )
            await pilot.press("down", "enter")
            assert memory_mgr.checkpointer.deleted_threads == ["session-a"]
            assert (
                ("inspection", "__sessions__", "metadata"),
                "session-a",
            ) in memory_mgr.store.deleted
            assert app._picker_mode is None

    asyncio.run(run_test())


def test_memory_tui_app_delete_memory_picker_deletes_selected_fact():
    async def run_test():
        memory_mgr = _FakeMemoryManager()
        session = MemoryCliSession(
            memory_mgr=memory_mgr,
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press(
                "/",
                "d",
                "e",
                "l",
                "e",
                "t",
                "e",
                "-",
                "m",
                "e",
                "m",
                "o",
                "r",
                "y",
                " ",
                "f",
                "a",
                "c",
                "t",
                "s",
                "enter",
            )
            assert app._picker_mode == "delete-memory"
            assert app._memory_picker_kind == "facts"
            assert app._memory_picker
            await pilot.press("enter")
            assert memory_mgr.store.deleted == [
                (("inspection", "operator", "facts"), "fact-1")
            ]
            assert app._picker_mode is None

    asyncio.run(run_test())


def test_memory_tui_app_delete_memory_picker_deletes_selected_location():
    async def run_test():
        memory_mgr = _FakeMemoryManager()
        session = MemoryCliSession(
            memory_mgr=memory_mgr,
            graph=_FakeGraph(),
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press(
                "/",
                "d",
                "e",
                "l",
                "e",
                "t",
                "e",
                "-",
                "m",
                "e",
                "m",
                "o",
                "r",
                "y",
                " ",
                "l",
                "o",
                "c",
                "a",
                "t",
                "i",
                "o",
                "n",
                "s",
                "enter",
            )
            assert app._picker_mode == "delete-memory"
            assert app._memory_picker_kind == "locations"
            assert app._memory_picker
            await pilot.press("enter")
            assert memory_mgr.store.deleted == [
                (("inspection", "operator", "spatial"), "point1")
            ]
            assert app._picker_mode is None

    asyncio.run(run_test())


def test_memory_tui_app_quiet_resume_does_not_render_history():
    async def run_test():
        graph = _FakeGraph()
        graph.messages = [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
        ]
        session = MemoryCliSession(
            memory_mgr=_FakeMemoryManager(),
            graph=graph,
            namespace="inspection",
            user_id="operator",
        )
        app = MemoryTuiApp(session, log_path=None)

        async with app.run_test() as pilot:
            await pilot.press(
                "/",
                "r",
                "e",
                "s",
                "u",
                "m",
                "e",
                " ",
                "s",
                "e",
                "s",
                "s",
                "i",
                "o",
                "n",
                "-",
                "a",
                " ",
                "-",
                "-",
                "q",
                "u",
                "i",
                "e",
                "t",
                "enter",
            )
            for _ in range(5):
                await pilot.pause()
            assert app.session.thread_id == "session-a"
            assert "Resumed session: session-a" in "\n".join(app._transcript)
            assert "old question" not in "\n".join(app._transcript)
            assert "old answer" not in "\n".join(app._transcript)

    asyncio.run(run_test())
