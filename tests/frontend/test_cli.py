import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
from rai.frontend.cli import (
    CliAgentEvent,
    CliTurn,
    MemoryCliSession,
    parse_cli_input,
    shutdown_tool_connectors,
)
from rai.frontend.tui import RAI_AGENT_THEME, ChatTextArea, MemoryTuiApp
from textual import events
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Markdown, RichLog


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
    assert [event.message.content for event in events if event.kind == "message"] == [
        "event answer"
    ]


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
            assert "ChatTextArea {\n        height: auto;" in app.CSS
            assert (
                "ChatTextArea {\n        height: auto;\n        min-height: 1;\n        max-height: 6;\n        border: tall #34424b;"
                in app.CSS
            )
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
            assert "border: round #2f3a42;" in app.CSS
            assert "border-right: solid #334048;" in app.CSS

    asyncio.run(run_test())


def test_memory_tui_app_keeps_ctrl_c_for_input_clear_not_quit():
    bindings = {binding.key: binding.action for binding in MemoryTuiApp.BINDINGS}

    assert bindings["ctrl+c"] == "quit"
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
            assert "Assistant\nanswer" in app._transcript

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
