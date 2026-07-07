from types import SimpleNamespace

from langchain_core.messages import AIMessage
from PIL import Image

from rai.frontend.cli import CliTurn, MemoryCliSession, parse_cli_input


class _FakeCheckpointer:
    def list(self, _config, limit=200):
        return [
            SimpleNamespace(config={"configurable": {"thread_id": "session-a"}}),
            SimpleNamespace(config={"configurable": {"thread_id": "session-b"}}),
        ][:limit]


class _FakeMemoryManager:
    @property
    def checkpointer(self):
        return _FakeCheckpointer()


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


def test_parse_cli_input_handles_commands_and_image_turn():
    assert parse_cli_input("/exit").should_exit is True
    assert parse_cli_input("/new").message == "new"
    assert parse_cli_input("/sessions").message == "sessions"
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
        thread_id="thread-1",
    )

    new_messages = session.invoke(CliTurn(text="hello", images=[str(image_path)]))

    assert len(new_messages) == 1
    assert new_messages[0].content == "answer"
    call = graph.calls[0]
    assert call["input"]["messages"][0].content == "hello"
    assert call["config"]["configurable"]["thread_id"] == "thread-1"
    assert call["context"].user_id == "operator"
    assert call["context"].namespace == "inspection"
    assert len(call["context"].transient_images) == 1
    assert call["context"].transient_images[0] != str(image_path)
    assert session.summary == "summary"


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
