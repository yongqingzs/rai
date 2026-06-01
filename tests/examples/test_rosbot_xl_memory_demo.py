# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path
from typing import Any

import rai.agents.langchain.core.react_agent as react_agent
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field
from rai.agents.langchain.core.react_agent import summarize_messages
from rai.frontend.memory_streamlit import collect_tool_call_entries
from rai.memory.graph import MemoryAgentContext, create_memory_react_agent
from rai.memory.long_term import format_long_term_item, list_long_term_memory_items
from rai.memory.session import delete_session, get_latest_session_id, load_thread_state
from rai.memory.users import add_user_profile, delete_user, get_user_ids

import rai_whoami.tools.robot_docs as robot_docs
from rai_whoami import WhoamiConfig, create_robot_docs_tool, load_whoami_config


def _load_demo_module():
    path = Path(__file__).parents[2] / "examples" / "rosbot-xl-memory-demo.py"
    spec = importlib.util.spec_from_file_location("rosbot_xl_memory_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Snapshot:
    def __init__(self, values):
        self.values = values


class _Graph:
    def __init__(self, values):
        self.values = values

    def get_state(self, config):
        assert config["configurable"]["thread_id"] == "thread-1"
        return _Snapshot(self.values)


class _Checkpoint:
    def __init__(self, thread_id):
        self.config = {"configurable": {"thread_id": thread_id}}


class _Checkpointer:
    def __init__(self):
        self.deleted = []
        self.checkpoints = [_Checkpoint("latest"), _Checkpoint("older")]

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)

    def list(self, config, limit=200):
        return self.checkpoints[:limit]


class _MemoryManager:
    def __init__(self):
        self.checkpointer = _Checkpointer()
        self.store = InMemoryStore()


class _GraphMemoryManager:
    def __init__(self):
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()


class _RecordingFakeChatModel(FakeListChatModel):
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        return super()._call(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def _has_conversation_summary_message(messages: list[BaseMessage]) -> bool:
    return any(
        isinstance(message.content, str)
        and message.content.startswith("[Conversation summary:")
        for message in messages
    )


def test_load_thread_state_reads_checkpoint_values():
    messages = [AIMessage(content="restored")]
    graph = _Graph({"messages": messages, "summary": "prior summary"})

    restored_messages, restored_summary = load_thread_state(graph, "thread-1")

    assert restored_messages == messages
    assert restored_summary == "prior summary"


def test_welcome_message_is_ai_message():
    demo = _load_demo_module()

    message = demo._welcome_message()

    assert isinstance(message, AIMessage)
    assert "persistent memory" in message.content


def test_load_robot_docs_config_reads_whoami_section(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[whoami]
enabled = true
root_dir = "docs/robot"
build_vector_db = true
k = 7
"""
    )

    config = load_whoami_config(str(config_path))

    assert config == WhoamiConfig(
        enabled=True,
        root_dir="docs/robot",
        build_vector_db=True,
        k=7,
    )


def test_create_robot_docs_tool_returns_none_when_disabled():
    tool = create_robot_docs_tool(WhoamiConfig(enabled=False))

    assert tool is None


def test_create_robot_docs_tool_wraps_whoami_query_tool(monkeypatch, tmp_path):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    for filename in ("index.faiss", "index.pkl", "vdb_kwargs.json"):
        (generated_dir / filename).write_text("{}")

    monkeypatch.setattr(
        robot_docs.RobotDocsQueryTool,
        "__init__",
        lambda self, **kwargs: BaseTool.__init__(self, **kwargs),
    )

    tool = create_robot_docs_tool(
        WhoamiConfig(enabled=True, root_dir=str(tmp_path), k=3),
        embeddings_model=None,
    )

    assert tool.name == "query_robot_docs"
    assert "static whoami documentation" in tool.description
    assert tool.root_dir == str(tmp_path)
    assert tool.k == 3


def test_create_robot_docs_tool_builds_vector_db_when_configured(
    monkeypatch,
    tmp_path,
):
    calls = []

    class _Source:
        @classmethod
        def from_directory(cls, root_dir):
            calls.append(("source", root_dir))
            return "source"

    class _Builder:
        def __init__(self, root_dir, embedding=None):
            calls.append(("builder", root_dir, embedding))

        def build(self, source):
            calls.append(("build", source))

    monkeypatch.setattr(robot_docs, "EmbodimentSource", _Source)
    monkeypatch.setattr(robot_docs, "FAISSBuilder", _Builder)
    monkeypatch.setattr(robot_docs, "has_vector_db", lambda root_dir: True)
    monkeypatch.setattr(
        robot_docs.RobotDocsQueryTool,
        "__init__",
        lambda self, **kwargs: BaseTool.__init__(self, **kwargs),
    )
    create_robot_docs_tool(
        WhoamiConfig(
            enabled=True,
            root_dir=str(tmp_path),
            build_vector_db=True,
        ),
        embeddings_model=None,
    )

    assert calls == [
        ("source", tmp_path),
        ("builder", tmp_path / "generated", None),
        ("build", "source"),
    ]


def test_build_memory_agent_includes_robot_docs_tool(monkeypatch, tmp_path):
    demo = _load_demo_module()
    memory_mgr = _MemoryManager()
    captured = {}

    class _Tool(BaseTool):
        name: str = "query_robot_docs"
        description: str = "robot docs"

        def _run(self):
            return "robot docs"

    monkeypatch.setattr(demo, "get_llm_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(demo, "_load_embodiment", lambda path: "embodiment")
    monkeypatch.setattr(demo, "create_robot_docs_tool", lambda *args: _Tool())

    def _create_agent(**kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(demo, "create_memory_agent_with_tools", _create_agent)

    graph = demo.build_memory_agent(
        memory_mgr,
        tmp_path / "embodiment.json",
        robot_docs_config=WhoamiConfig(enabled=True, root_dir="whoami"),
        embeddings_model=object(),
    )

    assert graph == "graph"
    assert "query_robot_docs" in [tool.name for tool in captured["extra_tools"]]
    assert captured["extra_prompt_sections"] == [demo.ROBOT_DOCS_PROMPT_SECTION]


def test_short_term_summary_is_state_not_message():
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"user turn {i} " + "x" * 120))
        messages.append(AIMessage(content=f"assistant turn {i} " + "y" * 120))

    result = summarize_messages(
        messages,
        existing_summary="previous summary",
        llm=FakeListChatModel(responses=["compressed summary"]),
        threshold=100,
        keep_recent=4,
    )

    assert result["summary"] == "compressed summary"
    assert result["messages"] == messages[-4:]
    assert not _has_conversation_summary_message(result["messages"])


def test_memory_graph_injects_summary_without_checkpointing_summary_message(
    monkeypatch,
):
    summaries = iter([f"summary {i}" for i in range(20)])
    monkeypatch.setattr(
        react_agent,
        "get_llm_model",
        lambda *args, **kwargs: FakeListChatModel(responses=[next(summaries)]),
    )
    llm = _RecordingFakeChatModel(responses=[f"answer {i}" for i in range(20)])
    memory_mgr = _GraphMemoryManager()
    graph = create_memory_react_agent(
        memory_mgr=memory_mgr,
        llm=llm,
        tools=[],
        system_prompt_builder=lambda context: "base system prompt",
        token_threshold=120,
        keep_recent=4,
    )
    config = {"configurable": {"thread_id": "short-term-summary"}}
    context = MemoryAgentContext(user_id="alice", namespace="default")

    for i in range(8):
        graph.invoke(
            {"messages": [HumanMessage(content=f"user turn {i} " + "x" * 140)]},
            config=config,
            context=context,
        )

    snapshot = graph.get_state(config)
    checkpoint_messages = snapshot.values["messages"]

    assert snapshot.values["summary"]
    assert not _has_conversation_summary_message(checkpoint_messages)
    assert len(checkpoint_messages) < 16
    assert isinstance(llm.calls[-1][0], SystemMessage)
    assert "## Short-Term Memory Summary" in llm.calls[-1][0].content
    assert snapshot.values["summary"] in llm.calls[-1][0].content


def test_collect_tool_call_entries_matches_outputs():
    ai_message = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "id": "call-1",
                "name": "save_location",
                "args": {"location_name": "Kitchen"},
            }
        ],
    )
    tool_message = ToolMessage(
        content="Location saved: 'Kitchen'",
        name="save_location",
        tool_call_id="call-1",
    )

    entries = collect_tool_call_entries([ai_message, tool_message])

    assert list(entries) == ["ai-1"]
    assert entries["ai-1"][0].name == "save_location"
    assert entries["ai-1"][0].args == {"location_name": "Kitchen"}
    assert entries["ai-1"][0].output == tool_message


def test_delete_session_deletes_checkpoint_thread():
    memory_mgr = _MemoryManager()

    delete_session(memory_mgr, "thread-1")

    assert memory_mgr.checkpointer.deleted == ["thread-1"]


def test_get_latest_session_id_uses_first_checkpoint():
    memory_mgr = _MemoryManager()

    assert get_latest_session_id(memory_mgr) == "latest"


def test_get_user_ids_merges_profiles_and_memory_namespaces():
    memory_mgr = _MemoryManager()
    add_user_profile(memory_mgr, "default", "bob")
    memory_mgr.store.put(
        ("default", "alice", "facts"),
        "fact-1",
        {"text": "The user likes green tea."},
    )

    assert get_user_ids(memory_mgr, "default") == ["alice", "bob", "default"]


def test_list_and_format_long_term_memory_items():
    memory_mgr = _MemoryManager()
    memory_mgr.store.put(
        ("default", "alice", "facts"),
        "fact-1",
        {"text": "The user likes green tea."},
    )
    memory_mgr.store.put(
        ("default", "alice", "spatial"),
        "loc_kitchen",
        {"location": "Kitchen", "pose": {"x": 1.0, "y": 2.0, "z": 0.0}},
    )

    items = list_long_term_memory_items(memory_mgr.store, "default", "alice")
    formatted = [format_long_term_item(*item[:1], item[2], item[3]) for item in items]

    assert len(items) == 2
    assert "The user likes green tea." in formatted
    assert "Kitchen (1.0, 2.0, 0.0)" in formatted


def test_delete_user_long_term_memory_deletes_all_user_items():
    memory_mgr = _MemoryManager()
    add_user_profile(memory_mgr, "default", "alice")
    memory_mgr.store.put(
        ("default", "alice", "facts"),
        "fact-1",
        {"text": "The user likes green tea."},
    )
    memory_mgr.store.put(
        ("default", "alice", "spatial"),
        "loc_kitchen",
        {"location": "Kitchen", "pose": {"x": 1.0, "y": 2.0, "z": 0.0}},
    )

    deleted = delete_user(memory_mgr, "default", "alice")

    assert deleted == 3
    assert list_long_term_memory_items(memory_mgr.store, "default", "alice") == []
    assert "alice" not in get_user_ids(memory_mgr, "default")


def test_deleted_user_profile_hides_stale_namespaces():
    memory_mgr = _MemoryManager()
    add_user_profile(memory_mgr, "default", "robot_user")
    memory_mgr.store.put(
        ("default", "robot_user", "facts"),
        "fact-1",
        {"text": "temporary memory"},
    )
    delete_user(memory_mgr, "default", "robot_user")

    # Simulate a store backend that still reports the old namespace after deletion.
    memory_mgr.store.put(
        ("default", "robot_user", "facts"),
        "stale-marker",
        {"text": "stale namespace marker"},
    )

    assert "robot_user" not in get_user_ids(memory_mgr, "default")


def test_delete_legacy_user_without_profile_hides_namespace_user():
    memory_mgr = _MemoryManager()
    memory_mgr.store.put(
        ("default", "robot_user", "facts"),
        "fact-1",
        {"text": "legacy memory"},
    )

    deleted = delete_user(memory_mgr, "default", "robot_user")

    assert deleted == 2
    assert "robot_user" not in get_user_ids(memory_mgr, "default")
