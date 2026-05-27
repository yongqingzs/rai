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

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from rai.frontend.memory_streamlit import collect_tool_call_entries
from rai.memory.long_term import format_long_term_item, list_long_term_memory_items
from rai.memory.session import delete_session, get_latest_session_id, load_thread_state
from rai.memory.users import add_user_profile, delete_user, get_user_ids


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
