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

from langchain_core.messages import AIMessage


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


def test_load_thread_state_reads_checkpoint_values():
    demo = _load_demo_module()
    messages = [AIMessage(content="restored")]
    graph = _Graph({"messages": messages, "summary": "prior summary"})

    restored_messages, restored_summary = demo._load_thread_state(graph, "thread-1")

    assert restored_messages == messages
    assert restored_summary == "prior summary"


def test_welcome_message_is_ai_message():
    demo = _load_demo_module()

    message = demo._welcome_message()

    assert isinstance(message, AIMessage)
    assert "persistent memory" in message.content
