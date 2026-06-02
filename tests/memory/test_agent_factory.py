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

import rai.memory.agent_factory as agent_factory
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore
from rai.memory.agent_factory import build_memory_system_prompt
from rai.memory.graph import MemoryAgentContext


class _MemoryManager:
    def __init__(self):
        self.store = InMemoryStore()
        self.checkpointer = object()


class _Tool(BaseTool):
    name: str = "extra_tool"
    description: str = "extra"

    def _run(self):
        return "ok"


def test_build_memory_system_prompt_injects_long_term_memory_and_extra_sections():
    prompt = build_memory_system_prompt(
        base_system_prompt="base",
        long_term_memory="- remembered fact",
        extra_sections=["extra section"],
    )

    assert "base" in prompt
    assert "- remembered fact" in prompt
    assert "save_fact" in prompt
    assert "extra section" in prompt


def test_create_memory_agent_with_tools_adds_memory_and_extra_tools(monkeypatch):
    memory_mgr = _MemoryManager()
    captured = {}

    def _create_memory_react_agent(**kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(
        agent_factory,
        "create_memory_react_agent",
        _create_memory_react_agent,
    )

    graph = agent_factory.create_memory_agent_with_tools(
        memory_mgr=memory_mgr,
        llm=object(),
        base_system_prompt_builder=lambda context: f"base for {context.user_id}",
        namespace="default",
        user_id="alice",
        extra_tools=[_Tool()],
        extra_prompt_sections=["extra section"],
    )

    assert graph == "graph"
    assert [tool.name for tool in captured["tools"]] == [
        "save_fact",
        "save_location",
        "forget_memory",
        "extra_tool",
    ]
    prompt = captured["system_prompt_builder"](
        MemoryAgentContext(user_id="alice", namespace="default")
    )
    assert "base for alice" in prompt
    assert "extra section" in prompt
