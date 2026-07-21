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

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field
from rai.context import ContextConfig
from rai.memory.config import MemoryConfig
from rai.memory.graph import MemoryAgentContext, create_memory_react_agent
from rai.memory.manager import MemoryManager


class _ScriptedToolModel(FakeListChatModel):
    scripted: list[AIMessage] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.scripted.pop(0))])


def test_multi_tool_turn_only_persists_root_namespace(tmp_path) -> None:
    @tool
    def inspect_point(point: int) -> str:
        """Inspect one location."""
        return f"point {point} inspected"

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "inspect_point", "args": {"point": index}, "id": f"c{index}"}
            ],
        )
        for index in range(6)
    ] + [AIMessage(content="inspection complete")]
    model = _ScriptedToolModel(responses=["unused"], scripted=scripted)
    manager = MemoryManager(
        config=MemoryConfig(
            enabled=True,
            backend="sqlite",
            short_term_path=str(tmp_path / "checkpoints.db"),
            long_term_path=str(tmp_path / "store.db"),
        )
    )
    manager.start()
    try:
        graph = create_memory_react_agent(
            manager,
            model,
            [inspect_point],
            lambda _context: "robot system prompt",
            context_config=ContextConfig(max_input_tokens=32_768),
        )
        result = manager._run_on_async_loop(
            graph.ainvoke(
                {"messages": [HumanMessage(content="inspect six points")]},
                {"configurable": {"thread_id": "long-task"}},
                context=MemoryAgentContext(user_id="operator", namespace="inspection"),
            )
        )

        async def checkpoint_namespaces():
            cursor = await manager.checkpointer.conn.execute(
                "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
                ("long-task",),
            )
            return [row[0] for row in await cursor.fetchall()]

        namespaces = manager._run_on_async_loop(checkpoint_namespaces())
    finally:
        manager.stop()

    assert namespaces == [""]
    assert result["messages"][-1].content == "inspection complete"
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 6
