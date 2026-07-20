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

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from rai.agents.langchain.core.react_agent import create_react_runnable
from rai.context import (
    ContextBudgetExceeded,
    ContextConfig,
    ContextManager,
    load_context_config,
)
from rai.context.manager import TRUNCATION_NOTICE
from rai.messages import HumanMultimodalMessage


def _manager(
    config: ContextConfig, responses: list[str] | None = None
) -> ContextManager:
    return ContextManager(
        config,
        summary_model_factory=lambda: FakeListChatModel(
            responses=responses or ["compressed history"]
        ),
    )


def test_load_context_config_uses_operational_budget(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[context]
max_input_tokens = 4096
trigger_ratio = 0.6
keep_ratio = 0.2
max_messages = 40
summary_max_tokens = 512
chars_per_token = 1.5
tokens_per_image = 768
""".strip()
    )

    config = load_context_config(config_path)

    assert config.max_input_tokens == 4096
    assert config.trigger_tokens == 2457
    assert config.keep_tokens == 819
    assert config.chars_per_token == 1.5
    assert config.tokens_per_image == 768


def test_few_large_historical_messages_are_compressed() -> None:
    config = ContextConfig(
        max_input_tokens=3000,
        trigger_ratio=0.5,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=300,
    )
    manager = _manager(config, ["intermediate summary", "old facts retained"])
    messages: list[BaseMessage] = [
        SystemMessage(content="system"),
        HumanMessage(content="old request " + "x" * 1500),
        AIMessage(content="old answer " + "y" * 1500),
        HumanMessage(content="new request"),
    ]

    prepared = manager.prepare(messages)

    assert prepared.compressed
    assert prepared.summary == "old facts retained"
    assert len(prepared.messages) == 2
    assert prepared.token_count <= config.max_input_tokens


def test_active_tool_calls_remain_countable_while_large_results_are_bounded() -> None:
    config = ContextConfig(
        max_input_tokens=1800,
        trigger_ratio=0.55,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=256,
    )
    manager = _manager(config)
    messages: list[BaseMessage] = [
        SystemMessage(content="robot system prompt"),
        HumanMessage(content="inspect point1 through point8"),
    ]
    for index in range(6):
        call_id = f"call-{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "analyze_scene",
                            "args": {"point": index},
                            "id": call_id,
                        }
                    ],
                ),
                ToolMessage(
                    content=f"point {index} result " + "detail " * 500,
                    tool_call_id=call_id,
                ),
            ]
        )

    prepared = manager.prepare(messages)

    calls = [m for m in prepared.messages if isinstance(m, AIMessage) and m.tool_calls]
    results = [m for m in prepared.messages if isinstance(m, ToolMessage)]
    assert len(calls) == 6
    assert len(results) == 6
    assert prepared.truncated_tool_results > 0
    assert prepared.token_count <= config.max_input_tokens
    assert any("Tool result truncated" in str(message.content) for message in results)


def test_large_tool_result_is_truncated_in_one_bounded_pass(monkeypatch) -> None:
    config = ContextConfig(
        max_input_tokens=6000,
        trigger_ratio=0.9,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=256,
        chars_per_token=2.0,
    )
    manager = _manager(config)
    message = ToolMessage(
        content="large visual payload " + "x" * 100_000,
        tool_call_id="large-result",
    )
    calls = 0
    original = manager._head_tail

    def counted_head_tail(text: str, token_limit: int, *, suffix: str) -> str:
        nonlocal calls
        calls += 1
        return original(text, token_limit, suffix=suffix)

    monkeypatch.setattr(manager, "_head_tail", counted_head_tail)
    fitted, truncated = manager._fit_tool_results(
        [message],
        summary="",
        tools=[],
        token_limit=manager._message_tokens(message) - 100,
    )

    assert truncated == 1
    assert calls == 1
    assert manager._message_tokens(fitted[0]) < manager._message_tokens(message)


def test_tool_truncation_falls_back_when_content_does_not_shrink(monkeypatch) -> None:
    manager = _manager(ContextConfig(chars_per_token=2.0))
    message = ToolMessage(content="x" * 1000, tool_call_id="stuck-result")
    calls = 0

    def unchanged(text: str, token_limit: int, *, suffix: str) -> str:
        nonlocal calls
        calls += 1
        return text

    monkeypatch.setattr(manager, "_head_tail", unchanged)
    fitted, truncated = manager._fit_tool_results(
        [message],
        summary="",
        tools=[],
        token_limit=100,
    )

    assert calls == 1
    assert truncated == 1
    assert fitted[0].content == TRUNCATION_NOTICE.strip()


def test_uncompressible_latest_user_input_fails_before_model_call() -> None:
    config = ContextConfig(
        max_input_tokens=512,
        trigger_ratio=0.6,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=64,
    )
    manager = _manager(config)

    with pytest.raises(ContextBudgetExceeded, match="max_input_tokens=512"):
        manager.prepare(
            [
                SystemMessage(content="system"),
                HumanMessage(content="z" * 5000),
            ]
        )


def test_tool_schemas_and_images_are_included_in_the_budget() -> None:
    @tool
    def inspect_component(component: str) -> str:
        """Inspect a named robot component and return its detailed status."""
        return component

    config = ContextConfig(
        max_input_tokens=4096,
        trigger_ratio=0.7,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=512,
        chars_per_token=2.0,
        tokens_per_image=900,
    )
    manager = _manager(config)
    text_only = [HumanMessage(content="inspect camera")]
    multimodal = [HumanMultimodalMessage(content="inspect camera", images=["aW1hZ2U="])]

    text_tokens = manager.count_tokens(text_only)
    with_tool_tokens = manager.count_tokens(text_only, tools=[inspect_component])
    image_tokens = manager.count_tokens(multimodal)

    assert with_tool_tokens > text_tokens
    assert image_tokens >= text_tokens + config.tokens_per_image


class _ScriptedToolModel(FakeListChatModel):
    scripted: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.scripted.pop(0))])


def test_async_invocation_continues_after_context_compression() -> None:
    model = _ScriptedToolModel(
        responses=["unused"],
        scripted=[AIMessage(content="answer after summary")],
    )
    config = ContextConfig(
        max_input_tokens=4000,
        trigger_ratio=0.5,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=256,
        chars_per_token=2.0,
    )
    manager = _manager(config, ["historical summary"])
    graph = create_react_runnable(llm=model, context_manager=manager)
    messages: list[BaseMessage] = [
        HumanMessage(content="old question " + "x" * 6000),
        AIMessage(content="old answer " + "y" * 6000),
        HumanMessage(content="new question"),
    ]

    result = asyncio.run(
        asyncio.wait_for(
            graph.ainvoke({"messages": messages}),
            timeout=3,
        )
    )

    assert result["summary"] == "historical summary"
    assert result["messages"][-1].content == "answer after summary"


def test_context_is_checked_between_tool_calls_in_one_user_turn() -> None:
    @tool
    def inspect_point(point: int) -> str:
        """Inspect one point."""
        return f"point {point} " + "visual detail " * 500

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "inspect_point", "args": {"point": index}, "id": f"c{index}"}
            ],
        )
        for index in range(4)
    ] + [AIMessage(content="inspection complete")]
    model = _ScriptedToolModel(responses=["unused"], scripted=scripted)
    config = ContextConfig(
        max_input_tokens=1600,
        trigger_ratio=0.55,
        keep_ratio=0.2,
        max_messages=80,
        summary_max_tokens=256,
    )
    manager = _manager(config)
    graph = create_react_runnable(
        llm=model,
        tools=[inspect_point],
        context_manager=manager,
    )

    result = graph.invoke({"messages": [HumanMessage(content="inspect four points")]})

    assert result["messages"][-1].content == "inspection complete"
    assert len(model.calls) == 5
    assert all(
        manager.count_tokens(call, tools=[inspect_point]) <= config.max_input_tokens
        for call in model.calls
    )
    assert any(
        "Tool result truncated" in str(message.content)
        for call in model.calls[1:]
        for message in call
        if isinstance(message, ToolMessage)
    )
