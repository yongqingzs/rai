# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import rai.agents.langchain.core.react_agent as react_agent
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field
from rai.agents.langchain.core.react_agent import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_TOKEN_THRESHOLD,
    estimate_tokens,
    summarize_messages,
)
from rai.memory.graph import MemoryAgentContext, create_memory_react_agent

OLD_SENTINEL = "OLD_CONTEXT_SENTINEL_7F31"
RECENT_SENTINEL = "RECENT_CONTEXT_SENTINEL_9A42"


class _MemoryManager:
    def __init__(self) -> None:
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


def _default_threshold_messages() -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for index in range(7):
        marker = OLD_SENTINEL if index == 0 else f"turn-{index}"
        messages.append(
            HumanMessage(content=f"user {marker} " + ("user-context " * 230))
        )
        recent = f" {RECENT_SENTINEL}" if index == 6 else ""
        messages.append(
            AIMessage(content=f"assistant turn-{index}{recent} " + ("answer " * 380))
        )
    return messages


def _print_report(
    *,
    title: str,
    input_messages: list[BaseMessage],
    summary: str,
    checkpoint_messages: list[BaseMessage],
    model_context: list[BaseMessage],
) -> None:
    print(f"\n=== {title} ===")
    print(
        "before: "
        f"messages={len(input_messages)}, estimated_tokens={estimate_tokens(input_messages)}, "
        f"threshold={DEFAULT_TOKEN_THRESHOLD}, keep_recent={DEFAULT_KEEP_RECENT}"
    )
    print(f"summary: {summary}")
    print(f"checkpoint_messages: {len(checkpoint_messages)}")
    print("main_model_context:")
    show_full = os.getenv("RAI_CONTEXT_FULL") == "1"
    for index, message in enumerate(model_context):
        content = str(message.content)
        displayed = content if show_full or message.type == "system" else content[:240]
        suffix = "" if len(displayed) == len(content) else " ..."
        print(f"  [{index}] {message.type} chars={len(content)}: {displayed}{suffix}")
    raw_recent = "\n".join(str(message.content) for message in model_context[1:])
    print(
        "checks: "
        f"summary_in_system={summary in str(model_context[0].content)}, "
        f"old_raw_removed={OLD_SENTINEL not in raw_recent}, "
        f"recent_preserved={RECENT_SENTINEL in raw_recent}"
    )


def test_default_threshold_requires_both_token_and_message_limits() -> None:
    messages = _default_threshold_messages()
    summary_model = FakeListChatModel(responses=["unused"])

    token_only = summarize_messages(
        messages[:DEFAULT_KEEP_RECENT],
        llm=summary_model,
    )
    message_only = summarize_messages(
        [HumanMessage(content=f"short-{index}") for index in range(13)],
        llm=summary_model,
    )

    assert estimate_tokens(messages[:DEFAULT_KEEP_RECENT]) > DEFAULT_TOKEN_THRESHOLD
    assert token_only["summary"] == ""
    assert token_only["messages"] == messages[:DEFAULT_KEEP_RECENT]
    assert message_only["summary"] == ""
    assert len(message_only["messages"]) == 13


def test_default_compression_exposes_checkpoint_and_actual_model_context(
    monkeypatch,
) -> None:
    input_messages = _default_threshold_messages()
    expected_summary = f"Earlier context preserved {OLD_SENTINEL}."
    monkeypatch.setattr(
        react_agent,
        "get_llm_model",
        lambda *args, **kwargs: FakeListChatModel(responses=[expected_summary]),
    )
    main_model = _RecordingFakeChatModel(responses=["final answer"])
    memory_mgr = _MemoryManager()
    graph = create_memory_react_agent(
        memory_mgr=memory_mgr,
        llm=main_model,
        tools=[],
        system_prompt_builder=lambda context: "Context compression test system prompt.",
    )
    config = {"configurable": {"thread_id": "context-compression-default"}}
    context = MemoryAgentContext(user_id="tester", namespace="tests")

    graph.invoke({"messages": input_messages}, config=config, context=context)

    snapshot = graph.get_state(config)
    summary = snapshot.values["summary"]
    checkpoint_messages = list(snapshot.values["messages"])
    model_context = main_model.calls[-1]
    system_message = model_context[0]
    raw_recent = "\n".join(str(message.content) for message in model_context[1:])

    assert len(input_messages) > DEFAULT_KEEP_RECENT
    assert estimate_tokens(input_messages) > DEFAULT_TOKEN_THRESHOLD
    assert summary == expected_summary
    assert isinstance(system_message, SystemMessage)
    assert "## Short-Term Memory Summary" in str(system_message.content)
    assert summary in str(system_message.content)
    assert len(model_context[1:]) == DEFAULT_KEEP_RECENT
    assert OLD_SENTINEL not in raw_recent
    assert RECENT_SENTINEL in raw_recent
    assert len(checkpoint_messages) == DEFAULT_KEEP_RECENT + 1

    _print_report(
        title="DETERMINISTIC CONTEXT COMPRESSION",
        input_messages=input_messages,
        summary=summary,
        checkpoint_messages=checkpoint_messages,
        model_context=model_context,
    )
