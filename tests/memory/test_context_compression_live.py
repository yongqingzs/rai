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

from typing import Any

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from rai import get_llm_model
from rai.agents.langchain.core.react_agent import estimate_tokens
from rai.memory.graph import MemoryAgentContext, create_memory_react_agent

OLD_SENTINEL = "LIVE_OLD_CONTEXT_SENTINEL_C281"
RECENT_SENTINEL = "LIVE_RECENT_CONTEXT_SENTINEL_D492"
LIVE_TOKEN_THRESHOLD = 300
LIVE_KEEP_RECENT = 4


class _MemoryManager:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()


class _ContextCapture(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        self.calls.extend([list(batch) for batch in messages])


def _live_messages() -> list[BaseMessage]:
    return [
        HumanMessage(
            content=(
                f"Remember this exact verification token: {OLD_SENTINEL}. "
                + ("old inspection context " * 30)
            )
        ),
        AIMessage(content=f"I will preserve {OLD_SENTINEL}. " + ("old response " * 30)),
        HumanMessage(content="Intermediate user context. " + ("middle " * 35)),
        AIMessage(content="Intermediate assistant context. " + ("middle " * 35)),
        HumanMessage(content="Recent user request. " + ("recent " * 35)),
        AIMessage(content="Recent assistant response. " + ("recent " * 35)),
        HumanMessage(content=f"Keep this recent token: {RECENT_SENTINEL}."),
        AIMessage(content="Acknowledge the recent token and answer briefly."),
    ]


def _print_live_report(
    *,
    input_messages: list[BaseMessage],
    summary: str,
    checkpoint_messages: list[BaseMessage],
    model_context: list[BaseMessage],
) -> None:
    print("\n=== LIVE MODEL CONTEXT COMPRESSION ===")
    print(
        "before: "
        f"messages={len(input_messages)}, estimated_tokens={estimate_tokens(input_messages)}, "
        f"threshold={LIVE_TOKEN_THRESHOLD}, keep_recent={LIVE_KEEP_RECENT}"
    )
    print(f"summary_from_real_model: {summary}")
    print(f"checkpoint_messages: {len(checkpoint_messages)}")
    print("actual_main_model_context:")
    for index, message in enumerate(model_context):
        print(f"  [{index}] {message.type}: {message.content}")
    raw_recent = "\n".join(str(message.content) for message in model_context[1:])
    print(
        "checks: "
        f"summary_in_system={summary in str(model_context[0].content)}, "
        f"old_raw_removed={OLD_SENTINEL not in raw_recent}, "
        f"recent_preserved={RECENT_SENTINEL in raw_recent}"
    )


@pytest.mark.billable
@pytest.mark.manual
def test_live_model_receives_compressed_context() -> None:
    input_messages = _live_messages()
    capture = _ContextCapture()
    memory_mgr = _MemoryManager()
    main_model = get_llm_model("complex_model", streaming=False)
    graph = create_memory_react_agent(
        memory_mgr=memory_mgr,
        llm=main_model,
        tools=[],
        system_prompt_builder=lambda context: "Live context compression test.",
        token_threshold=LIVE_TOKEN_THRESHOLD,
        keep_recent=LIVE_KEEP_RECENT,
    )
    config = {
        "configurable": {"thread_id": "context-compression-live"},
        "callbacks": [capture],
    }
    context = MemoryAgentContext(user_id="tester", namespace="tests")

    graph.invoke({"messages": input_messages}, config=config, context=context)

    snapshot = graph.get_state(config)
    summary = str(snapshot.values["summary"])
    checkpoint_messages = list(snapshot.values["messages"])
    model_context = next(
        batch
        for batch in reversed(capture.calls)
        if batch
        and isinstance(batch[0], SystemMessage)
        and "## Short-Term Memory Summary" in str(batch[0].content)
    )
    raw_recent = "\n".join(str(message.content) for message in model_context[1:])

    assert estimate_tokens(input_messages) > LIVE_TOKEN_THRESHOLD
    assert len(input_messages) > LIVE_KEEP_RECENT
    assert summary
    assert OLD_SENTINEL in summary
    assert summary in str(model_context[0].content)
    assert len(model_context[1:]) == LIVE_KEEP_RECENT
    assert OLD_SENTINEL not in raw_recent
    assert RECENT_SENTINEL in raw_recent
    assert len(checkpoint_messages) == LIVE_KEEP_RECENT + 1

    _print_live_report(
        input_messages=input_messages,
        summary=summary,
        checkpoint_messages=checkpoint_messages,
        model_context=model_context,
    )
