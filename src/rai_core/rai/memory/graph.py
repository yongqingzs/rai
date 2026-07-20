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

from dataclasses import dataclass, replace
from typing import Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import Annotated, TypedDict

import rai.agents.langchain.core.react_agent as react_agent
from rai.agents.langchain.core.react_agent import create_react_runnable
from rai.context import (
    ContextConfig,
    ContextManager,
    load_context_config,
)
from rai.memory.manager import MemoryManager
from rai.messages import HumanMultimodalMessage


@dataclass
class MemoryAgentContext:
    user_id: str
    namespace: str
    transient_images: list[str] | None = None


class MemoryState(TypedDict):
    messages: Annotated[
        List[
            AIMessage
            | HumanMessage
            | HumanMultimodalMessage
            | ToolMessage
            | SystemMessage
            | RemoveMessage
        ],
        add_messages,
    ]
    summary: str
    system_prompt: str


SystemPromptBuilder = Callable[[MemoryAgentContext], str]


def _message_text(message: HumanMessage) -> str:
    text = message.text
    if isinstance(text, str):
        return text
    return str(text)


def create_memory_react_agent(
    memory_mgr: MemoryManager,
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt_builder: SystemPromptBuilder,
    token_threshold: int | None = None,
    keep_recent: int | None = None,
    context_config: ContextConfig | None = None,
) -> Runnable[MemoryState, MemoryState]:
    """Create a memory-aware ReAct graph.

    The outer graph owns checkpointed short-term memory. The inner ReAct graph is
    intentionally stateless and receives a temporary system prompt at invocation time.
    """
    context_config = context_config or load_context_config()
    legacy_trigger = None
    legacy_keep = None
    if token_threshold is not None or keep_recent is not None:
        threshold = token_threshold or context_config.trigger_tokens
        retained = keep_recent or 12
        context_config = replace(
            context_config,
            max_input_tokens=max(context_config.max_input_tokens, threshold * 4),
        )
        legacy_trigger = [("tokens", threshold)]
        legacy_keep = ("messages", retained)

    context_manager = ContextManager(
        context_config,
        summary_model_factory=lambda: react_agent.get_llm_model(
            "simple_model", streaming=False
        ),
        trigger=legacy_trigger,
        keep=legacy_keep,
    )
    inner_agent = create_react_runnable(
        llm=llm,
        tools=tools,
        checkpointer=None,
        store=memory_mgr.store,
        context_manager=context_manager,
    )

    def enrich_prompt(state: MemoryState, runtime: Runtime[MemoryAgentContext]):
        system_content = system_prompt_builder(runtime.context)
        removals = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if isinstance(m, SystemMessage) and getattr(m, "id", None)
        ]
        return {
            "messages": removals,
            "system_prompt": system_content,
            "summary": state.get("summary", ""),
        }

    def run_react(
        state: MemoryState,
        runtime: Runtime[MemoryAgentContext],
        config: RunnableConfig,
    ):
        configurable = config.get("configurable", {})
        try:
            conversation_messages = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            transient_source: HumanMessage | None = None
            if runtime.context.transient_images:
                conversation_messages = list(conversation_messages)
                for index in range(len(conversation_messages) - 1, -1, -1):
                    message = conversation_messages[index]
                    if isinstance(message, HumanMessage):
                        transient_source = message
                        conversation_messages[index] = HumanMultimodalMessage(
                            content=_message_text(message),
                            images=runtime.context.transient_images,
                        )
                        break
            react_messages = [
                SystemMessage(content=state["system_prompt"]),
                *conversation_messages,
            ]
            result = inner_agent.invoke(
                {
                    "messages": react_messages,
                    "summary": state.get("summary", ""),
                },
                RunnableConfig(
                    callbacks=config.get("callbacks", []),
                    configurable=configurable,
                ),
                context=runtime.context,
            )
            effective_messages = [
                message
                for message in result["messages"]
                if not isinstance(message, SystemMessage)
            ]
            effective_messages = _restore_transient_human_message(
                effective_messages, transient_source
            )
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *effective_messages,
                ],
                "summary": result.get("summary", state.get("summary", "")),
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Agent error: {e}")],
                "summary": state.get("summary", ""),
            }

    async def arun_react(
        state: MemoryState,
        runtime: Runtime[MemoryAgentContext],
        config: RunnableConfig,
    ):
        configurable = config.get("configurable", {})
        try:
            conversation_messages = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            transient_source: HumanMessage | None = None
            if runtime.context.transient_images:
                conversation_messages = list(conversation_messages)
                for index in range(len(conversation_messages) - 1, -1, -1):
                    message = conversation_messages[index]
                    if isinstance(message, HumanMessage):
                        transient_source = message
                        conversation_messages[index] = HumanMultimodalMessage(
                            content=_message_text(message),
                            images=runtime.context.transient_images,
                        )
                        break
            react_messages = [
                SystemMessage(content=state["system_prompt"]),
                *conversation_messages,
            ]
            result = await inner_agent.ainvoke(
                {
                    "messages": react_messages,
                    "summary": state.get("summary", ""),
                },
                RunnableConfig(
                    callbacks=config.get("callbacks", []),
                    configurable=configurable,
                ),
                context=runtime.context,
            )
            effective_messages = [
                message
                for message in result["messages"]
                if not isinstance(message, SystemMessage)
            ]
            effective_messages = _restore_transient_human_message(
                effective_messages, transient_source
            )
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *effective_messages,
                ],
                "summary": result.get("summary", state.get("summary", "")),
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Agent error: {e}")],
                "summary": state.get("summary", ""),
            }

    builder = StateGraph(MemoryState, context_schema=MemoryAgentContext)
    builder.add_node("enrich_prompt", enrich_prompt)
    builder.add_node("react", RunnableCallable(run_react, arun_react, name="react"))

    builder.add_edge(START, "enrich_prompt")
    builder.add_edge("enrich_prompt", "react")
    builder.add_edge("react", END)

    return builder.compile(
        checkpointer=memory_mgr.checkpointer,
        store=memory_mgr.store,
    )


def _restore_transient_human_message(
    messages: list, source: HumanMessage | None
) -> list:
    if source is None:
        return messages
    source_text = _message_text(source)
    restored = list(messages)
    for index in range(len(restored) - 1, -1, -1):
        message = restored[index]
        if (
            isinstance(message, HumanMultimodalMessage)
            and _message_text(message) == source_text
        ):
            restored[index] = source
            break
    return restored
