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

from dataclasses import dataclass
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
from langgraph.runtime import Runtime
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import Annotated, TypedDict

from rai.agents.langchain.core.react_agent import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_TOKEN_THRESHOLD,
    create_react_runnable,
    summarize_messages,
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


def _inject_summary(system_prompt: str, summary: str) -> str:
    if not summary:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        "## Short-Term Memory Summary\n"
        "The following is an internal summary of earlier conversation in this "
        f"thread. Use it as context, not as a literal assistant message.\n{summary}"
    )


def create_memory_react_agent(
    memory_mgr: MemoryManager,
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt_builder: SystemPromptBuilder,
    token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> Runnable[MemoryState, MemoryState]:
    """Create a memory-aware ReAct graph.

    The outer graph owns checkpointed short-term memory. The inner ReAct graph is
    intentionally stateless and receives a temporary system prompt at invocation time.
    """
    inner_agent = create_react_runnable(
        llm=llm,
        tools=tools,
        checkpointer=None,
        store=memory_mgr.store,
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

    def summarize_node(state: MemoryState, runtime: Runtime[MemoryAgentContext]):
        result = summarize_messages(
            messages=state["messages"],
            existing_summary=state.get("summary", ""),
            threshold=token_threshold,
            keep_recent=keep_recent,
        )
        if result["messages"] is state["messages"]:
            return {"summary": result["summary"]}

        retained_ids = {m.id for m in result["messages"] if getattr(m, "id", None)}
        removals = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if getattr(m, "id", None) and m.id not in retained_ids
        ]
        return {"messages": removals + result["messages"], "summary": result["summary"]}

    def run_react(
        state: MemoryState,
        runtime: Runtime[MemoryAgentContext],
        config: RunnableConfig,
    ):
        configurable = config.get("configurable", {})
        try:
            system_prompt = _inject_summary(
                state["system_prompt"],
                state.get("summary", ""),
            )
            conversation_messages = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            if runtime.context.transient_images:
                conversation_messages = list(conversation_messages)
                for index in range(len(conversation_messages) - 1, -1, -1):
                    message = conversation_messages[index]
                    if isinstance(message, HumanMessage):
                        conversation_messages[index] = HumanMultimodalMessage(
                            content=_message_text(message),
                            images=runtime.context.transient_images,
                        )
                        break
            react_messages = [
                SystemMessage(content=system_prompt),
                *conversation_messages,
            ]
            result = inner_agent.invoke(
                {"messages": react_messages},
                RunnableConfig(
                    callbacks=config.get("callbacks", []),
                    configurable=configurable,
                ),
                context=runtime.context,
            )
            new_messages = list(result["messages"][len(react_messages) :])
            return {"messages": new_messages, "summary": state.get("summary", "")}
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
            system_prompt = _inject_summary(
                state["system_prompt"],
                state.get("summary", ""),
            )
            conversation_messages = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            if runtime.context.transient_images:
                conversation_messages = list(conversation_messages)
                for index in range(len(conversation_messages) - 1, -1, -1):
                    message = conversation_messages[index]
                    if isinstance(message, HumanMessage):
                        conversation_messages[index] = HumanMultimodalMessage(
                            content=_message_text(message),
                            images=runtime.context.transient_images,
                        )
                        break
            react_messages = [
                SystemMessage(content=system_prompt),
                *conversation_messages,
            ]
            result = await inner_agent.ainvoke(
                {"messages": react_messages},
                RunnableConfig(
                    callbacks=config.get("callbacks", []),
                    configurable=configurable,
                ),
                context=runtime.context,
            )
            new_messages = list(result["messages"][len(react_messages) :])
            return {"messages": new_messages, "summary": state.get("summary", "")}
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Agent error: {e}")],
                "summary": state.get("summary", ""),
            }

    builder = StateGraph(MemoryState, context_schema=MemoryAgentContext)
    builder.add_node("enrich_prompt", enrich_prompt)
    builder.add_node("summarize", summarize_node)
    builder.add_node("react", RunnableCallable(run_react, arun_react, name="react"))

    builder.add_edge(START, "enrich_prompt")
    builder.add_edge("enrich_prompt", "summarize")
    builder.add_edge("summarize", "react")
    builder.add_edge("react", END)

    return builder.compile(
        checkpointer=memory_mgr.checkpointer,
        store=memory_mgr.store,
    )
