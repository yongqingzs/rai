# Copyright (C) 2025 Robotec.AI
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

import json
from functools import partial
from typing import (
    List,
    Optional,
    cast,
)

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import Annotated, NotRequired, TypedDict

from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.agents.langchain.invocation_helpers import (
    ainvoke_llm_with_tracing,
    invoke_llm_with_tracing,
)
from rai.context import ContextManager, inject_summary
from rai.initialization import get_llm_model
from rai.messages import SystemMultimodalMessage

DEFAULT_TOKEN_THRESHOLD = 8192
DEFAULT_KEEP_RECENT = 12


class ReActAgentState(TypedDict):
    """State type for the react agent.

    Parameters
    ----------
    messages : Annotated[List[BaseMessage], operator.add]
        List of messages in the conversation (supports checkpointing)
    """

    messages: Annotated[List[BaseMessage], add_messages]
    summary: NotRequired[str]


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str | SystemMultimodalMessage],
    state: ReActAgentState,
    config: RunnableConfig,
):
    """Process messages using the LLM.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for processing
    state : ReActAgentState
        Current state containing messages
    config : RunnableConfig
        Configuration including callbacks for tracing

    Returns
    -------
    ReActAgentState
        New messages to append (operator.add reducer)

    Raises
    ------
    ValueError
        If state is invalid or LLM processing fails
    """
    new_messages: List[BaseMessage] = []
    has_system = isinstance(state["messages"][0], SystemMessage)

    if isinstance(system_prompt, SystemMultimodalMessage):
        if not has_system:
            new_messages.append(system_prompt)
    elif system_prompt:
        if not has_system:
            new_messages.append(SystemMessage(content=system_prompt))

    # Invoke LLM with system prompt and the current compacted summary prepended.
    all_msgs = _inject_state_summary(
        list(new_messages) + list(state["messages"]), state.get("summary", "")
    )
    ai_msg = invoke_llm_with_tracing(llm, all_msgs, config)
    new_messages.append(ai_msg)

    return {"messages": new_messages}


async def allm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str | SystemMultimodalMessage],
    state: ReActAgentState,
    config: RunnableConfig,
):
    """Async variant of ``llm_node`` used by LangGraph event streaming."""
    new_messages: List[BaseMessage] = []
    has_system = isinstance(state["messages"][0], SystemMessage)

    if isinstance(system_prompt, SystemMultimodalMessage):
        if not has_system:
            new_messages.append(system_prompt)
    elif system_prompt:
        if not has_system:
            new_messages.append(SystemMessage(content=system_prompt))

    all_msgs = _inject_state_summary(
        list(new_messages) + list(state["messages"]), state.get("summary", "")
    )
    ai_msg = await ainvoke_llm_with_tracing(llm, all_msgs, config)
    new_messages.append(ai_msg)

    return {"messages": new_messages}


def create_react_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str | SystemMultimodalMessage] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    context_manager: ContextManager | None = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    """Create a react agent that can process messages and optionally use tools.

    Parameters
    ----------
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    tools : Optional[List[BaseTool]], default=None
        List of tools the agent can use
    checkpointer : Optional[BaseCheckpointSaver], default=None
        Checkpointer for short-term (thread-scoped) memory persistence
    store : Optional[BaseStore], default=None
        Store for long-term (cross-session) memory persistence

    Returns
    -------
    Runnable[ReActAgentState, ReActAgentState]
        A runnable that processes messages and optionally uses tools

    Raises
    ------
    ValueError
        If tools are provided but invalid
    """
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)

    def _tool_runner_delta(tool_runner, state, config):
        """Wrap ToolRunner to return delta for operator.add reducer."""
        initial_len = len(state["messages"])
        result = tool_runner.invoke(state, config)
        new_msgs = list(result["messages"][initial_len:])
        return {"messages": new_msgs}

    async def _atool_runner_delta(tool_runner, state, config):
        """Async wrapper for ToolRunner used by ``astream_events``."""
        initial_len = len(state["messages"])
        result = await tool_runner.ainvoke(state, config)
        new_msgs = list(result["messages"][initial_len:])
        return {"messages": new_msgs}

    graph = StateGraph(ReActAgentState)

    def _prepare_context(state: ReActAgentState):
        if context_manager is None:
            return {"summary": state.get("summary", "")}
        prepared = context_manager.prepare(
            state["messages"],
            summary=state.get("summary", ""),
            tools=tools or [],
        )
        if (
            not prepared.compressed
            and prepared.truncated_tool_results == 0
            and prepared.summary == state.get("summary", "")
        ):
            return {"summary": prepared.summary}
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *prepared.messages,
            ],
            "summary": prepared.summary,
        }

    if context_manager is not None:
        graph.add_node("context_management", _prepare_context)
        graph.add_edge(START, "context_management")
        graph.add_edge("context_management", "llm")
    else:
        graph.add_edge(START, "llm")

    if tools:
        tool_runner = ToolRunner(tools)
        graph.add_node(
            "tools",
            RunnableCallable(
                partial(_tool_runner_delta, tool_runner),
                partial(_atool_runner_delta, tool_runner),
                name="tools",
            ),
        )
        graph.add_conditional_edges(
            "llm",
            tools_condition,
        )
        graph.add_edge(
            "tools", "context_management" if context_manager is not None else "llm"
        )
        # Bind tools to LLM
        bound_llm = cast(BaseChatModel, llm.bind_tools(tools))
        graph.add_node(
            "llm",
            RunnableCallable(
                partial(llm_node, bound_llm, system_prompt),
                partial(allm_node, bound_llm, system_prompt),
                name="llm",
            ),
        )
    else:
        graph.add_node(
            "llm",
            RunnableCallable(
                partial(llm_node, llm, system_prompt),
                partial(allm_node, llm, system_prompt),
                name="llm",
            ),
        )

    # Compile the graph
    return graph.compile(
        checkpointer=False if checkpointer is None else checkpointer,
        store=store,
    )


def _inject_state_summary(
    messages: List[BaseMessage], summary: str
) -> List[BaseMessage]:
    if not summary:
        return messages
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            updated = message.model_copy(
                update={"content": inject_summary(str(message.content), summary)}
            )
            return [*messages[:index], updated, *messages[index + 1 :]]
    return [SystemMessage(content=inject_summary("", summary)), *messages]


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Rough token estimate for a list of messages (~4 chars per token)."""
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += len(m.content) // 4
        elif isinstance(m.content, list):
            for c in m.content:
                if isinstance(c, dict) and "text" in c:
                    total += len(c["text"]) // 4
        if hasattr(m, "tool_calls"):
            for tc in m.tool_calls:
                total += len(json.dumps(tc)) // 4
    return total


def summarize_messages(
    messages: List[BaseMessage],
    existing_summary: str = "",
    llm: Optional[BaseChatModel] = None,
    config: Optional[RunnableConfig] = None,
    threshold: int = DEFAULT_TOKEN_THRESHOLD,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> dict:
    """Compress messages if they exceed token threshold.

    Parameters
    ----------
    messages : List[BaseMessage]
        Full conversation history
    existing_summary : str
        Previous summary (if any)
    llm : Optional[BaseChatModel]
        LLM to use for summarization (auto-selected if None)
    config : Optional[RunnableConfig]
        Runnable config for tracing
    threshold : int
        Token threshold before compression kicks in
    keep_recent : int
        Number of recent messages to always keep

    Returns
    -------
    dict
        {"messages": List[BaseMessage], "summary": str}
        If no compression needed, returns original messages and summary.
    """
    token_count = count_tokens_approximately(messages)
    if token_count <= threshold:
        return {"messages": messages, "summary": existing_summary}

    if llm is None:
        llm = get_llm_model("simple_model", streaming=False)

    # Separate system messages and conversational messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    middleware = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", threshold),
        keep=("messages", keep_recent),
        token_counter=count_tokens_approximately,
    )
    cutoff = middleware._determine_cutoff_index(conv_msgs)
    if cutoff <= 0:
        middleware = SummarizationMiddleware(
            model=llm,
            trigger=("tokens", threshold),
            keep=("tokens", max(1, threshold // 3)),
            token_counter=count_tokens_approximately,
        )
        cutoff = middleware._determine_cutoff_index(conv_msgs)
    if cutoff <= 0:
        return {"messages": messages, "summary": existing_summary}
    to_summarize, recent = middleware._partition_messages(conv_msgs, cutoff)
    summary_input: list[BaseMessage] = []
    if existing_summary:
        summary_input.append(
            HumanMessage(content=f"Previous conversation summary:\n{existing_summary}")
        )
    summary_input.extend(to_summarize)
    new_summary = middleware._create_summary(summary_input)

    # Rebuild message list with only real recent messages. The summary is returned
    # separately so callers can keep it as state instead of conversation history.
    compressed = system_msgs + recent

    return {"messages": compressed, "summary": new_summary}
