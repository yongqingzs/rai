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
import operator
from functools import partial
from typing import (
    List,
    Optional,
    cast,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import Annotated, TypedDict

from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.agents.langchain.invocation_helpers import (
    ainvoke_llm_with_tracing,
    invoke_llm_with_tracing,
)
from rai.initialization import get_llm_model
from rai.messages import SystemMultimodalMessage

SUMMARIZE_PROMPT = """Condense the following conversation into a brief summary while preserving:
- Key decisions and conclusions
- User preferences and stated facts
- Important context for continuing the dialogue
- Any tool results that are relevant

Write the summary in second person ("You are a robot...", "The user asked...").
Return ONLY the summary, no preamble.

Conversation:
{conversation}
"""

DEFAULT_TOKEN_THRESHOLD = 8192
DEFAULT_KEEP_RECENT = 12


class ReActAgentState(TypedDict):
    """State type for the react agent.

    Parameters
    ----------
    messages : Annotated[List[BaseMessage], operator.add]
        List of messages in the conversation (supports checkpointing)
    """

    messages: Annotated[List[BaseMessage], operator.add]


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

    # Invoke LLM with system prompt prepended
    all_msgs = list(new_messages) + list(state["messages"])
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

    all_msgs = list(new_messages) + list(state["messages"])
    ai_msg = await ainvoke_llm_with_tracing(llm, all_msgs, config)
    new_messages.append(ai_msg)

    return {"messages": new_messages}


def create_react_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str | SystemMultimodalMessage] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
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
        graph.add_edge("tools", "llm")
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
    return graph.compile(checkpointer=checkpointer, store=store)


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
    token_count = estimate_tokens(messages)
    if token_count <= threshold or len(messages) <= keep_recent:
        return {"messages": messages, "summary": existing_summary}

    if llm is None:
        llm = get_llm_model("simple_model", streaming=False)

    # Separate system messages and conversational messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # Keep recent messages, summarize the rest
    recent = conv_msgs[-keep_recent:]
    to_summarize = conv_msgs[:-keep_recent]

    # Build conversation text for summarization
    conv_text = "\n".join(f"{type(m).__name__}: {m.content}" for m in to_summarize)

    prompt = SUMMARIZE_PROMPT.format(conversation=conv_text)
    if existing_summary:
        prompt = f"Previous summary:\n{existing_summary}\n\n" + prompt

    from langchain_core.messages import HumanMessage

    summary_response = invoke_llm_with_tracing(
        llm, [HumanMessage(content=prompt)], config
    )
    new_summary = summary_response.content

    # Rebuild message list with only real recent messages. The summary is returned
    # separately so callers can keep it as state instead of conversation history.
    compressed = system_msgs + recent

    return {"messages": compressed, "summary": new_summary}
