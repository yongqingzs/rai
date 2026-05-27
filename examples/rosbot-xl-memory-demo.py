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

"""ROSBot XL demo with persistent memory layer.

Demonstrates:
- System memory: Embodiment JSON always in context
- Short-term memory: Conversation history with token-threshold summarization
- Long-term memory: Agent-driven CRUD via memory tools (facts + spatial)

Usage:
    uv run examples/rosbot-xl-memory-demo.py
"""

import json
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rai import get_embeddings_model, get_llm_model
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.frontend.memory_streamlit import render_memory_sidebar
from rai.memory import (
    MemoryAgentContext,
    MemoryManager,
    create_memory_react_agent,
    load_memory_config,
)
from rai.memory.long_term import render_long_term_memories
from rai.tools.memory import create_memory_tools
from rai.tools.time import WaitForSecondsTool

# --- Constants ---

EMBEDDING_PATH = Path(__file__).parent / "embodiments" / "rosbotxl_embodiment1.json"

SYSTEM_PROMPT_TEMPLATE = """You are a ROSBot XL robot assistant. You can move around, look at objects, and help with tasks.

## System Memory (Embodiment)
{embodiment}

## Long-Term Memory (Persisted Across Sessions)
The following facts and locations are fully loaded in your context. Use this knowledge when answering questions or planning actions. If none are listed, you have no stored memories yet.
{long_term_memory}

## Available Memory Tools
You have access to memory tools:
- save_fact: Save text facts that should persist across sessions
- save_location: Save structured spatial/location data
- forget_memory: Delete stored memories

Use these tools proactively:
- When the user shares preferences or important information, save them with save_fact
- When you identify or learn about a location with coordinates, use save_location with a pose like: {{"x": 1.0, "y": 2.0, "z": 0.0}}
- When the user asks to forget something, use forget_memory"""


def _load_embodiment(path: Path) -> str:
    """Load embodiment JSON and format as text for system prompt."""
    if not path.exists():
        return f"(Embodiment file not found at {path})"
    try:
        data = json.loads(path.read_text())
        desc = data.get("description", "")
        rules = data.get("rules", [])
        capabilities = data.get("capabilities", [])

        parts = [desc]
        if rules:
            parts.append("Rules:\n" + "\n".join(f"  - {r}" for r in rules))
        if capabilities:
            parts.append(
                "Capabilities:\n" + "\n".join(f"  - {c}" for c in capabilities)
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"(Error loading embodiment: {e})"


def build_memory_agent(
    memory_mgr: MemoryManager,
    embodiment_path: Path,
    user_id: str = "default",
    namespace: str = "default",
) -> object:
    """Build a memory-aware agent graph.

    Graph structure:
        START -> enrich_prompt -> summarize -> react

    - enrich_prompt: Load all LTM + inject system prompt with embodiment
    - summarize: Compress conversation when tokens exceed threshold
    - react: Inner ReAct agent with memory tools

    Long-term memories are loaded at each turn (no search needed).
    The agent has save/forget tools for memory.
    """
    llm = get_llm_model("complex_model", streaming=True)
    embodiment_text = _load_embodiment(embodiment_path)

    # Create memory tools (bound to store + namespace)
    memory_tools_dict = create_memory_tools(
        store=memory_mgr.store,
        namespace=namespace,
        user_id=user_id,
    )
    # All tools: save/forget memory + time tool (recall not needed, LTM is in context)
    all_tools = [
        memory_tools_dict["save_fact"],
        memory_tools_dict["save_location"],
        memory_tools_dict["forget"],
        WaitForSecondsTool(),
    ]

    def build_system_prompt(context: MemoryAgentContext) -> str:
        long_term_memory = render_long_term_memories(
            memory_mgr.store,
            context.namespace,
            context.user_id,
        )
        return SYSTEM_PROMPT_TEMPLATE.format(
            embodiment=embodiment_text,
            long_term_memory=long_term_memory,
        )

    graph = create_memory_react_agent(
        memory_mgr=memory_mgr,
        llm=llm,
        tools=all_tools,
        system_prompt_builder=build_system_prompt,
    )
    return graph


def initialize_memory_mgr() -> MemoryManager:
    """Initialize and return the memory manager."""
    config = load_memory_config()
    if not config.enabled:
        st.error("Memory is disabled. Enable it in config.toml [memory] section.")
        st.stop()

    if config.backend == "postgres" and not config.connection:
        st.error(
            "PostgreSQL backend selected but no connection string in config.toml. "
            "Add: connection = 'postgresql://user:pass@host:5432/db'"
        )
        st.stop()

    memory_mgr = MemoryManager(config=config)
    try:
        embeddings = get_embeddings_model()
        memory_mgr = MemoryManager(config=config, embeddings=embeddings)
    except Exception as e:
        st.warning(f"Could not load embeddings, semantic search disabled: {e}")
        memory_mgr = MemoryManager(config=config)

    memory_mgr.start()
    memory_mgr.setup()
    return memory_mgr


def _welcome_message() -> AIMessage:
    return AIMessage(
        content=(
            "Hi! I'm a robot with persistent memory. "
            "Our conversation is summarized when it gets long (short-term). "
            "I'll proactively save important facts and locations to long-term memory. "
            "Ask me to forget something and I'll delete matching memories directly."
        )
    )


def run_memory_app():
    """Run the Streamlit app with persistent memory."""
    st.set_page_config(page_title="RAI Memory Demo", page_icon=":robot:")
    st.title(":robot: ROSBot XL - Persistent Memory Demo")
    st.sidebar.header("Configuration")

    config = load_memory_config()
    st.sidebar.markdown(
        f"**Backend:** `{config.backend}`\n**Namespace:** `{config.namespace}`"
    )
    st.sidebar.markdown("---")

    # Initialize memory manager
    if "memory_mgr" not in st.session_state:
        st.session_state["memory_mgr"] = initialize_memory_mgr()
    memory_mgr = st.session_state["memory_mgr"]

    user_id = st.session_state.get("user_id", "default")
    if "graph" not in st.session_state or st.session_state.get("_last_user") != user_id:
        graph = build_memory_agent(
            memory_mgr,
            EMBEDDING_PATH,
            user_id=user_id,
            namespace=config.namespace,
        )
        st.session_state["graph"] = graph
        st.session_state["_last_user"] = user_id

    graph = st.session_state["graph"]

    sidebar_state = render_memory_sidebar(
        memory_mgr=memory_mgr,
        graph=graph,
        namespace=config.namespace,
        welcome_message_factory=_welcome_message,
    )
    if sidebar_state.user_id != user_id:
        graph = build_memory_agent(
            memory_mgr,
            EMBEDDING_PATH,
            user_id=sidebar_state.user_id,
            namespace=config.namespace,
        )
        st.session_state["graph"] = graph
        st.session_state["_last_user"] = sidebar_state.user_id
    st.sidebar.markdown("---")
    user_id = sidebar_state.user_id

    # --- Render messages ---

    for msg in sidebar_state.messages:
        if isinstance(msg, AIMessage) and msg.content:
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    # Render tool calls in sidebar
    st.sidebar.header("Tool Calls")
    for msg in sidebar_state.messages:
        if isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

    # --- User input ---

    prompt = st.chat_input()
    if not prompt:
        return

    # Normal conversation flow
    human_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(human_msg)
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        ctx = MemoryAgentContext(
            user_id=user_id,
            namespace=config.namespace,
        )

        input_state = {
            "messages": [human_msg],
        }

        result = streamlit_invoke(
            graph,
            callables=[st_callback],
            thread_id=sidebar_state.thread_id,
            context=ctx,
            input_state=input_state,
        )

        if result and "messages" in result:
            # Replace UI messages with the checkpointed thread state.
            st.session_state.messages = result["messages"]
            st.session_state["summary"] = result.get("summary", "")


if __name__ == "__main__":
    run_memory_app()
