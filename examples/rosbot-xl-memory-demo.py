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
from langchain_core.messages import AIMessage, ToolMessage
from rai import get_embeddings_model, get_llm_model
from rai.frontend.memory_streamlit import (
    render_chat_messages_with_tools,
    render_memory_chat_input,
    render_memory_sidebar,
)
from rai.memory import (
    MemoryManager,
    create_memory_agent_with_tools,
    load_memory_config,
)
from rai.tools.time import WaitForSecondsTool

from rai_whoami import WhoamiConfig, create_robot_docs_tool, load_whoami_config

# --- Constants ---

EMBEDDING_PATH = Path(__file__).parent / "embodiments" / "rosbotxl_embodiment1.json"

BASE_SYSTEM_PROMPT_TEMPLATE = """You are a ROSBot XL robot assistant. You can move around, look at objects, and help with tasks.

## System Memory (Embodiment)
{embodiment}"""

ROBOT_DOCS_PROMPT_SECTION = """## Robot Documentation Retrieval
If the query_robot_docs tool is available, use it for questions about the robot's static documentation: hardware specifications, sensors, capabilities, URDF details, manuals, or operating limits. Do not use it for user preferences, learned facts, remembered locations, or conversation memory."""


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
    robot_docs_config: WhoamiConfig | None = None,
    embeddings_model=None,
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
    robot_docs_config = robot_docs_config or load_whoami_config()
    robot_docs_tool = create_robot_docs_tool(robot_docs_config, embeddings_model)

    def build_base_system_prompt(_context) -> str:
        return BASE_SYSTEM_PROMPT_TEMPLATE.format(embodiment=embodiment_text)

    return create_memory_agent_with_tools(
        memory_mgr=memory_mgr,
        llm=llm,
        base_system_prompt_builder=build_base_system_prompt,
        namespace=namespace,
        user_id=user_id,
        base_tools=[WaitForSecondsTool()],
        extra_tools=[robot_docs_tool],
        extra_prompt_sections=[ROBOT_DOCS_PROMPT_SECTION] if robot_docs_tool else None,
    )


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
    robot_docs_config = load_whoami_config()
    st.sidebar.markdown(
        f"**Backend:** `{config.backend}`\n**Namespace:** `{config.namespace}`"
    )
    if robot_docs_config.enabled:
        st.sidebar.markdown(f"**Robot Docs:** `{robot_docs_config.root_dir}`")
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
            robot_docs_config=robot_docs_config,
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
            robot_docs_config=robot_docs_config,
        )
        st.session_state["graph"] = graph
        st.session_state["_last_user"] = sidebar_state.user_id
    st.sidebar.markdown("---")
    user_id = sidebar_state.user_id

    # --- Render messages ---

    render_chat_messages_with_tools(sidebar_state.messages)

    # Render tool calls in sidebar
    st.sidebar.header("Tool Calls")
    for msg in sidebar_state.messages:
        if isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

    render_memory_chat_input(graph, sidebar_state, config.namespace)


if __name__ == "__main__":
    run_memory_app()
