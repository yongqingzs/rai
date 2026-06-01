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
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import tomli
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from rai import get_embeddings_model, get_llm_model
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.frontend.memory_streamlit import (
    render_chat_messages_with_tools,
    render_memory_sidebar,
)
from rai.memory import (
    MemoryAgentContext,
    MemoryManager,
    create_memory_react_agent,
    load_memory_config,
)
from rai.memory.long_term import render_long_term_memories
from rai.tools.memory import create_memory_tools
from rai.tools.time import WaitForSecondsTool

from rai_whoami import EmbodimentSource
from rai_whoami.tools import QueryDatabaseTool
from rai_whoami.vector_db import FAISSBuilder

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
- When the user asks to forget something, use forget_memory

## Robot Documentation Retrieval
If the query_robot_docs tool is available, use it for questions about the robot's static documentation: hardware specifications, sensors, capabilities, URDF details, manuals, or operating limits. Do not use it for user preferences, learned facts, remembered locations, or conversation memory."""


@dataclass
class RobotDocsConfig:
    enabled: bool = False
    root_dir: str = ""
    build_vector_db: bool = False
    k: int = 4


class RobotDocsQueryTool(QueryDatabaseTool):
    name: str = "query_robot_docs"
    description: str = (
        "Rag 向量库查询工具。"
        "Search the robot's static whoami documentation, including hardware "
        "specs, sensors, capabilities, URDF/documentation details, and operating "
        "limits. Use this for robot documentation questions, not for user "
        "preferences, conversation memory, or learned locations."
    )


def load_robot_docs_config(config_path: str = "config.toml") -> RobotDocsConfig:
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    return RobotDocsConfig(**config_dict.get("whoami", {}))


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


def _has_whoami_vector_db(root_dir: Path) -> bool:
    generated_dir = root_dir / "generated"
    return (
        (generated_dir / "index.faiss").exists()
        and (generated_dir / "index.pkl").exists()
        and (generated_dir / "vdb_kwargs.json").exists()
    )


def _create_robot_docs_tool(
    config: RobotDocsConfig,
    embeddings_model=None,
) -> BaseTool | None:
    if not config.enabled:
        return None

    if not config.root_dir:
        raise ValueError("[whoami] root_dir must be set when enabled = true")

    root_dir = Path(config.root_dir)
    if config.build_vector_db:
        source = EmbodimentSource.from_directory(root_dir)
        FAISSBuilder(root_dir / "generated", embedding=embeddings_model).build(source)

    if not _has_whoami_vector_db(root_dir):
        raise FileNotFoundError(
            "Whoami vector DB not found. Expected generated/index.faiss, "
            "generated/index.pkl, and generated/vdb_kwargs.json under "
            f"{root_dir}. Build it with `build-whoami {root_dir} --build-vector-db` "
            "or set [whoami] build_vector_db = true."
        )

    return RobotDocsQueryTool(
        root_dir=str(root_dir),
        embeddings_model=embeddings_model,
        k=config.k,
    )


def build_memory_agent(
    memory_mgr: MemoryManager,
    embodiment_path: Path,
    user_id: str = "default",
    namespace: str = "default",
    robot_docs_config: RobotDocsConfig | None = None,
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
    robot_docs_config = robot_docs_config or load_robot_docs_config()
    robot_docs_tool = _create_robot_docs_tool(robot_docs_config, embeddings_model)
    if robot_docs_tool is not None:
        all_tools.append(robot_docs_tool)

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
    robot_docs_config = load_robot_docs_config()
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
            st.rerun()


if __name__ == "__main__":
    run_memory_app()
