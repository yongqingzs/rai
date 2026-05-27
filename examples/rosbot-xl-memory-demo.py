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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from rai import get_embeddings_model, get_llm_model
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.agents.langchain.core.react_agent import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_TOKEN_THRESHOLD,
    create_react_runnable,
    summarize_messages,
)
from rai.memory import MemoryManager, load_memory_config
from rai.tools.memory import MemoryTools, create_memory_tools
from rai.tools.time import WaitForSecondsTool
from typing_extensions import TypedDict

# --- Constants ---

EMBEDDING_PATH = Path(__file__).parent / "embodiments" / "rosbotxl_embodiment1.json"
MAX_LONG_TERM_FACTS = 20
MAX_LONG_TERM_SPATIAL = 20
MAX_LONG_TERM_CHARS = 8000

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


# --- State & Context ---


@dataclass
class AgentContext:
    user_id: str
    namespace: str


class MemoryState(TypedDict):
    messages: List[AIMessage | HumanMessage | ToolMessage | SystemMessage]
    summary: str


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


def _load_all_long_term_memories(
    store,
    namespace: str,
    user_id: str,
) -> str:
    """Load all stored facts and spatial data for system prompt injection."""
    all_memories = []
    limits = {"facts": MAX_LONG_TERM_FACTS, "spatial": MAX_LONG_TERM_SPATIAL}
    for schema in ("facts", "spatial"):
        ns = (namespace, user_id, schema)
        try:
            items = store.search(ns, query="", limit=limits[schema])
            for item in items:
                if schema == "facts":
                    all_memories.append(f"- {item.value.get('text', str(item.value))}")
                else:
                    loc = item.value.get("location", "unknown")
                    pose = item.value.get("pose")
                    objects = item.value.get("objects", [])
                    desc = item.value.get("description", "")
                    line = f"- {loc}"
                    if pose:
                        line += (
                            f" ({pose.get('x', '?')}, {pose.get('y', '?')}, "
                            f"{pose.get('z', '?')})"
                        )
                    if desc:
                        line += f" — {desc}"
                    if objects:
                        line += f" [{', '.join(objects)}]"
                    all_memories.append(line)
        except Exception:
            pass

    if not all_memories:
        return "none yet"
    rendered = "\n".join(all_memories)
    if len(rendered) > MAX_LONG_TERM_CHARS:
        rendered = rendered[:MAX_LONG_TERM_CHARS].rstrip()
        rendered += "\n..."
    return rendered


def build_memory_agent(
    memory_mgr: MemoryManager,
    embodiment_path: Path,
    user_id: str = "default",
    namespace: str = "default",
    token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[Runnable[MemoryState, MemoryState], MemoryTools, str]:
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
    memory_tools = MemoryTools(
        tools=memory_tools_dict,
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

    # Inner ReAct agent (handles tool use + reasoning)
    inner_agent = create_react_runnable(
        llm=llm,
        tools=all_tools,
        checkpointer=memory_mgr.checkpointer,
        store=memory_mgr.store,
    )

    # --- Graph nodes ---

    def enrich_prompt(state: MemoryState, runtime: Runtime[AgentContext]):
        """Load all LTM and inject system prompt with embodiment + memories."""
        ltm_text = _load_all_long_term_memories(
            memory_mgr.store,
            runtime.context.namespace,
            runtime.context.user_id,
        )
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            embodiment=embodiment_text,
            long_term_memory=ltm_text,
        )
        non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "messages": [SystemMessage(content=system_content)] + non_system,
            "summary": state.get("summary", ""),
        }

    def summarize_node(state: MemoryState, runtime: Runtime[AgentContext]):
        """Compress conversation when tokens exceed threshold."""
        llm_simple = get_llm_model("simple_model", streaming=False)
        result = summarize_messages(
            messages=state["messages"],
            existing_summary=state.get("summary", ""),
            llm=llm_simple,
            threshold=token_threshold,
            keep_recent=keep_recent,
        )
        return result

    def run_react(
        state: MemoryState, runtime: Runtime[AgentContext], config: RunnableConfig
    ):
        """Run the inner ReAct agent and return full updated state."""
        ctx = AgentContext(
            user_id=runtime.context.user_id,
            namespace=runtime.context.namespace,
        )
        configurable = config.get("configurable", {})
        try:
            result = inner_agent.invoke(
                {"messages": state["messages"]},
                RunnableConfig(
                    callbacks=config.get("callbacks", []),
                    configurable=configurable,
                ),
                context=ctx,
            )
            # Return full message list (replace reducer in outer graph)
            return {"messages": result["messages"], "summary": state.get("summary", "")}
        except Exception as e:
            error_msg = AIMessage(content=f"Agent error: {e}")
            return {
                "messages": list(state["messages"]) + [error_msg],
                "summary": state.get("summary", ""),
            }

    # --- Build graph ---

    builder = StateGraph(MemoryState, context_schema=AgentContext)
    builder.add_node("enrich_prompt", enrich_prompt)
    builder.add_node("summarize", summarize_node)
    builder.add_node("react", run_react)

    builder.add_edge(START, "enrich_prompt")
    builder.add_edge("enrich_prompt", "summarize")
    builder.add_edge("summarize", "react")
    builder.add_edge("react", END)

    graph = builder.compile(
        checkpointer=memory_mgr.checkpointer,
        store=memory_mgr.store,
    )
    return graph, memory_tools, embodiment_text


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


def _get_session_ids(memory_mgr: MemoryManager) -> List[str]:
    """Get unique thread_ids from the checkpointer."""
    thread_ids = set()
    for cp in memory_mgr.checkpointer.list(None, limit=200):
        tid = cp.config.get("configurable", {}).get("thread_id")
        if tid:
            thread_ids.add(tid)
    return sorted(thread_ids)


def _get_user_ids(memory_mgr: MemoryManager, namespace: str) -> List[str]:
    """Get unique user_ids from store namespaces."""
    user_ids = set()
    try:
        namespaces = memory_mgr.store.list_namespaces(prefix=(namespace,), limit=200)
        for ns in namespaces:
            if len(ns) >= 2:
                user_ids.add(ns[1])
    except Exception:
        pass
    return sorted(user_ids)


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

    # --- Sidebar: Session and User selection ---

    st.sidebar.subheader("Short-Term Memory")
    session_ids = _get_session_ids(memory_mgr)
    if not session_ids:
        st.sidebar.info("No sessions yet. Start a conversation.")
    selected_session = st.sidebar.selectbox(
        "Session (Thread)",
        options=session_ids if session_ids else ["(new session)"],
        index=0 if not session_ids else min(0, len(session_ids)),
        help="Different sessions keep separate conversation history.",
    )
    if st.sidebar.button("+ New Session"):
        st.session_state.thread_id = f"session-{int(time.time())}"
        st.rerun()

    st.sidebar.subheader("Long-Term Memory")
    user_ids = _get_user_ids(memory_mgr, config.namespace)
    if not user_ids:
        st.sidebar.info("No users with long-term memories yet.")
    selected_user = st.sidebar.selectbox(
        "User",
        options=user_ids if user_ids else ["(default)"],
        index=0,
        help="Long-term facts are scoped to the selected user.",
    )

    st.session_state.thread_id = (
        selected_session
        if selected_session != "(new session)"
        else st.session_state.get("thread_id", f"session-{int(time.time())}")
    )
    st.session_state.user_id = (
        selected_user if selected_user != "(default)" else "default"
    )

    # Clear messages when switching sessions
    current_thread = st.session_state.get("_last_thread")
    if current_thread and current_thread != st.session_state.thread_id:
        st.session_state["messages"] = [
            AIMessage(
                content="New session started. Previous conversation stored in memory."
            )
        ]
    st.session_state["_last_thread"] = st.session_state.thread_id

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Short-term**: Conversation history within the selected session.
    Compressed via summarization when tokens exceed threshold.

    **Long-term**: Agent-driven memory tools (facts + spatial).
    Persists across sessions, scoped to the selected user.

    **Forget**: Agent deletes matching memories directly.
    """)

    if st.sidebar.button("Clear Long-Term Memory"):
        _clear_long_term_memory(config, st.session_state.user_id)
        st.rerun()

    # --- Initialize agent graph ---

    user_id = st.session_state.user_id
    if "graph" not in st.session_state or st.session_state.get("_last_user") != user_id:
        graph, memory_tools, embodiment_text = build_memory_agent(
            memory_mgr,
            EMBEDDING_PATH,
            user_id=user_id,
            namespace=config.namespace,
        )
        st.session_state["graph"] = graph
        st.session_state["memory_tools"] = memory_tools
        st.session_state["_last_user"] = user_id
        st.session_state["messages"] = [
            AIMessage(
                content=(
                    "Hi! I'm a robot with persistent memory. "
                    "Our conversation is summarized when it gets long (short-term). "
                    "I'll proactively save important facts and locations to long-term memory. "
                    "Ask me to forget something and I'll delete matching memories directly."
                )
            )
        ]
        st.session_state["summary"] = ""

    graph = st.session_state["graph"]
    memory_tools = st.session_state["memory_tools"]

    # --- Render messages ---

    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage) and msg.content:
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    # Render tool calls in sidebar
    st.sidebar.header("Tool Calls")
    for msg in st.session_state.messages:
        if isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

    # --- User input ---

    prompt = st.chat_input()
    if not prompt:
        return

    # Normal conversation flow
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        ctx = AgentContext(
            user_id=user_id,
            namespace=config.namespace,
        )

        input_state = {
            "messages": st.session_state.messages,
            "summary": st.session_state.get("summary", ""),
        }

        result = streamlit_invoke(
            graph,
            callables=[st_callback],
            thread_id=st.session_state.thread_id,
            context=ctx,
            input_state=input_state,
        )

        if result and "messages" in result:
            # Replace full message list (graph may compress/enrich)
            st.session_state.messages = result["messages"]

            # Update summary if provided
            if "summary" in result:
                st.session_state["summary"] = result["summary"]


def _clear_long_term_memory(config, user_id: str):
    """Clear all long-term memories for the current user."""
    memory_mgr = MemoryManager(config=config)
    memory_mgr.start()
    ns_base = (config.namespace, user_id)
    for schema in ("facts", "spatial"):
        ns = (*ns_base, schema)
        try:
            items = memory_mgr.store.search(ns, query="", limit=1000)
            for item in items:
                memory_mgr.store.delete(ns, item.key)
        except Exception:
            pass
    memory_mgr.stop()
    st.success("Long-term memory cleared!")


if __name__ == "__main__":
    run_memory_app()
