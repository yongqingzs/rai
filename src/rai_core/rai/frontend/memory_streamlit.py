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

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.frontend.chat_input import render_multimodal_chat_input
from rai.frontend.multimodal import (
    collect_multimodal_tool_images,
    render_human_message,
    render_image_list,
)
from rai.memory.graph import MemoryAgentContext
from rai.memory.long_term import format_long_term_item, list_long_term_memory_items
from rai.memory.manager import MemoryManager
from rai.memory.session import (
    delete_session,
    get_latest_session_id,
    get_session_ids,
    graph_config,
    load_thread_state,
)
from rai.memory.users import add_user_profile, delete_user, get_user_ids
from rai.messages import HumanMultimodalMessage


@dataclass
class MemorySidebarState:
    thread_id: str
    user_id: str
    messages: list
    summary: str


@dataclass
class ToolCallRenderEntry:
    tool_call_id: str
    name: str
    args: Any
    output: ToolMessage | None = None


def default_welcome_message() -> AIMessage:
    return AIMessage(content="New conversation started.")


def collect_tool_call_entries(messages: list) -> dict[str, list[ToolCallRenderEntry]]:
    """Group persisted tool calls with their matching ToolMessage outputs."""
    outputs_by_id = {
        msg.tool_call_id: msg for msg in messages if isinstance(msg, ToolMessage)
    }
    entries_by_ai_message_id = {}
    for index, msg in enumerate(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue

        message_id = msg.id or f"ai-message-{index}"
        entries_by_ai_message_id[message_id] = [
            ToolCallRenderEntry(
                tool_call_id=tool_call.get("id", ""),
                name=tool_call.get("name", "tool"),
                args=tool_call.get("args", {}),
                output=outputs_by_id.get(tool_call.get("id", "")),
            )
            for tool_call in msg.tool_calls
        ]
    return entries_by_ai_message_id


def _format_tool_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, indent=2, ensure_ascii=False)


def render_chat_messages_with_tools(messages: list):
    """Render checkpointed chat messages, including recoverable tool call details."""
    tool_entries = collect_tool_call_entries(messages)
    multimodal_tool_images = collect_multimodal_tool_images(messages)
    for index, msg in enumerate(messages):
        if isinstance(msg, HumanMultimodalMessage):
            continue

        if isinstance(msg, HumanMessage):
            render_human_message(msg)
            continue

        if isinstance(msg, AIMessage):
            message_id = msg.id or f"ai-message-{index}"
            entries = tool_entries.get(message_id, [])
            if msg.content or entries:
                with st.chat_message("assistant"):
                    if msg.content:
                        st.write(msg.content)
                    for entry in entries:
                        with st.expander(f"Tool: {entry.name}", expanded=False):
                            st.caption("Input")
                            st.code(_format_tool_args(entry.args), language="json")
                            if entry.output is not None:
                                st.caption("Output")
                                st.code(entry.output.content, language="json")
                            images = multimodal_tool_images.get(entry.tool_call_id)
                            render_image_list(images)
            continue

        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in {
                entry.tool_call_id
                for entries in tool_entries.values()
                for entry in entries
            }:
                with st.chat_message("assistant"):
                    with st.expander(f"Tool: {msg.name}", expanded=False):
                        st.caption("Output")
                        st.code(msg.content, language="json")
                        render_image_list(multimodal_tool_images.get(msg.tool_call_id))


def render_memory_sidebar(
    memory_mgr: MemoryManager,
    graph,
    namespace: str,
    welcome_message_factory: Callable[[], AIMessage] = default_welcome_message,
) -> MemorySidebarState:
    """Render reusable Streamlit controls for memory session/user management."""
    st.sidebar.subheader("Short-Term Memory")
    session_ids = get_session_ids(memory_mgr)
    latest_session_id = get_latest_session_id(memory_mgr)
    if not session_ids:
        st.sidebar.info("No sessions yet. Start a conversation.")

    forced_thread_id = st.session_state.pop("_new_thread_id", None)
    current_thread_id = st.session_state.get("thread_id") or latest_session_id
    selected_thread_id = forced_thread_id or current_thread_id
    session_options = ["(new session)"] + session_ids
    if selected_thread_id and selected_thread_id not in session_options:
        session_options.insert(1, selected_thread_id)
    default_index = (
        session_options.index(selected_thread_id)
        if selected_thread_id in session_options
        else 0
    )
    selected_session = st.sidebar.selectbox(
        "Session (Thread)",
        options=session_options,
        index=default_index,
        help="Different sessions keep separate conversation history.",
    )

    if st.sidebar.button("+ New Session"):
        st.session_state["_new_thread_id"] = f"session-{int(time.time())}"
        st.rerun()

    thread_id = (
        selected_session
        if selected_session != "(new session)"
        else st.session_state.get("thread_id", f"session-{int(time.time())}")
    )
    st.session_state.thread_id = thread_id

    can_delete_session = thread_id in session_ids
    if st.sidebar.button("Delete Session", disabled=not can_delete_session):
        delete_session(memory_mgr, thread_id)
        st.session_state.pop("messages", None)
        st.session_state.pop("summary", None)
        st.session_state.pop("_last_thread", None)
        st.session_state["_new_thread_id"] = f"session-{int(time.time())}"
        st.rerun()

    st.sidebar.subheader("Long-Term Memory")
    user_ids = get_user_ids(memory_mgr, namespace)
    selected_user_id = st.session_state.pop("_selected_user_id", None)
    if selected_user_id and selected_user_id not in user_ids:
        user_ids.append(selected_user_id)
        user_ids = sorted(user_ids)
    default_user_index = (
        user_ids.index(selected_user_id)
        if selected_user_id in user_ids
        else user_ids.index(st.session_state.user_id)
        if st.session_state.get("user_id") in user_ids
        else 0
    )
    selected_user = st.sidebar.selectbox(
        "User",
        options=user_ids,
        index=default_user_index,
        help="Long-term facts are scoped to the selected user.",
    )

    with st.sidebar.popover("Add User"):
        with st.form("add_user_form", clear_on_submit=True):
            new_user_id = st.text_input("User ID")
            submitted = st.form_submit_button("Create User")
        if submitted and new_user_id.strip():
            clean_user_id = new_user_id.strip()
            add_user_profile(memory_mgr, namespace, clean_user_id)
            st.session_state["_selected_user_id"] = clean_user_id
            st.session_state.pop("messages", None)
            st.session_state.pop("summary", None)
            st.rerun()

    user_id = selected_user
    st.session_state.user_id = user_id

    long_term_items = list_long_term_memory_items(memory_mgr.store, namespace, user_id)
    if st.sidebar.button("Delete User", disabled=user_id == "default"):
        delete_user(memory_mgr, namespace, user_id)
        remaining_users = [
            uid for uid in get_user_ids(memory_mgr, namespace) if uid != user_id
        ]
        st.session_state["_selected_user_id"] = (
            "default" if "default" in remaining_users else remaining_users[0]
        )
        st.session_state.pop("messages", None)
        st.session_state.pop("summary", None)
        st.rerun()

    if not long_term_items:
        st.sidebar.info("No long-term memories for this user.")
    else:
        facts = [item for item in long_term_items if item[0] == "facts"]
        spatial = [item for item in long_term_items if item[0] == "spatial"]
        for label, group in (("Facts", facts), ("Locations", spatial)):
            if not group:
                continue
            with st.sidebar.expander(f"{label} ({len(group)})", expanded=False):
                for schema, ns, key, value in group:
                    st.caption(format_long_term_item(schema, key, value))
                    if st.button("Delete", key=f"delete_ltm_{schema}_{key}"):
                        memory_mgr.store.delete(ns, key)
                        if schema == "facts":
                            name = value.get("text", key)
                            if len(name) > 40:
                                name = name[:40] + "..."
                            item_desc = f"fact '{name}'"
                        else:
                            name = value.get("location", key)
                            item_desc = f"location '{name}'"
                        system_msg = HumanMessage(
                            content=f"[System Notification: The user deleted the long-term {item_desc} from the database via the UI. Please treat it as deleted/forgotten and do not mention or refer to it anymore.]",
                            additional_kwargs={"system_notification": True},
                        )
                        graph.update_state(
                            graph_config(thread_id), {"messages": [system_msg]}
                        )
                        if "messages" in st.session_state:
                            st.session_state.messages.append(system_msg)
                        st.rerun()

    current_thread = st.session_state.get("_last_thread")
    if (
        "messages" not in st.session_state
        or current_thread != thread_id
        or st.session_state.get("_messages_user") != user_id
    ):
        restored_messages, restored_summary = load_thread_state(graph, thread_id)
        st.session_state["messages"] = restored_messages or [welcome_message_factory()]
        st.session_state["summary"] = restored_summary
        st.session_state["_last_thread"] = thread_id
        st.session_state["_messages_user"] = user_id

    return MemorySidebarState(
        thread_id=thread_id,
        user_id=user_id,
        messages=st.session_state["messages"],
        summary=st.session_state.get("summary", ""),
    )


def render_memory_chat_input(
    graph,
    sidebar_state: MemorySidebarState,
    namespace: str,
):
    """Render chat input and invoke a memory graph for one user turn."""
    submission = render_multimodal_chat_input()
    if not submission:
        return None

    human_msg = HumanMessage(content=submission.text)
    st.session_state.messages.append(human_msg)
    st.chat_message("user").write(submission.text)

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        context = MemoryAgentContext(
            user_id=sidebar_state.user_id,
            namespace=namespace,
            transient_images=submission.images,
        )
        result = streamlit_invoke(
            graph,
            callables=[st_callback],
            thread_id=sidebar_state.thread_id,
            context=context,
            input_state={"messages": [human_msg]},
        )

        if result and "messages" in result:
            st.session_state.messages = result["messages"]
            st.session_state["summary"] = result.get("summary", "")
            st.rerun()
        return result
