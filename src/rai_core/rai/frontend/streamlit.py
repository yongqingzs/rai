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

from typing import Any, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable

from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.frontend.multimodal import (
    collect_multimodal_tool_images,
    render_human_message,
    render_image_list,
)
from rai.messages import HumanMultimodalMessage


def run_streamlit_app(
    agent: Runnable[Any, Any],
    page_title: str,
    initial_message: str,
    thread_id: Optional[str] = None,
):
    st.title(page_title)
    st.markdown("---")

    st.sidebar.header("Tool Calls History")

    if "graph" not in st.session_state:
        st.session_state["graph"] = agent

    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content=initial_message)]

    prompt = st.chat_input()
    multimodal_tool_images = collect_multimodal_tool_images(st.session_state.messages)
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMultimodalMessage):
            continue
        elif isinstance(msg, HumanMessage):
            render_human_message(msg)
        elif isinstance(msg, ToolMessage):
            with st.chat_message("assistant"):
                with st.expander(f"Tool: {msg.name}", expanded=False):
                    st.code(msg.content, language="json")
                    render_image_list(multimodal_tool_images.get(msg.tool_call_id))

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = get_streamlit_cb(st.container())
            streamlit_invoke(
                st.session_state["graph"],
                st.session_state.messages,
                [st_callback],
                thread_id=thread_id,
            )
