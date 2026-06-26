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

from __future__ import annotations

import base64
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

from rai.messages import HumanMultimodalMessage


def _render_image(image_b64: str) -> None:
    image_cv2 = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_b64), np.uint8),
        cv2.IMREAD_COLOR,
    )
    st.image(image_cv2, channels="BGR", use_container_width=True)


def render_human_multimodal_message(msg: HumanMultimodalMessage) -> None:
    content_parts = []
    if isinstance(msg.content, list):
        content_parts = [
            part.get("text", "")
            for part in msg.content
            if isinstance(part, dict) and part.get("type") == "text"
        ]

    if content_parts:
        st.chat_message("user").write("\n".join(filter(None, content_parts)))
    else:
        st.chat_message("user").write("")

    if isinstance(msg.images, list):
        for image in msg.images:
            _render_image(image)


def render_human_message(msg: HumanMessage) -> None:
    if msg.additional_kwargs.get("system_notification"):
        return
    st.chat_message("user").write(msg.content)


def collect_multimodal_tool_images(messages: list[Any]) -> dict[str, list[str]]:
    images_by_tool_call_id: dict[str, list[str]] = defaultdict(list)
    for msg in messages:
        if isinstance(msg, HumanMultimodalMessage) and getattr(msg, "tool_call_id", None):
            if isinstance(msg.images, list):
                images_by_tool_call_id[msg.tool_call_id].extend(msg.images)
    return dict(images_by_tool_call_id)


def render_tool_message_with_images(
    msg: ToolMessage,
    images: list[str] | None = None,
    *,
    expand_label: str | None = None,
) -> None:
    with st.chat_message("assistant"):
        with st.expander(expand_label or f"Tool: {msg.name}", expanded=False):
            st.caption("Output")
            st.code(msg.content, language="json")
            render_image_list(images)


def render_image_list(images: list[str] | None) -> None:
    if images:
        for image in images:
            _render_image(image)
