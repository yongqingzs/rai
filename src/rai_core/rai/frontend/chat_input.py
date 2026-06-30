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
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage

from rai.messages import HumanMultimodalMessage

DEFAULT_IMAGE_FILE_TYPES = ("png", "jpg", "jpeg", "webp")


@dataclass(frozen=True)
class ChatInputSubmission:
    text: str
    images: list[str]
    file_names: list[str]


def _as_file_list(files: Any) -> list[Any]:
    if files is None:
        return []
    if isinstance(files, list):
        return files
    return [files]


def _read_uploaded_file(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        return uploaded_file.read()
    raise TypeError(f"Unsupported uploaded file object: {type(uploaded_file)!r}")


def _encode_uploaded_images(files: Iterable[Any]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    file_names: list[str] = []
    for uploaded_file in files:
        image_bytes = _read_uploaded_file(uploaded_file)
        if not image_bytes:
            continue
        images.append(base64.b64encode(image_bytes).decode("utf-8"))
        file_names.append(getattr(uploaded_file, "name", "uploaded_image"))
    return images, file_names


def parse_chat_input_value(value: Any) -> ChatInputSubmission | None:
    """Normalize Streamlit chat input output into text plus uploaded images."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return ChatInputSubmission(text=text, images=[], file_names=[])

    text = str(getattr(value, "text", "") or "").strip()
    files = _as_file_list(getattr(value, "files", None))
    images, file_names = _encode_uploaded_images(files)

    if not text and images:
        text = "Analyze the uploaded image."

    if not text:
        return None

    return ChatInputSubmission(text=text, images=images, file_names=file_names)


def render_multimodal_chat_input(
    placeholder: str = "Your message",
    *,
    accept_file: bool | str = True,
    file_type: Sequence[str] | None = DEFAULT_IMAGE_FILE_TYPES,
    key: str | None = None,
) -> ChatInputSubmission | None:
    """Render a Streamlit chat input that can attach image files."""
    value = st.chat_input(
        placeholder,
        key=key,
        accept_file=accept_file,
        file_type=file_type,
    )
    return parse_chat_input_value(value)


def make_transient_user_message(
    text: str,
    images: Sequence[str] | None,
) -> HumanMessage | HumanMultimodalMessage:
    """Create the per-invocation user message, keeping persisted text separate."""
    if images:
        return HumanMultimodalMessage(content=text, images=list(images))
    return HumanMessage(content=text)


def _message_text(message: HumanMessage) -> str:
    text = message.text
    if isinstance(text, str):
        return text
    return str(text)


def replace_latest_user_message_with_transient_images(
    messages: Sequence[BaseMessage],
    images: Sequence[str] | None,
) -> list[BaseMessage]:
    """Return a copy where only the latest human message is temporarily multimodal."""
    if not images:
        return list(messages)

    replaced = list(messages)
    for index in range(len(replaced) - 1, -1, -1):
        msg = replaced[index]
        if isinstance(msg, HumanMessage):
            replaced[index] = HumanMultimodalMessage(
                content=_message_text(msg),
                images=list(images),
            )
            break
    return replaced
