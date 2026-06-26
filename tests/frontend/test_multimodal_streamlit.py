from __future__ import annotations

import base64
from contextlib import contextmanager

import pytest
from langchain_core.messages import HumanMessage

from rai.frontend.multimodal import render_human_message, render_human_multimodal_message
from rai.messages import HumanMultimodalMessage


@contextmanager
def _chat_message_stub(_role: str):
    yield


def _make_image_b64() -> str:
    return base64.b64encode(b"fake-image-bytes").decode("utf-8")


def test_render_human_message_ignores_system_notification(monkeypatch):
    writes = []

    class _ChatMessage:
        def write(self, content):
            writes.append(content)

    monkeypatch.setattr(
        "rai.frontend.multimodal.st.chat_message",
        lambda role: _ChatMessage(),
    )

    render_human_message(
        HumanMessage(content="hidden", additional_kwargs={"system_notification": True})
    )

    assert writes == []


def test_render_human_multimodal_message_renders_text_and_images(monkeypatch):
    writes = []
    images = []

    class _ChatMessage:
        def write(self, content):
            writes.append(content)

    monkeypatch.setattr(
        "rai.frontend.multimodal.st.chat_message",
        lambda role: _ChatMessage(),
    )
    monkeypatch.setattr("rai.frontend.multimodal.st.image", lambda *args, **kwargs: images.append((args, kwargs)))
    monkeypatch.setattr(
        "rai.frontend.multimodal.cv2.imdecode",
        lambda *args, **kwargs: "decoded-image",
    )

    render_human_multimodal_message(
        HumanMultimodalMessage(content="image attached", images=[_make_image_b64()])
    )

    assert writes == ["image attached"]
    assert len(images) == 1

