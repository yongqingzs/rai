from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage, ToolMessage

from rai.frontend.multimodal import (
    collect_multimodal_tool_images,
    render_human_message,
    render_tool_message_with_images,
)
from rai.messages import HumanMultimodalMessage


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


def test_collect_multimodal_tool_images_groups_images_by_tool_call_id():
    messages = [
        HumanMultimodalMessage(
            content="image attached",
            images=[_make_image_b64()],
            tool_call_id="call-1",
        ),
        HumanMultimodalMessage(
            content="another image",
            images=[_make_image_b64()],
            tool_call_id="call-1",
        ),
    ]

    images = collect_multimodal_tool_images(messages)

    assert list(images.keys()) == ["call-1"]
    assert len(images["call-1"]) == 2


def test_render_tool_message_with_images_renders_tool_output_and_images(monkeypatch):
    codes = []
    images = []

    class _ChatMessage:
        def write(self, content):
            writes.append(content)

    class _Expander:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _ChatMessageContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def expander(self, *args, **kwargs):
            return _Expander()

    monkeypatch.setattr("rai.frontend.multimodal.st.chat_message", lambda role: _ChatMessageContext())
    monkeypatch.setattr(
        "rai.frontend.multimodal.st.code",
        lambda content, language=None: codes.append((content, language)),
    )
    monkeypatch.setattr("rai.frontend.multimodal.st.image", lambda *args, **kwargs: images.append((args, kwargs)))
    monkeypatch.setattr(
        "rai.frontend.multimodal.cv2.imdecode",
        lambda *args, **kwargs: "decoded-image",
    )

    render_tool_message_with_images(
        ToolMessage(content="tool result", tool_call_id="call-1", name="center_gimbal_and_capture"),
        images=[_make_image_b64()],
    )

    assert codes == [("tool result", "json")]
    assert len(images) == 1
