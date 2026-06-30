from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from rai.frontend.chat_input import (
    parse_chat_input_value,
    replace_latest_user_message_with_transient_images,
)
from rai.messages import HumanMultimodalMessage


@dataclass
class _UploadedFile:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


@dataclass
class _ChatInputValue:
    text: str
    files: list[_UploadedFile]


def test_parse_chat_input_value_encodes_uploaded_images():
    submission = parse_chat_input_value(
        _ChatInputValue(
            text="Describe this",
            files=[_UploadedFile(name="panel.png", content=b"image-bytes")],
        )
    )

    assert submission is not None
    assert submission.text == "Describe this"
    assert submission.file_names == ["panel.png"]
    assert submission.images == ["aW1hZ2UtYnl0ZXM="]


def test_parse_chat_input_value_defaults_text_for_image_only_submission():
    submission = parse_chat_input_value(
        _ChatInputValue(
            text="",
            files=[_UploadedFile(name="panel.png", content=b"image-bytes")],
        )
    )

    assert submission is not None
    assert submission.text == "Analyze the uploaded image."
    assert submission.images == ["aW1hZ2UtYnl0ZXM="]


def test_replace_latest_user_message_with_transient_images_keeps_history_copy_plain():
    messages = [
        AIMessage(content="hello"),
        HumanMessage(content="Describe this"),
    ]

    invoke_messages = replace_latest_user_message_with_transient_images(
        messages,
        ["aW1hZ2UtYnl0ZXM="],
    )

    assert isinstance(messages[-1], HumanMessage)
    assert not isinstance(messages[-1], HumanMultimodalMessage)
    assert isinstance(invoke_messages[-1], HumanMultimodalMessage)
    assert invoke_messages[-1].images == ["aW1hZ2UtYnl0ZXM="]
