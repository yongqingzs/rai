import json
from pathlib import Path

from langchain_core.messages import ToolMessage
from rai.messages import (
    delete_session_artifacts,
    get_stored_artifacts,
    store_artifacts,
)
from rai.messages.multimodal import ToolMultimodalMessage


def test_store_artifacts_writes_directory_record(tmp_path: Path):
    image_b64 = "iVBORw0KGgo="
    root = tmp_path / "data" / "artifacts"

    store_artifacts(
        "tool/call:1",
        [
            {
                "summary": "captured",
                "images": [],
                "raw_images": [image_b64],
                "audios": [],
            }
        ],
        db_path=root,
    )

    artifact_dir = root / "tool_call_1"
    metadata_path = artifact_dir / "metadata.json"

    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["tool_call_id"] == "tool/call:1"
    assert metadata["artifacts"][0]["raw_images"] == ["raw_images_0000.png"]
    assert (artifact_dir / "raw_images_0000.png").exists()

    restored = get_stored_artifacts("tool/call:1", db_path=root)
    assert restored[0]["summary"] == "captured"
    assert restored[0]["raw_images"] == [image_b64]


def test_store_artifacts_externalizes_audio(tmp_path: Path):
    audio_b64 = "UklGRg=="
    root = tmp_path / "artifacts"

    store_artifacts(
        "audio-call",
        [{"summary": "audio", "audios": [audio_b64]}],
        db_path=root,
    )

    metadata = json.loads((root / "audio-call" / "metadata.json").read_text())
    assert metadata["artifacts"][0]["audios"] == ["audios_0000.bin"]
    assert get_stored_artifacts("audio-call", db_path=root)[0]["audios"] == [audio_b64]


def test_delete_session_artifacts_only_removes_owned_directories(tmp_path: Path):
    root = tmp_path / "artifacts"
    store_artifacts(
        "session-a-call",
        [{"summary": "a"}],
        db_path=root,
        thread_id="session-a",
    )
    store_artifacts(
        "session-b-call",
        [{"summary": "b"}],
        db_path=root,
        thread_id="session-b",
    )
    store_artifacts("legacy-call", [{"summary": "legacy"}], db_path=root)

    assert delete_session_artifacts("session-a", db_path=root) == 1
    assert not (root / "session-a-call").exists()
    assert (root / "session-b-call").exists()
    assert (root / "legacy-call").exists()


def test_empty_tool_images_postprocesses_to_plain_tool_message():
    msg = ToolMultimodalMessage(
        content="captured",
        name="center_gimbal_and_capture",
        tool_call_id="tool-call-1",
        images=[],
    )

    postprocessed = msg.postprocess()

    assert isinstance(postprocessed, ToolMessage)
    assert postprocessed.name == "center_gimbal_and_capture"
    assert postprocessed.tool_call_id == "tool-call-1"
    assert postprocessed.content == "captured"
