import json
from pathlib import Path

from langchain_core.messages import ToolMessage

from rai.messages import get_stored_artifacts, store_artifacts
from rai.messages.multimodal import ToolMultimodalMessage


def test_store_artifacts_writes_directory_record(tmp_path: Path):
    image_b64 = "iVBORw0KGgo="
    root = tmp_path / "data" / "artifacts"

    store_artifacts(
        "tool/call:1",
        [{"summary": "captured", "images": [], "raw_images": [image_b64], "audios": []}],
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
