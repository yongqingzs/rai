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

import base64

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from rai.agents.langchain.core import ToolRunner
from rai.messages import get_stored_artifacts


def test_tool_runner_externalizes_raw_images_before_checkpoint(tmp_path, monkeypatch):
    image_b64 = base64.b64encode(b"image-bytes" * 10_000).decode()

    @tool(response_format="content_and_artifact")
    def capture_raw_image():
        """Capture a raw image for later analysis."""
        return "captured", {
            "raw_images": [image_b64],
            "summary": "one image",
        }

    monkeypatch.chdir(tmp_path)
    runner = ToolRunner(tools=[capture_raw_image])
    call = ToolCall(name="capture_raw_image", args={}, id="capture-1")
    state = {"messages": [AIMessage(content="", tool_calls=[call])]}

    output = runner.invoke(state)

    result = output["messages"][-1]
    assert isinstance(result, ToolMessage)
    assert result.artifact == {
        "artifact_id": "capture-1",
        "summary": "one image",
        "images": 0,
        "raw_images": 1,
        "audios": 0,
    }
    serialized = JsonPlusSerializer().dumps_typed(output)[1]
    assert image_b64.encode() not in serialized
    assert len(serialized) < 10_000
    assert get_stored_artifacts("capture-1")[0]["raw_images"] == [image_b64]
