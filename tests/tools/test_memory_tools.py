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

from langgraph.store.memory import InMemoryStore

from rai.tools.memory import create_memory_tools


def test_forget_memory_deletes_matching_memories():
    store = InMemoryStore()
    tools = create_memory_tools(store=store, namespace="test", user_id="alice")

    tools["save_fact"].invoke({"fact": "The user likes green tea."})

    deletion_result = tools["forget"].invoke({"query": "green tea"})
    assert "Deleted 1 memories" in deletion_result
    assert tools["recall"].invoke({"query": "green tea", "memory_type": "facts"}) == (
        "No memories found matching 'green tea'"
    )


def test_save_location_accepts_structured_pose_and_json_string():
    store = InMemoryStore()
    tools = create_memory_tools(store=store, namespace="test", user_id="alice")

    result_structured = tools["save_location"].invoke(
        {
            "location_name": "Kitchen",
            "pose": {"x": -0.2175, "y": -0.8775, "z": 0.0},
        }
    )
    assert "Kitchen" in result_structured

    result_json = tools["save_location"].invoke(
        {
            "location_name": "Living Room",
            "pose": "{\"x\": -0.82, \"y\": 3.525, \"z\": 0.0}",
        }
    )
    assert "Living Room" in result_json

    recall_result = tools["recall"].invoke(
        {"query": "Living Room", "memory_type": "spatial"}
    )
    assert "Living Room" in recall_result
    assert '"x": -0.82' in recall_result
