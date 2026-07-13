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

import asyncio

from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError
from rai.tools.memory import create_memory_tools


class _Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class _AsyncOnlyStore:
    def __init__(self):
        self.values = {}
        self._loop = None

    def put(self, *args, **kwargs):
        raise RuntimeError("Synchronous calls to async store detected")

    def search(self, *args, **kwargs):
        raise RuntimeError("Synchronous calls to async store detected")

    def delete(self, *args, **kwargs):
        raise RuntimeError("Synchronous calls to async store detected")

    async def aput(self, namespace, key, value):
        self.values[(namespace, key)] = value

    async def asearch(self, namespace, query=None, limit=10):
        items = [
            _Item(key, value)
            for (ns, key), value in self.values.items()
            if ns == namespace and (not query or query.lower() in str(value).lower())
        ]
        return items[:limit]

    async def adelete(self, namespace, key):
        self.values.pop((namespace, key), None)


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
            "pose": {"x": -0.2175, "y": -0.8775, "z": 0.0, "yaw": 1.57},
        }
    )
    assert "Kitchen" in result_structured
    assert "yaw=1.5700" in result_structured

    result_json = tools["save_location"].invoke(
        {
            "location_name": "Living Room",
            "pose": '{"x": -0.82, "y": 3.525, "z": 0.0, "yaw": -0.5}',
        }
    )
    assert "Living Room" in result_json

    recall_result = tools["recall"].invoke(
        {"query": "Living Room", "memory_type": "spatial"}
    )
    assert "Living Room" in recall_result
    assert '"x": -0.82' in recall_result
    assert '"yaw": -0.5' in recall_result


def test_save_location_accepts_objects_as_json_string():
    store = InMemoryStore()
    tools = create_memory_tools(store=store, namespace="test", user_id="alice")

    result_empty = tools["save_location"].invoke(
        {
            "location_name": "Toilet",
            "pose": '{"x": 0, "y": 0, "z": 3.0, "yaw": 0.0}',
            "objects": "[]",
            "description": "The toilet located at the specified coordinates.",
        }
    )
    assert "Toilet" in result_empty

    result_objects = tools["save_location"].invoke(
        {
            "location_name": "Bathroom",
            "objects": '["sink", "door"]',
        }
    )
    assert "Bathroom" in result_objects

    recall_result = tools["recall"].invoke(
        {"query": "Bathroom", "memory_type": "spatial"}
    )
    assert "Bathroom" in recall_result
    assert "sink" in recall_result


def test_save_location_rejects_objects_json_object():
    store = InMemoryStore()
    tools = create_memory_tools(store=store, namespace="test", user_id="alice")

    try:
        tools["save_location"].invoke(
            {
                "location_name": "Invalid",
                "objects": '{"name": "sink"}',
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected objects JSON object to fail validation")


def test_memory_tools_use_async_store_methods_in_arun():
    async def run_test():
        store = _AsyncOnlyStore()
        tools = create_memory_tools(store=store, namespace="test", user_id="alice")

        fact_result = await tools["save_fact"].ainvoke(
            {"fact": "The user prefers inspection reports in Chinese."}
        )
        assert "Fact saved" in fact_result

        location_result = await tools["save_location"].ainvoke(
            {
                "location_name": "Tunnel Entrance",
                "pose": {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.25},
            }
        )
        assert "Tunnel Entrance" in location_result
        assert "yaw=0.2500" in location_result

        recall_result = await tools["recall"].ainvoke(
            {"query": "Tunnel", "memory_type": "spatial"}
        )
        assert "Tunnel Entrance" in recall_result

        delete_result = await tools["forget"].ainvoke({"query": "Chinese"})
        assert "Deleted 1 memories" in delete_result

    asyncio.run(run_test())
