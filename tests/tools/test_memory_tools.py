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
