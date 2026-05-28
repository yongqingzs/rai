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

from typing import List, Optional

from langchain_core.runnables import RunnableConfig

from rai.memory.manager import MemoryManager


def get_session_ids(memory_mgr: MemoryManager, limit: int = 200) -> List[str]:
    """Get unique thread IDs from the checkpointer."""
    thread_ids = set()
    for checkpoint in memory_mgr.checkpointer.list(None, limit=limit):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            thread_ids.add(thread_id)
    return sorted(thread_ids)


def get_latest_session_id(memory_mgr: MemoryManager, limit: int = 200) -> Optional[str]:
    """Get the latest thread ID reported by the checkpointer."""
    for checkpoint in memory_mgr.checkpointer.list(None, limit=limit):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            return thread_id
    return None


def delete_session(memory_mgr: MemoryManager, thread_id: str):
    """Delete a thread from the checkpointer."""
    memory_mgr.checkpointer.delete_thread(thread_id)


def graph_config(thread_id: str) -> RunnableConfig:
    return RunnableConfig({"configurable": {"thread_id": thread_id}})


def load_thread_state(graph, thread_id: str) -> tuple[list, str]:
    """Load checkpointed messages and summary for a graph thread."""
    snapshot = graph.get_state(graph_config(thread_id))
    values = snapshot.values or {}
    return list(values.get("messages", [])), values.get("summary", "")
