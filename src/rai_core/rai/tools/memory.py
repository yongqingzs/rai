# Copyright (C) 2025 Robotec.AI
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

"""Memory tools for agent-driven long-term memory CRUD.

Provides 4 tools:
- SaveFactTool: Save a text fact to long-term memory
- SaveLocationTool: Save structured spatial/location data
- RecallMemoryTool: Search/recall stored memories
- ForgetMemoryTool: Delete matching stored memories
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

# --- Tool Input Schemas ---


class SaveFactToolInput(BaseModel):
    """Input for saving a text fact to long-term memory."""

    fact: str = Field(..., description="The fact or piece of information to remember")


class SaveLocationToolInput(BaseModel):
    """Input for saving structured spatial/location data."""

    location_name: str = Field(
        ..., description="Name of the location (e.g. 'Kitchen', 'Living Room')"
    )
    pose: Optional[dict] = Field(
        default=None,
        description="Optional pose as dict with x, y, z, and optionally roll, pitch, yaw",
    )
    objects: Optional[list[str]] = Field(
        default=None,
        description="Optional list of objects at this location",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional natural language description of the location",
    )


class RecallMemoryToolInput(BaseModel):
    """Input for searching/recalling stored memories."""

    query: str = Field(
        ...,
        description="Search query. Use keywords like 'location', 'kitchen', 'user preference'",
    )
    memory_type: Optional[str] = Field(
        default=None,
        description="Filter by type: 'facts', 'spatial', or None for both",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )


class ForgetMemoryToolInput(BaseModel):
    """Input for requesting deletion of a stored memory."""

    query: str = Field(
        ...,
        description="Description of the memory to forget. Will match against stored facts.",
    )


# --- Factory for creating memory tools ---


def create_memory_tools(
    store: BaseStore,
    namespace: str,
    user_id: str,
):
    """Create memory tools bound to a store and namespace.

    Parameters
    ----------
    store : BaseStore
        LangGraph store instance for persistent storage
    namespace : str
        Base namespace (e.g. "default")
    user_id : str
        User ID for scoping memories

    Returns
    -------
    dict
        Dictionary with keys: save_fact, save_location, recall, forget
    """

    fact_ns = (namespace, user_id, "facts")
    spatial_ns = (namespace, user_id, "spatial")

    class SaveFactTool(BaseTool):
        """Save a text fact to long-term memory.

        Use this when the user shares information that should persist
        across sessions (preferences, habits, important facts).
        """

        name: str = "save_fact"
        description: str = (
            "Save a text fact to long-term memory. "
            "Use when the user shares information that should persist across sessions. "
            "Examples: preferences, habits, important dates, robot configuration."
        )
        args_schema: Type[SaveFactToolInput] = SaveFactToolInput

        def _run(self, fact: str) -> str:
            key = str(uuid.uuid4())
            value = {
                "text": fact,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            store.put(fact_ns, key, value)
            return f"Fact saved: '{fact}'"

        def _arun(self, fact: str) -> str:
            return self._run(fact)

    class SaveLocationTool(BaseTool):
        """Save structured spatial/location data to long-term memory.

        Use this when the user or agent identifies a named location
        with coordinates, objects, or landmarks.
        """

        name: str = "save_location"
        description: str = (
            "Save structured spatial/location data to long-term memory. "
            "Use when a named location with coordinates or objects is identified. "
            "Includes position, detected objects, and natural language description."
        )
        args_schema: Type[SaveLocationToolInput] = SaveLocationToolInput

        def _run(
            self,
            location_name: str,
            pose: Optional[dict] = None,
            objects: Optional[list[str]] = None,
            description: Optional[str] = None,
        ) -> str:
            key = f"loc_{location_name.lower().replace(' ', '_')}"
            value = {
                "location": location_name,
                "pose": pose,
                "objects": objects or [],
                "description": description or "",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            store.put(spatial_ns, key, value)
            result = f"Location saved: '{location_name}'"
            if pose:
                result += f" at ({pose.get('x', '?')}, {pose.get('y', '?')}, {pose.get('z', '?')})"
            return result

        def _arun(
            self,
            location_name: str,
            pose: Optional[dict] = None,
            objects: Optional[list[str]] = None,
            description: Optional[str] = None,
        ) -> str:
            return self._run(location_name, pose, objects, description)

    class RecallMemoryTool(BaseTool):
        """Search and recall stored memories.

        Searches both facts and spatial memories by semantic similarity.
        Returns matching memories with timestamps.
        """

        name: str = "recall_memory"
        description: str = (
            "Search and recall stored long-term memories. "
            "Searches both text facts and spatial location data. "
            "Returns matching memories with timestamps."
        )
        args_schema: Type[RecallMemoryToolInput] = RecallMemoryToolInput

        def _run(
            self,
            query: str,
            memory_type: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            namespaces = []
            if memory_type is None or memory_type == "facts":
                namespaces.append(("facts", fact_ns))
            if memory_type is None or memory_type == "spatial":
                namespaces.append(("spatial", spatial_ns))

            results = []
            for mtype, ns in namespaces:
                try:
                    items = store.search(ns, query=query, limit=limit)
                    for item in items:
                        results.append(
                            f"[{mtype}] {item.key}: {json.dumps(item.value)}"
                        )
                except Exception as e:
                    results.append(f"[{mtype}] Search error: {e}")

            if not results:
                return f"No memories found matching '{query}'"
            return f"Found {len(results)} memories:\n" + "\n".join(results[:limit])

        def _arun(
            self,
            query: str,
            memory_type: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            return self._run(query, memory_type, limit)

    class ForgetMemoryTool(BaseTool):
        """Delete matching stored memories."""

        name: str = "forget_memory"
        description: str = (
            "Request deletion of a stored memory. "
            "This tool directly deletes matching memories in the store."
        )
        args_schema: Type[ForgetMemoryToolInput] = ForgetMemoryToolInput

        def _run(self, query: str) -> str:
            # Search for matching memories in both namespaces
            namespaces = [("facts", fact_ns), ("spatial", spatial_ns)]
            matches = []
            for mtype, ns in namespaces:
                try:
                    items = store.search(ns, query=query, limit=10)
                    for item in items:
                        matches.append((mtype, ns, item.key, item.value))
                except Exception:
                    pass

            if not matches:
                return f"No memories found matching '{query}'"

            deleted = []
            for mtype, ns, key, _ in matches:
                try:
                    store.delete(ns, key)
                    deleted.append(f"[{mtype}] {key}")
                except Exception as e:
                    deleted.append(f"[{mtype}] {key}: error ({e})")

            return f"Deleted {len(deleted)} memories: {', '.join(deleted)}"

        def _arun(self, query: str) -> str:
            return self._run(query)

    return {
        "save_fact": SaveFactTool(),
        "save_location": SaveLocationTool(),
        "recall": RecallMemoryTool(),
        "forget": ForgetMemoryTool(),
    }


class MemoryTools:
    """Container for memory tools."""

    def __init__(self, tools: dict, store: BaseStore, namespace: str, user_id: str):
        self.tools = tools
        self.store = store
        self.namespace = namespace
        self.user_id = user_id
