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

import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, field_validator

# --- Tool Input Schemas ---


class SaveFactToolInput(BaseModel):
    """Input for saving a text fact to long-term memory."""

    fact: str = Field(..., description="The fact or piece of information to remember")


class SaveLocationToolInput(BaseModel):
    """Input for saving structured spatial/location data."""

    class PoseInput(BaseModel):
        """2D/3D position for a stored location."""

        x: float = Field(..., description="X coordinate in meters")
        y: float = Field(..., description="Y coordinate in meters")
        z: float = Field(..., description="Z coordinate in meters")
        yaw: Optional[float] = Field(
            default=None,
            description="Optional yaw angle of the orientation in radians",
        )

    location_name: str = Field(
        ..., description="Name of the location (e.g. 'Kitchen', 'Living Room')"
    )
    pose: Optional[PoseInput] = Field(
        default=None,
        description="Optional position with x, y, z, and optional yaw coordinates",
    )
    objects: Optional[list[str]] = Field(
        default=None,
        description="Optional list of objects at this location",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional natural language description of the location",
    )

    @field_validator("pose", mode="before")
    @classmethod
    def parse_pose(cls, value):
        if value is None or isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        return value

    @field_validator("objects", mode="before")
    @classmethod
    def parse_objects(cls, value):
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            if not value.strip():
                return []
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        return value


def _normalize_pose(
    pose: Optional[SaveLocationToolInput.PoseInput | dict | str],
) -> Optional[dict]:
    if pose is None:
        return None
    if isinstance(pose, SaveLocationToolInput.PoseInput):
        return pose.model_dump()
    if isinstance(pose, dict):
        return pose
    if isinstance(pose, str):
        parsed = json.loads(pose)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("pose must be a dict, PoseInput, JSON string, or None")


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

    async def _await_store_coro(coro):
        store_loop = getattr(store, "_loop", None)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if (
            store_loop is not None
            and store_loop is not current_loop
            and store_loop.is_running()
        ):
            future = asyncio.run_coroutine_threadsafe(coro, store_loop)
            return await asyncio.wrap_future(future)
        return await coro

    async def _aput(ns: tuple[str, ...], key: str, value: dict) -> None:
        if hasattr(store, "aput"):
            await _await_store_coro(store.aput(ns, key, value))
            return
        store.put(ns, key, value)

    async def _asearch(ns: tuple[str, ...], query: str, limit: int) -> list[Any]:
        if hasattr(store, "asearch"):
            return await _await_store_coro(store.asearch(ns, query=query, limit=limit))
        return store.search(ns, query=query, limit=limit)

    async def _adelete(ns: tuple[str, ...], key: str) -> None:
        if hasattr(store, "adelete"):
            await _await_store_coro(store.adelete(ns, key))
            return
        store.delete(ns, key)

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

        async def _arun(self, fact: str) -> str:
            key = str(uuid.uuid4())
            value = {
                "text": fact,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await _aput(fact_ns, key, value)
            return f"Fact saved: '{fact}'"

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
            pose: Optional[SaveLocationToolInput.PoseInput | dict | str] = None,
            objects: Optional[list[str] | str] = None,
            description: Optional[str] = None,
        ) -> str:
            key = f"loc_{location_name.lower().replace(' ', '_')}"
            pose_data = _normalize_pose(pose)
            value = {
                "location": location_name,
                "pose": pose_data,
                "objects": objects or [],
                "description": description or "",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            store.put(spatial_ns, key, value)
            result = f"Location saved: '{location_name}'"
            if pose_data:
                coords = (
                    f"{pose_data.get('x', '?')}, {pose_data.get('y', '?')}, "
                    f"{pose_data.get('z', '?')}"
                )
                if pose_data.get("yaw") is not None:
                    coords += f", yaw={pose_data.get('yaw'):.4f}"
                result += f" at ({coords})"
            return result

        async def _arun(
            self,
            location_name: str,
            pose: Optional[SaveLocationToolInput.PoseInput | dict | str] = None,
            objects: Optional[list[str] | str] = None,
            description: Optional[str] = None,
        ) -> str:
            key = f"loc_{location_name.lower().replace(' ', '_')}"
            pose_data = _normalize_pose(pose)
            value = {
                "location": location_name,
                "pose": pose_data,
                "objects": objects or [],
                "description": description or "",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await _aput(spatial_ns, key, value)
            result = f"Location saved: '{location_name}'"
            if pose_data:
                coords = (
                    f"{pose_data.get('x', '?')}, {pose_data.get('y', '?')}, "
                    f"{pose_data.get('z', '?')}"
                )
                if pose_data.get("yaw") is not None:
                    coords += f", yaw={pose_data.get('yaw'):.4f}"
                result += f" at ({coords})"
            return result

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

        async def _arun(
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
                    items = await _asearch(ns, query=query, limit=limit)
                    for item in items:
                        results.append(
                            f"[{mtype}] {item.key}: {json.dumps(item.value)}"
                        )
                except Exception as e:
                    results.append(f"[{mtype}] Search error: {e}")

            if not results:
                return f"No memories found matching '{query}'"
            return f"Found {len(results)} memories:\n" + "\n".join(results[:limit])

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

        async def _arun(self, query: str) -> str:
            namespaces = [("facts", fact_ns), ("spatial", spatial_ns)]
            matches = []
            for mtype, ns in namespaces:
                try:
                    items = await _asearch(ns, query=query, limit=10)
                    for item in items:
                        matches.append((mtype, ns, item.key, item.value))
                except Exception:
                    pass

            if not matches:
                return f"No memories found matching '{query}'"

            deleted = []
            for mtype, ns, key, _ in matches:
                try:
                    await _adelete(ns, key)
                    deleted.append(f"[{mtype}] {key}")
                except Exception as e:
                    deleted.append(f"[{mtype}] {key}: error ({e})")

            return f"Deleted {len(deleted)} memories: {', '.join(deleted)}"

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
