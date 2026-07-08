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

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from rai.memory.manager import MemoryManager


@dataclass(frozen=True)
class SessionSummary:
    thread_id: str
    created_at: float | None = None
    updated_at: float | None = None
    first_user_message: str = ""
    message_count: int = 0

    @property
    def created_at_display(self) -> str:
        if self.created_at is None:
            return "unknown"
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


def session_metadata_namespace(namespace: str) -> tuple[str, str, str]:
    return (namespace, "__sessions__", "metadata")


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


def delete_session_metadata(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
) -> None:
    memory_mgr.store.delete(session_metadata_namespace(namespace), thread_id)


def graph_config(thread_id: str) -> RunnableConfig:
    return RunnableConfig({"configurable": {"thread_id": thread_id}})


def load_thread_state(graph, thread_id: str) -> tuple[list, str]:
    """Load checkpointed messages and summary for a graph thread."""
    snapshot = graph.get_state(graph_config(thread_id))
    values = snapshot.values or {}
    return list(values.get("messages", [])), values.get("summary", "")


def record_session_activity(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
    *,
    first_user_message: str | None = None,
    message_count: int | None = None,
) -> None:
    ns = session_metadata_namespace(namespace)
    now = time.time()
    existing = _get_store_value(memory_mgr, ns, thread_id) or {}
    value = dict(existing)
    value.setdefault("thread_id", thread_id)
    value.setdefault("created_at", now)
    value["updated_at"] = now
    if first_user_message and not value.get("first_user_message"):
        value["first_user_message"] = first_user_message
    if message_count is not None:
        value["message_count"] = message_count
    memory_mgr.store.put(ns, thread_id, value)


def list_session_summaries(
    memory_mgr: MemoryManager,
    graph,
    namespace: str,
    limit: int = 200,
) -> list[SessionSummary]:
    metadata_by_thread = _load_session_metadata(memory_mgr, namespace, limit)
    summaries: list[SessionSummary] = []
    for thread_id in get_session_ids(memory_mgr, limit=limit):
        metadata = metadata_by_thread.get(thread_id, {})
        summaries.append(
            _summary_from_metadata(thread_id, metadata)
            if metadata
            else SessionSummary(thread_id=thread_id)
        )
    return sorted(
        summaries,
        key=lambda item: item.updated_at or item.created_at or 0.0,
        reverse=True,
    )


def session_summary_label(summary: SessionSummary, max_first_chars: int = 60) -> str:
    first = summary.first_user_message.strip() or "(empty)"
    if len(first) > max_first_chars:
        first = first[: max_first_chars - 3].rstrip() + "..."
    return f"{summary.created_at_display} | {first} | {summary.thread_id}"


def _load_session_metadata(
    memory_mgr: MemoryManager,
    namespace: str,
    limit: int,
) -> dict[str, dict[str, Any]]:
    ns = session_metadata_namespace(namespace)
    metadata: dict[str, dict[str, Any]] = {}
    try:
        for item in memory_mgr.store.search(ns, query="", limit=limit):
            metadata[item.key] = item.value
    except Exception:
        pass
    return metadata


def _summary_from_metadata(thread_id: str, metadata: dict[str, Any]) -> SessionSummary:
    return SessionSummary(
        thread_id=thread_id,
        created_at=metadata.get("created_at"),
        updated_at=metadata.get("updated_at"),
        first_user_message=metadata.get("first_user_message", ""),
        message_count=int(metadata.get("message_count", 0) or 0),
    )


def _get_store_value(
    memory_mgr: MemoryManager,
    namespace: tuple[str, str, str],
    key: str,
) -> dict[str, Any] | None:
    try:
        item = memory_mgr.store.get(namespace, key)
        return item.value if item is not None else None
    except Exception:
        return None
