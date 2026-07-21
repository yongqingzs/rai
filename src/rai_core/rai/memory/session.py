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

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnableConfig

from rai.memory.manager import MemoryManager
from rai.messages import delete_session_artifacts

_TRANSCRIPT_STORAGE_PAGE_SIZE = 100
_TRANSCRIPT_RECENT_KEY_LIMIT = 512


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


@dataclass(frozen=True)
class TranscriptPage:
    """One newest-first storage page, exposed as chronological messages."""

    messages: list[BaseMessage]
    next_offset: int | None


def session_metadata_namespace(namespace: str) -> tuple[str, str, str]:
    return (namespace, "__sessions__", "metadata")


def session_transcript_namespace(
    namespace: str, thread_id: str
) -> tuple[str, str, str, str]:
    return (namespace, "__sessions__", "transcript", thread_id)


def append_session_transcript_message(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
    message: BaseMessage,
    *,
    turn_id: str = "",
) -> bool:
    """Append one idempotent, non-indexed message to a session transcript."""
    return bool(
        append_session_transcript_messages(
            memory_mgr,
            namespace,
            thread_id,
            [message],
            turn_id=turn_id,
        )
    )


def append_session_transcript_messages(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
    messages: list[BaseMessage],
    *,
    turn_id: str = "",
) -> int:
    """Append messages while reading and updating each storage page once."""
    if not messages:
        return 0
    transcript_ns = session_transcript_namespace(namespace, thread_id)
    metadata_item = _get_store_item(memory_mgr, transcript_ns, "metadata")
    metadata = dict(metadata_item.value) if metadata_item is not None else {}
    recent_keys = list(metadata.get("recent_keys", []))
    recent_key_set = set(recent_keys)
    message_count = int(metadata.get("message_count", 0))
    pages: dict[int, list[dict[str, Any]]] = {}
    appended = 0
    sequence = time.time_ns()
    recorded_at = time.time()

    for index, (message, message_dict) in enumerate(
        zip(messages, messages_to_dict(messages), strict=True)
    ):
        message_turn_id = f"{turn_id}:{index}" if turn_id else ""
        key = _transcript_message_key(
            message,
            message_dict,
            turn_id=message_turn_id,
        )
        if key in recent_key_set:
            continue
        page_index = message_count // _TRANSCRIPT_STORAGE_PAGE_SIZE
        if page_index not in pages:
            page_item = _get_store_item(
                memory_mgr,
                transcript_ns,
                _transcript_page_key(page_index),
            )
            page_value = dict(page_item.value) if page_item is not None else {}
            pages[page_index] = list(page_value.get("entries", []))
        pages[page_index].append(
            {
                "key": key,
                "message": message_dict,
                "sequence": sequence + appended,
                "recorded_at": recorded_at,
            }
        )
        recent_keys.append(key)
        recent_key_set.add(key)
        message_count += 1
        appended += 1

    if not appended:
        return 0
    for page_index, entries in pages.items():
        _put_store_item(
            memory_mgr,
            transcript_ns,
            _transcript_page_key(page_index),
            {"entries": entries},
            index=False,
        )
    _put_store_item(
        memory_mgr,
        transcript_ns,
        "metadata",
        {
            "message_count": message_count,
            "recent_keys": recent_keys[-_TRANSCRIPT_RECENT_KEY_LIMIT:],
            "updated_at": time.time(),
        },
        index=False,
    )
    return appended


def load_session_transcript_page(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> TranscriptPage:
    """Load one transcript page without deserializing checkpoint snapshots."""
    if limit < 1:
        raise ValueError("Transcript page limit must be positive")
    transcript_ns = session_transcript_namespace(namespace, thread_id)
    metadata_item = _get_store_item(memory_mgr, transcript_ns, "metadata")
    message_count = (
        int(metadata_item.value.get("message_count", 0))
        if metadata_item is not None
        else 0
    )
    end = max(0, message_count - offset)
    start = max(0, end - limit)
    entries = _load_transcript_entry_range(memory_mgr, transcript_ns, start, end)
    message_dicts = [entry["message"] for entry in entries]
    messages = messages_from_dict(message_dicts) if message_dicts else []
    next_offset = offset + len(messages) if start > 0 else None
    return TranscriptPage(messages=messages, next_offset=next_offset)


def load_session_transcript(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
    *,
    page_size: int = 100,
) -> list[BaseMessage]:
    """Load a complete transcript in chronological order using bounded pages."""
    pages: list[list[BaseMessage]] = []
    offset: int | None = 0
    while offset is not None:
        page = load_session_transcript_page(
            memory_mgr,
            namespace,
            thread_id,
            limit=page_size,
            offset=offset,
        )
        pages.append(page.messages)
        offset = page.next_offset
    # Store pages arrive newest first, while each returned page is chronological.
    return [message for page in reversed(pages) for message in page]


def delete_session_transcript(
    memory_mgr: MemoryManager,
    namespace: str,
    thread_id: str,
) -> int:
    """Delete all transcript rows for one session in bounded batches."""
    transcript_ns = session_transcript_namespace(namespace, thread_id)
    metadata_item = _get_store_item(memory_mgr, transcript_ns, "metadata")
    if metadata_item is None:
        return 0
    message_count = int(metadata_item.value.get("message_count", 0))
    page_count = (
        message_count + _TRANSCRIPT_STORAGE_PAGE_SIZE - 1
    ) // _TRANSCRIPT_STORAGE_PAGE_SIZE
    for page_index in range(page_count):
        _delete_store_item(
            memory_mgr,
            transcript_ns,
            _transcript_page_key(page_index),
        )
    _delete_store_item(memory_mgr, transcript_ns, "metadata")
    return message_count


def get_session_ids(memory_mgr: MemoryManager, limit: int = 200) -> List[str]:
    """Get unique thread IDs from the checkpointer.

    Prefer ``get_session_ids_from_metadata`` for UI session lists. This function
    can be expensive for large checkpoint payloads because LangGraph
    checkpointer iteration may deserialize checkpoint blobs.
    """
    thread_ids = set()
    for checkpoint in memory_mgr.checkpointer.list(None, limit=limit):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            thread_ids.add(thread_id)
    return sorted(thread_ids)


def get_session_ids_from_metadata(
    memory_mgr: MemoryManager,
    namespace: str,
    limit: int = 200,
) -> List[str]:
    return sorted(_load_session_metadata(memory_mgr, namespace, limit).keys())


def get_latest_session_id(memory_mgr: MemoryManager, limit: int = 200) -> Optional[str]:
    """Get the latest thread ID reported by the checkpointer."""
    for checkpoint in memory_mgr.checkpointer.list(None, limit=limit):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            return thread_id
    return None


def get_latest_session_id_from_metadata(
    memory_mgr: MemoryManager,
    namespace: str,
    limit: int = 200,
) -> Optional[str]:
    summaries = list_session_summaries(memory_mgr, None, namespace, limit=limit)
    return summaries[0].thread_id if summaries else None


def delete_session(
    memory_mgr: MemoryManager,
    thread_id: str,
    namespace: str | None = None,
) -> int:
    """Delete a checkpoint thread and its session-owned persistent data."""
    delete_thread = getattr(memory_mgr, "delete_thread", None)
    if callable(delete_thread):
        delete_thread(thread_id)
    else:
        memory_mgr.checkpointer.delete_thread(thread_id)
    deleted = delete_session_artifacts(thread_id)
    if namespace is not None:
        deleted += delete_session_transcript(memory_mgr, namespace, thread_id)
    return deleted


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
    summaries = [
        _summary_from_metadata(thread_id, metadata)
        for thread_id, metadata in metadata_by_thread.items()
    ]
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


def _transcript_message_key(
    message: BaseMessage,
    message_dict: dict[str, Any],
    *,
    turn_id: str,
) -> str:
    message_id = getattr(message, "id", None)
    if message_id:
        return f"message:{message_id}"
    payload = json.dumps(message_dict, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"content:{turn_id or uuid.uuid4().hex}:{digest}"


def _transcript_page_key(page_index: int) -> str:
    return f"page:{page_index:012d}"


def _load_transcript_entry_range(
    memory_mgr: MemoryManager,
    namespace: tuple[str, ...],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    if start >= end:
        return []
    first_page = start // _TRANSCRIPT_STORAGE_PAGE_SIZE
    last_page = (end - 1) // _TRANSCRIPT_STORAGE_PAGE_SIZE
    entries: list[dict[str, Any]] = []
    for page_index in range(first_page, last_page + 1):
        item = _get_store_item(
            memory_mgr,
            namespace,
            _transcript_page_key(page_index),
        )
        if item is not None:
            entries.extend(item.value.get("entries", []))
    page_start = first_page * _TRANSCRIPT_STORAGE_PAGE_SIZE
    return entries[start - page_start : end - page_start]


def _put_store_item(
    memory_mgr: MemoryManager,
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    *,
    index: bool | list[str] | None,
) -> None:
    put = getattr(memory_mgr, "put_store_item", None)
    if callable(put):
        put(namespace, key, value, index=index)
        return
    memory_mgr.store.put(namespace, key, value, index=index)


def _get_store_item(
    memory_mgr: MemoryManager,
    namespace: tuple[str, ...],
    key: str,
) -> Any:
    get = getattr(memory_mgr, "get_store_item", None)
    if callable(get):
        return get(namespace, key)
    return memory_mgr.store.get(namespace, key)


def _delete_store_item(
    memory_mgr: MemoryManager,
    namespace: tuple[str, ...],
    key: str,
) -> None:
    delete = getattr(memory_mgr, "delete_store_item", None)
    if callable(delete):
        delete(namespace, key)
        return
    memory_mgr.store.delete(namespace, key)
