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

import asyncio
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import BaseStore, IndexConfig

from rai.memory.config import MemoryConfig, load_memory_config

logger = logging.getLogger(__name__)

_MULTIMODAL_MSGPACK_ALLOWLIST = (
    ("rai.messages.multimodal", "AIMultimodalMessage"),
    ("rai.messages.multimodal", "HumanMultimodalMessage"),
    ("rai.messages.multimodal", "SystemMultimodalMessage"),
    ("rai.messages.multimodal", "ToolMultimodalMessage"),
)


def _default_index_config(
    embeddings: Optional[Embeddings] = None,
) -> IndexConfig:
    if embeddings is not None:
        sample = embeddings.embed_query("sample")
        return IndexConfig(embed=embeddings, dims=len(sample))
    random_fn: Callable[[List[str]], List[List[float]]] = lambda _: [[0.1] * 768]
    return IndexConfig(embed=random_fn, dims=768)


def _memory_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=_MULTIMODAL_MSGPACK_ALLOWLIST,
    )


class MemoryManager:
    """Manages persistent memory layers (short-term checkpointer + long-term store).

    Short-term memory: Thread-scoped conversation history via LangGraph checkpointer.
    Long-term memory: Cross-session key-value store with semantic search via LangGraph store.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        config_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        if config is None:
            config = load_memory_config(config_path)
        self._config = config
        self._embeddings = embeddings
        self._checkpointer: Optional[BaseCheckpointSaver] = None
        self._store: Optional[BaseStore] = None
        self._cm_checker = None
        self._cm_store = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        if self._checkpointer is None:
            raise RuntimeError(
                "MemoryManager not started. Call start() or use 'with' context."
            )
        return self._checkpointer

    @property
    def store(self) -> BaseStore:
        if self._store is None:
            raise RuntimeError(
                "MemoryManager not started. Call start() or use 'with' context."
            )
        return self._store

    def _create_components(
        self,
    ) -> Tuple[BaseCheckpointSaver, BaseStore]:
        if self._config.backend == "sqlite":
            checkpointer, store, cm_checker, cm_store = self._run_on_async_loop(
                _create_sqlite_async_components(
                    self._config.short_term_path,
                    self._config.long_term_path,
                    self._embeddings,
                )
            )
            self._cm_checker = cm_checker
            self._cm_store = cm_store
            return checkpointer, store

        checkpointer, store, cm_checker, cm_store = _create_memory_components(
            backend=self._config.backend,
            short_term_path=self._config.short_term_path,
            long_term_path=self._config.long_term_path,
            connection=self._config.connection,
            embeddings=self._embeddings,
        )
        self._cm_checker = cm_checker
        self._cm_store = cm_store
        return checkpointer, store

    @property
    def async_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._async_loop

    def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        if self._async_loop is not None and self._async_loop.is_running():
            return self._async_loop

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=run_loop,
            name="rai-memory-asyncio",
            daemon=True,
        )
        thread.start()
        ready.wait()
        self._async_loop = loop
        self._async_thread = thread
        return loop

    def _run_on_async_loop(self, coro):
        loop = self._ensure_async_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def start(self):
        if not self._config.enabled:
            logger.warning("Memory is disabled in config.")
            return
        if self._checkpointer is not None:
            return
        self._checkpointer, self._store = self._create_components()
        logger.info(
            f"Memory started (backend={self._config.backend}, "
            f"namespace={self._config.namespace})"
        )

    def setup(self):
        if self._checkpointer is not None and hasattr(self._checkpointer, "setup"):
            result = self._checkpointer.setup()
            if asyncio.iscoroutine(result):
                self._run_on_async_loop(result)
        if self._store is not None and hasattr(self._store, "setup"):
            try:
                result = self._store.setup()
                if asyncio.iscoroutine(result):
                    self._run_on_async_loop(result)
            except Exception as e:
                logger.warning(
                    f"Store setup failed (semantic search may be unavailable): {e}"
                )

    def prune_checkpoints(self, thread_id: str) -> int:
        """Prune completed-turn SQLite history while retaining recent recovery points."""
        if (
            not self._config.checkpoint_prune_after_turn
            or self._config.backend != "sqlite"
            or self._checkpointer is None
        ):
            return 0
        deleted = self._run_on_async_loop(
            _prune_sqlite_thread_checkpoints(
                self._checkpointer,
                thread_id,
                self._config.checkpoint_keep_per_thread,
            )
        )
        self._warn_if_checkpoint_database_large()
        return deleted

    def _warn_if_checkpoint_database_large(self) -> None:
        path = self._config.short_term_path
        if path == ":memory:":
            return
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.exists():
            return
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        if size_mb > self._config.checkpoint_warn_mb:
            logger.warning(
                "Checkpoint database is %.1f MiB (configured warning limit: %d MiB). "
                "Pruning makes pages reusable but does not shrink the SQLite file; "
                "compact it during maintenance downtime.",
                size_mb,
                self._config.checkpoint_warn_mb,
            )

    def stop(self):
        if self._async_loop is not None:
            loop = self._async_loop
            try:
                asyncio.run_coroutine_threadsafe(
                    self._close_async_components(),
                    loop,
                ).result()
            finally:
                self._checkpointer = None
                self._store = None
                self._cm_checker = None
                self._cm_store = None
                loop.call_soon_threadsafe(loop.stop)
                if self._async_thread is not None:
                    self._async_thread.join(timeout=2)
                loop.close()
                self._async_loop = None
                self._async_thread = None
            return

        if self._checkpointer is not None:
            if hasattr(self._checkpointer, "close"):
                self._checkpointer.close()
            self._checkpointer = None
        if self._store is not None:
            if hasattr(self._store, "close"):
                self._store.close()
            self._store = None
        self._cm_checker = None
        self._cm_store = None

    async def _close_async_components(self) -> None:
        if self._store is not None:
            store_task = getattr(self._store, "_task", None)
            if store_task is not None and not store_task.done():
                store_task.cancel()
                try:
                    await store_task
                except asyncio.CancelledError:
                    pass
        if self._cm_checker is not None:
            await self._cm_checker.__aexit__(None, None, None)
        if self._cm_store is not None:
            await self._cm_store.__aexit__(None, None, None)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args: Any):
        self.stop()

    @contextmanager
    def context(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()


def _create_memory_components(
    backend: str,
    short_term_path: str,
    long_term_path: str,
    connection: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    if backend == "postgres":
        return _create_postgres_components(connection, embeddings)
    return _create_sqlite_components(short_term_path, long_term_path, embeddings)


async def _create_sqlite_async_components(
    short_term_path: str,
    long_term_path: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore
    from langgraph.store.sqlite.base import SqliteIndexConfig

    _ensure_sqlite_parent_dir(short_term_path)
    _ensure_sqlite_parent_dir(long_term_path)

    cm_checker = AsyncSqliteSaver.from_conn_string(short_term_path)
    checkpointer = await cm_checker.__aenter__()
    checkpointer.serde = _memory_serde()
    await checkpointer.setup()

    idx = _default_index_config(embeddings)
    sqlite_index = SqliteIndexConfig(embed=idx["embed"], dims=idx["dims"])
    cm_store = AsyncSqliteStore.from_conn_string(long_term_path, index=sqlite_index)
    store = await cm_store.__aenter__()
    await store.setup()
    return checkpointer, store, cm_checker, cm_store


def _create_sqlite_components(
    short_term_path: str,
    long_term_path: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore
    from langgraph.store.sqlite.base import SqliteIndexConfig

    _ensure_sqlite_parent_dir(short_term_path)
    _ensure_sqlite_parent_dir(long_term_path)

    cm_checker = SqliteSaver.from_conn_string(short_term_path)
    checkpointer = cm_checker.__enter__()
    checkpointer.serde = _memory_serde()
    checkpointer.setup()

    idx = _default_index_config(embeddings)
    sqlite_index = SqliteIndexConfig(embed=idx["embed"], dims=idx["dims"])
    cm_store = SqliteStore.from_conn_string(long_term_path, index=sqlite_index)
    store = cm_store.__enter__()
    store.setup()
    return checkpointer, store, cm_checker, cm_store


async def _prune_sqlite_thread_checkpoints(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
    keep: int,
) -> int:
    conn = getattr(checkpointer, "conn", None)
    if conn is None:
        return 0

    async with conn.execute(
        "SELECT checkpoint_ns, checkpoint_id FROM checkpoints "
        "WHERE thread_id = ? ORDER BY checkpoint_ns, checkpoint_id DESC",
        (thread_id,),
    ) as cursor:
        rows = await cursor.fetchall()

    seen_by_namespace: dict[str, int] = {}
    stale: list[tuple[str, str, str]] = []
    for checkpoint_ns, checkpoint_id in rows:
        count = seen_by_namespace.get(checkpoint_ns, 0)
        seen_by_namespace[checkpoint_ns] = count + 1
        if count >= keep:
            stale.append((thread_id, checkpoint_ns, checkpoint_id))
    if not stale:
        return 0

    await conn.executemany(
        "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? "
        "AND checkpoint_id = ?",
        stale,
    )
    await conn.executemany(
        "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
        "AND checkpoint_id = ?",
        stale,
    )
    await conn.commit()
    return len(stale)


def _ensure_sqlite_parent_dir(path: str) -> None:
    if path == ":memory:":
        return
    parent = Path(path).expanduser().parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def _create_postgres_components(
    connection: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    cm_checker = PostgresSaver.from_conn_string(connection)
    checkpointer = cm_checker.__enter__()
    checkpointer.serde = _memory_serde()
    checkpointer.setup()

    pg_index = None
    if embeddings is not None:
        from langgraph.store.postgres.base import PostgresIndexConfig

        idx = _default_index_config(embeddings)
        pg_index = PostgresIndexConfig(
            embed=idx["embed"],
            dims=idx["dims"],
        )
    cm_store = PostgresStore.from_conn_string(connection, index=pg_index)
    store_store = cm_store.__enter__()
    return checkpointer, store_store, cm_checker, cm_store
