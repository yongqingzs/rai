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

import logging
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore, IndexConfig

from rai.memory.config import MemoryConfig, load_memory_config

logger = logging.getLogger(__name__)


def _default_index_config(
    embeddings: Optional[Embeddings] = None,
) -> IndexConfig:
    if embeddings is not None:
        sample = embeddings.embed_query("sample")
        return IndexConfig(embed=embeddings, dims=len(sample))
    random_fn: Callable[[List[str]], List[List[float]]] = lambda _: [[0.1] * 768]
    return IndexConfig(embed=random_fn, dims=768)


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
            self._checkpointer.setup()
        if self._store is not None and hasattr(self._store, "setup"):
            try:
                self._store.setup()
            except Exception as e:
                logger.warning(
                    f"Store setup failed (semantic search may be unavailable): {e}"
                )

    def stop(self):
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


def _create_sqlite_components(
    short_term_path: str,
    long_term_path: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore
    from langgraph.store.sqlite.base import SqliteIndexConfig

    cm_checker = SqliteSaver.from_conn_string(short_term_path)
    checkpointer = cm_checker.__enter__()
    checkpointer.setup()

    idx = _default_index_config(embeddings)
    sqlite_index = SqliteIndexConfig(embed=idx["embed"], dims=idx["dims"])
    cm_store = SqliteStore.from_conn_string(long_term_path, index=sqlite_index)
    store_store = cm_store.__enter__()
    store_store.setup()
    return checkpointer, store_store, cm_checker, cm_store


def _create_postgres_components(
    connection: str,
    embeddings: Optional[Embeddings],
) -> Tuple[BaseCheckpointSaver, BaseStore, Any, Any]:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    cm_checker = PostgresSaver.from_conn_string(connection)
    checkpointer = cm_checker.__enter__()
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
