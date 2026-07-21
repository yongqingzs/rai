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

from rai.memory.config import MemoryConfig
from rai.memory.manager import MemoryManager


def test_sqlite_memory_manager_creates_parent_directories(tmp_path):
    short_term_path = tmp_path / "nested" / "memory" / "checkpoints.db"
    long_term_path = tmp_path / "nested" / "memory" / "store.db"
    config = MemoryConfig(
        enabled=True,
        backend="sqlite",
        short_term_path=str(short_term_path),
        long_term_path=str(long_term_path),
    )

    with MemoryManager(config=config) as memory_mgr:
        memory_mgr.setup()

    assert short_term_path.parent.exists()
    assert short_term_path.exists()
    assert long_term_path.exists()


def test_memory_manager_checkpoint_allowlist_includes_multimodal_messages():
    config = MemoryConfig(
        enabled=True,
        backend="sqlite",
        short_term_path=":memory:",
        long_term_path=":memory:",
    )

    memory_mgr = MemoryManager(config=config)
    memory_mgr.start()
    try:
        serde = memory_mgr.checkpointer.serde
        assert serde._custom_unpack_ext_hook is False
        assert serde._allowed_msgpack_modules is not True
        assert serde._allowed_msgpack_modules is not None
        assert (
            "rai.messages.multimodal",
            "HumanMultimodalMessage",
        ) in serde._allowed_msgpack_modules
        assert (
            "rai.messages.multimodal",
            "ToolMultimodalMessage",
        ) in serde._allowed_msgpack_modules
    finally:
        memory_mgr.stop()


def test_memory_manager_prunes_completed_thread_checkpoints(tmp_path):
    config = MemoryConfig(
        enabled=True,
        backend="sqlite",
        short_term_path=str(tmp_path / "checkpoints.db"),
        long_term_path=str(tmp_path / "store.db"),
        checkpoint_keep_per_thread=3,
    )
    memory_mgr = MemoryManager(config=config)
    memory_mgr.start()

    async def seed_checkpoints(start: int, stop: int):
        conn = memory_mgr.checkpointer.conn
        for index in range(start, stop):
            checkpoint_id = f"{index:04d}"
            await conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "thread-1",
                    "",
                    checkpoint_id,
                    "msgpack",
                    b"state" * 20_000,
                    b"meta",
                ),
            )
            await conn.execute(
                "INSERT INTO writes "
                "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                "channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "thread-1",
                    "",
                    checkpoint_id,
                    "task",
                    index,
                    "messages",
                    "msgpack",
                    b"value",
                ),
            )
        await conn.commit()

    async def seed_subgraph_checkpoints():
        conn = memory_mgr.checkpointer.conn
        for index in range(2):
            checkpoint_id = f"sub-{index}"
            await conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "thread-1",
                    "react:legacy",
                    checkpoint_id,
                    "msgpack",
                    b"subgraph-state",
                    b"meta",
                ),
            )
            await conn.execute(
                "INSERT INTO writes "
                "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                "channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "thread-1",
                    "react:legacy",
                    checkpoint_id,
                    "task",
                    index,
                    "messages",
                    "msgpack",
                    b"subgraph-value",
                ),
            )
        await conn.commit()

    async def counts():
        conn = memory_mgr.checkpointer.conn
        checkpoint_count = (
            await (await conn.execute("SELECT count(*) FROM checkpoints")).fetchone()
        )[0]
        writes_count = (
            await (await conn.execute("SELECT count(*) FROM writes")).fetchone()
        )[0]
        ids = [
            row[0]
            for row in await (
                await conn.execute(
                    "SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id"
                )
            ).fetchall()
        ]
        return checkpoint_count, writes_count, ids

    async def checkpoint_wal():
        await memory_mgr.checkpointer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    try:
        memory_mgr._run_on_async_loop(seed_checkpoints(0, 8))
        memory_mgr._run_on_async_loop(seed_subgraph_checkpoints())
        assert memory_mgr.prune_checkpoints("thread-1") == 7
        assert memory_mgr._run_on_async_loop(counts()) == (
            3,
            3,
            ["0005", "0006", "0007"],
        )
        memory_mgr._run_on_async_loop(checkpoint_wal())
        initial_size = (tmp_path / "checkpoints.db").stat().st_size
        for index in range(8, 28):
            memory_mgr._run_on_async_loop(seed_checkpoints(index, index + 1))
            memory_mgr.prune_checkpoints("thread-1")
        assert memory_mgr._run_on_async_loop(counts()) == (
            3,
            3,
            ["0025", "0026", "0027"],
        )
        memory_mgr._run_on_async_loop(checkpoint_wal())
        assert (tmp_path / "checkpoints.db").stat().st_size <= initial_size * 2
    finally:
        memory_mgr.stop()


def test_memory_manager_deletes_async_sqlite_thread(tmp_path):
    manager = MemoryManager(
        config=MemoryConfig(
            enabled=True,
            backend="sqlite",
            short_term_path=str(tmp_path / "checkpoints.db"),
            long_term_path=str(tmp_path / "store.db"),
        )
    )
    manager.start()

    async def seed_and_count():
        conn = manager.checkpointer.conn
        await conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("thread-1", "", "0001", "msgpack", b"state", b"meta"),
        )
        await conn.commit()
        return (
            await (
                await conn.execute(
                    "SELECT count(*) FROM checkpoints WHERE thread_id = 'thread-1'"
                )
            ).fetchone()
        )[0]

    try:
        assert manager._run_on_async_loop(seed_and_count()) == 1
        manager.delete_thread("thread-1")

        async def count():
            return (
                await (
                    await manager.checkpointer.conn.execute(
                        "SELECT count(*) FROM checkpoints WHERE thread_id = 'thread-1'"
                    )
                ).fetchone()
            )[0]

        assert manager._run_on_async_loop(count()) == 0
    finally:
        manager.stop()
