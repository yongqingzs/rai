import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from rai.memory.config import MemoryConfig
from rai.memory.manager import MemoryManager
from rai.memory.session import (
    append_session_transcript_messages,
    delete_session,
    load_session_transcript,
    load_session_transcript_page,
)


def test_sqlite_transcript_is_paginated_non_indexed_and_session_owned(tmp_path):
    store_path = Path(tmp_path) / "store.db"
    manager = MemoryManager(
        config=MemoryConfig(
            enabled=True,
            backend="sqlite",
            short_term_path=str(Path(tmp_path) / "checkpoints.db"),
            long_term_path=str(store_path),
            namespace="inspection",
        )
    )
    messages = [
        (
            HumanMessage(content=f"question {index}", id=f"user-{index}")
            if index % 2 == 0
            else AIMessage(content=f"answer {index}", id=f"assistant-{index}")
        )
        for index in range(205)
    ]

    manager.start()
    try:
        assert (
            append_session_transcript_messages(
                manager,
                "inspection",
                "session-a",
                messages,
            )
            == 205
        )

        newest_page = load_session_transcript_page(
            manager,
            "inspection",
            "session-a",
            limit=100,
        )
        restored = load_session_transcript(
            manager,
            "inspection",
            "session-a",
            page_size=100,
        )

        assert len(newest_page.messages) == 100
        assert newest_page.next_offset == 100
        assert [message.content for message in restored] == [
            message.content for message in messages
        ]

        with sqlite3.connect(store_path) as connection:
            assert (
                connection.execute("SELECT count(*) FROM store_vectors").fetchone()[0]
                == 0
            )

        delete_session(manager, "session-a", "inspection")
        assert (
            load_session_transcript(
                manager,
                "inspection",
                "session-a",
            )
            == []
        )
    finally:
        manager.stop()
