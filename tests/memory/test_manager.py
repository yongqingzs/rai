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
        assert ("rai.messages.multimodal", "HumanMultimodalMessage") in serde._allowed_msgpack_modules
        assert ("rai.messages.multimodal", "ToolMultimodalMessage") in serde._allowed_msgpack_modules
    finally:
        memory_mgr.stop()
