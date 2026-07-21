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

from dataclasses import dataclass
from typing import Optional

import tomli

_DEFAULT_MEMORY_CONFIG = """
memory section not found in config.toml. Add:

[memory]
enabled = true
backend = "sqlite"
short_term_path = "checkpoints.db"
long_term_path = "store.db"
# backend = "postgres"
# connection = "postgresql://user:pass@localhost:5432/dbname"
namespace = "default"
checkpoint_prune_after_turn = true
# Retained root-graph checkpoints per thread; completed subgraphs are removed.
checkpoint_keep_per_thread = 20
checkpoint_warn_mb = 512
"""


@dataclass
class MemoryConfig:
    enabled: bool = False
    backend: str = "sqlite"
    short_term_path: str = "checkpoints.db"
    long_term_path: str = "store.db"
    connection: str = ""
    namespace: str = "default"
    checkpoint_prune_after_turn: bool = True
    checkpoint_keep_per_thread: int = 20
    checkpoint_warn_mb: int = 512

    def __post_init__(self) -> None:
        if self.checkpoint_keep_per_thread < 1:
            raise ValueError("memory.checkpoint_keep_per_thread must be positive")
        if self.checkpoint_warn_mb < 1:
            raise ValueError("memory.checkpoint_warn_mb must be positive")


def load_memory_config(config_path: Optional[str] = None) -> MemoryConfig:
    if config_path is None:
        config_path = "config.toml"
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)

    if "memory" not in config_dict:
        return MemoryConfig()

    return MemoryConfig(**config_dict["memory"])
