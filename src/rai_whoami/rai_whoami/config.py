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

from dataclasses import dataclass
from typing import Literal, Optional

import tomli


@dataclass
class WhoamiRetrievalConfig:
    strategy: Literal["vector", "keyword", "hybrid"] = "vector"
    vector_k: int = 4
    keyword_k: int = 4
    final_k: int = 4
    score_threshold: float | None = None
    normalize_embeddings: bool = False
    distance_strategy: Literal["l2", "cosine", "inner_product"] = "l2"


@dataclass
class WhoamiConfig:
    enabled: bool = False
    root_dir: str = ""
    build_vector_db: bool = False
    k: int = 4
    retrieval: WhoamiRetrievalConfig | None = None

    def __post_init__(self):
        if self.retrieval is None:
            self.retrieval = WhoamiRetrievalConfig(
                vector_k=self.k,
                keyword_k=self.k,
                final_k=self.k,
            )
        elif isinstance(self.retrieval, dict):
            self.retrieval = WhoamiRetrievalConfig(**self.retrieval)


def load_whoami_config(config_path: Optional[str] = None) -> WhoamiConfig:
    if config_path is None:
        config_path = "config.toml"
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    return WhoamiConfig(**config_dict.get("whoami", {}))
