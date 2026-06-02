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
from typing import Optional

import tomli


@dataclass
class WhoamiConfig:
    enabled: bool = False
    root_dir: str = ""
    build_vector_db: bool = False
    k: int = 4


def load_whoami_config(config_path: Optional[str] = None) -> WhoamiConfig:
    if config_path is None:
        config_path = "config.toml"
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    return WhoamiConfig(**config_dict.get("whoami", {}))
