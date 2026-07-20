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
from pathlib import Path

import tomli


@dataclass(frozen=True)
class ContextConfig:
    """Operational context budget, independent of the model's advertised limit."""

    enabled: bool = True
    max_input_tokens: int = 12_288
    trigger_ratio: float = 0.67
    keep_ratio: float = 0.25
    max_messages: int = 80
    summary_max_tokens: int = 1_024
    chars_per_token: float = 2.0
    tokens_per_image: int = 1_024

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError("context.max_input_tokens must be greater than zero")
        if not 0 < self.trigger_ratio <= 1:
            raise ValueError("context.trigger_ratio must be in the range (0, 1]")
        if not 0 < self.keep_ratio < self.trigger_ratio:
            raise ValueError(
                "context.keep_ratio must be greater than zero and less than "
                "context.trigger_ratio"
            )
        if self.max_messages <= 1:
            raise ValueError("context.max_messages must be greater than one")
        if self.summary_max_tokens <= 0:
            raise ValueError("context.summary_max_tokens must be greater than zero")
        if self.summary_max_tokens >= self.max_input_tokens:
            raise ValueError(
                "context.summary_max_tokens must be less than context.max_input_tokens"
            )
        if self.chars_per_token <= 0:
            raise ValueError("context.chars_per_token must be greater than zero")
        if self.tokens_per_image <= 0:
            raise ValueError("context.tokens_per_image must be greater than zero")

    @property
    def trigger_tokens(self) -> int:
        return max(1, int(self.max_input_tokens * self.trigger_ratio))

    @property
    def keep_tokens(self) -> int:
        return max(1, int(self.max_input_tokens * self.keep_ratio))


def load_context_config(config_path: str | Path = "config.toml") -> ContextConfig:
    """Load the optional ``[context]`` section from a RAI TOML config."""

    path = Path(config_path)
    if not path.exists():
        return ContextConfig()
    with path.open("rb") as config_file:
        config = tomli.load(config_file)
    return ContextConfig(**config.get("context", {}))
