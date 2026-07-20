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

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.summarization import (
    ContextSize,
    DEFAULT_SUMMARY_PROMPT,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from rai.context.config import ContextConfig

SUMMARY_PREFIX = "## Short-Term Memory Summary"
TRUNCATION_NOTICE = "\n\n[Tool result truncated to fit the context budget.]"


class ContextBudgetExceeded(RuntimeError):
    """Raised before an LLM call when its effective input cannot fit safely."""


@dataclass(frozen=True)
class PreparedContext:
    messages: list[BaseMessage]
    summary: str
    token_count: int
    compressed: bool = False
    truncated_tool_results: int = 0


SummaryModelFactory = Callable[[], BaseChatModel]


def inject_summary(system_prompt: str, summary: str) -> str:
    if not summary:
        return system_prompt
    return (
        f"{system_prompt}\n\n{SUMMARY_PREFIX}\n"
        "The following is an internal summary of earlier conversation in this "
        f"thread. Use it as context, not as a literal assistant message.\n{summary}"
    )


class ContextManager:
    """Prepare a bounded model context using LangChain summarization primitives."""

    def __init__(
        self,
        config: ContextConfig,
        summary_model_factory: SummaryModelFactory,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize | None = None,
    ) -> None:
        self.config = config
        self._summary_model_factory = summary_model_factory
        self._trigger = trigger
        self._keep = keep
        self._token_counter = partial(
            count_tokens_approximately,
            chars_per_token=config.chars_per_token,
            tokens_per_image=config.tokens_per_image,
        )
        self._middleware: SummarizationMiddleware | None = None
        self._summary_input_budget = 0
        self._tool_token_cache: tuple[tuple[int, ...], int] | None = None

    def prepare(
        self,
        messages: Sequence[BaseMessage],
        *,
        summary: str = "",
        tools: Sequence[BaseTool] = (),
    ) -> PreparedContext:
        """Compact and bound the exact input that will be sent to the model."""

        copied = list(messages)
        if not self.config.enabled:
            return PreparedContext(
                messages=copied,
                summary=summary,
                token_count=self.count_tokens(copied, summary=summary, tools=tools),
            )

        system_messages = [m for m in copied if isinstance(m, SystemMessage)]
        conversation = [m for m in copied if not isinstance(m, SystemMessage)]
        total_tokens = self.count_tokens(copied, summary=summary, tools=tools)
        compressed = False

        should_compact = self._should_compact(conversation, total_tokens)
        current_turn_start = self._last_human_index(conversation)
        if should_compact and current_turn_start != 0:
            middleware = self._get_middleware()
            cutoff = middleware._determine_cutoff_index(conversation)
            # Keep the active user turn intact. ToolPolicy derives its deterministic
            # per-turn counters from these messages; old tool payloads are bounded
            # below without erasing the calls themselves.
            if current_turn_start is not None:
                cutoff = min(cutoff, current_turn_start)
            if cutoff > 0:
                to_summarize, conversation = middleware._partition_messages(
                    conversation, cutoff
                )
                summary = self._create_rolling_summary(
                    middleware,
                    to_summarize,
                    existing_summary=summary,
                )
                compressed = True

        prepared = system_messages + conversation
        prepared, truncated = self._fit_tool_results(
            prepared,
            summary=summary,
            tools=tools,
            token_limit=(
                self.config.trigger_tokens
                if should_compact
                else self.config.max_input_tokens
            ),
        )
        total_tokens = self.count_tokens(prepared, summary=summary, tools=tools)
        if total_tokens > self.config.max_input_tokens:
            raise ContextBudgetExceeded(
                "Effective context requires approximately "
                f"{total_tokens} tokens, exceeding configured context.max_input_tokens="
                f"{self.config.max_input_tokens}. Reduce the system prompt, tool schemas, "
                "or the latest user input, or raise the configured operational budget."
            )

        return PreparedContext(
            messages=prepared,
            summary=summary,
            token_count=total_tokens,
            compressed=compressed,
            truncated_tool_results=truncated,
        )

    def _should_compact(
        self, messages: Sequence[BaseMessage], total_tokens: int
    ) -> bool:
        trigger = self._trigger or [
            ("tokens", self.config.trigger_tokens),
            ("messages", self.config.max_messages),
        ]
        clauses = trigger if isinstance(trigger, list) else [trigger]
        return any(
            (kind == "tokens" and total_tokens >= value)
            or (kind == "messages" and len(messages) >= value)
            or (
                kind == "fraction"
                and total_tokens >= int(self.config.max_input_tokens * value)
            )
            for kind, value in clauses
        )

    def count_tokens(
        self,
        messages: Sequence[BaseMessage],
        *,
        summary: str = "",
        tools: Sequence[BaseTool] = (),
    ) -> int:
        effective = list(messages)
        if summary:
            effective = self._messages_with_summary(effective, summary)
        return self._token_counter(effective) + self._tool_tokens(tools)

    def _tool_tokens(self, tools: Sequence[BaseTool]) -> int:
        key = tuple(id(tool) for tool in tools)
        if self._tool_token_cache is None or self._tool_token_cache[0] != key:
            self._tool_token_cache = (
                key,
                self._token_counter([], tools=list(tools)),
            )
        return self._tool_token_cache[1]

    def _get_middleware(self) -> SummarizationMiddleware:
        if self._middleware is None:
            summary_prompt_tokens = self._token_counter(
                [HumanMessage(content=DEFAULT_SUMMARY_PROMPT.format(messages=""))]
            )
            # Reserve room for both the summary response and the middleware's
            # structured prompt.
            available_input = (
                self.config.max_input_tokens
                - self.config.summary_max_tokens
                - summary_prompt_tokens
            )
            if available_input < 128:
                raise ContextBudgetExceeded(
                    "Configured context.max_input_tokens="
                    f"{self.config.max_input_tokens} is too small for the "
                    "summarization prompt and summary_max_tokens reserve."
                )
            # Later chunks also carry the previous rolling summary.
            self._summary_input_budget = max(
                128,
                available_input - self.config.summary_max_tokens,
            )
            self._middleware = SummarizationMiddleware(
                model=self._summary_model_factory(),
                trigger=self._trigger
                or [
                    ("tokens", self.config.trigger_tokens),
                    ("messages", self.config.max_messages),
                ],
                keep=self._keep or ("tokens", self.config.keep_tokens),
                token_counter=self._token_counter,
                trim_tokens_to_summarize=None,
            )
        return self._middleware

    def _create_rolling_summary(
        self,
        middleware: SummarizationMiddleware,
        messages: Sequence[AnyMessage],
        *,
        existing_summary: str,
    ) -> str:
        chunks: list[list[AnyMessage]] = []
        chunk: list[AnyMessage] = []
        chunk_tokens = 0
        for message in messages:
            message_tokens = self._message_tokens(message)
            if chunk and chunk_tokens + message_tokens > self._summary_input_budget:
                chunks.append(chunk)
                chunk = []
                chunk_tokens = 0
            if message_tokens > self._summary_input_budget:
                content = self._head_tail(
                    str(message.content),
                    self._summary_input_budget,
                    suffix="\n\n[Message truncated for summarization.]",
                )
                message = HumanMessage(content=f"{type(message).__name__}: {content}")
                message_tokens = self._message_tokens(message)
            chunk.append(message)
            chunk_tokens += message_tokens
        if chunk:
            chunks.append(chunk)

        rolling_summary = existing_summary
        for summary_chunk in chunks:
            summary_input: list[AnyMessage] = []
            if rolling_summary:
                summary_input.append(
                    HumanMessage(
                        content=(f"Previous conversation summary:\n{rolling_summary}")
                    )
                )
            summary_input.extend(summary_chunk)
            rolling_summary = self._limit_text_tokens(
                middleware._create_summary(summary_input),
                self.config.summary_max_tokens,
            )
        return rolling_summary

    @staticmethod
    def _messages_with_summary(
        messages: list[BaseMessage], summary: str
    ) -> list[BaseMessage]:
        for index, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                updated = message.model_copy(
                    update={"content": inject_summary(str(message.content), summary)}
                )
                return [*messages[:index], updated, *messages[index + 1 :]]
        return [SystemMessage(content=inject_summary("", summary)), *messages]

    def _fit_tool_results(
        self,
        messages: list[BaseMessage],
        *,
        summary: str,
        tools: Sequence[BaseTool],
        token_limit: int,
    ) -> tuple[list[BaseMessage], int]:
        result = list(messages)
        truncated = 0
        while (
            current := self.count_tokens(result, summary=summary, tools=tools)
        ) > token_limit:
            candidates = [
                (index, self._message_tokens(message))
                for index, message in enumerate(result)
                if isinstance(message, ToolMessage)
                and isinstance(message.content, str)
                and self._message_tokens(message) > 64
            ]
            if not candidates:
                break
            index, message_tokens = max(candidates, key=lambda item: item[1])
            excess = current - token_limit
            target = max(32, message_tokens - excess - 32)
            message = cast(ToolMessage, result[index])
            content = self._head_tail(
                str(message.content),
                target,
                suffix=TRUNCATION_NOTICE,
            )
            result[index] = message.model_copy(update={"content": content})
            truncated += 1
        return result, truncated

    @staticmethod
    def _last_human_index(messages: Sequence[BaseMessage]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return index
        return None

    def _message_tokens(self, message: BaseMessage) -> int:
        return self._token_counter([message])

    def _limit_text_tokens(self, text: Any, token_limit: int) -> str:
        value = str(text)
        if self._token_counter([HumanMessage(content=value)]) <= token_limit:
            return value
        return self._head_tail(value, token_limit, suffix="\n\n[Summary truncated.]")

    def _head_tail(self, text: str, token_limit: int, *, suffix: str) -> str:
        if not text:
            return suffix.strip()
        char_budget = max(
            32,
            int(token_limit * self.config.chars_per_token) - len(suffix),
        )
        if len(text) <= char_budget:
            return text
        head_size = max(1, int(char_budget * 0.7))
        tail_size = max(1, char_budget - head_size)
        return f"{text[:head_size]}\n...\n{text[-tail_size:]}{suffix}"
