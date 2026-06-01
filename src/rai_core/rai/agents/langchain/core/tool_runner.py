# Copyright (C) 2024 Robotec.AI
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


import json
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.tool_node import msg_content_output
from langgraph.utils.runnable import RunnableCallable
from pydantic import ValidationError

from rai.messages import MultimodalArtifact, ToolMultimodalMessage, store_artifacts


@dataclass(frozen=True)
class ToolPolicy:
    max_calls_per_turn: int | None = None
    max_consecutive_calls: int | None = None
    block_similar_args: bool = False
    similar_args_threshold: float = 0.75


@dataclass
class ToolCallRecord:
    name: str
    args: Any
    output: str | None = None


@dataclass
class ToolCallGuard:
    max_total_calls_per_turn: int = 8
    default_policy: ToolPolicy = field(
        default_factory=lambda: ToolPolicy(
            max_calls_per_turn=4,
            max_consecutive_calls=3,
        )
    )
    policies: dict[str, ToolPolicy] = field(default_factory=dict)

    @classmethod
    def with_default_policies(cls) -> "ToolCallGuard":
        return cls(
            policies={
                "query_robot_docs": ToolPolicy(
                    max_calls_per_turn=2,
                    max_consecutive_calls=1,
                    block_similar_args=True,
                    similar_args_threshold=0.3,
                ),
                "save_fact": ToolPolicy(
                    max_calls_per_turn=5,
                    max_consecutive_calls=2,
                    block_similar_args=True,
                    similar_args_threshold=0.8,
                ),
                "save_location": ToolPolicy(
                    max_calls_per_turn=5,
                    max_consecutive_calls=2,
                    block_similar_args=True,
                    similar_args_threshold=0.8,
                ),
                "forget_memory": ToolPolicy(
                    max_calls_per_turn=3,
                    max_consecutive_calls=2,
                    block_similar_args=True,
                    similar_args_threshold=0.8,
                ),
            }
        )

    def check(
        self,
        call: ToolCall,
        messages: list[Any],
        current_call_index: int = 0,
    ) -> str | None:
        records = self._current_turn_records(messages, current_call_index)
        policy = self.policies.get(call["name"], self.default_policy)
        current_name = call["name"]
        current_args = call.get("args", {})

        if len(records) + 1 > self.max_total_calls_per_turn:
            return (
                f"Tool call blocked: this user turn already used {len(records)} "
                "tool calls. Use the available tool results and answer the user "
                "directly."
            )

        same_tool_records = [
            record for record in records if record.name == current_name
        ]
        if (
            policy.max_calls_per_turn is not None
            and len(same_tool_records) + 1 > policy.max_calls_per_turn
        ):
            return (
                f"Tool call blocked: {current_name} was already called "
                f"{len(same_tool_records)} time(s) in this user turn. Use the "
                "available result(s) and answer the user directly."
            )

        if policy.max_consecutive_calls is not None:
            consecutive = 0
            for record in reversed(records):
                if record.name != current_name:
                    break
                consecutive += 1
            if consecutive + 1 > policy.max_consecutive_calls:
                return (
                    f"Tool call blocked: {current_name} was called repeatedly "
                    "without another step in between. Use the previous result and "
                    "answer the user directly."
                )

        if policy.block_similar_args:
            for record in same_tool_records:
                similarity = self._args_similarity(record.args, current_args)
                if similarity >= policy.similar_args_threshold:
                    return (
                        f"Tool call blocked: {current_name} was already called with "
                        "similar arguments in this user turn. Use the previous result "
                        "and answer the user directly."
                    )

        return None

    def _current_turn_records(
        self,
        messages: list[Any],
        current_call_index: int,
    ) -> list[ToolCallRecord]:
        turn_messages = self._messages_since_last_human(messages)
        outputs_by_id = {
            message.tool_call_id: str(message.content)
            for message in turn_messages
            if isinstance(message, ToolMessage)
        }
        records: list[ToolCallRecord] = []
        for message_index, message in enumerate(turn_messages):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            limit = len(message.tool_calls)
            if message_index == len(turn_messages) - 1:
                limit = min(limit, current_call_index)
            for call in message.tool_calls[:limit]:
                records.append(
                    ToolCallRecord(
                        name=call["name"],
                        args=call.get("args", {}),
                        output=outputs_by_id.get(call.get("id", "")),
                    )
                )
        return records

    @staticmethod
    def _messages_since_last_human(messages: list[Any]) -> list[Any]:
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return messages[index:]
        return messages

    @staticmethod
    def _args_similarity(left: Any, right: Any) -> float:
        left_text = ToolCallGuard._normalize_args(left)
        right_text = ToolCallGuard._normalize_args(right)
        if not left_text or not right_text:
            return 0.0
        left_tokens = set(left_text.split())
        right_tokens = set(right_text.split())
        token_similarity = (
            len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if left_tokens and right_tokens
            else 0.0
        )
        if len(left_tokens) > 1 and len(right_tokens) > 1 and token_similarity == 0:
            return 0.0
        sequence_similarity = SequenceMatcher(None, left_text, right_text).ratio()
        return max(token_similarity, sequence_similarity)

    @staticmethod
    def _normalize_args(args: Any) -> str:
        if isinstance(args, dict) and "query" in args:
            value = args["query"]
        else:
            value = args
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return "".join(char.lower() if char.isalnum() else " " for char in text).strip()


class ToolRunner(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        logger: Optional[logging.Logger] = None,
        tool_call_guard: ToolCallGuard | None = None,
    ) -> None:
        super().__init__(self._func, name=name, tags=tags, trace=False)
        self.logger = logger or logging.getLogger(__name__)
        self.tool_call_guard = tool_call_guard or ToolCallGuard.with_default_policies()
        self.tools_by_name: Dict[str, BaseTool] = {}
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_

    def get_messages(self, input: dict[str, Any]) -> List:
        """Get fields from from input that will be processed."""
        return input.get("messages", [])

    def update_input_with_outputs(
        self, input: dict[str, Any], outputs: List[Any]
    ) -> None:
        """Update input with tool outputs."""
        input["messages"].extend(outputs)

    def _func(self, input: dict[str, Any], config: RunnableConfig) -> Any:
        config["max_concurrency"] = (
            1  # TODO(maciejmajek): use better mechanism for task queueing
        )
        messages = self.get_messages(input)
        if not messages:
            raise ValueError("No message found in input")

        message = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(index_and_call: tuple[int, ToolCall]):
            index, call = index_and_call
            blocked_reason = self.tool_call_guard.check(call, messages, index)
            if blocked_reason is not None:
                self.logger.info(
                    f"Blocked tool call: {call['name']}, args: {call['args']}. "
                    f"Reason: {blocked_reason}"
                )
                return ToolMessage(
                    content=blocked_reason,
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )

            self.logger.info(f"Running tool: {call['name']}, args: {call['args']}")
            artifact = None

            try:
                ts = time.perf_counter()
                output = self.tools_by_name[call["name"]].invoke(call, config)  # type: ignore
                te = time.perf_counter() - ts
                self.logger.info(
                    f"Tool {call['name']} completed in {te:.2f} seconds. Tool output: {str(output.content)[:100]}{'...' if len(str(output.content)) > 100 else ''}"
                )
                self.logger.debug(
                    f"Tool {call['name']} output: \n\n{str(output.content)}"
                )
            except ValidationError as e:
                errors = e.errors()
                for error in errors:
                    error.pop(
                        "url"
                    )  # get rid of the  https://errors.pydantic.dev/... url

                error_message = f"""
                                    Validation error in tool {call["name"]}:
                                    {e.title}
                                    Number of errors: {e.error_count()}
                                    Errors:
                                    {json.dumps(errors, indent=2)}
                                """
                self.logger.info(error_message)
                output = ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            except Exception as e:
                self.logger.info(f'Error in "{call["name"]}", error: {e}')
                output = ToolMessage(
                    content=f"Failed to run tool. Error: {e}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )

            if output.artifact is not None:
                artifact = output.artifact
                if not isinstance(artifact, dict):
                    raise ValueError(
                        "Artifact must be a dictionary with optional keys: 'images', 'audios'"
                    )

                artifact = cast(MultimodalArtifact, artifact)
                store_artifacts(output.tool_call_id, [artifact])

            if artifact is not None and (
                len(artifact.get("images", [])) > 0
                or len(artifact.get("audios", [])) > 0
            ):  # multimodal case, we currently support images and audios artifacts
                return ToolMultimodalMessage(
                    content=msg_content_output(output.content),
                    name=call["name"],
                    tool_call_id=call["id"],
                    images=artifact.get("images", []),
                    audios=artifact.get("audios", []),
                )

            return output

        with get_executor_for_config(config) as executor:
            indexed_tool_calls = list(enumerate(message.tool_calls))
            raw_outputs = [*executor.map(run_one, indexed_tool_calls)]
            outputs: List[Any] = []
            for raw_output in raw_outputs:
                if isinstance(raw_output, ToolMultimodalMessage):
                    outputs.extend(
                        raw_output.postprocess()
                    )  # openai please allow tool messages with images!
                else:
                    outputs.append(raw_output)

            # because we can't answer an aiMessage with an alternating sequence of tool and human messages
            # we sort the messages by type so that the tool messages are sent first
            # for more information see implementation of ToolMultimodalMessage.postprocess
            outputs.sort(key=lambda x: x.__class__.__name__, reverse=True)

            self.update_input_with_outputs(input, outputs)
            return input


class SubAgentToolRunner(ToolRunner):
    """ToolRunner that works with 'step_messages' key used by subagents"""

    def get_messages(self, input: dict[str, Any]) -> List:
        return input.get("step_messages", [])

    def update_input_with_outputs(
        self, input: dict[str, Any], outputs: List[Any]
    ) -> None:
        input["messages"].extend(outputs)
        input["step_messages"].extend(outputs)
