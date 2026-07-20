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

import logging

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from rai.agents.langchain.core import ToolCallGuard, ToolPolicy, ToolRunner
from rai.messages import (
    HumanMultimodalMessage,
    ToolMultimodalMessage,
    preprocess_image,
)
from rai.tools.ros2.cli import ros2_topic


@tool(response_format="content_and_artifact")
def get_image():
    """Get an image from the camera"""
    return "Image retrieved", {
        "images": [preprocess_image("docs/imgs/o3deSimulation.png")]
    }


def test_tool_runner_invalid_call():
    runner = ToolRunner(tools=[ros2_topic], logger=logging.getLogger(__name__))
    tool_call = ToolCall(name="bad_fn", args={"command": "list"}, id="12345")
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)
    assert isinstance(output["messages"][0], AIMessage), (
        "First message is not an AIMessage"
    )
    assert isinstance(output["messages"][1], ToolMessage), (
        "Tool output is not a tool message"
    )
    assert output["messages"][1].status == "error"


def test_tool_runner():
    runner = ToolRunner(tools=[ros2_topic], logger=logging.getLogger(__name__))

    tool_call = ToolCall(name="ros2_topic", args={"command": "list"}, id="12345")
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)
    assert isinstance(output["messages"][0], AIMessage), (
        "First message is not an AIMessage"
    )
    assert isinstance(output["messages"][1], ToolMessage), (
        "Tool output is not a tool message"
    )
    assert len(output["messages"][-1].content) > 0, (
        "Tool output is empty. At least rosout should be visible."
    )


def test_tool_runner_multimodal():
    runner = ToolRunner(
        tools=[ros2_topic, get_image], logger=logging.getLogger(__name__)
    )

    tool_call = ToolCall(name="get_image", args={}, id="12345")
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)

    assert isinstance(output["messages"][0], AIMessage), (
        "First message is not an AIMessage"
    )
    assert isinstance(output["messages"][1], ToolMultimodalMessage), (
        "Tool output is not a multimodal message"
    )
    assert isinstance(output["messages"][2], HumanMultimodalMessage), (
        "Human output is not a multimodal message"
    )


@tool
def echo_tool(value: str) -> str:
    """Echo a value."""
    return value


@tool
def query_robot_docs(query: str) -> str:
    """Search robot docs."""
    return f"docs: {query}"


@tool
def save_fact(fact: str) -> str:
    """Save a fact."""
    return f"saved: {fact}"


def test_tool_runner_blocks_default_excessive_same_tool_calls():
    guard = ToolCallGuard(
        max_total_calls_per_turn=8,
        default_policy=ToolPolicy(max_calls_per_turn=1),
    )
    runner = ToolRunner(tools=[echo_tool], tool_call_guard=guard)
    first_call = ToolCall(name="echo_tool", args={"value": "first"}, id="1")
    second_call = ToolCall(name="echo_tool", args={"value": "second"}, id="2")
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[first_call]),
            ToolMessage(content="first", name="echo_tool", tool_call_id="1"),
            AIMessage(content="", tool_calls=[second_call]),
        ]
    }

    output = runner.invoke(state)

    assert output["messages"][-1].status == "error"
    assert "already called 1 time" in output["messages"][-1].content


def test_tool_runner_blocks_similar_robot_docs_queries():
    runner = ToolRunner(tools=[query_robot_docs, echo_tool])
    first_call = ToolCall(
        name="query_robot_docs",
        args={"query": "weight mass dimensions"},
        id="1",
    )
    echo_call = ToolCall(name="echo_tool", args={"value": "step"}, id="2")
    second_call = ToolCall(
        name="query_robot_docs",
        args={"query": "chassis size weight kg dimensions"},
        id="3",
    )
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[first_call]),
            ToolMessage(
                content="Result 1\nContent:\ncamera size only",
                name="query_robot_docs",
                tool_call_id="1",
            ),
            AIMessage(content="", tool_calls=[echo_call]),
            ToolMessage(content="step", name="echo_tool", tool_call_id="2"),
            AIMessage(content="", tool_calls=[second_call]),
        ]
    }

    output = runner.invoke(state)

    assert output["messages"][-1].status == "error"
    assert "similar arguments" in output["messages"][-1].content


def test_tool_runner_allows_distinct_robot_docs_queries_after_other_tool():
    runner = ToolRunner(tools=[query_robot_docs, echo_tool])
    docs_call = ToolCall(
        name="query_robot_docs",
        args={"query": "camera parameters"},
        id="1",
    )
    echo_call = ToolCall(name="echo_tool", args={"value": "step"}, id="2")
    second_docs_call = ToolCall(
        name="query_robot_docs",
        args={"query": "operating limits"},
        id="3",
    )
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[docs_call]),
            ToolMessage(
                content="camera docs",
                name="query_robot_docs",
                tool_call_id="1",
            ),
            AIMessage(content="", tool_calls=[echo_call]),
            ToolMessage(content="step", name="echo_tool", tool_call_id="2"),
            AIMessage(content="", tool_calls=[second_docs_call]),
        ]
    }

    output = runner.invoke(state)

    assert output["messages"][-1].status == "success"
    assert output["messages"][-1].content == "docs: operating limits"


def test_tool_runner_blocks_similar_memory_save_fact_calls():
    runner = ToolRunner(tools=[save_fact])
    first_call = ToolCall(
        name="save_fact",
        args={"fact": "The user likes green tea."},
        id="1",
    )
    second_call = ToolCall(
        name="save_fact",
        args={"fact": "User likes green tea"},
        id="2",
    )
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[first_call]),
            ToolMessage(content="saved", name="save_fact", tool_call_id="1"),
            AIMessage(content="", tool_calls=[second_call]),
        ]
    }

    output = runner.invoke(state)

    assert output["messages"][-1].status == "error"
    assert "similar arguments" in output["messages"][-1].content
