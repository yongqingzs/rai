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

from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from rai.memory.graph import MemoryAgentContext, MemoryState, create_memory_react_agent
from rai.memory.long_term import render_long_term_memories
from rai.memory.manager import MemoryManager
from rai.tools.memory import create_memory_tools

MEMORY_SYSTEM_PROMPT_TEMPLATE = """{base_system_prompt}

## Long-Term Memory (Persisted Across Sessions)
The following facts and locations are fully loaded in your context. Use this knowledge when answering questions or planning actions. If none are listed, you have no stored memories yet.
{long_term_memory}

## Available Memory Tools
You have access to memory tools:
- save_fact: Save text facts that should persist across sessions
- save_location: Save structured spatial/location data
- forget_memory: Delete stored memories

Use these tools proactively:
- When the user shares preferences or important information, save them with save_fact
- When you identify or learn about a location with coordinates, use save_location with a pose like: {{"x": 1.0, "y": 2.0, "z": 0.0}}
- When the user asks to forget something, use forget_memory"""


def create_default_memory_tools(
    memory_mgr: MemoryManager,
    namespace: str,
    user_id: str,
) -> list[BaseTool]:
    memory_tools = create_memory_tools(
        store=memory_mgr.store,
        namespace=namespace,
        user_id=user_id,
    )
    return [
        memory_tools["save_fact"],
        memory_tools["save_location"],
        memory_tools["forget"],
    ]


def build_memory_system_prompt(
    base_system_prompt: str,
    long_term_memory: str,
    extra_sections: Sequence[str] | None = None,
) -> str:
    prompt = MEMORY_SYSTEM_PROMPT_TEMPLATE.format(
        base_system_prompt=base_system_prompt,
        long_term_memory=long_term_memory,
    )
    if extra_sections:
        prompt += "\n\n" + "\n\n".join(section for section in extra_sections if section)
    return prompt


def create_memory_agent_with_tools(
    memory_mgr: MemoryManager,
    llm: BaseChatModel,
    base_system_prompt_builder: Callable[[MemoryAgentContext], str],
    namespace: str,
    user_id: str,
    base_tools: Sequence[BaseTool] | None = None,
    extra_tools: Sequence[BaseTool | None] | None = None,
    extra_prompt_sections: Sequence[str] | None = None,
) -> Runnable[MemoryState, MemoryState]:
    tools = [
        *create_default_memory_tools(memory_mgr, namespace=namespace, user_id=user_id),
        *(base_tools or []),
        *[tool for tool in (extra_tools or []) if tool is not None],
    ]

    def system_prompt_builder(context: MemoryAgentContext) -> str:
        long_term_memory = render_long_term_memories(
            memory_mgr.store,
            context.namespace,
            context.user_id,
        )
        return build_memory_system_prompt(
            base_system_prompt=base_system_prompt_builder(context),
            long_term_memory=long_term_memory,
            extra_sections=extra_prompt_sections,
        )

    return create_memory_react_agent(
        memory_mgr=memory_mgr,
        llm=llm,
        tools=tools,
        system_prompt_builder=system_prompt_builder,
    )
