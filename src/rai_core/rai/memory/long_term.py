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

from typing import Any

from langgraph.store.base import BaseStore

MAX_LONG_TERM_FACTS = 20
MAX_LONG_TERM_SPATIAL = 20
MAX_LONG_TERM_CHARS = 8000


def list_long_term_memory_items(store: BaseStore, namespace: str, user_id: str):
    items = []
    for schema in ("facts", "spatial"):
        ns = (namespace, user_id, schema)
        try:
            for item in store.search(ns, query="", limit=200):
                items.append((schema, ns, item.key, item.value))
        except Exception:
            pass
    return items


def format_long_term_item(schema: str, key: str, value: dict[str, Any]) -> str:
    if schema == "facts":
        text = value.get("text", str(value))
        return f"{text[:80]}{'...' if len(text) > 80 else ''}"

    location = value.get("location", key)
    pose = value.get("pose")
    if pose:
        output = (
            f"{location} ({pose.get('x', '?')}, {pose.get('y', '?')}, "
            f"{pose.get('z', '?')}"
        )
        if pose.get("yaw") is not None:
            output += f", yaw={pose.get('yaw'):.4f}"
        output += ")"
        return output
    return str(location)


def render_long_term_memories(
    store: BaseStore,
    namespace: str,
    user_id: str,
    fact_limit: int = MAX_LONG_TERM_FACTS,
    spatial_limit: int = MAX_LONG_TERM_SPATIAL,
    max_chars: int = MAX_LONG_TERM_CHARS,
) -> str:
    """Render facts and spatial memories for prompt injection."""
    all_memories = []
    limits = {"facts": fact_limit, "spatial": spatial_limit}
    for schema in ("facts", "spatial"):
        ns = (namespace, user_id, schema)
        try:
            items = store.search(ns, query="", limit=limits[schema])
            for item in items:
                all_memories.append(
                    f"- {format_long_term_item(schema, item.key, item.value)}"
                )
        except Exception:
            pass

    if not all_memories:
        return "none yet"
    rendered = "\n".join(all_memories)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip()
        rendered += "\n..."
    return rendered
