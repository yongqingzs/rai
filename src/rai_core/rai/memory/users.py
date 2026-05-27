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

import time
from typing import List

from rai.memory.manager import MemoryManager


def user_profile_namespace(namespace: str) -> tuple[str, str, str]:
    return (namespace, "__users__", "profiles")


def add_user_profile(memory_mgr: MemoryManager, namespace: str, user_id: str):
    memory_mgr.store.put(
        user_profile_namespace(namespace),
        user_id,
        {"user_id": user_id, "created_at": time.time(), "deleted": False},
    )


def get_user_ids(memory_mgr: MemoryManager, namespace: str) -> List[str]:
    """Get unique user IDs from profiles and memory namespaces."""
    user_ids = {"default"}
    deleted_user_ids = set()
    try:
        profiles = memory_mgr.store.search(
            user_profile_namespace(namespace), query="", limit=200
        )
        for profile in profiles:
            profile_user_id = profile.value.get("user_id", profile.key)
            if profile.value.get("deleted"):
                deleted_user_ids.add(profile_user_id)
                user_ids.discard(profile_user_id)
            else:
                user_ids.add(profile_user_id)
    except Exception:
        pass

    try:
        namespaces = memory_mgr.store.list_namespaces(prefix=(namespace,), limit=200)
        for ns in namespaces:
            if len(ns) >= 2 and ns[1] != "__users__" and ns[1] not in deleted_user_ids:
                user_ids.add(ns[1])
    except Exception:
        pass
    return sorted(user_ids)


def delete_user(memory_mgr: MemoryManager, namespace: str, user_id: str) -> int:
    """Delete a user profile and all long-term memory items for that user."""
    deleted = 0
    try:
        memory_mgr.store.put(
            user_profile_namespace(namespace),
            user_id,
            {"user_id": user_id, "deleted": True, "deleted_at": time.time()},
        )
        deleted += 1
    except Exception:
        pass

    for schema in ("facts", "spatial"):
        ns = (namespace, user_id, schema)
        try:
            items = memory_mgr.store.search(ns, query="", limit=1000)
            for item in items:
                memory_mgr.store.delete(ns, item.key)
                deleted += 1
        except Exception:
            pass
    return deleted
