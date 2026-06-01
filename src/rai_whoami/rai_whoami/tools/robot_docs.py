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

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool

from rai_whoami.config import WhoamiConfig
from rai_whoami.models import EmbodimentSource
from rai_whoami.tools.vector_db import QueryDatabaseTool
from rai_whoami.vector_db import FAISSBuilder


class RobotDocsQueryTool(QueryDatabaseTool):
    name: str = "query_robot_docs"
    description: str = (
        "RAG vector database query tool. Search the robot's static whoami "
        "documentation, including hardware specs, sensors, capabilities, "
        "URDF/documentation details, and operating limits. Use this for robot "
        "documentation questions, not for user preferences, conversation "
        "memory, or learned locations."
    )


def has_vector_db(root_dir: str | Path) -> bool:
    generated_dir = Path(root_dir) / "generated"
    return (
        (generated_dir / "index.faiss").exists()
        and (generated_dir / "index.pkl").exists()
        and (generated_dir / "vdb_kwargs.json").exists()
    )


def ensure_vector_db(
    root_dir: str | Path,
    embeddings_model: Embeddings | None = None,
    build_vector_db: bool = False,
) -> None:
    root_path = Path(root_dir)
    if build_vector_db:
        source = EmbodimentSource.from_directory(root_path)
        FAISSBuilder(root_path / "generated", embedding=embeddings_model).build(source)

    if not has_vector_db(root_path):
        raise FileNotFoundError(
            "Whoami vector DB not found. Expected generated/index.faiss, "
            "generated/index.pkl, and generated/vdb_kwargs.json under "
            f"{root_path}. Build it with `build-whoami {root_path} --build-vector-db` "
            "or set [whoami] build_vector_db = true."
        )


def create_robot_docs_tool(
    config: WhoamiConfig,
    embeddings_model: Embeddings | None = None,
) -> BaseTool | None:
    if not config.enabled:
        return None

    if not config.root_dir:
        raise ValueError("[whoami] root_dir must be set when enabled = true")

    ensure_vector_db(
        config.root_dir,
        embeddings_model=embeddings_model,
        build_vector_db=config.build_vector_db,
    )
    return RobotDocsQueryTool(
        root_dir=config.root_dir,
        embeddings_model=embeddings_model,
        k=config.k,
    )
