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

from langchain_core.tools import BaseTool

import rai_whoami.tools.robot_docs as robot_docs
from rai_whoami import WhoamiConfig, create_robot_docs_tool, load_whoami_config


def test_load_whoami_config_reads_whoami_section(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[whoami]
enabled = true
root_dir = "docs/robot"
build_vector_db = true
k = 7

[whoami.retrieval]
strategy = "hybrid"
vector_k = 9
keyword_k = 5
final_k = 3
score_threshold = 0.8
normalize_embeddings = true
distance_strategy = "cosine"
"""
    )

    config = load_whoami_config(str(config_path))

    assert config.enabled is True
    assert config.root_dir == "docs/robot"
    assert config.build_vector_db is True
    assert config.k == 7
    assert config.retrieval.strategy == "hybrid"
    assert config.retrieval.vector_k == 9
    assert config.retrieval.keyword_k == 5
    assert config.retrieval.final_k == 3
    assert config.retrieval.score_threshold == 0.8
    assert config.retrieval.normalize_embeddings is True
    assert config.retrieval.distance_strategy == "cosine"


def test_create_robot_docs_tool_returns_none_when_disabled():
    tool = create_robot_docs_tool(WhoamiConfig(enabled=False))

    assert tool is None


def test_create_robot_docs_tool_wraps_whoami_query_tool(monkeypatch, tmp_path):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    for filename in ("index.faiss", "index.pkl", "vdb_kwargs.json"):
        (generated_dir / filename).write_text("{}")

    monkeypatch.setattr(
        robot_docs.RobotDocsQueryTool,
        "__init__",
        lambda self, **kwargs: BaseTool.__init__(self, **kwargs),
    )

    tool = create_robot_docs_tool(
        WhoamiConfig(enabled=True, root_dir=str(tmp_path), k=3),
        embeddings_model=None,
    )

    assert tool.name == "query_robot_docs"
    assert "static whoami documentation" in tool.description
    assert tool.root_dir == str(tmp_path)
    assert tool.k == 3


def test_create_robot_docs_tool_builds_vector_db_when_configured(
    monkeypatch,
    tmp_path,
):
    calls = []

    class _Source:
        @classmethod
        def from_directory(cls, root_dir):
            calls.append(("source", root_dir))
            return "source"

    class _Builder:
        def __init__(
            self,
            root_dir,
            embedding=None,
            distance_strategy="l2",
            normalize_embeddings=False,
        ):
            calls.append(
                (
                    "builder",
                    root_dir,
                    embedding,
                    distance_strategy,
                    normalize_embeddings,
                )
            )

        def build(self, source):
            calls.append(("build", source))

    monkeypatch.setattr(robot_docs, "EmbodimentSource", _Source)
    monkeypatch.setattr(robot_docs, "FAISSBuilder", _Builder)
    monkeypatch.setattr(robot_docs, "has_vector_db", lambda root_dir: True)
    monkeypatch.setattr(
        robot_docs.RobotDocsQueryTool,
        "__init__",
        lambda self, **kwargs: BaseTool.__init__(self, **kwargs),
    )
    create_robot_docs_tool(
        WhoamiConfig(
            enabled=True,
            root_dir=str(tmp_path),
            build_vector_db=True,
            retrieval={
                "distance_strategy": "cosine",
                "normalize_embeddings": True,
            },
        ),
        embeddings_model=None,
    )

    assert calls == [
        ("source", tmp_path),
        ("builder", tmp_path / "generated", None, "cosine", True),
        ("build", "source"),
    ]
