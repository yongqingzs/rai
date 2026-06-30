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

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rai_whoami.tools.vector_db import QueryDatabaseTool, format_retrieved_documents
from rai_whoami.vector_db.faiss import PrefixedEmbeddings, split_documents_for_vector_db


def test_split_documents_for_vector_db_splits_plain_documents():
    document = Document(
        page_content="alpha beta gamma " * 80,
        metadata={"source": "manual.txt", "page": 3},
    )

    chunks = split_documents_for_vector_db(
        [document],
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert {chunk.metadata["source"] for chunk in chunks} == {"manual.txt"}
    assert {chunk.metadata["page"] for chunk in chunks} == {3}
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_split_documents_for_vector_db_uses_markdown_headers_before_recursive_split():
    document = Document(
        page_content=(
            "# Overview\nROSBot XL overview text.\n\n## Sensors\ncamera lidar imu " * 40
        ),
        metadata={"source": "robot.md"},
    )

    chunks = split_documents_for_vector_db(
        [document],
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "robot.md" for chunk in chunks)
    assert any(chunk.metadata.get("Header 1") == "Overview" for chunk in chunks)
    assert any(chunk.metadata.get("Header 2") == "Sensors" for chunk in chunks)


def test_format_retrieved_documents_outputs_source_page_and_content():
    documents = [
        Document(
            page_content="The maximum payload is 5 kg.",
            metadata={"source": "manual.pdf", "page": 2},
        ),
        Document(
            page_content="The robot has a front RGB camera.",
            metadata={"source": "sensors.md"},
        ),
    ]

    output = format_retrieved_documents(documents)

    assert "Document(" not in output
    assert "Result 1" in output
    assert "Source: manual.pdf" in output
    assert "Page: 2" in output
    assert "Content:\nThe maximum payload is 5 kg." in output
    assert "Source: sensors.md" in output
    assert "Page: unknown" in output


def test_format_retrieved_documents_handles_empty_results():
    assert format_retrieved_documents([]) == "No matching documents found."


def test_keyword_search_prioritizes_exact_markdown_content():
    tool = QueryDatabaseTool.model_construct(
        strategy="keyword",
        keyword_k=2,
        final_k=2,
        keyword_documents=[
            Document(
                page_content="### 1.1 机身尺寸与重量\n- 底盘长度：435 mm\n- 底盘宽度：330 mm",
                metadata={"source": "manual.md", "chunk_index": 1},
            ),
            Document(
                page_content="#### 3.2.1 目标坐标系\n导航目标使用 map 坐标系。",
                metadata={"source": "manual.md", "chunk_index": 2},
            ),
        ],
    )

    results = tool._keyword_search("告诉我机器人的机身尺寸")

    assert results[0].metadata["chunk_index"] == 1


class _RecordingEmbeddings(Embeddings):
    def __init__(self):
        self.document_texts: list[list[str]] = []
        self.query_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return [float(len(text))]


def test_prefixed_embeddings_adds_search_prefixes():
    embeddings = _RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(embeddings)

    wrapped.embed_documents(["hello", "world"])
    wrapped.embed_query("robot size")

    assert embeddings.document_texts == [["search_document: hello", "search_document: world"]]
    assert embeddings.query_texts == ["search_query: robot size"]
