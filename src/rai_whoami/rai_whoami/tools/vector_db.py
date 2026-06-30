# Copyright (C) 2025 Robotec.AI
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
from typing import Any, Literal, Type

from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from rai_whoami.models import EmbodimentSource
from rai_whoami.vector_db.faiss import get_faiss_client, split_documents_for_vector_db


class QueryDatabaseToolInput(BaseModel):
    query: str = Field(..., description="The query to search the database with")


def _format_metadata_value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def format_retrieved_documents(documents: list[Document]) -> str:
    if len(documents) == 0:
        return "No matching documents found."

    formatted: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = _format_metadata_value(metadata.get("source"))
        page = _format_metadata_value(metadata.get("page"))
        content = document.page_content.strip()
        formatted.append(
            f"Result {index}\nSource: {source}\nPage: {page}\nContent:\n{content}"
        )
    return "\n\n".join(formatted)


class QueryDatabaseTool(BaseTool):
    name: str = "query_database"
    description: str = "Query the database with a natural language query"
    args_schema: Type[QueryDatabaseToolInput] = QueryDatabaseToolInput

    database_type: Literal["faiss"] = Field(
        default="faiss", description="The type of database to use"
    )
    root_dir: str = Field(..., description="The root directory of the database")
    embeddings_model: Embeddings | None = None

    k: int = Field(default=4, description="The number of results to return")
    strategy: Literal["vector", "keyword", "hybrid"] = Field(default="vector")
    vector_k: int = Field(default=4)
    keyword_k: int = Field(default=4)
    final_k: int = Field(default=4)
    score_threshold: float | None = Field(default=None)
    distance_strategy: Literal["l2", "cosine", "inner_product"] = Field(default="l2")
    normalize_embeddings: bool = Field(default=False)
    vdb_client: VectorStore | None = None
    keyword_documents: list[Document] = Field(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if "vector_k" not in kwargs:
            self.vector_k = self.k
        if "keyword_k" not in kwargs:
            self.keyword_k = self.k
        if "final_k" not in kwargs:
            self.final_k = self.k
        if self.database_type == "faiss":
            self.vdb_client = get_faiss_client(
                str(Path(self.root_dir) / "generated"),
                self.embeddings_model,
                distance_strategy=self.distance_strategy,
                normalize_embeddings=self.normalize_embeddings,
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        if self.strategy in ("keyword", "hybrid"):
            source = EmbodimentSource.from_directory(self.root_dir)
            self.keyword_documents = split_documents_for_vector_db(source.documentation)

    def _run(self, query: str) -> str:
        documents = self._retrieve(query)
        return format_retrieved_documents(documents[: self.final_k])

    def _retrieve(self, query: str) -> list[Document]:
        if self.strategy == "vector":
            return self._vector_search(query)
        if self.strategy == "keyword":
            return self._keyword_search(query)
        return self._merge_documents(
            [*self._keyword_search(query), *self._vector_search(query)]
        )

    def _vector_search(self, query: str) -> list[Document]:
        if self.score_threshold is None:
            return self.vdb_client.similarity_search(query, k=self.vector_k)
        scored = self.vdb_client.similarity_search_with_score(query, k=self.vector_k)
        return [
            document
            for document, score in scored
            if float(score) <= self.score_threshold
        ]

    def _keyword_search(self, query: str) -> list[Document]:
        scored = [
            (document, _keyword_score(query, document))
            for document in self.keyword_documents
        ]
        scored = [(document, score) for document, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [document for document, _ in scored[: self.keyword_k]]

    def _merge_documents(self, documents: list[Document]) -> list[Document]:
        seen: set[tuple[str, Any]] = set()
        merged: list[Document] = []
        for document in documents:
            metadata = document.metadata or {}
            key = (str(metadata.get("source", "")), metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
        return merged


def _keyword_score(query: str, document: Document) -> float:
    query = query.strip().lower()
    if not query:
        return 0.0

    metadata = document.metadata or {}
    headers = " ".join(
        str(metadata.get(key, ""))
        for key in ("Header 1", "Header 2", "Header 3", "Header 4")
    ).lower()
    content = document.page_content.lower()

    score = 0.0
    if query in headers:
        score += 20.0
    if query in content:
        score += 8.0

    terms = [
        term
        for term in query.replace("，", " ")
        .replace("？", " ")
        .replace("?", " ")
        .replace(",", " ")
        .split()
        if term
    ]
    if not terms:
        terms = [query]
    if len(query) >= 4:
        terms.extend(
            {
                query[index : index + size]
                for size in (2, 3, 4)
                for index in range(0, len(query) - size + 1)
            }
        )
    for term in terms:
        if term in headers:
            score += 6.0
        if term in content:
            score += 2.0
    return score
