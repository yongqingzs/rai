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

import inspect
import json
import os
from importlib import import_module
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rai.initialization import get_embeddings_model

from rai_whoami.models import EmbodimentSource
from rai_whoami.vector_db.builder import VectorDBBuilder

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
FAISS_DISTANCE_STRATEGIES = {
    "l2": DistanceStrategy.EUCLIDEAN_DISTANCE,
    "cosine": DistanceStrategy.COSINE,
    "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
}
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


def _is_markdown_document(document: Document) -> bool:
    source = str(document.metadata.get("source", "")).lower()
    return source.endswith(".md") or source.endswith(".markdown")


def split_documents_for_vector_db(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split loaded documentation into chunks suitable for vector search."""
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    chunks: list[Document] = []
    for document in documents:
        if _is_markdown_document(document):
            markdown_chunks = markdown_splitter.split_text(document.page_content)
            for markdown_chunk in markdown_chunks:
                markdown_chunk.metadata = {
                    **document.metadata,
                    **markdown_chunk.metadata,
                }
            split_chunks = recursive_splitter.split_documents(markdown_chunks)
        else:
            split_chunks = recursive_splitter.split_documents([document])

        for index, chunk in enumerate(split_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": index,
            }
        chunks.extend(split_chunks)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class NormalizedEmbeddings(Embeddings):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            _normalize_vector(vector)
            for vector in self.embeddings.embed_documents(texts)
        ]

    def embed_query(self, text: str) -> List[float]:
        return _normalize_vector(self.embeddings.embed_query(text))


def _prepare_embeddings(
    embeddings: Embeddings,
    distance_strategy: str,
    normalize_embeddings: bool,
) -> tuple[Embeddings, bool]:
    if distance_strategy == "inner_product" and normalize_embeddings:
        return NormalizedEmbeddings(embeddings), False
    return embeddings, normalize_embeddings


class FAISSBuilder(VectorDBBuilder):
    def __init__(
        self,
        root_dir: str = "faiss/",
        embedding: Optional[Embeddings] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        distance_strategy: str = "l2",
        normalize_embeddings: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.distance_strategy = distance_strategy
        self.normalize_embeddings = normalize_embeddings
        if embedding is None:
            embedding, model_kwargs = cast(
                Tuple[Embeddings, Dict[str, Any]],
                get_embeddings_model(return_kwargs=True),
            )
        embedding, self.faiss_normalize_l2 = _prepare_embeddings(
            embedding,
            distance_strategy=self.distance_strategy,
            normalize_embeddings=self.normalize_embeddings,
        )
        super().__init__(
            root_dir=root_dir, embedding=embedding, model_kwargs=model_kwargs
        )

    def _build(self, data: EmbodimentSource):
        if len(data.documentation) == 0:
            raise ValueError("No documents found")
        os.makedirs(self.root_dir, exist_ok=True)
        documents = split_documents_for_vector_db(
            data.documentation,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if len(documents) == 0:
            raise ValueError("No document chunks found")
        db = FAISS.from_documents(
            documents,
            self.embedding,
            distance_strategy=FAISS_DISTANCE_STRATEGIES[self.distance_strategy],
            normalize_L2=self.faiss_normalize_l2,
        )
        db.save_local(self.root_dir.as_posix())
        c = str(db.__class__).strip("<>").replace("class '", "").replace("'", "")
        new_kwargs = {
            "vectorstore": {"class": c},
            "embeddings": self.model_kwargs,
            "chunking": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "markdown_headers": MARKDOWN_HEADERS_TO_SPLIT_ON,
            },
            "retrieval": {
                "distance_strategy": self.distance_strategy,
                "normalize_embeddings": self.normalize_embeddings,
            },
        }
        self.model_kwargs = new_kwargs
        self.dump_model_kwargs()
        return db


def get_class_from_string(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def initialize_embeddings(class_path: str, **kwargs: Any) -> Embeddings:
    c = get_class_from_string(class_path)
    kwargs = {k: kwargs[k] for k in inspect.signature(c).parameters if k in kwargs}
    return c(**kwargs)


def get_faiss_client(
    root_dir: str,
    embeddings_model: Embeddings | None = None,
    distance_strategy: str | None = None,
    normalize_embeddings: bool | None = None,
) -> FAISS:
    vdb_kwargs = json.load(open(Path(root_dir) / "vdb_kwargs.json"))
    if embeddings_model is None:
        embeddings_model = initialize_embeddings(
            vdb_kwargs["embeddings"]["class"], **vdb_kwargs["embeddings"]
        )
    retrieval_kwargs = vdb_kwargs.get("retrieval", {})
    distance_strategy = distance_strategy or retrieval_kwargs.get(
        "distance_strategy", "l2"
    )
    normalize_embeddings = (
        normalize_embeddings
        if normalize_embeddings is not None
        else retrieval_kwargs.get("normalize_embeddings", False)
    )
    embeddings_model, faiss_normalize_l2 = _prepare_embeddings(
        embeddings_model,
        distance_strategy=distance_strategy,
        normalize_embeddings=normalize_embeddings,
    )

    vdb_client = FAISS.load_local(
        folder_path=root_dir,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True,
        distance_strategy=FAISS_DISTANCE_STRATEGIES[distance_strategy],
        normalize_L2=faiss_normalize_l2,
    )
    return vdb_client
