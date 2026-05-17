"""Agno VectorDb backed by turbovec's quantized index.

Install with: ``pip install turbovec[agno]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._turbovec import IdMapIndex

try:
    from agno.knowledge.document import Document
    from agno.knowledge.embedder import Embedder
    from agno.vectordb.base import VectorDb
    from agno.vectordb.distance import Distance
except ImportError as exc:
    raise ImportError(
        "agno is required to use turbovec.agno. "
        "Install with: pip install turbovec[agno]"
    ) from exc


_INDEX_FILENAME = "index.tvim"
_STORE_FILENAME = "docstore.pkl"


class TurboQuantVectorDb(VectorDb):
    """Agno VectorDb backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2-4 bits per dimension using TurboQuant.
    Supports insert, upsert, search, and delete operations.
    """

    def __init__(
        self,
        *,
        dim: int = 1536,
        bit_width: int = 4,
        embedder: Optional[Embedder] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "turbovec", **kwargs)
        self.dim = dim
        self.bit_width = bit_width
        self.embedder = embedder
        self.path = path

        self._index: Optional[IdMapIndex] = None
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._str_to_u64: Dict[str, int] = {}
        self._u64_to_str: Dict[int, str] = {}
        self._next_u64: int = 0

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    def create(self) -> None:
        if self._index is None:
            if self.path and Path(self.path).exists():
                self._load(self.path)
            else:
                self._index = IdMapIndex(self.dim, self.bit_width)

    async def async_create(self) -> None:
        self.create()

    def name_exists(self, name: str) -> bool:
        return name in self._docs

    async def async_name_exists(self, name: str) -> bool:
        return self.name_exists(name)

    def id_exists(self, id: str) -> bool:
        return id in self._str_to_u64

    def content_hash_exists(self, content_hash: str) -> bool:
        for doc_meta in self._docs.values():
            if doc_meta.get("content_hash") == content_hash:
                return True
        return False

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._index is None:
            self.create()

        for doc in documents:
            if doc.embedding is None:
                continue
            vec = np.asarray(doc.embedding, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec[None, :]
            if not vec.flags["C_CONTIGUOUS"]:
                vec = np.ascontiguousarray(vec)

            handle = self._issue_handle()
            handle_arr = np.array([handle], dtype=np.uint64)
            self._index.add_with_ids(vec, handle_arr)

            doc_id = doc.id or str(handle)
            self._str_to_u64[doc_id] = handle
            self._u64_to_str[handle] = doc_id
            self._docs[doc_id] = {
                "name": doc.name,
                "content": doc.content,
                "meta_data": doc.meta_data,
                "content_hash": content_hash,
            }

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.insert(content_hash, documents, filters)

    def upsert_available(self) -> bool:
        return True

    def upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._index is None:
            self.create()

        for doc in documents:
            doc_id = doc.id
            if doc_id and doc_id in self._str_to_u64:
                self.delete_by_id(doc_id)
        self.insert(content_hash, documents, filters)

    async def async_upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.upsert(content_hash, documents, filters)

    def search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
        if self._index is None or len(self._index) == 0:
            return []

        if self.embedder is None:
            return []

        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            return []

        qvec = np.asarray(query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        k = min(limit, len(self._index))
        scores, handles = self._index.search(qvec, k)

        results: List[Document] = []
        for score, handle in zip(scores[0], handles[0]):
            doc_id = self._u64_to_str.get(int(handle))
            if doc_id is None:
                continue
            doc_data = self._docs.get(doc_id)
            if doc_data is None:
                continue

            if self.similarity_threshold and float(score) < self.similarity_threshold:
                continue

            results.append(
                Document(
                    id=doc_id,
                    name=doc_data.get("name"),
                    content=doc_data.get("content", ""),
                    meta_data=doc_data.get("meta_data", {}),
                )
            )
        return results

    async def async_search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
        return self.search(query, limit, filters)

    def drop(self) -> None:
        self._index = None
        self._docs.clear()
        self._str_to_u64.clear()
        self._u64_to_str.clear()
        self._next_u64 = 0

    async def async_drop(self) -> None:
        self.drop()

    def exists(self) -> bool:
        return self._index is not None and len(self._index) > 0

    async def async_exists(self) -> bool:
        return self.exists()

    def delete(self) -> bool:
        self.drop()
        return True

    def delete_by_id(self, id: str) -> bool:
        handle = self._str_to_u64.pop(id, None)
        if handle is None:
            return False
        self._u64_to_str.pop(handle, None)
        self._docs.pop(id, None)
        self._index.remove(handle)
        return True

    def delete_by_name(self, name: str) -> bool:
        to_delete = [did for did, meta in self._docs.items() if meta.get("name") == name]
        for did in to_delete:
            self.delete_by_id(did)
        return len(to_delete) > 0

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        to_delete = []
        for did, meta in self._docs.items():
            doc_meta = meta.get("meta_data", {})
            if all(doc_meta.get(k) == v for k, v in metadata.items()):
                to_delete.append(did)
        for did in to_delete:
            self.delete_by_id(did)
        return len(to_delete) > 0

    def delete_by_content_id(self, content_id: str) -> bool:
        return self.delete_by_id(content_id)

    def get_supported_search_types(self) -> List[str]:
        return ["similarity"]

    def save(self, folder_path: Optional[str] = None) -> None:
        path = folder_path or self.path
        if path is None:
            return
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "wb") as f:
            pickle.dump(
                {
                    "docs": self._docs,
                    "str_to_u64": self._str_to_u64,
                    "next_u64": self._next_u64,
                },
                f,
            )

    def _load(self, folder_path: str) -> None:
        folder = Path(folder_path)
        self._index = IdMapIndex.load(str(folder / _INDEX_FILENAME))
        store_path = folder / _STORE_FILENAME
        if store_path.exists():
            with open(store_path, "rb") as f:
                state = pickle.load(f)
            self._docs = state["docs"]
            self._str_to_u64 = state["str_to_u64"]
            self._next_u64 = state["next_u64"]
            self._u64_to_str = {h: s for s, h in self._str_to_u64.items()}


__all__ = ["TurboQuantVectorDb"]
