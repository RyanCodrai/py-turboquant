from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("agno")

from agno.knowledge.document import Document

from turbovec import IdMapIndex
from turbovec.agno import TurboQuantVectorDb


class StubEmbedder:
    """Deterministic embedder for tests."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def get_embedding(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.tolist()


def _make_doc(content: str, doc_id: str, embedding_dim: int = 64) -> Document:
    rng = np.random.default_rng(abs(hash(content)) % (2**32))
    v = rng.standard_normal(embedding_dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return Document(id=doc_id, name=doc_id, content=content, embedding=v.tolist())


def test_create_initializes_index():
    db = TurboQuantVectorDb(dim=64, bit_width=4)
    db.create()
    assert db.exists() is False  # empty index
    assert db._index is not None


def test_insert_and_search():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    docs = [_make_doc("apple", "d1"), _make_doc("banana", "d2"), _make_doc("cherry", "d3")]
    db.insert("hash1", docs)

    assert db.exists() is True
    assert len(db._index) == 3

    results = db.search("apple", limit=2)
    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)


def test_upsert_replaces_existing():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    doc1 = _make_doc("version1", "same-id")
    db.insert("h1", [doc1])
    assert db._docs["same-id"]["content"] == "version1"

    doc2 = _make_doc("version2", "same-id")
    db.upsert("h2", [doc2])
    assert db._docs["same-id"]["content"] == "version2"
    assert len(db._index) == 1


def test_delete_by_id():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    docs = [_make_doc("a", "d1"), _make_doc("b", "d2")]
    db.insert("h1", docs)

    assert db.delete_by_id("d1") is True
    assert db.delete_by_id("ghost") is False
    assert len(db._index) == 1
    assert "d1" not in db._docs


def test_delete_by_name():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    doc = _make_doc("content", "d1")
    doc.name = "my-file.pdf"
    db.insert("h1", [doc])

    assert db.delete_by_name("my-file.pdf") is True
    assert db.delete_by_name("nonexistent") is False


def test_delete_by_metadata():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    doc = _make_doc("content", "d1")
    doc.meta_data = {"source": "web"}
    db.insert("h1", [doc])

    assert db.delete_by_metadata({"source": "web"}) is True
    assert len(db._docs) == 0


def test_content_hash_exists():
    db = TurboQuantVectorDb(dim=64, bit_width=4)
    db.create()

    doc = _make_doc("content", "d1")
    db.insert("unique-hash", [doc])

    assert db.content_hash_exists("unique-hash") is True
    assert db.content_hash_exists("other-hash") is False


def test_drop_clears_everything():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    db.insert("h1", [_make_doc("x", "d1")])
    db.drop()

    assert db._index is None
    assert len(db._docs) == 0


def test_save_and_load(tmp_path):
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder, path=str(tmp_path))
    db.create()

    db.insert("h1", [_make_doc("hello", "d1"), _make_doc("world", "d2")])
    db.save()

    db2 = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder, path=str(tmp_path))
    db2.create()  # should load from path

    assert len(db2._index) == 2
    assert "d1" in db2._docs


def test_empty_search_returns_empty():
    embedder = StubEmbedder(dim=64)
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=embedder)
    db.create()

    results = db.search("anything", limit=5)
    assert results == []


def test_search_without_embedder_returns_empty():
    db = TurboQuantVectorDb(dim=64, bit_width=4, embedder=None)
    db.create()
    db.insert("h1", [_make_doc("x", "d1")])

    results = db.search("query", limit=5)
    assert results == []


def test_id_exists():
    db = TurboQuantVectorDb(dim=64, bit_width=4)
    db.create()
    db.insert("h1", [_make_doc("x", "d1")])

    assert db.id_exists("d1") is True
    assert db.id_exists("d2") is False


def test_get_supported_search_types():
    db = TurboQuantVectorDb(dim=64, bit_width=4)
    assert db.get_supported_search_types() == ["similarity"]
