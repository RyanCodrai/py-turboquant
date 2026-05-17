# Agno integration

`turbovec.agno.TurboQuantVectorDb` implements Agno's `VectorDb` interface backed by an `IdMapIndex`, supporting insert, upsert, search, and delete with O(1) id-based operations.

## Install

```bash
pip install turbovec[agno]
```

## Basic usage

```python
from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from turbovec.agno import TurboQuantVectorDb

vector_db = TurboQuantVectorDb(
    dim=1536,
    bit_width=4,
    embedder=OpenAIEmbedder(),
)

knowledge = Knowledge(vector_db=vector_db)
knowledge.load_text("Turbovec compresses vectors to 4 bits per dimension.")

agent = Agent(knowledge=knowledge)
agent.print_response("What does turbovec do?")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | `1536` | Embedding dimension |
| `bit_width` | `int` | `4` | Quantization bits (2 or 4) |
| `embedder` | `Embedder` | `None` | Agno embedder for search queries |
| `path` | `str` | `None` | Directory for save/load persistence |
| `similarity_threshold` | `float` | `None` | Minimum score to include in results |

## Save / load

```python
vector_db = TurboQuantVectorDb(dim=1536, bit_width=4, path="./my-store")
vector_db.create()  # loads from path if exists, else creates new

# ... add documents ...

vector_db.save()  # persists to path
```

## Delete

```python
vector_db.delete_by_id("doc-123")
vector_db.delete_by_name("my-document.pdf")
vector_db.delete_by_metadata({"source": "web"})
```

## Known limitations

- No metadata filtering during search. Post-search filtering can be done at the application layer.
- Side-car docstore uses pickle. Only load from trusted sources.
- Search is brute-force over quantized vectors. Suitable for corpora up to ~10M vectors.
