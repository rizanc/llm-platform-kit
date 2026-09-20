"""Qdrant-backed store. Production path. Tests don't import this."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vecstore import Chunk, VectorStore


class QdrantVectorStore(VectorStore):
    """Production store wrapping Qdrant client.

    Index config (recommended for >100k vectors):
        hnsw_config:
          m: 16
          ef_construct: 128
        optimizer_config:
          indexing_threshold: 0   # index immediately
        quantization_config:
          scalar: { type: int8, quantile: 0.99 }

    Payload indexes MUST exist on every metadata filter field.
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "chunks", dim: int = 768):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, ScalarQuantization,
            ScalarQuantizationConfig, HnswConfigDiff,
        )
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(type="int8", quantile=0.99, always_ram=True),
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )

    def add(self, chunks: Iterable[Chunk], dense: np.ndarray, sparse: list[dict[str, float]]):
        from qdrant_client.models import PointStruct, SparseVector
        points = []
        chunks = list(chunks)
        for c, d, s in zip(chunks, dense, sparse):
            points.append(PointStruct(
                id=hash(c.chunk_id) & 0x7FFFFFFFFFFFFFFF,  # signed int64
                vector={"dense": d.tolist(), "sparse": SparseVector(indices=list(range(len(s))), values=list(s.values()))},
                payload={**c.__dict__, **c.metadata},
            ))
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, dense, sparse, k, filters=None):
        from qdrant_client.models import SparseVector
        q_filter = _to_qdrant_filter(filters)
        results = self.client.search(
            self.collection,
            query_vector=("dense", dense.tolist()),
            query_filter=q_filter,
            limit=k,
            with_payload=True,
        )
        return [(str(r.id), float(r.score)) for r in results]

    def count(self):
        return self.client.count(self.collection).count

    def snapshot(self, path):
        """Qdrant snapshots are server-side. Local fallback: dump via /collections/{name}/points."""
        from qdrant_client import SnapshotLocation
        snap = self.client.create_snapshot(self.collection)
        # download from snapshot URL — left as exercise for ops.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "snapshot_name.txt").write_text(snap.name)

    def restore(self, path):
        name = (Path(path) / "snapshot_name.txt").read_text().strip()
        self.client.restore_snapshot(self.collection, location=f"{name}")


def _to_qdrant_filter(filters: dict | None):
    if not filters:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
    conditions = []
    for k, v in filters.items():
        if isinstance(v, dict):
            op = v.get("op", "eq")
            target = v["value"]
            if op == "eq":
                conditions.append(FieldCondition(key=k, match=MatchValue(value=target)))
            elif op == "in":
                conditions.append(FieldCondition(key=k, match=MatchAny(any=target)))
            elif op == "gte":
                conditions.append(FieldCondition(key=k, range=Range(gte=target)))
            elif op == "lte":
                conditions.append(FieldCondition(key=k, range=Range(lte=target)))
        else:
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=conditions)