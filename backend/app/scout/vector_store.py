import uuid
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import redis

from app.core.config import settings
from app.scout.observability import scout_op


@dataclass(frozen=True)
class VectorMatch:
    patch_id: uuid.UUID
    score: float


class ScoutVectorStore:
    def __init__(self) -> None:
        self.index_name = settings.SCOUT_REDIS_INDEX
        self.dim = settings.SCOUT_EMBEDDING_DIM
        self.prefix = "scout:patch:"
        self.client: redis.Redis | None = None
        try:
            self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
            self.client.ping()
        except Exception:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def ensure_index(self) -> None:
        if self.client is None:
            return
        try:
            raw_info = self.client.execute_command("FT.INFO", self.index_name)
            existing_dim = _extract_vector_dim(raw_info)
            if existing_dim == self.dim:
                return
            self.reset_index()
            return
        except Exception:
            pass

        try:
            self.client.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self.prefix,
                "SCHEMA",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                self.dim,
                "DISTANCE_METRIC",
                "COSINE",
                "patch_id",
                "TAG",
                "location_id",
                "TAG",
                "center_lat",
                "NUMERIC",
                "center_lon",
                "NUMERIC",
            )
        except Exception:
            return

    def reset_index(self) -> None:
        if self.client is None:
            return
        try:
            self.client.execute_command("FT.DROPINDEX", self.index_name, "DD")
        except Exception:
            pass
        self.ensure_index()

    def clear_vectors(self) -> None:
        if self.client is None:
            return
        cursor = 0
        while True:
            cursor, keys = cast(
                tuple[int, list[bytes]],
                self.client.scan(cursor=cursor, match=f"{self.prefix}*", count=500),
            )
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break

    @scout_op("upsert_patch_embedding")
    def upsert_patch(
        self,
        patch_id: uuid.UUID,
        location_id: uuid.UUID,
        center_lat: float,
        center_lon: float,
        embedding: np.ndarray,
    ) -> str:
        key = f"{self.prefix}{patch_id}"
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim}-dim embedding, got {embedding.shape}")
        if self.client is None:
            return key
        self.ensure_index()
        vector = embedding.astype(np.float32).tobytes()
        self.client.hset(
            key,
            mapping={
                "patch_id": str(patch_id),
                "location_id": str(location_id),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "embedding": vector,
            },
        )
        return key

    @scout_op("search_patch_embeddings")
    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[VectorMatch]:
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim}-dim embedding, got {embedding.shape}")
        if self.client is None:
            return []
        self.ensure_index()
        vector = embedding.astype(np.float32).tobytes()
        try:
            raw = self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                f"*=>[KNN {top_k} @embedding $vec AS score]",
                "PARAMS",
                "2",
                "vec",
                vector,
                "SORTBY",
                "score",
                "RETURN",
                "2",
                "patch_id",
                "score",
                "DIALECT",
                "2",
            )
            return self._parse_search_results(raw)
        except Exception:
            return self._linear_scan(embedding, top_k)

    def _parse_search_results(self, raw: list[Any]) -> list[VectorMatch]:
        matches: list[VectorMatch] = []
        for index in range(2, len(raw), 2):
            fields = raw[index]
            data = _field_list_to_dict(fields)
            patch_id = data.get(b"patch_id") or data.get("patch_id")
            score = data.get(b"score") or data.get("score") or 0
            if patch_id is None:
                continue
            matches.append(
                VectorMatch(
                    patch_id=uuid.UUID(_decode(patch_id)),
                    score=float(_decode(score)),
                )
            )
        return matches

    def _linear_scan(self, embedding: np.ndarray, top_k: int) -> list[VectorMatch]:
        if self.client is None:
            return []
        cursor = 0
        matches: list[VectorMatch] = []
        query = embedding.astype(np.float32)
        while True:
            cursor, keys = cast(
                tuple[int, list[bytes]],
                self.client.scan(cursor=cursor, match=f"{self.prefix}*", count=500),
            )
            for key in keys:
                data = cast(dict[bytes, bytes], self.client.hgetall(_decode(key)))
                patch_id = data.get(b"patch_id")
                vector = data.get(b"embedding")
                if patch_id is None or vector is None:
                    continue
                candidate = np.frombuffer(vector, dtype=np.float32)
                if candidate.shape != query.shape:
                    continue
                distance = 1 - float(np.dot(query, candidate))
                matches.append(VectorMatch(uuid.UUID(_decode(patch_id)), distance))
            if cursor == 0:
                break
        return sorted(matches, key=lambda item: item.score)[:top_k]


def _field_list_to_dict(fields: list[Any]) -> dict[Any, Any]:
    return {fields[index]: fields[index + 1] for index in range(0, len(fields), 2)}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _extract_vector_dim(raw_info: Any) -> int | None:
    if not isinstance(raw_info, list):
        return None
    for index, value in enumerate(raw_info):
        if value == b"attributes" or value == "attributes":
            attributes = raw_info[index + 1] if index + 1 < len(raw_info) else []
            if not isinstance(attributes, list):
                return None
            for attribute in attributes:
                if not isinstance(attribute, list):
                    continue
                data = _field_list_to_dict(attribute)
                identifier = data.get(b"identifier") or data.get("identifier")
                if _decode(identifier) != "embedding":
                    continue
                dim = data.get(b"dim") or data.get("dim") or data.get(b"DIM") or data.get("DIM")
                if dim is not None:
                    return int(_decode(dim))
    return None
