import uuid
from typing import Any, cast

import numpy as np
from sqlmodel import Session, col, select

from app.models import ScoutImagePatch, ScoutRasterAsset
from app.scout.embedding import active_embedding_model
from app.scout.geo import haversine_meters
from app.scout.vector_store import ScoutVectorStore, VectorMatch


def store_patch_embedding(patch: ScoutImagePatch, embedding: np.ndarray) -> None:
    patch.embedding = embedding.astype(np.float32).tolist()


def search_patch_embeddings(
    session: Session,
    embedding: np.ndarray,
    top_k: int = 5,
    *,
    hint_lat: float | None = None,
    hint_lon: float | None = None,
    radius_km: float | None = None,
) -> list[VectorMatch]:
    fetch_k = top_k * 5 if hint_lat is not None and hint_lon is not None else top_k
    vector = embedding.astype(np.float32).tolist()
    embedding_column = cast(Any, ScoutImagePatch).__table__.c.embedding
    distance = embedding_column.cosine_distance(vector)
    rows = session.exec(
        select(
            ScoutImagePatch.id,
            ScoutImagePatch.center_lat,
            ScoutImagePatch.center_lon,
            distance.label("score"),
        )
        .where(
            col(ScoutImagePatch.embedding).is_not(None),
            ScoutImagePatch.embedding_model == active_embedding_model(),
            ScoutImagePatch.embedding_dim == int(embedding.shape[0]),
        )
        .order_by(distance)
        .limit(fetch_k)
    ).all()

    matches = [
        VectorMatch(patch_id=uuid.UUID(str(patch_id)), score=float(score))
        for patch_id, center_lat, center_lon, score in rows
    ]

    if hint_lat is None or hint_lon is None or radius_km is None:
        return matches[:top_k]

    radius_m = radius_km * 1000.0
    filtered = [
        match
        for match in matches
        if _patch_within_radius(session, match.patch_id, hint_lat, hint_lon, radius_m)
    ]
    return filtered[:top_k] if filtered else matches[:top_k]


def _patch_within_radius(
    session: Session,
    patch_id: uuid.UUID,
    hint_lat: float,
    hint_lon: float,
    radius_m: float,
) -> bool:
    patch = session.get(ScoutImagePatch, patch_id)
    if patch is None:
        return False
    return haversine_meters(hint_lat, hint_lon, patch.center_lat, patch.center_lon) <= radius_m


def rebuild_redis_from_postgres(
    session: Session, store: ScoutVectorStore, *, clear: bool = False
) -> int:
    if not store.available:
        return 0
    if clear:
        store.reset_index()
    else:
        store.ensure_index()

    rows = session.exec(
        select(ScoutImagePatch, ScoutRasterAsset.location_id)
        .join(ScoutRasterAsset)
        .where(
            col(ScoutImagePatch.embedding).is_not(None),
            ScoutImagePatch.embedding_model == active_embedding_model(),
            ScoutImagePatch.embedding_dim == store.dim,
        )
    ).all()
    indexed = 0
    for patch, location_id in rows:
        if patch.embedding is None:
            continue
        patch.redis_key = store.upsert_patch(
            patch_id=patch.id,
            location_id=location_id,
            center_lat=patch.center_lat,
            center_lon=patch.center_lon,
            embedding=np.asarray(patch.embedding, dtype=np.float32),
        )
        indexed += 1
    session.commit()
    return indexed


def count_postgres_embeddings(session: Session) -> int:
    return len(
        session.exec(
            select(ScoutImagePatch.id).where(
                col(ScoutImagePatch.embedding).is_not(None),
                ScoutImagePatch.embedding_model == active_embedding_model(),
            )
        ).all()
    )
