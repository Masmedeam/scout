import uuid
from typing import Any, cast

import numpy as np
from sqlmodel import Session, col, select

from app.models import ScoutImagePatch, ScoutRasterAsset
from app.scout.embedding import MODEL_NAME
from app.scout.vector_store import ScoutVectorStore, VectorMatch


def store_patch_embedding(patch: ScoutImagePatch, embedding: np.ndarray) -> None:
    patch.embedding = embedding.astype(np.float32).tolist()


def search_patch_embeddings(
    session: Session, embedding: np.ndarray, top_k: int = 5
) -> list[VectorMatch]:
    vector = embedding.astype(np.float32).tolist()
    embedding_column = cast(Any, ScoutImagePatch).__table__.c.embedding
    distance = embedding_column.cosine_distance(vector)
    rows = session.exec(
        select(ScoutImagePatch.id, distance.label("score"))
        .where(
            col(ScoutImagePatch.embedding).is_not(None),
            ScoutImagePatch.embedding_model == MODEL_NAME,
            ScoutImagePatch.embedding_dim == int(embedding.shape[0]),
        )
        .order_by(distance)
        .limit(top_k)
    ).all()
    return [
        VectorMatch(patch_id=uuid.UUID(str(patch_id)), score=float(score))
        for patch_id, score in rows
    ]


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
            ScoutImagePatch.embedding_model == MODEL_NAME,
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
                ScoutImagePatch.embedding_model == MODEL_NAME,
            )
        ).all()
    )
