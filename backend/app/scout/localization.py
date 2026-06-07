"""End-to-end visual localization: retrieval + optional fine geometric re-rank."""

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session

from app.core.config import settings
from app.models import ScoutImagePatch
from app.scout.embedding import embed_image
from app.scout.fine_match import FineMatchResult, fine_match_patch
from app.scout.geo import GeoBounds
from app.scout.postgres_vector_store import (
    rebuild_redis_from_postgres,
    search_patch_embeddings,
)
from app.scout.vector_store import ScoutVectorStore


@dataclass(frozen=True)
class LocalizedMatch:
    patch_id: uuid.UUID
    retrieval_score: float
    center_lat: float
    center_lon: float
    west: float
    south: float
    east: float
    north: float
    file_path: str
    fine_match: FineMatchResult | None = None

    @property
    def predicted_lat(self) -> float:
        if self.fine_match is not None:
            return self.fine_match.refined_lat
        return self.center_lat

    @property
    def predicted_lon(self) -> float:
        if self.fine_match is not None:
            return self.fine_match.refined_lon
        return self.center_lon

    @property
    def combined_score(self) -> float:
        if self.fine_match is not None:
            return self.fine_match.match_score
        return max(0.0, 1.0 - self.retrieval_score)


@dataclass(frozen=True)
class LocalizationResult:
    matches: list[LocalizedMatch]
    predicted_lat: float | None
    predicted_lon: float | None
    confidence: float
    method: str


def localize_image(
    *,
    session: Session,
    query_path: str | Path,
    top_k: int = 5,
    hint_lat: float | None = None,
    hint_lon: float | None = None,
    radius_km: float | None = None,
) -> LocalizationResult:
    store = ScoutVectorStore()
    embedding = embed_image(query_path)
    candidate_k = top_k * settings.SCOUT_RERANK_CANDIDATES if settings.SCOUT_FINEMATCH_ENABLED else top_k

    vector_matches = store.search(embedding, top_k=candidate_k)
    if hint_lat is not None and hint_lon is not None:
        vector_matches = search_patch_embeddings(
            session,
            embedding,
            top_k=candidate_k,
            hint_lat=hint_lat,
            hint_lon=hint_lon,
            radius_km=radius_km,
        )
    elif not vector_matches:
        vector_matches = search_patch_embeddings(session, embedding, top_k=candidate_k)
        if vector_matches and store.available:
            rebuild_redis_from_postgres(session, store, clear=True)

    localized: list[LocalizedMatch] = []
    for vector_match in vector_matches:
        patch = session.get(ScoutImagePatch, vector_match.patch_id)
        if patch is None:
            continue
        fine = None
        if settings.SCOUT_FINEMATCH_ENABLED:
            fine = fine_match_patch(
                query_path=query_path,
                patch_path=patch.file_path,
                bounds=GeoBounds(
                    west=patch.west,
                    south=patch.south,
                    east=patch.east,
                    north=patch.north,
                ),
                patch_width=patch.width,
                patch_height=patch.height,
            )
        localized.append(
            LocalizedMatch(
                patch_id=patch.id,
                retrieval_score=vector_match.score,
                center_lat=patch.center_lat,
                center_lon=patch.center_lon,
                west=patch.west,
                south=patch.south,
                east=patch.east,
                north=patch.north,
                file_path=patch.file_path,
                fine_match=fine,
            )
        )

    ranked = _rank_matches(localized)
    ranked = ranked[:top_k]
    best = ranked[0] if ranked else None
    confidence = _confidence(ranked)

    return LocalizationResult(
        matches=ranked,
        predicted_lat=best.predicted_lat if best else None,
        predicted_lon=best.predicted_lon if best else None,
        confidence=confidence,
        method="retrieval+finematch" if settings.SCOUT_FINEMATCH_ENABLED else "retrieval",
    )


def _rank_matches(matches: list[LocalizedMatch]) -> list[LocalizedMatch]:
    if not matches:
        return []
    if settings.SCOUT_FINEMATCH_ENABLED:
        with_fine = [item for item in matches if item.fine_match is not None]
        if with_fine:
            return sorted(with_fine, key=lambda item: item.combined_score, reverse=True)
    return sorted(matches, key=lambda item: item.retrieval_score)


def _confidence(matches: list[LocalizedMatch]) -> float:
    if not matches:
        return 0.0
    best = matches[0]
    if best.fine_match is not None:
        return min(1.0, best.fine_match.match_score)
    if len(matches) < 2:
        return 0.5
    gap = matches[1].retrieval_score - best.retrieval_score
    if gap >= settings.SCOUT_VPS_MIN_SCORE_GAP:
        return min(1.0, 0.4 + gap * 2)
    return max(0.1, 0.3 + gap)
