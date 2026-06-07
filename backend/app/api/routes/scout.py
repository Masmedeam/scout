import json
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlmodel import col, func, select
from starlette.responses import FileResponse

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    ScoutCoveragePublic,
    ScoutExampleImage,
    ScoutFlightEvalPublic,
    ScoutFusionStatePublic,
    ScoutImagePatch,
    ScoutImportResult,
    ScoutIndexStatus,
    ScoutLiveSessionPublic,
    ScoutLiveSessionSummary,
    ScoutLiveTrackPoint,
    ScoutLocation,
    ScoutLocationCreate,
    ScoutLocationPublic,
    ScoutLocationsPublic,
    ScoutRasterAsset,
    ScoutRasterAssetPublic,
    ScoutSearchMatch,
    ScoutSearchRun,
    ScoutSearchRunPublic,
)
from app.scout.coverage import compute_coverage
from app.scout.embedding import active_embedding_model, embed_image
from app.scout.flight_log import evaluate_flight_log, load_manifest, summarize_results
from app.scout.geo import GeoBounds, haversine_meters
from app.scout.live_session import (
    LiveSessionResult,
    LiveTrackPoint,
    get_live_session,
    process_live_video,
)
from app.scout.localization import localize_image
from app.scout.postgres_vector_store import (
    count_postgres_embeddings,
    store_patch_embedding,
)
from app.scout.query_examples import get_query_example_path, read_query_examples
from app.scout.storage import ensure_storage_dirs, save_upload
from app.scout.tiling import tile_raster
from app.scout.vector_store import ScoutVectorStore
from app.scout.vio_fusion import VioDelta, VpsFix, get_tracker

router = APIRouter(prefix="/scout", tags=["scout"])


@router.get("/locations", response_model=ScoutLocationsPublic)
def read_locations(session: SessionDep, _current_user: CurrentUser) -> Any:
    count = session.exec(select(func.count()).select_from(ScoutLocation)).one()
    locations = session.exec(
        select(ScoutLocation).order_by(col(ScoutLocation.created_at).desc())
    ).all()
    return ScoutLocationsPublic(
        data=[ScoutLocationPublic.model_validate(location) for location in locations],
        count=count,
    )


@router.post("/locations", response_model=ScoutLocationPublic)
def create_location(
    *, session: SessionDep, _current_user: CurrentUser, location_in: ScoutLocationCreate
) -> Any:
    _validate_bounds(
        GeoBounds(
            west=location_in.west,
            south=location_in.south,
            east=location_in.east,
            north=location_in.north,
        )
    )
    existing = session.exec(
        select(ScoutLocation).where(ScoutLocation.name == location_in.name)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Location already exists")
    location = ScoutLocation.model_validate(location_in)
    session.add(location)
    session.commit()
    session.refresh(location)
    return location


@router.get("/index/status", response_model=ScoutIndexStatus)
def read_index_status(session: SessionDep, _current_user: CurrentUser) -> Any:
    store = ScoutVectorStore()
    return ScoutIndexStatus(
        locations=session.exec(select(func.count()).select_from(ScoutLocation)).one(),
        raster_assets=session.exec(
            select(func.count()).select_from(ScoutRasterAsset)
        ).one(),
        patches=session.exec(select(func.count()).select_from(ScoutImagePatch)).one(),
        postgres_vectors=count_postgres_embeddings(session),
        redis_available=store.available,
        redis_index=store.index_name,
        embedding_model=active_embedding_model(),
        embedding_dim=store.dim,
    )


@router.get("/index/coverage", response_model=ScoutCoveragePublic)
def read_index_coverage(session: SessionDep, _current_user: CurrentUser) -> Any:
    report = compute_coverage(session)
    return ScoutCoveragePublic(
        coverage_percent=report.coverage_percent,
        patch_count=report.patch_count,
        raster_asset_count=report.raster_asset_count,
        bounds_west=report.bounds_west,
        bounds_south=report.bounds_south,
        bounds_east=report.bounds_east,
        bounds_north=report.bounds_north,
        uncovered_cell_count=len(report.uncovered_cells),
    )


@router.get("/examples", response_model=list[ScoutExampleImage])
def read_examples(
    session: SessionDep, _current_user: CurrentUser, limit: int = 8
) -> Any:
    if limit < 1 or limit > 24:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 24")
    holdouts = read_query_examples()[:limit]
    if holdouts:
        return [
            ScoutExampleImage(
                id=example["id"],
                label=example["label"],
                center_lat=example["center_lat"],
                center_lon=example["center_lon"],
                preview_url=example["preview_url"],
            )
            for example in holdouts
        ]

    statement = (
        select(ScoutImagePatch)
        .join(ScoutRasterAsset)
        .join(ScoutLocation)
        .where(ScoutLocation.name == "San Francisco")
        .order_by(col(ScoutRasterAsset.source_provider), col(ScoutImagePatch.pixel_x))
        .limit(limit)
    )
    patches = session.exec(statement).all()
    return [
        ScoutExampleImage(
            id=str(patch.id),
            label=_example_label(patch),
            center_lat=patch.center_lat,
            center_lon=patch.center_lon,
            preview_url=_patch_preview_url(patch.id),
        )
        for patch in patches
    ]


@router.get("/examples/{example_id}/image")
def read_example_image(
    _current_user: CurrentUser, example_id: str
) -> FileResponse:
    path = get_query_example_path(example_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Example not found")
    return FileResponse(_safe_storage_path(str(path)))


@router.get("/patches/{patch_id}/image")
def read_patch_image(
    session: SessionDep, _current_user: CurrentUser, patch_id: uuid.UUID
) -> FileResponse:
    patch = session.get(ScoutImagePatch, patch_id)
    if patch is None:
        raise HTTPException(status_code=404, detail="Patch not found")
    path = _safe_storage_path(patch.file_path)
    return FileResponse(path)


@router.get("/patches/{patch_id}/download")
def download_patch_image(
    session: SessionDep, _current_user: CurrentUser, patch_id: uuid.UUID
) -> FileResponse:
    patch = session.get(ScoutImagePatch, patch_id)
    if patch is None:
        raise HTTPException(status_code=404, detail="Patch not found")
    path = _safe_storage_path(patch.file_path)
    return FileResponse(path, filename=path.name, media_type="image/jpeg")


@router.post("/imagery/import", response_model=ScoutImportResult)
def import_imagery(
    *,
    session: SessionDep,
    _current_user: CurrentUser,
    image: Annotated[UploadFile, File()],
    location_name: Annotated[str, Form()],
    west: Annotated[float, Form()],
    south: Annotated[float, Form()],
    east: Annotated[float, Form()],
    north: Annotated[float, Form()],
    description: Annotated[str | None, Form()] = None,
    source_provider: Annotated[str, Form()] = "local",
    source_uri: Annotated[str | None, Form()] = None,
    license: Annotated[str | None, Form()] = None,
    capture_date: Annotated[str | None, Form()] = None,
    patch_size: Annotated[int, Form()] = 256,
    overlap: Annotated[int, Form()] = 64,
) -> Any:
    bounds = GeoBounds(west=west, south=south, east=east, north=north)
    _validate_bounds(bounds)

    raster_path = save_upload(image, "rasters")
    try:
        width, height, patch_payloads = tile_raster(
            raster_path, bounds, patch_size=patch_size, overlap=overlap
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    location = _get_or_create_location(
        session=session,
        name=location_name,
        description=description,
        bounds=bounds,
    )
    asset = ScoutRasterAsset(
        location_id=location.id,
        source_provider=source_provider,
        source_uri=source_uri,
        license=license,
        capture_date=capture_date,
        file_path=str(raster_path),
        width=width,
        height=height,
        west=west,
        south=south,
        east=east,
        north=north,
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)

    store = ScoutVectorStore()
    redis_indexed = 0
    for payload in patch_payloads:
        patch = ScoutImagePatch(
            raster_asset_id=asset.id,
            embedding_model=active_embedding_model(),
            embedding_dim=store.dim,
            redis_key="",
            **payload,
        )
        session.add(patch)
        session.flush()
        embedding = embed_image(patch.file_path)
        store_patch_embedding(patch, embedding)
        redis_key = store.upsert_patch(
            patch_id=patch.id,
            location_id=location.id,
            center_lat=patch.center_lat,
            center_lon=patch.center_lon,
            embedding=embedding,
        )
        patch.redis_key = redis_key
        if store.available:
            redis_indexed += 1

    session.commit()
    session.refresh(location)
    session.refresh(asset)
    return ScoutImportResult(
        location=ScoutLocationPublic.model_validate(location),
        raster_asset=ScoutRasterAssetPublic.model_validate(asset),
        patches_created=len(patch_payloads),
        redis_indexed=redis_indexed,
    )


@router.post("/search/image", response_model=ScoutSearchRunPublic)
def search_image(
    *,
    session: SessionDep,
    _current_user: CurrentUser,
    image: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form()] = 5,
) -> Any:
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    query_path = save_upload(image, "queries")
    try:
        _assert_readable_image(query_path)
        localization = localize_image(session=session, query_path=query_path, top_k=top_k)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    matches: list[ScoutSearchMatch] = []
    for localized in localization.matches:
        fine = localized.fine_match
        matches.append(
            ScoutSearchMatch(
                patch_id=localized.patch_id,
                score=localized.combined_score,
                center_lat=localized.center_lat,
                center_lon=localized.center_lon,
                west=localized.west,
                south=localized.south,
                east=localized.east,
                north=localized.north,
                file_path=localized.file_path,
                preview_url=_patch_preview_url(localized.patch_id),
                retrieval_score=localized.retrieval_score,
                fine_match_score=fine.match_score if fine else None,
                fine_match_inliers=fine.inlier_count if fine else None,
                refined_lat=fine.refined_lat if fine else None,
                refined_lon=fine.refined_lon if fine else None,
            )
        )

    run = ScoutSearchRun(
        query_image_path=str(query_path),
        top_k=top_k,
        predicted_lat=localization.predicted_lat,
        predicted_lon=localization.predicted_lon,
        matches=[match.model_dump(mode="json") for match in matches],
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return ScoutSearchRunPublic(
        id=run.id,
        created_at=run.created_at,
        query_image_path=run.query_image_path,
        top_k=run.top_k,
        predicted_lat=run.predicted_lat,
        predicted_lon=run.predicted_lon,
        confidence=localization.confidence,
        method=localization.method,
        matches=matches,
    )


@router.post("/fusion/init", response_model=ScoutFusionStatePublic)
def init_fusion(
    *,
    _current_user: CurrentUser,
    session_id: Annotated[str, Form()],
    lat: Annotated[float, Form()],
    lon: Annotated[float, Form()],
) -> Any:
    state = get_tracker(session_id).initialize(lat=lat, lon=lon)
    return ScoutFusionStatePublic(
        session_id=session_id,
        lat=state.lat,
        lon=state.lon,
        source=state.source,
        vps_confidence=state.vps_confidence,
        timestamp=state.timestamp,
    )


@router.post("/fusion/vio", response_model=ScoutFusionStatePublic)
def fusion_vio_update(
    *,
    _current_user: CurrentUser,
    session_id: Annotated[str, Form()],
    delta_north_m: Annotated[float, Form()],
    delta_east_m: Annotated[float, Form()],
) -> Any:
    tracker = get_tracker(session_id)
    try:
        state = tracker.apply_vio(
            VioDelta(delta_north_m=delta_north_m, delta_east_m=delta_east_m)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ScoutFusionStatePublic(
        session_id=session_id,
        lat=state.lat,
        lon=state.lon,
        source=state.source,
        vps_confidence=state.vps_confidence,
        timestamp=state.timestamp,
    )


@router.post("/fusion/vps", response_model=ScoutFusionStatePublic)
def fusion_vps_update(
    *,
    session: SessionDep,
    _current_user: CurrentUser,
    session_id: Annotated[str, Form()],
    image: Annotated[UploadFile | None, File()] = None,
    lat: Annotated[float | None, Form()] = None,
    lon: Annotated[float | None, Form()] = None,
    confidence: Annotated[float, Form()] = 0.8,
) -> Any:
    fix_lat = lat
    fix_lon = lon
    fix_confidence = confidence

    if image is not None:
        query_path = save_upload(image, "queries")
        try:
            _assert_readable_image(query_path)
            localization = localize_image(session=session, query_path=query_path, top_k=3)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if localization.predicted_lat is None or localization.predicted_lon is None:
            raise HTTPException(status_code=422, detail="VPS could not localize query image")
        fix_lat = localization.predicted_lat
        fix_lon = localization.predicted_lon
        fix_confidence = localization.confidence

    if fix_lat is None or fix_lon is None:
        raise HTTPException(status_code=400, detail="Provide lat/lon or an image for VPS fix")

    tracker = get_tracker(session_id)
    try:
        state = tracker.apply_vps(
            VpsFix(lat=fix_lat, lon=fix_lon, confidence=fix_confidence)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ScoutFusionStatePublic(
        session_id=session_id,
        lat=state.lat,
        lon=state.lon,
        source=state.source,
        vps_confidence=state.vps_confidence,
        timestamp=state.timestamp,
    )


@router.get("/fusion/{session_id}", response_model=ScoutFusionStatePublic)
def read_fusion_state(
    *, _current_user: CurrentUser, session_id: str
) -> Any:
    tracker = get_tracker(session_id)
    if tracker.state is None:
        raise HTTPException(status_code=404, detail="Fusion session not initialized")
    state = tracker.state
    return ScoutFusionStatePublic(
        session_id=session_id,
        lat=state.lat,
        lon=state.lon,
        source=state.source,
        vps_confidence=state.vps_confidence,
        timestamp=state.timestamp,
    )


@router.post("/live/session", response_model=ScoutLiveSessionPublic)
def create_live_session(
    *,
    session: SessionDep,
    _current_user: CurrentUser,
    video: Annotated[UploadFile, File()],
    altitude_m: Annotated[float, Form()] = settings.SCOUT_VO_DEFAULT_ALTITUDE_M,
    fps: Annotated[float, Form()] = settings.SCOUT_LIVE_FRAME_FPS,
) -> Any:
    if fps <= 0 or fps > 10:
        raise HTTPException(status_code=400, detail="fps must be between 0 and 10")
    if altitude_m <= 0 or altitude_m > 500:
        raise HTTPException(status_code=400, detail="altitude_m must be between 0 and 500")

    video_path = save_upload(video, "queries/live-uploads")
    result = process_live_video(
        session=session,
        video_path=video_path,
        altitude_m=altitude_m,
        fps=fps,
    )
    return _live_session_public(result)


@router.get("/live/{session_id}", response_model=ScoutLiveSessionPublic)
def read_live_session(*, _current_user: CurrentUser, session_id: str) -> Any:
    result = get_live_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    return _live_session_public(result)


@router.get("/live/{session_id}/summary", response_model=ScoutLiveSessionSummary)
def read_live_session_summary(*, _current_user: CurrentUser, session_id: str) -> Any:
    result = get_live_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    return ScoutLiveSessionSummary(
        session_id=result.session_id,
        status=result.status,
        frame_count=result.frame_count,
        vps_fix_count=result.vps_fix_count,
        median_confidence=result.median_confidence,
        track_length_m=_track_length_meters(result.track),
    )


@router.post("/flight-log/eval", response_model=ScoutFlightEvalPublic)
def eval_flight_log(
    *,
    session: SessionDep,
    _current_user: CurrentUser,
    manifest: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form()] = 5,
) -> Any:
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    eval_dir = ensure_storage_dirs()["queries"] / "flight-log"
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_dir / manifest.filename
    manifest_path.write_bytes(manifest.file.read())
    frames = load_manifest(manifest_path)
    results = evaluate_flight_log(session=session, frames=frames, top_k=top_k)
    summary = summarize_results(results)
    report_path = eval_dir / "eval_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    return ScoutFlightEvalPublic(
        frames_evaluated=summary.get("frames_evaluated", 0),
        median_error_meters=summary.get("median_error_meters"),
        mean_error_meters=summary.get("mean_error_meters"),
        max_error_meters=summary.get("max_error_meters"),
        within_50m=summary.get("within_50m"),
        within_250m=summary.get("within_250m"),
    )


def _get_or_create_location(
    *, session: SessionDep, name: str, description: str | None, bounds: GeoBounds
) -> ScoutLocation:
    location = session.exec(select(ScoutLocation).where(ScoutLocation.name == name)).first()
    if location:
        return location
    location = ScoutLocation(
        name=name,
        description=description,
        west=bounds.west,
        south=bounds.south,
        east=bounds.east,
        north=bounds.north,
    )
    session.add(location)
    session.commit()
    session.refresh(location)
    return location


def _validate_bounds(bounds: GeoBounds) -> None:
    try:
        bounds.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _assert_readable_image(path: str | Path) -> None:
    with Image.open(path) as image:
        image.verify()


def _patch_preview_url(patch_id: object) -> str:
    return f"{settings.API_V1_STR}/scout/patches/{patch_id}/image"


def _safe_storage_path(path: str) -> Path:
    storage_root = Path(settings.SCOUT_STORAGE_DIR).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="File is outside Scout storage")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return resolved


def _example_label(patch: ScoutImagePatch) -> str:
    stem = Path(patch.file_path).stem.replace("-", " ").replace("_", " ")
    return stem.title()


def _live_session_public(result: LiveSessionResult) -> ScoutLiveSessionPublic:
    return ScoutLiveSessionPublic(
        session_id=result.session_id,
        status=result.status,
        frame_count=result.frame_count,
        vps_fix_count=result.vps_fix_count,
        median_confidence=result.median_confidence,
        track=[
            ScoutLiveTrackPoint(
                frame_id=point.frame_id,
                lat=point.lat,
                lon=point.lon,
                source=point.source,
                vps_confidence=point.vps_confidence,
                timestamp_s=point.timestamp_s,
            )
            for point in result.track
        ],
        error=result.error,
    )


def _track_length_meters(track: list[LiveTrackPoint]) -> float | None:
    if len(track) < 2:
        return None
    total = 0.0
    for prev, curr in zip(track, track[1:], strict=False):
        total += haversine_meters(prev.lat, prev.lon, curr.lat, curr.lon)
    return total
