import base64
import json
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlmodel import col, func, select
from starlette.responses import FileResponse

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    ScoutExampleImage,
    ScoutImagePatch,
    ScoutImportResult,
    ScoutIndexStatus,
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
from app.scout.embedding import MODEL_NAME, embed_image
from app.scout.geo import GeoBounds
from app.scout.postgres_vector_store import (
    count_postgres_embeddings,
    rebuild_redis_from_postgres,
    search_patch_embeddings,
    store_patch_embedding,
)
from app.scout.query_examples import get_query_example_path, read_query_examples
from app.scout.storage import save_upload
from app.scout.tiling import tile_raster
from app.scout.vector_store import ScoutVectorStore

router = APIRouter(prefix="/scout", tags=["scout"])

SAFETY_CATEGORIES = [
    "smoke",
    "visible fire",
    "person lying on the ground or fallen",
    "aggressive behavior",
    "physical fight",
    "weapon visible",
    "traffic collision",
    "pedestrian in roadway danger",
    "crowd crush or panic",
    "structural collapse",
    "flooding",
    "downed power line",
    "hazardous spill",
    "vehicle driving dangerously",
    "person trapped or stranded",
    "medical distress",
    "unattended child in danger",
    "blocked emergency access",
    "large debris hazard",
    "suspicious unattended package",
]

SAFETY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "emergency_detected": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "scene_description": {"type": "string"},
        "detected_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                    "recommended_action": {"type": "string"},
                    "bounding_box": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "number", "minimum": 0, "maximum": 1},
                            "y": {"type": "number", "minimum": 0, "maximum": 1},
                            "width": {"type": "number", "minimum": 0, "maximum": 1},
                            "height": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                },
                "required": [
                    "category",
                    "severity",
                    "confidence",
                    "evidence",
                    "recommended_action",
                    "bounding_box",
                ],
            },
        },
        "limitations": {"type": "string"},
    },
    "required": [
        "emergency_detected",
        "confidence",
        "summary",
        "scene_description",
        "detected_issues",
        "limitations",
    ],
}


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
        embedding_model=MODEL_NAME,
        embedding_dim=store.dim,
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
            embedding_model=MODEL_NAME,
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
        embedding = embed_image(query_path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    store = ScoutVectorStore()
    vector_matches = store.search(embedding, top_k=top_k)
    if not vector_matches:
        vector_matches = search_patch_embeddings(session, embedding, top_k=top_k)
        if vector_matches and store.available:
            rebuild_redis_from_postgres(session, store, clear=True)
    matches: list[ScoutSearchMatch] = []
    for vector_match in vector_matches:
        patch = session.get(ScoutImagePatch, vector_match.patch_id)
        if patch is None:
            continue
        matches.append(
            ScoutSearchMatch(
                patch_id=patch.id,
                score=vector_match.score,
                center_lat=patch.center_lat,
                center_lon=patch.center_lon,
                west=patch.west,
                south=patch.south,
                east=patch.east,
                north=patch.north,
                file_path=patch.file_path,
                preview_url=_patch_preview_url(patch.id),
            )
        )

    predicted_lat = matches[0].center_lat if matches else None
    predicted_lon = matches[0].center_lon if matches else None
    run = ScoutSearchRun(
        query_image_path=str(query_path),
        top_k=top_k,
        predicted_lat=predicted_lat,
        predicted_lon=predicted_lon,
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
        matches=matches,
    )


@router.post("/analyze/safety")
def analyze_safety(
    *,
    _current_user: CurrentUser,
    image: Annotated[UploadFile, File()],
) -> Any:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured")

    query_path = save_upload(image, "queries")
    try:
        _assert_readable_image(query_path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    image_data = base64.b64encode(Path(query_path).read_bytes()).decode("ascii")
    prompt = (
        "Analyze this single drone livestream frame for public-safety concerns. "
        "Check these 20 categories: "
        f"{', '.join(SAFETY_CATEGORIES)}. "
        "Return emergency_detected=true only when visible evidence suggests an "
        "active or likely urgent safety issue. Do not infer emergency from normal "
        "traffic, normal crowds, rooftops, shadows, blur, or uncertainty alone. "
        "Use detected_issues only for issues with visible evidence. Mention limits "
        "from the single-frame aerial viewpoint. For each detected issue, include "
        "a tight bounding_box around the visible evidence using normalized image "
        "coordinates where x and y are the top-left corner and width and height "
        "are fractions of the full image from 0 to 1."
    )
    payload = {
        "model": settings.OPENAI_SAFETY_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scout_safety_analysis",
                "strict": True,
                "schema": SAFETY_SCHEMA,
            }
        },
    }

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _openai_error_detail(exc.response)
        raise HTTPException(status_code=502, detail=detail)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}")

    data = response.json()
    try:
        analysis = json.loads(_extract_response_text(data))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI returned an invalid safety payload: {exc}"
        )

    analysis["categories_checked"] = SAFETY_CATEGORIES
    analysis["model"] = settings.OPENAI_SAFETY_MODEL
    return analysis


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


def _extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                return content["text"]
    raise KeyError("response text not found")


def _openai_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"OpenAI request failed with status {response.status_code}"
    message = payload.get("error", {}).get("message")
    if isinstance(message, str):
        return message
    return f"OpenAI request failed with status {response.status_code}"
