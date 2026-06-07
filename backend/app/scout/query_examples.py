import json
from pathlib import Path

from PIL import Image, ImageEnhance
from sqlmodel import Session, select

from app.core.config import settings
from app.models import ScoutImagePatch
from app.scout.embedding import embed_image
from app.scout.geo import GeoBounds, haversine_meters, pixel_to_lat, pixel_to_lon
from app.scout.postgres_vector_store import search_patch_embeddings
from app.scout.storage import ensure_storage_dirs
from app.scout.vector_store import ScoutVectorStore

EXAMPLE_MANIFEST = "examples.json"
MAX_VALIDATION_ERROR_METERS = 250.0


def create_query_examples(session: Session, limit: int = 10) -> list[dict]:
    dirs = ensure_storage_dirs()
    examples_dir = dirs["queries"] / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for old_file in examples_dir.glob("*.jpg"):
        old_file.unlink()

    patches = sorted(
        session.exec(select(ScoutImagePatch)).all(),
        key=lambda item: (-item.center_lat, item.center_lon),
    )
    if not patches:
        return []

    store = ScoutVectorStore()
    examples: list[dict] = []
    for index, patch in enumerate(patches):
        if index % 3:
            continue

        source_path = Path(patch.file_path)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        if not source_path.exists():
            continue

        example_number = len(examples) + 1
        example_id = f"sf-holdout-{example_number:02d}"
        output_path = examples_dir / f"{example_id}.jpg"
        crop_x, crop_y, crop_size = write_transformed_patch_example(
            source_path=source_path,
            output_path=output_path,
            variant_index=index,
        )

        query_embedding = embed_image(output_path)
        if store.available:
            matches = store.search(query_embedding, top_k=1)
        else:
            matches = search_patch_embeddings(session, query_embedding, top_k=1)
        if not matches and store.available:
            matches = search_patch_embeddings(session, query_embedding, top_k=1)
        if not matches:
            output_path.unlink(missing_ok=True)
            continue

        matched_patch = session.get(ScoutImagePatch, matches[0].patch_id)
        if matched_patch is None:
            output_path.unlink(missing_ok=True)
            continue
        error_meters = haversine_meters(
            patch.center_lat,
            patch.center_lon,
            matched_patch.center_lat,
            matched_patch.center_lon,
        )
        if error_meters > MAX_VALIDATION_ERROR_METERS:
            output_path.unlink(missing_ok=True)
            continue

        center_x = crop_x + crop_size / 2
        center_y = crop_y + crop_size / 2
        bounds = GeoBounds(
            west=patch.west,
            south=patch.south,
            east=patch.east,
            north=patch.north,
        )
        examples.append(
            {
                "id": example_id,
                "label": f"SF holdout {example_number:02d}",
                "center_lat": pixel_to_lat(bounds, center_y, patch.height),
                "center_lon": pixel_to_lon(bounds, center_x, patch.width),
                "file_path": str(output_path),
                "preview_url": f"{settings.API_V1_STR}/scout/examples/{example_id}/image",
            }
        )
        if len(examples) >= limit:
            break

    manifest_path(examples_dir).write_text(json.dumps(examples, indent=2))
    return examples


def read_query_examples() -> list[dict]:
    examples_dir = ensure_storage_dirs()["queries"] / "examples"
    manifest = manifest_path(examples_dir)
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())


def get_query_example_path(example_id: str) -> Path | None:
    for example in read_query_examples():
        if example["id"] == example_id:
            return Path(example["file_path"])
    return None


def manifest_path(examples_dir: Path) -> Path:
    return examples_dir / EXAMPLE_MANIFEST


def write_transformed_patch_example(
    *, source_path: Path, output_path: Path, variant_index: int
) -> tuple[int, int, int]:
    crop_sizes = [224, 232, 240, 216, 228]
    offsets = [(8, -6), (-10, 9), (12, 4), (-6, -10), (4, 12)]
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        crop_size = min(crop_sizes[variant_index % len(crop_sizes)], width, height)
        offset_x, offset_y = offsets[variant_index % len(offsets)]
        crop_x = max(0, min(width - crop_size, (width - crop_size) // 2 + offset_x))
        crop_y = max(0, min(height - crop_size, (height - crop_size) // 2 + offset_y))
        crop = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        crop = crop.resize((256, 256), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Contrast(crop).enhance(0.96 + 0.02 * (variant_index % 4))
        crop = ImageEnhance.Brightness(crop).enhance(0.97 + 0.02 * (variant_index % 3))
        crop.save(output_path, format="JPEG", quality=92)
    return crop_x, crop_y, crop_size
