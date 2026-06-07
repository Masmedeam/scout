import argparse
import logging
from pathlib import Path
from typing import NamedTuple, cast

import httpx
from PIL import Image
from sqlmodel import Session, select

from app.core.db import engine
from app.models import ScoutImagePatch, ScoutLocation, ScoutRasterAsset
from app.scout.embedding import MODEL_NAME, embed_image
from app.scout.geo import GeoBounds
from app.scout.query_examples import create_query_examples
from app.scout.storage import ensure_storage_dirs
from app.scout.tiling import tile_raster
from app.scout.vector_store import ScoutVectorStore

USGS_NAIP_PLUS_EXPORT = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPPlus/ImageServer/exportImage"
)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class SfChip(NamedTuple):
    name: str
    bounds: GeoBounds


class SfArea(NamedTuple):
    name: str
    lon: float
    lat: float


SF_QUICK_CHIPS = [
    SfChip("golden-gate-park", GeoBounds(-122.510, 37.765, -122.454, 37.790)),
    SfChip("downtown", GeoBounds(-122.420, 37.775, -122.382, 37.802)),
    SfChip("mission", GeoBounds(-122.430, 37.740, -122.390, 37.770)),
    SfChip("presidio-marina", GeoBounds(-122.475, 37.785, -122.425, 37.812)),
]

SF_AREAS = [
    SfArea("downtown", -122.401, 37.790),
    SfArea("financial-district", -122.399, 37.794),
    SfArea("soma", -122.405, 37.778),
    SfArea("mission", -122.414, 37.759),
    SfArea("castro", -122.435, 37.760),
    SfArea("haight", -122.448, 37.771),
    SfArea("golden-gate-park-east", -122.455, 37.769),
    SfArea("golden-gate-park-west", -122.488, 37.770),
    SfArea("presidio", -122.461, 37.798),
    SfArea("marina", -122.438, 37.803),
    SfArea("bayview", -122.393, 37.729),
    SfArea("sunset", -122.487, 37.748),
]

SCALE_LEVELS = [
    ("low-altitude", 0.014, 0.010),
    ("mid-altitude", 0.030, 0.022),
    ("high-altitude", 0.060, 0.044),
]

SF_BOUNDS = GeoBounds(-122.515, 37.703, -122.355, 37.812)


def main() -> None:
    args = parse_args()
    dirs = ensure_storage_dirs()
    seed_dir = dirs["rasters"] / "sf-naip-plus"
    seed_dir.mkdir(parents=True, exist_ok=True)

    store = ScoutVectorStore()
    with Session(engine) as session:
        if args.clear:
            clear_sf_seed(session=session, store=store)

        location = get_or_create_sf_location(session)
        total_patches = 0
        total_indexed = 0
        chips = build_chips(args.profile)
        if args.chips is not None:
            chips = chips[: args.chips]

        for chip in chips:
            raster_path = download_chip(
                chip=chip,
                destination=seed_dir / f"{chip.name}.jpg",
                image_size=args.image_size,
                force=args.force_download,
            )
            width, height, patch_payloads = tile_raster(
                raster_path,
                chip.bounds,
                patch_size=args.patch_size,
                overlap=args.overlap,
            )
            asset = ScoutRasterAsset(
                location_id=location.id,
                source_provider="USGS NAIP Plus",
                source_uri=USGS_NAIP_PLUS_EXPORT,
                license="USGS public domain imagery service",
                capture_date=None,
                file_path=str(raster_path),
                width=width,
                height=height,
                west=chip.bounds.west,
                south=chip.bounds.south,
                east=chip.bounds.east,
                north=chip.bounds.north,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)

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
                patch.redis_key = store.upsert_patch(
                    patch_id=patch.id,
                    location_id=location.id,
                    center_lat=patch.center_lat,
                    center_lon=patch.center_lon,
                    embedding=embedding,
                )
                total_indexed += int(store.available)

            session.commit()
            total_patches += len(patch_payloads)
            logger.info(
                f"Indexed {chip.name}: {len(patch_payloads)} patches "
                f"from {width}x{height} imagery"
            )

    logger.info(
        f"SF seed complete: {len(chips)} chips, {total_patches} patches, "
        f"{total_indexed} Redis upserts"
    )
    if args.examples:
        with Session(engine) as session:
            examples = create_query_examples(session=session, limit=args.examples)
        logger.info(f"Created {len(examples)} holdout query examples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Scout with SF NAIP imagery.")
    parser.add_argument("--profile", choices=("quick", "expanded"), default="expanded")
    parser.add_argument("--chips", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def build_chips(profile: str) -> list[SfChip]:
    if profile == "quick":
        return SF_QUICK_CHIPS
    return [
        SfChip(
            name=f"{area.name}-{scale_name}",
            bounds=bounded_chip(area.lon, area.lat, lon_span, lat_span),
        )
        for area in SF_AREAS
        for scale_name, lon_span, lat_span in SCALE_LEVELS
    ]


def bounded_chip(lon: float, lat: float, lon_span: float, lat_span: float) -> GeoBounds:
    west = max(SF_BOUNDS.west, lon - lon_span / 2)
    east = min(SF_BOUNDS.east, lon + lon_span / 2)
    south = max(SF_BOUNDS.south, lat - lat_span / 2)
    north = min(SF_BOUNDS.north, lat + lat_span / 2)
    return GeoBounds(west=west, south=south, east=east, north=north)


def get_or_create_sf_location(session: Session) -> ScoutLocation:
    location = session.exec(
        select(ScoutLocation).where(ScoutLocation.name == "San Francisco")
    ).first()
    if location:
        return location
    location = ScoutLocation(
        name="San Francisco",
        description="Scout seed area for SF NAIP Plus imagery tests",
        west=-122.515,
        south=37.703,
        east=-122.355,
        north=37.812,
    )
    session.add(location)
    session.commit()
    session.refresh(location)
    return location


def clear_sf_seed(session: Session, store: ScoutVectorStore) -> None:
    location = session.exec(
        select(ScoutLocation).where(ScoutLocation.name == "San Francisco")
    ).first()
    if location:
        session.delete(location)
        session.commit()
    if store.client is not None:
        cursor = 0
        while True:
            cursor, keys = cast(
                tuple[int, list[str]],
                store.client.scan(cursor=cursor, match=f"{store.prefix}*", count=500),
            )
            if keys:
                store.client.delete(*keys)
            if cursor == 0:
                break


def download_chip(
    *, chip: SfChip, destination: Path, image_size: int, force: bool
) -> Path:
    if destination.exists() and not force:
        return destination

    params = {
        "bbox": (
            f"{chip.bounds.west},{chip.bounds.south},"
            f"{chip.bounds.east},{chip.bounds.north}"
        ),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{image_size},{image_size}",
        "format": "jpgpng",
        "f": "image",
    }
    response = httpx.get(
        USGS_NAIP_PLUS_EXPORT,
        params=params,
        timeout=60,
        follow_redirects=True,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)

    with Image.open(destination) as image:
        image.verify()
    return destination


if __name__ == "__main__":
    main()
