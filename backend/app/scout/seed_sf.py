import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import httpx
from PIL import Image
from sqlmodel import Session, select

from app.core.db import engine
from app.models import ScoutImagePatch, ScoutLocation, ScoutRasterAsset
from app.scout.embedding import MODEL_NAME, embed_image
from app.scout.geo import GeoBounds
from app.scout.postgres_vector_store import store_patch_embedding
from app.scout.query_examples import create_query_examples
from app.scout.storage import ensure_storage_dirs
from app.scout.tiling import tile_raster
from app.scout.vector_store import ScoutVectorStore

USGS_NAIP_PLUS_EXPORT = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPPlus/ImageServer/exportImage"
)
NOAA_RGB_8BIT_EXPORT = (
    "https://maps1.coast.noaa.gov/arcgis/rest/services/"
    "Imagery/3Band_RGB_8Bit_Imagery/ImageServer/exportImage"
)
SENTINEL2_EXPORT = (
    "https://sentinel.arcgis.com/arcgis/rest/services/"
    "Sentinel2/ImageServer/exportImage"
)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ArcgisImageSource(NamedTuple):
    slug: str
    provider: str
    export_url: str
    license: str
    source_uri: str
    rendering_rule: str | None = None
    capture_date: str | None = None
    legacy_stem: bool = False


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

SF_DIVERSE_AREAS = [
    *SF_AREAS,
    SfArea("civic-center", -122.419, 37.779),
    SfArea("north-beach-chinatown", -122.408, 37.800),
    SfArea("embarcadero-ferry-building", -122.393, 37.795),
    SfArea("oracle-park", -122.389, 37.778),
    SfArea("chase-center-mission-bay", -122.387, 37.768),
    SfArea("twin-peaks", -122.447, 37.752),
    SfArea("ocean-beach", -122.510, 37.758),
    SfArea("lake-merced", -122.494, 37.724),
    SfArea("sf-zoo", -122.503, 37.733),
    SfArea("balboa-park", -122.443, 37.725),
    SfArea("port-of-sf", -122.379, 37.753),
    SfArea("hunters-point", -122.373, 37.728),
    SfArea("treasure-island", -122.371, 37.824),
    SfArea("yerba-buena-island", -122.365, 37.811),
    SfArea("alcatraz", -122.423, 37.827),
    SfArea("golden-gate-bridge-south", -122.479, 37.808),
    SfArea("bay-bridge-west", -122.377, 37.798),
]

SCALE_LEVELS = [
    ("low-altitude", 0.014, 0.010),
    ("mid-altitude", 0.030, 0.022),
    ("high-altitude", 0.060, 0.044),
]

SF_BOUNDS = GeoBounds(-122.515, 37.703, -122.355, 37.835)

IMAGERY_SOURCES = {
    "usgs-naip-plus": ArcgisImageSource(
        slug="usgs-naip-plus",
        provider="USGS NAIP Plus",
        export_url=USGS_NAIP_PLUS_EXPORT,
        license="USGS public domain imagery service",
        source_uri=USGS_NAIP_PLUS_EXPORT,
        legacy_stem=True,
    ),
    "usgs-naip-plus-false-color": ArcgisImageSource(
        slug="usgs-naip-plus-false-color",
        provider="USGS NAIP Plus false color",
        export_url=USGS_NAIP_PLUS_EXPORT,
        license="USGS public domain imagery service",
        source_uri=USGS_NAIP_PLUS_EXPORT,
        rendering_rule='{"rasterFunction":"FalseColorComposite"}',
    ),
    "noaa-rgb-8bit": ArcgisImageSource(
        slug="noaa-rgb-8bit",
        provider="NOAA Coastal Imagery RGB 8-bit",
        export_url=NOAA_RGB_8BIT_EXPORT,
        license="NOAA public imagery service; verify dataset-specific metadata",
        source_uri=NOAA_RGB_8BIT_EXPORT,
    ),
    "sentinel2-natural-color": ArcgisImageSource(
        slug="sentinel2-natural-color",
        provider="Sentinel-2 natural color",
        export_url=SENTINEL2_EXPORT,
        license=(
            "Source: Esri, European Commission, European Space Agency, "
            "Amazon Web Services"
        ),
        source_uri=SENTINEL2_EXPORT,
        rendering_rule='{"rasterFunction":"Natural Color with DRA"}',
    ),
    "sentinel2-color-infrared": ArcgisImageSource(
        slug="sentinel2-color-infrared",
        provider="Sentinel-2 color infrared",
        export_url=SENTINEL2_EXPORT,
        license=(
            "Source: Esri, European Commission, European Space Agency, "
            "Amazon Web Services"
        ),
        source_uri=SENTINEL2_EXPORT,
        rendering_rule='{"rasterFunction":"Color Infrared with DRA"}',
    ),
}


def main() -> None:
    args = parse_args()
    dirs = ensure_storage_dirs()

    store = ScoutVectorStore()
    with Session(engine) as session:
        if args.clear:
            clear_sf_seed(session=session, store=store)

        location = get_or_create_sf_location(session)
        total_patches = 0
        total_indexed = 0
        total_skipped = 0
        sources = build_sources(args.sources, args.profile)
        chips = build_chips(args.profile)
        if args.chips is not None:
            chips = chips[: args.chips]

        for source in sources:
            source_dir = dirs["rasters"] / source.slug
            source_dir.mkdir(parents=True, exist_ok=True)

            for chip in chips:
                destination = source_dir / raster_filename(source=source, chip=chip)
                if not args.reindex_existing and raster_asset_exists(
                    session=session, path=destination
                ):
                    total_skipped += 1
                    logger.info(f"Skipping existing {source.slug}/{chip.name}")
                    continue

                try:
                    raster_path = download_chip(
                        chip=chip,
                        source=source,
                        destination=destination,
                        image_size=args.image_size,
                        force=args.force_download,
                    )
                except Exception as exc:
                    total_skipped += 1
                    logger.info(f"Skipping {source.slug}/{chip.name}: {exc}")
                    continue

                width, height, patch_payloads = tile_raster(
                    raster_path,
                    chip.bounds,
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                )
                asset = ScoutRasterAsset(
                    location_id=location.id,
                    source_provider=source.provider,
                    source_uri=source.source_uri,
                    license=source.license,
                    capture_date=source.capture_date,
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
                    store_patch_embedding(patch, embedding)
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
                    f"Indexed {source.slug}/{chip.name}: {len(patch_payloads)} "
                    f"patches from {width}x{height} imagery"
                )

    logger.info(
        f"SF seed complete: {len(sources)} sources, {len(chips)} chips, "
        f"{total_patches} patches, {total_indexed} Redis upserts, "
        f"{total_skipped} skipped"
    )
    if args.examples:
        with Session(engine) as session:
            examples = create_query_examples(session=session, limit=args.examples)
        logger.info(f"Created {len(examples)} holdout query examples")


def raster_asset_exists(*, session: Session, path: Path) -> bool:
    return (
        session.exec(
            select(ScoutRasterAsset).where(ScoutRasterAsset.file_path == str(path))
        ).first()
        is not None
    )


def raster_filename(*, source: ArcgisImageSource, chip: SfChip) -> str:
    if source.legacy_stem:
        return f"{chip.name}.jpg"
    return f"{source.slug}_{chip.name}.jpg"


def build_sources(sources_arg: str | None, profile: str) -> list[ArcgisImageSource]:
    if sources_arg is None:
        source_slugs = (
            ["usgs-naip-plus"]
            if profile != "diverse"
            else [
                "usgs-naip-plus",
                "usgs-naip-plus-false-color",
                "noaa-rgb-8bit",
                "sentinel2-natural-color",
                "sentinel2-color-infrared",
            ]
        )
    else:
        source_slugs = [
            source.strip() for source in sources_arg.split(",") if source.strip()
        ]

    sources: list[ArcgisImageSource] = []
    for source_slug in source_slugs:
        source = IMAGERY_SOURCES.get(source_slug)
        if source is None:
            choices = ", ".join(sorted(IMAGERY_SOURCES))
            raise ValueError(f"Unknown imagery source {source_slug!r}; choose {choices}")
        sources.append(source)
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Scout with SF aerial imagery.")
    parser.add_argument(
        "--profile", choices=("quick", "expanded", "diverse"), default="expanded"
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated source slugs. Defaults to usgs-naip-plus, or to "
            "usgs-naip-plus,usgs-naip-plus-false-color,noaa-rgb-8bit,"
            "sentinel2-natural-color,sentinel2-color-infrared for --profile "
            "diverse."
        ),
    )
    parser.add_argument("--chips", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--reindex-existing",
        action="store_true",
        help="Create new DB rows even when the raster file path already exists.",
    )
    return parser.parse_args()


def build_chips(profile: str) -> list[SfChip]:
    if profile == "quick":
        return SF_QUICK_CHIPS
    if profile == "diverse":
        return build_diverse_chips()
    return [
        SfChip(
            name=f"{area.name}-{scale_name}",
            bounds=bounded_chip(area.lon, area.lat, lon_span, lat_span),
        )
        for area in SF_AREAS
        for scale_name, lon_span, lat_span in SCALE_LEVELS
    ]


def build_diverse_chips() -> list[SfChip]:
    compact_levels = [
        ("scene", 0.022, 0.016),
        ("context", 0.044, 0.032),
    ]
    chips = [
        SfChip(
            name=f"{area.name}-{scale_name}",
            bounds=bounded_chip(area.lon, area.lat, lon_span, lat_span),
        )
        for area in SF_DIVERSE_AREAS
        for scale_name, lon_span, lat_span in compact_levels
    ]

    # Add a sparse citywide grid so less-famous neighborhoods are represented.
    lon_steps = [-122.500, -122.470, -122.440, -122.410, -122.380]
    lat_steps = [37.720, 37.745, 37.770, 37.795, 37.820]
    chips.extend(
        SfChip(
            name=f"city-grid-{row}-{col}",
            bounds=bounded_chip(lon, lat, 0.026, 0.020),
        )
        for row, lat in enumerate(lat_steps, start=1)
        for col, lon in enumerate(lon_steps, start=1)
    )
    return chips


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
        location.west = SF_BOUNDS.west
        location.south = SF_BOUNDS.south
        location.east = SF_BOUNDS.east
        location.north = SF_BOUNDS.north
        session.add(location)
        session.commit()
        session.refresh(location)
        return location
    location = ScoutLocation(
        name="San Francisco",
        description="Scout seed area for SF aerial imagery tests",
        west=SF_BOUNDS.west,
        south=SF_BOUNDS.south,
        east=SF_BOUNDS.east,
        north=SF_BOUNDS.north,
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
        store.reset_index()


def download_chip(
    *,
    chip: SfChip,
    source: ArcgisImageSource,
    destination: Path,
    image_size: int,
    force: bool,
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
    if source.rendering_rule is not None:
        params["renderingRule"] = source.rendering_rule

    response = httpx.get(
        source.export_url,
        params=params,
        timeout=60,
        follow_redirects=True,
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"export returned {response.headers.get('content-type')}")

    with Image.open(BytesIO(response.content)) as image:
        if image.mode in {"RGBA", "LA"}:
            alpha = image.getchannel("A")
            if alpha.getextrema() == (0, 0):
                raise ValueError("export returned fully transparent no-data imagery")
        image.convert("RGB").save(destination, format="JPEG", quality=92)

    with Image.open(destination) as image:
        image.verify()
    return destination


if __name__ == "__main__":
    main()
