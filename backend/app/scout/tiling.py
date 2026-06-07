from pathlib import Path

from PIL import Image

from app.scout.geo import GeoBounds, pixel_to_lat, pixel_to_lon
from app.scout.observability import scout_op
from app.scout.storage import ensure_storage_dirs


@scout_op("tile_raster")
def tile_raster(
    raster_path: str | Path,
    bounds: GeoBounds,
    patch_size: int = 256,
    overlap: int = 64,
) -> tuple[int, int, list[dict]]:
    bounds.validate()
    if patch_size < 32:
        raise ValueError("patch_size must be at least 32")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must be greater than or equal to 0 and less than patch_size")

    patch_dir = ensure_storage_dirs()["patches"]
    raster_path = Path(raster_path)
    patches: list[dict] = []

    with Image.open(raster_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        step = patch_size - overlap

        xs = _tile_offsets(width, patch_size, step)
        ys = _tile_offsets(height, patch_size, step)
        for y in ys:
            for x in xs:
                right = min(x + patch_size, width)
                lower = min(y + patch_size, height)
                tile = image.crop((x, y, right, lower))
                patch_path = patch_dir / f"{raster_path.stem}_{x}_{y}.jpg"
                tile.save(patch_path, format="JPEG", quality=92)

                west = pixel_to_lon(bounds, x, width)
                east = pixel_to_lon(bounds, right, width)
                north = pixel_to_lat(bounds, y, height)
                south = pixel_to_lat(bounds, lower, height)
                center_lon = pixel_to_lon(bounds, (x + right) / 2, width)
                center_lat = pixel_to_lat(bounds, (y + lower) / 2, height)

                patches.append(
                    {
                        "file_path": str(patch_path),
                        "pixel_x": x,
                        "pixel_y": y,
                        "width": right - x,
                        "height": lower - y,
                        "west": west,
                        "south": south,
                        "east": east,
                        "north": north,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                    }
                )

    return width, height, patches


def _tile_offsets(length: int, patch_size: int, step: int) -> list[int]:
    if length <= patch_size:
        return [0]
    offsets = list(range(0, length - patch_size + 1, step))
    final_offset = length - patch_size
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets
