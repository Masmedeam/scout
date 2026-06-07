"""Compute geographic coverage of indexed Scout patches."""

from dataclasses import dataclass

from sqlmodel import Session, select

from app.models import ScoutImagePatch, ScoutRasterAsset
from app.scout.geo import GeoBounds
from app.scout.seed_sf import SF_BOUNDS


@dataclass(frozen=True)
class CoverageCell:
    row: int
    col: int
    west: float
    south: float
    east: float
    north: float
    covered: bool


@dataclass(frozen=True)
class CoverageReport:
    coverage_percent: float
    patch_count: int
    raster_asset_count: int
    bounds_west: float
    bounds_south: float
    bounds_east: float
    bounds_north: float
    uncovered_cells: list[CoverageCell]


def compute_coverage(session: Session, *, grid_rows: int = 20, grid_cols: int = 16) -> CoverageReport:
    patches = session.exec(
        select(ScoutImagePatch).join(ScoutRasterAsset)
    ).all()
    raster_count = session.exec(select(ScoutRasterAsset)).all()

    if not patches:
        cells = _build_grid_cells(grid_rows, grid_cols)
        return CoverageReport(
            coverage_percent=0.0,
            patch_count=0,
            raster_asset_count=len(raster_count),
            bounds_west=SF_BOUNDS.west,
            bounds_south=SF_BOUNDS.south,
            bounds_east=SF_BOUNDS.east,
            bounds_north=SF_BOUNDS.north,
            uncovered_cells=cells,
        )

    west = min(patch.west for patch in patches)
    south = min(patch.south for patch in patches)
    east = max(patch.east for patch in patches)
    north = max(patch.north for patch in patches)

    lon_step = (SF_BOUNDS.east - SF_BOUNDS.west) / grid_cols
    lat_step = (SF_BOUNDS.north - SF_BOUNDS.south) / grid_rows
    covered_cells = 0
    uncovered: list[CoverageCell] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_west = SF_BOUNDS.west + col * lon_step
            cell_east = cell_west + lon_step
            cell_south = SF_BOUNDS.south + row * lat_step
            cell_north = cell_south + lat_step
            cell_bounds = GeoBounds(cell_west, cell_south, cell_east, cell_north)
            covered = any(_bounds_overlap(cell_bounds, patch) for patch in patches)
            if covered:
                covered_cells += 1
            else:
                uncovered.append(
                    CoverageCell(
                        row=row + 1,
                        col=col + 1,
                        west=cell_west,
                        south=cell_south,
                        east=cell_east,
                        north=cell_north,
                        covered=False,
                    )
                )

    total_cells = grid_rows * grid_cols
    coverage_percent = (covered_cells / total_cells) * 100.0

    return CoverageReport(
        coverage_percent=coverage_percent,
        patch_count=len(patches),
        raster_asset_count=len(raster_count),
        bounds_west=west,
        bounds_south=south,
        bounds_east=east,
        bounds_north=north,
        uncovered_cells=uncovered,
    )


def _bounds_overlap(cell: GeoBounds, patch: ScoutImagePatch) -> bool:
    return not (
        patch.east <= cell.west
        or patch.west >= cell.east
        or patch.north <= cell.south
        or patch.south >= cell.north
    )


def _build_grid_cells(grid_rows: int, grid_cols: int) -> list[CoverageCell]:
    lon_step = (SF_BOUNDS.east - SF_BOUNDS.west) / grid_cols
    lat_step = (SF_BOUNDS.north - SF_BOUNDS.south) / grid_rows
    cells: list[CoverageCell] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_west = SF_BOUNDS.west + col * lon_step
            cell_east = cell_west + lon_step
            cell_south = SF_BOUNDS.south + row * lat_step
            cell_north = cell_south + lat_step
            cells.append(
                CoverageCell(
                    row=row + 1,
                    col=col + 1,
                    west=cell_west,
                    south=cell_south,
                    east=cell_east,
                    north=cell_north,
                    covered=False,
                )
            )
    return cells
