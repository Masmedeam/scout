"""Ingest and evaluate GPS-labeled drone flight frames."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlmodel import Session

from app.scout.geo import haversine_meters
from app.scout.localization import localize_image


@dataclass(frozen=True)
class FlightFrame:
    frame_id: str
    file_path: Path
    true_lat: float
    true_lon: float
    timestamp: str | None = None


@dataclass(frozen=True)
class FlightEvalResult:
    frame_id: str
    true_lat: float
    true_lon: float
    pred_lat: float
    pred_lon: float
    error_meters: float
    confidence: float
    method: str


def load_manifest(manifest_path: Path) -> list[FlightFrame]:
    if manifest_path.suffix.lower() == ".json":
        return _load_json_manifest(manifest_path)
    return _load_csv_manifest(manifest_path)


def evaluate_flight_log(
    *,
    session: Session,
    frames: list[FlightFrame],
    top_k: int = 5,
) -> list[FlightEvalResult]:
    results: list[FlightEvalResult] = []
    for frame in frames:
        if not frame.file_path.exists():
            continue
        localization = localize_image(
            session=session,
            query_path=frame.file_path,
            top_k=top_k,
        )
        if localization.predicted_lat is None or localization.predicted_lon is None:
            continue
        error_meters = haversine_meters(
            frame.true_lat,
            frame.true_lon,
            localization.predicted_lat,
            localization.predicted_lon,
        )
        results.append(
            FlightEvalResult(
                frame_id=frame.frame_id,
                true_lat=frame.true_lat,
                true_lon=frame.true_lon,
                pred_lat=localization.predicted_lat,
                pred_lon=localization.predicted_lon,
                error_meters=error_meters,
                confidence=localization.confidence,
                method=localization.method,
            )
        )
    return results


def summarize_results(results: list[FlightEvalResult]) -> dict:
    if not results:
        return {"frames_evaluated": 0}
    errors = [item.error_meters for item in results]
    sorted_errors = sorted(errors)
    return {
        "frames_evaluated": len(results),
        "median_error_meters": sorted_errors[len(sorted_errors) // 2],
        "mean_error_meters": sum(errors) / len(errors),
        "max_error_meters": max(errors),
        "within_50m": sum(error <= 50 for error in errors),
        "within_250m": sum(error <= 250 for error in errors),
        "results": [asdict(item) for item in results],
    }


def write_example_manifest(frames: list[FlightFrame], output_path: Path) -> None:
    rows = [
        {
            "frame_id": frame.frame_id,
            "file_path": str(frame.file_path),
            "lat": frame.true_lat,
            "lon": frame.true_lon,
            "timestamp": frame.timestamp or "",
        }
        for frame in frames
    ]
    output_path.write_text(json.dumps(rows, indent=2))


def _load_csv_manifest(manifest_path: Path) -> list[FlightFrame]:
    base_dir = manifest_path.parent
    frames: list[FlightFrame] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            file_path = Path(row.get("file_path") or row.get("path") or "")
            if not file_path.is_absolute():
                file_path = base_dir / file_path
            frames.append(
                FlightFrame(
                    frame_id=row.get("frame_id") or row.get("id") or f"frame-{index:04d}",
                    file_path=file_path,
                    true_lat=float(row["lat"]),
                    true_lon=float(row["lon"]),
                    timestamp=row.get("timestamp") or None,
                )
            )
    return frames


def _load_json_manifest(manifest_path: Path) -> list[FlightFrame]:
    base_dir = manifest_path.parent
    payload = json.loads(manifest_path.read_text())
    frames: list[FlightFrame] = []
    for index, row in enumerate(payload, start=1):
        file_path = Path(row["file_path"])
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        frames.append(
            FlightFrame(
                frame_id=row.get("frame_id") or row.get("id") or f"frame-{index:04d}",
                file_path=file_path,
                true_lat=float(row["lat"]),
                true_lon=float(row["lon"]),
                timestamp=row.get("timestamp"),
            )
        )
    return frames
