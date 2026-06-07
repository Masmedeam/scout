"""Evaluate Scout localization against pangeo-uav SF drone query images."""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from sqlmodel import Session

from app.core.db import engine
from app.scout.embedding import embed_image, model_name
from app.scout.geo import haversine_meters
from app.scout.storage import ensure_storage_dirs
from app.scout.vector_store import ScoutVectorStore

PANGEO_REPO = "Fahim17/pangeo-uav"
COORD_PATTERN = re.compile(r"_(-?\d+\.\d+)_(-?\d+\.\d+)/")

# Known SF drone queries with ground-truth coordinates encoded in the path.
PANGEO_SF_SAMPLES = [
    "drone/SanFrancisco/47B-UX3fH6fLIpAKCt0GjA_37.747424_-122.405549/0.jpeg",
    "drone/SanFrancisco/RZW5tyUrtyAlYzaEyLuJeQ_37.784506_-122.415557/0.jpeg",
    "drone/SanFrancisco/x1Ei2cHn0ctAWOUKQdSSJA_37.751709_-122.392244/0.jpeg",
    "drone/SanFrancisco/mIN5akGw6h0tsJkgQb4Dtg_37.748152_-122.420764/0.jpeg",
    "drone/SanFrancisco/xXQGuPOwHeZ5l4-WA7qOSw_37.768789_-122.443432/0.jpeg",
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    sample: str
    true_lat: float
    true_lon: float
    pred_lat: float
    pred_lon: float
    error_meters: float
    match_score: float


def main() -> None:
    args = parse_args()
    store = ScoutVectorStore()
    if not store.available:
        raise SystemExit("Redis vector store is unavailable. Start docker compose first.")

    eval_dir = ensure_storage_dirs()["queries"] / "pangeo-uav"
    eval_dir.mkdir(parents=True, exist_ok=True)

    samples = PANGEO_SF_SAMPLES[: args.limit]
    results: list[EvalResult] = []
    for sample in samples:
        result = evaluate_sample(
            store=store,
            sample=sample,
            eval_dir=eval_dir,
            top_k=args.top_k,
            force_download=args.force_download,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("No pangeo-uav samples could be evaluated.")

    errors = [item.error_meters for item in results]
    summary = {
        "embedding_model": model_name(),
        "samples_evaluated": len(results),
        "median_error_meters": sorted(errors)[len(errors) // 2],
        "mean_error_meters": sum(errors) / len(errors),
        "max_error_meters": max(errors),
        "within_250m": sum(error <= 250 for error in errors),
        "within_500m": sum(error <= 500 for error in errors),
        "results": [asdict(item) for item in results],
    }

    report_path = eval_dir / "eval_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"pangeo-uav eval: {len(results)} samples, "
        f"median={summary['median_error_meters']:.0f}m, "
        f"mean={summary['mean_error_meters']:.0f}m, "
        f"<=250m={summary['within_250m']}/{len(results)}"
    )
    logger.info(f"Report written to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scout localization on pangeo-uav SF drone queries."
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def evaluate_sample(
    *,
    store: ScoutVectorStore,
    sample: str,
    eval_dir: Path,
    top_k: int,
    force_download: bool,
) -> EvalResult | None:
    coords = parse_coords(sample)
    if coords is None:
        logger.warning(f"Skipping sample with unparseable coords: {sample}")
        return None
    true_lat, true_lon = coords

    local_path = eval_dir / sample
    if force_download or not local_path.exists():
        try:
            downloaded = hf_hub_download(
                repo_id=PANGEO_REPO,
                repo_type="dataset",
                filename=sample,
                local_dir=str(eval_dir),
            )
            local_path = Path(downloaded)
        except Exception as error:
            logger.warning(
                f"Could not download {sample} from HuggingFace "
                f"(dataset may be gated — set HF_TOKEN): {error}"
            )
            return None

    matches = store.search(embed_image(local_path), top_k=top_k)
    if not matches:
        logger.warning(f"No vector matches for {sample}")
        return None

    with Session(engine) as session:
        from app.models import ScoutImagePatch

        matched_patch = session.get(ScoutImagePatch, matches[0].patch_id)
        if matched_patch is None:
            logger.warning(f"Matched patch missing in Postgres for {sample}")
            return None
        pred_lat = matched_patch.center_lat
        pred_lon = matched_patch.center_lon

    error_meters = haversine_meters(true_lat, true_lon, pred_lat, pred_lon)
    result = EvalResult(
        sample=sample,
        true_lat=true_lat,
        true_lon=true_lon,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        error_meters=error_meters,
        match_score=matches[0].score,
    )
    logger.info(
        f"{Path(sample).parent.name}: error={error_meters:.0f}m "
        f"(true={true_lat:.5f},{true_lon:.5f} "
        f"pred={pred_lat:.5f},{pred_lon:.5f})"
    )
    return result


def parse_coords(sample: str) -> tuple[float, float] | None:
    match = COORD_PATTERN.search(sample)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


if __name__ == "__main__":
    main()
