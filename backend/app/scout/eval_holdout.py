"""Evaluate Scout localization against holdout query examples."""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlmodel import Session

from app.core.db import engine
from app.scout.embedding import active_embedding_model
from app.scout.geo import haversine_meters
from app.scout.localization import localize_image
from app.scout.query_examples import create_query_examples, read_query_examples
from app.scout.storage import ensure_storage_dirs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    query_id: str
    true_lat: float
    true_lon: float
    pred_lat: float
    pred_lon: float
    error_meters: float
    confidence: float
    method: str


def main() -> None:
    args = parse_args()
    with Session(engine) as session:
        if args.regenerate:
            create_query_examples(session=session, limit=args.limit)
        examples = read_query_examples()[: args.limit]
        if not examples:
            raise SystemExit(
                "No holdout examples found. Run seed with --examples or pass --regenerate."
            )

        results: list[EvalResult] = []
        for example in examples:
            result = evaluate_example(session=session, example=example, top_k=args.top_k)
            if result is not None:
                results.append(result)

    if not results:
        raise SystemExit("No holdout examples could be evaluated.")

    errors = [item.error_meters for item in results]
    summary = {
        "embedding_model": active_embedding_model(),
        "eval_source": "holdout",
        "samples_evaluated": len(results),
        "median_error_meters": sorted(errors)[len(errors) // 2],
        "mean_error_meters": sum(errors) / len(errors),
        "max_error_meters": max(errors),
        "within_250m": sum(error <= 250 for error in errors),
        "within_500m": sum(error <= 500 for error in errors),
        "results": [asdict(item) for item in results],
    }

    report_dir = ensure_storage_dirs()["queries"] / "eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "holdout_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"holdout eval: {len(results)} samples, "
        f"median={summary['median_error_meters']:.0f}m, "
        f"mean={summary['mean_error_meters']:.0f}m"
    )
    logger.info(f"Report written to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scout localization on holdout query examples."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def evaluate_example(*, session: Session, example: dict, top_k: int) -> EvalResult | None:
    query_path = Path(example["file_path"])
    if not query_path.is_absolute():
        query_path = Path.cwd() / query_path
    if not query_path.exists():
        return None

    localization = localize_image(session=session, query_path=query_path, top_k=top_k)
    if localization.predicted_lat is None or localization.predicted_lon is None:
        return None

    error_meters = haversine_meters(
        float(example["center_lat"]),
        float(example["center_lon"]),
        localization.predicted_lat,
        localization.predicted_lon,
    )
    logger.info(f"{example['id']}: error={error_meters:.0f}m conf={localization.confidence:.2f}")
    return EvalResult(
        query_id=example["id"],
        true_lat=float(example["center_lat"]),
        true_lon=float(example["center_lon"]),
        pred_lat=localization.predicted_lat,
        pred_lon=localization.predicted_lon,
        error_meters=error_meters,
        confidence=localization.confidence,
        method=localization.method,
    )


if __name__ == "__main__":
    main()
