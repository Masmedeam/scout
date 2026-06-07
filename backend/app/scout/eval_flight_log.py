"""CLI for evaluating GPS-labeled flight log manifests."""

import argparse
import json
import logging
from pathlib import Path

from sqlmodel import Session

from app.core.db import engine
from app.scout.flight_log import evaluate_flight_log, load_manifest, summarize_results
from app.scout.storage import ensure_storage_dirs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    frames = load_manifest(manifest_path)
    with Session(engine) as session:
        results = evaluate_flight_log(session=session, frames=frames, top_k=args.top_k)
    summary = summarize_results(results)
    report_dir = ensure_storage_dirs()["queries"] / "flight-log"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "eval_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"flight-log eval: {summary.get('frames_evaluated', 0)} frames, "
        f"median={summary.get('median_error_meters', 'n/a')}m"
    )
    logger.info(f"Report written to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Scout on a flight log manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
