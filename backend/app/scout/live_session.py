"""Process uploaded drone video into a fused VPS+VO track."""

import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from sqlmodel import Session

from app.core.config import settings
from app.scout.aerial_slam import AerialSlamTracker
from app.scout.localization import localize_image
from app.scout.storage import ensure_storage_dirs
from app.scout.video_frames import extract_frames
from app.scout.vio_fusion import FusionTracker, VioDelta, VpsFix
from app.scout.visual_odometry import estimate_frame_delta


@dataclass
class LiveTrackPoint:
    frame_id: str
    lat: float
    lon: float
    source: str
    vps_confidence: float
    timestamp_s: float


@dataclass
class LiveSessionResult:
    session_id: str
    status: str
    track: list[LiveTrackPoint] = field(default_factory=list)
    vps_fix_count: int = 0
    frame_count: int = 0
    median_confidence: float = 0.0
    error: str | None = None


_live_sessions: dict[str, LiveSessionResult] = {}


def get_live_session(session_id: str) -> LiveSessionResult | None:
    return _live_sessions.get(session_id)


def process_live_video(
    *,
    session: Session,
    video_path: Path,
    altitude_m: float | None = None,
    fps: float | None = None,
) -> LiveSessionResult:
    session_id = str(uuid.uuid4())
    result = LiveSessionResult(session_id=session_id, status="processing")
    _live_sessions[session_id] = result

    live_dir = ensure_storage_dirs()["queries"] / "live" / session_id
    live_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = extract_frames(
            video_path=video_path,
            output_dir=live_dir,
            fps=fps or settings.SCOUT_LIVE_FRAME_FPS,
        )
        if not frames:
            result.status = "failed"
            result.error = "No frames extracted from video"
            return result

        tracker = FusionTracker()
        slam = AerialSlamTracker(altitude_m=altitude_m or settings.SCOUT_VO_DEFAULT_ALTITUDE_M)
        vps_fix_count = 0

        for index, frame in enumerate(frames):
            run_vps = index == 0 or index % settings.SCOUT_LIVE_VPS_INTERVAL_FRAMES == 0

            if index == 0:
                slam.step(frame.file_path)
            elif index > 0:
                delta = estimate_frame_delta(
                    prev_path=frames[index - 1].file_path,
                    curr_path=frame.file_path,
                    altitude_m=altitude_m,
                    slam_tracker=slam,
                )
                if delta is not None and tracker.state is not None:
                    tracker.apply_vio(
                        VioDelta(
                            delta_north_m=delta.delta_north_m,
                            delta_east_m=delta.delta_east_m,
                            timestamp=time(),
                        )
                    )

            if run_vps:
                hint_lat = tracker.state.lat if tracker.state else None
                hint_lon = tracker.state.lon if tracker.state else None
                localization = localize_image(
                    session=session,
                    query_path=frame.file_path,
                    top_k=3,
                    hint_lat=hint_lat,
                    hint_lon=hint_lon,
                    radius_km=settings.SCOUT_LIVE_GEO_PRIOR_RADIUS_KM
                    if hint_lat is not None
                    else None,
                )
                apply_vps = (
                    localization.predicted_lat is not None
                    and localization.predicted_lon is not None
                    and (
                        tracker.state is None
                        or localization.confidence
                        >= settings.SCOUT_LIVE_VPS_MIN_APPLY_CONFIDENCE
                    )
                )
                if apply_vps:
                    if tracker.state is None:
                        tracker.initialize(
                            lat=localization.predicted_lat,
                            lon=localization.predicted_lon,
                        )
                        state = tracker.state
                        if state is not None:
                            state.source = "vps"
                            state.vps_confidence = localization.confidence
                    else:
                        state = tracker.apply_vps(
                            VpsFix(
                                lat=localization.predicted_lat,
                                lon=localization.predicted_lon,
                                confidence=localization.confidence,
                                timestamp=time(),
                            )
                        )
                    vps_fix_count += 1

            if tracker.state is None:
                continue

            result.track.append(
                LiveTrackPoint(
                    frame_id=frame.frame_id,
                    lat=tracker.state.lat,
                    lon=tracker.state.lon,
                    source=tracker.state.source,
                    vps_confidence=tracker.state.vps_confidence,
                    timestamp_s=frame.timestamp_s,
                )
            )

        confidences = [point.vps_confidence for point in result.track if point.vps_confidence > 0]
        result.frame_count = len(result.track)
        result.vps_fix_count = vps_fix_count
        result.median_confidence = (
            statistics.median(confidences) if confidences else 0.0
        )
        result.status = "completed" if result.track else "failed"
        if not result.track:
            result.error = "VPS could not localize any frame"
        return result
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        return result
