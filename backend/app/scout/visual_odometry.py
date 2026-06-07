"""Visual odometry adapters for live drone tracking."""

from pathlib import Path

from app.core.config import settings
from app.scout.aerial_slam import SlamDelta, estimate_nadir_delta

# Backward-compatible alias used by live_session.
VoDelta = SlamDelta


def estimate_frame_delta(
    *,
    prev_path: str | Path,
    curr_path: str | Path,
    altitude_m: float | None = None,
    slam_tracker=None,
) -> SlamDelta | None:
    altitude = altitude_m if altitude_m is not None else settings.SCOUT_VO_DEFAULT_ALTITUDE_M
    if slam_tracker is not None:
        return slam_tracker.step(curr_path)
    if settings.SCOUT_SLAM_MODEL == "homography":
        return estimate_nadir_delta(
            prev_path=prev_path,
            curr_path=curr_path,
            altitude_m=altitude,
        )
    return _legacy_optical_flow(prev_path, curr_path, altitude)


def _legacy_optical_flow(
    prev_path: str | Path, curr_path: str | Path, altitude_m: float
) -> SlamDelta | None:
    import cv2
    import numpy as np

    from app.scout.aerial_slam import _ground_width_m, _load_gray

    prev_gray = _load_gray(prev_path, 640)
    curr_gray = _load_gray(curr_path, 640)
    if prev_gray is None or curr_gray is None:
        return None

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if prev_pts is None or len(prev_pts) < 8:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if next_pts is None or status is None:
        return None

    good_prev = prev_pts[status.reshape(-1) == 1]
    good_next = next_pts[status.reshape(-1) == 1]
    if len(good_prev) < 8:
        return None

    dx_px = float(np.median(good_next[:, 0, 0] - good_prev[:, 0, 0]))
    dy_px = float(np.median(good_next[:, 0, 1] - good_prev[:, 0, 1]))
    height, width = prev_gray.shape[:2]
    ground_width_m = _ground_width_m(altitude_m, settings.SCOUT_VO_CAMERA_FOV_DEG)
    meters_per_px_x = ground_width_m / width
    meters_per_px_y = ground_width_m * (height / width) / height
    return SlamDelta(
        delta_north_m=-dy_px * meters_per_px_y,
        delta_east_m=dx_px * meters_per_px_x,
        inlier_count=len(good_prev),
        inlier_ratio=len(good_prev) / max(len(prev_pts), 1),
        yaw_deg=0.0,
    )
