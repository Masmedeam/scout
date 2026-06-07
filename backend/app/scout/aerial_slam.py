"""Nadir aerial SLAM frontend for drone video.

Assumes a downward-looking camera over flat terrain. Estimates frame-to-frame
metric motion with ORB features and a partial affine (similarity) model, then
optionally smooths motion when match quality is weak.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings


@dataclass(frozen=True)
class SlamDelta:
    delta_north_m: float
    delta_east_m: float
    inlier_count: int
    inlier_ratio: float
    yaw_deg: float


@dataclass
class AerialSlamTracker:
    """Stateful nadir VO / SLAM frontend with optional delta smoothing."""

    altitude_m: float
    camera_fov_deg: float = settings.SCOUT_VO_CAMERA_FOV_DEG
    max_side: int = 960
    _orb: cv2.ORB = field(default_factory=lambda: cv2.ORB_create(nfeatures=2000))
    _prev_gray: np.ndarray | None = field(default=None, repr=False)
    _ema_north: float = 0.0
    _ema_east: float = 0.0

    def reset(self) -> None:
        self._prev_gray = None
        self._ema_north = 0.0
        self._ema_east = 0.0

    def step(self, frame_path: str | Path) -> SlamDelta | None:
        gray = _load_gray(frame_path, self.max_side)
        if gray is None:
            return None
        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        delta = estimate_nadir_delta(
            prev_gray=self._prev_gray,
            curr_gray=gray,
            altitude_m=self.altitude_m,
            camera_fov_deg=self.camera_fov_deg,
            orb=self._orb,
        )
        self._prev_gray = gray
        if delta is None:
            return None

        alpha = min(1.0, max(0.15, delta.inlier_ratio))
        self._ema_north = self._ema_north * (1 - alpha) + delta.delta_north_m * alpha
        self._ema_east = self._ema_east * (1 - alpha) + delta.delta_east_m * alpha
        return SlamDelta(
            delta_north_m=self._ema_north,
            delta_east_m=self._ema_east,
            inlier_count=delta.inlier_count,
            inlier_ratio=delta.inlier_ratio,
            yaw_deg=delta.yaw_deg,
        )


def estimate_nadir_delta(
    *,
    prev_path: str | Path | None = None,
    curr_path: str | Path | None = None,
    prev_gray: np.ndarray | None = None,
    curr_gray: np.ndarray | None = None,
    altitude_m: float | None = None,
    camera_fov_deg: float | None = None,
    orb: cv2.ORB | None = None,
) -> SlamDelta | None:
    altitude = altitude_m if altitude_m is not None else settings.SCOUT_VO_DEFAULT_ALTITUDE_M
    fov_deg = camera_fov_deg if camera_fov_deg is not None else settings.SCOUT_VO_CAMERA_FOV_DEG

    if prev_gray is None and prev_path is not None:
        prev_gray = _load_gray(prev_path)
    if curr_gray is None and curr_path is not None:
        curr_gray = _load_gray(curr_path)
    if prev_gray is None or curr_gray is None:
        return None

    detector = orb or cv2.ORB_create(nfeatures=2000)
    prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)
    curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)
    if (
        prev_desc is None
        or curr_desc is None
        or len(prev_kp) < 12
        or len(curr_kp) < 12
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(prev_desc, curr_desc, k=2)
    good = [
        match_pair[0]
        for match_pair in knn
        if len(match_pair) == 2 and match_pair[0].distance < 0.75 * match_pair[1].distance
    ]
    if len(good) < 8:
        return None

    src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=2000,
        confidence=0.99,
    )
    if transform is None or inlier_mask is None:
        return None

    inlier_count = int(inlier_mask.sum())
    if inlier_count < 8:
        return None
    inlier_ratio = inlier_count / len(good)

    dx_px = float(transform[0, 2])
    dy_px = float(transform[1, 2])
    yaw_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))

    height, width = prev_gray.shape[:2]
    ground_width_m = _ground_width_m(altitude, fov_deg)
    meters_per_px_x = ground_width_m / width
    meters_per_px_y = ground_width_m * (height / width) / height

    delta_east_m = dx_px * meters_per_px_x
    delta_north_m = -dy_px * meters_per_px_y
    return SlamDelta(
        delta_north_m=delta_north_m,
        delta_east_m=delta_east_m,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        yaw_deg=yaw_deg,
    )


def _load_gray(path: str | Path, max_side: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _ground_width_m(altitude_m: float, fov_deg: float) -> float:
    fov_rad = np.radians(fov_deg)
    return 2.0 * altitude_m * np.tan(fov_rad / 2.0)
