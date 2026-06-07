"""Geometric re-ranking: align query image to candidate patches via feature matching."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.scout.geo import GeoBounds, pixel_to_lat, pixel_to_lon
from app.scout.observability import scout_op


@dataclass(frozen=True)
class FineMatchResult:
    inlier_count: int
    inlier_ratio: float
    match_score: float
    refined_lat: float
    refined_lon: float


@scout_op("fine_match_patch")
def fine_match_patch(
    *,
    query_path: str | Path,
    patch_path: str | Path,
    bounds: GeoBounds,
    patch_width: int,
    patch_height: int,
) -> FineMatchResult | None:
    query_gray = _load_gray(query_path, max_side=640)
    patch_gray = _load_gray(patch_path, max_side=640)
    if query_gray is None or patch_gray is None:
        return None

    orb = cv2.ORB_create(nfeatures=2000)
    query_kp, query_desc = orb.detectAndCompute(query_gray, None)
    patch_kp, patch_desc = orb.detectAndCompute(patch_gray, None)
    if (
        query_desc is None
        or patch_desc is None
        or len(query_kp) < 8
        or len(patch_kp) < 8
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = matcher.match(query_desc, patch_desc)
    if len(raw_matches) < 8:
        return None

    raw_matches = sorted(raw_matches, key=lambda item: item.distance)[:200]
    query_pts = np.float32([query_kp[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
    patch_pts = np.float32([patch_kp[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(query_pts, patch_pts, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return None

    inlier_count = int(mask.ravel().sum())
    if inlier_count < 8:
        return None
    inlier_ratio = inlier_count / len(raw_matches)

    query_h, query_w = query_gray.shape[:2]
    patch_h, patch_w = patch_gray.shape[:2]
    center = np.float32([[[query_w / 2, query_h / 2]]])
    mapped = cv2.perspectiveTransform(center, homography)[0][0]

    scale_x = patch_width / patch_w
    scale_y = patch_height / patch_h
    patch_x = float(mapped[0] * scale_x)
    patch_y = float(mapped[1] * scale_y)

    if not (0 <= patch_x <= patch_width and 0 <= patch_y <= patch_height):
        return None

    refined_lon = pixel_to_lon(bounds, patch_x, patch_width)
    refined_lat = pixel_to_lat(bounds, patch_y, patch_height)
    match_score = inlier_ratio * min(1.0, inlier_count / 40.0)

    return FineMatchResult(
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        match_score=match_score,
        refined_lat=refined_lat,
        refined_lon=refined_lon,
    )


def _load_gray(path: str | Path, *, max_side: int) -> np.ndarray | None:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return gray
