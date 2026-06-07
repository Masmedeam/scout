"""Fuse high-rate VIO deltas with low-rate VPS global corrections."""

import math
from dataclasses import dataclass, field
from time import time

from app.core.config import settings


@dataclass
class PoseState:
    lat: float
    lon: float
    timestamp: float
    source: str = "init"
    vps_confidence: float = 0.0


@dataclass
class VioDelta:
    delta_north_m: float
    delta_east_m: float
    timestamp: float | None = None


@dataclass
class VpsFix:
    lat: float
    lon: float
    confidence: float
    timestamp: float | None = None


@dataclass
class FusionTracker:
    state: PoseState | None = None
    history: list[PoseState] = field(default_factory=list)

    def initialize(self, *, lat: float, lon: float) -> PoseState:
        now = time()
        self.state = PoseState(lat=lat, lon=lon, timestamp=now, source="init")
        self._record()
        return self.state

    def apply_vio(self, delta: VioDelta) -> PoseState:
        if self.state is None:
            raise ValueError("Fusion tracker is not initialized")
        lat, lon = _offset_meters(
            self.state.lat,
            self.state.lon,
            delta.delta_north_m,
            delta.delta_east_m,
        )
        now = delta.timestamp or time()
        self.state = PoseState(lat=lat, lon=lon, timestamp=now, source="vio")
        self._record()
        return self.state

    def apply_vps(self, fix: VpsFix) -> PoseState:
        if self.state is None:
            now = fix.timestamp or time()
            self.state = PoseState(
                lat=fix.lat,
                lon=fix.lon,
                timestamp=now,
                source="vps",
                vps_confidence=fix.confidence,
            )
            self._record()
            return self.state

        alpha = min(1.0, fix.confidence * settings.SCOUT_FUSION_VPS_WEIGHT)
        lat = self.state.lat * (1 - alpha) + fix.lat * alpha
        lon = self.state.lon * (1 - alpha) + fix.lon * alpha
        now = fix.timestamp or time()
        self.state = PoseState(
            lat=lat,
            lon=lon,
            timestamp=now,
            source="fused",
            vps_confidence=fix.confidence,
        )
        self._record()
        return self.state

    def _record(self) -> None:
        if self.state is None:
            return
        self.history.append(self.state)
        if len(self.history) > settings.SCOUT_FUSION_HISTORY_LIMIT:
            self.history = self.history[-settings.SCOUT_FUSION_HISTORY_LIMIT :]


_trackers: dict[str, FusionTracker] = {}


def get_tracker(session_id: str) -> FusionTracker:
    tracker = _trackers.get(session_id)
    if tracker is None:
        tracker = FusionTracker()
        _trackers[session_id] = tracker
    return tracker


def _offset_meters(
    lat: float, lon: float, north_m: float, east_m: float
) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(0.1, abs(math.cos(math.radians(lat))))
    return lat + north_m / meters_per_deg_lat, lon + east_m / meters_per_deg_lon
