import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GeoBounds:
    west: float
    south: float
    east: float
    north: float

    def validate(self) -> None:
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("longitude bounds must be between -180 and 180")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("latitude bounds must be between -90 and 90")


def pixel_to_lon(bounds: GeoBounds, x: float, image_width: int) -> float:
    return bounds.west + (x / image_width) * (bounds.east - bounds.west)


def pixel_to_lat(bounds: GeoBounds, y: float, image_height: int) -> float:
    return bounds.north - (y / image_height) * (bounds.north - bounds.south)


def haversine_meters(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float
) -> float:
    radius_meters = 6_371_000.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_meters * math.atan2(math.sqrt(value), math.sqrt(1 - value))
