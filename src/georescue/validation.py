"""Input validation and geometry helpers for GeoRescue AI.

This module sits between user input (Streamlit forms, script arguments)
and the pipeline. It ensures that:

- Bounding boxes and event dates are valid *before* any network call.
- The AOI area is within the configured limit.
- Geographic inputs can be safely reprojected to a suitable projected CRS
  for accurate area and length calculations.
- Helper functions for choosing a projected CRS are centralised so the
  rest of the codebase does not duplicate UTM-zone logic.

The module performs **no network I/O** and can be tested in isolation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.models import BoundingBox, EventDefinition

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Area computations
# ---------------------------------------------------------------------------

def bbox_area_sqkm(bbox: BoundingBox) -> float:
    """Return the geodesic area of a bounding box in square kilometres.

    Uses an equal-area projection (World Mollweide / EPSG:54009) which is
    accurate to a few percent at mid-latitudes and remains valid across
    the antimeridian. This is sufficient for enforcing the AOI limit; the
    actual flood-extent area is computed in a local UTM zone later.
    """
    geom = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    try:
        mollweide = CRS.from_epsg(54009)
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, mollweide, always_xy=True)
        projected = shapely_transform(transformer.transform, geom)
        return float(projected.area) / 1_000_000.0
    except Exception as exc:  # pragma: no cover - pyproj should not fail
        # Fall back to a rough spherical estimate. Better a plausible
        # upper bound than a crash during validation.
        log.warning("Mollweide projection failed (%s); using spherical estimate.", exc)
        import math
        lat_rad = math.radians((bbox.min_lat + bbox.max_lat) / 2.0)
        dlat_km = (bbox.max_lat - bbox.min_lat) * 111.32
        dlon_km = (bbox.max_lon - bbox.min_lon) * 111.32 * math.cos(lat_rad)
        return abs(dlat_km * dlon_km)


# ---------------------------------------------------------------------------
# Projected CRS selection
# ---------------------------------------------------------------------------

def choose_projected_crs(bbox: BoundingBox) -> CRS:
    """Return a projected CRS suitable for area/length computation on ``bbox``.

    Prefers the UTM zone containing the bbox centroid. Falls back to the
    configured fallback CRS (default EPSG:3857) if a UTM zone cannot be
    determined (e.g. near the poles).
    """
    centroid_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    centroid_lat = (bbox.min_lat + bbox.max_lat) / 2.0

    if -80.0 <= centroid_lat <= 84.0:
        zone = int((centroid_lon + 180.0) / 6.0) + 1
        zone = max(1, min(60, zone))
        epsg = 32600 + zone if centroid_lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)

    settings = get_settings()
    log.warning(
        "Latitude %.4f outside UTM range; using fallback CRS %s.",
        centroid_lat,
        settings.fallback_projected_crs,
    )
    return CRS.from_user_input(settings.fallback_projected_crs)


def reproject_geometry(geom: BaseGeometry, src_crs: CRS, dst_crs: CRS) -> BaseGeometry:
    """Reproject a Shapely geometry between two CRSes."""
    if src_crs == dst_crs:
        return geom
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when user input fails validation with an actionable message."""


def validate_event(
    event: EventDefinition,
    settings: Optional[Settings] = None,
) -> None:
    """Validate an :class:`EventDefinition` against application limits.

    Raises :class:`ValidationError` with a human-readable message if any
    check fails. The Pydantic model already enforces per-field constraints;
    this function adds whole-request checks (e.g. AOI size).
    """
    settings = settings or get_settings()

    area = bbox_area_sqkm(event.aoi)
    if area > settings.max_aoi_sqkm:
        raise ValidationError(
            f"AOI area is {area:,.0f} km², which exceeds the configured limit "
            f"of {settings.max_aoi_sqkm:,.0f} km². "
            f"Draw a smaller area or raise GEORESCUE_MAX_AOI_SQKM in .env."
        )

    if area <= 0:
        raise ValidationError("AOI area is zero or negative. Check the bbox coordinates.")

    # Warn (not fail) on suspiciously small AOIs.
    if area < 1.0:
        log.warning(
            "AOI area is only %.3f km². Sentinel-1 GRD is 10 m resolution; "
            "very small AOIs may yield few useful pixels.",
            area,
        )

    if event.pre_days < 1 or event.post_days < 1:
        raise ValidationError("pre_days and post_days must each be at least 1.")


# ---------------------------------------------------------------------------
# Formatting helpers used by the dashboard and reports
# ---------------------------------------------------------------------------

def format_bbox(bbox: BoundingBox) -> str:
    """Return a compact, human-readable representation of a bbox."""
    return (
        f"W{bbox.min_lon:.4f} S{bbox.min_lat:.4f} "
        f"E{bbox.max_lon:.4f} N{bbox.max_lat:.4f}"
    )


def bbox_to_wkt(bbox: BoundingBox) -> str:
    """Return the bbox as a WKT POLYGON in EPSG:4326."""
    poly = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    return poly.wkt


def parse_bbox_string(text: str) -> BoundingBox:
    """Parse a user-supplied bbox string.

    Accepts either comma- or whitespace-separated values in the order
    ``min_lon, min_lat, max_lon, max_lat``.

    Raises :class:`ValidationError` with a clear message on bad input.
    """
    if not text or not text.strip():
        raise ValidationError(
            "Bounding box is empty. Enter four numbers: "
            "min_lon, min_lat, max_lon, max_lat."
        )
    cleaned = text.replace(",", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) != 4:
        raise ValidationError(
            f"Expected 4 numbers (min_lon, min_lat, max_lon, max_lat), "
            f"got {len(parts)}: {text!r}"
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValidationError(
            f"Could not parse bbox numbers from {text!r}: {exc}"
        ) from exc

    try:
        return BoundingBox(
            min_lon=min_lon, min_lat=min_lat,
            max_lon=max_lon, max_lat=max_lat,
        )
    except Exception as exc:
        raise ValidationError(f"Invalid bbox: {exc}") from exc


def bounds_intersect(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if two ``(minx, miny, maxx, maxy)`` boxes overlap."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("bounds_intersect expects two 4-element sequences")
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def ensure_polygon(geom: BaseGeometry) -> Polygon:
    """Return a Polygon from ``geom``, raising if it is not polygonal."""
    if geom.is_empty:
        raise ValidationError("Geometry is empty.")
    if isinstance(geom, Polygon):
        return geom
    raise ValidationError(f"Expected a Polygon, got {geom.geom_type}.")


__all__ = [
    "ValidationError",
    "bbox_area_sqkm",
    "bbox_to_wkt",
    "bounds_intersect",
    "choose_projected_crs",
    "ensure_polygon",
    "format_bbox",
    "parse_bbox_string",
    "reproject_geometry",
    "validate_event",
]