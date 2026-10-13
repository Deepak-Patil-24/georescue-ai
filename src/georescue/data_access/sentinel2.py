"""Sentinel-2 L2A search for GeoRescue AI.

This module provides a companion to the Sentinel-1 pipeline. Sentinel-2
is used for *corroboration* when clouds permit; it is never a mandatory
input. If no suitable scene exists, the caller records that fact rather
than substituting anything.

Only Sentinel-2 **L2A** (surface-reflectance) products are requested,
because the L2A scene-classification layer (SCL) is the cleanest
documented cloud mask available without additional processing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from georescue.config import Settings, get_settings
from georescue.data_access import DataAccessError, NoResultsError
from georescue.data_access.copernicus import CopernicusClient
from georescue.logging_config import get_logger
from georescue.models import BoundingBox, SceneMetadata

log = get_logger(__name__)

_ODATA_COLLECTION = "SENTINEL-2"
_PRODUCT_TYPE_L2A = "S2MSI2A"


def _format_odata_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _bbox_polygon_wkt(bbox: BoundingBox) -> str:
    return (
        f"POLYGON(({bbox.min_lon} {bbox.min_lat},"
        f"{bbox.max_lon} {bbox.min_lat},"
        f"{bbox.max_lon} {bbox.max_lat},"
        f"{bbox.min_lon} {bbox.max_lat},"
        f"{bbox.min_lon} {bbox.min_lat}))"
    )


def _build_filter(
    bbox: BoundingBox,
    start: datetime,
    end: datetime,
    *,
    max_cloud_cover: Optional[float],
) -> str:
    parts: list[str] = []
    parts.append(f"ContentDate/Start ge {_format_odata_datetime(start)}")
    parts.append(f"ContentDate/Start le {_format_odata_datetime(end)}")
    parts.append(
        "OData.CSC.Intersects(area=geography'SRID=4326;"
        f"{_bbox_polygon_wkt(bbox)}')"
    )
    parts.append(
        "Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{_PRODUCT_TYPE_L2A}')"
    )
    if max_cloud_cover is not None:
        # cloudCover is stored as an IntegerAttribute (0-100).
        # OData filter uses the typed value object.
        parts.append(
            "Attributes/OData.CSC.IntegerAttribute/any("
            f"att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.IntegerAttribute/Value le {int(max_cloud_cover)})"
        )
    return " and ".join(parts)


def _attr_value(record: dict[str, Any], name: str) -> Optional[str]:
    attrs = record.get("Attributes") or []
    for att in attrs:
        if att.get("Name") == name:
            if "Value" in att and att["Value"] is not None:
                return str(att["Value"])
            for typed_key in (
                "OData.CSC.StringAttribute",
                "OData.CSC.IntegerAttribute",
                "OData.CSC.DoubleAttribute",
            ):
                nested = att.get(typed_key)
                if isinstance(nested, dict) and "Value" in nested:
                    return str(nested["Value"])
    return None


def _record_to_scene(record: dict[str, Any]) -> SceneMetadata:
    content_date = record.get("ContentDate") or {}
    start_str = content_date.get("Start") or record.get("ModificationDate")
    if not start_str:
        raise DataAccessError(
            f"Sentinel-2 record {record.get('Id')} has no acquisition date."
        )
    acquisition = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

    cloud_str = _attr_value(record, "cloudCover")
    try:
        cloud = float(cloud_str) if cloud_str is not None else None
    except ValueError:
        cloud = None

    size = record.get("ContentLength")

    return SceneMetadata(
        scene_id=str(record.get("Id", "")),
        platform="sentinel-2",
        acquisition_datetime=acquisition,
        relative_orbit=None,  # Sentinel-2 uses a different tiling scheme.
        orbit_direction=None,
        polarisation=None,
        cloud_cover=cloud,
        product_type=_PRODUCT_TYPE_L2A,
        processing_level="L2A",
        tile_or_granule_id=record.get("Name"),
        download_url=None,
        size_bytes=int(size) if isinstance(size, (int, float)) else None,
        footprint_geojson=record.get("GeoFootprint"),
        source="CDSE",
    )


def search_sentinel2(
    bbox: BoundingBox,
    start: date,
    end: date,
    *,
    max_cloud_cover: Optional[float] = 30.0,
    max_results: int = 100,
    settings: Optional[Settings] = None,
    client: Optional[CopernicusClient] = None,
) -> list[SceneMetadata]:
    """Search CDSE for Sentinel-2 L2A products intersecting ``bbox``."""
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)

    filter_expr = _build_filter(bbox, start_dt, end_dt, max_cloud_cover=max_cloud_cover)

    owns_client = client is None
    client = client or CopernicusClient(settings=settings)
    try:
        records = client.search_products(
            collection=_ODATA_COLLECTION,
            filter_expression=filter_expr,
            top=max_results,
        )
    finally:
        if owns_client:
            client.close()

    scenes: list[SceneMetadata] = []
    seen: set[str] = set()
    for record in records:
        try:
            scene = _record_to_scene(record)
        except DataAccessError as exc:
            log.warning("Skipping malformed Sentinel-2 record: %s", exc)
            continue
        if scene.scene_id in seen:
            continue
        seen.add(scene.scene_id)
        scenes.append(scene)

    scenes.sort(key=lambda s: s.acquisition_datetime)
    log.info(
        "Sentinel-2 search returned %d unique scenes between %s and %s.",
        len(scenes), start, end,
    )
    return scenes


def select_sentinel2_corroboration(
    bbox: BoundingBox,
    event_date: date,
    *,
    pre_days: int = 12,
    post_days: int = 12,
    max_cloud_cover: float = 30.0,
    settings: Optional[Settings] = None,
    client: Optional[CopernicusClient] = None,
) -> Optional[SceneMetadata]:
    """Return the best post-event Sentinel-2 scene for corroboration, or None.

    Unlike Sentinel-1, optical data are optional. This function returns
    ``None`` (not an exception) when no suitable scene exists, so the
    caller can continue with SAR-only analysis.
    """
    post_start = event_date + timedelta(days=1)
    post_end = event_date + timedelta(days=post_days)

    try:
        scenes = search_sentinel2(
            bbox, post_start, post_end,
            max_cloud_cover=max_cloud_cover,
            settings=settings,
            client=client,
        )
    except NoResultsError:
        return None

    if not scenes:
        log.info("No Sentinel-2 L2A scene met the cloud-cover limit; skipping optical corroboration.")
        return None

    # Prefer the earliest suitable post-event scene: closest in time to the
    # event, least opportunity for scene change unrelated to the flood.
    scenes.sort(key=lambda s: (s.cloud_cover if s.cloud_cover is not None else 999.0,
                                s.acquisition_datetime))
    best = scenes[0]
    log.info(
        "Selected Sentinel-2 corroboration scene: %s on %s (cloud %.1f%%).",
        best.scene_id, best.acquisition_datetime.date(),
        best.cloud_cover if best.cloud_cover is not None else -1,
    )
    return best


__all__ = [
    "search_sentinel2",
    "select_sentinel2_corroboration",
]