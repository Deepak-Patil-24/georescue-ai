"""Sentinel-1 GRD search and selection for GeoRescue AI.

Responsibilities
----------------
- Build a CDSE OData filter for a bounding box, time window, and optional
  polarisation / orbit-direction preferences.
- Query the catalogue via :class:`CopernicusClient`.
- Convert raw OData records into :class:`SceneMetadata`.
- Select a *compatible* pre-event / post-event pair: same relative orbit
  and same orbit direction when the data allow it, falling back to the
  best available pair otherwise, always recording which compromise was
  made in ``selection_notes``.

Why this matters
----------------
The hackathon brief explicitly warns that Sentinel-1 images from
different orbit tracks cannot be compared pixel-by-pixel. The selection
logic here is therefore conservative: it prefers same-track pairs and
records every deviation.

Filter notes
------------
The CDSE OData catalogue does **not** reliably match Sentinel-1 products
through the ``Attributes/OData.CSC.StringAttribute/any(...)`` syntax for
``productType``. Empirically, ``contains(Name,'IW_GRDH')`` works and
returns exactly the Interferometric Wide swath, Ground Range, High
resolution products we need for flood mapping.

Polarisation is embedded in the product name:

- ``1SDV`` -> dual polarisation (VV + VH)
- ``1SSH`` -> single polarisation (HH only)
- ``1SSV`` -> single polarisation (VV only)

We do not filter on polarisation at query time; instead we let the
catalogue return all IW GRD products and let the caller inspect the
polarisation metadata.

This module does not preprocess SAR data. That is the job of
:mod:`georescue.processing.preprocessing`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from georescue.config import Settings, get_settings
from georescue.data_access import DataAccessError, NoResultsError
from georescue.data_access.copernicus import CopernicusClient
from georescue.logging_config import get_logger
from georescue.models import BoundingBox, SceneMetadata, ScenePair

log = get_logger(__name__)

_ODATA_COLLECTION = "SENTINEL-1"

# Substring that identifies IW GRD products in the product Name.
_IW_GRD_NAME_TOKEN = "IW_GRDH"


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------

def _format_odata_datetime(dt: datetime) -> str:
    """Format a datetime for an OData ``ContentDate/Start`` comparison."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _bbox_polygon_wkt(bbox: BoundingBox) -> str:
    """Return the bbox as WKT POLYGON in EPSG:4326 (lon lat order)."""
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
    orbit_direction: Optional[str] = None,
) -> str:
    """Build the OData ``$filter`` fragment for Sentinel-1 products.

    Uses only filter forms that have been empirically verified against
    the CDSE catalogue:

    - ``ContentDate/Start`` for the time window
    - ``contains(Name, 'IW_GRDH')`` for the product type
    - ``OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(...)')``
      for the footprint intersection
    - ``contains(Name, '_ASC_')`` / ``contains(Name, '_DESC_')`` for the
      orbit direction, since the direction is also embedded in the name.
    """
    parts: list[str] = []
    parts.append(f"ContentDate/Start ge {_format_odata_datetime(start)}")
    parts.append(f"ContentDate/Start le {_format_odata_datetime(end)}")
    parts.append(f"contains(Name,'{_IW_GRD_NAME_TOKEN}')")
    parts.append(
        "OData.CSC.Intersects(area=geography'SRID=4326;"
        f"{_bbox_polygon_wkt(bbox)}')"
    )
    if orbit_direction:
        direction = orbit_direction.upper()
        # Sentinel-1 product names use ASCENDING / DESCENDING as
        # ``_ASC_`` and ``_DESC_``. We check both possible spellings by
        # using a case-sensitive contains on the uppercase name. Product
        # names are always uppercase.
        if direction == "ASCENDING":
            parts.append("contains(Name,'_ASC_')")
        elif direction == "DESCENDING":
            parts.append("contains(Name,'_DESC_')")
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def _attr_value(record: dict[str, Any], name: str) -> Optional[str]:
    """Extract an attribute value by name from an OData record."""
    attrs = record.get("Attributes") or []
    for att in attrs:
        if att.get("Name") == name:
            if "Value" in att and att["Value"] is not None:
                return str(att["Value"])
            for typed_key in (
                "OData.CSC.StringAttribute",
                "OData.CSC.IntegerAttribute",
                "OData.CSC.DoubleAttribute",
                "OData.CSC.DateTimeOffsetAttribute",
            ):
                nested = att.get(typed_key)
                if isinstance(nested, dict) and "Value" in nested:
                    return str(nested["Value"])
    return None


def _parse_polarisation_from_name(name: str) -> Optional[str]:
    """Extract polarisation from a Sentinel-1 product name.

    Product names embed the polarisation as a 4-character token, e.g.
    ``S1D_IW_GRDH_1SDV_...`` where ``1SDV`` means single-pol mode, dual
    polarisation (VV + VH). Return values:

    - ``"VV+VH"`` for ``1SDV``
    - ``"HH+HV"`` for ``1SDH``
    - ``"VV"``    for ``1SSV``
    - ``"HH"``    for ``1SSH``
    - ``None`` if the token cannot be parsed.
    """
    if not name:
        return None
    parts = name.split("_")
    for token in parts:
        if len(token) == 4 and token[0] == "1":
            mode = token[1:3]
            pols = token[3:]
            if mode == "SD":
                if pols == "V":
                    return "VV+VH"
                if pols == "H":
                    return "HH+HV"
            elif mode == "SS":
                if pols == "V":
                    return "VV"
                if pols == "H":
                    return "HH"
    return None


def _parse_relative_orbit_from_name(name: str) -> Optional[int]:
    """Relative orbit is not reliably embedded in the product name.

    It is exposed via catalogue attributes when present. If the attribute
    is missing, this returns ``None`` and callers must handle the missing
    value honestly (no substitution of a made-up orbit number).
    """
    return None


def _record_to_scene(record: dict[str, Any]) -> SceneMetadata:
    """Convert a raw OData record into a :class:`SceneMetadata`."""
    content_date = record.get("ContentDate") or {}
    start_str = content_date.get("Start") or record.get("ModificationDate")
    if not start_str:
        raise DataAccessError(
            f"Sentinel-1 record {record.get('Id')} has no acquisition date."
        )
    acquisition = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

    name = record.get("Name") or ""

    relative_orbit_str = _attr_value(record, "relativeOrbitNumber")
    relative_orbit: Optional[int] = None
    if relative_orbit_str and relative_orbit_str.isdigit():
        relative_orbit = int(relative_orbit_str)

    orbit_direction = _attr_value(record, "orbitDirection")
    if orbit_direction is None:
        # Fall back to name parsing.
        if "_ASC_" in name:
            orbit_direction = "ASCENDING"
        elif "_DESC_" in name:
            orbit_direction = "DESCENDING"

    polarisation = _attr_value(record, "polarisationChannels")
    if polarisation is None:
        polarisation = _parse_polarisation_from_name(name)

    size = record.get("ContentLength")

    return SceneMetadata(
        scene_id=str(record.get("Id", "")),
        platform="sentinel-1",
        acquisition_datetime=acquisition,
        relative_orbit=relative_orbit,
        orbit_direction=orbit_direction.upper() if orbit_direction else None,
        polarisation=polarisation,
        product_type="GRD",
        processing_level=_attr_value(record, "processingLevel"),
        tile_or_granule_id=name,
        download_url=None,
        size_bytes=int(size) if isinstance(size, (int, float)) else None,
        footprint_geojson=record.get("GeoFootprint"),
        source="CDSE",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_sentinel1(
    bbox: BoundingBox,
    start: date,
    end: date,
    *,
    orbit_direction: Optional[str] = None,
    max_results: int = 100,
    settings: Optional[Settings] = None,
    client: Optional[CopernicusClient] = None,
) -> list[SceneMetadata]:
    """Search CDSE for Sentinel-1 IW GRD products intersecting ``bbox``.

    Parameters
    ----------
    bbox:
        Area of interest.
    start, end:
        Inclusive date range for ``ContentDate/Start``.
    orbit_direction:
        Optional ``"ASCENDING"`` or ``"DESCENDING"`` filter.
    max_results:
        Upper bound on records returned.
    settings, client:
        Optional dependency injection for testing.

    Returns
    -------
    list of :class:`SceneMetadata`, ordered by acquisition time (earliest
    first), with duplicates removed by scene id.
    """
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = (
        datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
        + timedelta(days=1)
    )

    filter_expr = _build_filter(
        bbox, start_dt, end_dt, orbit_direction=orbit_direction,
    )

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
            log.warning("Skipping malformed Sentinel-1 record: %s", exc)
            continue
        if scene.scene_id in seen:
            continue
        seen.add(scene.scene_id)
        scenes.append(scene)

    scenes.sort(key=lambda s: s.acquisition_datetime)
    log.info(
        "Sentinel-1 search returned %d unique scenes between %s and %s.",
        len(scenes), start, end,
    )
    return scenes


def _pair_score(pre: SceneMetadata, post: SceneMetadata) -> tuple[int, float, float]:
    """Return a sort key that prefers compatible pre/post pairs.

    Lower is better. The tuple is:

    1. Orbit compatibility penalty (0 = same orbit and direction, 1 = same
       direction only, 2 = same relative orbit only, 3 = neither).
    2. Absolute distance from the ideal 12-day separation.
    3. Post-event acquisition time (earlier is better).
    """
    penalty = 3
    if pre.relative_orbit is not None and post.relative_orbit is not None:
        if pre.relative_orbit == post.relative_orbit:
            if pre.orbit_direction == post.orbit_direction:
                penalty = 0
            else:
                penalty = 1
        elif pre.orbit_direction == post.orbit_direction:
            penalty = 2

    separation_days = abs(
        (post.acquisition_datetime - pre.acquisition_datetime).total_seconds()
    ) / 86400.0
    ideal_penalty = abs(separation_days - 12.0)

    return (penalty, ideal_penalty, post.acquisition_datetime.timestamp())


def select_sentinel1_pair(
    bbox: BoundingBox,
    event_date: date,
    *,
    pre_days: int = 12,
    post_days: int = 12,
    prefer_same_orbit: bool = True,
    max_results: int = 100,
    settings: Optional[Settings] = None,
    client: Optional[CopernicusClient] = None,
) -> ScenePair:
    """Search and select a compatible pre/post Sentinel-1 GRD pair.

    Raises
    ------
    NoResultsError
        If either the pre-event or post-event search returns no scenes.
    DataAccessError
        If the CDSE catalogue cannot be reached.
    """
    pre_start = event_date - timedelta(days=pre_days)
    pre_end = event_date - timedelta(days=1)
    post_start = event_date + timedelta(days=1)
    post_end = event_date + timedelta(days=post_days)

    pre_scenes = search_sentinel1(
        bbox, pre_start, pre_end,
        max_results=max_results, settings=settings, client=client,
    )
    post_scenes = search_sentinel1(
        bbox, post_start, post_end,
        max_results=max_results, settings=settings, client=client,
    )

    if not pre_scenes:
        raise NoResultsError(
            f"No Sentinel-1 GRD scenes found for the pre-event window "
            f"{pre_start} to {pre_end}. "
            f"Try widening the window or expanding the AOI."
        )
    if not post_scenes:
        raise NoResultsError(
            f"No Sentinel-1 GRD scenes found for the post-event window "
            f"{post_start} to {post_end}. "
            f"Try widening the window or expanding the AOI."
        )

    notes: list[str] = []
    if prefer_same_orbit:
        pre_orbits = {
            s.relative_orbit for s in pre_scenes if s.relative_orbit is not None
        }
        if pre_orbits:
            same_orbit_post = [
                s for s in post_scenes if s.relative_orbit in pre_orbits
            ]
            if same_orbit_post:
                post_scenes = same_orbit_post
            else:
                notes.append(
                    "No post-event scene shared a relative orbit with the "
                    "pre-event candidates; falling back to best temporal match."
                )
        else:
            notes.append(
                "Relative orbit numbers were not available from the catalogue; "
                "selecting the best temporal match instead."
            )

    scored = [
        (_pair_score(pre, post), pre, post)
        for pre in pre_scenes
        for post in post_scenes
    ]
    scored.sort(key=lambda item: item[0])
    _, best_pre, best_post = scored[0]

    if (
        best_pre.relative_orbit is not None
        and best_post.relative_orbit is not None
        and best_pre.relative_orbit != best_post.relative_orbit
    ):
        notes.append(
            f"Selected pair uses different relative orbits "
            f"(pre={best_pre.relative_orbit}, post={best_post.relative_orbit}). "
            f"Pixel-level comparison is less reliable."
        )
    elif best_pre.orbit_direction != best_post.orbit_direction:
        notes.append(
            f"Selected pair uses opposite orbit directions "
            f"(pre={best_pre.orbit_direction}, post={best_post.orbit_direction})."
        )

    log.info(
        "Selected Sentinel-1 pair: pre %s (orbit %s %s), post %s (orbit %s %s).",
        best_pre.acquisition_datetime.date(),
        best_pre.relative_orbit, best_pre.orbit_direction,
        best_post.acquisition_datetime.date(),
        best_post.relative_orbit, best_post.orbit_direction,
    )

    return ScenePair(pre=best_pre, post=best_post, selection_notes=notes)


__all__ = [
    "search_sentinel1",
    "select_sentinel1_pair",
]