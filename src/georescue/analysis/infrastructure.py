"""Infrastructure exposure analysis.

This module intersects the candidate flood extent with pre-event OSM
infrastructure (buildings, roads, bridges) to identify features that are
**potentially exposed** to flooding.

Critical terminology
--------------------
- **Intersecting** = the feature geometry intersects the flood polygon.
- **Potentially exposed** = intersecting, or within a small buffer.
- **Confirmed damaged** = NOT CALCULATED. We have no ground truth. We
  never claim a building or road is destroyed merely because it
  intersects a flood polygon.

Data source
-----------
Pre-event OSM snapshot via the ohsome API. The snapshot date must be
strictly before the event date. This is enforced by
:mod:`georescue.data_access.historical_osm`.

Handling large AOIs and rate limits
-----------------------------------
The ohsome API returns parquet for extraction queries and enforces a
rate limit (HTTP 429). We mitigate both problems by:

1. Splitting the AOI into a grid of smaller cells.
2. Querying all three infrastructure filters in a single request per
   tile, minimising the total number of requests.
3. Sleeping between tiles to stay under the rate limit.
4. Retrying on HTTP 429 with exponential backoff.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from rasterio.crs import CRS

from georescue.config import Settings, get_settings
from georescue.data_access import DataAccessError, NoResultsError
from georescue.data_access.historical_osm import query_osm_features
from georescue.logging_config import get_logger
from georescue.models import BoundingBox
from georescue.processing.raster_utils import utm_crs_for_bbox

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OSM_SNAPSHOT = date(2026, 7, 27)

# Cell size for tiled ohsome queries (degrees). At 27°N, 0.10° is roughly
# 11 km × 10 km, which keeps each request well under the API size limit
# while minimising the total number of requests.
DEFAULT_CELL_SIZE_DEG = 0.10

# Pause between tiles to stay under the ohsome rate limit (seconds).
TILE_PAUSE_S = 5.0

# Retry configuration for HTTP 429.
MAX_RETRIES = 4
INITIAL_BACKOFF_S = 3.0

# Filter expressions for the three infrastructure layers.
FILTER_BUILDINGS = "building=* and geometry:polygon"
FILTER_ROADS = "highway=* and geometry:line"
FILTER_BRIDGES = "bridge=* and geometry:line"

ALL_FILTERS = (FILTER_BUILDINGS, FILTER_ROADS, FILTER_BRIDGES)


# ---------------------------------------------------------------------------
# Tiling helper for large AOIs
# ---------------------------------------------------------------------------

def _split_bbox_into_cells(
    bbox: BoundingBox,
    cell_size_deg: float,
) -> list[BoundingBox]:
    """Split a bbox into a grid of smaller bboxes."""
    cells: list[BoundingBox] = []
    lat = bbox.min_lat
    while lat < bbox.max_lat:
        next_lat = min(lat + cell_size_deg, bbox.max_lat)
        lon = bbox.min_lon
        while lon < bbox.max_lon:
            next_lon = min(lon + cell_size_deg, bbox.max_lon)
            cells.append(BoundingBox(
                min_lon=lon, min_lat=lat,
                max_lon=next_lon, max_lat=next_lat,
            ))
            lon = next_lon
        lat = next_lat
    return cells


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _query_with_retry(
    cell: BoundingBox,
    snapshot_timestamp: date,
    *,
    event_date: Optional[date],
    settings: Settings,
    max_retries: int = MAX_RETRIES,
) -> dict[str, list[dict[str, Any]]]:
    """Call query_osm_features with exponential backoff on rate-limit errors.

    Detects HTTP 429 in the error message and retries after a growing
    delay. Non-retryable errors are raised immediately.
    """
    backoff = INITIAL_BACKOFF_S
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return query_osm_features(
                cell, snapshot_timestamp,
                filters=ALL_FILTERS,
                event_date=event_date,
                settings=settings,
            )
        except DataAccessError as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "rate limit" in msg.lower()
            if not is_rate_limit or attempt == max_retries:
                raise
            log.info(
                "  rate limited (attempt %d/%d); sleeping %.1f s.",
                attempt, max_retries, backoff,
            )
            time.sleep(backoff)
            backoff *= 2.0
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise DataAccessError("_query_with_retry exhausted retries without success")


# ---------------------------------------------------------------------------
# Tiled fetch
# ---------------------------------------------------------------------------

def _fetch_all_osm_features_tiled(
    bbox: BoundingBox,
    snapshot_timestamp: date,
    *,
    event_date: Optional[date],
    settings: Settings,
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch buildings, roads, and bridges in one query per tile.

    Returns a dict keyed by filter expression, each mapping to a list of
    GeoJSON features. Features are deduplicated by OSM ID across tiles.
    """
    cells = _split_bbox_into_cells(bbox, cell_size_deg)
    log.info(
        "Fetching OSM infrastructure from %d tile(s) of %.3f° × %.3f°...",
        len(cells), cell_size_deg, cell_size_deg,
    )

    seen_ids: dict[str, set[str]] = {f: set() for f in ALL_FILTERS}
    collected: dict[str, list[dict[str, Any]]] = {f: [] for f in ALL_FILTERS}

    tiles_with_features = 0
    tiles_with_any_result = 0
    tiles_failed = 0

    for i, cell in enumerate(cells, 1):
        try:
            result = _query_with_retry(
                cell, snapshot_timestamp,
                event_date=event_date,
                settings=settings,
            )
            tiles_with_any_result += 1

            tile_added = 0
            for filt in ALL_FILTERS:
                features = result.get(filt, [])
                for feat in features:
                    feat_id = feat.get("id")
                    if feat_id and feat_id in seen_ids[filt]:
                        continue
                    if feat_id:
                        seen_ids[filt].add(feat_id)
                    collected[filt].append(feat)
                    tile_added += 1

            if tile_added > 0:
                tiles_with_features += 1

            log.info(
                "  tile %d/%d: +%d new features "
                "(B=%d, R=%d, Br=%d cumulative)",
                i, len(cells), tile_added,
                len(collected[FILTER_BUILDINGS]),
                len(collected[FILTER_ROADS]),
                len(collected[FILTER_BRIDGES]),
            )

        except NoResultsError:
            log.info("  tile %d/%d: empty", i, len(cells))
        except DataAccessError as exc:
            tiles_failed += 1
            log.warning("  tile %d/%d failed: %s", i, len(cells), exc)

        if i < len(cells):
            time.sleep(TILE_PAUSE_S)

    log.info(
        "Tiled fetch complete: %d tiles queried, %d returned data, "
        "%d with matching features, %d failed.",
        len(cells), tiles_with_any_result, tiles_with_features, tiles_failed,
    )
    log.info(
        "  total features: %d buildings, %d roads, %d bridges",
        len(collected[FILTER_BUILDINGS]),
        len(collected[FILTER_ROADS]),
        len(collected[FILTER_BRIDGES]),
    )

    if tiles_failed > 0:
        log.warning(
            "%d tile(s) failed after retries. Results may be incomplete.",
            tiles_failed,
        )

    # Attach fetch metadata so callers can report data completeness.
    collected["__fetch_metadata__"] = {
        "tiles_total": len(cells),
        "tiles_with_features": tiles_with_features,
        "tiles_with_any_result": tiles_with_any_result,
        "tiles_failed": tiles_failed,
        "cell_size_deg": cell_size_deg,
    }
    return collected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flood_extent_gdf(flood_geojson_path: Path) -> gpd.GeoDataFrame:
    """Load the flood extent GeoJSON into a GeoDataFrame (WGS84)."""
    path = Path(flood_geojson_path)
    if not path.is_file():
        raise DataAccessError(
            f"Flood extent GeoJSON not found: {path}. "
            f"Run Batch 3C vectorization first."
        )
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise DataAccessError(f"Flood extent GeoDataFrame is empty: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    return gdf.to_crs(epsg=4326)


def _safe_union(geoseries: gpd.GeoSeries):
    """Return the union of a GeoSeries, compatible with both geopandas versions."""
    if hasattr(geoseries, "union_all"):
        return geoseries.union_all()
    return geoseries.unary_union


def _osm_features_to_gdf(
    features: list[dict[str, Any]],
    *,
    geometry_type: str = "any",
) -> gpd.GeoDataFrame:
    """Convert a list of ohsome GeoJSON features to a GeoDataFrame."""
    if not features:
        return gpd.GeoDataFrame(
            columns=["geometry", "osm_id", "osm_type", "tags"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    rows: list[dict[str, Any]] = []
    for feat in features:
        geom_dict = feat.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue

        if geometry_type == "polygon" and geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if geometry_type == "line" and geom.geom_type not in ("LineString", "MultiLineString"):
            continue

        props = feat.get("properties", {})
        rows.append({
            "geometry": geom,
            "osm_id": props.get("@osmId"),
            "osm_type": props.get("@osmType"),
            "tags": props.get("tags", {}),
        })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["geometry", "osm_id", "osm_type", "tags"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Exposure computation
# ---------------------------------------------------------------------------

def compute_building_exposure(
    flood_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    *,
    projected_crs: CRS,
    buffer_m: float = 0.0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Find buildings intersecting the flood extent."""
    if buildings_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "buildings_intersecting_count": 0,
            "buildings_exposed_area_sqm": 0.0,
            "warnings": ["No buildings retrieved from OSM."],
        }

    flood_proj = flood_gdf.to_crs(projected_crs)
    buildings_proj = buildings_gdf.to_crs(projected_crs)

    if buffer_m > 0:
        flood_proj = flood_proj.copy()
        flood_proj["geometry"] = flood_proj.geometry.buffer(buffer_m)

    try:
        joined = gpd.sjoin(
            buildings_proj,
            flood_proj[["geometry"]],
            how="inner",
            predicate="intersects",
        )
    except Exception as exc:
        log.warning("Spatial join failed for buildings: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "buildings_intersecting_count": 0,
            "buildings_exposed_area_sqm": 0.0,
            "warnings": [f"Building spatial join failed: {exc}"],
        }

    if "osm_id" in joined.columns:
        joined = joined.drop_duplicates(subset=["osm_id"], keep="first")

    if not joined.empty:
        flood_union = _safe_union(flood_proj.geometry)
        joined["intersect_area_sqm"] = joined.geometry.intersection(flood_union).area
    else:
        joined["intersect_area_sqm"] = 0.0

    total_area = float(joined["intersect_area_sqm"].sum())
    summary = {
        "buildings_intersecting_count": int(len(joined)),
        "buildings_exposed_area_sqm": round(total_area, 2),
        "buildings_exposed_area_hectares": round(total_area / 10_000.0, 4),
        "buffer_m": buffer_m,
        "warnings": [],
    }
    return joined, summary


def compute_road_exposure(
    flood_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    *,
    projected_crs: CRS,
    buffer_m: float = 0.0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Find roads intersecting the flood extent and compute affected length."""
    if roads_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "roads_intersecting_count": 0,
            "roads_affected_length_km": 0.0,
            "warnings": ["No roads retrieved from OSM."],
        }

    flood_proj = flood_gdf.to_crs(projected_crs)
    roads_proj = roads_gdf.to_crs(projected_crs)

    if buffer_m > 0:
        flood_proj = flood_proj.copy()
        flood_proj["geometry"] = flood_proj.geometry.buffer(buffer_m)

    try:
        joined = gpd.sjoin(
            roads_proj,
            flood_proj[["geometry"]],
            how="inner",
            predicate="intersects",
        )
    except Exception as exc:
        log.warning("Spatial join failed for roads: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "roads_intersecting_count": 0,
            "roads_affected_length_km": 0.0,
            "warnings": [f"Road spatial join failed: {exc}"],
        }

    if "osm_id" in joined.columns:
        joined = joined.drop_duplicates(subset=["osm_id"], keep="first")

    if not joined.empty:
        flood_union = _safe_union(flood_proj.geometry)
        joined["intersect_length_m"] = joined.geometry.intersection(flood_union).length
    else:
        joined["intersect_length_m"] = 0.0

    total_length_m = float(joined["intersect_length_m"].sum())
    summary = {
        "roads_intersecting_count": int(len(joined)),
        "roads_affected_length_km": round(total_length_m / 1000.0, 3),
        "buffer_m": buffer_m,
        "warnings": [],
    }
    return joined, summary


def compute_bridge_exposure(
    flood_gdf: gpd.GeoDataFrame,
    bridges_gdf: gpd.GeoDataFrame,
    *,
    projected_crs: CRS,
    buffer_m: float = 0.0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Find bridges intersecting the flood extent."""
    if bridges_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "bridges_intersecting_count": 0,
            "warnings": ["No bridges retrieved from OSM."],
        }

    flood_proj = flood_gdf.to_crs(projected_crs)
    bridges_proj = bridges_gdf.to_crs(projected_crs)

    if buffer_m > 0:
        flood_proj = flood_proj.copy()
        flood_proj["geometry"] = flood_proj.geometry.buffer(buffer_m)

    try:
        joined = gpd.sjoin(
            bridges_proj,
            flood_proj[["geometry"]],
            how="inner",
            predicate="intersects",
        )
    except Exception as exc:
        log.warning("Spatial join failed for bridges: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs), {
            "bridges_intersecting_count": 0,
            "warnings": [f"Bridge spatial join failed: {exc}"],
        }

    if "osm_id" in joined.columns:
        joined = joined.drop_duplicates(subset=["osm_id"], keep="first")

    summary = {
        "bridges_intersecting_count": int(len(joined)),
        "buffer_m": buffer_m,
        "warnings": [],
    }
    return joined, summary


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def analyze_infrastructure_exposure(
    flood_geojson_path: Path,
    aoi_bbox: BoundingBox,
    *,
    osm_snapshot: date = DEFAULT_OSM_SNAPSHOT,
    event_date: date = date(2026, 8, 26),
    buffer_m: float = 0.0,
    output_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
) -> dict[str, Any]:
    """Run the full infrastructure exposure analysis."""
    settings = settings or get_settings()
    output_dir = output_dir or (settings.output_path / "infrastructure")
    output_dir.mkdir(parents=True, exist_ok=True)

    projected_crs = utm_crs_for_bbox(aoi_bbox)
    warnings: list[str] = []

    # --- Load flood extent ---
    flood_gdf = _flood_extent_gdf(flood_geojson_path)
    log.info("Loaded flood extent: %d polygon(s).", len(flood_gdf))

    # --- Fetch OSM data ---
    log.info("Fetching pre-event OSM snapshot %s...", osm_snapshot)

    try:
        features_by_filter = _fetch_all_osm_features_tiled(
            aoi_bbox, osm_snapshot,
            event_date=event_date,
            settings=settings,
            cell_size_deg=cell_size_deg,
        )
    except Exception as exc:
        warnings.append(f"OSM fetch failed: {exc}")
        log.error("OSM fetch failed: %s", exc)
        features_by_filter = {f: [] for f in ALL_FILTERS}

    fetch_meta = features_by_filter.pop("__fetch_metadata__", {})

    buildings_features = features_by_filter.get(FILTER_BUILDINGS, [])
    roads_features = features_by_filter.get(FILTER_ROADS, [])
    bridges_features = features_by_filter.get(FILTER_BRIDGES, [])

    # --- Convert to GeoDataFrames ---
    buildings_gdf = _osm_features_to_gdf(buildings_features, geometry_type="polygon")
    roads_gdf = _osm_features_to_gdf(roads_features, geometry_type="line")
    bridges_gdf = _osm_features_to_gdf(bridges_features, geometry_type="line")

    log.info(
        "OSM GeoDataFrames: %d buildings, %d roads, %d bridges.",
        len(buildings_gdf), len(roads_gdf), len(bridges_gdf),
    )

    # --- Compute exposure ---
    buildings_exposed, buildings_summary = compute_building_exposure(
        flood_gdf, buildings_gdf,
        projected_crs=projected_crs, buffer_m=buffer_m,
    )
    roads_exposed, roads_summary = compute_road_exposure(
        flood_gdf, roads_gdf,
        projected_crs=projected_crs, buffer_m=buffer_m,
    )
    bridges_exposed, bridges_summary = compute_bridge_exposure(
        flood_gdf, bridges_gdf,
        projected_crs=projected_crs, buffer_m=buffer_m,
    )

    warnings.extend(buildings_summary.get("warnings", []))
    warnings.extend(roads_summary.get("warnings", []))
    warnings.extend(bridges_summary.get("warnings", []))

    # --- Write outputs ---
    buildings_path = output_dir / "buildings_exposed.geojson"
    roads_path = output_dir / "roads_exposed.geojson"
    bridges_path = output_dir / "bridges_exposed.geojson"

    for gdf, path, label in (
        (buildings_exposed, buildings_path, "buildings"),
        (roads_exposed, roads_path, "roads"),
        (bridges_exposed, bridges_path, "bridges"),
    ):
        if gdf.empty:
            log.info("No exposed %s to write.", label)
            continue
        export_gdf = gdf.to_crs(epsg=4326)
        for col in ("index_right", "intersect_area_sqm", "intersect_length_m"):
            if col in export_gdf.columns:
                export_gdf = export_gdf.drop(columns=[col])
        export_gdf.to_file(path, driver="GeoJSON")
        log.info("Wrote %s (%d features).", path.name, len(export_gdf))

    # --- Summary JSON ---
    summary = {
        "flood_extent_path": str(flood_geojson_path),
        "osm_snapshot": osm_snapshot.isoformat(),
        "event_date": event_date.isoformat(),
        "buffer_m": buffer_m,
        "cell_size_deg": cell_size_deg,
        "buildings": buildings_summary,
        "roads": roads_summary,
        "bridges": bridges_summary,
        "warnings": warnings,
        "note": (
            "Counts reflect infrastructure intersecting the candidate "
            "flood extent. They are NOT confirmed damage assessments."
        ),
    }
    summary_path = output_dir / "infrastructure_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Wrote summary: %s", summary_path.name)

    return summary


__all__ = [
    "DEFAULT_OSM_SNAPSHOT",
    "analyze_infrastructure_exposure",
    "compute_bridge_exposure",
    "compute_building_exposure",
    "compute_road_exposure",
]