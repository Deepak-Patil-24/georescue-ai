"""Settlement connectivity and access disruption analysis.

This module answers the third core hackathon question: **who may be cut off?**

Method
------
1. Fetch pre-event OSM features for a bounding box: settlements, hospitals,
   towns, and the road network.
2. Build a NetworkX graph whose nodes are road-segment endpoints
   (intersections and dead-ends) and whose edges are road segments with
   length attributes.
3. Merge nearby graph nodes to close gaps caused by inexact OSM endpoint
   drawing (a common reason the OSM road network looks fragmented).
4. Snap settlements and destinations to their nearest graph node.
5. Compute baseline connectivity: for each settlement, can it reach at
   least one destination through the pre-event road network?
6. Identify road segments whose intersection with the candidate flood
   extent exceeds a configurable fraction of their length. These are
   marked as **potentially impassable** in the modelled scenario.
7. Remove those edges from the graph and recompute connectivity.
8. Report settlements that were connected in the baseline but become
   disconnected in the scenario.

Critical terminology
--------------------
- "Modelled access disruption scenario" — not a real-world prediction.
- "Potentially cut off" — based on the graph model, not verified by ground
  truth.
- "Potentially impassable road" — a road intersecting the flood above a
  threshold. It is **not** confirmed impassable.

Limitations
-----------
- OSM road networks are incomplete in remote areas, especially in the
  Himalaya. Missing roads cause false positives (settlements appear
  disconnected) and false negatives (missing settlements).
- Road segments do not encode elevation or bridge status in most OSM
  data. A flooded segment may in reality be a bridge that remains above
  water, or a mountain road that stays dry above the flood line.
- The disruption rule is a heuristic. It does not model water depth,
  debris, or structural damage.
- The result is a scenario that helps prioritise investigation, not a
  verified accessibility assessment.
"""

from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
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

DEFAULT_CELL_SIZE_DEG = 0.10
TILE_PAUSE_S = 6.0  # 10 requests/minute, safely under the 15/min sliding window
MAX_RETRIES = 4
INITIAL_BACKOFF_S = 10.0

# Road segment disruption threshold: a segment is treated as potentially
# impassable if this fraction of its length intersects the flood extent.
DEFAULT_DISRUPTION_FRACTION = 0.20

# Node-merge tolerance in metres. Road segments whose endpoints are within
# this distance are treated as connected. This closes the small gaps in
# OSM road networks caused by inexact endpoint drawing.
DEFAULT_NODE_MERGE_M = 50.0

# Settlement and destination filters.
FILTER_ROADS = "highway=* and geometry:line"
FILTER_SETTLEMENTS = "place in (village,hamlet,town) and geometry:point"
FILTER_TOWNS = "place in (town,city) and geometry:point"
FILTER_HOSPITALS = "amenity in (hospital,clinic) and geometry:point"

ALL_FILTERS = (
    FILTER_ROADS,
    FILTER_SETTLEMENTS,
    FILTER_TOWNS,
    FILTER_HOSPITALS,
)

EXCLUDED_HIGHWAY_TYPES = {
    "footway", "path", "steps", "cycleway", "bridleway", "pedestrian",
    "corridor", "construction", "proposed", "raceway",
}


# ---------------------------------------------------------------------------
# Tiled OSM fetch
# ---------------------------------------------------------------------------

def _split_bbox_into_cells(bbox: BoundingBox, cell_size_deg: float) -> list[BoundingBox]:
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


def _query_with_retry(
    cell: BoundingBox,
    snapshot_timestamp: date,
    *,
    event_date: Optional[date],
    settings: Settings,
    max_retries: int = MAX_RETRIES,
) -> dict[str, list[dict[str, Any]]]:
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
            is_transient = (
                "timeout" in msg.lower()
                or "read timed out" in msg.lower()
                or "name resolution" in msg.lower()
                or "failed to resolve" in msg.lower()
            )
            if (not (is_rate_limit or is_transient)) or attempt == max_retries:
                raise
            log.info(
                "  transient error (attempt %d/%d); sleeping %.1f s.",
                attempt, max_retries, backoff,
            )
            time.sleep(backoff)
            backoff *= 2.0
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise DataAccessError("_query_with_retry exhausted retries")


def _fetch_all_features_tiled(
    bbox: BoundingBox,
    snapshot_timestamp: date,
    *,
    event_date: Optional[date],
    settings: Settings,
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
) -> dict[str, list[dict[str, Any]]]:
    cells = _split_bbox_into_cells(bbox, cell_size_deg)
    log.info("Fetching connectivity features from %d tiles...", len(cells))

    seen_ids: dict[str, set[str]] = {f: set() for f in ALL_FILTERS}
    collected: dict[str, list[dict[str, Any]]] = {f: [] for f in ALL_FILTERS}
    tiles_failed = 0

    for i, cell in enumerate(cells, 1):
        try:
            result = _query_with_retry(
                cell, snapshot_timestamp,
                event_date=event_date,
                settings=settings,
            )
            for filt in ALL_FILTERS:
                for feat in result.get(filt, []):
                    feat_id = feat.get("id")
                    if feat_id and feat_id in seen_ids[filt]:
                        continue
                    if feat_id:
                        seen_ids[filt].add(feat_id)
                    collected[filt].append(feat)

            log.info(
                "  tile %d/%d: cumulative (R=%d, S=%d, T=%d, H=%d)",
                i, len(cells),
                len(collected[FILTER_ROADS]),
                len(collected[FILTER_SETTLEMENTS]),
                len(collected[FILTER_TOWNS]),
                len(collected[FILTER_HOSPITALS]),
            )
        except NoResultsError:
            log.info("  tile %d/%d: empty", i, len(cells))
        except DataAccessError as exc:
            tiles_failed += 1
            log.warning("  tile %d/%d failed: %s", i, len(cells), exc)

        if i < len(cells):
            time.sleep(TILE_PAUSE_S)

    if tiles_failed:
        log.warning("%d tile(s) failed; results may be incomplete.", tiles_failed)
    log.info(
        "Fetch complete: %d roads, %d settlements, %d towns, %d hospitals",
        len(collected[FILTER_ROADS]),
        len(collected[FILTER_SETTLEMENTS]),
        len(collected[FILTER_TOWNS]),
        len(collected[FILTER_HOSPITALS]),
    )
    collected["__fetch_metadata__"] = {
        "tiles_total": len(cells),
        "tiles_failed": tiles_failed,
        "cell_size_deg": cell_size_deg,
    }
    return collected


# ---------------------------------------------------------------------------
# GeoDataFrame conversion
# ---------------------------------------------------------------------------

def _features_to_gdf(features, *, geometry_type: str = "any") -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame(
            columns=["geometry", "osm_id", "osm_type", "tags"],
            geometry="geometry", crs="EPSG:4326",
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

        if geometry_type == "line" and geom.geom_type not in ("LineString", "MultiLineString"):
            continue
        if geometry_type == "point" and geom.geom_type != "Point":
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
            geometry="geometry", crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _segment_endpoints(geom) -> tuple[tuple[float, float], tuple[float, float]]:
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        return (coords[0][:2], coords[-1][:2])
    if geom.geom_type == "MultiLineString":
        first = list(geom.geoms[0].coords)[0]
        last = list(geom.geoms[-1].coords)[-1]
        return (first[:2], last[:2])
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def build_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    *,
    projected_crs: CRS,
    snap_to_m: float = 5.0,
) -> nx.MultiGraph:
    """Build a NetworkX MultiGraph from road segments.

    Nodes are endpoints rounded to ``snap_to_m`` metres in the projected
    CRS. This merges endpoints that are very close together.

    Use :func:`merge_nearby_nodes` afterwards to close larger gaps.
    """
    roads_proj = roads_gdf.to_crs(projected_crs)
    graph = nx.MultiGraph()
    skipped = 0

    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        tags = row.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        highway = tags.get("highway")
        if highway in EXCLUDED_HIGHWAY_TYPES:
            skipped += 1
            continue

        try:
            start, end = _segment_endpoints(geom)
        except ValueError:
            skipped += 1
            continue

        # Snap to snap_to_m grid.
        a = (round(start[0] / snap_to_m) * snap_to_m, round(start[1] / snap_to_m) * snap_to_m)
        b = (round(end[0] / snap_to_m) * snap_to_m, round(end[1] / snap_to_m) * snap_to_m)

        if a == b:
            skipped += 1
            continue

        graph.add_edge(
            a, b,
            key=str(row.get("osm_id") or f"{a}-{b}"),
            osm_id=row.get("osm_id"),
            highway=highway,
            length_m=float(geom.length),
            geometry=geom,
        )

    log.info(
        "Road graph (pre-merge): %d nodes, %d edges (%d segments skipped).",
        graph.number_of_nodes(), graph.number_of_edges(), skipped,
    )
    return graph


def merge_nearby_nodes(
    graph: nx.MultiGraph,
    *,
    tolerance_m: float,
) -> nx.MultiGraph:
    """Merge graph nodes whose coordinates are within ``tolerance_m``.

    Uses a grid-based bucketing approach: nodes are assigned to a cell of
    size ``tolerance_m`` in the projected CRS, and nodes in the same cell
    or in adjacent cells are grouped if their distances are below the
    tolerance. Each group is replaced by its centroid.

    This closes gaps in OSM road networks caused by inexact endpoint
    drawing, which is very common in remote areas.
    """
    if tolerance_m <= 0:
        return graph

    from scipy.spatial import cKDTree

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return graph

    coords = np.array(nodes, dtype=float)
    tree = cKDTree(coords)

    # Query pairs within tolerance.
    pairs = tree.query_pairs(r=tolerance_m, output_type="ndarray")

    if len(pairs) == 0:
        log.info("Node merge: no nodes within %.0f m.", tolerance_m)
        return graph

    # Union-Find to group nodes.
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(int(i), int(j))

    # Build groups keyed by representative index.
    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)

    # Compute centroid for each group.
    node_remap: dict[tuple[float, float], tuple[float, float]] = {}
    for rep, members in groups.items():
        if len(members) == 1:
            continue
        cx = float(np.mean([nodes[m][0] for m in members]))
        cy = float(np.mean([nodes[m][1] for m in members]))
        centroid = (cx, cy)
        for m in members:
            node_remap[nodes[m]] = centroid

    if not node_remap:
        log.info("Node merge: no multi-node groups found.")
        return graph

    # Build a new graph with remapped nodes.
    new_graph = nx.MultiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        u2 = node_remap.get(u, u)
        v2 = node_remap.get(v, v)
        if u2 == v2:
            # Self-loop after merge; skip.
            continue
        new_graph.add_edge(u2, v2, key=key, **data)

    log.info(
        "Node merge: %d -> %d nodes, %d -> %d edges.",
        graph.number_of_nodes(), new_graph.number_of_nodes(),
        graph.number_of_edges(), new_graph.number_of_edges(),
    )
    return new_graph


def _snap_points_to_graph(
    points_gdf: gpd.GeoDataFrame,
    graph: nx.MultiGraph,
    *,
    projected_crs: CRS,
    max_snap_distance_m: float = 5000.0,
) -> dict[str, tuple[float, float]]:
    """Snap points to nearest graph node using KDTree for speed."""
    if points_gdf.empty or graph.number_of_nodes() == 0:
        return {}

    points_proj = points_gdf.to_crs(projected_crs)
    nodes = list(graph.nodes())
    node_array = np.array(nodes, dtype=float)

    from scipy.spatial import cKDTree
    tree = cKDTree(node_array)

    snapped: dict[str, tuple[float, float]] = {}
    for _, row in points_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        point = geom.centroid if geom.geom_type != "Point" else geom
        dist, idx = tree.query((point.x, point.y))
        if dist > max_snap_distance_m:
            continue
        snapped[str(row.get("osm_id"))] = (
            float(node_array[idx, 0]), float(node_array[idx, 1])
        )

    log.info("Snapped %d of %d points to graph nodes.", len(snapped), len(points_gdf))
    return snapped


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------

def compute_connectivity(
    graph: nx.MultiGraph,
    settlements: dict[str, tuple[float, float]],
    destinations: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Compute per-settlement connectivity to the nearest destination."""
    if not destinations:
        log.warning("No destinations; connectivity cannot be computed.")
        return {
            sid: {
                "connected": False,
                "distance_m": math.inf,
                "destination_id": None,
                "snapped_node": node,
            }
            for sid, node in settlements.items()
        }

    distances_to_dest: dict[Any, float] = {}
    nearest_dest: dict[Any, Any] = {}

    for dest_id, dest_node in destinations.items():
        if dest_node not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph, dest_node, weight="length_m",
            )
        except Exception as exc:
            log.warning("Dijkstra from %s failed: %s", dest_id, exc)
            continue
        for node, dist in lengths.items():
            if node not in distances_to_dest or dist < distances_to_dest[node]:
                distances_to_dest[node] = dist
                nearest_dest[node] = dest_id

    results: dict[str, dict[str, Any]] = {}
    for sid, node in settlements.items():
        if node in distances_to_dest:
            results[sid] = {
                "connected": True,
                "distance_m": float(distances_to_dest[node]),
                "destination_id": nearest_dest.get(node),
                "snapped_node": node,
            }
        else:
            results[sid] = {
                "connected": False,
                "distance_m": math.inf,
                "destination_id": None,
                "snapped_node": node,
            }

    n_connected = sum(1 for r in results.values() if r["connected"])
    log.info(
        "Connectivity: %d of %d settlements can reach a destination.",
        n_connected, len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Flood-based disruption
# ---------------------------------------------------------------------------

def identify_disrupted_edges(
    graph: nx.MultiGraph,
    flood_geometry,
    *,
    disruption_fraction: float = DEFAULT_DISRUPTION_FRACTION,
) -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    """Return edges whose flooded fraction exceeds ``disruption_fraction``."""
    if flood_geometry is None or flood_geometry.is_empty:
        return []

    disrupted: list[tuple[tuple[float, float], tuple[float, float], str]] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        length = data.get("length_m") or geom.length
        if length <= 0:
            continue
        try:
            intersected = geom.intersection(flood_geometry).length
        except Exception:
            continue
        if intersected / length >= disruption_fraction:
            disrupted.append((u, v, key))

    log.info(
        "Disruption: %d of %d edges flood >%.0f%% of their length.",
        len(disrupted), graph.number_of_edges(), disruption_fraction * 100,
    )
    return disrupted


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def analyze_connectivity(
    flood_geojson_path: Path,
    aoi_bbox: BoundingBox,
    *,
    osm_snapshot: date = DEFAULT_OSM_SNAPSHOT,
    event_date: date = date(2026, 8, 26),
    disruption_fraction: float = DEFAULT_DISRUPTION_FRACTION,
    node_merge_m: float = DEFAULT_NODE_MERGE_M,
    output_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
) -> dict[str, Any]:
    """Run the full settlement connectivity analysis."""
    settings = settings or get_settings()
    output_dir = output_dir or (settings.output_path / "connectivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    projected_crs = utm_crs_for_bbox(aoi_bbox)
    warnings: list[str] = []

    # --- Load flood extent ---
    flood_path = Path(flood_geojson_path)
    if not flood_path.is_file():
        raise DataAccessError(f"Flood extent not found: {flood_path}")
    flood_gdf = gpd.read_file(flood_path)
    if flood_gdf.crs is None:
        flood_gdf = flood_gdf.set_crs(epsg=4326, allow_override=True)
    flood_gdf = flood_gdf.to_crs(projected_crs)
    flood_geometry = (
        flood_gdf.geometry.union_all()
        if hasattr(flood_gdf.geometry, "union_all")
        else flood_gdf.geometry.unary_union
    )
    log.info("Loaded flood extent: %d polygon(s).", len(flood_gdf))

    # --- Fetch OSM features ---
    try:
        features = _fetch_all_features_tiled(
            aoi_bbox, osm_snapshot,
            event_date=event_date,
            settings=settings,
            cell_size_deg=cell_size_deg,
        )
    except Exception as exc:
        raise DataAccessError(f"OSM fetch failed: {exc}") from exc

    fetch_meta = features.pop("__fetch_metadata__", {})

    roads_gdf = _features_to_gdf(features.get(FILTER_ROADS, []), geometry_type="line")
    settlements_gdf = _features_to_gdf(features.get(FILTER_SETTLEMENTS, []), geometry_type="point")
    towns_gdf = _features_to_gdf(features.get(FILTER_TOWNS, []), geometry_type="point")
    hospitals_gdf = _features_to_gdf(features.get(FILTER_HOSPITALS, []), geometry_type="point")

    log.info(
        "Fetched: %d road segments, %d settlements, %d towns, %d hospitals.",
        len(roads_gdf), len(settlements_gdf), len(towns_gdf), len(hospitals_gdf),
    )

    if roads_gdf.empty:
        raise DataAccessError("No road segments retrieved.")

    # --- Build graph, then merge nearby nodes ---
    graph = build_road_graph(roads_gdf, projected_crs=projected_crs)
    graph = merge_nearby_nodes(graph, tolerance_m=node_merge_m)

    # --- Snap settlements and destinations ---
    settlements_snapped = _snap_points_to_graph(
        settlements_gdf, graph, projected_crs=projected_crs,
    )
    destinations_snapped: dict[str, tuple[float, float]] = {}
    for dest_gdf, prefix in ((towns_gdf, "town"), (hospitals_gdf, "hospital")):
        snapped = _snap_points_to_graph(dest_gdf, graph, projected_crs=projected_crs)
        for k, v in snapped.items():
            destinations_snapped[f"{prefix}-{k}"] = v

    # --- Baseline connectivity ---
    baseline = compute_connectivity(graph, settlements_snapped, destinations_snapped)

    # --- Disruption scenario ---
    disrupted_edges = identify_disrupted_edges(
        graph, flood_geometry,
        disruption_fraction=disruption_fraction,
    )
    disrupted_graph = graph.copy()
    disrupted_graph.remove_edges_from(disrupted_edges)

    scenario = compute_connectivity(
        disrupted_graph, settlements_snapped, destinations_snapped,
    )

    # --- Compare ---
    newly_disconnected: list[dict[str, Any]] = []
    for sid, base in baseline.items():
        scen = scenario.get(sid, {})
        if base["connected"] and not scen.get("connected", False):
            newly_disconnected.append({
                "settlement_id": sid,
                "baseline_distance_m": base["distance_m"],
                "baseline_destination": base["destination_id"],
            })

    log.info(
        "Newly disconnected: %d of %d baseline-connected settlements.",
        len(newly_disconnected),
        sum(1 for r in baseline.values() if r["connected"]),
    )

    # --- Save disrupted roads layer ---
    if disrupted_edges:
        disrupted_rows = []
        for u, v, key in disrupted_edges:
            data = graph.get_edge_data(u, v, key)
            if data is None:
                continue
            geom = data.get("geometry")
            if geom is None:
                continue
            disrupted_rows.append({
                "osm_id": data.get("osm_id"),
                "highway": data.get("highway"),
                "length_m": data.get("length_m"),
                "geometry": geom,
            })
        if disrupted_rows:
            disrupted_gdf = gpd.GeoDataFrame(
                disrupted_rows, geometry="geometry", crs=projected_crs,
            ).to_crs(epsg=4326)
            disrupted_gdf.to_file(
                output_dir / "disrupted_roads.geojson", driver="GeoJSON",
            )
            log.info("Wrote disrupted_roads.geojson.")

    # --- Save disconnected settlements layer ---
    if newly_disconnected:
        settlement_ids = {d["settlement_id"] for d in newly_disconnected}
        disconnected_gdf = settlements_gdf[
            settlements_gdf["osm_id"].astype(str).isin(settlement_ids)
        ].copy()
        if not disconnected_gdf.empty:
            distance_map = {
                d["settlement_id"]: d["baseline_distance_m"]
                for d in newly_disconnected
            }
            disconnected_gdf["baseline_distance_m"] = (
                disconnected_gdf["osm_id"].astype(str).map(distance_map)
            )
            disconnected_gdf.to_file(
                output_dir / "disconnected_settlements.geojson", driver="GeoJSON",
            )
            log.info("Wrote disconnected_settlements.geojson.")

    # --- Summary ---
    summary = {
        "aoi_bbox": list(aoi_bbox.as_tuple()),
        "osm_snapshot": osm_snapshot.isoformat(),
        "event_date": event_date.isoformat(),
        "disruption_fraction": disruption_fraction,
        "node_merge_m": node_merge_m,
        "fetch_metadata": fetch_meta,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "counts": {
            "settlements_in_graph": len(settlements_snapped),
            "destinations_in_graph": len(destinations_snapped),
            "baseline_connected": sum(1 for r in baseline.values() if r["connected"]),
            "scenario_connected": sum(1 for r in scenario.values() if r["connected"]),
            "newly_disconnected": len(newly_disconnected),
            "disrupted_edges": len(disrupted_edges),
        },
        "newly_disconnected_settlements": newly_disconnected[:100],
        "warnings": warnings,
        "note": (
            "Modelled access disruption scenario. Not a verified accessibility "
            "assessment. Potentially impassable roads are those intersecting "
            "the flood extent over at least {:.0f}% of their length.".format(
                disruption_fraction * 100
            )
        ),
    }
    with open(output_dir / "connectivity_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Wrote connectivity_summary.json.")

    return summary


__all__ = [
    "analyze_connectivity",
    "build_road_graph",
    "compute_connectivity",
    "identify_disrupted_edges",
    "merge_nearby_nodes",
]