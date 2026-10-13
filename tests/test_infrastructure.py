"""Tests for the infrastructure module.

Exercises the low-level building/road exposure functions on synthetic
geometries. These tests do not require OSM access — they build
GeoDataFrames in memory from the fixtures in ``conftest.py``.

The fixtures provide:

- A 1 km² square flood polygon in UTM 45N.
- Three building footprints, two inside the flood polygon.
- Three road lines, two crossing the flood polygon.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from rasterio.crs import CRS
from shapely.geometry import LineString, Polygon, box

from georescue.analysis.infrastructure import (
    compute_building_exposure,
    compute_road_exposure,
)


UTM45N = CRS.from_epsg(32645)


def _gdf_from_geoms(geoms, crs_epsg: int = 32645) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"osm_id": [str(i) for i in range(len(geoms))]},
        geometry=geoms,
        crs=f"EPSG:{crs_epsg}",
    )


# ---------------------------------------------------------------------------
# Building exposure
# ---------------------------------------------------------------------------

def test_building_exposure_counts_only_intersecting(
    synthetic_flood_polygon: Polygon,
    synthetic_buildings_in_flood,
):
    """Two of three buildings intersect the flood polygon."""
    flood_gdf = _gdf_from_geoms([synthetic_flood_polygon])
    buildings_gdf = _gdf_from_geoms(synthetic_buildings_in_flood)

    exposed, summary = compute_building_exposure(
        flood_gdf, buildings_gdf,
        projected_crs=UTM45N, buffer_m=0.0,
    )

    assert summary["buildings_intersecting_count"] == 2, (
        f"Expected 2 buildings inside, got "
        f"{summary['buildings_intersecting_count']}"
    )
    # Two 50 m × 50 m buildings = 5000 m² total
    assert summary["buildings_exposed_area_sqm"] == pytest.approx(5000.0, rel=0.01)


def test_building_exposure_empty_input():
    """With no buildings, the summary reports zero and a warning."""
    flood_gdf = _gdf_from_geoms([box(0, 0, 100, 100)])
    empty = gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:32645")

    exposed, summary = compute_building_exposure(
        flood_gdf, empty,
        projected_crs=UTM45N,
    )
    assert summary["buildings_intersecting_count"] == 0
    assert summary["buildings_exposed_area_sqm"] == 0.0
    assert "warnings" in summary


def test_building_exposure_no_intersections():
    """Buildings far from the flood polygon should not be counted."""
    flood_gdf = _gdf_from_geoms([box(0, 0, 100, 100)])
    far_buildings = _gdf_from_geoms([box(500, 500, 550, 550)])

    exposed, summary = compute_building_exposure(
        flood_gdf, far_buildings,
        projected_crs=UTM45N,
    )
    assert summary["buildings_intersecting_count"] == 0


# ---------------------------------------------------------------------------
# Road exposure
# ---------------------------------------------------------------------------

def test_road_exposure_counts_intersecting_lines(
    synthetic_flood_polygon: Polygon,
    synthetic_roads_in_flood,
):
    """Two of three roads intersect the flood polygon."""
    flood_gdf = _gdf_from_geoms([synthetic_flood_polygon])
    roads_gdf = _gdf_from_geoms(synthetic_roads_in_flood)

    exposed, summary = compute_road_exposure(
        flood_gdf, roads_gdf,
        projected_crs=UTM45N, buffer_m=0.0,
    )

    assert summary["roads_intersecting_count"] == 2, (
        f"Expected 2 roads crossing, got "
        f"{summary['roads_intersecting_count']}"
    )
    # Road 1 crosses 1000 m fully inside the flood box.
    # Road 2 is 600 m long, all inside the box.
    # Total intersection length should be at least 1000 m.
    assert summary["roads_affected_length_km"] >= 1.0, (
        f"Expected at least 1 km, got "
        f"{summary['roads_affected_length_km']} km"
    )


def test_road_exposure_empty_input():
    flood_gdf = _gdf_from_geoms([box(0, 0, 100, 100)])
    empty = gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:32645")

    exposed, summary = compute_road_exposure(
        flood_gdf, empty,
        projected_crs=UTM45N,
    )
    assert summary["roads_intersecting_count"] == 0
    assert summary["roads_affected_length_km"] == 0.0


def test_road_exposure_length_is_intersection_only():
    """A road that extends far outside the flood should only count its
    intersection length.

    A 2000 m line where only 100 m is inside the flood polygon should
    report ~0.1 km, not 2.0 km.
    """
    flood_polygon = box(0, 0, 100, 100)
    # A line from x=-950 to x=+1050 at y=50.
    # It intersects the flood polygon from x=0 to x=100, a length of 100 m.
    road = LineString([(-950.0, 50.0), (1050.0, 50.0)])

    flood_gdf = _gdf_from_geoms([flood_polygon])
    roads_gdf = _gdf_from_geoms([road])

    exposed, summary = compute_road_exposure(
        flood_gdf, roads_gdf,
        projected_crs=UTM45N,
    )
    assert summary["roads_intersecting_count"] == 1
    assert summary["roads_affected_length_km"] == pytest.approx(0.1, abs=0.001), (
        f"Expected 0.1 km intersection, got "
        f"{summary['roads_affected_length_km']}"
    )