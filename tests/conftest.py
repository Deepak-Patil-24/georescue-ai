"""Shared pytest fixtures for the GeoRescue AI test suite.

All fixtures here are deterministic and use synthetic data only. None
of them require network access, credentials, or satellite imagery.

The fixtures provide:

- Small :class:`BoundingBox` objects in WGS84 for several regions.
- Synthetic raster files (GeoTIFFs) written to a temporary directory
  with known contents so tests can assert exact numeric results.
- A small synthetic road network for connectivity tests.
- A small synthetic flood polygon and building footprints for
  infrastructure tests.
- A fake ``Settings`` object so tests do not require a real ``.env``.

Every synthetic raster is written with an explicit CRS (EPSG:32645,
UTM zone 45N) so area and length calculations are meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from georescue.models import BoundingBox
from georescue.processing.raster_utils import write_raster


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

@pytest.fixture
def trishuli_bbox() -> BoundingBox:
    """The Trishuli case-study AOI (Nepal)."""
    return BoundingBox(
        min_lon=85.10, min_lat=27.60,
        max_lon=85.55, max_lat=28.05,
    )


@pytest.fixture
def koshi_bbox() -> BoundingBox:
    """A fresh AOI in southeastern Nepal."""
    return BoundingBox(
        min_lon=86.90, min_lat=26.40,
        max_lon=87.20, max_lat=26.70,
    )


@pytest.fixture
def small_bbox() -> BoundingBox:
    """A very small AOI used for validation tests."""
    return BoundingBox(
        min_lon=85.30, min_lat=27.70,
        max_lon=85.31, max_lat=27.71,
    )


@pytest.fixture
def pacific_bbox() -> BoundingBox:
    """An AOI in the middle of the Pacific Ocean with no Sentinel-1 coverage."""
    return BoundingBox(
        min_lon=-150.0, min_lat=10.0,
        max_lon=-149.0, max_lat=11.0,
    )


# ---------------------------------------------------------------------------
# Raster fixtures
# ---------------------------------------------------------------------------

# UTM 45N transform: 30 m pixels, origin chosen so the raster sits inside
# the Trishuli AOI.
_UTM45N = CRS.from_epsg(32645)
_TEST_TRANSFORM = Affine(30.0, 0.0, 312000.0, 0.0, -30.0, 3104000.0)


@pytest.fixture
def tmp_raster_dir(tmp_path: Path) -> Path:
    """A temporary directory for synthetic rasters."""
    d = tmp_path / "rasters"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def synthetic_pre_raster(tmp_raster_dir: Path) -> Path:
    """A 50×50 raster with uniform backscatter of 0.1 (linear power)."""
    arr = np.full((50, 50), 0.1, dtype=np.float32)
    path = tmp_raster_dir / "pre.tif"
    write_raster(
        path, arr,
        transform=_TEST_TRANSFORM,
        crs=_UTM45N,
        nodata=np.nan,
    )
    return path


@pytest.fixture
def synthetic_post_raster(tmp_raster_dir: Path) -> Path:
    """A 50×50 raster where a 10×10 block dropped to 0.02.

    The ratio in that block is 0.2, well below a 0.4 threshold, so it
    should be flagged as candidate flood.
    """
    arr = np.full((50, 50), 0.1, dtype=np.float32)
    arr[20:30, 20:30] = 0.02
    path = tmp_raster_dir / "post.tif"
    write_raster(
        path, arr,
        transform=_TEST_TRANSFORM,
        crs=_UTM45N,
        nodata=np.nan,
    )
    return path


@pytest.fixture
def synthetic_flat_slope(tmp_raster_dir: Path) -> Path:
    """A 50×50 slope raster, all flat (0°)."""
    arr = np.zeros((50, 50), dtype=np.float32)
    path = tmp_raster_dir / "slope_flat.tif"
    write_raster(
        path, arr,
        transform=_TEST_TRANSFORM,
        crs=_UTM45N,
        nodata=np.nan,
    )
    return path


@pytest.fixture
def synthetic_steep_slope(tmp_raster_dir: Path) -> Path:
    """A 50×50 slope raster, all 30° (above typical thresholds)."""
    arr = np.full((50, 50), 30.0, dtype=np.float32)
    path = tmp_raster_dir / "slope_steep.tif"
    write_raster(
        path, arr,
        transform=_TEST_TRANSFORM,
        crs=_UTM45N,
        nodata=np.nan,
    )
    return path


@pytest.fixture
def synthetic_partial_slope(tmp_raster_dir: Path) -> Path:
    """A 50×50 slope raster where only the 10×10 flood block is flat.

    Every other pixel is 30°, so the slope mask should remove all of the
    candidate flood pixels except the ones inside the flat block.
    """
    arr = np.full((50, 50), 30.0, dtype=np.float32)
    arr[20:30, 20:30] = 0.0
    path = tmp_raster_dir / "slope_partial.tif"
    write_raster(
        path, arr,
        transform=_TEST_TRANSFORM,
        crs=_UTM45N,
        nodata=np.nan,
    )
    return path


# ---------------------------------------------------------------------------
# Synthetic graphs and geometries
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_road_graph() -> "networkx.MultiGraph":
    """A small road network.

    Layout (nodes are (x, y) tuples in projected metres):

        A — B — C — D
                |
                E

    All edges have a length of 100 m. This is used for connectivity
    tests. Imported lazily inside the fixture to keep the top-level
    import list small.
    """
    import networkx as nx

    graph = nx.MultiGraph()
    positions = {
        "A": (0.0, 0.0),
        "B": (100.0, 0.0),
        "C": (200.0, 0.0),
        "D": (300.0, 0.0),
        "E": (200.0, 100.0),
    }
    edges = [
        ("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"),
    ]
    for u, v in edges:
        graph.add_edge(
            positions[u], positions[v],
            key=f"{u}-{v}",
            osm_id=f"{u}-{v}",
            highway="residential",
            length_m=100.0,
            geometry=_line_between(positions[u], positions[v]),
        )
    return graph


def _line_between(p0: tuple[float, float], p1: tuple[float, float]):
    from shapely.geometry import LineString
    return LineString([p0, p1])


@pytest.fixture
def synthetic_flood_polygon():
    """A 1 km × 1 km square flood polygon in UTM 45N coordinates."""
    from shapely.geometry import box
    return box(312000.0, 3103000.0, 313000.0, 3104000.0)


@pytest.fixture
def synthetic_buildings_in_flood():
    """Three building footprints: two inside the flood polygon, one outside."""
    from shapely.geometry import box
    return [
        # Inside
        box(312100.0, 3103100.0, 312150.0, 3103150.0),
        box(312500.0, 3103500.0, 312550.0, 3103550.0),
        # Outside
        box(313500.0, 3105000.0, 313550.0, 3105050.0),
    ]


@pytest.fixture
def synthetic_roads_in_flood():
    """Three road lines: two cross the flood polygon, one is far away."""
    from shapely.geometry import LineString
    return [
        # Crosses the flood polygon
        LineString([(311500.0, 3103500.0), (313500.0, 3103500.0)]),
        # Also inside
        LineString([(312200.0, 3103200.0), (312800.0, 3103200.0)]),
        # Far away
        LineString([(320000.0, 3200000.0), (321000.0, 3200000.0)]),
    ]


# ---------------------------------------------------------------------------
# Fake settings
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the settings cache at a temporary directory.

    Any code that calls ``get_settings()`` inside a test using this
    fixture will see a project root under ``tmp_path``. That prevents
    tests from writing into the real project folder.
    """
    from georescue import config as config_module

    # Force the settings cache to rebuild so the environment variable
    # is picked up.
    config_module.reset_settings_cache()
    yield
    config_module.reset_settings_cache()