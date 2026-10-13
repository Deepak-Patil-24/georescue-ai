"""Tests for the evaluation module.

Exercises the IoU, precision, recall, and F1 calculations on synthetic
GeoJSON files with known overlap.

The evaluation module reads two GeoJSON files (candidate and reference)
and rasterizes them to a common grid. It returns a metrics dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from georescue.evaluation.metrics import evaluate_flood_extent
from georescue.models import BoundingBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_polygon_geojson(
    path: Path,
    polygons: list[list[tuple[float, float]]],
) -> None:
    """Write a FeatureCollection with the given polygon rings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for i, ring in enumerate(polygons):
        features.append({
            "type": "Feature",
            "id": f"f-{i}",
            "properties": {"id": f"f-{i}"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        })
    payload = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Perfect agreement
# ---------------------------------------------------------------------------

def test_perfect_overlap_gives_iou_one(tmp_path: Path):
    """When candidate and reference are identical, IoU should be 1."""
    square = [
        (85.30, 27.70),
        (85.32, 27.70),
        (85.32, 27.72),
        (85.30, 27.72),
        (85.30, 27.70),
    ]
    candidate = tmp_path / "candidate.geojson"
    reference = tmp_path / "reference.geojson"
    _write_polygon_geojson(candidate, [square])
    _write_polygon_geojson(reference, [square])

    aoi = BoundingBox(min_lon=85.25, min_lat=27.65, max_lon=85.35, max_lat=27.75)
    result = evaluate_flood_extent(
        candidate, reference,
        aoi_bbox=aoi,
        resolution_m=30.0,
    )
    assert result["evaluated"] is True
    assert result["iou"] == pytest.approx(1.0, abs=0.02)
    assert result["precision"] == pytest.approx(1.0, abs=0.02)
    assert result["recall"] == pytest.approx(1.0, abs=0.02)
    assert result["f1"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# No overlap
# ---------------------------------------------------------------------------

def test_no_overlap_gives_iou_zero(tmp_path: Path):
    """Two disjoint squares: IoU = 0, precision = 0, recall = 0."""
    candidate_poly = [
        (85.30, 27.70),
        (85.31, 27.70),
        (85.31, 27.71),
        (85.30, 27.71),
        (85.30, 27.70),
    ]
    reference_poly = [
        (85.33, 27.73),
        (85.34, 27.73),
        (85.34, 27.74),
        (85.33, 27.74),
        (85.33, 27.73),
    ]
    candidate = tmp_path / "candidate.geojson"
    reference = tmp_path / "reference.geojson"
    _write_polygon_geojson(candidate, [candidate_poly])
    _write_polygon_geojson(reference, [reference_poly])

    aoi = BoundingBox(min_lon=85.25, min_lat=27.65, max_lon=85.40, max_lat=27.80)
    result = evaluate_flood_extent(
        candidate, reference,
        aoi_bbox=aoi,
        resolution_m=30.0,
    )
    assert result["evaluated"] is True
    assert result["iou"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Partial overlap
# ---------------------------------------------------------------------------

def test_partial_overlap_metrics_plausible(tmp_path: Path):
    """Candidate and reference overlap by exactly half.

    The two 0.02° × 0.02° squares overlap on a 0.01° × 0.02° strip,
    which is half of each square.
    """
    candidate_poly = [
        (85.30, 27.70),
        (85.32, 27.70),
        (85.32, 27.72),
        (85.30, 27.72),
        (85.30, 27.70),
    ]
    reference_poly = [
        (85.31, 27.70),
        (85.33, 27.70),
        (85.33, 27.72),
        (85.31, 27.72),
        (85.31, 27.70),
    ]
    candidate = tmp_path / "candidate.geojson"
    reference = tmp_path / "reference.geojson"
    _write_polygon_geojson(candidate, [candidate_poly])
    _write_polygon_geojson(reference, [reference_poly])

    aoi = BoundingBox(min_lon=85.25, min_lat=27.65, max_lon=85.40, max_lat=27.80)
    result = evaluate_flood_extent(
        candidate, reference,
        aoi_bbox=aoi,
        resolution_m=30.0,
    )
    assert result["evaluated"] is True
    # Intersection is half of each square, so:
    #   IoU = intersection / union = 0.5A / 1.5A = 1/3 ≈ 0.33
    #   Precision = intersection / candidate = 0.5A / A = 0.5
    #   Recall = intersection / reference = 0.5A / A = 0.5
    assert 0.25 < result["iou"] < 0.45, f"IoU out of range: {result['iou']}"
    assert 0.40 < result["precision"] < 0.60
    assert 0.40 < result["recall"] < 0.60
    assert 0.40 < result["f1"] < 0.60


# ---------------------------------------------------------------------------
# Missing reference file
# ---------------------------------------------------------------------------

def test_missing_reference_file(tmp_path: Path):
    """If the reference file is missing, the result is marked not evaluated."""
    candidate_poly = [
        (85.30, 27.70),
        (85.31, 27.70),
        (85.31, 27.71),
        (85.30, 27.71),
        (85.30, 27.70),
    ]
    candidate = tmp_path / "candidate.geojson"
    _write_polygon_geojson(candidate, [candidate_poly])

    aoi = BoundingBox(min_lon=85.25, min_lat=27.65, max_lon=85.35, max_lat=27.75)
    result = evaluate_flood_extent(
        candidate, tmp_path / "does_not_exist.geojson",
        aoi_bbox=aoi,
        resolution_m=30.0,
    )
    assert result["evaluated"] is False
    assert "not found" in result["reason"].lower()