"""Tests for the validation module.

These tests exercise:

- Geodesic area calculation for bounding boxes.
- Parsing of user-supplied bbox strings.
- Rejection of invalid bounding boxes.
- Projected CRS selection (UTM zone chosen from the bbox centroid).
- Event validation against the AOI-size limit.

All tests use synthetic inputs. No network, no credentials, no data
downloads.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pyproj import CRS

from georescue.models import BoundingBox, EventDefinition
from georescue.validation import (
    ValidationError,
    bbox_area_sqkm,
    bbox_to_wkt,
    bounds_intersect,
    choose_projected_crs,
    format_bbox,
    parse_bbox_string,
    validate_event,
)


# ---------------------------------------------------------------------------
# bbox_area_sqkm
# ---------------------------------------------------------------------------

def test_bbox_area_plausible_for_nepal(trishuli_bbox):
    """The Trishuli AOI is roughly 0.45° × 0.45° at 28°N.

    That is approximately 45 km × 50 km ≈ 2200–2600 km². We allow a
    generous window to accommodate the exact projection used.
    """
    area = bbox_area_sqkm(trishuli_bbox)
    assert 2000.0 < area < 3500.0, f"Expected ~2500 km², got {area}"


def test_bbox_area_matches_manual_estimate(koshi_bbox):
    """A 0.30° × 0.30° box at 26.5°N is roughly 900 km²."""
    area = bbox_area_sqkm(koshi_bbox)
    assert 700.0 < area < 1200.0, f"Expected ~900 km², got {area}"


def test_bbox_area_zero_for_degenerate_box():
    """A zero-width bbox should be rejected by the model."""
    with pytest.raises(Exception):
        BoundingBox(min_lon=10.0, min_lat=10.0, max_lon=10.0, max_lat=20.0)


# ---------------------------------------------------------------------------
# parse_bbox_string
# ---------------------------------------------------------------------------

def test_parse_bbox_string_comma_separated():
    bbox = parse_bbox_string("85.10, 27.60, 85.55, 28.05")
    assert bbox.min_lon == pytest.approx(85.10)
    assert bbox.min_lat == pytest.approx(27.60)
    assert bbox.max_lon == pytest.approx(85.55)
    assert bbox.max_lat == pytest.approx(28.05)


def test_parse_bbox_string_whitespace_separated():
    bbox = parse_bbox_string("85.10 27.60 85.55 28.05")
    assert bbox.min_lon == pytest.approx(85.10)
    assert bbox.max_lat == pytest.approx(28.05)


def test_parse_bbox_string_rejects_wrong_count():
    with pytest.raises(ValidationError, match="Expected 4 numbers"):
        parse_bbox_string("85.10, 27.60, 85.55")


def test_parse_bbox_string_rejects_non_numeric():
    with pytest.raises(ValidationError):
        parse_bbox_string("a, b, c, d")


def test_parse_bbox_string_rejects_inverted_box():
    with pytest.raises(ValidationError):
        parse_bbox_string("85.55, 27.60, 85.10, 28.05")


def test_parse_bbox_string_rejects_empty():
    with pytest.raises(ValidationError):
        parse_bbox_string("")


# ---------------------------------------------------------------------------
# BoundingBox model constraints
# ---------------------------------------------------------------------------

def test_bbox_rejects_inverted_lon():
    with pytest.raises(Exception):
        BoundingBox(min_lon=20.0, min_lat=10.0, max_lon=10.0, max_lat=20.0)


def test_bbox_rejects_inverted_lat():
    with pytest.raises(Exception):
        BoundingBox(min_lon=10.0, min_lat=20.0, max_lon=20.0, max_lat=10.0)


def test_bbox_rejects_out_of_range_lon():
    with pytest.raises(Exception):
        BoundingBox(min_lon=-200.0, min_lat=0.0, max_lon=10.0, max_lat=10.0)


# ---------------------------------------------------------------------------
# choose_projected_crs
# ---------------------------------------------------------------------------

def test_choose_projected_crs_trishuli(trishuli_bbox):
    """Nepal (85°E, 28°N) is in UTM zone 45 North."""
    crs = choose_projected_crs(trishuli_bbox)
    assert crs.to_epsg() == 32645


def test_choose_projected_crs_koshi(koshi_bbox):
    """Nepal (87°E, 26°N) is also in UTM zone 45 North."""
    crs = choose_projected_crs(koshi_bbox)
    assert crs.to_epsg() == 32645


def test_choose_projected_crs_southern_hemisphere():
    """A bbox at 30°S should be in a southern UTM zone (EPSG 327xx)."""
    bbox = BoundingBox(min_lon=30.0, min_lat=-30.0, max_lon=31.0, max_lat=-29.0)
    crs = choose_projected_crs(bbox)
    epsg = crs.to_epsg()
    assert epsg is not None
    assert 32700 <= epsg < 32800, f"Expected southern UTM, got EPSG {epsg}"


# ---------------------------------------------------------------------------
# validate_event
# ---------------------------------------------------------------------------

def test_validate_event_accepts_small_aoi(trishuli_bbox):
    event = EventDefinition(
        aoi=trishuli_bbox,
        event_date=date(2026, 8, 26),
    )
    # Should not raise
    validate_event(event)


def test_validate_event_rejects_oversized_aoi():
    """An AOI larger than the configured maximum must be rejected."""
    huge = BoundingBox(min_lon=-10.0, min_lat=-10.0, max_lon=30.0, max_lat=30.0)
    event = EventDefinition(aoi=huge, event_date=date(2026, 8, 26))
    with pytest.raises(ValidationError, match="exceeds the configured limit"):
        validate_event(event)


def test_validate_event_rejects_zero_pre_days(trishuli_bbox):
    """Pydantic itself rejects pre_days=0, so we catch that earlier."""
    with pytest.raises(Exception):
        EventDefinition(
            aoi=trishuli_bbox,
            event_date=date(2026, 8, 26),
            pre_days=0,
        )


def test_event_definition_rejects_future_event_date(trishuli_bbox):
    """Events in the future are not analysable."""
    future = date.today() + timedelta(days=30)
    with pytest.raises(Exception):
        EventDefinition(aoi=trishuli_bbox, event_date=future)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def test_format_bbox_readable(trishuli_bbox):
    text = format_bbox(trishuli_bbox)
    assert "85.1000" in text
    assert "27.6000" in text
    assert "85.5500" in text
    assert "28.0500" in text


def test_bbox_to_wkt_roundtrip(trishuli_bbox):
    wkt = bbox_to_wkt(trishuli_bbox)
    assert wkt.startswith("POLYGON")
    assert "85.1" in wkt


def test_bounds_intersect_overlapping():
    assert bounds_intersect((0, 0, 10, 10), (5, 5, 15, 15))


def test_bounds_intersect_disjoint():
    assert not bounds_intersect((0, 0, 10, 10), (20, 20, 30, 30))


def test_bounds_intersect_touching():
    """Touching edges count as intersecting."""
    assert bounds_intersect((0, 0, 10, 10), (10, 0, 20, 10))