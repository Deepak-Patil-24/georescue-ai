"""Tests for the reporting module.

Exercises:

- Building an :class:`AnalysisEvidence` object from synthetic summary
  files on disk.
- Generating the deterministic situation report.
- Validating that the LLM response validator rejects invented numbers
  and accepts real numbers.

All tests use synthetic files written to a pytest temporary directory.
No network, no LLM calls, no external services.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from georescue.models import BoundingBox, EventDefinition
from georescue.reporting.evidence import (
    ATTRIBUTION_DEM,
    ATTRIBUTION_OSM,
    ATTRIBUTION_SENTINEL,
    build_evidence,
)
from georescue.reporting.llm_copilot import validate_llm_response
from georescue.reporting.situation_report import (
    build_deterministic_report,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers to write synthetic summary files
# ---------------------------------------------------------------------------

def _write_infrastructure_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "flood_extent_path": "candidate_flood_extent.geojson",
        "osm_snapshot": "2026-07-27",
        "event_date": "2026-08-26",
        "buffer_m": 0.0,
        "cell_size_deg": 0.1,
        "buildings": {
            "buildings_intersecting_count": 42,
            "buildings_exposed_area_sqm": 38568.78,
            "buildings_exposed_area_hectares": 3.8569,
            "buffer_m": 0.0,
            "warnings": [],
        },
        "roads": {
            "roads_intersecting_count": 125,
            "roads_affected_length_km": 6.4,
            "buffer_m": 0.0,
            "warnings": [],
        },
        "bridges": {
            "bridges_intersecting_count": 3,
            "buffer_m": 0.0,
            "warnings": [],
        },
        "warnings": [],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_connectivity_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aoi_bbox": [85.1, 27.6, 85.55, 28.05],
        "osm_snapshot": "2026-07-27",
        "event_date": "2026-08-26",
        "disruption_fraction": 0.2,
        "node_merge_m": 50.0,
        "graph": {"nodes": 100, "edges": 120},
        "counts": {
            "settlements_in_graph": 25,
            "destinations_in_graph": 3,
            "baseline_connected": 10,
            "scenario_connected": 9,
            "newly_disconnected": 1,
            "disrupted_edges": 4,
        },
        "newly_disconnected_settlements": [
            {
                "settlement_id": "1234567890",
                "baseline_distance_m": 1500.0,
                "baseline_destination": "hospital-98765",
            }
        ],
        "warnings": [],
        "note": "Modelled access disruption scenario.",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_flood_extent_geojson(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "flood-0000",
                "properties": {
                    "id": "flood-0000",
                    "area_sqkm": 0.5,
                    "area_hectares": 50.0,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [85.30, 27.70],
                        [85.31, 27.70],
                        [85.31, 27.71],
                        [85.30, 27.71],
                        [85.30, 27.70],
                    ]],
                },
            },
            {
                "type": "Feature",
                "id": "flood-0001",
                "properties": {
                    "id": "flood-0001",
                    "area_sqkm": 0.25,
                    "area_hectares": 25.0,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [85.32, 27.72],
                        [85.33, 27.72],
                        [85.33, 27.73],
                        [85.32, 27.73],
                        [85.32, 27.72],
                    ]],
                },
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Evidence building
# ---------------------------------------------------------------------------

def test_build_evidence_from_synthetic_summaries(tmp_path: Path):
    """Build evidence from synthetic files and check that all metrics
    have the expected values."""
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"

    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
        aoi_name="Test AOI",
    )

    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )

    # 5 infra metrics + 2 flood + 4 connectivity + 1 disrupted edges = 12
    assert len(evidence.metrics) == 12

    # Look up specific metrics
    by_id = {m.metric_id: m for m in evidence.metrics}
    assert by_id["flood.area_sqkm"].value == pytest.approx(0.75, rel=0.01)
    assert by_id["flood.polygon_count"].value == 2
    assert by_id["infra.buildings_intersecting"].value == 42
    assert by_id["infra.roads_affected_km"].value == pytest.approx(6.4, rel=0.01)
    assert by_id["conn.newly_disconnected"].value == 1

    # Attributions present
    assert ATTRIBUTION_SENTINEL in evidence.attributions
    assert ATTRIBUTION_DEM in evidence.attributions
    assert ATTRIBUTION_OSM in evidence.attributions


def test_build_evidence_missing_files_reports_warnings(tmp_path: Path):
    """When files are missing, evidence records warnings rather than
    silently substituting empty values."""
    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=tmp_path / "missing.geojson",
        infrastructure_summary=tmp_path / "missing.json",
        connectivity_summary=tmp_path / "missing.json",
    )
    assert evidence.flood_extent is None
    assert evidence.infrastructure is None
    assert evidence.connectivity is None
    assert len(evidence.warnings) >= 3


# ---------------------------------------------------------------------------
# Deterministic situation report
# ---------------------------------------------------------------------------

def test_deterministic_report_contains_real_numbers(tmp_path: Path):
    """The report should quote the same numbers as the evidence object."""
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"
    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
        aoi_name="Test AOI",
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )
    report = build_deterministic_report(evidence)

    assert report.generated_by == "deterministic"
    # The report should mention the building count from the evidence.
    assert "42" in report.body_markdown
    # The report should mention the destroyed area (0.75 km²).
    assert "0.75" in report.body_markdown or "0.8" in report.body_markdown


def test_deterministic_report_written_to_disk(tmp_path: Path):
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"
    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )
    report = build_deterministic_report(evidence)
    out = tmp_path / "report.md"
    write_report(report, out)
    assert out.is_file()
    assert out.stat().st_size > 500


# ---------------------------------------------------------------------------
# LLM validator
# ---------------------------------------------------------------------------

def test_validator_accepts_real_number(tmp_path: Path):
    """A number that exists in the evidence should pass validation."""
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"
    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )

    # 42 is the building count and appears in the evidence.
    ok, problems = validate_llm_response(
        "There are 42 buildings inside the possibly flooded areas.",
        evidence,
    )
    assert ok, f"Expected validation to pass, but problems: {problems}"
    assert problems == []


def test_validator_rejects_invented_number(tmp_path: Path):
    """A number that does not exist in the evidence should be rejected."""
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"
    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )

    # 12345 does not appear anywhere in the evidence.
    ok, problems = validate_llm_response(
        "There were 12345 buildings destroyed by the flood.",
        evidence,
    )
    assert not ok, "Validator should reject an invented number"
    assert any("12345" in p for p in problems)


def test_validator_allows_unit_conversions(tmp_path: Path):
    """A distance reported in metres when the evidence has kilometres
    should be accepted."""
    infra = tmp_path / "infrastructure_summary.json"
    conn = tmp_path / "connectivity_summary.json"
    flood = tmp_path / "candidate_flood_extent.geojson"
    _write_infrastructure_summary(infra)
    _write_connectivity_summary(conn)
    _write_flood_extent_geojson(flood)

    event = EventDefinition(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
    )
    evidence = build_evidence(
        event,
        flood_extent_geojson=flood,
        infrastructure_summary=infra,
        connectivity_summary=conn,
    )

    # Evidence has 0.75 km². Response writes 750,000 m².
    # The validator should recognise the unit conversion.
    ok, problems = validate_llm_response(
        "The total area is 750000 square metres.",
        evidence,
    )
    assert ok, f"Validator should accept unit conversions, but: {problems}"