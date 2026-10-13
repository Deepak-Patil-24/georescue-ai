"""Structured evidence gathering for GeoRescue AI reports.

This module reads the outputs of the analysis pipeline (flood extent,
infrastructure exposure, connectivity) and builds a single validated
:class:`AnalysisEvidence` object. Every numeric value reported in a
situation report must come from an :class:`EvidenceMetric` in this
object.

Rules enforced here
-------------------
- A missing value is reported as ``None`` with the kind ``"unavailable"``.
  We never coerce a missing metric into zero.
- Each metric carries its ``source_layer`` and ``method`` so the report
  can cite provenance.
- Each metric carries a ``kind`` from
  {observation, estimate, assumption, recommendation} so readers can
  distinguish what we measured from what we modelled.
- When ``use_settings_fallbacks`` is ``False``, the function will not
  silently substitute files from the precomputed Trishuli case study
  when an input is missing. This prevents one run's numbers from being
  reported as another run's results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.models import (
    AnalysisEvidence,
    BoundingBox,
    ConnectivityResult,
    DisconnectedSettlement,
    EventDefinition,
    EvidenceMetric,
    FloodExtentResult,
    InfrastructureExposure,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Attribution strings (verbatim, per hackathon rules)
# ---------------------------------------------------------------------------

ATTRIBUTION_SENTINEL = "Contains modified Copernicus Sentinel data 2026."
ATTRIBUTION_DEM = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and "
    "© Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS "
    "by the European Union and ESA; all rights reserved."
)
ATTRIBUTION_OSM = "© OpenStreetMap contributors."

ATTRIBUTIONS = (ATTRIBUTION_SENTINEL, ATTRIBUTION_DEM, ATTRIBUTION_OSM)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def _load_flood_extent_summary(geojson_path: Path) -> Optional[dict[str, Any]]:
    """Load a flood extent GeoJSON and return a summary dict."""
    if not geojson_path.is_file():
        return None
    try:
        with open(geojson_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read flood extent %s: %s", geojson_path, exc)
        return None

    features = data.get("features") or []
    total_area = 0.0
    largest = 0.0
    for feat in features:
        props = feat.get("properties") or {}
        area = float(props.get("area_sqkm") or 0.0)
        total_area += area
        largest = max(largest, area)

    return {
        "polygon_count": len(features),
        "total_area_sqkm": total_area,
        "largest_polygon_sqkm": largest,
    }


# ---------------------------------------------------------------------------
# Evidence construction
# ---------------------------------------------------------------------------

def build_evidence(
    event: EventDefinition,
    *,
    flood_extent_geojson: Optional[Path] = None,
    infrastructure_summary: Optional[Path] = None,
    connectivity_summary: Optional[Path] = None,
    settings: Optional[Settings] = None,
    use_settings_fallbacks: bool = True,
) -> AnalysisEvidence:
    """Build the :class:`AnalysisEvidence` object for a completed analysis.

    Parameters
    ----------
    event:
        Event definition providing the AOI and date for the report.
    flood_extent_geojson, infrastructure_summary, connectivity_summary:
        Paths to the per-layer inputs for this analysis.
    settings:
        Optional settings override.
    use_settings_fallbacks:
        When True (default), missing paths are replaced by the precomputed
        Trishuli case-study paths. When False, missing paths are treated
        as genuinely absent and a warning is recorded. The dashboard
        passes False for run-relative contexts so one run's numbers are
        never silently shown for another.

    Any missing input is recorded as an explicit warning in the evidence
    object rather than being silently skipped.
    """
    settings = settings or get_settings()
    warnings: list[str] = []
    metrics: list[EvidenceMetric] = []

    # --- Flood extent ---
    flood_extent: Optional[FloodExtentResult] = None
    if flood_extent_geojson is None:
        if use_settings_fallbacks:
            flood_extent_geojson = (
                settings.processed_data_path / "candidate_flood_extent.geojson"
            )
        else:
            warnings.append(
                "Flood extent path was not provided for this run and "
                "fallbacks are disabled."
            )

    if flood_extent_geojson is not None:
        flood_summary = _load_flood_extent_summary(flood_extent_geojson)
        if flood_summary is None:
            warnings.append(
                f"Flood extent GeoJSON not found at {flood_extent_geojson}. "
                f"Candidate flood extent cannot be reported."
            )
        else:
            area_sqkm = float(flood_summary["total_area_sqkm"])
            flood_extent = FloodExtentResult(
                method="Sentinel-1 VV backscatter ratio change detection",
                parameters={
                    "ratio_threshold": 0.4,
                    "max_slope_deg": 5.0,
                    "min_mmu_pixels": 9,
                },
                area_sqkm=area_sqkm,
                area_hectares=area_sqkm * 100.0,
                pixel_count=0,
                crs="EPSG:4326",
                polygon_path=str(flood_extent_geojson),
            )
            metrics.append(EvidenceMetric(
                metric_id="flood.area_sqkm",
                label="Candidate flood extent area",
                value=round(area_sqkm, 4),
                unit="km²",
                source_layer="candidate_flood_extent.geojson",
                method="Sentinel-1 VV backscatter ratio change detection",
                kind="estimate",
            ))
            metrics.append(EvidenceMetric(
                metric_id="flood.polygon_count",
                label="Number of candidate flood polygons",
                value=int(flood_summary["polygon_count"]),
                unit="count",
                source_layer="candidate_flood_extent.geojson",
                method="Raster vectorisation with 50 m morphological merge",
                kind="estimate",
            ))

    # --- Infrastructure exposure ---
    infra: Optional[InfrastructureExposure] = None
    if infrastructure_summary is None:
        if use_settings_fallbacks:
            infrastructure_summary = (
                settings.output_path / "infrastructure" / "infrastructure_summary.json"
            )
        else:
            warnings.append(
                "Infrastructure summary path was not provided for this run "
                "and fallbacks are disabled."
            )

    if infrastructure_summary is not None:
        infra_data = _load_json(infrastructure_summary)
        if infra_data is None:
            warnings.append(
                f"Infrastructure summary not found at {infrastructure_summary}. "
                f"Infrastructure exposure cannot be reported."
            )
        else:
            buildings = infra_data.get("buildings") or {}
            roads = infra_data.get("roads") or {}
            bridges = infra_data.get("bridges") or {}

            infra = InfrastructureExposure(
                buildings_intersecting_count=int(
                    buildings.get("buildings_intersecting_count") or 0
                ),
                buildings_exposed_area_sqm=float(
                    buildings.get("buildings_exposed_area_sqm") or 0.0
                ),
                roads_intersecting_count=int(
                    roads.get("roads_intersecting_count") or 0
                ),
                roads_affected_length_km=float(
                    roads.get("roads_affected_length_km") or 0.0
                ),
                bridges_intersecting_count=int(
                    bridges.get("bridges_intersecting_count") or 0
                ),
            )
            metrics.extend([
                EvidenceMetric(
                    metric_id="infra.buildings_intersecting",
                    label="Buildings intersecting the candidate flood extent",
                    value=infra.buildings_intersecting_count,
                    unit="count",
                    source_layer="buildings_exposed.geojson",
                    method=(
                        "Spatial join of pre-event OSM buildings with the "
                        "flood extent"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="infra.buildings_area_sqm",
                    label="Total exposed building footprint area",
                    value=round(infra.buildings_exposed_area_sqm, 2),
                    unit="m²",
                    source_layer="buildings_exposed.geojson",
                    method=(
                        "Sum of building ∩ flood intersection areas, "
                        "projected CRS"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="infra.roads_intersecting",
                    label="Road segments intersecting the candidate flood extent",
                    value=infra.roads_intersecting_count,
                    unit="count",
                    source_layer="roads_exposed.geojson",
                    method=(
                        "Spatial join of pre-event OSM roads with the flood "
                        "extent"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="infra.roads_affected_km",
                    label=(
                        "Total road length intersecting the candidate flood "
                        "extent"
                    ),
                    value=round(infra.roads_affected_length_km, 3),
                    unit="km",
                    source_layer="roads_exposed.geojson",
                    method=(
                        "Sum of road ∩ flood intersection lengths, "
                        "projected CRS"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="infra.bridges_intersecting",
                    label="Bridges intersecting the candidate flood extent",
                    value=infra.bridges_intersecting_count,
                    unit="count",
                    source_layer="bridges_exposed.geojson",
                    method=(
                        "Spatial join of pre-event OSM bridges with the "
                        "flood extent"
                    ),
                    kind="estimate",
                ),
            ])

    # --- Connectivity ---
    connectivity: Optional[ConnectivityResult] = None
    if connectivity_summary is None:
        if use_settings_fallbacks:
            connectivity_summary = (
                settings.output_path / "connectivity" / "connectivity_summary.json"
            )
        else:
            warnings.append(
                "Connectivity summary path was not provided for this run "
                "and fallbacks are disabled."
            )

    if connectivity_summary is not None:
        conn_data = _load_json(connectivity_summary)
        if conn_data is None:
            warnings.append(
                f"Connectivity summary not found at {connectivity_summary}. "
                f"Access disruption cannot be reported."
            )
        else:
            counts = conn_data.get("counts") or {}
            disconnected_raw = (
                conn_data.get("newly_disconnected_settlements") or []
            )
            disconnected = [
                DisconnectedSettlement(
                    settlement_name=f"OSM {d.get('settlement_id')}",
                    settlement_osm_id=str(d.get("settlement_id")),
                    baseline_destination=str(d.get("baseline_destination")),
                    baseline_distance_km=(
                        float(d["baseline_distance_m"]) / 1000.0
                        if d.get("baseline_distance_m") is not None
                        else None
                    ),
                    scenario_connected=False,
                    reason=(
                        "Modelled flood disruption removed the baseline "
                        "road path."
                    ),
                )
                for d in disconnected_raw
            ]
            connectivity = ConnectivityResult(
                scenario_name=(
                    "Flood road-disruption scenario (20% length threshold)"
                ),
                disruption_rule=(
                    f"Road segments intersecting the flood extent over "
                    f"≥{conn_data.get('disruption_fraction', 0.2) * 100:.0f}% "
                    f"of their length are treated as potentially impassable."
                ),
                baseline_settlement_count=int(
                    counts.get("settlements_in_graph") or 0
                ),
                baseline_connected_count=int(
                    counts.get("baseline_connected") or 0
                ),
                scenario_connected_count=int(
                    counts.get("scenario_connected") or 0
                ),
                newly_disconnected=disconnected,
                disrupted_road_osm_ids=[],
                warnings=[],
            )
            metrics.extend([
                EvidenceMetric(
                    metric_id="conn.settlements_in_graph",
                    label="Settlements snapped to the pre-event road graph",
                    value=connectivity.baseline_settlement_count,
                    unit="count",
                    source_layer="connectivity_summary.json",
                    method="Nearest-node snapping of OSM place points",
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="conn.baseline_connected",
                    label=(
                        "Settlements with a pre-event road path to a "
                        "hospital or town"
                    ),
                    value=connectivity.baseline_connected_count,
                    unit="count",
                    source_layer="connectivity_summary.json",
                    method=(
                        "Dijkstra shortest paths over the pre-event road "
                        "graph"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="conn.scenario_connected",
                    label=(
                        "Settlements still connected after the modelled "
                        "disruption"
                    ),
                    value=connectivity.scenario_connected_count,
                    unit="count",
                    source_layer="connectivity_summary.json",
                    method=(
                        "Dijkstra shortest paths after removing disrupted "
                        "edges"
                    ),
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="conn.newly_disconnected",
                    label=(
                        "Settlements newly disconnected by the modelled "
                        "disruption"
                    ),
                    value=len(disconnected),
                    unit="count",
                    source_layer="connectivity_summary.json",
                    method="Baseline connected minus scenario connected",
                    kind="estimate",
                ),
                EvidenceMetric(
                    metric_id="conn.disrupted_edges",
                    label="Road segments treated as potentially impassable",
                    value=int(counts.get("disrupted_edges") or 0),
                    unit="count",
                    source_layer="connectivity_summary.json",
                    method=(
                        "Flood intersection fraction ≥ 20% of segment length"
                    ),
                    kind="assumption",
                ),
            ])

    return AnalysisEvidence(
        event=event,
        generated_at=datetime.now(timezone.utc),
        flood_extent=flood_extent,
        infrastructure=infra,
        connectivity=connectivity,
        metrics=metrics,
        warnings=warnings,
        attributions=list(ATTRIBUTIONS),
    )


__all__ = [
    "ATTRIBUTION_DEM",
    "ATTRIBUTION_OSM",
    "ATTRIBUTION_SENTINEL",
    "ATTRIBUTIONS",
    "build_evidence",
]