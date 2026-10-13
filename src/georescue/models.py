"""Pydantic data models shared across GeoRescue AI.

These models define the contract between pipeline stages. They are
intentionally explicit so that:

- Every stage validates its inputs and outputs.
- The evidence object sent to the LLM is well-typed and traceable.
- Missing values are represented as ``None`` (or ``"unavailable"``), never
  silently coerced to zero.
- ``extra="forbid"`` prevents accidental schema drift.

Terminology discipline
----------------------
- ``candidate_flood_extent`` — output of the change-detection algorithm.
- ``potentially_exposed`` — infrastructure intersecting or near that extent.
- ``modelled_disruption`` — network scenario, not observed impassability.

We never use the words "confirmed damage" or "confirmed destruction" for
algorithmic outputs. Those terms are reserved for independently verified
ground truth, which this system does not produce.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Area of interest and event definition
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Geographic bounding box in WGS84 (EPSG:4326).

    Order: ``min_lon, min_lat, max_lon, max_lat``. Matches GeoJSON
    convention (west, south, east, north).
    """

    model_config = ConfigDict(extra="forbid")

    min_lon: float = Field(..., ge=-180.0, le=180.0)
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)

    @field_validator("max_lon")
    @classmethod
    def _lon_order(cls, v: float, info) -> float:
        min_lon = info.data.get("min_lon")
        if min_lon is not None and v <= min_lon:
            raise ValueError(
                f"max_lon ({v}) must be strictly greater than min_lon ({min_lon})"
            )
        return v

    @field_validator("max_lat")
    @classmethod
    def _lat_order(cls, v: float, info) -> float:
        min_lat = info.data.get("min_lat")
        if min_lat is not None and v <= min_lat:
            raise ValueError(
                f"max_lat ({v}) must be strictly greater than min_lat ({min_lat})"
            )
        return v

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(min_lon, min_lat, max_lon, max_lat)``."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def to_geojson_polygon(self) -> dict:
        """Return a GeoJSON Polygon (ring closed, counter-clockwise)."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat],
            ]],
        }


class EventDefinition(BaseModel):
    """User-supplied analysis request."""

    model_config = ConfigDict(extra="forbid")

    aoi: BoundingBox
    event_date: date
    pre_days: int = Field(default=12, ge=1, le=90)
    post_days: int = Field(default=12, ge=1, le=90)
    max_cloud_cover: float = Field(default=30.0, ge=0.0, le=100.0)
    prefer_same_relative_orbit: bool = True
    aoi_name: Optional[str] = None

    @field_validator("event_date")
    @classmethod
    def _not_in_future(cls, v: date) -> date:
        if v > date.today():
            # Allow up to 1 day in the future to tolerate time-zone quirks.
            from datetime import timedelta
            if v > date.today() + timedelta(days=1):
                raise ValueError(
                    f"event_date {v} is in the future; cannot analyse an "
                    f"event that has not yet occurred."
                )
        return v


# ---------------------------------------------------------------------------
# Satellite scene metadata
# ---------------------------------------------------------------------------

class SceneMetadata(BaseModel):
    """Metadata for a single satellite scene candidate or selection."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    platform: Literal["sentinel-1", "sentinel-2"]
    acquisition_datetime: datetime
    relative_orbit: Optional[int] = None
    orbit_direction: Optional[str] = None  # "ASCENDING" | "DESCENDING"
    polarisation: Optional[str] = None     # e.g. "VV", "VH", "VV+VH"
    cloud_cover: Optional[float] = None    # Sentinel-2 only, percent
    product_type: Optional[str] = None     # e.g. "GRD", "SLC", "L2A"
    processing_level: Optional[str] = None
    tile_or_granule_id: Optional[str] = None
    download_url: Optional[str] = None
    size_bytes: Optional[int] = None
    footprint_geojson: Optional[dict] = None
    source: str = "CDSE"


class ScenePair(BaseModel):
    """A selected pre-event / post-event scene pair, with selection notes."""

    model_config = ConfigDict(extra="forbid")

    pre: SceneMetadata
    post: SceneMetadata
    selection_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Flood extent
# ---------------------------------------------------------------------------

class FloodExtentResult(BaseModel):
    """Result of the flood-detection stage.

    ``method`` must be a short human-readable description of the algorithm
    actually used. ``parameters`` records thresholds and settings for
    reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    method: str
    parameters: dict
    area_sqkm: float = Field(..., ge=0.0)
    area_hectares: float = Field(..., ge=0.0)
    pixel_count: int = Field(..., ge=0)
    pixel_size_m: Optional[float] = None
    crs: str
    mask_path: Optional[str] = None
    polygon_path: Optional[str] = None
    raster_path: Optional[str] = None
    pre_scene: Optional[SceneMetadata] = None
    post_scene: Optional[SceneMetadata] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Infrastructure exposure
# ---------------------------------------------------------------------------

class InfrastructureExposure(BaseModel):
    """Counts and identifiers of potentially exposed infrastructure.

    Every count here means *intersecting the candidate flood extent or its
    buffer*, not *confirmed damaged*.
    """

    model_config = ConfigDict(extra="forbid")

    buildings_intersecting_count: int = Field(..., ge=0)
    buildings_exposed_area_sqm: float = Field(..., ge=0.0)
    roads_intersecting_count: int = Field(..., ge=0)
    roads_affected_length_km: float = Field(..., ge=0.0)
    bridges_intersecting_count: int = Field(..., ge=0)
    buildings_layer_path: Optional[str] = None
    roads_layer_path: Optional[str] = None
    bridges_layer_path: Optional[str] = None
    sample_building_osm_ids: list[str] = Field(default_factory=list)
    sample_road_osm_ids: list[str] = Field(default_factory=list)
    sample_bridge_osm_ids: list[str] = Field(default_factory=list)
    buffer_m: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Settlement connectivity
# ---------------------------------------------------------------------------

class DisconnectedSettlement(BaseModel):
    """A settlement whose modelled connectivity changed after disruption."""

    model_config = ConfigDict(extra="forbid")

    settlement_name: str
    settlement_osm_id: Optional[str] = None
    baseline_destination: Optional[str] = None
    baseline_distance_km: Optional[float] = None
    scenario_connected: bool
    reason: str


class ConnectivityResult(BaseModel):
    """Result of the access-disruption analysis.

    The output is explicitly framed as a *modelled scenario*.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_name: str
    disruption_rule: str
    baseline_settlement_count: int = Field(..., ge=0)
    baseline_connected_count: int = Field(..., ge=0)
    scenario_connected_count: int = Field(..., ge=0)
    newly_disconnected: list[DisconnectedSettlement] = Field(default_factory=list)
    disrupted_road_osm_ids: list[str] = Field(default_factory=list)
    destination_type: str = "hospital_or_town"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence and reporting
# ---------------------------------------------------------------------------

class EvidenceMetric(BaseModel):
    """One traceable numerical fact produced by the application.

    Every number in a situation report must reference an ``EvidenceMetric``.
    """

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    label: str
    value: float | int | str | None
    unit: Optional[str] = None
    source_layer: str
    method: str
    kind: Literal["observation", "estimate", "assumption", "recommendation"]


class AnalysisEvidence(BaseModel):
    """Structured evidence object passed to the situation-report generator."""

    model_config = ConfigDict(extra="forbid")

    event: EventDefinition
    generated_at: datetime
    flood_extent: Optional[FloodExtentResult] = None
    infrastructure: Optional[InfrastructureExposure] = None
    connectivity: Optional[ConnectivityResult] = None
    metrics: list[EvidenceMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attributions: list[str] = Field(default_factory=list)


class SituationReport(BaseModel):
    """A generated one-page situation report."""

    model_config = ConfigDict(extra="forbid")

    title: str
    generated_at: datetime
    generated_by: Literal["llm", "deterministic"]
    body_markdown: str
    evidence: AnalysisEvidence
    notes: list[str] = Field(default_factory=list)