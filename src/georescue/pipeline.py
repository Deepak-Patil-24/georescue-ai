"""End-to-end pipeline orchestrator for GeoRescue AI.

This module is the single entry point for running a fresh flood analysis
on a user-supplied area and event date. It is designed for two callers:

1. The Streamlit dashboard (``app.py``), which lets a rescuer pick an
   area and dates and press "Run flood assessment" without editing any
   code.
2. The command-line scripts in ``scripts/``, which wrap this module with
   argument parsing for developer use.

Design principles
-----------------
- **Search first, download later.** The caller must be able to see
  available satellite scenes and estimated download sizes before any
  bytes are pulled from CDSE.
- **One run, one folder.** Every invocation creates a unique folder
  under ``outputs/runs/<run_id>/``. Nothing is written to a shared
  location that could collide with another run.
- **No stale outputs.** The run folder is created fresh. If a run with
  the same slug already exists, a numeric suffix is appended.
- **Honest failures.** If a stage fails, the pipeline raises a specific
  exception and does not attempt to continue with missing inputs.
- **No fabrication.** The pipeline never writes a placeholder flood
  extent, a synthetic raster, or an invented metric.

The module does not contain any AOI-specific constants. Every AOI,
date, threshold, and output path is a parameter.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from georescue.analysis.connectivity import analyze_connectivity
from georescue.analysis.infrastructure import analyze_infrastructure_exposure
from georescue.config import Settings, get_settings
from georescue.data_access import DataAccessError, NoResultsError
from georescue.data_access.copernicus import CopernicusClient
from georescue.data_access.dem import download_dem_for_bbox, mosaic_dem_tiles
from georescue.data_access.sentinel1 import (
    search_sentinel1,
    select_sentinel1_pair,
)
from georescue.logging_config import get_logger
from georescue.models import BoundingBox, EventDefinition, SceneMetadata, ScenePair
from georescue.processing.flood_detection import (
    FloodDetectionResult,
    detect_flood_change,
    save_flood_mask,
    save_ratio_raster,
)
from georescue.processing.preprocessing import preprocess_sentinel1_product
from georescue.processing.raster_utils import (
    align_raster_to_reference,
    bbox_in_crs,
    compute_slope_degrees,
    utm_crs_for_bbox,
    write_raster,
)
from georescue.processing.vectorization import (
    reproject_geojson_to_wgs84,
    vectorize_flood_mask,
    write_geojson,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Default search window in days on either side of the event date. The
# Sentinel-1 revisit is ~12 days per orbit, so 15 days covers at least
# one usable pre-event and one usable post-event acquisition on most
# tracks. Callers may override.
DEFAULT_WINDOW_DAYS = 15

# Estimated download size warning threshold. If the sum of the two
# selected products exceeds this, the caller should ask for confirmation.
DEFAULT_SIZE_WARNING_MB = 800.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete a stage.

    The message is user-facing: it should explain what failed and what
    the caller can do about it.
    """


class NoSuitableImageryError(PipelineError):
    """Raised when no compatible Sentinel-1 pair exists for the request."""


class DownloadConfirmationRequired(PipelineError):
    """Raised when the caller has not confirmed a large download.

    The caller is expected to show the estimated size to the user and
    re-invoke ``run_full_analysis`` with ``confirm_download=True``.
    """


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SceneAvailability:
    """Structured summary of satellite scene availability for an AOI.

    Returned by :func:`search_only`. Contains enough information for a
    user interface to present availability and estimated size without
    downloading anything.
    """

    aoi: BoundingBox
    event_date: date
    pre_scenes: list[SceneMetadata]
    post_scenes: list[SceneMetadata]
    selected_pair: Optional[ScenePair]
    estimated_download_mb: float
    selection_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_pair(self) -> bool:
        return self.selected_pair is not None

    @property
    def total_scenes(self) -> int:
        return len(self.pre_scenes) + len(self.post_scenes)


@dataclass
class PipelineResult:
    """Paths and structured metrics from a completed pipeline run."""

    run_id: str
    run_dir: Path
    event: EventDefinition
    scene_pair: ScenePair
    flood_extent_geojson: Path
    flood_mask_path: Path
    flood_ratio_path: Path
    area_sqkm: float
    area_hectares: float
    pixel_count: int
    infrastructure_summary: dict[str, Any]
    connectivity_summary: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    warnings: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest for this run.

        Written to ``manifest.json`` inside the run folder so downstream
        tools can inspect what the run produced without re-running.
        """
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "event": {
                "aoi_name": self.event.aoi_name,
                "event_date": self.event.event_date.isoformat(),
                "pre_days": self.event.pre_days,
                "post_days": self.event.post_days,
                "bbox": list(self.event.aoi.as_tuple()),
            },
            "scene_pair": {
                "pre": _scene_to_dict(self.scene_pair.pre),
                "post": _scene_to_dict(self.scene_pair.post),
                "selection_notes": self.scene_pair.selection_notes,
            },
            "flood_extent": {
                "area_sqkm": self.area_sqkm,
                "area_hectares": self.area_hectares,
                "pixel_count": self.pixel_count,
                "geojson": str(self.flood_extent_geojson),
            },
            "infrastructure_summary": self.infrastructure_summary,
            "connectivity_summary": self.connectivity_summary,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_s": (self.finished_at - self.started_at).total_seconds(),
            "warnings": self.warnings,
        }


def _scene_to_dict(scene: SceneMetadata) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "platform": scene.platform,
        "acquisition_datetime": scene.acquisition_datetime.isoformat(),
        "relative_orbit": scene.relative_orbit,
        "orbit_direction": scene.orbit_direction,
        "polarisation": scene.polarisation,
        "product_type": scene.product_type,
        "processing_level": scene.processing_level,
        "tile_or_granule_id": scene.tile_or_granule_id,
        "size_bytes": scene.size_bytes,
    }


# ---------------------------------------------------------------------------
# Run folder management
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_run_id(aoi_name: Optional[str], event_date: date) -> str:
    """Return a filesystem-safe run identifier from an AOI name and date.

    Example: ``("Koshi Riverplain", 2026-08-26)`` -> ``"koshi_riverplain_20260826"``.
    If ``aoi_name`` is None or empty, only the date is used.
    """
    base = ""
    if aoi_name:
        slug = _SLUG_RE.sub("_", aoi_name.strip().lower()).strip("_")
        base = slug
    date_part = event_date.strftime("%Y%m%d")
    if base:
        return f"{base}_{date_part}"
    return f"aoi_{date_part}"


def make_unique_run_dir(
    root: Path,
    run_id: str,
) -> Path:
    """Return a unique run directory under ``root``.

    If ``root/run_id`` exists, a numeric suffix ``_2``, ``_3``, ... is
    appended until a free name is found. The directory is created with
    parents. The function is atomic: two concurrent callers will not
    receive the same path because the successful creator wins the
    ``mkdir`` call.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_id
    if not candidate.exists():
        candidate.mkdir(parents=False)
        return candidate
    # Find the next free suffix.
    for n in range(2, 1000):
        candidate = root / f"{run_id}_{n}"
        if not candidate.exists():
            candidate.mkdir(parents=False)
            return candidate
    raise PipelineError(
        f"Could not create a unique run folder under {root} for run_id={run_id!r} "
        f"after 999 attempts. Clean up old runs or change the AOI name."
    )


# ---------------------------------------------------------------------------
# Search stage
# ---------------------------------------------------------------------------

def search_only(
    aoi_bbox: BoundingBox,
    event_date: date,
    *,
    pre_days: int = DEFAULT_WINDOW_DAYS,
    post_days: int = DEFAULT_WINDOW_DAYS,
    aoi_name: Optional[str] = None,
    settings: Optional[Settings] = None,
    client: Optional[CopernicusClient] = None,
) -> SceneAvailability:
    """Search CDSE for Sentinel-1 scenes and select a compatible pair.

    Does **not** download anything. The caller is expected to show the
    result to the user and ask for confirmation before invoking
    :func:`run_full_analysis`.

    Parameters
    ----------
    aoi_bbox:
        Area of interest in WGS84.
    event_date:
        Event date.
    pre_days, post_days:
        Search window on either side of the event date.
    aoi_name:
        Optional human-readable AOI name (used for the run_id).
    settings:
        Optional settings override.
    client:
        Optional pre-authenticated CDSE client.

    Returns
    -------
    :class:`SceneAvailability`
    """
    settings = settings or get_settings()

    warnings: list[str] = []
    pre_scenes: list[SceneMetadata] = []
    post_scenes: list[SceneMetadata] = []
    selected_pair: Optional[ScenePair] = None
    selection_notes: list[str] = []
    estimated_mb = 0.0

    owns_client = client is None
    client = client or CopernicusClient(settings=settings)

    try:
        # Search pre and post windows.
        pre_start = event_date - timedelta(days=pre_days)
        pre_end = event_date - timedelta(days=1)
        post_start = event_date + timedelta(days=1)
        post_end = event_date + timedelta(days=post_days)

        try:
            pre_scenes = search_sentinel1(
                aoi_bbox, pre_start, pre_end,
                max_results=50, settings=settings, client=client,
            )
        except (DataAccessError, NoResultsError) as exc:
            warnings.append(f"Pre-event search failed: {exc}")

        try:
            post_scenes = search_sentinel1(
                aoi_bbox, post_start, post_end,
                max_results=50, settings=settings, client=client,
            )
        except (DataAccessError, NoResultsError) as exc:
            warnings.append(f"Post-event search failed: {exc}")

        # Attempt to select a compatible pair. select_sentinel1_pair
        # re-runs the search internally, so we call it only when both
        # lists have content.
        if pre_scenes and post_scenes:
            try:
                selected_pair = select_sentinel1_pair(
                    aoi_bbox, event_date,
                    pre_days=pre_days, post_days=post_days,
                    settings=settings, client=client,
                )
                selection_notes = list(selected_pair.selection_notes)

                # Sum estimated sizes of the two selected products.
                for scene in (selected_pair.pre, selected_pair.post):
                    if scene.size_bytes:
                        estimated_mb += scene.size_bytes / (1024 * 1024)
            except (DataAccessError, NoResultsError) as exc:
                warnings.append(f"Pair selection failed: {exc}")

        # If we could not select a pair but do have scenes, estimate
        # the size of the smallest pre and smallest post scene.
        if selected_pair is None and pre_scenes and post_scenes:
            pre_sizes = [
                s.size_bytes for s in pre_scenes if s.size_bytes
            ]
            post_sizes = [
                s.size_bytes for s in post_scenes if s.size_bytes
            ]
            if pre_sizes and post_sizes:
                estimated_mb = (
                    min(pre_sizes) + min(post_sizes)
                ) / (1024 * 1024)
                warnings.append(
                    "No compatible pre/post pair with matching orbit was "
                    "found. The estimate is based on the smallest available "
                    "scenes and may not produce a pixel-comparable analysis."
                )

    finally:
        if owns_client:
            client.close()

    if not pre_scenes:
        warnings.append(
            f"No pre-event Sentinel-1 scenes found between {event_date - timedelta(days=pre_days)} "
            f"and {event_date - timedelta(days=1)}."
        )
    if not post_scenes:
        warnings.append(
            f"No post-event Sentinel-1 scenes found between {event_date + timedelta(days=1)} "
            f"and {event_date + timedelta(days=post_days)}."
        )

    return SceneAvailability(
        aoi=aoi_bbox,
        event_date=event_date,
        pre_scenes=pre_scenes,
        post_scenes=post_scenes,
        selected_pair=selected_pair,
        estimated_download_mb=round(estimated_mb, 1),
        selection_notes=selection_notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float], None]


def _emit(
    callback: Optional[ProgressCallback],
    message: str,
    fraction: float,
) -> None:
    """Call the progress callback if provided, then log."""
    log.info("[%.0f%%] %s", fraction * 100, message)
    if callback is not None:
        try:
            callback(message, fraction)
        except Exception as exc:  # pragma: no cover - user code
            log.warning("Progress callback raised: %s", exc)


def run_full_analysis(
    event: EventDefinition,
    *,
    settings: Optional[Settings] = None,
    confirm_download: bool = False,
    size_warning_mb: float = DEFAULT_SIZE_WARNING_MB,
    ratio_threshold: float = 0.4,
    max_slope_deg: float = 5.0,
    min_mmu_pixels: int = 9,
    fill_hole_pixels: int = 4,
    merge_gap_m: float = 50.0,
    progress: Optional[ProgressCallback] = None,
    availability: Optional[SceneAvailability] = None,
) -> PipelineResult:
    """Run the full Sentinel-1 flood assessment for a user-supplied event.

    Parameters
    ----------
    event:
        Validated :class:`EventDefinition`. Supplies the AOI, event date,
        and pre/post search windows.
    settings:
        Optional settings override.
    confirm_download:
        Set to ``True`` after the user has confirmed that a potentially
        large download may proceed. If ``False`` and the estimated
        download size exceeds ``size_warning_mb``, the function raises
        :class:`DownloadConfirmationRequired` before touching the network.
    size_warning_mb:
        Threshold above which ``confirm_download`` is required.
    ratio_threshold, max_slope_deg, min_mmu_pixels, fill_hole_pixels:
        Flood-detection parameters, forwarded unchanged to the detector.
    merge_gap_m:
        Vectorization merge gap.
    progress:
        Optional callback ``(message, fraction)``. Called at each stage.
    availability:
        Optional pre-computed :class:`SceneAvailability`. If provided,
        the search stage is skipped and this data is used. Useful when
        the caller has already displayed the search results to a user.

    Returns
    -------
    :class:`PipelineResult`

    Raises
    ------
    NoSuitableImageryError
        If no compatible Sentinel-1 pair exists.
    DownloadConfirmationRequired
        If the estimated download is large and confirmation was not given.
    PipelineError
        For downstream stage failures.
    """
    settings = settings or get_settings()
    started_at = datetime.now(timezone.utc)

    # ---- Stage 0: search if not already done ----------------------------
    if availability is None:
        _emit(progress, "Searching for suitable satellite imagery...", 0.02)
        availability = search_only(
            event.aoi, event.event_date,
            pre_days=event.pre_days, post_days=event.post_days,
            aoi_name=event.aoi_name, settings=settings,
        )

    if availability.selected_pair is None:
        detail = "; ".join(availability.warnings) or (
            "No compatible Sentinel-1 pair was found for the requested "
            "area and dates. Try widening the search window, moving the "
            "AOI, or choosing different dates."
        )
        raise NoSuitableImageryError(detail)

    pair = availability.selected_pair

    # ---- Stage 0b: download size confirmation ---------------------------
    if (
        availability.estimated_download_mb > size_warning_mb
        and not confirm_download
    ):
        raise DownloadConfirmationRequired(
            f"The selected Sentinel-1 products total approximately "
            f"{availability.estimated_download_mb:.0f} MB. Re-run with "
            f"confirm_download=True to proceed."
        )

    # ---- Prepare the run folder ----------------------------------------
    run_id = make_run_id(event.aoi_name, event.event_date)
    run_root = settings.output_path / "runs"
    run_dir = make_unique_run_dir(run_root, run_id)

    # Subdirectories inside the run folder.
    run_interim = run_dir / "interim"
    run_processed = run_dir / "processed"
    run_interim.mkdir(parents=True, exist_ok=True)
    run_processed.mkdir(parents=True, exist_ok=True)

    _emit(progress, f"Created run folder: {run_dir.name}", 0.05)

    result_holder: dict[str, Any] = {
        "run_dir": run_dir,
        "warnings": list(availability.warnings),
    }
    try:
        result = _run_pipeline_body(
            event=event,
            pair=pair,
            run_dir=run_dir,
            run_interim=run_interim,
            run_processed=run_processed,
            settings=settings,
            ratio_threshold=ratio_threshold,
            max_slope_deg=max_slope_deg,
            min_mmu_pixels=min_mmu_pixels,
            fill_hole_pixels=fill_hole_pixels,
            merge_gap_m=merge_gap_m,
            progress=progress,
            started_at=started_at,
            initial_warnings=list(availability.warnings),
        )
    except Exception as exc:
        # Log the failure into the run folder for later inspection.
        failure_note = {
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(run_dir / "FAILURE.json", "w", encoding="utf-8") as fh:
                json.dump(failure_note, fh, indent=2)
        except OSError:
            pass
        raise

    # Persist the manifest.
    manifest = result.to_manifest()
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    _emit(progress, "Done.", 1.0)
    return result


def _run_pipeline_body(
    *,
    event: EventDefinition,
    pair: ScenePair,
    run_dir: Path,
    run_interim: Path,
    run_processed: Path,
    settings: Settings,
    ratio_threshold: float,
    max_slope_deg: float,
    min_mmu_pixels: int,
    fill_hole_pixels: int,
    merge_gap_m: float,
    progress: Optional[ProgressCallback],
    started_at: datetime,
    initial_warnings: list[str],
) -> PipelineResult:
    """The actual pipeline body. Separated from the public entry point
    so that failures can be logged into the run folder consistently."""

    warnings = list(initial_warnings)

    # ---- Stage 1: DEM --------------------------------------------------
    _emit(progress, "Downloading and preparing elevation data...", 0.10)
    dem_dir = run_interim / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    dem_tiles = download_dem_for_bbox(event.aoi, dem_dir)
    if not dem_tiles:
        raise PipelineError(
            "No Copernicus DEM tiles intersect the AOI. "
            "The AOI may be outside the 60°S to 60°N coverage."
        )
    dem_mosaic = dem_dir / "dem_mosaic.tif"
    mosaic_dem_tiles(dem_tiles, dem_mosaic, bbox=event.aoi)

    dem_utm = _reproject_dem_to_utm(dem_mosaic, event.aoi, run_interim)
    _emit(progress, f"Elevation data ready: {dem_utm.name}", 0.18)

    # ---- Stage 2: locate or download Sentinel-1 products ---------------
    raw_dir = settings.raw_data_path / "sentinel1"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pre_zip = _resolve_or_download_product(
        pair.pre, raw_dir,
        progress=progress, fraction_base=0.20, fraction_span=0.15,
        settings=settings,
    )
    _emit(progress, f"Pre-event scene ready: {pre_zip.name}", 0.36)

    post_zip = _resolve_or_download_product(
        pair.post, raw_dir,
        progress=progress, fraction_base=0.36, fraction_span=0.15,
        settings=settings,
    )
    _emit(progress, f"Post-event scene ready: {post_zip.name}", 0.52)

    # ---- Stage 3: terrain correction -----------------------------------
    _emit(progress, "Correcting radar terrain distortion...", 0.54)
    pre_tc = run_interim / "pre_terrain_corrected.tif"
    preprocess_sentinel1_product(
        pre_zip, event.aoi, dem_utm, pre_tc,
        measurement_group="IW/VV", correct_radiometry=None,
    )

    post_tc = run_interim / "post_terrain_corrected.tif"
    preprocess_sentinel1_product(
        post_zip, event.aoi, dem_utm, post_tc,
        measurement_group="IW/VV", correct_radiometry=None,
    )
    _emit(progress, "Radar imagery ready.", 0.68)

    # ---- Stage 4: alignment --------------------------------------------
    post_aligned = run_interim / "post_aligned.tif"
    align_raster_to_reference(post_tc, pre_tc, post_aligned)
    _emit(progress, "Images aligned.", 0.72)

    # ---- Stage 5: slope ------------------------------------------------
    slope_path = run_interim / "slope.tif"
    slope, slope_profile = compute_slope_degrees(dem_utm)
    write_raster(
        slope_path, slope,
        transform=slope_profile["transform"],
        crs=slope_profile["crs"],
        nodata=float("nan"),
        dtype="float32",
    )
    slope_aligned = run_interim / "slope_aligned.tif"
    align_raster_to_reference(slope_path, pre_tc, slope_aligned)
    _emit(progress, "Terrain analysis ready.", 0.76)

    # ---- Stage 6: flood detection --------------------------------------
    _emit(progress, "Detecting possible flood areas...", 0.78)
    detection: FloodDetectionResult = detect_flood_change(
        pre_tc, post_aligned,
        ratio_threshold=ratio_threshold,
        max_slope_deg=max_slope_deg,
        slope_path=slope_aligned,
        min_mmu_pixels=min_mmu_pixels,
        fill_hole_pixels=fill_hole_pixels,
    )
    warnings.extend(detection.warnings or [])

    mask_path = run_processed / "candidate_flood_mask.tif"
    ratio_path = run_processed / "backscatter_ratio.tif"
    save_flood_mask(detection, mask_path)
    save_ratio_raster(detection, ratio_path)
    _emit(
        progress,
        f"Detected {detection.pixel_count} candidate flood pixels "
        f"({detection.area_sqkm:.3f} km²).",
        0.82,
    )

    # ---- Stage 7: vectorization ----------------------------------------
    flood_geojson = run_processed / "candidate_flood_extent.geojson"
    vector = vectorize_flood_mask(
        mask_path, merge_gap_m=merge_gap_m,
    )
    vector_wgs84 = reproject_geojson_to_wgs84(
        vector, vector["properties"]["crs"],
    )
    write_geojson(vector_wgs84, flood_geojson)
    _emit(
        progress,
        f"Vectorised into {vector['properties']['polygon_count']} polygons.",
        0.86,
    )

    # ---- Stage 8: infrastructure ---------------------------------------
    _emit(progress, "Analysing buildings, roads, and bridges...", 0.88)
    try:
        infra_summary = analyze_infrastructure_exposure(
            flood_geojson, event.aoi,
            osm_snapshot=_resolve_snapshot_date(event),
            event_date=event.event_date,
            buffer_m=0.0,
            output_dir=run_dir / "infrastructure",
            settings=settings,
        )
    except Exception as exc:
        warnings.append(f"Infrastructure analysis failed: {exc}")
        infra_summary = {"error": str(exc)}

    # ---- Stage 9: connectivity -----------------------------------------
    _emit(progress, "Analysing road access...", 0.94)
    try:
        conn_summary = analyze_connectivity(
            flood_geojson, event.aoi,
            osm_snapshot=_resolve_snapshot_date(event),
            event_date=event.event_date,
            disruption_fraction=0.20,
            output_dir=run_dir / "connectivity",
            settings=settings,
        )
    except Exception as exc:
        warnings.append(f"Connectivity analysis failed: {exc}")
        conn_summary = {"error": str(exc)}

    finished_at = datetime.now(timezone.utc)
    _emit(progress, "Analysis complete.", 1.0)

    return PipelineResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        event=event,
        scene_pair=pair,
        flood_extent_geojson=flood_geojson,
        flood_mask_path=mask_path,
        flood_ratio_path=ratio_path,
        area_sqkm=detection.area_sqkm,
        area_hectares=detection.area_hectares,
        pixel_count=detection.pixel_count,
        infrastructure_summary=infra_summary,
        connectivity_summary=conn_summary,
        started_at=started_at,
        finished_at=finished_at,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_snapshot_date(event: EventDefinition) -> date:
    """Return the OSM snapshot date for a given event.

    Uses the default snapshot (2026-07-27) unless the event is earlier,
    in which case the snapshot is set to 30 days before the event.
    """
    default = date(2026, 7, 27)
    if event.event_date <= default:
        return event.event_date - timedelta(days=30)
    return default


def _reproject_dem_to_utm(
    dem_mosaic: Path,
    aoi: BoundingBox,
    run_interim: Path,
) -> Path:
    """Reproject a DEM mosaic to the AOI's UTM CRS with -9999 nodata."""
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dst_crs = utm_crs_for_bbox(aoi)
    dst_path = run_interim / "dem_utm.tif"

    with rasterio.open(dem_mosaic) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "deflate",
            "nodata": -9999.0,
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=-9999.0,
                resampling=Resampling.bilinear,
            )

    return dst_path


def _resolve_or_download_product(
    scene: SceneMetadata,
    raw_dir: Path,
    *,
    progress: Optional[ProgressCallback],
    fraction_base: float,
    fraction_span: float,
    settings: Settings,
) -> Path:
    """Return the local path to a scene zip, downloading it if missing."""
    name = scene.tile_or_granule_id or scene.scene_id
    if not name.endswith(".zip"):
        name = name + ".zip"
    target = raw_dir / name

    if target.is_file() and target.stat().st_size > 0:
        log.info("Using cached product: %s", target.name)
        return target

    _emit(progress, f"Downloading {name}...", fraction_base)
    client = CopernicusClient(settings=settings)
    try:
        client.download_product(scene.scene_id, target)
    finally:
        client.close()
    _emit(progress, f"Downloaded {name}.", fraction_base + fraction_span)
    return target


__all__ = [
    "DEFAULT_SIZE_WARNING_MB",
    "DEFAULT_WINDOW_DAYS",
    "DownloadConfirmationRequired",
    "NoSuitableImageryError",
    "PipelineError",
    "PipelineResult",
    "SceneAvailability",
    "make_run_id",
    "make_unique_run_dir",
    "run_full_analysis",
    "search_only",
]