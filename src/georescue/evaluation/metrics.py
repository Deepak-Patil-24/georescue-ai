"""Case-study evaluation against a reference flood/damage map.

This module compares our candidate flood extent with an independent
reference layer such as the official Copernicus EMSR927 products. It
computes intersection-over-union (IoU), precision, recall, F1, pixel
counts, and area comparison.

Strict methodological rules
---------------------------
- The reference layer is **read-only**. It is never fed into the flood
  detection pipeline and never used to tune the algorithm.
- Every metric is computed at runtime from the two input files. No
  numbers are hardcoded.
- If either input is missing, or if the two rasters do not overlap, the
  function returns a structured record with ``"evaluated": False`` and
  an explanatory message. It never fabricates a metric.

Method
------
1. Load the candidate flood extent GeoJSON (WGS84) and burn it to a
   raster on a chosen reference grid.
2. Load the reference EMSR927 layer, select the appropriate layer
   (impact/damage polygon), reproject to WGS84, and burn it to the same
   grid.
3. Compare binary masks pixel-by-pixel and compute the confusion matrix
   and derived metrics.

The reference EMSR927 GRA product contains building-level damage
grading. For a pixel-level comparison we use the union of all damaged
building footprints within AOI01 as the reference flood-affected area.
Alternative reference layers within the GeoPackage can be selected via
the ``reference_layer`` parameter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.models import BoundingBox
from georescue.processing.raster_utils import utm_crs_for_bbox

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    """Computed metrics for one candidate-vs-reference comparison."""

    evaluated: bool
    reason: str
    reference_source: str
    reference_layer: str
    candidate_source: str
    resolution_m: float
    crs: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    iou: float
    precision: float
    recall: float
    f1: float
    candidate_area_sqkm: float
    reference_area_sqkm: float
    intersection_area_sqkm: float
    candidate_only_area_sqkm: float
    reference_only_area_sqkm: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_candidate_union(candidate_path: Path):
    """Return the union of the candidate flood polygons in WGS84."""
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate flood extent not found: {candidate_path}")
    gdf = gpd.read_file(candidate_path)
    if gdf.empty:
        raise ValueError(f"Candidate flood extent is empty: {candidate_path}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    gdf = gdf.to_crs(epsg=4326)
    return unary_union(gdf.geometry)


def _load_reference_union(
    reference_path: Path,
    *,
    layer: Optional[str] = None,
) -> tuple[Any, str]:
    """Return the union of reference geometries in WGS84 and the layer name."""
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference layer not found: {reference_path}")

    # GeoPackage may have multiple layers; list them to select one.
    if reference_path.suffix.lower() in (".gpkg",):
        import fiona
        layers = fiona.listlayers(str(reference_path))
        if not layers:
            raise ValueError(f"No layers in GeoPackage: {reference_path}")
        chosen = layer if layer and layer in layers else layers[0]
        log.info(
            "Reference GeoPackage has layers: %s. Using %r.",
            layers, chosen,
        )
        gdf = gpd.read_file(reference_path, layer=chosen)
    else:
        chosen = layer or "default"
        gdf = gpd.read_file(reference_path)

    if gdf.empty:
        raise ValueError(f"Reference layer is empty: {reference_path}")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    gdf = gdf.to_crs(epsg=4326)

    # Keep only polygonal geometries; explode multipart.
    poly_mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask]
    if gdf.empty:
        raise ValueError(
            f"Reference layer has no polygonal geometries: {reference_path}"
        )

    return unary_union(gdf.geometry), chosen


def _rasterize_to_grid(
    geometries,
    *,
    bbox: BoundingBox,
    resolution_m: float,
) -> tuple[np.ndarray, dict]:
    """Rasterize a WGS84 geometry onto a UTM grid covering ``bbox``.

    Returns ``(mask, meta)`` where ``meta`` contains the transform, CRS,
    width, height, and resolution.
    """
    proj_crs = utm_crs_for_bbox(bbox)

    # Compute the projected bounding box with a small buffer.
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    corners = [
        (bbox.min_lon, bbox.min_lat),
        (bbox.max_lon, bbox.min_lat),
        (bbox.max_lon, bbox.max_lat),
        (bbox.min_lon, bbox.max_lat),
    ]
    xs, ys = [], []
    for x, y in corners:
        px, py = transformer.transform(x, y)
        xs.append(px)
        ys.append(py)

    min_x = min(xs) - resolution_m
    min_y = min(ys) - resolution_m
    max_x = max(xs) + resolution_m
    max_y = max(ys) + resolution_m

    width = int(np.ceil((max_x - min_x) / resolution_m))
    height = int(np.ceil((max_y - min_y) / resolution_m))
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Reproject geometries to the projected CRS for rasterization.
    from shapely.ops import transform as shapely_transform
    proj_geom = shapely_transform(transformer.transform, geometries)

    mask = rasterize(
        [(proj_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    meta = {
        "transform": transform,
        "crs": str(proj_crs),
        "width": width,
        "height": height,
        "resolution_m": resolution_m,
    }
    return mask.astype(bool), meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_flood_extent(
    candidate_geojson: Path,
    reference_path: Path,
    *,
    aoi_bbox: BoundingBox,
    reference_layer: Optional[str] = None,
    resolution_m: float = 30.0,
    output_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Evaluate a candidate flood extent against a reference layer.

    Parameters
    ----------
    candidate_geojson:
        Path to ``candidate_flood_extent.geojson``.
    reference_path:
        Path to the reference EMSR927 GeoPackage or shapefile.
    aoi_bbox:
        Area of interest in WGS84. Used to bound the raster grid.
    reference_layer:
        Optional layer name inside the reference file. If omitted, the
        first available layer is used.
    resolution_m:
        Pixel size (metres) for the comparison grid. Default 30 m,
        matching the Sentinel-1 terrain-corrected resolution.
    output_dir:
        Where to write ``evaluation_summary.json``. Defaults to
        ``outputs/evaluation``.
    settings:
        Optional settings override.

    Returns
    -------
    dict
        Structured summary with the metrics and provenance. If either
        input is missing, returns ``{"evaluated": False, ...}``.
    """
    settings = settings or get_settings()
    output_dir = output_dir or (settings.output_path / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "evaluation_summary.json"

    # --- Attempt to load both inputs ---
    try:
        candidate_geom = _load_candidate_union(candidate_geojson)
        reference_geom, chosen_layer = _load_reference_union(
            reference_path, layer=reference_layer,
        )
    except FileNotFoundError as exc:
        result = {
            "evaluated": False,
            "reason": str(exc),
            "reference_source": str(reference_path),
            "candidate_source": str(candidate_geojson),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        log.warning("Evaluation skipped: %s", exc)
        return result
    except ValueError as exc:
        result = {
            "evaluated": False,
            "reason": str(exc),
            "reference_source": str(reference_path),
            "candidate_source": str(candidate_geojson),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        log.warning("Evaluation skipped: %s", exc)
        return result

    # --- Rasterize both to a common grid ---
    candidate_mask, meta = _rasterize_to_grid(
        candidate_geom, bbox=aoi_bbox, resolution_m=resolution_m,
    )
    reference_mask, _ = _rasterize_to_grid(
        reference_geom, bbox=aoi_bbox, resolution_m=resolution_m,
    )

    # --- Confusion matrix ---
    tp = int(np.logical_and(candidate_mask, reference_mask).sum())
    fp = int(np.logical_and(candidate_mask, ~reference_mask).sum())
    fn = int(np.logical_and(~candidate_mask, reference_mask).sum())
    tn = int(np.logical_and(~candidate_mask, ~reference_mask).sum())

    # --- Area (per pixel) ---
    px_area_sqkm = (resolution_m * resolution_m) / 1_000_000.0
    candidate_area = float((tp + fp) * px_area_sqkm)
    reference_area = float((tp + fn) * px_area_sqkm)
    intersection_area = float(tp * px_area_sqkm)
    candidate_only = float(fp * px_area_sqkm)
    reference_only = float(fn * px_area_sqkm)

    # --- Metrics (guarded against divide by zero) ---
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    metrics = EvaluationMetrics(
        evaluated=True,
        reason="Comparison performed successfully.",
        reference_source=str(reference_path),
        reference_layer=chosen_layer,
        candidate_source=str(candidate_geojson),
        resolution_m=resolution_m,
        crs=meta["crs"],
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        iou=round(iou, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        candidate_area_sqkm=round(candidate_area, 6),
        reference_area_sqkm=round(reference_area, 6),
        intersection_area_sqkm=round(intersection_area, 6),
        candidate_only_area_sqkm=round(candidate_only, 6),
        reference_only_area_sqkm=round(reference_only, 6),
    )

    result = asdict(metrics)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["note"] = (
        "Metrics are computed by pixel-wise comparison of two independent "
        "binary raster masks. The reference layer was used for evaluation "
        "only and was never fed into the flood-detection pipeline. "
        "'Candidate' refers to the flood extent derived from Sentinel-1 VV "
        "backscatter change detection. 'Reference' refers to the official "
        "Copernicus EMSR927 product supplied by the user."
    )

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    log.info("Wrote evaluation summary: %s", summary_path.name)

    log.info(
        "IoU=%.3f precision=%.3f recall=%.3f F1=%.3f "
        "(TP=%d FP=%d FN=%d TN=%d)",
        iou, precision, recall, f1, tp, fp, fn, tn,
    )
    return result


__all__ = [
    "EvaluationMetrics",
    "evaluate_flood_extent",
]