"""Vectorization of raster flood masks into georeferenced polygons.

This module turns the boolean candidate flood mask produced by
:mod:`georescue.processing.flood_detection` into georeferenced polygons
that can be:

- Overlaid on interactive maps (Folium).
- Intersected with OpenStreetMap infrastructure (Batch 4).
- Used as input to settlement connectivity analysis (Batch 5).
- Exported as GeoJSON for external tools.

Design rules
------------
- Areas are computed in a projected CRS (the raster's own CRS), never in
  degrees. The :func:`polygon_area_sqkm` helper enforces this.
- Polygons smaller than a configured minimum area are removed. This is
  the *polygon-level* minimum mapping unit, distinct from the pixel-level
  MMU applied inside ``flood_detection.py``.
- Fragmented polygons that are close together are merged into coherent
  clusters. This matches the physical reality of a flood event, which
  usually forms a connected extent along a river corridor rather than
  hundreds of disconnected patches.
- Interior rings (holes) are preserved when they are large enough to be
  meaningful, and filled when they are noise.
- Every polygon carries a small set of provenance properties:
  ``area_sqkm``, ``source``, ``method``, and a stable identifier.

Terminology
-----------
The output is called **candidate flood extent**. It is the vectorized
form of the candidate flood mask. It is not "confirmed flood extent",
"confirmed damage", or "observed flooding".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rasterio import features
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from georescue.logging_config import get_logger
from georescue.processing import ProcessingError
from georescue.processing.raster_utils import read_raster

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Raster -> polygons
# ---------------------------------------------------------------------------

def raster_mask_to_polygons(
    mask: np.ndarray,
    transform: Any,
    crs: CRS | str,
) -> list[Any]:
    """Convert a boolean raster mask to a list of Shapely polygons.

    Parameters
    ----------
    mask:
        2-D boolean array. ``True`` cells become part of polygons.
    transform:
        The raster's affine transform (from rasterio).
    crs:
        The raster's CRS (rasterio CRS or EPSG string). Used only to tag
        the polygons; the coordinates themselves are in the raster CRS.

    Returns
    -------
    List of Shapely Polygon objects in the raster's CRS.
    """
    if mask.dtype != bool:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    if not mask_bool.any():
        return []

    shapes_iter = features.shapes(
        mask_bool.astype(np.uint8),
        mask=mask_bool,
        transform=transform,
        connectivity=8,
    )

    polygons: list[Any] = []
    for geom_dict, value in shapes_iter:
        if value != 1:
            continue
        try:
            geom = shape(geom_dict)
        except Exception as exc:
            log.warning("Skipping malformed polygon geometry: %s", exc)
            continue
        if geom.is_empty:
            continue
        polygons.append(geom)

    log.info("Vectorization produced %d raw polygons.", len(polygons))
    return polygons


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def polygon_area_sqkm(geom: Any, crs: CRS) -> float:
    """Return the area of a geometry in km².

    Raises :class:`ProcessingError` if the geometry's CRS is geographic,
    because degree-based areas are not meaningful.
    """
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        raise ProcessingError(
            "polygon_area_sqkm requires a projected CRS; got a geographic one. "
            "Reproject the geometry to UTM (or another equal-area CRS) first."
        )
    return float(geom.area) / 1_000_000.0


def merge_nearby_polygons(
    polygons: list[Any],
    gap_m: float,
    *,
    min_area_sqkm: float = 0.0,
    crs: Optional[CRS] = None,
) -> list[Any]:
    """Merge polygons whose boundaries are within ``gap_m`` metres.

    A flood event usually produces a connected extent along a river
    corridor. The raster mask, however, often has small breaks caused by
    non-flood pixels (vegetation, terrain artifacts, speckle). This
    function fills those breaks by:

    1. Buffering every polygon outward by ``gap_m / 2``.
    2. Taking the union of all buffered geometries.
    3. Buffering the union inward by ``gap_m / 2`` to restore original
       shape.
    4. Optionally discarding merged clusters below ``min_area_sqkm``.

    This is the standard morphological closing operation expressed with
    Shapely.

    Parameters
    ----------
    polygons:
        Input polygons in a projected CRS with metre units.
    gap_m:
        Maximum gap in metres that will be closed. A value of 50 m means
        two polygons separated by a 50 m strip of non-flood pixels will
        be merged into one.
    min_area_sqkm:
        If positive, merged clusters below this area are dropped. Use 0
        to keep all clusters.
    crs:
        Optional CRS of the polygons (only needed if ``min_area_sqkm``
        is used).

    Returns
    -------
    List of merged geometries. Each element may be a Polygon or a
    MultiPolygon if the merge produced one.
    """
    if gap_m <= 0 or len(polygons) <= 1:
        return list(polygons)

    half = gap_m / 2.0

    # Buffer outward, union, buffer inward. This is the morphological
    # closing operation.
    try:
        buffered = [geom.buffer(half, resolution=4) for geom in polygons]
        union = unary_union(buffered)
        closed = union.buffer(-half, resolution=4)
    except Exception as exc:
        log.warning("merge_nearby_polygons failed: %s. Returning inputs.", exc)
        return list(polygons)

    # Flatten to a list of individual geometries.
    if closed.is_empty:
        return list(polygons)
    if closed.geom_type == "MultiPolygon":
        merged = list(closed.geoms)
    elif closed.geom_type == "Polygon":
        merged = [closed]
    else:
        # GeometryCollection or unexpected type: extract Polygon and
        # MultiPolygon parts.
        collected: list[Any] = []
        for geom in getattr(closed, "geoms", []):
            if geom.geom_type == "Polygon":
                collected.append(geom)
            elif geom.geom_type == "MultiPolygon":
                collected.extend(geom.geoms)
        merged = collected if collected else list(polygons)

    # Drop merged clusters that fell below the minimum area.
    if min_area_sqkm > 0 and crs is not None:
        kept: list[Any] = []
        removed = 0
        for geom in merged:
            try:
                area = polygon_area_sqkm(geom, crs)
            except ProcessingError:
                kept.append(geom)
                continue
            if area >= min_area_sqkm:
                kept.append(geom)
            else:
                removed += 1
        if removed:
            log.info(
                "merge_nearby_polygons: dropped %d merged clusters below "
                "%.4f km².", removed, min_area_sqkm,
            )
        merged = kept

    log.info(
        "Merged %d polygons into %d cluster(s) using a %.0f m gap.",
        len(polygons), len(merged), gap_m,
    )
    return merged


def filter_polygons_by_min_area(
    polygons: list[Any],
    min_area_sqkm: float,
    crs: CRS,
) -> tuple[list[Any], int]:
    """Remove polygons smaller than ``min_area_sqkm``.

    Returns ``(kept_polygons, removed_count)``.
    """
    kept: list[Any] = []
    removed = 0
    for geom in polygons:
        try:
            area = polygon_area_sqkm(geom, crs)
        except ProcessingError:
            removed += 1
            continue
        if area >= min_area_sqkm:
            kept.append(geom)
        else:
            removed += 1
    log.info(
        "Minimum-area filter: kept %d of %d polygons (removed %d).",
        len(kept), len(polygons), removed,
    )
    return kept, removed


def simplify_polygons(
    polygons: list[Any],
    tolerance_m: float,
) -> list[Any]:
    """Apply Douglas–Peucker simplification in the raster's metre CRS."""
    if tolerance_m <= 0:
        return polygons
    simplified: list[Any] = []
    for geom in polygons:
        try:
            simplified_geom = geom.simplify(tolerance_m, preserve_topology=True)
        except Exception as exc:
            log.warning("Simplification failed for one polygon: %s", exc)
            simplified.append(geom)
            continue
        if simplified_geom.is_empty:
            continue
        simplified.append(simplified_geom)
    return simplified


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def vectorize_flood_mask(
    mask_path: Path,
    *,
    min_area_sqkm: float = 0.005,     # 0.5 ha
    simplify_tolerance_m: float = 15.0,
    merge_gap_m: float = 50.0,
) -> dict[str, Any]:
    """Convert a flood mask GeoTIFF into a GeoJSON FeatureCollection.

    Pipeline
    --------
    1. Read the mask raster.
    2. Raster -> raw Shapely polygons.
    3. Simplify each polygon (Douglas–Peucker).
    4. Merge polygons within ``merge_gap_m`` into coherent clusters.
    5. Drop clusters smaller than ``min_area_sqkm``.
    6. Build GeoJSON features with provenance properties.

    Parameters
    ----------
    mask_path:
        Path to the flood mask raster. Non-zero pixels are considered
        flood. ``255`` (nodata from ``save_flood_mask``) is treated as
        not-flood.
    min_area_sqkm:
        Polygons smaller than this are removed. Default 0.005 km²
        (0.5 ha) matches the pixel-level MMU used during detection.
    simplify_tolerance_m:
        Douglas–Peucker simplification tolerance in metres. Default
        15 m, roughly half a pixel.
    merge_gap_m:
        Maximum gap between polygons (in metres) that will be closed by
        the morphological merge step. Set to 0 to disable merging.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection.
    """
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        raise ProcessingError(f"Flood mask not found: {mask_path}")

    array, profile = read_raster(mask_path)
    crs = profile["crs"]
    transform = profile["transform"]

    if crs is None:
        raise ProcessingError(
            f"Flood mask {mask_path.name} has no CRS; cannot vectorize safely."
        )
    if array.ndim != 2:
        raise ProcessingError(
            f"Flood mask has unexpected shape {array.shape}; expected 2-D."
        )

    # NaN and 255 both mean "not flood".
    valid = np.isfinite(array)
    mask_bool = valid & (array > 0) & (array != 255)

    flood_pixels = int(mask_bool.sum())
    if flood_pixels == 0:
        log.warning(
            "Flood mask contains no candidate flood pixels; "
            "returning empty FeatureCollection."
        )
        return {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "source": mask_path.name,
                "crs": str(crs),
                "flood_pixels": 0,
                "total_area_sqkm": 0.0,
                "polygon_count": 0,
                "raw_polygon_count": 0,
                "merged_from": 0,
                "merge_gap_m": merge_gap_m,
            },
        }

    # Raster -> polygons.
    raw_polygons = raster_mask_to_polygons(mask_bool, transform, crs)
    if not raw_polygons:
        raise ProcessingError(
            "Vectorization returned zero polygons despite positive mask."
        )
    raw_count = len(raw_polygons)

    # Simplify before merging; the merge then operates on the simplified
    # geometries, which is faster and produces cleaner output.
    polygons = simplify_polygons(raw_polygons, simplify_tolerance_m)

    # Merge nearby polygons into coherent clusters.
    if merge_gap_m > 0:
        polygons = merge_nearby_polygons(
            polygons,
            merge_gap_m,
            min_area_sqkm=0.0,  # filter after merge, not during
            crs=crs,
        )

    # Minimum-area filter applies to the merged clusters.
    polygons, removed_count = filter_polygons_by_min_area(
        polygons, min_area_sqkm, crs,
    )

    # Build GeoJSON features.
    features_list: list[dict[str, Any]] = []
    total_area = 0.0
    for idx, geom in enumerate(polygons):
        area = polygon_area_sqkm(geom, crs)
        total_area += area
        feature = {
            "type": "Feature",
            "id": f"flood-{idx:04d}",
            "properties": {
                "id": f"flood-{idx:04d}",
                "area_sqkm": round(area, 6),
                "area_hectares": round(area * 100.0, 4),
                "source": mask_path.name,
                "method": "Sentinel-1 VV backscatter ratio change detection",
                "geometry_crs": str(crs),
                "note": "candidate flood extent — not confirmed flood damage",
            },
            "geometry": mapping(geom),
        }
        features_list.append(feature)

    log.info(
        "Vectorization complete: %d clusters, %.4f km² total "
        "(raw polygons: %d, merge gap: %.0f m).",
        len(features_list), total_area, raw_count, merge_gap_m,
    )

    return {
        "type": "FeatureCollection",
        "features": features_list,
        "properties": {
            "source": mask_path.name,
            "crs": str(crs),
            "flood_pixels": flood_pixels,
            "total_area_sqkm": round(total_area, 6),
            "polygon_count": len(features_list),
            "raw_polygon_count": raw_count,
            "merged_from": raw_count - len(features_list),
            "removed_by_min_area": removed_count,
            "min_area_sqkm": min_area_sqkm,
            "simplify_tolerance_m": simplify_tolerance_m,
            "merge_gap_m": merge_gap_m,
        },
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_geojson(
    geojson: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a GeoJSON FeatureCollection to disk, atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(geojson, fh, ensure_ascii=False)
        temp_path.replace(output_path)
    except OSError as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise ProcessingError(
            f"Could not write GeoJSON to {output_path}: {exc}"
        ) from exc

    log.info("Wrote GeoJSON: %s", output_path.name)
    return output_path


def reproject_geojson_to_wgs84(
    geojson: dict[str, Any],
    src_crs: CRS | str,
) -> dict[str, Any]:
    """Reproject every geometry in a GeoJSON FeatureCollection to WGS84."""
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    if isinstance(src_crs, str):
        src_crs = CRS.from_user_input(src_crs)
    if src_crs.to_epsg() == 4326:
        return geojson

    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

    new_features: list[dict[str, Any]] = []
    for feat in geojson.get("features", []):
        geom_dict = feat.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
            reproj = shapely_transform(transformer.transform, geom)
        except Exception as exc:
            log.warning("Skipping feature that failed reprojection: %s", exc)
            continue
        new_feat = {
            "type": "Feature",
            "id": feat.get("id"),
            "properties": dict(feat.get("properties", {})),
            "geometry": mapping(reproj),
        }
        new_feat["properties"]["geometry_crs"] = "EPSG:4326"
        new_features.append(new_feat)

    out = {
        "type": "FeatureCollection",
        "features": new_features,
        "properties": dict(geojson.get("properties", {})),
    }
    out["properties"]["crs"] = "EPSG:4326"
    return out


__all__ = [
    "filter_polygons_by_min_area",
    "merge_nearby_polygons",
    "polygon_area_sqkm",
    "raster_mask_to_polygons",
    "reproject_geojson_to_wgs84",
    "simplify_polygons",
    "vectorize_flood_mask",
    "write_geojson",
]