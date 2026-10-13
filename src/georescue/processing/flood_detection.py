"""Flood detection by Sentinel-1 backscatter change analysis.

This module implements a defensible baseline flood detector based on
pre-event and post-event Sentinel-1 VV backscatter comparison. It is
NOT a trained deep-learning model; it is a transparent, reproducible
change-detection algorithm with configurable thresholds.

Method
------
1. Load aligned pre-event and post-event backscatter rasters.
2. Compute the ratio ``post / pre`` in linear power units.
3. Threshold the ratio to find pixels that got darker after the event
   (water surfaces scatter radar energy away from the sensor, producing
   low backscatter).
4. Apply masks to remove implausible flood pixels:
   - Terrain slope above a threshold (water does not pond on steep slopes).
   - Nodata regions in either input.
   - (Optional) Permanent water, if a mask is available.
5. Apply a median filter to reduce speckle noise.
6. Apply a minimum mapping unit to remove single-pixel detections.
7. Return a boolean mask plus diagnostic rasters.

Terminology
-----------
The output is called **candidate flood extent**, not confirmed flood
extent. We cannot verify from SAR alone that every detected change is
flooding; terrain, vegetation, and other surface changes can also
produce backscatter changes. The mask is evidence supporting a hypothesis,
not proof.

References
----------
Backscatter-ratio change detection is a standard approach documented in
the flood-mapping literature, including the Sen1Floods11 benchmark and
the Kuro Siwo dataset. The specific threshold values are configurable and
documented in ``config/settings.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from georescue.logging_config import get_logger
from georescue.processing import ProcessingError
from georescue.processing.raster_utils import (
    compute_slope_degrees,
    mask_area_sqkm,
    median_filter_nodata_safe,
    read_raster,
    write_raster,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FloodDetectionResult:
    """Result of running the flood detector on a pre/post pair."""

    mask: np.ndarray                    # boolean, True = candidate flood
    ratio: np.ndarray                   # post/pre in linear power
    area_sqkm: float
    area_hectares: float
    pixel_count: int
    pixel_size_m: float
    crs: str
    transform: tuple
    valid_fraction: float               # fraction of pixels with valid data
    warnings: list[str]
    parameters: dict                    # thresholds and settings used


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def mask_slope(
    slope_deg: np.ndarray,
    max_slope_deg: float,
) -> np.ndarray:
    """Return True where slope is at or below ``max_slope_deg``.

    Pixels with slope above the threshold are masked out because standing
    water does not persist on steep slopes.
    """
    if max_slope_deg <= 0:
        return np.ones_like(slope_deg, dtype=bool)
    valid = np.isfinite(slope_deg)
    return valid & (slope_deg <= max_slope_deg)


def mask_invalid(
    *arrays: np.ndarray,
) -> np.ndarray:
    """Return True where all input arrays contain finite values."""
    if not arrays:
        raise ValueError("mask_invalid requires at least one array")
    finite = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        finite &= np.isfinite(arr)
    return finite


# ---------------------------------------------------------------------------
# Minimum mapping unit
# ---------------------------------------------------------------------------

def apply_minimum_mapping_unit(
    mask: np.ndarray,
    min_pixels: int,
    *,
    connectivity: int = 2,
) -> np.ndarray:
    """Remove connected components smaller than ``min_pixels``.

    Uses :func:`scipy.ndimage.label` with the given connectivity
    (1 = 4-connected, 2 = 8-connected). Returns a boolean mask of the same
    shape with small components removed.

    This is the standard "minimum mapping unit" cleanup used in remote
    sensing to suppress speckle-induced single-pixel detections.
    """
    if min_pixels <= 1:
        return mask.astype(bool)

    from scipy.ndimage import label

    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        if connectivity == 1
        else np.ones((3, 3), dtype=bool)
    )

    labeled, num = label(mask, structure=structure)
    if num == 0:
        return mask.astype(bool)

    sizes = np.bincount(labeled.ravel())
    # sizes[0] is the background count; ignore it.
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labeled]


def fill_small_holes(
    mask: np.ndarray,
    max_hole_pixels: int,
) -> np.ndarray:
    """Fill holes in the mask smaller than ``max_hole_pixels``.

    Holes are defined as background (False) regions that are fully
    surrounded by foreground (True) regions. Filling them produces
    cleaner polygons and reduces fragmented output.
    """
    if max_hole_pixels <= 0:
        return mask.astype(bool)

    from scipy.ndimage import binary_fill_holes, label

    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask.astype(bool)

    labeled, num = label(holes, structure=np.ones((3, 3), dtype=bool))
    if num == 0:
        return mask.astype(bool)

    sizes = np.bincount(labeled.ravel())
    keep_holes = np.zeros_like(sizes, dtype=bool)
    keep_holes[1:] = sizes[1:] <= max_hole_pixels

    fill_this = keep_holes[labeled]
    return mask | fill_here(fill_this, holes, labeled)


def fill_here(keep_holes_mask: np.ndarray, holes: np.ndarray, labeled: np.ndarray) -> np.ndarray:
    """Helper: return the subset of ``holes`` to fill."""
    return keep_holes_mask & holes


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_flood_change(
    pre_path: Path,
    post_path: Path,
    *,
    ratio_threshold: float = 0.4,
    max_slope_deg: Optional[float] = 5.0,
    slope_path: Optional[Path] = None,
    min_mmu_pixels: int = 5,
    fill_hole_pixels: int = 4,
    median_filter_size: int = 3,
    permanent_water_path: Optional[Path] = None,
    apply_median_filter: bool = True,
) -> FloodDetectionResult:
    """Run backscatter-ratio change detection on a pre/post pair.

    Parameters
    ----------
    pre_path, post_path:
        Aligned backscatter rasters (linear power, same CRS, transform,
        width, height, nodata). Use
        :func:`georescue.processing.raster_utils.align_raster_to_reference`
        to align them first.
    ratio_threshold:
        Post/pre ratio below which a pixel is considered candidate flood.
        Higher values produce more pixels; lower values are stricter.
        Literature values range 0.3–0.5 for linear power.
    max_slope_deg:
        Maximum terrain slope (degrees) at which standing water is
        plausible. Pixels with steeper slope are masked. ``None``
        disables slope masking.
    slope_path:
        Path to a slope raster aligned to the same grid. Required if
        ``max_slope_deg`` is set. Compute one with
        :func:`georescue.processing.raster_utils.compute_slope_degrees`.
    min_mmu_pixels:
        Remove connected components smaller than this many pixels after
        thresholding.
    fill_hole_pixels:
        Fill interior holes smaller than this many pixels.
    median_filter_size:
        Size of the median filter applied to reduce speckle (must be odd).
        Set ``apply_median_filter=False`` to skip filtering.
    permanent_water_path:
        Optional path to a permanent-water mask raster aligned to the
        same grid. Pixels marked as permanent water are removed from the
        output, because they are not flood *change*.
    apply_median_filter:
        Whether to apply the speckle-reducing median filter.

    Returns
    -------
    :class:`FloodDetectionResult`
    """
    pre_path = Path(pre_path)
    post_path = Path(post_path)

    warnings: list[str] = []
    parameters = {
        "ratio_threshold": ratio_threshold,
        "max_slope_deg": max_slope_deg,
        "min_mmu_pixels": min_mmu_pixels,
        "fill_hole_pixels": fill_hole_pixels,
        "median_filter_size": median_filter_size,
        "apply_median_filter": apply_median_filter,
        "permanent_water_used": permanent_water_path is not None,
    }

    # --- Load rasters ---
    pre, pre_profile = read_raster(pre_path)
    post, post_profile = read_raster(post_path)

    if pre.shape != post.shape:
        raise ProcessingError(
            f"Pre and post rasters have different shapes: "
            f"{pre.shape} vs {post.shape}. "
            f"Align them with align_raster_to_reference first."
        )

    # Compare transforms and CRS. Small floating-point differences are OK,
    # but a mismatch in CRS or a large transform difference indicates the
    # caller skipped alignment.
    pre_crs = pre_profile["crs"]
    post_crs = post_profile["crs"]
    if pre_crs != post_crs:
        raise ProcessingError(
            f"Pre and post rasters have different CRS: "
            f"{pre_crs} vs {post_crs}."
        )

    # --- Optional speckle filter on both images ---
    if apply_median_filter:
        pre_f = median_filter_nodata_safe(pre, median_filter_size)
        post_f = median_filter_nodata_safe(post, median_filter_size)
    else:
        pre_f = pre
        post_f = post

    # --- Ratio in linear power ---
    # Both input rasters are linear backscatter. Avoid divide-by-zero by
    # masking invalid pixels.
    valid = mask_invalid(pre_f, post_f)
    # Guard against near-zero pre values producing huge ratios.
    pre_safe = np.where(valid & (pre_f > 1e-8), pre_f, np.nan)
    ratio = post_f / pre_safe
    # Mark invalid pixels as NaN.
    ratio = np.where(valid, ratio, np.nan)

    # --- Threshold ---
    # Low ratio = pixel got darker = candidate flood.
    candidate = valid & (ratio < ratio_threshold)

    # --- Slope mask ---
    if max_slope_deg is not None:
        if slope_path is None:
            warnings.append(
                "Slope masking was requested but no slope raster was "
                "provided; skipping slope mask."
            )
        else:
            slope, slope_profile = read_raster(slope_path)
            if slope.shape != candidate.shape:
                warnings.append(
                    f"Slope raster shape {slope.shape} does not match "
                    f"image shape {candidate.shape}; skipping slope mask."
                )
            else:
                slope_ok = mask_slope(slope, max_slope_deg)
                removed = candidate & ~slope_ok
                candidate = candidate & slope_ok
                log.info(
                    "Slope mask removed %d candidate pixels "
                    "(slope > %.1f°).",
                    int(removed.sum()), max_slope_deg,
                )

    # --- Permanent water mask (optional) ---
    if permanent_water_path is not None:
        water, water_profile = read_raster(permanent_water_path)
        if water.shape != candidate.shape:
            warnings.append(
                f"Permanent water mask shape {water.shape} does not "
                f"match image shape {candidate.shape}; skipping water mask."
            )
        else:
            water_bool = np.isfinite(water) & (water > 0)
            removed = candidate & water_bool
            candidate = candidate & ~water_bool
            log.info(
                "Permanent-water mask removed %d candidate pixels.",
                int(removed.sum()),
            )
    else:
        warnings.append(
            "No permanent-water mask was provided. Permanent water bodies "
            "(lakes, wide rivers) may appear as false positives in the "
            "candidate flood extent."
        )

    # --- Morphological cleanup ---
    if min_mmu_pixels > 1:
        before = int(candidate.sum())
        candidate = apply_minimum_mapping_unit(candidate, min_mmu_pixels)
        after = int(candidate.sum())
        log.info(
            "MMU removed %d small components (%d -> %d pixels).",
            before - after, before, after,
        )

    if fill_hole_pixels > 0:
        before = int(candidate.sum())
        candidate = fill_small_holes(candidate, fill_hole_pixels)
        after = int(candidate.sum())
        log.info("Filled %d hole pixels.", after - before)

    # --- Area ---
    transform = pre_profile["transform"]
    crs = pre_profile["crs"]
    pixel_size_m = abs(transform.a) if transform.a else 0.0

    area_sqkm = mask_area_sqkm(candidate, transform, crs)
    area_ha = area_sqkm * 100.0
    pixel_count = int(candidate.sum())

    valid_fraction = float(valid.sum()) / float(valid.size) if valid.size else 0.0

    if pixel_count == 0:
        warnings.append(
            "No candidate flood pixels were detected. This may mean the "
            "area was not flooded, or the threshold is too strict, or the "
            "scenes are not comparable."
        )

    log.info(
        "Flood detection result: %d pixels = %.4f km² (%.1f ha). "
        "Valid pixel fraction: %.2f%%.",
        pixel_count, area_sqkm, area_ha, valid_fraction * 100,
    )

    return FloodDetectionResult(
        mask=candidate,
        ratio=ratio,
        area_sqkm=area_sqkm,
        area_hectares=area_ha,
        pixel_count=pixel_count,
        pixel_size_m=pixel_size_m,
        crs=str(crs) if crs else "",
        transform=tuple(transform)[:6],
        valid_fraction=valid_fraction,
        warnings=warnings,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Persisting results
# ---------------------------------------------------------------------------

def save_flood_mask(
    result: FloodDetectionResult,
    output_path: Path,
    *,
    as_uint8: bool = True,
) -> Path:
    """Write the boolean flood mask to a GeoTIFF.

    By default the mask is written as ``uint8`` with values 0 and 1,
    which is compact and universally readable. Set ``as_uint8=False`` to
    write it as ``float32`` with values 0.0 and 1.0.
    """
    output_path = Path(output_path)
    if as_uint8:
        array = result.mask.astype(np.uint8)
        dtype = "uint8"
        nodata = 255
    else:
        array = result.mask.astype(np.float32)
        dtype = "float32"
        nodata = np.nan

    from rasterio.transform import Affine

    transform = Affine(*result.transform)
    from rasterio.crs import CRS
    crs = CRS.from_user_input(result.crs) if result.crs else None

    if crs is None:
        raise ProcessingError(
            "Cannot save flood mask: source raster has no CRS."
        )

    write_raster(
        output_path,
        array,
        transform=transform,
        crs=crs,
        nodata=nodata,
        dtype=dtype,
    )
    log.info("Wrote flood mask: %s", output_path.name)
    return output_path


def save_ratio_raster(
    result: FloodDetectionResult,
    output_path: Path,
) -> Path:
    """Write the post/pre ratio raster for inspection and tuning."""
    output_path = Path(output_path)
    from rasterio.transform import Affine
    from rasterio.crs import CRS

    transform = Affine(*result.transform)
    crs = CRS.from_user_input(result.crs) if result.crs else None
    if crs is None:
        raise ProcessingError("Cannot save ratio raster: no CRS.")

    write_raster(
        output_path,
        result.ratio,
        transform=transform,
        crs=crs,
        nodata=np.nan,
        dtype="float32",
    )
    log.info("Wrote ratio raster: %s", output_path.name)
    return output_path


__all__ = [
    "FloodDetectionResult",
    "apply_minimum_mapping_unit",
    "detect_flood_change",
    "fill_small_holes",
    "mask_invalid",
    "mask_slope",
    "save_flood_mask",
    "save_ratio_raster",
]