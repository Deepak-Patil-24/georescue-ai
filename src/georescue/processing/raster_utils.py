"""Low-level raster utilities for GeoRescue AI.

This module provides the primitives that every other processing module
builds on. It is deliberately dependency-light: only ``rasterio``,
``rioxarray``, ``numpy``, ``pyproj``, and ``shapely`` are required at
import time.

Design rules
------------
- Every function that opens a raster uses a context manager so file
  handles are released on error.
- Nodata is preserved explicitly. A missing value is never silently
  converted to zero.
- CRS mismatches are handled by :func:`align_raster_to_reference`, which
  reprojects the source onto the reference grid using bilinear resampling
  for continuous data. Callers choose nearest-neighbour for categorical
  data.
- Area and length are computed in a projected CRS, never in degrees.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from georescue.config import get_settings
from georescue.logging_config import get_logger
from georescue.models import BoundingBox
from georescue.processing import ProcessingError, RasterAlignmentError

log = get_logger(__name__)

# Default resampling methods for the two data classes we handle.
_CONTINUOUS_RESAMPLING = Resampling.bilinear
_CATEGORICAL_RESAMPLING = Resampling.nearest


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def utm_epsg_for_bbox(bbox: BoundingBox) -> int:
    """Return the EPSG code of the UTM zone containing the bbox centroid.

    Falls back to the configured fallback CRS if the bbox is outside the
    UTM validity range (-80 to 84 degrees latitude).
    """
    centroid_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    centroid_lat = (bbox.min_lat + bbox.max_lat) / 2.0

    if not (-80.0 <= centroid_lat <= 84.0):
        fallback = get_settings().fallback_projected_crs
        log.warning(
            "Latitude %.4f is outside the UTM range; using fallback CRS %s.",
            centroid_lat, fallback,
        )
        return CRS.from_user_input(fallback).to_epsg()

    zone = int(math.floor((centroid_lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return 32600 + zone if centroid_lat >= 0 else 32700 + zone


def utm_crs_for_bbox(bbox: BoundingBox) -> CRS:
    """Return the UTM CRS for a bbox as a :class:`rasterio.crs.CRS`."""
    return CRS.from_epsg(utm_epsg_for_bbox(bbox))


def bbox_in_crs(bbox: BoundingBox, dst_crs: CRS) -> tuple[float, float, float, float]:
    """Reproject a WGS84 bbox into ``dst_crs`` and return axis-aligned bounds.

    Returns ``(min_x, min_y, max_x, max_y)`` in the target CRS, expressed
    in that CRS's units (metres for UTM, feet for some state-plane CRSes,
    etc.). This is **not** a :class:`BoundingBox`, because that type is
    reserved for WGS84 lon/lat coordinates.

    The result is the axis-aligned bounding box of the reprojected
    corners. It is slightly larger than the true reprojected footprint,
    which is safe for clipping (we never lose data).
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), dst_crs, always_xy=True,
    )
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
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Raster metadata
# ---------------------------------------------------------------------------

def get_raster_info(path: Path) -> dict:
    """Return basic metadata for a raster without reading pixel data."""
    path = Path(path)
    if not path.exists():
        raise ProcessingError(f"Raster not found: {path}")
    try:
        with rasterio.open(path) as src:
            return {
                "path": str(path),
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform)[:6],
                "bounds": tuple(src.bounds),
                "nodata": src.nodata,
                "res": src.res,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise ProcessingError(
            f"Could not open raster {path}: {exc}. "
            f"The file may be corrupt or in an unsupported format."
        ) from exc


def read_raster(
    path: Path,
    *,
    band: int = 1,
    window: Optional[Window] = None,
) -> Tuple[np.ndarray, dict]:
    """Read a single band from a raster.

    Returns ``(array, profile)`` where ``profile`` is a dict containing
    ``transform``, ``crs``, ``nodata``, and ``res``. The array is a 2-D
    ``float32`` array with nodata values preserved as ``np.nan``.
    """
    path = Path(path)
    if not path.exists():
        raise ProcessingError(f"Raster not found: {path}")

    try:
        with rasterio.open(path) as src:
            if band < 1 or band > src.count:
                raise ProcessingError(
                    f"Band {band} requested but {path.name} has {src.count} band(s)."
                )
            data = src.read(band, window=window).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            profile = {
                "transform": src.transform,
                "crs": src.crs,
                "nodata": nodata,
                "res": src.res,
                "width": src.width,
                "height": src.height,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise ProcessingError(f"Could not read raster {path}: {exc}") from exc

    return data, profile


def write_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform: Affine,
    crs: CRS,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "deflate",
) -> Path:
    """Write a 2-D array as a single-band GeoTIFF.

    Uses atomic write (writes to ``.part`` then renames) so a crash does
    not leave a partially written file that looks valid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim != 2:
        raise ProcessingError(
            f"write_raster expects a 2-D array, got shape {array.shape}."
        )

    temp_path = path.with_suffix(path.suffix + ".part")
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "compress": compress,
    }
    try:
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
        temp_path.replace(path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise ProcessingError(f"Could not write raster {path}: {exc}") from exc

    return path


# ---------------------------------------------------------------------------
# Clipping and alignment
# ---------------------------------------------------------------------------

def clip_raster_to_bbox(
    src_path: Path,
    dst_path: Path,
    bbox: BoundingBox,
    *,
    nodata: float = np.nan,
) -> Path:
    """Clip a raster to a bbox and write the result.

    The bbox is interpreted in the raster's own CRS. Callers that have a
    WGS84 bbox should reproject it with :func:`bbox_in_crs` first.
    """
    src_path = Path(src_path)
    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise ProcessingError(
                    f"Raster {src_path} has no CRS; cannot clip safely."
                )
            geom = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
            try:
                out_image, out_transform = rio_mask(
                    src, [geom.__geo_interface__], crop=True, nodata=nodata,
                )
            except ValueError as exc:
                raise ProcessingError(
                    f"Bbox does not overlap raster {src_path.name}: {exc}"
                ) from exc

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": nodata,
                "compress": "deflate",
            })
    except rasterio.errors.RasterioIOError as exc:
        raise ProcessingError(f"Could not open {src_path}: {exc}") from exc

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    with rasterio.open(temp_path, "w", **out_meta) as dst:
        dst.write(out_image)
    temp_path.replace(dst_path)
    return dst_path


def align_raster_to_reference(
    src_path: Path,
    reference_path: Path,
    dst_path: Path,
    *,
    resampling: Resampling = _CONTINUOUS_RESAMPLING,
    nodata: float = np.nan,
) -> Path:
    """Reproject and resample ``src_path`` onto the grid of ``reference_path``.

    After this call, the two rasters share CRS, transform, width, and
    height, so pixel ``(i, j)`` in one corresponds to the same ground
    location as pixel ``(i, j)`` in the other. This is a prerequisite for
    pixel-wise change detection.

    Parameters
    ----------
    resampling:
        ``Resampling.bilinear`` (default) for continuous data such as
        backscatter. Use ``Resampling.nearest`` for categorical data such
        as land-cover masks.
    """
    src_path = Path(src_path)
    reference_path = Path(reference_path)
    dst_path = Path(dst_path)

    try:
        with rasterio.open(reference_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_nodata = ref.nodata
    except rasterio.errors.RasterioIOError as exc:
        raise RasterAlignmentError(
            f"Could not open reference raster {reference_path}: {exc}"
        ) from exc

    if ref_crs is None:
        raise RasterAlignmentError(
            f"Reference raster {reference_path.name} has no CRS."
        )

    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise RasterAlignmentError(
                    f"Source raster {src_path.name} has no CRS."
                )

            out_array = np.full(
                (1, ref_height, ref_width), nodata, dtype=np.float32,
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=out_array[0],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=nodata,
                resampling=resampling,
            )
    except rasterio.errors.RasterioIOError as exc:
        raise RasterAlignmentError(f"Could not open {src_path}: {exc}") from exc

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": ref_height,
        "width": ref_width,
        "transform": ref_transform,
        "crs": ref_crs,
        "nodata": nodata,
        "compress": "deflate",
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(out_array)
    temp_path.replace(dst_path)
    return dst_path


# ---------------------------------------------------------------------------
# Statistics and area
# ---------------------------------------------------------------------------

def pixel_area_sqkm(transform: Affine, crs: CRS) -> float:
    """Return the area of one pixel in square kilometres.

    If ``crs`` is projected, the area is ``|a * e - b * d| / 1e6`` where
    ``a, b, d, e`` are the linear parts of the affine transform. If the
    CRS is geographic (degrees), the pixel area is approximated at the
    raster centre — this is only used for sanity checks, never for final
    area reporting.
    """
    a, b, _c, d, e, _f = transform[:6]
    if crs.is_projected:
        return abs(a * e - b * d) / 1_000_000.0

    # Geographic CRS: approximate using the midpoint latitude.
    centre_lat = _f + e * 0.5 if e else 0.0
    if centre_lat == 0.0:
        # Fall back to the transform origin's y.
        centre_lat = _f
    lat_rad = math.radians(centre_lat)
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    return abs(a * metres_per_deg_lon) * abs(e * metres_per_deg_lat) / 1_000_000.0


def mask_area_sqkm(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> float:
    """Return the area of ``True`` pixels in a boolean mask, in km².

    For projected CRSes this is exact (up to pixel discretisation). For
    geographic CRSes the result is an approximation; callers that need
    accuracy should reproject the mask to a projected CRS first.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return float(mask.sum()) * pixel_area_sqkm(transform, crs)


def geometry_area_sqkm(
    geom: BaseGeometry,
    crs: CRS,
) -> float:
    """Return the area of a Shapely geometry in km².

    The geometry must already be in a projected CRS. If it is in a
    geographic CRS, this raises :class:`ProcessingError` — callers must
    reproject first, because degree-based areas are meaningless.
    """
    if crs.is_geographic:
        raise ProcessingError(
            "geometry_area_sqkm requires a projected CRS. "
            "Reproject the geometry to UTM (or another equal-area CRS) first."
        )
    return float(geom.area) / 1_000_000.0


def valid_pixel_fraction(array: np.ndarray) -> float:
    """Return the fraction of finite (non-nodata) pixels in an array."""
    if array.size == 0:
        return 0.0
    finite = np.isfinite(array)
    return float(finite.sum()) / float(array.size)


# ---------------------------------------------------------------------------
# Speckle filtering
# ---------------------------------------------------------------------------

def median_filter_nodata_safe(
    array: np.ndarray,
    size: int = 3,
) -> np.ndarray:
    """Apply a median filter that preserves NaN nodata regions.

    Uses :func:`scipy.ndimage.median_filter` on a copy where NaNs are
    temporarily replaced with a large finite value, then restores NaN in
    the original nodata positions.
    """
    if size < 3 or size % 2 == 0:
        raise ProcessingError(f"median filter size must be odd and >= 3, got {size}")

    from scipy.ndimage import median_filter

    nan_mask = ~np.isfinite(array)
    if not nan_mask.any():
        return median_filter(array, size=size).astype(np.float32)

    # Replace NaN with a sentinel that is unlikely to appear in real data.
    sentinel = np.nanmax(array[np.isfinite(array)]) if np.isfinite(array).any() else 0.0
    sentinel = max(sentinel * 100.0, 1e6)
    filled = np.where(nan_mask, sentinel, array)
    filtered = median_filter(filled, size=size)
    filtered[nan_mask] = np.nan
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Slope from DEM (used for terrain masking)
# ---------------------------------------------------------------------------

def compute_slope_degrees(dem_path: Path) -> Tuple[np.ndarray, dict]:
    """Compute terrain slope in degrees from a DEM raster.

    Uses the Horn method via :func:`numpy.gradient`. Returns the slope
    array and the source raster's profile. Slope is computed on the DEM's
    own grid; the caller aligns it to the SAR grid if needed.
    """
    dem, profile = read_raster(dem_path)
    transform = profile["transform"]
    crs = profile["crs"]

    if crs is None:
        raise ProcessingError(f"DEM {dem_path} has no CRS; cannot compute slope.")

    if not crs.is_projected:
        raise ProcessingError(
            "compute_slope_degrees requires a DEM in a projected CRS "
            "with metre units. Reproject the DEM first."
        )

    # Replace NaN with a local median so gradients are not polluted.
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    dy, dx = np.gradient(filled)
    dx /= abs(transform.a) or 1.0
    dy /= abs(transform.e) or 1.0
    slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[~np.isfinite(dem)] = np.nan
    return slope_deg, profile


__all__ = [
    "align_raster_to_reference",
    "bbox_in_crs",
    "clip_raster_to_bbox",
    "compute_slope_degrees",
    "geometry_area_sqkm",
    "get_raster_info",
    "mask_area_sqkm",
    "median_filter_nodata_safe",
    "pixel_area_sqkm",
    "read_raster",
    "utm_crs_for_bbox",
    "utm_epsg_for_bbox",
    "valid_pixel_fraction",
    "write_raster",
]