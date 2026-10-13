"""Sentinel-1 GRD preprocessing for GeoRescue AI.

This module converts a downloaded Sentinel-1 Ground Range Detected (GRD)
product into an analysis-ready backscatter raster on a fixed target grid,
using `Sarsen <https://github.com/bopen/sarsen>`_ for geometric terrain
correction (GTC).

Pipeline
--------
1. Open the Sentinel-1 product with :class:`sarsen.Sentinel1SarProduct`.
2. Run geometric terrain correction against a DEM that has already been
   resampled to the target projected CRS and pixel size.
3. Clip the resulting raster to the AOI.
4. Write the result as a single-band GeoTIFF.

The output raster is aligned to the DEM's grid, which we will generate in
:mod:`georescue.data_access.dem`. That means the pre-event and post-event
raster share the same transform, CRS, width, and height, so pixel-wise
change detection is valid.

Sarsen reads Sentinel-1 GRD products from a compressed ``.zip`` or an
uncompressed ``.SAFE`` directory [citation:1]. We pass the ``.zip`` path
directly.

Reference
---------
- Sarsen documentation: https://sarsen.readthedocs.io/
- Sarsen GitHub: https://github.com/bopen/sarsen
- Sarsen feature list: https://pypi.org/project/sarsen/ [citation:6]
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from georescue.logging_config import get_logger
from georescue.models import BoundingBox
from georescue.processing import ProcessingError
from georescue.processing.raster_utils import (
    bbox_in_crs,
    utm_crs_for_bbox,
    write_raster,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Product extraction
# ---------------------------------------------------------------------------

def extract_safe_from_zip(zip_path: Path, work_dir: Path) -> Path:
    """Extract a Sentinel-1 ``.zip`` archive and return the ``.SAFE`` folder.

    Sarsen can read compressed SAFE products directly [citation:1], but
    extracting once and reusing the folder is faster when we process the
    same product more than once (for example, both VV and VH). The
    extracted folder is left in ``work_dir`` so subsequent calls reuse it.

    Parameters
    ----------
    zip_path:
        Path to the downloaded Sentinel-1 ``.zip`` file.
    work_dir:
        Directory to extract into. Created if missing.

    Returns
    -------
    Path to the extracted ``.SAFE`` directory.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise ProcessingError(f"Sentinel-1 zip not found: {zip_path}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Sentinel-1 product archives contain a single top-level .SAFE folder.
    safe_name = zip_path.stem  # e.g. S1D_IW_GRDH_1SDV_...SAFE
    safe_dir = work_dir / safe_name

    if safe_dir.is_dir() and (safe_dir / "manifest.safe").is_file():
        log.info("Reusing extracted SAFE folder: %s", safe_dir.name)
        return safe_dir

    log.info("Extracting %s -> %s", zip_path.name, work_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(work_dir)
    except zipfile.BadZipFile as exc:
        raise ProcessingError(
            f"Sentinel-1 archive is corrupt: {zip_path.name}: {exc}"
        ) from exc
    except OSError as exc:
        raise ProcessingError(
            f"Could not extract {zip_path.name}: {exc}. "
            f"Check free disk space in {work_dir}."
        ) from exc

    if not safe_dir.is_dir():
        # Some archives nest the SAFE folder one level deeper.
        candidates = list(work_dir.glob("*.SAFE"))
        if len(candidates) == 1:
            safe_dir = candidates[0]
        else:
            raise ProcessingError(
                f"Could not find a .SAFE folder inside {zip_path.name}. "
                f"Found: {[p.name for p in candidates]}"
            )

    if not (safe_dir / "manifest.safe").is_file():
        raise ProcessingError(
            f"Extracted folder {safe_dir} has no manifest.safe; "
            f"the archive may be incomplete."
        )

    log.info("Extracted SAFE folder: %s", safe_dir.name)
    return safe_dir


# ---------------------------------------------------------------------------
# Terrain correction
# ---------------------------------------------------------------------------

def terrain_correct_grd(
    safe_dir: Path,
    dem_path: Path,
    output_path: Path,
    *,
    measurement_group: str = "IW/VV",
    correct_radiometry: Optional[str] = None,
) -> Path:
    """Apply geometric terrain correction to a Sentinel-1 GRD product.

    Parameters
    ----------
    safe_dir:
        Path to an uncompressed ``.SAFE`` directory.
    dem_path:
        Path to a DEM raster already resampled to the target projected
        CRS and pixel size. Sarsen uses the DEM's grid for the output.
    output_path:
        Where to write the terrain-corrected GeoTIFF.
    measurement_group:
        Sarsen measurement group selector, e.g. ``"IW/VV"`` or ``"IW/VH"``
        [citation:6].
    correct_radiometry:
        ``None`` for geometric terrain correction (GTC) only. Pass
        ``"gamma_nearest"`` to also apply radiometric terrain correction
        (RTC) [citation:6]. RTC is more correct for quantitative
        backscatter comparison but is roughly 5x slower.

    Returns
    -------
    Path to the written GeoTIFF.
    """
    try:
        import sarsen
    except ImportError as exc:
        raise ProcessingError(
            "Sarsen is required for Sentinel-1 preprocessing but is not "
            "installed. Run: pip install sarsen"
        ) from exc

    safe_dir = Path(safe_dir)
    dem_path = Path(dem_path)
    output_path = Path(output_path)

    if not safe_dir.is_dir():
        raise ProcessingError(f"SAFE folder not found: {safe_dir}")
    if not dem_path.is_file():
        raise ProcessingError(f"DEM not found: {dem_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sentinel-1 GRD measurement groups are exposed as "IW/VV" and "IW/VH".
    log.info(
        "Terrain correction: product=%s group=%s dem=%s -> %s",
        safe_dir.name, measurement_group, dem_path.name, output_path.name,
    )

    try:
        product = sarsen.Sentinel1SarProduct(
            str(safe_dir),
            measurement_group=measurement_group,
        )
    except Exception as exc:
        raise ProcessingError(
            f"Sarsen could not open {safe_dir.name} with measurement "
            f"group '{measurement_group}': {exc}"
        ) from exc

    try:
        # ``correct_radiometry=None`` performs GTC only.
        # ``correct_radiometry="gamma_nearest"`` performs RTC.
        corrected = sarsen.terrain_correction(
            product,
            dem_urlpath=str(dem_path),
            correct_radiometry=correct_radiometry,
        )
    except Exception as exc:
        raise ProcessingError(
            f"Sarsen terrain correction failed for {safe_dir.name}: {exc}"
        ) from exc

    # ``corrected`` is an xarray.DataArray. Write it via rioxarray so the
    # CRS and transform are preserved. We do not assume rioxarray is
    # importable at module load; it is a runtime requirement here only.
    try:
        import rioxarray  # noqa: F401
    except ImportError as exc:
        raise ProcessingError(
            "rioxarray is required to write Sarsen output but is not "
            "installed. Run: pip install rioxarray"
        ) from exc

    # Use a temp filename that preserves the .tif extension, because
    # rasterio refuses to write to an unknown extension.
    temp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    try:
        corrected.rio.to_raster(str(temp_path), dtype="float32")
        temp_path.replace(output_path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise ProcessingError(
            f"Could not write terrain-corrected raster {output_path}: {exc}"
        ) from exc

    log.info(
        "Terrain correction complete: %s (%.1f MB)",
        output_path.name,
        output_path.stat().st_size / (1024 * 1024),
    )
    return output_path


# ---------------------------------------------------------------------------
# Clipping to AOI
# ---------------------------------------------------------------------------

def clip_to_aoi(
    src_path: Path,
    dst_path: Path,
    aoi_bbox_wgs84: BoundingBox,
    *,
    nodata: float = np.nan,
) -> Path:
    """Clip a raster to an AOI expressed in WGS84.

    The AOI bbox is reprojected into the raster's own CRS before clipping,
    so the caller always passes the AOI in EPSG:4326 regardless of the
    raster CRS. See [citation:10] for the ``transform_bounds`` pattern
    used by rasterio users.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise ProcessingError(
                    f"Raster {src_path.name} has no CRS; cannot clip."
                )

            # Reproject the AOI bbox into the raster's CRS.
            min_x, min_y, max_x, max_y = bbox_in_crs(aoi_bbox_wgs84, src.crs)
            geom = box(min_x, min_y, max_x, max_y)

            try:
                out_image, out_transform = rio_mask(
                    src,
                    [geom.__geo_interface__],
                    crop=True,
                    nodata=nodata,
                )
            except ValueError as exc:
                raise ProcessingError(
                    f"AOI does not overlap raster {src_path.name}: {exc}"
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

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    try:
        with rasterio.open(temp_path, "w", **out_meta) as dst:
            dst.write(out_image)
        temp_path.replace(dst_path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise ProcessingError(
            f"Could not write clipped raster {dst_path}: {exc}"
        ) from exc

    log.info(
        "Clipped to AOI: %s (%d x %d pixels)",
        dst_path.name, out_meta["width"], out_meta["height"],
    )
    return dst_path


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def preprocess_sentinel1_product(
    zip_path: Path,
    aoi_bbox_wgs84: BoundingBox,
    dem_path: Path,
    output_path: Path,
    *,
    measurement_group: str = "IW/VV",
    correct_radiometry: Optional[str] = None,
    work_dir: Optional[Path] = None,
) -> Path:
    """End-to-end preprocessing of a Sentinel-1 GRD product.

    Steps
    -----
    1. Extract the SAFE folder from the ``.zip`` (if not already extracted).
    2. Apply terrain correction using the provided DEM.
    3. Clip the result to the AOI.
    4. Write the clipped raster to ``output_path``.

    Parameters
    ----------
    zip_path:
        Downloaded Sentinel-1 ``.zip`` file.
    aoi_bbox_wgs84:
        Area of interest in WGS84.
    dem_path:
        DEM raster already resampled to the target projected CRS.
    output_path:
        Where to write the final analysis-ready GeoTIFF.
    measurement_group:
        ``"IW/VV"`` (default) or ``"IW/VH"``.
    correct_radiometry:
        ``None`` for GTC; ``"gamma_nearest"`` for RTC.
    work_dir:
        Scratch directory for SAFE extraction. Defaults to a sibling
        ``_extracted`` folder next to the ``.zip``.

    Returns
    -------
    Path to the final clipped raster.
    """
    zip_path = Path(zip_path)
    if work_dir is None:
        work_dir = zip_path.parent / "_extracted"
    work_dir = Path(work_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: extract SAFE.
    safe_dir = extract_safe_from_zip(zip_path, work_dir)

    # Step 2: terrain correction into an intermediate GeoTIFF.
    intermediate = output_path.parent / (output_path.stem + "_full.tif")
    if intermediate.exists():
        log.info("Reusing cached terrain-corrected raster: %s", intermediate.name)
    else:
        terrain_correct_grd(
            safe_dir,
            dem_path,
            intermediate,
            measurement_group=measurement_group,
            correct_radiometry=correct_radiometry,
        )

    # Step 3: clip to AOI.
    clip_to_aoi(intermediate, output_path, aoi_bbox_wgs84)

    # Do not delete the intermediate: it is useful for debugging and
    # re-running downstream steps without re-running Sarsen.
    return output_path


__all__ = [
    "clip_to_aoi",
    "extract_safe_from_zip",
    "preprocess_sentinel1_product",
    "terrain_correct_grd",
]