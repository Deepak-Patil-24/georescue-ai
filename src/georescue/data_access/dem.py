"""Copernicus DEM 30 m (WorldDEM-30) access from the AWS Open Data bucket.

The Copernicus DEM 30 m dataset is published on AWS Open Data as a public
S3 bucket. Each 1° × 1° tile is a Cloud-Optimized GeoTIFF with a
deterministic naming convention:

    Copernicus_DSM_COG_10_<N|S><lat:02d>_00_<E|W><lon:03d>_00_DEM/
        Copernicus_DSM_COG_10_<N|S><lat:02d>_00_<E|W><lon:03d>_00_DEM.tif

Example for 27°N, 85°E:
    Copernicus_DSM_COG_10_N27_00_E085_00_DEM/
        Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif

Reference: https://registry.opendata.aws/copernicus-dem/

This module downloads only the tiles intersecting the AOI and can
optionally mosaic them into a single GeoTIFF using rasterio. It does not
require credentials.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import requests

from georescue.config import Settings, get_settings
from georescue.data_access import DataAccessError
from georescue.logging_config import get_logger
from georescue.models import BoundingBox

log = get_logger(__name__)

_TIMEOUT = (10, 300)


def _tile_key(lat: int, lon: int) -> str:
    """Return the DEM tile directory/key prefix for integer-degree lat/lon.

    ``lat`` is the *floor* of the tile's south edge, ``lon`` the floor of
    the west edge. Names follow the Copernicus convention:
    ``N27_00`` or ``S03_00`` for latitude, ``E085_00`` or ``W120_00`` for
    longitude.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}_00"
    lon_str = f"{abs(lon):03d}_00"
    base = f"Copernicus_DSM_COG_10_{ns}{lat_str}_{ew}{lon_str}_DEM"
    return f"{base}/{base}.tif"


def _tiles_for_bbox(bbox: BoundingBox) -> list[tuple[int, int]]:
    """Return integer (lat, lon) pairs of 1° tiles covering the bbox."""
    lat_min = int(math.floor(bbox.min_lat))
    lat_max = int(math.floor(bbox.max_lat))
    lon_min = int(math.floor(bbox.min_lon))
    lon_max = int(math.floor(bbox.max_lon))

    tiles: list[tuple[int, int]] = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            # DEM only covers 60°S..60°N in the WorldDEM-30 product line.
            if lat < -60 or lat > 60:
                continue
            tiles.append((lat, lon))
    return tiles


def download_dem_tile(
    lat: int,
    lon: int,
    destination_dir: Path,
    *,
    settings: Optional[Settings] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Download one 1° DEM tile.

    Returns the local path, or ``None`` when the tile does not exist in
    the bucket (common at high latitudes). Missing tiles are not an error;
    the caller decides how to handle incomplete coverage.
    """
    settings = settings or get_settings()
    base_url = settings.copernicus_dem_s3_base.rstrip("/")
    key = _tile_key(lat, lon)
    url = f"{base_url}/{key}"

    destination = Path(destination_dir) / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        log.info("Using cached DEM tile %s.", destination.name)
        return destination

    log.info("Downloading DEM tile %s", destination.name)
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
            if response.status_code == 404:
                log.warning("DEM tile %s not available in the bucket.", key)
                return None
            if response.status_code >= 400:
                raise DataAccessError(
                    f"DEM tile download failed with HTTP {response.status_code} "
                    f"for {key}."
                )
            temp = destination.with_suffix(destination.suffix + ".part")
            try:
                with open(temp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                temp.replace(destination)
            except Exception:
                if temp.exists():
                    try:
                        temp.unlink()
                    except OSError:
                        pass
                raise
    except requests.RequestException as exc:
        raise DataAccessError(
            f"DEM tile download failed for {key}: {exc}"
        ) from exc

    log.info("Downloaded DEM tile %s (%.1f MB).",
             destination.name, destination.stat().st_size / (1024 * 1024))
    return destination


def download_dem_for_bbox(
    bbox: BoundingBox,
    destination_dir: Path,
    *,
    settings: Optional[Settings] = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download all DEM tiles covering ``bbox``.

    Returns the list of successfully downloaded tile paths. Missing tiles
    (404) are skipped with a warning.
    """
    tiles = _tiles_for_bbox(bbox)
    if not tiles:
        log.warning(
            "No Copernicus DEM 30 m tiles intersect the AOI "
            "(WorldDEM-30 covers 60°S to 60°N)."
        )
        return []

    paths: list[Path] = []
    for lat, lon in tiles:
        path = download_dem_tile(
            lat, lon, destination_dir,
            settings=settings, overwrite=overwrite,
        )
        if path is not None:
            paths.append(path)
    return paths


def mosaic_dem_tiles(
    tile_paths: Iterable[Path],
    output_path: Path,
    *,
    bbox: Optional[BoundingBox] = None,
) -> Path:
    """Mosaic DEM tiles into a single GeoTIFF.

    If ``bbox`` is supplied, the mosaic is clipped to that bbox. Uses
    rasterio's merge and, when needed, warp utilities. Requires the
    ``rasterio`` package (a hard dependency of GeoRescue AI).

    Raises :class:`DataAccessError` on any failure so the caller can
    present a clear message.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    from rasterio.windows import from_bounds
    from rasterio.transform import from_bounds as transform_from_bounds

    paths = [Path(p) for p in tile_paths]
    if not paths:
        raise DataAccessError("mosaic_dem_tiles called with no tiles.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sources = [rasterio.open(p) for p in paths]
    except Exception as exc:
        raise DataAccessError(f"Could not open DEM tiles for mosaicking: {exc}") from exc

    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "deflate",
        })

        if bbox is None:
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(mosaic)
            log.info("Wrote DEM mosaic %s (%.1f MB).",
                     output_path.name, output_path.stat().st_size / (1024 * 1024))
            return output_path

        # Clip to bbox. Warp the mosaic into EPSG:4326 first if needed.
        src_crs = sources[0].crs
        if src_crs is None:
            raise DataAccessError("DEM tiles have no CRS; cannot clip safely.")
        wgs84 = CRS.from_epsg(4326)

        if src_crs != wgs84:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, wgs84, mosaic.shape[2], mosaic.shape[1], *rasterio.transform.array_bounds(
                    mosaic.shape[1], mosaic.shape[2], transform
                ),
            )
            warped = mosaic
            warped_transform = dst_transform
            warped_width = dst_width
            warped_height = dst_height
            warped_data = __import__("numpy").empty(
                (mosaic.shape[0], dst_height, dst_width), dtype=mosaic.dtype
            )
            reproject(
                source=mosaic,
                destination=warped_data,
                src_transform=transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=wgs84,
                resampling=Resampling.bilinear,
            )
            mosaic = warped_data
            transform = warped_transform
            width, height = warped_width, warped_height
        else:
            width, height = mosaic.shape[2], mosaic.shape[1]

        window = from_bounds(
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat,
            transform=transform,
        )
        window = window.intersection(
            __import__("rasterio").windows.Window(0, 0, width, height)
        )
        if window.width <= 0 or window.height <= 0:
            raise DataAccessError(
                "Requested bbox does not overlap the DEM mosaic."
            )

        clipped = mosaic[
            :,
            int(window.row_off):int(window.row_off + window.height),
            int(window.col_off):int(window.col_off + window.width),
        ]
        clipped_transform = rasterio.windows.transform(window, transform)

        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform,
            "crs": wgs84,
        })
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(clipped)
        log.info("Wrote clipped DEM %s (%.1f MB).",
                 output_path.name, output_path.stat().st_size / (1024 * 1024))
        return output_path
    finally:
        for src in sources:
            try:
                src.close()
            except Exception:
                pass


__all__ = [
    "download_dem_for_bbox",
    "download_dem_tile",
    "mosaic_dem_tiles",
]