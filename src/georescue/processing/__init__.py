"""Processing layer for GeoRescue AI.

This sub-package transforms raw satellite data into analysis-ready
products. Responsibilities, in pipeline order:

1. :mod:`~georescue.processing.raster_utils`
   Low-level raster helpers: CRS selection, alignment, clipping,
   resampling, nodata handling, area computation.

2. :mod:`~georescue.processing.preprocessing`
   Sentinel-1 GRD -> analysis-ready backscatter via Sarsen. Applies
   geometric terrain correction and (optionally) radiometric terrain
   correction, then aligns the result to the AOI grid.

3. :mod:`~georescue.processing.flood_detection`
   Change detection between pre-event and post-event backscatter rasters,
   producing a candidate flood mask. Includes terrain slope masking and
   optional permanent-water masking.

4. :mod:`~georescue.processing.vectorization`
   Convert the candidate flood mask into georeferenced polygons, compute
   accurate area in a projected CRS, and filter small polygons by minimum
   mapping unit.

No module in this package performs network I/O. Data acquisition is the
responsibility of :mod:`georescue.data_access`.
"""

from __future__ import annotations


class ProcessingError(RuntimeError):
    """Raised when a processing stage cannot complete.

    The message is written to be shown directly to the user. It should
    explain what failed and what the user can do about it.
    """


class RasterAlignmentError(ProcessingError):
    """Raised when two rasters cannot be aligned to a common grid."""


__all__ = [
    "ProcessingError",
    "RasterAlignmentError",
]