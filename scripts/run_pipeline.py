"""End-to-end pipeline runner for the Syapru Besi (EMSR927 AOI01) case study.

Runs the complete Sentinel-1 flood mapping pipeline on the small AOI
that overlaps the EMSR927 reference layer, so the resulting flood
extent can be evaluated against the official Copernicus damage product.

1. Sentinel-1 GRD -> terrain-corrected backscatter (Sarsen)
2. Align pre and post rasters to a common grid
3. Compute slope from the DEM for terrain masking
4. Run flood change detection
5. Save candidate flood extent in km² and hectares

Usage
-----
    python scripts/run_pipeline.py

Prerequisites
-------------
- Two Sentinel-1 products in ``data/raw/sentinel1/`` matching the
  globs PRE_ZIP_GLOB and POST_ZIP_GLOB.
- DEM at ``data/interim/dem_syapru_utm45n.tif`` (download and reproject
  with the commands in the setup guide).
- Sarsen is installed.

Intermediate files are cached in ``data/interim/`` and reused on
re-run, so the script can be safely interrupted and restarted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from georescue.config import get_settings  # noqa: E402
from georescue.logging_config import configure_logging, get_logger  # noqa: E402
from georescue.models import BoundingBox  # noqa: E402
from georescue.processing.flood_detection import (  # noqa: E402
    detect_flood_change,
    save_flood_mask,
    save_ratio_raster,
)
from georescue.processing.preprocessing import (  # noqa: E402
    preprocess_sentinel1_product,
)
from georescue.processing.raster_utils import (  # noqa: E402
    align_raster_to_reference,
    compute_slope_degrees,
    get_raster_info,
    write_raster,
)
from rasterio.enums import Resampling  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Case-study configuration: Syapru Besi (EMSR927 AOI01)
# ---------------------------------------------------------------------------

# Small AOI centered on the EMSR927 AOI01 damage footprint, with margin
# for the Sentinel-1 IW swath and the DEM edge.
AOI = BoundingBox(
    min_lon=85.25,
    min_lat=28.10,
    max_lon=85.42,
    max_lat=28.24,
)

# Sentinel-1 pre-event and post-event zips. Both match the same relative
# orbit (ASCENDING) around the 26 August 2026 event date, so the pair is
# pixel-comparable.
PRE_ZIP_GLOB = "*20260816T122141*20260816T122206*.zip"
POST_ZIP_GLOB = "*20260828T122141*20260828T122206*.zip"

# Flood-detection thresholds. Same as the Trishuli run; adjust here if
# the new AOI needs different tuning (e.g., lower ratio threshold for
# higher-elevation valleys).
RATIO_THRESHOLD = 0.4
MAX_SLOPE_DEG = 15.0
MIN_MMU_PIXELS = 3  # ~0.25 ha at 30 m, appropriate for a small AOI
FILL_HOLE_PIXELS = 4

# Suffix appended to every intermediate and output filename so this run
# does not overwrite the original Trishuli outputs.
SUFFIX = "_syapru"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_unique(pattern: str, base_dir: Path) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise SystemExit(
            f"No file matching {pattern!r} in {base_dir}. "
            f"Run scripts/download_case_study.py first with the Syapru "
            f"AOI configured."
        )
    if len(matches) > 1:
        log.warning(
            "Multiple files match %s; using first: %s",
            pattern, matches[0].name,
        )
    return matches[0]


def _stage(label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    configure_logging("INFO")
    settings = get_settings()

    raw_dir = settings.raw_data_path / "sentinel1"
    interim = settings.interim_data_path
    processed = settings.processed_data_path

    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # --- Locate inputs ---
    try:
        pre_zip = _find_unique(PRE_ZIP_GLOB, raw_dir)
    except SystemExit as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Have you run scripts/download_case_study.py with the "
            "Syapru Besi AOI? The pre-event zip is missing.",
            file=sys.stderr,
        )
        return 2

    try:
        post_zip = _find_unique(POST_ZIP_GLOB, raw_dir)
    except SystemExit as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "The post-event zip is missing. Ensure both files were "
            "downloaded into data/raw/sentinel1/.",
            file=sys.stderr,
        )
        return 2

    dem_path = interim / f"dem_syapru_utm45n.tif"

    print(f"Pre-event product:  {pre_zip.name}")
    print(f"Post-event product: {post_zip.name}")
    print(f"DEM:                {dem_path.name}")

    if not dem_path.is_file():
        print(
            f"\nERROR: DEM not found at {dem_path}.\n"
            f"Run the DEM download and reprojection commands from the "
            f"setup guide before continuing.",
            file=sys.stderr,
        )
        return 2

    # --- Stage 1: terrain correction (pre-event) ---
    pre_tc = interim / f"pre_terrain_corrected{SUFFIX}.tif"
    _stage("Stage 1/6: Terrain correction (pre-event)")
    t0 = time.time()
    if pre_tc.exists():
        print(f"Reusing cached result: {pre_tc.name}")
    else:
        preprocess_sentinel1_product(
            pre_zip,
            AOI,
            dem_path,
            pre_tc,
            measurement_group="IW/VV",
            correct_radiometry=None,
        )
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  output:  {pre_tc.name}")

    # --- Stage 2: terrain correction (post-event) ---
    post_tc = interim / f"post_terrain_corrected{SUFFIX}.tif"
    _stage("Stage 2/6: Terrain correction (post-event)")
    t0 = time.time()
    if post_tc.exists():
        print(f"Reusing cached result: {post_tc.name}")
    else:
        preprocess_sentinel1_product(
            post_zip,
            AOI,
            dem_path,
            post_tc,
            measurement_group="IW/VV",
            correct_radiometry=None,
        )
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  output:  {post_tc.name}")

    # --- Stage 3: align post to pre grid ---
    _stage("Stage 3/6: Align post-event raster to pre-event grid")
    post_aligned = interim / f"post_aligned{SUFFIX}.tif"
    if post_aligned.exists():
        print(f"Reusing cached result: {post_aligned.name}")
    else:
        align_raster_to_reference(
            post_tc,
            pre_tc,
            post_aligned,
            resampling=Resampling.bilinear,
        )
    info_pre = get_raster_info(pre_tc)
    info_post = get_raster_info(post_aligned)
    print(f"  pre  grid: {info_pre['height']} x {info_pre['width']}")
    print(f"  post grid: {info_post['height']} x {info_post['width']}")
    if (
        info_pre["height"] != info_post["height"]
        or info_pre["width"] != info_post["width"]
    ):
        print(
            "  ERROR: pre and post rasters are not on the same grid.",
            file=sys.stderr,
        )
        return 3
    print("  grids match ✓")

    # --- Stage 4: compute slope from DEM ---
    _stage("Stage 4/6: Compute terrain slope from DEM")
    slope_path = interim / f"slope_aoi{SUFFIX}.tif"
    if slope_path.exists():
        print(f"Reusing cached result: {slope_path.name}")
    else:
        slope, slope_profile = compute_slope_degrees(dem_path)
        write_raster(
            slope_path,
            slope,
            transform=slope_profile["transform"],
            crs=slope_profile["crs"],
            nodata=float("nan"),
            dtype="float32",
        )
    print(f"  output: {slope_path.name}")

    # --- Stage 5: align slope to SAR grid ---
    _stage("Stage 5/6: Align slope to SAR grid")
    slope_aligned = interim / f"slope_aligned{SUFFIX}.tif"
    if slope_aligned.exists():
        print(f"Reusing cached result: {slope_aligned.name}")
    else:
        align_raster_to_reference(
            slope_path,
            pre_tc,
            slope_aligned,
            resampling=Resampling.bilinear,
        )
    print(f"  output: {slope_aligned.name}")

    # --- Stage 6: flood detection ---
    _stage("Stage 6/6: Flood change detection")
    result = detect_flood_change(
        pre_tc,
        post_aligned,
        ratio_threshold=RATIO_THRESHOLD,
        max_slope_deg=MAX_SLOPE_DEG,
        slope_path=slope_aligned,
        min_mmu_pixels=MIN_MMU_PIXELS,
        fill_hole_pixels=FILL_HOLE_PIXELS,
    )

    # --- Save outputs ---
    mask_path = processed / f"candidate_flood_mask{SUFFIX}.tif"
    ratio_path = processed / f"backscatter_ratio{SUFFIX}.tif"
    save_flood_mask(result, mask_path)
    save_ratio_raster(result, ratio_path)

    # --- Report ---
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"  Candidate flood area:  {result.area_sqkm:.4f} km²")
    print(f"  Candidate flood area:  {result.area_hectares:.2f} ha")
    print(f"  Pixel count:           {result.pixel_count}")
    print(f"  Pixel size:            {result.pixel_size_m:.2f} m")
    print(f"  Valid pixel fraction:  {result.valid_fraction * 100:.2f}%")
    print(f"  CRS:                   {result.crs}")

    if result.warnings:
        print("\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    print(f"\n  Outputs:")
    print(f"    mask:  {mask_path.relative_to(settings.project_root)}")
    print(f"    ratio: {ratio_path.relative_to(settings.project_root)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())