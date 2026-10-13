"""Tests for the flood-detection module.

Exercises:

- The slope mask helper on synthetic slope rasters.
- The minimum-mapping-unit cleanup on synthetic masks.
- The hole-fill helper on synthetic masks.
- End-to-end ratio-based change detection on a synthetic pre/post pair
  with a known 10×10 flood block.

All tests use only the fixtures defined in ``conftest.py``. No network,
no credentials, no satellite imagery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from georescue.processing.flood_detection import (
    apply_minimum_mapping_unit,
    detect_flood_change,
    fill_small_holes,
    mask_invalid,
    mask_slope,
)


# ---------------------------------------------------------------------------
# mask_slope
# ---------------------------------------------------------------------------

def test_mask_slope_keeps_flat_pixels():
    slope = np.array([[0.0, 1.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    mask = mask_slope(slope, max_slope_deg=5.0)
    assert mask.dtype == bool
    assert mask.all(), "All values <= 5° should pass the mask"


def test_mask_slope_removes_steep_pixels():
    slope = np.array([[0.0, 10.0], [15.0, 3.0]], dtype=np.float32)
    mask = mask_slope(slope, max_slope_deg=5.0)
    assert mask[0, 0]
    assert not mask[0, 1], "10° should be removed"
    assert not mask[1, 0], "15° should be removed"
    assert mask[1, 1]


def test_mask_slope_ignores_nan():
    slope = np.array([[0.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    mask = mask_slope(slope, max_slope_deg=5.0)
    assert mask[0, 0]
    assert not mask[0, 1], "NaN should be treated as invalid"


def test_mask_slope_disabled_returns_all_true():
    """max_slope_deg <= 0 disables the mask (all valid pixels kept)."""
    slope = np.array([[0.0, 45.0], [80.0, 90.0]], dtype=np.float32)
    mask = mask_slope(slope, max_slope_deg=0.0)
    assert mask.all()


# ---------------------------------------------------------------------------
# mask_invalid
# ---------------------------------------------------------------------------

def test_mask_invalid_requires_finite_in_all_arrays():
    a = np.array([[1.0, 2.0], [np.nan, 4.0]])
    b = np.array([[1.0, np.nan], [3.0, 4.0]])
    mask = mask_invalid(a, b)
    assert mask[0, 0]  # both finite
    assert not mask[0, 1]  # b is NaN
    assert not mask[1, 0]  # a is NaN
    assert mask[1, 1]


# ---------------------------------------------------------------------------
# apply_minimum_mapping_unit
# ---------------------------------------------------------------------------

def test_mmu_removes_single_pixel_components():
    """A 3×3 mask with one isolated pixel should drop it when MMU > 1."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True  # isolated
    mask[5:8, 5:8] = True  # 3×3 = 9 pixels
    out = apply_minimum_mapping_unit(mask, min_pixels=5)
    assert not out[0, 0], "isolated pixel should be removed"
    assert out[5:8, 5:8].all()


def test_mmu_keeps_large_components():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True  # 10×10 = 100 pixels
    out = apply_minimum_mapping_unit(mask, min_pixels=5)
    assert out.sum() == 100


def test_mmu_no_op_when_min_pixels_is_one():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    out = apply_minimum_mapping_unit(mask, min_pixels=1)
    assert out.sum() == 1, "MMU=1 should keep single pixels"


# ---------------------------------------------------------------------------
# fill_small_holes
# ---------------------------------------------------------------------------

def test_fill_small_holes_fills_interior():
    """A hole fully enclosed by a solid ring should be filled."""
    mask = np.ones((10, 10), dtype=bool)
    mask[5, 5] = False
    out = fill_small_holes(mask, max_hole_pixels=5)
    assert out[5, 5], "single-pixel hole should be filled"
    assert out.sum() == 100


def test_fill_small_holes_leaves_large_holes():
    """A 4×4 hole (16 px) should not be filled when the limit is 5 px."""
    mask = np.ones((20, 20), dtype=bool)
    mask[8:12, 8:12] = False
    out = fill_small_holes(mask, max_hole_pixels=5)
    assert not out[8:12, 8:12].any(), "large hole should remain"


def test_fill_small_holes_no_holes_returns_unchanged():
    mask = np.ones((5, 5), dtype=bool)
    out = fill_small_holes(mask, max_hole_pixels=4)
    assert out.sum() == mask.sum()


# ---------------------------------------------------------------------------
# detect_flood_change — full pipeline on synthetic data
# ---------------------------------------------------------------------------

def test_detect_flood_change_finds_expected_block(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
):
    """The synthetic post raster has a 10×10 block at ratio 0.2.

    With threshold 0.4, no median filter, no slope mask, and no MMU,
    we expect exactly 100 flood pixels inside the block.

    Area calculation:
        100 pixels × (30 m × 30 m) = 90,000 m² = 0.09 km² = 9 ha
    """
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.4,
        max_slope_deg=None,          # disable slope mask
        min_mmu_pixels=1,            # disable MMU
        fill_hole_pixels=0,          # disable hole fill
        apply_median_filter=False,   # important: get the exact block
    )
    assert result.pixel_count == 100, (
        f"Expected 100 pixels, got {result.pixel_count}"
    )
    # 100 pixels × 900 m²/pixel = 90,000 m² = 0.09 km²
    assert result.area_sqkm == pytest.approx(0.09, rel=0.01), (
        f"Expected 0.09 km², got {result.area_sqkm}"
    )
    assert result.area_hectares == pytest.approx(9.0, rel=0.01), (
        f"Expected 9.0 ha, got {result.area_hectares}"
    )


def test_detect_flood_change_with_median_filter_reduces_edges(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
):
    """With the median filter enabled, edges of the flood block are smoothed.

    The exact count varies slightly with filter implementation, but it
    must be less than or equal to the unfiltered 100.
    """
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.4,
        max_slope_deg=None,
        min_mmu_pixels=1,
        fill_hole_pixels=0,
        apply_median_filter=True,
    )
    assert 0 < result.pixel_count <= 100, (
        f"Filtered count should be in (0, 100], got {result.pixel_count}"
    )


def test_detect_flood_change_slope_mask_removes_block(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
    synthetic_steep_slope: Path,
):
    """With a steep-slope raster, the entire flood block is masked out."""
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.4,
        max_slope_deg=5.0,
        slope_path=synthetic_steep_slope,
        min_mmu_pixels=1,
        fill_hole_pixels=0,
        apply_median_filter=False,
    )
    assert result.pixel_count == 0, (
        f"Steep slope should remove all flood pixels, got {result.pixel_count}"
    )


def test_detect_flood_change_slope_mask_preserves_flat_block(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
    synthetic_partial_slope: Path,
):
    """Only the flood block is flat in the partial slope raster.

    All the pixels inside the block should survive the slope mask.
    """
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.4,
        max_slope_deg=5.0,
        slope_path=synthetic_partial_slope,
        min_mmu_pixels=1,
        fill_hole_pixels=0,
        apply_median_filter=False,
    )
    assert result.pixel_count == 100


def test_detect_flood_change_threshold_too_strict(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
):
    """A very low ratio threshold (0.05) should detect nothing.

    The block ratio is 0.2, so it is above 0.05.
    """
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.05,
        max_slope_deg=None,
        min_mmu_pixels=1,
        fill_hole_pixels=0,
        apply_median_filter=False,
    )
    assert result.pixel_count == 0


def test_detect_flood_change_result_shape(
    synthetic_pre_raster: Path,
    synthetic_post_raster: Path,
):
    """The result mask and ratio arrays match the input shape."""
    result = detect_flood_change(
        synthetic_pre_raster,
        synthetic_post_raster,
        ratio_threshold=0.4,
        max_slope_deg=None,
        min_mmu_pixels=1,
        fill_hole_pixels=0,
        apply_median_filter=False,
    )
    assert result.mask.shape == (50, 50)
    assert result.ratio.shape == (50, 50)
    assert result.crs == "EPSG:32645"
    assert result.pixel_size_m == pytest.approx(30.0, abs=0.01)