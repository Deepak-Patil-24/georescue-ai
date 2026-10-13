"""Tests for the pipeline orchestration utilities.

These tests do not require any credentials, downloads, or network
access. They exercise:

- Run id generation.
- Unique run directory creation.
- The download confirmation gate in ``run_full_analysis``.

The download confirmation test uses a synthetic ``SceneAvailability``
object so the pipeline can be tested without touching the CDSE
catalogue.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from georescue.models import BoundingBox, EventDefinition, SceneMetadata, ScenePair
from georescue.pipeline import (
    DownloadConfirmationRequired,
    SceneAvailability,
    make_run_id,
    make_unique_run_dir,
    run_full_analysis,
)


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

def test_make_run_id_normalizes_name():
    """A human name with spaces and capitals becomes a slug."""
    rid = make_run_id("Trishuli corridor (case study)", date(2026, 8, 26))
    assert rid == "trishuli_corridor_case_study_20260826"


def test_make_run_id_simple_name():
    rid = make_run_id("Koshi", date(2026, 8, 26))
    assert rid == "koshi_20260826"


def test_make_run_id_no_name():
    """Without a name, only the date is used."""
    rid = make_run_id(None, date(2026, 8, 26))
    assert rid == "aoi_20260826"


def test_make_run_id_empty_name():
    rid = make_run_id("", date(2026, 8, 26))
    assert rid == "aoi_20260826"


# ---------------------------------------------------------------------------
# make_unique_run_dir
# ---------------------------------------------------------------------------

def test_make_unique_run_dir_creates_first(tmp_path: Path):
    d = make_unique_run_dir(tmp_path, "test_run")
    assert d.is_dir()
    assert d.name == "test_run"


def test_make_unique_run_dir_appends_suffix(tmp_path: Path):
    """When a folder with the same id exists, a numeric suffix is added."""
    d1 = make_unique_run_dir(tmp_path, "test_run")
    d2 = make_unique_run_dir(tmp_path, "test_run")
    d3 = make_unique_run_dir(tmp_path, "test_run")
    assert d1.name == "test_run"
    assert d2.name == "test_run_2"
    assert d3.name == "test_run_3"


def test_make_unique_run_dir_creates_parent(tmp_path: Path):
    """The parent directory is created if it does not exist."""
    nested = tmp_path / "does" / "not" / "exist"
    d = make_unique_run_dir(nested, "run_a")
    assert d.is_dir()
    assert nested.is_dir()


# ---------------------------------------------------------------------------
# SceneAvailability sanity
# ---------------------------------------------------------------------------

def _make_scene(scene_id: str, dt: datetime, size_bytes: int = 1_000_000_000) -> SceneMetadata:
    return SceneMetadata(
        scene_id=scene_id,
        platform="sentinel-1",
        acquisition_datetime=dt,
        relative_orbit=45,
        orbit_direction="ASCENDING",
        polarisation="VV",
        product_type="GRD",
        tile_or_granule_id=f"{scene_id}.SAFE",
        size_bytes=size_bytes,
    )


def test_scene_availability_has_pair_flag():
    pre = _make_scene("pre", datetime(2026, 8, 16, tzinfo=timezone.utc))
    post = _make_scene("post", datetime(2026, 8, 28, tzinfo=timezone.utc))
    pair = ScenePair(pre=pre, post=post)
    avail = SceneAvailability(
        aoi=BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0),
        event_date=date(2026, 8, 26),
        pre_scenes=[pre],
        post_scenes=[post],
        selected_pair=pair,
        estimated_download_mb=2000.0,
    )
    assert avail.has_pair is True
    assert avail.total_scenes == 2


def test_scene_availability_no_pair_flag():
    avail = SceneAvailability(
        aoi=BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0),
        event_date=date(2026, 8, 26),
        pre_scenes=[],
        post_scenes=[],
        selected_pair=None,
        estimated_download_mb=0.0,
    )
    assert avail.has_pair is False


# ---------------------------------------------------------------------------
# Download confirmation gate
# ---------------------------------------------------------------------------

def test_run_full_analysis_requires_confirmation_for_large_download(
    tmp_path: Path,
):
    """A large estimated download should trigger the confirmation error
    before any network access."""
    pre = _make_scene("pre", datetime(2026, 8, 16, tzinfo=timezone.utc))
    post = _make_scene("post", datetime(2026, 8, 28, tzinfo=timezone.utc))
    pair = ScenePair(pre=pre, post=post)
    availability = SceneAvailability(
        aoi=BoundingBox(min_lon=85.1, min_lat=27.6, max_lon=85.55, max_lat=28.05),
        event_date=date(2026, 8, 26),
        pre_scenes=[pre],
        post_scenes=[post],
        selected_pair=pair,
        estimated_download_mb=2400.0,
    )
    event = EventDefinition(
        aoi=availability.aoi,
        event_date=availability.event_date,
    )
    with pytest.raises(DownloadConfirmationRequired, match="2400"):
        run_full_analysis(
            event,
            availability=availability,
            confirm_download=False,
            size_warning_mb=800.0,
        )


def test_run_full_analysis_raises_when_no_pair():
    """A SceneAvailability with no pair should raise NoSuitableImageryError
    before any network access."""
    from georescue.pipeline import NoSuitableImageryError

    availability = SceneAvailability(
        aoi=BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0),
        event_date=date(2026, 8, 26),
        pre_scenes=[],
        post_scenes=[],
        selected_pair=None,
        estimated_download_mb=0.0,
        warnings=["No scenes found."],
    )
    event = EventDefinition(
        aoi=availability.aoi,
        event_date=availability.event_date,
    )
    with pytest.raises(NoSuitableImageryError):
        run_full_analysis(
            event,
            availability=availability,
            confirm_download=True,
        )