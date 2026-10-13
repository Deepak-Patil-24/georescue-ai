"""Download the pre-event and post-event Sentinel-1 scenes for the case study.

This script is a one-off helper for the August 2026 Trishuli flood case
study. It is NOT part of the application's runtime; the dashboard and the
general pipeline select scenes automatically.

What it does
------------
1. Searches CDSE for Sentinel-1 IW GRD products over the case-study AOI.
2. Restricts to the two target dates (pre-event and post-event). The
   search does not filter by orbit direction at the API level, because
   the CDSE OData name filter for orbit direction is unreliable. Instead,
   we retrieve all scenes near the target date and prefer ASCENDING in
   Python, falling back to DESCENDING with a recorded warning.
3. Picks the scene whose acquisition date matches the target exactly.
4. Downloads both products as ``.zip`` files into ``data/raw/sentinel1/``.
5. Verifies file sizes and prints a summary.

Usage
-----
    python scripts/download_case_study.py

Prerequisites
-------------
- ``.env`` must contain valid CDSE OAuth client credentials.
- At least 5 GB free disk space.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Make ``georescue`` importable when run from the project root without
# requiring PYTHONPATH to be set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from georescue.config import get_settings  # noqa: E402
from georescue.data_access import DataAccessError, NoResultsError  # noqa: E402
from georescue.data_access.copernicus import CopernicusClient  # noqa: E402
from georescue.data_access.sentinel1 import search_sentinel1  # noqa: E402
from georescue.logging_config import configure_logging  # noqa: E402
from georescue.models import BoundingBox, SceneMetadata  # noqa: E402


# ---------------------------------------------------------------------------
# Case-study configuration
# ---------------------------------------------------------------------------
# Values match the Trishuli / Bhote Koshi corridor AOI used elsewhere in
# the project. They are intentionally duplicated here so this script is
# self-contained and can be run without importing config/settings.py.

AOI = BoundingBox(
    min_lon=85.25,
    min_lat=28.10,
    max_lon=85.42,
    max_lat=28.24,
)

PRE_EVENT_DATE = date(2026, 8, 16)
POST_EVENT_DATE = date(2026, 8, 28)

# We search a small window around each target date so the script is
# robust to the exact acquisition hour.
PRE_WINDOW_DAYS = 1
POST_WINDOW_DAYS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_scene(
    client: CopernicusClient,
    target_date: date,
    window_days: int,
) -> SceneMetadata:
    """Return the best Sentinel-1 scene on or near ``target_date``.

    Strategy:
    1. Search without orbit-direction filter (the OData name filter for
       orbit direction is unreliable).
    2. Deduplicate by product name.
    3. Prefer scenes with acquisition date == target_date; otherwise take
       the closest date.
    4. Within the selected date, prefer ASCENDING; fall back to DESCENDING
       with a printed warning.
    """
    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)

    print(f"\nSearching for scenes near {target_date} "
          f"(window {start} .. {end})...")

    scenes = search_sentinel1(
        AOI,
        start,
        end,
        max_results=50,
        client=client,
    )

    # The catalogue often returns duplicate entries for the same product
    # (one per asset type). Deduplicate by product name.
    seen_names: set[str] = set()
    unique: list[SceneMetadata] = []
    for scene in scenes:
        name = scene.tile_or_granule_id or scene.scene_id
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append(scene)

    if not unique:
        raise NoResultsError(
            f"No Sentinel-1 scenes found near {target_date}."
        )

    print(f"  Candidates ({len(unique)} unique):")
    for s in unique:
        print(f"    - {s.acquisition_datetime.date()} "
              f"{s.orbit_direction} {s.tile_or_granule_id[:55]}")

    # Prefer scenes whose acquisition date matches the target exactly.
    exact = [s for s in unique if s.acquisition_datetime.date() == target_date]
    pool = exact if exact else unique

    # Within the pool, prefer ASCENDING; fall back to DESCENDING.
    ascending = [s for s in pool if s.orbit_direction == "ASCENDING"]
    if ascending:
        chosen = ascending[0]
    else:
        descending = [s for s in pool if s.orbit_direction == "DESCENDING"]
        if descending:
            print("  WARNING: no ASCENDING scene on target date; "
                  "using DESCENDING.")
            chosen = descending[0]
        else:
            chosen = pool[0]

    print(f"  Selected:")
    print(f"    name:   {chosen.tile_or_granule_id}")
    print(f"    id:     {chosen.scene_id}")
    print(f"    date:   {chosen.acquisition_datetime}")
    print(f"    orbit:  {chosen.orbit_direction}")
    print(f"    polar:  {chosen.polarisation}")
    if chosen.size_bytes:
        print(f"    size:   {chosen.size_bytes / (1024**2):.1f} MB")
    return chosen


def _download(
    client: CopernicusClient,
    scene: SceneMetadata,
    destination_dir: Path,
) -> Path:
    """Download ``scene`` as a ``.zip`` file into ``destination_dir``."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    # CDSE serves the product archive as ``<ProductName>.zip``.
    name = scene.tile_or_granule_id or scene.scene_id
    if not name.endswith(".zip"):
        name = name + ".zip"
    target = destination_dir / name

    print(f"\nDownloading {name}")
    print(f"  destination: {target}")
    if target.exists():
        size_mb = target.stat().st_size / (1024**2)
        print(f"  already present ({size_mb:.1f} MB); skipping.")
        return target

    return client.download_product(scene.scene_id, target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    configure_logging("INFO")
    settings = get_settings()

    if not settings.has_cdse_credentials:
        print("ERROR: CDSE credentials are not configured in .env.",
              file=sys.stderr)
        print("       Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.",
              file=sys.stderr)
        return 2

    raw_dir = settings.raw_data_path / "sentinel1"
    print(f"Download directory: {raw_dir}")

    client = CopernicusClient(settings=settings)
    pre_path: Path | None = None
    post_path: Path | None = None
    try:
        pre_scene = _find_scene(client, PRE_EVENT_DATE, PRE_WINDOW_DAYS)
        post_scene = _find_scene(client, POST_EVENT_DATE, POST_WINDOW_DAYS)

        print("\n=== Selected pair ===")
        print(f"  PRE:  {pre_scene.tile_or_granule_id}")
        print(f"        {pre_scene.acquisition_datetime} "
              f"({pre_scene.orbit_direction})")
        print(f"  POST: {post_scene.tile_or_granule_id}")
        print(f"        {post_scene.acquisition_datetime} "
              f"({post_scene.orbit_direction})")

        days_apart = abs(
            (post_scene.acquisition_datetime
             - pre_scene.acquisition_datetime).days
        )
        print(f"  Separation: {days_apart} days")

        if pre_scene.orbit_direction != post_scene.orbit_direction:
            print("  WARNING: pre and post scenes have different orbit "
                  "directions; pixel-comparison will be less reliable.")

        pre_path = _download(client, pre_scene, raw_dir)
        post_path = _download(client, post_scene, raw_dir)

    except NoResultsError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except DataAccessError as exc:
        print(f"\nDATA ACCESS ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print("\n=== Download complete ===")
    for p in (pre_path, post_path):
        if p is None:
            continue
        size_mb = p.stat().st_size / (1024**2)
        print(f"  {p.name}  ({size_mb:.1f} MB)")

    print(f"\nFiles stored in: {raw_dir}")
    print("Next step: run the preprocessing pipeline on these two products.")
    return 0


if __name__ == "__main__":
    sys.exit(main())