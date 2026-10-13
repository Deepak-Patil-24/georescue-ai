"""Run the Trishuli case study through the full pipeline.

This script is a thin wrapper around :func:`georescue.pipeline.run_full_analysis`.
It runs the complete Sentinel-1 flood assessment for the original
Trishuli case study and writes all outputs into a **unique new run
folder** under ``outputs/runs/``.

Safety
------
- This script **never** overwrites the precomputed Trishuli results
  that live at ``data/processed/`` and ``outputs/``.
- Every invocation creates a new folder
  ``outputs/runs/trishuli_case_study_YYYYMMDD[_N]/``.
- If the run fails partway through, the partial folder remains with a
  ``FAILURE.json`` file describing the error. It is not added to the
  dashboard's run selector until it completes successfully.

Usage
-----
    python scripts/run_case_study.py
    python scripts/run_case_study.py --confirm-download
    python scripts/run_case_study.py --event-date 2026-08-26 --window 15

By default the script refuses to start if the estimated download size
exceeds 800 MB. Pass ``--confirm-download`` to override that and proceed
with the download. This mirrors the dashboard's mandatory confirmation
checkbox.

Prerequisites
-------------
- ``.env`` contains valid CDSE credentials and an ohsome API key.
- The Sentinel-1 products for the chosen dates are already cached, or
  the CDSE account has enough quota to download them.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Make ``georescue`` importable when the script is run from the project
# root without PYTHONPATH being set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from georescue.config import get_settings  # noqa: E402
from georescue.logging_config import configure_logging, get_logger  # noqa: E402
from georescue.models import BoundingBox, EventDefinition  # noqa: E402
from georescue.pipeline import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    DownloadConfirmationRequired,
    NoSuitableImageryError,
    PipelineError,
    run_full_analysis,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Case-study defaults
# ---------------------------------------------------------------------------

TRISHULI_BBOX = BoundingBox(
    min_lon=85.10, min_lat=27.60,
    max_lon=85.55, max_lat=28.05,
)
TRISHULI_EVENT_DATE = date(2026, 8, 26)
TRISHULI_AOI_NAME = "Trishuli case study"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Sentinel-1 flood assessment for the Trishuli "
            "case study. Outputs go to a new folder under outputs/runs/."
        ),
    )
    parser.add_argument(
        "--event-date",
        type=date.fromisoformat,
        default=TRISHULI_EVENT_DATE,
        help=(
            f"Event date (YYYY-MM-DD). Default: "
            f"{TRISHULI_EVENT_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=(
            f"Search window in days on either side of the event date. "
            f"Default: {DEFAULT_WINDOW_DAYS}."
        ),
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help=(
            "Confirm the estimated download size (may exceed 800 MB) and "
            "proceed. Without this flag the script refuses to start if the "
            "estimated download is large."
        ),
    )
    parser.add_argument(
        "--aoi-name",
        default=TRISHULI_AOI_NAME,
        help="Human-readable name used for the run folder id.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for this run. Default: INFO.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    settings = get_settings()

    if not settings.has_cdse_credentials:
        print(
            "ERROR: CDSE credentials are not configured. Add "
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to .env.",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("  GeoRescue AI — Trishuli case study")
    print("=" * 70)
    print(f"  Area:        {args.aoi_name}")
    print(f"  Bounding box: {TRISHULI_BBOX.as_tuple()}")
    print(f"  Event date:   {args.event_date.isoformat()}")
    print(f"  Window:       ±{args.window} days")
    print(f"  Confirm download: {args.confirm_download}")
    print()

    event = EventDefinition(
        aoi=TRISHULI_BBOX,
        event_date=args.event_date,
        pre_days=args.window,
        post_days=args.window,
        aoi_name=args.aoi_name,
    )

    def _progress(message: str, fraction: float) -> None:
        bar = "#" * int(40 * fraction) + "-" * (40 - int(40 * fraction))
        print(f"  [{bar}] {fraction * 100:5.1f}%  {message}")

    t0 = time.time()
    try:
        result = run_full_analysis(
            event,
            confirm_download=args.confirm_download,
            progress=_progress,
        )
    except DownloadConfirmationRequired as exc:
        print()
        print(f"  DOWNLOAD CONFIRMATION REQUIRED")
        print(f"    {exc}")
        print()
        print("  Re-run with --confirm-download to proceed.")
        return 3
    except NoSuitableImageryError as exc:
        print()
        print(f"  NO SUITABLE IMAGERY")
        print(f"    {exc}")
        return 4
    except PipelineError as exc:
        print()
        print(f"  PIPELINE ERROR")
        print(f"    {exc}")
        return 5
    except Exception as exc:
        print()
        print(f"  UNEXPECTED ERROR: {type(exc).__name__}")
        print(f"    {exc}")
        return 6

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"  Run folder:      {result.run_dir}")
    print(f"  Flood extent:    {result.flood_extent_geojson}")
    print(f"  Area (km²):      {result.area_sqkm:.4f}")
    print(f"  Area (hectares): {result.area_hectares:.2f}")
    print(f"  Pixel count:     {result.pixel_count}")
    print()
    print(f"  Elapsed: {elapsed:.1f} s")
    if result.warnings:
        print()
        print("  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    print()
    print("  The precomputed Trishuli results at data/processed/ and "
          "outputs/infrastructure|connectivity/ have NOT been modified.")
    print("  To view the new run, refresh the dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())