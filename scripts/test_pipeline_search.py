"""Test the search-only stage of the pipeline module.

This script exercises ``georescue.pipeline.search_only`` against several
AOIs to verify that the search works and returns sensible availability
data. It does **not** download any satellite products.

Usage
-----
    python scripts/test_pipeline_search.py

Prerequisites
-------------
- ``.env`` must contain valid CDSE OAuth credentials (CDSE_CLIENT_ID,
  CDSE_CLIENT_SECRET).
- The ohsome key is not required for this test.

Expected runtime
----------------
About 30-60 seconds. Each AOI requires one or two CDSE catalogue
queries.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from georescue.config import get_settings  # noqa: E402
from georescue.logging_config import configure_logging  # noqa: E402
from georescue.models import BoundingBox  # noqa: E402
from georescue.pipeline import (  # noqa: E402
    NoSuitableImageryError,
    PipelineError,
    search_only,
)


CASES = [
    {
        "name": "Trishuli corridor (case study)",
        "bbox": BoundingBox(
            min_lon=85.10, min_lat=27.60,
            max_lon=85.55, max_lat=28.05,
        ),
        "event_date": date(2026, 8, 26),
    },
    {
        "name": "Koshi riverplain (fresh AOI test)",
        "bbox": BoundingBox(
            min_lon=86.90, min_lat=26.40,
            max_lon=87.20, max_lat=26.70,
        ),
        "event_date": date(2026, 8, 26),
    },
    {
        "name": "Pacific Ocean (should return no imagery)",
        "bbox": BoundingBox(
            min_lon=-150.0, min_lat=10.0,
            max_lon=-149.0, max_lat=11.0,
        ),
        "event_date": date(2026, 8, 26),
    },
]


def _print_availability(availability) -> None:
    print(f"  event date: {availability.event_date.isoformat()}")
    print(f"  pre-event scenes found:  {len(availability.pre_scenes)}")
    print(f"  post-event scenes found: {len(availability.post_scenes)}")
    print(f"  total scenes: {availability.total_scenes}")

    if availability.selected_pair is not None:
        pre = availability.selected_pair.pre
        post = availability.selected_pair.post
        print(f"  selected pre:  {pre.acquisition_datetime.date()} "
              f"{pre.orbit_direction} orbit {pre.relative_orbit} "
              f"pol {pre.polarisation}")
        print(f"  selected post: {post.acquisition_datetime.date()} "
              f"{post.orbit_direction} orbit {post.relative_orbit} "
              f"pol {post.polarisation}")
        days_apart = abs(
            (post.acquisition_datetime - pre.acquisition_datetime).days
        )
        print(f"  temporal separation: {days_apart} days")
    else:
        print("  selected pair: NONE (no compatible pair found)")

    print(f"  estimated download: {availability.estimated_download_mb:.1f} MB")

    if availability.selection_notes:
        print("  selection notes:")
        for note in availability.selection_notes:
            print(f"    - {note}")

    if availability.warnings:
        print("  warnings:")
        for w in availability.warnings:
            print(f"    - {w}")


def main() -> int:
    configure_logging("WARNING")  # quieter; we print our own messages

    settings = get_settings()
    if not settings.has_cdse_credentials:
        print(
            "ERROR: CDSE credentials are not configured. Add CDSE_CLIENT_ID "
            "and CDSE_CLIENT_SECRET to .env before running this test.",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("  search_only() test")
    print("=" * 70)

    failures = 0
    for case in CASES:
        print(f"\n--- {case['name']} ---")
        print(f"  bbox: {case['bbox'].as_tuple()}")
        try:
            availability = search_only(
                case["bbox"], case["event_date"],
                aoi_name=case["name"],
            )
            _print_availability(availability)

            # Sanity checks that would indicate a real problem.
            if "Trishuli" in case["name"]:
                if not availability.has_pair:
                    print("  STATUS: FAIL — expected a compatible pair")
                    failures += 1
                else:
                    print("  STATUS: OK")
            elif "Koshi" in case["name"]:
                # We expect coverage but not necessarily a pair; report.
                if availability.total_scenes == 0:
                    print("  STATUS: WARNING — no scenes found at all")
                else:
                    print("  STATUS: OK (scenes found)")
            elif "Pacific" in case["name"]:
                if availability.total_scenes == 0:
                    print("  STATUS: OK (correctly reports no imagery)")
                else:
                    print(f"  STATUS: WARNING — found "
                          f"{availability.total_scenes} scenes in open ocean; "
                          f"verify the search is behaving correctly.")
        except NoSuitableImageryError as exc:
            print(f"  NO SUITABLE IMAGERY: {exc}")
        except PipelineError as exc:
            print(f"  PIPELINE ERROR: {exc}")
            failures += 1
        except Exception as exc:  # pragma: no cover - unexpected
            print(f"  UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 70)
    if failures:
        print(f"  RESULT: {failures} case(s) failed")
        return 1
    print("  RESULT: all cases handled as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())