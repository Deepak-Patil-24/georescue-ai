"""Place-name geocoding for GeoRescue AI.

Uses the public Nominatim service from OpenStreetMap to convert a place
name (for example "Kathmandu", "Koshi riverplain", or "Syapru Besi")
into a WGS84 bounding box that the pipeline can use directly.

Usage policy
------------
Nominatim's usage policy requires:

- A descriptive ``User-Agent`` header identifying the application.
- At most 1 request per second.
- No heavy or bulk usage.

This module enforces the rate limit and sets a User-Agent that names
GeoRescue AI. It does not support bulk geocoding.

Attribution
-----------
Results are derived from OpenStreetMap data:
"© OpenStreetMap contributors."

Failure behaviour
-----------------
If the service is unavailable, returns a non-JSON body, or returns no
results, the module raises :class:`GeocodingError` with an actionable
message. The caller (dashboard) is expected to fall back to coordinate
entry.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from georescue.logging_config import get_logger
from georescue.models import BoundingBox

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

# Descriptive User-Agent as required by the Nominatim usage policy.
# Do not remove or shorten this.
_USER_AGENT = (
    "GeoRescue-AI/0.1 (Multimodal AI Hackathon 2026 prototype; "
    "contact: project maintainers via hackathon channel)"
)

# Request timeouts (connect, read) in seconds.
_TIMEOUT = (5, 15)

# Minimum seconds between consecutive requests to Nominatim. The service
# requires at most 1 request per second; we use slightly more than 1 to
# be safe.
_MIN_INTERVAL_S = 1.2

# Limit on the number of results requested from Nominatim. We only need
# the first one with a valid bounding box.
_RESULT_LIMIT = 5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GeocodingError(RuntimeError):
    """Raised when a place cannot be geocoded.

    The message is user-facing and should explain what the caller can do.
    """


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """A thread-safe rate limiter that enforces a minimum interval
    between calls."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval = min_interval_s
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_limiter = _RateLimiter(_MIN_INTERVAL_S)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GeocodeResult:
    """Result of a successful place-name lookup."""

    display_name: str
    bbox: BoundingBox
    place_type: Optional[str]
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geocode_place_name(
    query: str,
    *,
    expand_fraction: float = 0.15,
    min_half_width_deg: float = 0.02,
) -> GeocodeResult:
    """Look up a place name and return a WGS84 bounding box.

    Parameters
    ----------
    query:
        Human-readable place name, e.g. "Kathmandu" or "Koshi riverplain".
    expand_fraction:
        Fraction of the bbox width and height to add as padding on each
        side, so the AOI is slightly larger than the raw OSM match. This
        avoids clipping small features. Default 0.15 = 15%.
    min_half_width_deg:
        Minimum half-width of the returned bbox in degrees. Small places
        (e.g. a single village) would otherwise produce AOIs too small
        for the Sentinel-1 analysis. Default 0.02° ≈ 2.2 km at the
        equator.

    Returns
    -------
    :class:`GeocodeResult`

    Raises
    ------
    GeocodingError
        If the query is empty, no place matched, the service is
        unreachable, or the response is malformed.
    """
    query = (query or "").strip()
    if not query:
        raise GeocodingError("Enter a place name to search.")

    params = {
        "q": query,
        "format": "json",
        "limit": _RESULT_LIMIT,
        "addressdetails": 0,
    }
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }

    _limiter.wait()

    try:
        response = requests.get(
            NOMINATIM_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise GeocodingError(
            f"Could not reach the place-name service: {exc}. "
            f"Check your internet connection or enter coordinates directly."
        ) from exc

    if response.status_code == 429:
        raise GeocodingError(
            "The place-name service is rate limiting requests. "
            "Wait a few seconds and try again, or enter coordinates directly."
        )
    if response.status_code >= 400:
        raise GeocodingError(
            f"Place-name service returned HTTP {response.status_code}. "
            f"Try again later or enter coordinates directly."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingError(
            "Place-name service returned a response that was not JSON. "
            "Try again later or enter coordinates directly."
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise GeocodingError(
            f"No place named {query!r} was found. Try a different name, "
            f"or enter coordinates directly."
        )

    # Pick the first result with a valid bounding box.
    for entry in payload:
        bbox = _parse_bbox(entry)
        if bbox is None:
            continue
        # Expand and enforce a minimum size.
        bbox = _expand_bbox(bbox, expand_fraction, min_half_width_deg)
        return GeocodeResult(
            display_name=str(entry.get("display_name") or query),
            bbox=bbox,
            place_type=entry.get("type"),
            raw=entry,
        )

    raise GeocodingError(
        f"Place {query!r} was found but its bounding box could not be "
        f"read. Try a different name, or enter coordinates directly."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_bbox(entry: dict[str, Any]) -> Optional[BoundingBox]:
    """Extract a BoundingBox from a Nominatim result, or return None."""
    raw = entry.get("boundingbox")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        south, north, west, east = (float(x) for x in raw)
    except (TypeError, ValueError):
        return None

    if not (-90.0 <= south < north <= 90.0):
        return None
    if not (-180.0 <= west < east <= 180.0):
        # Handle antimeridian-crossing bboxes by rejecting them; rare and
        # complex. The user can enter coordinates directly.
        return None

    try:
        return BoundingBox(
            min_lon=west, min_lat=south,
            max_lon=east, max_lat=north,
        )
    except Exception:
        return None


def _expand_bbox(
    bbox: BoundingBox,
    expand_fraction: float,
    min_half_width_deg: float,
) -> BoundingBox:
    """Expand a bbox and enforce a minimum width/height in degrees."""
    width = bbox.max_lon - bbox.min_lon
    height = bbox.max_lat - bbox.min_lat

    # Enforce minimum size.
    if width < 2 * min_half_width_deg:
        cx = (bbox.min_lon + bbox.max_lon) / 2.0
        bbox = BoundingBox(
            min_lon=cx - min_half_width_deg,
            min_lat=bbox.min_lat,
            max_lon=cx + min_half_width_deg,
            max_lat=bbox.max_lat,
        )
        width = bbox.max_lon - bbox.min_lon
    if height < 2 * min_half_width_deg:
        cy = (bbox.min_lat + bbox.max_lat) / 2.0
        bbox = BoundingBox(
            min_lon=bbox.min_lon,
            min_lat=cy - min_half_width_deg,
            max_lon=bbox.max_lon,
            max_lat=cy + min_half_width_deg,
        )
        height = bbox.max_lat - bbox.min_lat

    # Expand by the requested fraction on each side.
    pad_lon = width * expand_fraction
    pad_lat = height * expand_fraction

    new_min_lon = max(-180.0, bbox.min_lon - pad_lon)
    new_max_lon = min(180.0, bbox.max_lon + pad_lon)
    new_min_lat = max(-90.0, bbox.min_lat - pad_lat)
    new_max_lat = min(90.0, bbox.max_lat + pad_lat)

    if new_max_lon <= new_min_lon or new_max_lat <= new_min_lat:
        # Expansion collapsed the box; return the original.
        return bbox

    return BoundingBox(
        min_lon=new_min_lon, min_lat=new_min_lat,
        max_lon=new_max_lon, max_lat=new_max_lat,
    )


__all__ = [
    "GeocodeResult",
    "GeocodingError",
    "geocode_place_name",
]