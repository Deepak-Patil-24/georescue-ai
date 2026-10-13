"""User-facing defaults for GeoRescue AI.

This module intentionally contains *only* default values that a developer
or judge might want to tweak without touching application logic. It is
imported by ``app.py`` and by the case-study script. It does NOT read
environment variables — that is the job of
:mod:`georescue.config`.

Rules
-----
- These values are fallbacks. The dashboard lets users override them.
- Never put credentials here.
- Keep values realistic for the Himalayan corridor used in the case study,
  but remember the pipeline must work anywhere on Earth.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Default case-study AOI (Trishuli / Bhote Koshi corridor, Nepal)
# ---------------------------------------------------------------------------
# This is a *default value only*. The dashboard and case-study script let
# the user supply any AOI and event date. Nothing downstream is hardcoded
# to this location.

DEFAULT_EVENT_DATE: Final[str] = "2026-08-26"

DEFAULT_AOI_BBOX: Final[tuple[float, float, float, float]] = (
    85.10,   # min_lon
    27.60,   # min_lat
    85.55,   # max_lon
    28.05,   # max_lat
)

DEFAULT_AOI_NAME: Final[str] = "Trishuli / Bhote Koshi corridor (case study default)"

# ---------------------------------------------------------------------------
# Temporal search windows
# ---------------------------------------------------------------------------

DEFAULT_PRE_DAYS: Final[int] = 12
DEFAULT_POST_DAYS: Final[int] = 12

# ---------------------------------------------------------------------------
# Sentinel-1 selection preferences
# ---------------------------------------------------------------------------

PREFER_SAME_RELATIVE_ORBIT: Final[bool] = True
S1_PREFERRED_POLARISATION: Final[str] = "VV"
S1_ALLOWED_ORBIT_DIRECTIONS: Final[tuple[str, ...]] = ("ASCENDING", "DESCENDING")

# ---------------------------------------------------------------------------
# Sentinel-2 optical settings
# ---------------------------------------------------------------------------

DEFAULT_MAX_CLOUD_COVER: Final[float] = 30.0

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

# Backscatter ratio threshold (linear power) below which a post-event pixel
# is considered a *candidate* flooded pixel. Values from the literature
# typically fall between 0.3 and 0.5; the dashboard exposes this.
DEFAULT_S1_RATIO_THRESHOLD: Final[float] = 0.4

# Minimum mapping unit for candidate flood polygons, in hectares.
DEFAULT_MIN_MMU_HA: Final[float] = 0.5

# Slope threshold above which pixels are masked out as implausible for
# standing water, in degrees.
DEFAULT_MAX_SLOPE_DEG: Final[float] = 5.0

# Buffer around the flood polygon for the "potentially exposed" layer, in
# metres.
DEFAULT_EXPOSURE_BUFFER_M: Final[float] = 0.0

# ---------------------------------------------------------------------------
# Connectivity settings
# ---------------------------------------------------------------------------

# A road segment is treated as *potentially impassable* in the modelled
# scenario when the fraction of its length intersecting the candidate
# flood extent exceeds this value.
DEFAULT_ROAD_DISRUPTION_FRACTION: Final[float] = 0.20

# Settlement classes considered as "settlements" for connectivity analysis.
SETTLEMENT_PLACE_VALUES: Final[tuple[str, ...]] = ("village", "hamlet", "town")

# Destination classes for connectivity (hospital or town).
DESTINATION_PLACE_VALUES: Final[tuple[str, ...]] = ("town", "city")
DESTINATION_AMENITY_VALUES: Final[tuple[str, ...]] = ("hospital", "clinic")

# ---------------------------------------------------------------------------
# Area limits
# ---------------------------------------------------------------------------

MAX_AOI_AREA_SQKM: Final[float] = 5000.0

# ---------------------------------------------------------------------------
# Attribution strings (must be preserved verbatim in reports and exports)
# ---------------------------------------------------------------------------

ATTRIBUTION_SENTINEL: Final[str] = (
    "Contains modified Copernicus Sentinel data 2026."
)

ATTRIBUTION_DEM: Final[str] = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and "
    "© Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS "
    "by the European Union and ESA; all rights reserved."
)

ATTRIBUTION_OSM: Final[str] = "© OpenStreetMap contributors."

ATTRIBUTIONS: Final[tuple[str, ...]] = (
    ATTRIBUTION_SENTINEL,
    ATTRIBUTION_DEM,
    ATTRIBUTION_OSM,
)