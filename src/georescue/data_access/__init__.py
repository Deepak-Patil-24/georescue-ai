"""Data access clients for GeoRescue AI.

Every client in this sub-package:

- performs network I/O only through :mod:`requests`
- reads credentials exclusively from :class:`georescue.config.Settings`
- raises :class:`DataAccessError` (defined here) with a clear, user-facing
  message on failure
- never falls back to synthetic or fabricated data

Public entry points
-------------------
- :class:`~georescue.data_access.copernicus.CopernicusClient` — shared
  OAuth / STAC / OData helper used by the Sentinel clients.
- :mod:`~georescue.data_access.sentinel1` — Sentinel-1 GRD search and
  download.
- :mod:`~georescue.data_access.sentinel2` — Sentinel-2 L2A search and
  download.
- :mod:`~georescue.data_access.historical_osm` — ohsome historical OSM
  client.
- :mod:`~georescue.data_access.dem` — Copernicus DEM 30 m from AWS.
"""

from __future__ import annotations


class DataAccessError(RuntimeError):
    """Raised when a remote data source cannot be reached or queried.

    The message is written to be shown directly to the user. It should
    explain what failed and what the user can do about it.
    """


class AuthenticationError(DataAccessError):
    """Raised when authentication with a remote service fails."""


class NoResultsError(DataAccessError):
    """Raised when a query succeeds but returns no usable results.

    This is distinct from a network failure: the service is up, the query
    is well-formed, but no matching data exist for the requested area and
    time window. The dashboard should present this as a coverage problem,
    not as an error.
    """


__all__ = [
    "AuthenticationError",
    "DataAccessError",
    "NoResultsError",
]