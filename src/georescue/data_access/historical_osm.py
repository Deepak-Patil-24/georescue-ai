"""Historical OpenStreetMap access via the ohsome API (v2).

The hackathon rules require pre-event OSM data. This module uses the
ohsome ``extraction/features`` endpoint with an explicit ``time``
parameter so that the returned features are a snapshot of OSM *as it was*
at that timestamp.

Rules enforced by this module
-----------------------------
- The requested timestamp must be in the past (or the literal string
  ``"latest"``, which ohsome interprets as the most recent snapshot).
- If the caller supplies an event date, the snapshot timestamp must be
  strictly earlier than the event date, otherwise :class:`DataAccessError`
  is raised. We do **not** silently fall back to current OSM data.
- The ohsome API key must be configured. If it is missing, a clear
  :class:`AuthenticationError` is raised rather than a 403 from the server.

Failure handling
----------------
ohsome v2 serves extraction responses as GeoParquet. In practice, the
``.parquet`` endpoint occasionally returns a body that cannot be parsed
as Parquet (observed as ``Parquet magic bytes not found in footer``).
When that happens, we do **not** silently return zero features. We
retry the same query against the ``.json`` output format, which serves
plain GeoJSON. If both formats fail, a :class:`DataAccessError` is
raised so the caller can record the failure honestly. A silent zero
would be misleading; a raised error is not.

Reference
---------
ohsome API v2 documentation:
    https://docs.ohsome.org/ohsome-api/v2/
Endpoints used:
    https://api.heigit.org/ohsome-api/v2/extraction/features.parquet
    https://api.heigit.org/ohsome-api/v2/extraction/features.json
Authentication:
    ``authorization: <API_KEY>`` header. The key is obtained free of
    charge from https://account.heigit.org.
"""

from __future__ import annotations

import io
import json as _json
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional, Union

import requests

from georescue.config import Settings, get_settings
from georescue.data_access import (
    AuthenticationError,
    DataAccessError,
    NoResultsError,
)
from georescue.logging_config import get_logger
from georescue.models import BoundingBox

log = get_logger(__name__)

_TIMEOUT = (10, 300)

# Minimum plausible size for a parquet file. A valid GeoParquet response
# is always larger than this.
_MIN_PARQUET_BYTES = 100

# The Parquet magic bytes at the start of a valid Parquet file.
_PARQUET_MAGIC = b"PAR1"

# Type alias: callers may pass a date, a datetime, or the literal "latest".
TimeLike = Union[date, datetime, str]


# ---------------------------------------------------------------------------
# Time and bbox helpers
# ---------------------------------------------------------------------------

def _format_ohsome_time(ts: TimeLike) -> str:
    """Format a timestamp as an ISO-8601 string accepted by ohsome v2."""
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ts.strftime("%Y-%m-%dT00:00:00Z")


def _snapshot_date_for_rule_check(ts: TimeLike) -> Optional[date]:
    """Return the snapshot date for pre-event rule enforcement, or None."""
    if isinstance(ts, str):
        if ts.lower() == "latest":
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    if isinstance(ts, datetime):
        return ts.date()
    return ts


def _bbox_to_aoi_array(bbox: BoundingBox) -> list[float]:
    """Return the AOI as a 4-element array for the v2 JSON body."""
    return [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _auth_headers(settings: Settings, *, accept: str) -> dict[str, str]:
    """Return request headers, including the ohsome API key."""
    if not settings.ohsome_api_key:
        raise AuthenticationError(
            "OHSOME_API_KEY is not configured. The ohsome API v2 requires "
            "an API key for every request. Register for a free key at "
            "https://account.heigit.org and add it to your .env file as "
            "OHSOME_API_KEY=<your key>."
        )
    return {
        "User-Agent": "GeoRescue-AI/0.1 (hackathon prototype)",
        "authorization": settings.ohsome_api_key,
        "Accept": accept,
    }


# ---------------------------------------------------------------------------
# Parquet response parsing
# ---------------------------------------------------------------------------

def _parse_parquet_response(content: bytes) -> list[dict[str, Any]]:
    """Parse an ohsome v2 parquet response into GeoJSON Feature dicts."""
    import pandas as pd
    from shapely import wkb, wkt

    if not content:
        return []

    try:
        df = pd.read_parquet(io.BytesIO(content))
    except Exception as exc:
        raise DataAccessError(
            f"Could not read ohsome parquet response: {exc}"
        ) from exc

    if df.empty:
        return []

    geom_col = None
    for candidate in ("geometry", "geom"):
        if candidate in df.columns:
            geom_col = candidate
            break
    if geom_col is None:
        log.warning(
            "ohsome parquet response has no geometry column; columns=%s",
            list(df.columns),
        )
        return []

    features: list[dict[str, Any]] = []
    for _idx, row in df.iterrows():
        geom_value = row[geom_col]
        geom_obj = _coerce_geometry(geom_value, wkb, wkt)
        if geom_obj is None:
            continue
        try:
            geom_dict = geom_obj.__geo_interface__
        except AttributeError:
            continue

        osm_id = _first_present(row, ("osm_id", "@osmId", "id"))
        osm_type = _first_present(row, ("osm_type", "@osmType", "type"))
        tags = _coerce_tags(row.get("tags") if "tags" in df.columns else None)

        props = {
            "@osmId": None if osm_id is None else str(osm_id),
            "@osmType": None if osm_type is None else str(osm_type),
            "@snapshotTimestamp": row.get("@snapshotTimestamp")
                if "@snapshotTimestamp" in df.columns else None,
            "@lastEdit": row.get("@lastEdit")
                if "@lastEdit" in df.columns else None,
            "tags": tags,
        }

        features.append({
            "type": "Feature",
            "id": f"{props['@osmType']}/{props['@osmId']}",
            "properties": props,
            "geometry": geom_dict,
        })

    return features


def _coerce_geometry(value: Any, wkb_mod: Any, wkt_mod: Any) -> Any:
    """Return a Shapely geometry from a parquet cell, or None."""
    if value is None:
        return None
    if hasattr(value, "__geo_interface__"):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return wkb_mod.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return wkt_mod.loads(value)
        except Exception:
            return None
    return None


def _first_present(row: Any, names: tuple[str, ...]) -> Any:
    """Return the first non-None value among ``names`` in a pandas row."""
    for name in names:
        try:
            value = row[name]
        except (KeyError, IndexError):
            continue
        if value is not None:
            return value
    return None


def _coerce_tags(value: Any) -> dict[str, Any]:
    """Normalise the OSM tags cell into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# GeoJSON response parsing (fallback)
# ---------------------------------------------------------------------------

def _parse_geojson_response(content: bytes) -> list[dict[str, Any]]:
    """Parse an ohsome v2 GeoJSON response into Feature dicts.

    GeoJSON is the alternate output format for ohsome v2. Its structure
    is simpler than Parquet: it is a FeatureCollection we can walk
    directly.
    """
    if not content:
        return []
    try:
        payload = _json.loads(content)
    except ValueError as exc:
        raise DataAccessError(
            f"Could not parse ohsome GeoJSON response: {exc}"
        ) from exc

    features_in = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features_in, list):
        return []

    out: list[dict[str, Any]] = []
    for feat in features_in:
        if not isinstance(feat, dict):
            continue
        geometry = feat.get("geometry")
        if not isinstance(geometry, dict) or "type" not in geometry:
            continue
        props_in = feat.get("properties") or {}
        props = {
            "@osmId": str(props_in.get("@osmId") or props_in.get("osm_id") or ""),
            "@osmType": str(props_in.get("@osmType") or props_in.get("osm_type") or ""),
            "@snapshotTimestamp": props_in.get("@snapshotTimestamp"),
            "@lastEdit": props_in.get("@lastEdit"),
            "tags": props_in.get("tags") or {},
        }
        out.append({
            "type": "Feature",
            "id": feat.get("id") or f"{props['@osmType']}/{props['@osmId']}",
            "properties": props,
            "geometry": geometry,
        })
    return out


# ---------------------------------------------------------------------------
# Per-format request helpers
# ---------------------------------------------------------------------------

def _post_ohsome(
    url: str,
    *,
    session: requests.Session,
    aoi_array: list[float],
    time_value: str,
    filt: str,
) -> requests.Response:
    """POST a single extraction query to ohsome and handle HTTP errors.

    Shared by the Parquet and GeoJSON paths so error handling is
    identical for both.
    """
    json_body = {
        "aoi": aoi_array,
        "time": time_value,
        "filter": filt,
        "clip": True,
    }
    try:
        response = session.post(url, json=json_body, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise DataAccessError(
            f"ohsome API request failed: {exc}. "
            f"Check your internet connection or try again later."
        ) from exc

    if response.status_code == 401:
        raise AuthenticationError(
            "ohsome API rejected the request with HTTP 401. "
            "Your OHSOME_API_KEY may be invalid or expired. "
            "Check the key at https://account.heigit.org."
        )
    if response.status_code == 403:
        raise AuthenticationError(
            "ohsome API rejected the request with HTTP 403. "
            "This usually means the API key is missing, invalid, "
            "or does not have permission for this endpoint. "
            "Verify OHSOME_API_KEY in your .env file."
        )
    if response.status_code == 404:
        raise DataAccessError(
            f"ohsome endpoint not found at {url}. "
            f"Check OHSOME_BASE_URL in your .env file. "
            f"Expected base: https://api.heigit.org/ohsome-api/v2"
        )
    if response.status_code == 413:
        raise DataAccessError(
            "ohsome rejected the request as too large. "
            "Reduce the AOI size or narrow the filter."
        )
    if response.status_code >= 400:
        raise DataAccessError(
            f"ohsome returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return response


def _try_parquet(
    session: requests.Session,
    base: str,
    aoi_array: list[float],
    time_value: str,
    filt: str,
) -> Optional[list[dict[str, Any]]]:
    """Attempt the Parquet endpoint.

    Returns the parsed features, or ``None`` if the response body was not
    usable Parquet. ``None`` signals "this format failed; try the other
    format" rather than "zero features".
    """
    url = f"{base}/extraction/features.parquet"
    response = _post_ohsome(
        url,
        session=session,
        aoi_array=aoi_array,
        time_value=time_value,
        filt=filt,
    )
    body = response.content or b""

    if len(body) < _MIN_PARQUET_BYTES:
        log.info(
            "ohsome parquet response for %r is %d bytes; trying GeoJSON.",
            filt, len(body),
        )
        return None
    if body[:4] != _PARQUET_MAGIC:
        log.warning(
            "ohsome parquet response for %r does not start with PAR1 "
            "(first 4 bytes: %r); trying GeoJSON.",
            filt, body[:4],
        )
        return None
    try:
        return _parse_parquet_response(body)
    except DataAccessError as exc:
        log.warning(
            "ohsome parquet body for %r could not be parsed: %s; "
            "trying GeoJSON.",
            filt, exc,
        )
        return None


def _try_geojson(
    session: requests.Session,
    base: str,
    aoi_array: list[float],
    time_value: str,
    filt: str,
) -> Optional[list[dict[str, Any]]]:
    """Attempt the GeoJSON endpoint.

    Returns the parsed features, or ``None`` if the response body was not
    usable GeoJSON.
    """
    url = f"{base}/extraction/features.json"
    response = _post_ohsome(
        url,
        session=session,
        aoi_array=aoi_array,
        time_value=time_value,
        filt=filt,
    )
    body = response.content or b""
    if not body:
        log.info("ohsome GeoJSON response for %r is empty.", filt)
        return None
    try:
        return _parse_geojson_response(body)
    except DataAccessError as exc:
        log.warning(
            "ohsome GeoJSON body for %r could not be parsed: %s.",
            filt, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def query_osm_features(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    filters: Iterable[str],
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
    allow_geojson_fallback: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Query historical OSM features by filter group.

    Parameters
    ----------
    bbox:
        Area of interest.
    snapshot_timestamp:
        OSM snapshot time. May be a :class:`date`, :class:`datetime`, or
        the literal string ``"latest"``. Must be before ``event_date``
        when provided (except for ``"latest"``, which is refused if an
        event date is set).
    filters:
        One or more ohsome filter expressions. Each is queried separately
        so the caller can group results by theme (e.g. buildings, roads).
    event_date:
        Optional event date used to enforce the pre-event rule.
    settings:
        Optional settings override.
    allow_geojson_fallback:
        When True (default), a Parquet response that cannot be parsed is
        retried against the GeoJSON endpoint. When False, a Parquet
        parse failure is treated as a hard error so the caller sees it.

    Returns
    -------
    dict mapping each filter expression to a list of GeoJSON features.

    Raises
    ------
    AuthenticationError
        If the ohsome API key is not configured or is rejected.
    DataAccessError
        If the API is unreachable, returns an HTTP error, or returns a
        body that cannot be parsed in either supported format.
    NoResultsError
        If the API succeeds but returns zero features for all filters.
    """
    settings = settings or get_settings()

    # --- Enforce pre-event rule (local check, no network) ---
    snap_date = _snapshot_date_for_rule_check(snapshot_timestamp)
    if event_date is not None:
        if snap_date is None:
            raise DataAccessError(
                f"Snapshot timestamp {snapshot_timestamp!r} cannot be verified "
                f"as being before the event date {event_date}. "
                f"Use an explicit pre-event date instead of 'latest'."
            )
        if snap_date >= event_date:
            raise DataAccessError(
                f"OSM snapshot timestamp {snap_date} must be strictly before "
                f"the event date {event_date}. Refusing to use post-event OSM."
            )
    if snap_date is not None and snap_date >= date.today():
        raise DataAccessError(
            f"OSM snapshot timestamp {snap_date} must be in the past."
        )

    base = settings.ohsome_base_url.rstrip("/")
    time_value = _format_ohsome_time(snapshot_timestamp)
    aoi_array = _bbox_to_aoi_array(bbox)

    results: dict[str, list[dict[str, Any]]] = {}
    session = requests.Session()

    try:
        for filt in filters:
            log.info(
                "ohsome v2 query: aoi=%s time=%s filter=%s",
                aoi_array, time_value, filt,
            )

            # --- Primary: Parquet ---
            session.headers.clear()
            session.headers.update(_auth_headers(settings, accept="application/octet-stream"))
            features = _try_parquet(
                session, base, aoi_array, time_value, filt,
            )

            # --- Fallback: GeoJSON ---
            if features is None and allow_geojson_fallback:
                log.info(
                    "Falling back to GeoJSON for filter %r.", filt,
                )
                session.headers.clear()
                session.headers.update(_auth_headers(settings, accept="application/geo+json"))
                features = _try_geojson(
                    session, base, aoi_array, time_value, filt,
                )

            if features is None:
                # Both formats failed. This is a real failure, not a
                # data gap. Raise so the caller can record it honestly.
                raise DataAccessError(
                    f"ohsome returned a response for filter {filt!r} that "
                    f"could not be parsed in Parquet or GeoJSON format. "
                    f"This is a data-access error, not a data gap. "
                    f"Re-run the query, or check the ohsome service status."
                )

            log.info("ohsome filter %r returned %d features.", filt, len(features))
            results[filt] = features

    finally:
        session.close()

    if all(len(v) == 0 for v in results.values()):
        raise NoResultsError(
            f"ohsome returned no OSM features for the requested AOI "
            f"and snapshot {snapshot_timestamp}. "
            f"The area may be outside OSM coverage, or the filters may "
            f"be too restrictive."
        )

    return results


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def fetch_buildings(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    """Fetch OSM building footprints."""
    filt = "building=* and geometry:polygon"
    result = query_osm_features(
        bbox, snapshot_timestamp,
        filters=[filt], event_date=event_date, settings=settings,
    )
    return result.get(filt, [])


def fetch_roads(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    """Fetch OSM road centrelines."""
    filt = "highway=* and geometry:line"
    result = query_osm_features(
        bbox, snapshot_timestamp,
        filters=[filt], event_date=event_date, settings=settings,
    )
    return result.get(filt, [])


def fetch_bridges(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    """Fetch OSM features tagged as bridges."""
    filt = "bridge=* and geometry:line"
    result = query_osm_features(
        bbox, snapshot_timestamp,
        filters=[filt], event_date=event_date, settings=settings,
    )
    return result.get(filt, [])


def fetch_settlements(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    """Fetch OSM settlement nodes (village, hamlet, town)."""
    filt = "place in (village,hamlet,town) and geometry:point"
    result = query_osm_features(
        bbox, snapshot_timestamp,
        filters=[filt], event_date=event_date, settings=settings,
    )
    return result.get(filt, [])


def fetch_destinations(
    bbox: BoundingBox,
    snapshot_timestamp: TimeLike,
    *,
    event_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch destination features for connectivity analysis."""
    town_filter = "place in (town,city) and geometry:point"
    hospital_filter = "amenity in (hospital,clinic) and geometry:point"
    result = query_osm_features(
        bbox, snapshot_timestamp,
        filters=[town_filter, hospital_filter],
        event_date=event_date, settings=settings,
    )
    return {
        "towns": result.get(town_filter, []),
        "hospitals": result.get(hospital_filter, []),
    }


__all__ = [
    "fetch_bridges",
    "fetch_buildings",
    "fetch_destinations",
    "fetch_roads",
    "fetch_settlements",
    "query_osm_features",
]