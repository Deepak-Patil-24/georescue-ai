"""Copernicus Data Space Ecosystem (CDSE) client.

Provides a single :class:`CopernicusClient` that handles:

- OAuth 2.0 authentication against the CDSE identity service, with
  automatic token refresh. Prefers the **Client Credentials** flow when
  ``CDSE_CLIENT_ID`` and ``CDSE_CLIENT_SECRET`` are configured, and falls
  back to the legacy password-grant flow otherwise.
- Catalogue queries via the CDSE OData ``Products`` endpoint.
- Streaming download of a product by ID.

Reference (documented CDSE endpoints)
-------------------------------------
- Token:    https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
- OData:    https://catalogue.dataspace.copernicus.eu/odata/v1/Products
- Download: https://download.dataspace.copernicus.eu/odata/v1/Products(<id>)/$value

Only documented CDSE flows are used. No other client IDs, secrets, or
undocumented endpoints are assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.data_access import (
    AuthenticationError,
    DataAccessError,
)

log = get_logger(__name__)

# Public client ID used by the legacy password-grant flow.
_CDSE_PUBLIC_CLIENT_ID = "cdse-public"

# Request timeouts (connect, read) in seconds.
_TIMEOUT_SEARCH = (10, 60)
_TIMEOUT_DOWNLOAD = (10, 300)
_TIMEOUT_TOKEN = (10, 30)


@dataclass
class _Token:
    access_token: str
    expires_at: float  # unix epoch seconds

    @property
    def is_valid(self) -> bool:
        # Refresh 60 s before expiry to avoid races.
        return time.time() < self.expires_at - 60.0


class CopernicusClient:
    """Client for the Copernicus Data Space Ecosystem.

    Construction does **not** require credentials. Authentication is
    deferred until the first catalogue or download call, so code paths
    that only build the client (e.g. UI capability checks) can do so
    safely without credentials.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._token: Optional[_Token] = None
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "GeoRescue-AI/0.1 (hackathon prototype)",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _ensure_credentials(self) -> None:
        if not self._settings.has_cdse_credentials:
            raise AuthenticationError(
                "Copernicus Data Space credentials are not configured. "
                "Register for a free account at https://dataspace.copernicus.eu, "
                "create an OAuth client under 'User settings' -> 'OAuth clients' "
                "with grant type 'Client Credentials', then set "
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in your .env file."
            )

    def _request_token(self) -> _Token:
        """Fetch a fresh OAuth access token.

        Uses Client Credentials when configured; otherwise uses the legacy
        password grant with the public ``cdse-public`` client ID.
        """
        self._ensure_credentials()

        if self._settings.uses_cdse_oauth_client:
            payload = {
                "grant_type": "client_credentials",
                "client_id": self._settings.cdse_client_id,
                "client_secret": self._settings.cdse_client_secret,
            }
            flow = "client_credentials"
        else:
            payload = {
                "client_id": _CDSE_PUBLIC_CLIENT_ID,
                "grant_type": "password",
                "username": self._settings.cdse_username,
                "password": self._settings.cdse_password,
            }
            flow = "password"

        try:
            response = self._session.post(
                self._settings.cdse_token_url,
                data=payload,
                timeout=_TIMEOUT_TOKEN,
            )
        except requests.RequestException as exc:
            raise AuthenticationError(
                f"Could not reach the CDSE identity service. "
                f"Check your internet connection. Underlying error: {exc}"
            ) from exc

        if response.status_code == 401:
            raise AuthenticationError(
                f"CDSE rejected the {flow} credentials (HTTP 401). "
                f"Verify CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in your .env file."
            )
        if response.status_code >= 400:
            raise AuthenticationError(
                f"CDSE token request ({flow}) failed with HTTP "
                f"{response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AuthenticationError(
                f"CDSE token response was not JSON: {response.text[:500]}"
            ) from exc

        access_token = data.get("access_token")
        expires_in = data.get("expires_in")
        if not access_token or not isinstance(expires_in, (int, float)):
            raise AuthenticationError(
                f"CDSE token response missing access_token or expires_in: {data}"
            )

        log.info("Obtained CDSE access token via %s flow (expires in %s s).",
                 flow, expires_in)
        return _Token(
            access_token=str(access_token),
            expires_at=time.time() + float(expires_in),
        )

    def _get_token(self, *, force_refresh: bool = False) -> str:
        if force_refresh or self._token is None or not self._token.is_valid:
            self._token = self._request_token()
        assert self._token is not None
        return self._token.access_token

    def _auth_headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token(force_refresh=force_refresh)}"}

    # ------------------------------------------------------------------
    # OData search
    # ------------------------------------------------------------------
    def search_products(
        self,
        *,
        collection: str,
        filter_expression: str,
        top: int = 100,
        order_by: Optional[str] = "ContentDate/Start asc",
    ) -> list[dict[str, Any]]:
        """Query the CDSE OData ``Products`` endpoint.

        Parameters
        ----------
        collection:
            CDSE collection name, e.g. ``"SENTINEL-1"`` or ``"SENTINEL-2"``.
        filter_expression:
            OData ``$filter`` expression built by the caller.
        top:
            Maximum number of products to return (paginated internally).
        order_by:
            OData ``$orderby`` expression, or ``None`` for default order.
        """
        if top <= 0:
            raise ValueError("top must be positive")

        url = f"{self._settings.cdse_odata_url}/Products"
        page_size = min(top, 100)
        collected: list[dict[str, Any]] = []
        skip = 0

        while len(collected) < top:
            # Use a list of tuples instead of a dict, because the OData API
            # expects repeated $expand parameters rather than a
            # comma-joined value.
            params: list[tuple[str, Any]] = [
                ("$filter", f"Collection/Name eq '{collection}' and ({filter_expression})"),
                ("$top", page_size),
                ("$skip", skip),
                ("$expand", "Attributes"),
                ("$expand", "Assets"),
            ]
            if order_by:
                params.append(("$orderby", order_by))

            try:
                response = self._session.get(
                    url,
                    params=params,
                    headers=self._auth_headers(),
                    timeout=_TIMEOUT_SEARCH,
                )
            except requests.RequestException as exc:
                raise DataAccessError(
                    f"Could not reach the CDSE catalogue. Check your "
                    f"internet connection. Underlying error: {exc}"
                ) from exc

            if response.status_code == 401:
                log.warning("CDSE returned 401; refreshing token and retrying once.")
                try:
                    response = self._session.get(
                        url,
                        params=params,
                        headers=self._auth_headers(force_refresh=True),
                        timeout=_TIMEOUT_SEARCH,
                    )
                except requests.RequestException as exc:
                    raise DataAccessError(f"CDSE catalogue retry failed: {exc}") from exc

            if response.status_code >= 400:
                raise DataAccessError(
                    f"CDSE catalogue query failed with HTTP "
                    f"{response.status_code}: {response.text[:500]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise DataAccessError(
                    f"CDSE catalogue returned non-JSON: {response.text[:500]}"
                ) from exc

            records = payload.get("value", [])
            if not records:
                break

            collected.extend(records)
            if len(records) < page_size:
                break
            skip += page_size

        return collected[:top]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download_product(
        self,
        product_id: str,
        destination: Path,
        *,
        chunk_size: int = 1024 * 1024,
        overwrite: bool = False,
    ) -> Path:
        """Download a product by ID to ``destination``.

        Writes atomically (``.part`` file renamed on success) so a partial
        download is never mistaken for a complete one. Returns the path to
        the completed file.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            log.info("Using cached product at %s.", destination)
            return destination

        url = (
            f"https://download.dataspace.copernicus.eu/odata/v1/"
            f"Products({product_id})/$value"
        )
        temp_path = destination.with_suffix(destination.suffix + ".part")

        log.info("Downloading product %s -> %s", product_id, destination.name)
        try:
            response = self._session.get(
                url,
                headers=self._auth_headers(),
                stream=True,
                timeout=_TIMEOUT_DOWNLOAD,
                allow_redirects=True,
            )

            if response.status_code == 401:
                log.warning("Download returned 401; refreshing token and retrying.")
                response.close()
                response = self._session.get(
                    url,
                    headers=self._auth_headers(force_refresh=True),
                    stream=True,
                    timeout=_TIMEOUT_DOWNLOAD,
                    allow_redirects=True,
                )

            if response.status_code >= 400:
                response.close()
                raise DataAccessError(
                    f"CDSE download failed with HTTP {response.status_code} "
                    f"for product {product_id}: {response.text[:500]}"
                )

            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with open(temp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
            response.close()

            if total and written < total:
                raise DataAccessError(
                    f"Download of {product_id} was truncated: "
                    f"{written} of {total} bytes received."
                )
        except requests.RequestException as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise DataAccessError(
                f"Download of {product_id} failed: {exc}"
            ) from exc
        except DataAccessError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        temp_path.replace(destination)
        log.info("Download complete: %s (%.1f MB)", destination.name,
                 destination.stat().st_size / (1024 * 1024))
        return destination

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "CopernicusClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["CopernicusClient"]