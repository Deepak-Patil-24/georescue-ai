"""Runtime configuration for GeoRescue AI.

Loads settings from environment variables (via ``python-dotenv`` if a
``.env`` file is present) and exposes them as a validated Pydantic model.

Design notes
------------
- Credentials are never required at import time. The application can start
  without any external API key; the LLM copilot falls back to a
  deterministic report generator.
- All paths are resolved relative to the project root, discovered from
  this file's location, so the package works regardless of the current
  working directory.
- No module in the package should read ``os.environ`` directly. Use
  :func:`get_settings` instead so validation and defaults are centralised.

LLM provider selection
----------------------
The copilot supports two providers:

- **Gemini** (Google AI Studio, free tier). Preferred when
  ``GEMINI_API_KEY`` is set.
- **DeepSeek** (paid). Used when ``DEEPSEEK_API_KEY`` is set and Gemini is
  not configured.

When neither key is set, the copilot runs in deterministic mode.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def _discover_project_root() -> Path:
    """Return the project root directory."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "requirements.txt").is_file() and (candidate / "app.py").is_file():
            return candidate
    return here.parent.parent.parent


PROJECT_ROOT: Path = _discover_project_root()

_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Validated runtime settings for GeoRescue AI."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- CDSE OAuth Client Credentials -------------------------------------
    cdse_client_id: Optional[str] = Field(default=None, alias="CDSE_CLIENT_ID")
    cdse_client_secret: Optional[str] = Field(default=None, alias="CDSE_CLIENT_SECRET")

    cdse_username: Optional[str] = Field(default=None, alias="CDSE_USERNAME")
    cdse_password: Optional[str] = Field(default=None, alias="CDSE_PASSWORD")

    cdse_token_url: str = Field(
        default=(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
            "protocol/openid-connect/token"
        ),
        alias="CDSE_TOKEN_URL",
    )
    cdse_stac_url: str = Field(
        default="https://catalogue.dataspace.copernicus.eu/stac",
        alias="CDSE_STAC_URL",
    )
    cdse_odata_url: str = Field(
        default="https://catalogue.dataspace.copernicus.eu/odata/v1",
        alias="CDSE_ODATA_URL",
    )

    # --- ohsome API --------------------------------------------------------
    ohsome_base_url: str = Field(
        default="https://api.heigit.org/ohsome-api/v2",
        alias="OHSOME_BASE_URL",
    )
    ohsome_api_key: Optional[str] = Field(default=None, alias="OHSOME_API_KEY")

    # --- Copernicus DEM ----------------------------------------------------
    copernicus_dem_s3_base: str = Field(
        default="https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/",
        alias="COPERNICUS_DEM_S3_BASE",
    )

    # --- Gemini (Google AI Studio, free tier, preferred) -------------------
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_api_base: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        alias="GEMINI_API_BASE",
    )
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # --- DeepSeek (paid, fallback) -----------------------------------------
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_API_BASE",
    )
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # --- Copilot mode ------------------------------------------------------
    # "auto"        -> use whichever LLM key is configured
    # "deterministic" -> never call an LLM
    copilot_mode: str = Field(default="auto", alias="GEORESCUE_COPILOT_MODE")

    # --- Runtime settings --------------------------------------------------
    log_level: str = Field(default="INFO", alias="GEORESCUE_LOG_LEVEL")
    data_dir: Path = Field(default=Path("data"), alias="GEORESCUE_DATA_DIR")
    output_dir: Path = Field(default=Path("outputs"), alias="GEORESCUE_OUTPUT_DIR")
    report_dir: Path = Field(default=Path("reports"), alias="GEORESCUE_REPORT_DIR")

    max_aoi_sqkm: float = Field(default=5000.0, alias="GEORESCUE_MAX_AOI_SQKM")
    fallback_projected_crs: str = Field(
        default="EPSG:3857",
        alias="GEORESCUE_FALLBACK_PROJECTED_CRS",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(
                f"GEORESCUE_LOG_LEVEL must be one of {sorted(allowed)}, got {v!r}"
            )
        return upper

    @field_validator("max_aoi_sqkm")
    @classmethod
    def _validate_max_aoi(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("GEORESCUE_MAX_AOI_SQKM must be positive")
        return v

    @field_validator("copilot_mode")
    @classmethod
    def _validate_copilot_mode(cls, v: str) -> str:
        allowed = {"auto", "deterministic", "llm"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(
                f"GEORESCUE_COPILOT_MODE must be one of {sorted(allowed)}, got {v!r}"
            )
        return lower

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_path(self) -> Path:
        return (PROJECT_ROOT / self.data_dir).resolve()

    @property
    def output_path(self) -> Path:
        return (PROJECT_ROOT / self.output_dir).resolve()

    @property
    def report_path(self) -> Path:
        return (PROJECT_ROOT / self.report_dir).resolve()

    @property
    def raw_data_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def interim_data_path(self) -> Path:
        return self.data_path / "interim"

    @property
    def processed_data_path(self) -> Path:
        return self.data_path / "processed"

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------
    @property
    def has_cdse_credentials(self) -> bool:
        if self.cdse_client_id and self.cdse_client_secret:
            return True
        return bool(self.cdse_username) and bool(self.cdse_password)

    @property
    def uses_cdse_oauth_client(self) -> bool:
        return bool(self.cdse_client_id) and bool(self.cdse_client_secret)

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_deepseek_key(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def has_llm_key(self) -> bool:
        """True when any LLM provider key is configured."""
        if self.copilot_mode == "deterministic":
            return False
        return self.has_gemini_key or self.has_deepseek_key

    @property
    def llm_provider(self) -> Optional[str]:
        """Return the active LLM provider name, or None for deterministic."""
        if self.copilot_mode == "deterministic":
            return None
        if self.has_gemini_key:
            return "gemini"
        if self.has_deepseek_key:
            return "deepseek"
        return None

    @property
    def llm_api_key(self) -> Optional[str]:
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        return None

    @property
    def llm_api_base(self) -> Optional[str]:
        if self.llm_provider == "gemini":
            return self.gemini_api_base
        if self.llm_provider == "deepseek":
            return self.deepseek_api_base
        return None

    @property
    def llm_model(self) -> Optional[str]:
        if self.llm_provider == "gemini":
            return self.gemini_model
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return None

    @property
    def has_ohsome_key(self) -> bool:
        return bool(self.ohsome_api_key)

    # ------------------------------------------------------------------
    # Side effects
    # ------------------------------------------------------------------
    def ensure_directories(self) -> None:
        for path in (
            self.data_path,
            self.raw_data_path,
            self.interim_data_path,
            self.processed_data_path,
            self.output_path,
            self.report_path,
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Could not create directory {path!r}. "
                    f"Check permissions. Underlying error: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear the cached settings. Intended for tests."""
    get_settings.cache_clear()