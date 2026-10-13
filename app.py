"""GeoRescue AI — Streamlit dashboard.

Main entry point for the interactive dashboard. Provides:

1. A **run selector** that switches between the precomputed Trishuli
   case study and any completed run under ``outputs/runs/``.
2. Nine view sections that read from the currently selected run.
3. A **Run new analysis** page that lets a rescuer pick an area by map
   drawing, coordinate entry, or place-name search; check satellite
   scene availability; confirm large downloads; and launch the full
   pipeline in a background thread.

The dashboard never writes to the precomputed Trishuli case-study
locations. New runs are written to unique folders under
``outputs/runs/<run_id>/`` and become visible in the run selector only
after they complete successfully.

On a deployed environment where the standard ``data/processed`` and
``outputs`` folders are not present (because they are gitignored), the
dashboard reads the small output files from the ``deploy_files/``
bundle instead. The "Run new analysis" page is shown but disabled,
because the free cloud tier does not provide persistent storage for
the ~2.4 GB satellite downloads.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the georescue package importable when the app runs from the
# repository root on Streamlit Cloud. Locally, PYTHONPATH=src handles
# this; on the cloud, there is no environment variable, so we add
# src/ to sys.path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from georescue.config import get_settings
from georescue.logging_config import configure_logging, get_logger
from georescue.models import BoundingBox, EventDefinition
from georescue.reporting.evidence import ATTRIBUTIONS, build_evidence
from georescue.reporting.llm_copilot import answer_question, generate_report
from georescue.reporting.situation_report import build_deterministic_report, write_report
from georescue.data_access.geocoding import GeocodingError, geocode_place_name
from georescue.pipeline import (
    DownloadConfirmationRequired,
    NoSuitableImageryError,
    PipelineError,
    PipelineResult,
    SceneAvailability,
    run_full_analysis,
    search_only,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GeoRescue AI — Flood Damage Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Defaults (preserved from the original dashboard)
# ---------------------------------------------------------------------------

DEFAULT_BBOX = (85.10, 27.60, 85.55, 28.05)
DEFAULT_EVENT_DATE = date(2026, 8, 26)
DEFAULT_AOI_NAME = "Trishuli corridor (case study)"
TRISHULI_RUN_ID = "trishuli_precomputed"
TRISHULI_DISPLAY = "Trishuli case study (precomputed)"

SUGGESTED_QUESTIONS = [
    "Which settlements are potentially cut off?",
    "How many buildings intersect the flood extent?",
    "What is the total affected road length?",
    "What are the biggest limitations?",
    "What should we investigate first?",
]

DOWNLOAD_CONFIRM_MB = 800.0


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------

def _resolve_data_roots() -> tuple[Path, Path]:
    """Return (processed_root, outputs_root) for the current environment.

    Local environment: standard paths ``data/processed`` and ``outputs``.
    Cloud deployment: the small bundle under ``deploy_files/``.

    The decision is made by checking whether the standard
    ``data/processed`` folder contains the precomputed flood extent.
    """
    project_root = get_settings().project_root
    standard_processed = project_root / "data" / "processed"
    standard_outputs = project_root / "outputs"

    bundle_processed = project_root / "deploy_files" / "data" / "processed"
    bundle_outputs = project_root / "deploy_files" / "outputs"

    if (standard_processed / "candidate_flood_extent.geojson").is_file():
        return standard_processed, standard_outputs

    if (bundle_processed / "candidate_flood_extent.geojson").is_file():
        return bundle_processed, bundle_outputs

    return standard_processed, standard_outputs


def _is_cloud_deploy() -> bool:
    """True when the app is running from the deploy_files bundle."""
    project_root = get_settings().project_root
    standard = project_root / "data" / "processed" / "candidate_flood_extent.geojson"
    bundle = (
        project_root / "deploy_files" / "data" / "processed"
        / "candidate_flood_extent.geojson"
    )
    return (not standard.is_file()) and bundle.is_file()


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """All paths for a single selectable run."""

    run_id: str
    display_name: str
    is_precomputed: bool
    run_dir: Optional[Path]

    flood_extent_geojson: Optional[Path]
    flood_mask_path: Optional[Path]
    infrastructure_summary: Optional[Path]
    connectivity_summary: Optional[Path]

    buildings_geojson: Optional[Path]
    roads_geojson: Optional[Path]
    bridges_geojson: Optional[Path]
    disconnected_geojson: Optional[Path]
    disrupted_roads_geojson: Optional[Path]

    reports_dir: Optional[Path]

    event_date: Optional[date] = None
    aoi_bbox: Optional[BoundingBox] = None
    aoi_name: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _first_existing(*candidates: Optional[Path]) -> Optional[Path]:
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


def _list_precomputed_trishuli(settings) -> RunContext:
    """Return the RunContext for the original Trishuli case study."""
    processed, outputs = _resolve_data_roots()
    reports = settings.report_path

    return RunContext(
        run_id=TRISHULI_RUN_ID,
        display_name=TRISHULI_DISPLAY,
        is_precomputed=True,
        run_dir=None,
        flood_extent_geojson=_first_existing(
            processed / "candidate_flood_extent.geojson",
        ),
        flood_mask_path=_first_existing(
            processed / "candidate_flood_mask.tif",
        ),
        infrastructure_summary=_first_existing(
            outputs / "infrastructure" / "infrastructure_summary.json",
        ),
        connectivity_summary=_first_existing(
            outputs / "connectivity" / "connectivity_summary.json",
        ),
        buildings_geojson=_first_existing(
            outputs / "infrastructure" / "buildings_exposed.geojson",
        ),
        roads_geojson=_first_existing(
            outputs / "infrastructure" / "roads_exposed.geojson",
        ),
        bridges_geojson=_first_existing(
            outputs / "infrastructure" / "bridges_exposed.geojson",
        ),
        disconnected_geojson=_first_existing(
            outputs / "connectivity" / "disconnected_settlements.geojson",
        ),
        disrupted_roads_geojson=_first_existing(
            outputs / "connectivity" / "disrupted_roads.geojson",
        ),
        reports_dir=reports,
        event_date=date(2026, 8, 26),
        aoi_name=DEFAULT_AOI_NAME,
        warnings=[],
    )


def _list_completed_runs(settings) -> list[RunContext]:
    """Enumerate completed runs under ``outputs/runs/``."""
    _, outputs = _resolve_data_roots()
    runs_root = outputs / "runs"
    if not runs_root.is_dir():
        return []

    contexts: list[RunContext] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        flood_geojson = run_dir / "processed" / "candidate_flood_extent.geojson"
        if not manifest_path.is_file() or not flood_geojson.is_file():
            continue

        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            continue

        event_meta = manifest.get("event") or {}
        aoi_name = event_meta.get("aoi_name") or run_dir.name
        event_date_str = event_meta.get("event_date")
        try:
            event_date_val = date.fromisoformat(event_date_str) if event_date_str else None
        except (ValueError, TypeError):
            event_date_val = None

        bbox_list = event_meta.get("bbox")
        aoi_bbox = None
        if isinstance(bbox_list, list) and len(bbox_list) == 4:
            try:
                aoi_bbox = BoundingBox(
                    min_lon=float(bbox_list[0]),
                    min_lat=float(bbox_list[1]),
                    max_lon=float(bbox_list[2]),
                    max_lat=float(bbox_list[3]),
                )
            except Exception:
                aoi_bbox = None

        contexts.append(RunContext(
            run_id=run_dir.name,
            display_name=(
                f"{aoi_name} — "
                f"{event_date_val.isoformat() if event_date_val else 'unknown date'} "
                f"({run_dir.name})"
            ),
            is_precomputed=False,
            run_dir=run_dir,
            flood_extent_geojson=flood_geojson,
            flood_mask_path=_first_existing(run_dir / "processed" / "candidate_flood_mask.tif"),
            infrastructure_summary=_first_existing(run_dir / "infrastructure" / "infrastructure_summary.json"),
            connectivity_summary=_first_existing(run_dir / "connectivity" / "connectivity_summary.json"),
            buildings_geojson=_first_existing(run_dir / "infrastructure" / "buildings_exposed.geojson"),
            roads_geojson=_first_existing(run_dir / "infrastructure" / "roads_exposed.geojson"),
            bridges_geojson=_first_existing(run_dir / "infrastructure" / "bridges_exposed.geojson"),
            disconnected_geojson=_first_existing(run_dir / "connectivity" / "disconnected_settlements.geojson"),
            disrupted_roads_geojson=_first_existing(run_dir / "connectivity" / "disrupted_roads.geojson"),
            reports_dir=settings.report_path,
            event_date=event_date_val,
            aoi_bbox=aoi_bbox,
            aoi_name=aoi_name,
            warnings=[],
        ))

    contexts.sort(key=lambda c: c.run_dir.stat().st_mtime if c.run_dir else 0, reverse=True)
    return contexts


def list_available_runs(settings) -> list[RunContext]:
    """Return every selectable run, newest first, precomputed last."""
    runs = _list_completed_runs(settings)
    runs.append(_list_precomputed_trishuli(settings))
    return runs


def get_selected_run(settings) -> RunContext:
    """Return the RunContext for the currently selected run id."""
    all_runs = list_available_runs(settings)
    selected_id = st.session_state.get("selected_run_id")

    for ctx in all_runs:
        if ctx.run_id == selected_id:
            return ctx

    for ctx in all_runs:
        if not ctx.is_precomputed:
            st.session_state["selected_run_id"] = ctx.run_id
            return ctx

    precomputed = _list_precomputed_trishuli(settings)
    st.session_state["selected_run_id"] = precomputed.run_id
    return precomputed


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_json_cached(path_str: str, mtime: float) -> Optional[dict[str, Any]]:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def load_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _load_json_cached(str(path), mtime)


@st.cache_data(show_spinner=False)
def _load_geojson_cached(path_str: str, mtime: float) -> Optional[gpd.GeoDataFrame]:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        return gdf.to_crs(epsg=4326)
    except Exception as exc:
        log.warning("Could not read GeoJSON %s: %s", path, exc)
        return None


def load_geojson(path: Optional[Path]) -> Optional[gpd.GeoDataFrame]:
    if path is None or not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _load_geojson_cached(str(path), mtime)


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def render_metric(label: str, value: Any, unit: str = "") -> None:
    if value is None:
        st.metric(label, "not calculated")
    elif isinstance(value, float):
        st.metric(label, f"{value:,.2f} {unit}".strip())
    elif isinstance(value, int):
        st.metric(label, f"{value:,} {unit}".strip())
    else:
        st.metric(label, f"{value} {unit}".strip())


def _build_folium_map(
    flood_gdf, buildings_gdf, roads_gdf, bridges_gdf,
    disconnected_gdf, disrupted_roads_gdf, center, zoom=11,
) -> folium.Map:
    m = folium.Map(
        location=[center[1], center[0]],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        name="OpenStreetMap Humanitarian",
        attr="© OpenStreetMap contributors, Tiles style by HOT",
        max_zoom=19,
    ).add_to(m)

    if disrupted_roads_gdf is not None and not disrupted_roads_gdf.empty:
        folium.GeoJson(
            disrupted_roads_gdf,
            name=(
                f"Red — Road sections that may be impassable "
                f"({len(disrupted_roads_gdf)})"
            ),
            style_function=lambda _: {"color": "#d62728", "weight": 2.5, "opacity": 0.85},
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ("osm_id", "highway", "length_m")
                        if c in disrupted_roads_gdf.columns],
                aliases=["OSM ID", "Road type", "Length (m)"], sticky=False,
            ),
        ).add_to(m)

    if roads_gdf is not None and not roads_gdf.empty:
        folium.GeoJson(
            roads_gdf,
            name=(
                f"Orange — Road sections overlapping possibly flooded "
                f"areas ({len(roads_gdf)})"
            ),
            style_function=lambda _: {"color": "#ff7f0e", "weight": 1.6, "opacity": 0.9},
        ).add_to(m)

    if bridges_gdf is not None and not bridges_gdf.empty:
        folium.GeoJson(
            bridges_gdf,
            name=f"Brown — Bridges possibly affected ({len(bridges_gdf)})",
            style_function=lambda _: {"color": "#8c564b", "weight": 4, "opacity": 1.0},
        ).add_to(m)

    if buildings_gdf is not None and not buildings_gdf.empty:
        folium.GeoJson(
            buildings_gdf,
            name=f"Blue — Buildings inside possibly flooded areas ({len(buildings_gdf)})",
            style_function=lambda _: {
                "fillColor": "#1f77b4", "color": "#1f77b4",
                "weight": 0.4, "fillOpacity": 0.55,
            },
        ).add_to(m)

    if flood_gdf is not None and not flood_gdf.empty:
        folium.GeoJson(
            flood_gdf,
            name=(
                f"Blue — Areas identified as possibly flooded "
                f"({len(flood_gdf)} polygons)"
            ),
            style_function=lambda _: {
                "fillColor": "#1e90ff", "color": "#0b5ed7",
                "weight": 1.2, "fillOpacity": 0.45,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ("id", "area_sqkm", "area_hectares")
                        if c in flood_gdf.columns],
                aliases=["ID", "Area (km²)", "Area (ha)"], sticky=False,
            ),
        ).add_to(m)

    if disconnected_gdf is not None and not disconnected_gdf.empty:
        folium.GeoJson(
            disconnected_gdf,
            name=(
                f"Red dot — Settlements that may be cut off "
                f"({len(disconnected_gdf)})"
            ),
            marker=folium.CircleMarker(
                radius=8, color="#000000", weight=2,
                fill=True, fillColor="#ff2d2d", fillOpacity=0.95,
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ("osm_id", "baseline_distance_m")
                        if c in disconnected_gdf.columns],
                aliases=["OSM ID", "Mapped road distance before the event (m)"],
                sticky=False,
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(settings) -> dict[str, Any]:
    st.sidebar.title("🛰️ GeoRescue AI")
    st.sidebar.caption(
        "Research prototype for the Multimodal AI Hackathon 2026, Track B. "
        "Not an operational emergency-response system."
    )

    st.sidebar.divider()

    # --- Run selector --------------------------------------------------
    all_runs = list_available_runs(settings)
    run_labels = [ctx.display_name for ctx in all_runs]
    run_ids = [ctx.run_id for ctx in all_runs]

    current_id = st.session_state.get("selected_run_id")
    try:
        default_index = run_ids.index(current_id)
    except (ValueError, AttributeError):
        default_index = 0

    st.sidebar.subheader("Currently viewing")
    selected_label = st.sidebar.selectbox(
        "Which assessment would you like to view?",
        options=run_labels,
        index=default_index,
        key="run_selector_widget",
    )
    selected_index = run_labels.index(selected_label)
    st.session_state["selected_run_id"] = run_ids[selected_index]

    # --- Navigation ----------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("Sections")

    pages = [
        "Run new analysis",
        "Overview",
        "Map",
        "Flood mapping",
        "Infrastructure exposure",
        "Access disruption",
        "Situation report",
        "Copilot",
        "Data provenance",
        "Export",
    ]
    page = st.sidebar.radio("Section", pages, key="page_selector")

    # --- Inputs (preserved) -------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("Analysis parameters")
    aoi_name = st.sidebar.text_input("Area name", value=DEFAULT_AOI_NAME)
    min_lon = st.sidebar.number_input("West longitude", value=DEFAULT_BBOX[0], format="%.4f")
    min_lat = st.sidebar.number_input("South latitude", value=DEFAULT_BBOX[1], format="%.4f")
    max_lon = st.sidebar.number_input("East longitude", value=DEFAULT_BBOX[2], format="%.4f")
    max_lat = st.sidebar.number_input("North latitude", value=DEFAULT_BBOX[3], format="%.4f")

    event_date = st.sidebar.date_input("Event date", value=DEFAULT_EVENT_DATE)
    pre_days = st.sidebar.slider("Days to search before the event", 3, 30, 12)
    post_days = st.sidebar.slider("Days to search after the event", 3, 30, 12)

    # --- Capability status --------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("System status")
    st.sidebar.write(
        f"**Satellite access (CDSE):** "
        f"{'✅ configured' if settings.has_cdse_credentials else '❌ missing credentials'}"
    )
    st.sidebar.write(
        f"**OpenStreetMap access (ohsome):** "
        f"{'✅ configured' if settings.has_ohsome_key else '❌ missing key'}"
    )
    llm_label = settings.llm_provider or "deterministic only"
    st.sidebar.write(
        f"**Report copilot:** "
        f"{'✅ ' + llm_label if settings.has_llm_key else '⚠️ ' + llm_label}"
    )

    return {
        "aoi_name": aoi_name,
        "min_lon": min_lon, "min_lat": min_lat,
        "max_lon": max_lon, "max_lat": max_lat,
        "event_date": event_date,
        "pre_days": pre_days, "post_days": post_days,
        "page": page,
    }


# ---------------------------------------------------------------------------
# "Run new analysis" page
# ---------------------------------------------------------------------------

def _run_pipeline_thread(
    event: EventDefinition,
    confirm_download: bool,
    progress_queue: "queue.Queue[tuple[str, Any]]",
    result_holder: dict,
) -> None:
    def _progress(message: str, fraction: float) -> None:
        progress_queue.put(("progress", (message, fraction)))

    try:
        result = run_full_analysis(
            event,
            confirm_download=confirm_download,
            progress=_progress,
        )
        result_holder["result"] = result
        progress_queue.put(("done", result))
    except Exception as exc:
        result_holder["error"] = exc
        progress_queue.put(("error", exc))


def section_run_new_analysis(settings, inputs: dict[str, Any]) -> None:
    st.header("Run a new flood assessment")

    # On the free cloud deployment, persistent storage is not available.
    # Show the interface, but disable the run with a clear explanation.
    if _is_cloud_deploy():
        st.warning(
            "**This deployed version cannot run new analyses.** "
            "A full analysis downloads approximately 2.4 GB of Sentinel-1 "
            "imagery and needs persistent cloud storage to keep it between "
            "runs. The free hosting tier does not provide that storage. "
            "The full workflow — search, download, and analysis — is "
            "demonstrated in the project video and runs locally. "
            "The interface is shown below so you can see how it works."
        )

    st.caption(
        "Choose an area of interest and an event date. The system will "
        "search for suitable satellite imagery before downloading anything. "
        "You can select the area by drawing on the map, typing coordinates, "
        "or searching for a place name."
    )

    st.subheader("1. Choose the area to analyse")

    aoi_method = st.radio(
        "How would you like to choose the area?",
        [
            "Search by place name",
            "Enter coordinates",
            "Draw on a map",
        ],
        horizontal=True,
        key="aoi_method",
    )

    bbox: Optional[BoundingBox] = None
    aoi_label = ""

    if aoi_method == "Search by place name":
        place_query = st.text_input(
            "Place name",
            value="",
            placeholder="e.g. Biratnagar, Syapru Besi, Kathmandu",
        )
        if st.button("Find this place"):
            try:
                result = geocode_place_name(place_query)
                bbox = result.bbox
                aoi_label = result.display_name
                st.session_state["new_aoi_bbox"] = bbox.as_tuple()
                st.session_state["new_aoi_label"] = aoi_label
                st.success(
                    f"Found: {result.display_name}. Bounding box filled below."
                )
            except GeocodingError as exc:
                st.error(str(exc))

    elif aoi_method == "Enter coordinates":
        st.caption(
            "Enter the four corners of the area. West must be less than "
            "East. South must be less than North."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            west = st.number_input("West longitude", value=86.90, format="%.4f", key="new_west")
        with c2:
            south = st.number_input("South latitude", value=26.40, format="%.4f", key="new_south")
        with c3:
            east = st.number_input("East longitude", value=87.20, format="%.4f", key="new_east")
        with c4:
            north = st.number_input("North latitude", value=26.70, format="%.4f", key="new_north")
        try:
            bbox = BoundingBox(
                min_lon=west, min_lat=south, max_lon=east, max_lat=north,
            )
            aoi_label = f"Manual bbox ({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f})"
            st.session_state["new_aoi_bbox"] = bbox.as_tuple()
            st.session_state["new_aoi_label"] = aoi_label
        except Exception as exc:
            st.error(f"Invalid bounding box: {exc}")
            bbox = None

    else:
        st.caption(
            "Draw a rectangle on the map. The coordinates are captured "
            "automatically when you release the mouse."
        )
        draw_map = folium.Map(
            location=[27.7, 85.3], zoom_start=8, tiles=None,
        )
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            name="OpenStreetMap Humanitarian",
            attr="© OpenStreetMap contributors, Tiles style by HOT",
            max_zoom=19,
        ).add_to(draw_map)
        folium.plugins.Draw(
            export=False,
            draw_options={
                "polyline": False, "polygon": True,
                "circle": False, "circlemarker": False,
                "marker": False,
                "rectangle": {"shapeOptions": {"color": "#1e90ff"}},
            },
            edit_options={"edit": False, "remove": True},
        ).add_to(draw_map)

        draw_result = st_folium(
            draw_map, width=None, height=500,
            returned_objects=["all_drawings"],
            key="draw_map",
        )

        if draw_result and draw_result.get("all_drawings"):
            drawings = draw_result["all_drawings"]
            if isinstance(drawings, dict):
                drawings = [drawings]
            latest = drawings[-1]
            geom = latest.get("geometry") or {}
            geom_type = geom.get("type")
            coords = geom.get("coordinates") or []

            try:
                if geom_type == "Polygon":
                    ring = coords[0]
                else:
                    ring = []
                if ring:
                    lons = [pt[0] for pt in ring]
                    lats = [pt[1] for pt in ring]
                    bbox = BoundingBox(
                        min_lon=min(lons), min_lat=min(lats),
                        max_lon=max(lons), max_lat=max(lats),
                    )
                    aoi_label = "Drawn polygon"
                    st.session_state["new_aoi_bbox"] = bbox.as_tuple()
                    st.session_state["new_aoi_label"] = aoi_label
                    st.success(
                        f"Drawn area captured: "
                        f"{bbox.min_lon:.3f}, {bbox.min_lat:.3f}, "
                        f"{bbox.max_lon:.3f}, {bbox.max_lat:.3f}"
                    )
            except Exception as exc:
                st.error(f"Could not read the drawn area: {exc}")

    current_bbox = bbox
    if current_bbox is None and st.session_state.get("new_aoi_bbox"):
        try:
            tup = st.session_state["new_aoi_bbox"]
            current_bbox = BoundingBox(
                min_lon=tup[0], min_lat=tup[1], max_lon=tup[2], max_lat=tup[3],
            )
        except Exception:
            current_bbox = None

    if current_bbox is not None:
        st.info(
            f"**Selected area:** {aoi_label or 'unnamed'}  \n"
            f"West {current_bbox.min_lon:.4f} · "
            f"South {current_bbox.min_lat:.4f} · "
            f"East {current_bbox.max_lon:.4f} · "
            f"North {current_bbox.max_lat:.4f}"
        )

    st.subheader("2. Choose the event date and search window")
    event_date = st.date_input(
        "Event date", value=DEFAULT_EVENT_DATE, key="new_event_date",
    )
    c1, c2 = st.columns(2)
    with c1:
        pre_days = st.slider(
            "Days to search before the event", 3, 30, 12, key="new_pre_days",
        )
    with c2:
        post_days = st.slider(
            "Days to search after the event", 3, 30, 12, key="new_post_days",
        )

    st.subheader("3. Check whether suitable satellite imagery is available")
    st.caption(
        "Searching checks the Copernicus catalogue. It does not download "
        "anything yet."
    )

    if st.button("Search for imagery", type="primary"):
        if current_bbox is None:
            st.error("Please choose an area of interest first.")
        else:
            with st.spinner("Searching the satellite catalogue..."):
                try:
                    availability = search_only(
                        current_bbox, event_date,
                        pre_days=pre_days, post_days=post_days,
                        aoi_name=aoi_label or "new_aoi",
                        settings=settings,
                    )
                    st.session_state["new_availability"] = availability
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    st.session_state.pop("new_availability", None)

    availability: Optional[SceneAvailability] = st.session_state.get("new_availability")

    if availability is not None:
        _render_availability(availability)

        st.subheader("4. Confirm and run")

        size_mb = availability.estimated_download_mb
        needs_confirm = size_mb > DOWNLOAD_CONFIRM_MB

        if size_mb > 0:
            est_minutes = size_mb / 600.0
            duration_text = f"approximately {est_minutes:.0f} minutes"
        else:
            duration_text = "unknown"

        confirm_ok = True
        if needs_confirm:
            st.warning(
                f"This run will download approximately **{size_mb/1024:.2f} GB** "
                f"of satellite data. Estimated download time: {duration_text}. "
                f"The analysis will then take a few minutes to run."
            )
            confirm_ok = st.checkbox(
                f"I understand this will download about {size_mb/1024:.2f} GB "
                f"and may take {duration_text}.",
                key="download_confirm",
            )
        else:
            st.info(
                f"Estimated download: {size_mb:.1f} MB. "
                f"Estimated duration: {duration_text}."
            )

        running = st.session_state.get("active_run_thread") is not None
        block_run_in_cloud = _is_cloud_deploy()

        run_button = st.button(
            "Run flood assessment",
            type="primary",
            disabled=(
                (not availability.has_pair)
                or (not confirm_ok)
                or running
                or block_run_in_cloud
            ),
        )

        if block_run_in_cloud:
            st.info(
                "This button is disabled on the deployed version. "
                "Run the project locally to use it."
            )
        elif running:
            st.info(
                "A previous analysis is still running. Please wait for it "
                "to finish before starting another."
            )

        if run_button and availability.has_pair and not block_run_in_cloud:
            _start_run(settings, availability, event_date, pre_days, post_days,
                       aoi_label or "new_aoi")
            st.rerun()

    _render_active_run_progress(settings)

    if st.button("Refresh this page"):
        st.rerun()


def _render_availability(availability: SceneAvailability) -> None:
    st.markdown("### Search results")
    st.write(f"Pre-event scenes found: **{len(availability.pre_scenes)}**")
    st.write(f"Post-event scenes found: **{len(availability.post_scenes)}**")

    if availability.selected_pair is None:
        st.error(
            "No compatible pre-event / post-event pair was found. "
            "Try widening the search window, moving the area, or choosing "
            "different dates."
        )
        for w in availability.warnings:
            st.warning(w)
        return

    pre = availability.selected_pair.pre
    post = availability.selected_pair.post
    days_apart = abs((post.acquisition_datetime - pre.acquisition_datetime).days)

    st.success(
        f"Found a compatible pair: {pre.acquisition_datetime.date()} "
        f"({pre.orbit_direction}, orbit {pre.relative_orbit}) and "
        f"{post.acquisition_datetime.date()} "
        f"({post.orbit_direction}, orbit {post.relative_orbit}), "
        f"{days_apart} days apart."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Estimated download", f"{availability.estimated_download_mb:.0f} MB")
    with c2:
        st.metric("Pre-event scene", f"{pre.acquisition_datetime.date()}")
    with c3:
        st.metric("Post-event scene", f"{post.acquisition_datetime.date()}")

    with st.expander("Technical details of the selected scenes"):
        st.write(f"**Pre-event product:** `{pre.tile_or_granule_id}`")
        st.write(f"**Pre-event acquisition:** {pre.acquisition_datetime.isoformat()}")
        st.write(f"**Pre-event orbit:** {pre.orbit_direction} / {pre.relative_orbit}")
        st.write(f"**Pre-event polarisation:** {pre.polarisation}")
        st.write(f"**Post-event product:** `{post.tile_or_granule_id}`")
        st.write(f"**Post-event acquisition:** {post.acquisition_datetime.isoformat()}")
        st.write(f"**Post-event orbit:** {post.orbit_direction} / {post.relative_orbit}")
        st.write(f"**Post-event polarisation:** {post.polarisation}")
        if availability.selection_notes:
            st.write("**Selection notes:**")
            for note in availability.selection_notes:
                st.write(f"- {note}")

    for w in availability.warnings:
        st.warning(w)


def _start_run(
    settings,
    availability: SceneAvailability,
    event_date: date,
    pre_days: int,
    post_days: int,
    aoi_name: str,
) -> None:
    event = EventDefinition(
        aoi=availability.aoi,
        event_date=event_date,
        pre_days=pre_days,
        post_days=post_days,
        aoi_name=aoi_name,
    )
    progress_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    result_holder: dict = {}
    confirm_download = availability.estimated_download_mb > DOWNLOAD_CONFIRM_MB

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(event, confirm_download, progress_queue, result_holder),
        daemon=True,
    )
    thread.start()

    st.session_state["active_run_thread"] = thread
    st.session_state["active_run_queue"] = progress_queue
    st.session_state["active_run_result"] = result_holder
    st.session_state["active_run_availability"] = availability
    st.session_state["active_run_log"] = []
    st.session_state["active_run_fraction"] = 0.0
    st.session_state["active_run_message"] = "Starting..."
    st.session_state["active_run_error"] = None
    st.session_state["active_run_done_result"] = None


def _render_active_run_progress(settings) -> None:
    progress_queue: Optional["queue.Queue[tuple[str, Any]]"] = st.session_state.get("active_run_queue")
    if progress_queue is None:
        return

    log_lines = st.session_state.get("active_run_log", [])
    try:
        while True:
            kind, payload = progress_queue.get_nowait()
            if kind == "progress":
                message, fraction = payload
                st.session_state["active_run_message"] = message
                st.session_state["active_run_fraction"] = float(fraction)
                log_lines.append(f"[{fraction*100:5.1f}%] {message}")
            elif kind == "done":
                st.session_state["active_run_done_result"] = payload
                st.session_state["active_run_message"] = "Completed."
                st.session_state["active_run_fraction"] = 1.0
            elif kind == "error":
                st.session_state["active_run_error"] = payload
                st.session_state["active_run_message"] = f"Failed: {payload}"
    except queue.Empty:
        pass

    st.session_state["active_run_log"] = log_lines

    st.markdown("### Progress")
    st.progress(st.session_state.get("active_run_fraction", 0.0))
    st.write(st.session_state.get("active_run_message", ""))

    with st.expander("Stage-by-stage log", expanded=False):
        for line in log_lines:
            st.write(line)

    done_result = st.session_state.get("active_run_done_result")
    error = st.session_state.get("active_run_error")

    if done_result is not None:
        new_run_id = done_result.run_id
        st.session_state["selected_run_id"] = new_run_id
        st.success(
            f"Analysis complete. The new run is now selected in the sidebar "
            f"as {new_run_id}."
        )
        _clear_active_run()

    elif error is not None:
        st.error(f"Analysis failed: {error}")
        if st.button("Clear error and start over"):
            _clear_active_run()
            st.rerun()

    else:
        thread = st.session_state.get("active_run_thread")
        if thread is not None and thread.is_alive():
            time.sleep(1.0)
            st.rerun()


def _clear_active_run() -> None:
    for key in [
        "active_run_thread", "active_run_queue", "active_run_result",
        "active_run_availability", "active_run_log", "active_run_fraction",
        "active_run_message", "active_run_error", "active_run_done_result",
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Read-only view sections
# ---------------------------------------------------------------------------

def section_overview(run: RunContext, flood_gdf, infra_summary, conn_summary) -> None:
    st.header("Overview")
    if run.is_precomputed:
        st.caption(
            "This is the precomputed Trishuli case study. Use the sidebar "
            "selector to switch runs, or run a new analysis."
        )
    else:
        st.caption(f"Currently viewing run: **{run.run_id}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if flood_gdf is not None and not flood_gdf.empty and "area_sqkm" in flood_gdf.columns:
            render_metric(
                "Areas possibly flooded",
                round(float(flood_gdf["area_sqkm"].sum()), 3), "km²",
            )
        else:
            render_metric("Areas possibly flooded", None)
    with c2:
        if infra_summary:
            render_metric(
                "Buildings inside those areas",
                infra_summary.get("buildings", {}).get("buildings_intersecting_count"),
            )
        else:
            render_metric("Buildings inside those areas", None)
    with c3:
        if infra_summary:
            render_metric(
                "Road length overlapping those areas",
                infra_summary.get("roads", {}).get("roads_affected_length_km"), "km",
            )
        else:
            render_metric("Road length overlapping those areas", None)
    with c4:
        if conn_summary:
            render_metric(
                "Settlements that may be cut off",
                conn_summary.get("counts", {}).get("newly_disconnected"),
            )
        else:
            render_metric("Settlements that may be cut off", None)

    if run.aoi_bbox is not None:
        st.caption(
            f"Area: **{run.aoi_name or 'unnamed'}** — "
            f"West {run.aoi_bbox.min_lon:.4f}, South {run.aoi_bbox.min_lat:.4f}, "
            f"East {run.aoi_bbox.max_lon:.4f}, North {run.aoi_bbox.max_lat:.4f}"
        )


def section_map(run: RunContext, gdfs: dict) -> None:
    st.header("Map")
    st.caption(
        "Toggle the layers on or off using the map control in the top-right "
        "corner. The label of each layer begins with its colour on the map, "
        "so the control doubles as a legend. All shapes are the actual "
        "outputs of the current run."
    )
    flood_gdf = gdfs.get("flood")
    if flood_gdf is None or flood_gdf.empty:
        st.info("No flood extent available for this run.")
        return
    bounds = flood_gdf.total_bounds
    center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    m = _build_folium_map(
        flood_gdf=flood_gdf,
        buildings_gdf=gdfs.get("buildings"),
        roads_gdf=gdfs.get("roads"),
        bridges_gdf=gdfs.get("bridges"),
        disconnected_gdf=gdfs.get("disconnected"),
        disrupted_roads_gdf=gdfs.get("disrupted_roads"),
        center=center,
    )
    st_folium(m, width=None, height=650, returned_objects=[])


def section_flood(run: RunContext, flood_gdf) -> None:
    st.header("Flood mapping")
    st.caption(
        "These are the **areas the satellite analysis identified as "
        "possibly flooded**. They are a candidate flood extent, not a "
        "confirmed flood boundary."
    )
    if flood_gdf is None or flood_gdf.empty:
        st.info("No flood extent available for this run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if "area_sqkm" in flood_gdf.columns:
            render_metric(
                "Total area identified as possibly flooded",
                round(float(flood_gdf["area_sqkm"].sum()), 3), "km²",
            )
        else:
            render_metric("Total area identified as possibly flooded", None)
    with c2:
        st.metric("Number of separate patches", f"{len(flood_gdf):,}")

    st.subheader("What this means for a responder")
    st.markdown(
        "- The blue areas on the map show where the radar signal changed "
        "in a way consistent with flooding between the two satellite "
        "passes.\n"
        "- This is a satellite-based estimate. Ground verification is "
        "needed before treating any area as confirmed flooded.\n"
        "- Very steep terrain, permanent water bodies, and vegetation can "
        "produce false positives or false negatives."
    )
    with st.expander("Technical details (method, parameters, sources)"):
        st.markdown(
            "- **Method:** Sentinel-1 VV backscatter ratio change detection.\n"
            "- **Threshold:** pixels with a post/pre ratio below the "
            "configured threshold are flagged, with a slope mask and "
            "minimum mapping unit applied.\n"
            "- **Sources:** Copernicus Sentinel-1 (CDSE), Copernicus "
            "WorldDEM-30, and OpenStreetMap (pre-event snapshot).\n"
            "- **Reproducibility:** the exact thresholds and parameters are "
            "recorded in the run's manifest.json."
        )


def section_infrastructure(run: RunContext, infra_summary) -> None:
    st.header("Infrastructure exposure")
    st.caption(
        "These counts describe **infrastructure that lies inside or "
        "overlaps the areas identified as possibly flooded**. They are "
        "potential exposure, not confirmed damage."
    )
    if not infra_summary:
        st.info("No infrastructure summary available for this run.")
        return

    b = infra_summary.get("buildings", {}) or {}
    r = infra_summary.get("roads", {}) or {}
    br = infra_summary.get("bridges", {}) or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric(
            "Buildings inside those areas",
            b.get("buildings_intersecting_count"),
        )
        render_metric(
            "Total building footprint inside",
            b.get("buildings_exposed_area_hectares"), "ha",
        )
    with c2:
        render_metric(
            "Road sections overlapping those areas",
            r.get("roads_intersecting_count"),
        )
        render_metric(
            "Length of road overlapping",
            r.get("roads_affected_length_km"), "km",
        )
    with c3:
        render_metric(
            "Bridges overlapping those areas",
            br.get("bridges_intersecting_count"),
        )

    warnings = (
        (b.get("warnings") or [])
        + (r.get("warnings") or [])
        + (br.get("warnings") or [])
    )
    for w in warnings:
        st.warning(w)

    st.subheader("What this means for a responder")
    st.markdown(
        "- A building in this list has its footprint touching a possibly "
        "flooded area. It has not necessarily sustained damage.\n"
        "- A road in this list has a section overlapping a possibly flooded "
        "area. It may or may not be impassable.\n"
        "- Buildings and bridges listed here are worth checking on the "
        "ground before deciding on access routes."
    )
    with st.expander("Technical details"):
        st.markdown(
            f"- **OSM snapshot used:** {infra_summary.get('osm_snapshot', 'n/a')}\n"
            f"- **Tile size for OSM queries:** {infra_summary.get('cell_size_deg', 'n/a')}°\n"
            "- **Method:** geometric intersection between OSM feature "
            "geometries and the vectorised candidate flood extent, computed "
            "in the AOI's UTM projection.\n"
            "- **Note:** 'intersecting' means the feature's geometry touches "
            "the candidate flood polygon. It is not a damage assessment."
        )


def section_access(run: RunContext, conn_summary) -> None:
    st.header("Access disruption")
    st.caption(
        "This is a **modelled scenario**. A road section overlapping a "
        "possibly flooded area is treated as potentially impassable. "
        "It is not confirmed closed."
    )
    if not conn_summary:
        st.info("No connectivity summary available for this run.")
        return

    counts = conn_summary.get("counts", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Settlements in the road graph", counts.get("settlements_in_graph"))
    with c2:
        render_metric("Connected at baseline", counts.get("baseline_connected"))
    with c3:
        render_metric("Still connected in the scenario", counts.get("scenario_connected"))
    with c4:
        render_metric("May have lost their road connection", counts.get("newly_disconnected"))

    disconnected = conn_summary.get("newly_disconnected_settlements", []) or []
    if disconnected:
        st.subheader("Settlements that may have lost their mapped road connection")
        st.dataframe(pd.DataFrame(disconnected), use_container_width=True)
    else:
        st.success(
            "No settlements lost their mapped road connection under the "
            "current modelled scenario."
        )

    st.subheader("What this means for a responder")
    st.markdown(
        "- The listed settlements had a road connection to a hospital or "
        "town before the event, and lose it in the model when flooded road "
        "sections are treated as impassable.\n"
        "- A flooded road is not automatically impassable. Bridges, "
        "elevated sections, and alternate routes may still be usable.\n"
        "- These settlements should be prioritised for ground verification."
    )
    with st.expander("Technical details"):
        st.markdown(
            f"- **Scenario rule:** {conn_summary.get('note', 'n/a')}\n"
            f"- **Disruption fraction:** {conn_summary.get('disruption_fraction', 'n/a')}\n"
            f"- **Node merge tolerance:** {conn_summary.get('node_merge_m', 'n/a')} m"
        )


def section_report(run: RunContext, evidence) -> None:
    st.header("Situation report")
    st.caption(
        "The report is generated from the metrics computed for the current "
        "run. Every number in the report traces to an actual analysis result."
    )

    c1, _ = st.columns([1, 3])
    with c1:
        if st.button("Generate report", type="primary"):
            with st.spinner("Generating report..."):
                st.session_state["situation_report"] = generate_report(evidence)

    if "situation_report" not in st.session_state:
        with st.spinner("Generating initial report..."):
            st.session_state["situation_report"] = generate_report(evidence)

    report = st.session_state["situation_report"]
    st.caption(
        f"Source: **{report.generated_by}** — "
        f"generated at {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    st.markdown(report.body_markdown)

    if report.notes:
        with st.expander("Notes"):
            for n in report.notes:
                st.write(f"- {n}")

    report_path = get_settings().report_path / f"situation_report_{run.run_id}.md"
    write_report(report, report_path)
    with open(report_path, "rb") as fh:
        st.download_button(
            "Download report (Markdown)",
            data=fh.read(),
            file_name=f"situation_report_{run.run_id}.md",
            mime="text/markdown",
        )


def section_copilot(run: RunContext, evidence) -> None:
    st.header("Copilot")
    st.caption(
        "Ask questions about the current run. Every number in the answer is "
        "grounded in the computed metrics. If the language model proposes an "
        "unsupported number, the response is rejected and a rule-based answer "
        "is used instead."
    )

    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []

    # If a previous "Ask" scheduled a clear, do it now — before the text
    # input widget is instantiated. Deleting the widget's key lets
    # Streamlit create a fresh, empty widget on this rerun.
    if st.session_state.pop("copilot_clear_pending", False):
        st.session_state.pop("copilot_input", None)

    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i].button(q, key=f"suggest_{i}"):
            with st.spinner("Thinking..."):
                result = answer_question(evidence, q)
            st.session_state["copilot_history"].append(
                {"question": q, "result": result}
            )

    question = st.text_input("Your question", key="copilot_input")
    if st.button("Ask", type="primary"):
        if question.strip():
            with st.spinner("Thinking..."):
                result = answer_question(evidence, question.strip())
            st.session_state["copilot_history"].append(
                {"question": question.strip(), "result": result}
            )
            st.session_state["copilot_clear_pending"] = True
            st.rerun()

    for entry in reversed(st.session_state["copilot_history"]):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A** _(source: {entry['result']['source']})_")
        st.markdown(entry["result"]["answer"])
        for w in entry["result"].get("warnings", []):
            st.warning(w)
        st.divider()


def section_provenance(run: RunContext, evidence) -> None:
    st.header("Data provenance and limitations")
    st.caption("Every result traces to a source file and a processing step.")

    st.subheader("Current run")
    st.write(f"**Run id:** `{run.run_id}`")
    if run.run_dir is not None:
        st.write(f"**Run folder:** `{run.run_dir}`")
    if run.event_date:
        st.write(f"**Event date:** {run.event_date.isoformat()}")
    if run.aoi_bbox is not None:
        st.write(
            f"**Area:** {run.aoi_name or 'unnamed'} "
            f"({run.aoi_bbox.min_lon:.4f}, {run.aoi_bbox.min_lat:.4f}, "
            f"{run.aoi_bbox.max_lon:.4f}, {run.aoi_bbox.max_lat:.4f})"
        )

    metrics_rows = [
        {
            "Metric ID": m.metric_id,
            "Label": m.label,
            "Value": m.value,
            "Unit": m.unit or "",
            "Kind": m.kind,
            "Source": m.source_layer,
        }
        for m in evidence.metrics
    ]
    if metrics_rows:
        st.subheader("Computed metrics")
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    st.subheader("Attributions")
    for a in ATTRIBUTIONS:
        st.write(f"- {a}")

    if evidence.warnings:
        st.subheader("Warnings")
        for w in evidence.warnings:
            st.warning(w)
    else:
        st.success("No analysis warnings.")


def section_export(run: RunContext, evidence, gdfs: dict, infra_summary, conn_summary) -> None:
    st.header("Export")
    st.caption("Download the generated results for this run.")

    c1, c2 = st.columns(2)

    def _download(label: str, path: Optional[Path], file_name: str, mime: str) -> None:
        if path is None or not path.is_file():
            return
        with open(path, "rb") as fh:
            st.download_button(label, data=fh.read(), file_name=file_name, mime=mime)

    with c1:
        _download(
            "Areas identified as possibly flooded (GeoJSON)",
            run.flood_extent_geojson,
            "candidate_flood_extent.geojson",
            "application/geo+json",
        )
        _download(
            "Buildings inside possibly flooded areas (GeoJSON)",
            run.buildings_geojson,
            "buildings_exposed.geojson",
            "application/geo+json",
        )
        _download(
            "Road sections overlapping possibly flooded areas (GeoJSON)",
            run.roads_geojson,
            "roads_exposed.geojson",
            "application/geo+json",
        )
    with c2:
        _download(
            "Bridges possibly affected (GeoJSON)",
            run.bridges_geojson,
            "bridges_exposed.geojson",
            "application/geo+json",
        )
        _download(
            "Settlements that may be cut off (GeoJSON)",
            run.disconnected_geojson,
            "disconnected_settlements.geojson",
            "application/geo+json",
        )
        _download(
            "Road sections that may be impassable (GeoJSON)",
            run.disrupted_roads_geojson,
            "disrupted_roads.geojson",
            "application/geo+json",
        )

    summary_payload = {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir) if run.run_dir else None,
        "event_date": run.event_date.isoformat() if run.event_date else None,
        "aoi_name": run.aoi_name,
        "aoi_bbox": list(run.aoi_bbox.as_tuple()) if run.aoi_bbox else None,
        "metrics": [
            {
                "id": m.metric_id,
                "label": m.label,
                "value": m.value,
                "unit": m.unit,
                "kind": m.kind,
                "source_layer": m.source_layer,
                "method": m.method,
            }
            for m in evidence.metrics
        ],
        "warnings": evidence.warnings,
        "attributions": evidence.attributions,
        "infrastructure_summary": infra_summary,
        "connectivity_summary": conn_summary,
    }
    st.download_button(
        "Analysis summary (JSON)",
        data=json.dumps(summary_payload, indent=2, ensure_ascii=False),
        file_name=f"analysis_summary_{run.run_id}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------

def build_run_evidence(run: RunContext) -> Any:
    """Build an AnalysisEvidence object for the currently selected run.

    Fallbacks to the precomputed Trishuli case-study paths are disabled
    for run-relative contexts. Every path must come from the selected
    run's RunContext. If a path is None, that layer is genuinely absent
    for this run.
    """
    if run.aoi_bbox is not None and run.event_date is not None:
        event = EventDefinition(
            aoi=run.aoi_bbox,
            event_date=run.event_date,
            aoi_name=run.aoi_name or run.run_id,
        )
    else:
        event = EventDefinition(
            aoi=BoundingBox(
                min_lon=DEFAULT_BBOX[0], min_lat=DEFAULT_BBOX[1],
                max_lon=DEFAULT_BBOX[2], max_lat=DEFAULT_BBOX[3],
            ),
            event_date=DEFAULT_EVENT_DATE,
            aoi_name=DEFAULT_AOI_NAME,
        )

    return build_evidence(
        event,
        flood_extent_geojson=run.flood_extent_geojson,
        infrastructure_summary=run.infrastructure_summary,
        connectivity_summary=run.connectivity_summary,
        use_settings_fallbacks=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging("INFO")
    settings = get_settings()

    inputs = render_sidebar(settings)

    run = get_selected_run(settings)
    evidence = build_run_evidence(run)

    gdfs = {
        "flood": load_geojson(run.flood_extent_geojson),
        "buildings": load_geojson(run.buildings_geojson),
        "roads": load_geojson(run.roads_geojson),
        "bridges": load_geojson(run.bridges_geojson),
        "disconnected": load_geojson(run.disconnected_geojson),
        "disrupted_roads": load_geojson(run.disrupted_roads_geojson),
    }

    infra_summary = load_json(run.infrastructure_summary)
    conn_summary = load_json(run.connectivity_summary)

    page = inputs["page"]

    if page == "Run new analysis":
        section_run_new_analysis(settings, inputs)
    elif page == "Overview":
        section_overview(run, gdfs["flood"], infra_summary, conn_summary)
    elif page == "Map":
        section_map(run, gdfs)
    elif page == "Flood mapping":
        section_flood(run, gdfs["flood"])
    elif page == "Infrastructure exposure":
        section_infrastructure(run, infra_summary)
    elif page == "Access disruption":
        section_access(run, conn_summary)
    elif page == "Situation report":
        section_report(run, evidence)
    elif page == "Copilot":
        section_copilot(run, evidence)
    elif page == "Data provenance":
        section_provenance(run, evidence)
    elif page == "Export":
        section_export(run, evidence, gdfs, infra_summary, conn_summary)


if __name__ == "__main__":
    main()