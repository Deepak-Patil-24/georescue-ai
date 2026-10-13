"""Deterministic situation report generator.

This module produces a one-page Markdown situation report from an
:class:`AnalysisEvidence` object. It runs entirely in Python and does not
depend on any LLM. It is used directly when no LLM API key is configured,
and it is used as the fallback when an LLM response fails validation.

Report structure
----------------
1. Event and analysis summary
2. Candidate flood extent
3. Potentially exposed infrastructure
4. Modelled access disruption
5. Areas warranting further investigation
6. Data limitations and uncertainty
7. Sources and attributions

Every number quoted in the report is sourced from an
:class:`EvidenceMetric`. If a metric is missing, the report says so
explicitly rather than substituting a value.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.models import AnalysisEvidence, SituationReport
from georescue.validation import format_bbox

log = get_logger(__name__)


def _metric_lookup(evidence: AnalysisEvidence, metric_id: str):
    for m in evidence.metrics:
        if m.metric_id == metric_id:
            return m
    return None


def _fmt(value, unit: str = "") -> str:
    if value is None:
        return "not calculated"
    if isinstance(value, float):
        return f"{value:,.2f} {unit}".strip()
    if isinstance(value, int):
        return f"{value:,} {unit}".strip()
    return f"{value} {unit}".strip()


def build_deterministic_report(
    evidence: AnalysisEvidence,
    *,
    title: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> SituationReport:
    """Build a one-page Markdown situation report from evidence."""
    settings = settings or get_settings()

    event = evidence.event
    bbox = event.aoi
    aoi_name = event.aoi_name or "User-specified area"

    title = title or f"Situation report — {aoi_name} — {event.event_date.isoformat()}"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"**Generated:** {evidence.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"**Method:** deterministic report from computed metrics"
    )
    lines.append("")

    # ---- 1. Event and analysis summary ----
    lines.append("## 1. Event and analysis summary")
    lines.append("")
    lines.append(f"- **Area of interest:** {aoi_name}")
    lines.append(f"- **Bounding box (WGS84):** {format_bbox(bbox)}")
    lines.append(f"- **Event date:** {event.event_date.isoformat()}")
    lines.append(f"- **Pre-event window:** {event.pre_days} days before the event")
    lines.append(f"- **Post-event window:** {event.post_days} days after the event")
    lines.append("")

    # ---- 2. Candidate flood extent ----
    lines.append("## 2. Candidate flood extent")
    lines.append("")
    if evidence.flood_extent is None:
        lines.append("_Not calculated: flood extent not available._")
    else:
        area_metric = _metric_lookup(evidence, "flood.area_sqkm")
        poly_metric = _metric_lookup(evidence, "flood.polygon_count")
        lines.append(
            f"- **Mapped candidate flood area:** "
            f"{_fmt(area_metric.value if area_metric else None, 'km²')} "
            f"({_fmt((area_metric.value * 100) if area_metric and area_metric.value else None, 'ha')})"
        )
        lines.append(
            f"- **Number of candidate flood polygons:** "
            f"{_fmt(poly_metric.value if poly_metric else None)}"
        )
        lines.append(
            f"- **Detection method:** {evidence.flood_extent.method}"
        )
    lines.append("")
    lines.append(
        "> The mapped extent is a **candidate** flood extent derived from "
        "Sentinel-1 VV backscatter change detection. It is not confirmed "
        "flood damage and it is not a verified flood extent."
    )
    lines.append("")

    # ---- 3. Potentially exposed infrastructure ----
    lines.append("## 3. Potentially exposed infrastructure")
    lines.append("")
    if evidence.infrastructure is None:
        lines.append("_Not calculated: infrastructure exposure not available._")
    else:
        infra = evidence.infrastructure
        lines.append(
            f"- **Buildings intersecting the candidate flood extent:** "
            f"{infra.buildings_intersecting_count:,}"
        )
        lines.append(
            f"- **Total exposed building footprint:** "
            f"{infra.buildings_exposed_area_sqm:,.0f} m² "
            f"({infra.buildings_exposed_area_sqm / 10_000:,.2f} ha)"
        )
        lines.append(
            f"- **Road segments intersecting the candidate flood extent:** "
            f"{infra.roads_intersecting_count:,}"
        )
        lines.append(
            f"- **Total road length intersecting the candidate flood extent:** "
            f"{infra.roads_affected_length_km:,.2f} km"
        )
        lines.append(
            f"- **Bridges intersecting the candidate flood extent:** "
            f"{infra.bridges_intersecting_count:,}"
        )
    lines.append("")
    lines.append(
        "> These counts represent **intersection** with the candidate "
        "flood extent. They do **not** indicate confirmed damage. A "
        "building whose footprint touches the flood polygon may or may "
        "not have sustained structural damage."
    )
    lines.append("")

    # ---- 4. Modelled access disruption ----
    lines.append("## 4. Modelled access disruption")
    lines.append("")
    if evidence.connectivity is None:
        lines.append("_Not calculated: connectivity analysis not available._")
    else:
        c = evidence.connectivity
        lines.append(f"- **Scenario:** {c.scenario_name}")
        lines.append(f"- **Rule:** {c.disruption_rule}")
        lines.append(
            f"- **Settlements in the pre-event road graph:** "
            f"{c.baseline_settlement_count:,}"
        )
        lines.append(
            f"- **Settlements with a pre-event road path to a hospital or town:** "
            f"{c.baseline_connected_count:,}"
        )
        lines.append(
            f"- **Settlements still connected after the modelled disruption:** "
            f"{c.scenario_connected_count:,}"
        )
        lines.append(
            f"- **Settlements newly disconnected by the modelled disruption:** "
            f"{len(c.newly_disconnected)}"
        )
        if c.newly_disconnected:
            lines.append("")
            lines.append("**Potentially cut-off settlements:**")
            lines.append("")
            for s in c.newly_disconnected:
                dist = (
                    f"baseline distance {s.baseline_distance_km:.2f} km"
                    if s.baseline_distance_km is not None
                    else "baseline distance not calculated"
                )
                lines.append(
                    f"- OSM ID `{s.settlement_osm_id}` — "
                    f"was connected to `{s.baseline_destination}` "
                    f"({dist}); disconnected in the modelled scenario."
                )
    lines.append("")
    lines.append(
        "> This is a **modelled scenario**. A flooded road segment is not "
        "automatically impassable. It is labelled here as **potentially "
        "impassable** to help prioritise investigation."
    )
    lines.append("")

    # ---- 5. Areas warranting further investigation ----
    lines.append("## 5. Areas warranting further investigation")
    lines.append("")
    if evidence.connectivity and evidence.connectivity.newly_disconnected:
        lines.append(
            "Priority investigation targets are the settlements listed in "
            "Section 4 as newly disconnected. For each, a field team should "
            "verify:"
        )
        lines.append("")
        lines.append("- Whether the mapped flood extent reached the access road.")
        lines.append("- Whether the road is passable by vehicle.")
        lines.append("- Whether any alternative route exists that OSM does not record.")
        lines.append(
            "- Whether the settlement has another source of access "
            "(helipad, footpath, seasonal bridge)."
        )
    else:
        lines.append(
            "The modelled scenario did not identify any settlements that "
            "lose connectivity. This does not rule out access disruption; "
            "it means the analysis did not find one under the current "
            "thresholds and OSM data."
        )
    lines.append("")

    # ---- 6. Data limitations ----
    lines.append("## 6. Data limitations and uncertainty")
    lines.append("")
    lines.append(
        "- Sentinel-1 revisit is roughly 6–12 days per orbit; sudden "
        "events cannot be warned about in advance."
    )
    lines.append(
        "- The candidate flood extent is produced by backscatter ratio "
        "change detection with a fixed threshold. It is a baseline "
        "algorithm, not a trained deep-learning model."
    )
    lines.append(
        "- Permanent water masking was not applied in this analysis; "
        "standing water bodies may appear as false positives."
    )
    lines.append(
        "- Terrain slope masking removes candidate pixels on slopes "
        "above 5°, which reduces false positives in valleys but may "
        "also suppress real flood pixels on steep terrain."
    )
    lines.append(
        "- OSM coverage in remote Himalayan regions is incomplete. "
        "Missing buildings, roads, or settlements cause both false "
        "positives and false negatives."
    )
    lines.append(
        "- OSM roads do not encode elevation or bridge status. A flooded "
        "segment may in reality be a bridge that remains above water."
    )
    lines.append(
        "- Cloud cover prevents Sentinel-2 corroboration during monsoon "
        "conditions. This analysis is SAR-only."
    )
    if evidence.warnings:
        lines.append("")
        lines.append("**Additional warnings from the analysis pipeline:**")
        for w in evidence.warnings:
            lines.append(f"- {w}")
    lines.append("")

    # ---- 7. Sources and attributions ----
    lines.append("## 7. Sources and attributions")
    lines.append("")
    for a in evidence.attributions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append(
        "This report was generated by GeoRescue AI, a research prototype "
        "for the Multimodal AI Hackathon 2026, Track B. It is not an "
        "operational emergency-response system."
    )

    body = "\n".join(lines)

    return SituationReport(
        title=title,
        generated_at=datetime.now(timezone.utc),
        generated_by="deterministic",
        body_markdown=body,
        evidence=evidence,
        notes=["Deterministic report generated without an LLM."],
    )


def write_report(report: SituationReport, output_path: Path) -> Path:
    """Write a situation report to a Markdown file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report.body_markdown)
    log.info("Wrote situation report: %s", output_path.name)
    return output_path


__all__ = [
    "build_deterministic_report",
    "write_report",
]