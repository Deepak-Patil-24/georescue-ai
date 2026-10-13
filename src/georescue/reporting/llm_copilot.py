"""Evidence-grounded LLM copilot for GeoRescue AI.

Supports two providers transparently:

- **Gemini** via Google AI Studio (free tier, preferred).
- **DeepSeek** via the DeepSeek API (paid, fallback).

The copilot never sends confidential credentials, personal data, or
unnecessary information to any provider. The API key is read from
``.env`` and never hardcoded.

Every number the LLM produces is validated against the evidence. If any
number is unsupported, the response is rejected and the deterministic
fallback is returned instead.

Bilingual output
----------------
When the LLM is available, every response (situation report and copilot
answer) is produced in **English first, then Nepali**, under a clear
"नेपाली अनुवाद" heading. The numeric validator runs on the whole
response, so no unsupported number can pass in either language.

When the LLM is unavailable, the deterministic fallback responds in
English with a short Nepali disclaimer at the top, so the Nepali
requirement is still visible to the reader.

Place-name lookup
-----------------
Optional: a small helper resolves OSM identifiers to human-readable
place names via the public Nominatim service. The lookup is cached in
memory and rate-limited to one request per second as Nominatim
requires. If Nominatim is unreachable or returns nothing for an ID, the
ID is kept as-is and the name is left as ``None``. No name is ever
invented.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests

from georescue.config import Settings, get_settings
from georescue.logging_config import get_logger
from georescue.models import AnalysisEvidence, SituationReport
from georescue.reporting.situation_report import build_deterministic_report

log = get_logger(__name__)


# Nepali disclaimer prepended to deterministic answers so the Nepali
# requirement is visibly satisfied even when the LLM is unavailable.
_NEPALI_DISCLAIMER = (
    "**नेपाली अनुवाद उपलब्ध छैन / Nepali translation unavailable**  \n"
    "यो जवाफ अंग्रेजीमा छ। नेपाली अनुवाद तब मात्र उपलब्ध हुन्छ जब "
    "भाषा मोडेल उपलब्ध हुन्छ।  \n"
    "_(This answer is in English. Nepali translation is available only "
    "when the language model is available.)_\n\n---\n\n"
)


# ---------------------------------------------------------------------------
# Place-name lookup (Nominatim)
# ---------------------------------------------------------------------------

# Descriptive User-Agent as Nominatim requires. Do not shorten.
_NOMINATIM_USER_AGENT = (
    "GeoRescue-AI/0.1 (hackathon prototype; contact via project repository)"
)

# Nominatim base URL. The lookup appends ``/lookup?osm_ids=...``.
_NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"

# Timeouts for the lookup request, in seconds (connect, read).
_NOMINATIM_TIMEOUT = (5, 10)

# Minimum seconds between consecutive Nominatim requests. The usage
# policy allows one request per second; we use slightly more to be safe.
_NOMINATIM_MIN_INTERVAL_S = 1.2

# In-memory cache: OSM id string -> resolved name or None.
# The lock protects concurrent access in case the dashboard is loaded
# by multiple sessions in the same process.
_PLACE_NAME_CACHE: dict[str, Optional[str]] = {}
_PLACE_NAME_CACHE_LOCK = threading.Lock()


class _NominatimRateLimiter:
    """Thread-safe rate limiter for Nominatim requests."""

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


_NOMINATIM_LIMITER = _NominatimRateLimiter(_NOMINATIM_MIN_INTERVAL_S)


def _osm_id_to_nominatim_form(osm_id: str) -> Optional[str]:
    """Convert a bare numeric OSM id to a Nominatim ``N123`` / ``W123`` form."""
    osm_id = str(osm_id or "").strip()
    if not osm_id:
        return None
    if osm_id[:1].upper() in {"N", "W", "R"} and osm_id[1:].isdigit():
        return osm_id.upper()
    if osm_id.isdigit():
        return f"N{osm_id}"
    return None


def _query_nominatim_lookup(osm_ids: Iterable[str]) -> dict[str, Optional[str]]:
    """Call the Nominatim lookup endpoint for a batch of OSM ids."""
    result: dict[str, Optional[str]] = {str(i): None for i in osm_ids}

    nominatim_ids = []
    id_to_input: dict[str, str] = {}
    for raw in osm_ids:
        form = _osm_id_to_nominatim_form(str(raw))
        if form:
            nominatim_ids.append(form)
            id_to_input[form] = str(raw)

    if not nominatim_ids:
        return result

    params = {
        "osm_ids": ",".join(nominatim_ids),
        "format": "json",
        "addressdetails": 0,
        "accept-language": "en",
    }
    headers = {
        "User-Agent": _NOMINATIM_USER_AGENT,
        "Accept": "application/json",
    }

    _NOMINATIM_LIMITER.wait()

    try:
        response = requests.get(
            _NOMINATIM_LOOKUP_URL,
            params=params,
            headers=headers,
            timeout=_NOMINATIM_TIMEOUT,
        )
    except requests.RequestException as exc:
        log.info("Nominatim lookup failed: %s", exc)
        return result

    if response.status_code in (403, 429):
        log.info(
            "Nominatim returned HTTP %d; skipping place-name lookup.",
            response.status_code,
        )
        return result
    if response.status_code >= 400:
        log.info(
            "Nominatim returned HTTP %d; skipping place-name lookup.",
            response.status_code,
        )
        return result

    try:
        payload = response.json()
    except ValueError:
        log.info("Nominatim returned non-JSON; skipping place-name lookup.")
        return result

    if not isinstance(payload, list):
        return result

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        osm_type = str(entry.get("osm_type") or "").upper()
        osm_id = entry.get("osm_id")
        if osm_id is None:
            continue
        prefix = {"NODE": "N", "WAY": "W", "RELATION": "R"}.get(osm_type, "")
        if not prefix:
            continue
        form = f"{prefix}{osm_id}"
        input_id = id_to_input.get(form)
        if input_id is None:
            continue
        name = entry.get("name")
        if not name:
            display = entry.get("display_name") or ""
            name = str(display).split(",")[0].strip() or None
        if name:
            result[input_id] = str(name)

    return result


def lookup_place_names(osm_ids: Iterable[str]) -> dict[str, Optional[str]]:
    """Resolve OSM identifiers to human-readable place names."""
    ids = [str(i) for i in osm_ids if i is not None and str(i).strip()]
    if not ids:
        return {}

    unique_ids = list(dict.fromkeys(ids))

    resolved: dict[str, Optional[str]] = {}
    to_query: list[str] = []

    with _PLACE_NAME_CACHE_LOCK:
        for i in unique_ids:
            if i in _PLACE_NAME_CACHE:
                resolved[i] = _PLACE_NAME_CACHE[i]
            else:
                to_query.append(i)

    if to_query:
        try:
            queried = _query_nominatim_lookup(to_query)
        except Exception as exc:
            log.info("Place-name lookup raised an unexpected error: %s", exc)
            queried = {i: None for i in to_query}

        with _PLACE_NAME_CACHE_LOCK:
            for i in to_query:
                value = queried.get(i)
                _PLACE_NAME_CACHE[i] = value
                resolved[i] = value

    return resolved


def _settlement_osm_ids(evidence: AnalysisEvidence) -> list[str]:
    """Return every settlement OSM id referenced by the connectivity data."""
    if evidence.connectivity is None:
        return []
    ids: list[str] = []
    for s in evidence.connectivity.newly_disconnected:
        if s.settlement_osm_id:
            ids.append(str(s.settlement_osm_id))
    return ids


# ---------------------------------------------------------------------------
# Evidence -> JSON for the LLM
# ---------------------------------------------------------------------------

def _flood_extent_to_dict(evidence: AnalysisEvidence) -> Optional[dict[str, Any]]:
    """Serialize the flood-extent sub-object for the LLM."""
    fe = evidence.flood_extent
    if fe is None:
        return None
    return {
        "method": fe.method,
        "parameters": fe.parameters,
        "area_sqkm": fe.area_sqkm,
        "area_hectares": fe.area_hectares,
        "pixel_count": fe.pixel_count,
        "pixel_size_m": fe.pixel_size_m,
        "crs": fe.crs,
    }


def _infrastructure_to_dict(evidence: AnalysisEvidence) -> Optional[dict[str, Any]]:
    """Serialize the infrastructure sub-object for the LLM."""
    infra = evidence.infrastructure
    if infra is None:
        return None
    return {
        "buildings_intersecting_count": infra.buildings_intersecting_count,
        "buildings_exposed_area_sqm": infra.buildings_exposed_area_sqm,
        "roads_intersecting_count": infra.roads_intersecting_count,
        "roads_affected_length_km": infra.roads_affected_length_km,
        "bridges_intersecting_count": infra.bridges_intersecting_count,
        "buffer_m": infra.buffer_m,
        "warnings": list(infra.warnings or []),
    }


def _connectivity_to_dict(evidence: AnalysisEvidence) -> Optional[dict[str, Any]]:
    """Serialize the connectivity sub-object for the LLM."""
    c = evidence.connectivity
    if c is None:
        return None
    return {
        "scenario_name": c.scenario_name,
        "disruption_rule": c.disruption_rule,
        "baseline_settlement_count": c.baseline_settlement_count,
        "baseline_connected_count": c.baseline_connected_count,
        "scenario_connected_count": c.scenario_connected_count,
        "destination_type": c.destination_type,
        "disrupted_road_osm_ids": list(c.disrupted_road_osm_ids or []),
        "newly_disconnected": [
            {
                "settlement_osm_id": s.settlement_osm_id,
                "baseline_destination": s.baseline_destination,
                "baseline_distance_km": s.baseline_distance_km,
                "scenario_connected": s.scenario_connected,
                "reason": s.reason,
            }
            for s in c.newly_disconnected
        ],
    }


def evidence_to_llm_payload(evidence: AnalysisEvidence) -> dict[str, Any]:
    """Return a compact, structured JSON payload for the LLM."""
    def _metric_to_dict(m) -> dict[str, Any]:
        return {
            "id": m.metric_id,
            "label": m.label,
            "value": m.value,
            "unit": m.unit,
            "source_layer": m.source_layer,
            "method": m.method,
            "kind": m.kind,
        }

    payload: dict[str, Any] = {
        "event": {
            "event_date": evidence.event.event_date.isoformat(),
            "aoi_name": evidence.event.aoi_name or "User-specified area",
            "aoi_bbox": list(evidence.event.aoi.as_tuple()),
            "pre_days": evidence.event.pre_days,
            "post_days": evidence.event.post_days,
        },
        "metrics": [_metric_to_dict(m) for m in evidence.metrics],
        "warnings": list(evidence.warnings or []),
        "attributions": list(evidence.attributions or []),
    }

    flood_extent = _flood_extent_to_dict(evidence)
    if flood_extent is not None:
        payload["flood_extent"] = flood_extent

    infrastructure = _infrastructure_to_dict(evidence)
    if infrastructure is not None:
        payload["infrastructure"] = infrastructure

    connectivity = _connectivity_to_dict(evidence)
    if connectivity is not None:
        payload["connectivity"] = connectivity

    osm_ids = _settlement_osm_ids(evidence)
    if osm_ids:
        try:
            payload["place_names"] = lookup_place_names(osm_ids)
        except Exception as exc:
            log.info("Place-name lookup failed entirely: %s", exc)
            payload["place_names"] = {i: None for i in osm_ids}

    notes: list[str] = []
    if evidence.flood_extent is None:
        notes.append("Flood extent layer is absent for this run.")
    if evidence.infrastructure is None:
        notes.append("Infrastructure layer is absent for this run.")
    if evidence.connectivity is None:
        notes.append("Connectivity layer is absent for this run.")
    payload["evidence_completeness"] = {
        "flood_extent": evidence.flood_extent is not None,
        "infrastructure": evidence.infrastructure is not None,
        "connectivity": evidence.connectivity is not None,
        "notes": notes,
    }

    return payload


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the GeoRescue AI copilot. You help rescue workers, volunteers, and emergency-response teams understand a flood assessment and decide what to check first.

YOUR AUDIENCE
A rescuer in the field. They may be reading on a phone, in a hurry, and may have no technical or data-science background. Every answer must be understandable to them.

THREE GOALS, IN THIS PRIORITY ORDER
1. ACCURATE: Every fact comes from the evidence JSON. Never invent, guess, or infer beyond it.
2. UNDERSTANDABLE: Plain everyday English. No jargon. No metric IDs in the main body.
3. ACTIONABLE: Say what the finding means and what to check next.

If accuracy and simplicity ever conflict, choose accuracy and explain the uncertainty simply.

CRITICAL SAFETY RULES — breaking these is a failure:
1. Use ONLY the numbers provided in the evidence JSON. Never invent, round, or approximate a number that is not present.
2. If a requested number is not present, say "this information is not available in the current analysis."
3. Never invent village names, road conditions, flood locations, affected people, hospital accessibility, or rescue requirements.
4. Never claim that a satellite image alone confirms damage, casualties, trapped people, or road closures.
5. Describe mapped flood areas as "areas that may be flooded", never as "confirmed flooding".
6. Describe a building inside a mapped area as "potentially exposed", never as "damaged".
7. Describe a village flagged by the road analysis as "may have lost road access", never as "isolated" or "cut off".
8. Describe the road model as "a scenario", not as a verified assessment.
9. If the user asks about a scenario or value that is not in the evidence (for example, a different threshold that has not been run), say so honestly and explain what would need to be done. Do NOT invent the answer.
10. Never use results from a different location, event, or run. Use only the evidence provided in the current message.
11. Keep every answer under 400 words in the English portion. The Nepali portion may be a close translation of the same content.

BILINGUAL OUTPUT — REQUIRED FOR EVERY RESPONSE
- Produce the answer FIRST in English, then provide a Nepali translation under a heading.
- Use this exact structure:

  [English answer, following the rules below]

  ---

  ### नेपाली अनुवाद

  [Nepali translation of the same answer, containing the same numbers.]

- The Nepali section MUST contain the same numbers as the English section. Do not add numbers. Do not omit numbers.
- The Nepali section MUST preserve the same safety language (candidate, may be, potentially exposed, modelled scenario).
- If you are not confident in a Nepali word, transliterate the English word or keep the English term in parentheses. Accuracy is more important than elegant Nepali.
- If the question is a one-line factual answer, still provide a one-line Nepali translation.

PLAIN-ENGLISH VOCABULARY — use these replacements:
- "candidate flood extent" -> "areas that may be flooded"
- "infrastructure exposure" -> "buildings, roads, and bridges that may be affected"
- "potentially disconnected settlements" -> "villages that may have lost road access"
- "connectivity analysis" -> "checking whether roads to villages remain accessible"
- "SAR satellite imagery" -> "satellite images that can detect changes on the ground, even through clouds"
- "backscatter change" -> "a change in how the ground reflects the satellite signal"
- "road graph" -> "a digital map of how roads connect places"
- "OpenStreetMap" -> "an open map of roads, buildings, and places"
- "digital elevation model" -> "a map of land height"
- "model output" -> "the result of the analysis"
- "baseline" -> "the situation before the flood"
- "scenario" -> "a possible situation based on stated assumptions"
- "false positive" -> "an area the system flags that may not actually be flooded"
- "validation" -> "checking whether a result is correct"
- "modelled road disruption scenario" -> "in this scenario, we assume"
- "edge" or "edge removal" -> "road section"
- "baseline destination" -> "the hospital or town it could reach before the event"
- "newly disconnected" -> "may have lost road access"
- "snapped to the road graph" -> "placed on the road map"
- "algorithmic estimate" -> "our best estimate from satellite data"

If you must use any technical term that is not in the list above, explain it in plain English immediately.

NUMBER FORMATTING
- Round excessive decimals for readability. Write "about 2 km" not "2.030521 km". Write "about 1.1 km" not "1.111636 km".
- Never change the meaning of a number. Rounding is for readability only.
- Do not confuse buildings with roads. Do not confuse road segments (count) with road kilometres (length).
- If a question has multiple parts, answer every part using its own number.

ANSWER STRUCTURE — use this when the answer is more than one sentence:
**What we found**
One or two sentences giving the direct answer in plain English.

**Why it matters**
One or two sentences on the possible impact on people, buildings, roads, or emergency access.

**What to check first**
A short, prioritized list of practical actions.

**Important note**
Mention uncertainty or missing information only when relevant.

For very short factual questions, a single short paragraph is enough. Do not force the structure onto trivial answers. The Nepali section should follow the same structure if the English section uses it.

VILLAGE NAMES AND IDS
- The evidence payload may include a `place_names` map from OSM id to a real village or town name.
- When a name is available in that map, use it in preference to the numeric id.
- When the map has `null` for an id, or when the id is not in the map at all, say honestly that the village name is not available in the current data and include the numeric id as a reference.
- Never invent a village name to sound more useful. Never guess a name from the id or from geography.

TECHNICAL FOOTER
- After the Nepali section, add a short section titled "Evidence references (for review)" listing the metric IDs used, one per line, in the format `metric_id = value`.
- Keep it to 8 lines or fewer.
- Omit the footer if no metric IDs are relevant.
- Metric IDs are allowed ONLY in this footer. Never in the English body or the Nepali section.

RECOMMENDATIONS
- Base priorities on the findings in the evidence.
- Explain why each priority matters.
- Present recommendations as things to verify, not as proof that an emergency exists.
- Do not invent alternative routes, safe evacuation paths, or accessible hospitals.

Answer in clear, professional English followed by Nepali."""


def build_user_prompt(evidence_json: dict[str, Any], question: str) -> str:
    return (
        "Evidence JSON (this is the ONLY factual data you may use):\n"
        "```json\n"
        f"{json.dumps(evidence_json, indent=2, ensure_ascii=False)}\n"
        "```\n\n"
        f"Question: {question}\n\n"
        "Reminder:\n"
        "- Write the answer for a rescuer who has no technical background.\n"
        "- Use plain English in the main body. No metric IDs, no jargon.\n"
        "- Round long decimals for readability, without changing the meaning.\n"
        "- Prioritize accuracy first, understandability second, actionability third.\n"
        "- Put metric IDs only in the optional 'Evidence references (for review)' footer.\n"
        "- Use the `place_names` map when it has a real name for a village id. "
        "If it shows null, say the name is not available and keep the id as reference.\n"
        "- Provide the answer FIRST in English, THEN the same content in Nepali "
        "under the heading '### नेपाली अनुवाद'."
    )


# ---------------------------------------------------------------------------
# LLM call (provider-agnostic)
# ---------------------------------------------------------------------------

def call_llm(
    question: str,
    evidence_json: dict[str, Any],
    *,
    settings: Optional[Settings] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Call the configured LLM provider (Gemini or DeepSeek)."""
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider is None:
        log.info("No LLM provider configured; skipping LLM call.")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("OpenAI SDK not installed; cannot call LLM. "
                    "Run: pip install openai")
        return None

    try:
        client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
            timeout=timeout,
        )
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(evidence_json, question)},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        error_msg = str(exc)

        if "402" in error_msg or "Insufficient Balance" in error_msg:
            log.error(
                "%s returned HTTP 402 (Insufficient Balance). "
                "Top up your %s account or switch provider.",
                provider, provider,
            )
        elif "401" in error_msg or "invalid_api_key" in error_msg.lower():
            log.error(
                "%s returned HTTP 401. Check the API key in .env.", provider,
            )
        elif "429" in error_msg or "rate" in error_msg.lower():
            log.warning("%s rate limit hit; try again later.", provider)
        else:
            log.warning("%s call failed: %s", provider, error_msg[:200])
        return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError):
        log.warning("%s response was missing choices/message.", provider)
        return None

    if not content or not isinstance(content, str):
        log.warning("%s returned an empty message.", provider)
        return None

    return content.strip()


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def validate_llm_response(
    response_text: str,
    evidence: AnalysisEvidence,
    *,
    tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    """Check that every number in the LLM response appears in the evidence.

    Because the response may now contain an English and a Nepali section
    with the same numbers repeated, the validator will see each number
    twice. That is fine: both copies match the same whitelisted value.
    """
    allowed_numbers: set[float] = set()
    allowed_id_strings: set[str] = set()

    def _add_number(v: Any) -> None:
        if v is None:
            return
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return
        if not (fv == fv):
            return
        allowed_numbers.add(fv)
        for factor in (
            100.0,
            1000.0,
            10000.0,
            1_000_000.0,
        ):
            allowed_numbers.add(fv * factor)
            if fv != 0:
                allowed_numbers.add(fv / factor)

    def _add_numbers_from_text(text: str) -> None:
        if not isinstance(text, str):
            return
        for token in _NUMBER_RE.findall(text):
            try:
                allowed_numbers.add(float(token.replace(",", "")))
            except ValueError:
                continue

    def _add_from_dict(data: Any) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                _add_from_dict(value)
                _add_numbers_from_text(str(key))
        elif isinstance(data, (list, tuple)):
            for item in data:
                _add_from_dict(item)
        elif isinstance(data, (int, float)):
            _add_number(data)
        elif isinstance(data, str):
            _add_numbers_from_text(data)

    event = evidence.event
    _add_numbers_from_text(event.event_date.isoformat())
    _add_numbers_from_text(event.aoi_name or "")
    _add_number(event.pre_days)
    _add_number(event.post_days)
    _add_number(event.aoi.min_lon)
    _add_number(event.aoi.min_lat)
    _add_number(event.aoi.max_lon)
    _add_number(event.aoi.max_lat)

    for m in evidence.metrics:
        if isinstance(m.value, (int, float)):
            _add_number(m.value)
        elif isinstance(m.value, str):
            _add_numbers_from_text(m.value)
        _add_numbers_from_text(m.label or "")
        _add_numbers_from_text(m.method or "")
        _add_numbers_from_text(m.source_layer or "")

    if evidence.connectivity:
        c = evidence.connectivity
        _add_number(c.baseline_settlement_count)
        _add_number(c.baseline_connected_count)
        _add_number(c.scenario_connected_count)
        _add_number(len(c.newly_disconnected))
        _add_numbers_from_text(c.scenario_name or "")
        _add_numbers_from_text(c.disruption_rule or "")
        _add_numbers_from_text(c.destination_type or "")

        for s in c.newly_disconnected:
            if s.settlement_osm_id:
                allowed_id_strings.add(str(s.settlement_osm_id))
            if s.baseline_destination:
                destination = str(s.baseline_destination)
                allowed_id_strings.add(destination)
                if "-" in destination:
                    _, suffix = destination.rsplit("-", 1)
                    allowed_id_strings.add(suffix)
            if s.baseline_distance_km is not None:
                _add_number(s.baseline_distance_km)

    if evidence.flood_extent:
        fe = evidence.flood_extent
        _add_number(fe.area_sqkm)
        _add_number(fe.area_hectares)
        _add_number(fe.pixel_count)
        if fe.pixel_size_m is not None:
            _add_number(fe.pixel_size_m)
        _add_numbers_from_text(fe.method or "")
        _add_numbers_from_text(fe.crs or "")
        _add_from_dict(fe.parameters)

    if evidence.infrastructure:
        infra = evidence.infrastructure
        _add_number(infra.buildings_intersecting_count)
        _add_number(infra.buildings_exposed_area_sqm)
        _add_number(infra.roads_intersecting_count)
        _add_number(infra.roads_affected_length_km)
        _add_number(infra.bridges_intersecting_count)
        _add_number(infra.buffer_m)
        for w in infra.warnings or []:
            _add_numbers_from_text(w)

    mentioned_numbers: list[float] = []
    for token in _NUMBER_RE.findall(response_text):
        try:
            mentioned_numbers.append(float(token.replace(",", "")))
        except ValueError:
            continue

    prose_safe_ints = set(float(i) for i in range(0, 16))
    problems: list[str] = []

    for val in mentioned_numbers:
        if val in prose_safe_ints:
            continue

        if val in allowed_numbers:
            continue
        matched = False
        for allowed in allowed_numbers:
            if abs(allowed) > 1000:
                tol = max(abs(allowed) * 0.01, 1.0)
            else:
                tol = max(tolerance * max(abs(allowed), 1.0), 0.05)
            if abs(val - allowed) <= tol:
                matched = True
                break
        if matched:
            continue

        if abs(val - int(val)) < 1e-6:
            as_int_str = str(int(val))
            if as_int_str in allowed_id_strings:
                continue

        problems.append(f"Number {val} does not appear in the evidence.")

    return (len(problems) == 0, problems)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    evidence: AnalysisEvidence,
    *,
    use_llm: Optional[bool] = None,
    settings: Optional[Settings] = None,
) -> SituationReport:
    """Generate a situation report, LLM-backed if available, else deterministic."""
    settings = settings or get_settings()

    if use_llm is None:
        use_llm = settings.has_llm_key

    if not use_llm:
        report = build_deterministic_report(evidence, settings=settings)
        report.notes.append(
            "Deterministic mode (no LLM provider configured or disabled)."
        )
        return report

    payload = evidence_to_llm_payload(evidence)
    question = (
        "Write a one-page situation report following the standard structure: "
        "event and analysis summary, candidate flood extent, potentially "
        "exposed infrastructure, modelled access disruption, areas warranting "
        "further investigation, data limitations and uncertainty, and sources "
        "and attributions. Use only the numbers in the evidence JSON. "
        "Provide the report in English first, then the same report in Nepali "
        "under the heading '### नेपाली अनुवाद'."
    )

    response_text = call_llm(question, payload, settings=settings)
    if response_text is None:
        log.info("Falling back to deterministic report (LLM unavailable).")
        fallback = build_deterministic_report(evidence, settings=settings)
        fallback.notes.append("LLM unavailable or returned no content; deterministic fallback used.")
        return fallback

    is_valid, problems = validate_llm_response(response_text, evidence)
    if not is_valid:
        log.warning(
            "LLM response failed numeric validation: %s. Falling back to deterministic.",
            problems[:5],
        )
        fallback = build_deterministic_report(evidence, settings=settings)
        fallback.notes.append(
            f"LLM response rejected for {len(problems)} unsupported numbers; "
            f"deterministic fallback used."
        )
        return fallback

    return SituationReport(
        title=f"Situation report (LLM, bilingual) — {evidence.event.event_date.isoformat()}",
        generated_at=datetime.now(timezone.utc),
        generated_by="llm",
        body_markdown=response_text,
        evidence=evidence,
        notes=[f"LLM response from {settings.llm_provider} passed numeric validation."],
    )


def answer_question(
    evidence: AnalysisEvidence,
    question: str,
    *,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Answer a rescuer question grounded on the evidence."""
    settings = settings or get_settings()

    if not settings.has_llm_key:
        return {
            "answer": _NEPALI_DISCLAIMER + _deterministic_answer(evidence, question),
            "source": "deterministic",
            "warnings": ["No LLM key configured; response is rule-based."],
        }

    payload = evidence_to_llm_payload(evidence)
    response_text = call_llm(question, payload, settings=settings)
    if response_text is None:
        return {
            "answer": _NEPALI_DISCLAIMER + _deterministic_answer(evidence, question),
            "source": "deterministic",
            "warnings": [f"LLM ({settings.llm_provider}) call failed; response is rule-based."],
        }

    is_valid, problems = validate_llm_response(response_text, evidence)
    if not is_valid:
        return {
            "answer": _NEPALI_DISCLAIMER + _deterministic_answer(evidence, question),
            "source": "deterministic",
            "warnings": [
                f"LLM response rejected for unsupported numbers: {problems[:3]}"
            ],
        }

    return {
        "answer": response_text,
        "source": settings.llm_provider or "llm",
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Deterministic responder
# ---------------------------------------------------------------------------

def _format_settlement(
    osm_id: Optional[str],
    place_names: Optional[dict[str, Optional[str]]] = None,
) -> str:
    """Return a plain-English reference for a settlement."""
    if not osm_id:
        return "an unnamed settlement"
    if place_names:
        name = place_names.get(str(osm_id))
        if name:
            return f"{name} (OSM ID {osm_id})"
    return f"settlement with OSM ID {osm_id}"


def _deterministic_answer(evidence: AnalysisEvidence, question: str) -> str:
    """Rule-based responder for common rescuer questions."""
    q = question.lower()

    place_names: dict[str, Optional[str]] = {}
    try:
        osm_ids = _settlement_osm_ids(evidence)
        if osm_ids:
            place_names = lookup_place_names(osm_ids)
    except Exception as exc:
        log.info("Deterministic fallback could not resolve place names: %s", exc)

    # --- Settlement / access-disruption intent (highest priority) ---
    if any(kw in q for kw in (
        "cut off", "cutoff", "disconnect", "isolat", "unreachable",
        "no longer", "settlement", "village", "who is",
    )):
        if evidence.connectivity:
            c = evidence.connectivity
            answer = (
                f"In the modelled disruption scenario, {c.baseline_connected_count} "
                f"of {c.baseline_settlement_count} settlements had a valid pre-event "
                f"road path to a hospital or town. After applying the flood disruption "
                f"rule, {c.scenario_connected_count} remained connected, so "
                f"{len(c.newly_disconnected)} settlements were newly disconnected.\n\n"
            )
            if c.newly_disconnected:
                answer += "Potentially cut-off settlements:\n"
                for s in c.newly_disconnected:
                    dist = (
                        f"{s.baseline_distance_km:.2f} km"
                        if s.baseline_distance_km is not None
                        else "unknown"
                    )
                    label = _format_settlement(s.settlement_osm_id, place_names)
                    destination = s.baseline_destination or "an unknown destination"
                    answer += (
                        f"- {label} — baseline distance to "
                        f"destination {destination}: {dist}\n"
                    )
            answer += (
                "\nThis is a modelled scenario, not a verified accessibility "
                "assessment. A flooded road segment is labelled as potentially "
                "impassable, not confirmed impassable."
            )
            return answer
        return "Settlement connectivity was not calculated."

    # --- Building-specific intent ---
    if any(kw in q for kw in ("building", "houses", "house", "structures", "footprint")):
        if evidence.infrastructure:
            return (
                f"{evidence.infrastructure.buildings_intersecting_count} buildings intersect "
                f"the candidate flood extent, with a total exposed footprint of "
                f"{evidence.infrastructure.buildings_exposed_area_sqm:.0f} m² "
                f"({evidence.infrastructure.buildings_exposed_area_sqm / 10_000:.2f} ha). "
                f"This is potential exposure, not confirmed damage."
            )
        return "Building exposure was not calculated."

    # --- Bridge-specific intent ---
    if any(kw in q for kw in ("bridge", "bridges")):
        if evidence.infrastructure:
            return (
                f"{evidence.infrastructure.bridges_intersecting_count} bridges "
                f"intersect the candidate flood extent. A bridge whose geometry "
                f"intersects the flood polygon is potentially exposed, not "
                f"confirmed damaged."
            )
        return "Bridge exposure was not calculated."

    # --- Road-specific intent ---
    if any(kw in q for kw in ("road", "roads", "highway", "streets")):
        if evidence.infrastructure:
            return (
                f"{evidence.infrastructure.roads_intersecting_count} road segments "
                f"({evidence.infrastructure.roads_affected_length_km:.2f} km total) "
                f"intersect the candidate flood extent. This is a measure of "
                f"potential exposure, not confirmed impassability."
            )
        return "Road exposure was not calculated."

    # --- Flood extent questions ---
    if any(kw in q for kw in (
        "flood area", "flood extent", "flooded area", "flood size",
        "how much", "km2", "km²",
    )):
        if evidence.flood_extent:
            return (
                f"The mapped candidate flood extent covers approximately "
                f"{evidence.flood_extent.area_sqkm:.2f} km² "
                f"({evidence.flood_extent.area_hectares:.2f} ha). "
                f"This is a candidate extent from Sentinel-1 VV backscatter "
                f"change detection, not confirmed flood damage."
            )
        return "Flood extent was not calculated."

    # --- Hospital / destination questions ---
    if "hospital" in q or "clinic" in q:
        if evidence.connectivity:
            return (
                f"The pre-event road graph contains "
                f"{evidence.connectivity.baseline_settlement_count} settlements and "
                f"their nearest hospitals or towns are used as connectivity "
                f"destinations. Specific hospital names are not included in "
                f"the current evidence object; the OSM identifiers are "
                f"available in connectivity_summary.json."
            )
        return "Hospital data was not included."

    # --- Limitation questions ---
    if any(kw in q for kw in ("limitation", "accurate", "reliab", "confiden")):
        return (
            "Key limitations: (1) Sentinel-1 revisit is 6–12 days; sudden events "
            "cannot be warned about in advance. (2) The flood detector is a "
            "backscatter ratio baseline, not a trained model. (3) OSM coverage "
            "in remote Himalayan regions is incomplete. (4) Roads do not encode "
            "elevation or bridge status. (5) Cloud cover limits Sentinel-2 "
            "corroboration. (6) All outputs are candidate extents and potential "
            "exposure, not confirmed damage."
        )

    # --- Method questions ---
    if any(kw in q for kw in ("method", "how did", "algorithm", "detect")):
        return (
            "The flood extent is derived from Sentinel-1 VV backscatter "
            "change detection with a ratio threshold of 0.4 in linear power. "
            "Terrain slope masking (≤5°) and a minimum mapping unit of 9 "
            "pixels (~0.75 ha at 30 m) are applied. Infrastructure exposure "
            "uses spatial joins against a pre-event OSM snapshot. "
            "Connectivity is computed with Dijkstra shortest paths over "
            "a NetworkX road graph."
        )

    # --- Recommendations ---
    if any(kw in q for kw in ("recommend", "priorit", "first", "what should")):
        if evidence.connectivity and evidence.connectivity.newly_disconnected:
            lines = [
                "Priority investigation targets are the settlements listed as "
                "newly disconnected. For each, a field team should verify:",
            ]
            for s in evidence.connectivity.newly_disconnected:
                label = _format_settlement(s.settlement_osm_id, place_names)
                lines.append(f"- {label}")
            lines.append(
                "For each, check: (1) whether the mapped flood extent reached "
                "the access road, (2) whether the road is passable by vehicle, "
                "(3) whether any alternative route exists that OSM does not "
                "record, and (4) whether the settlement has another source of "
                "access (helipad, footpath, seasonal bridge)."
            )
            return "\n".join(lines)
        return (
            "The modelled scenario did not identify any newly disconnected "
            "settlements. A field team should still verify the exposed "
            "infrastructure counts in the situation report, especially "
            "bridges, which are critical single points of failure."
        )

    # --- Fallback ---
    return (
        "I can answer questions about the candidate flood extent, potentially "
        "exposed buildings, roads, and bridges, and modelled settlement access "
        "disruption. Please rephrase your question or ask about one of those."
    )


__all__ = [
    "answer_question",
    "build_user_prompt",
    "call_llm",
    "evidence_to_llm_payload",
    "generate_report",
    "lookup_place_names",
    "validate_llm_response",
]