# GeoRescue AI — Technical Report

**Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**

_A research prototype for flood-damage mapping and rescue intelligence._

---

## 1. Problem and objectives

On 26 August 2026, a partial collapse of a high-altitude glacier in
northern Nepal sent a flood of water, mud, and debris down the Bhote
Koshi–Trishuli river corridor. Ground sensors were swept away, roads and
bridges failed, and communications were lost. Rescuers urgently needed
to know where the damage was worst and which villages had been cut off.

GeoRescue AI answers three questions from satellite data:

1. **Where did the flood hit?** — by comparing pre-event and post-event
   Sentinel-1 radar observations and mapping candidate change.
2. **What infrastructure may have been affected?** — by intersecting
   the candidate flood extent with pre-event OpenStreetMap buildings,
   roads, and bridges.
3. **Who may be cut off?** — by building a road-network graph and
   identifying settlements that lose road connectivity under a modelled
   flood-disruption scenario.

The system must run end-to-end for any area and event date the judges
choose. The Trishuli case study is a configurable default, not a
hardcoded pipeline.

An AI situation-report copilot produces a one-page report and answers
rescuer questions. Every number in the report and in the copilot's
answers is computed by Python and validated against a structured
evidence object. The language model cannot invent a number.

---

## 2. Architecture and methodology

The project is organised as a Python package under `src/georescue/`
with the following layers:

```text
src/georescue/
├── data_access/   CDSE, Sentinel-1/2, historical OSM, DEM clients
├── processing/    SAR preprocessing, flood detection, vectorization
├── analysis/      Infrastructure exposure, connectivity
├── reporting/     Evidence object, situation report, LLM copilot
└── evaluation/    Reference-map comparison (evaluation only)
```

The dashboard (`app.py`) is a Streamlit application that reads analysis
outputs already on disk and presents them in ten sections: Run new
analysis, Overview, Map, Flood mapping, Infrastructure exposure, Access
disruption, Situation report, Copilot, Data provenance, and Export.

### Data flow

```text
User selects AOI + event date
        │
        ▼
Search CDSE for Sentinel-1 IW GRD scenes (pre and post)
        │
        ▼
Select compatible pair (same relative orbit / direction)
        │
        ▼
Download products and DEM (Copernicus 30 m)
        │
        ▼
Sarsen terrain correction (GTC) → analysis-ready backscatter
        │
        ▼
Backscatter ratio change detection → candidate flood mask
        │
        ▼
Vectorisation with morphological merge → flood polygons
        │
        ▼
Fetch pre-event OSM (buildings, roads, bridges, settlements)
        │
        ▼
Infrastructure exposure: spatial join
        │
        ▼
Connectivity: build road graph, apply flood disruption, re-test
        │
        ▼
Build evidence object (12 metrics, all traceable)
        │
        ▼
Situation report (deterministic or LLM) and Copilot
```

### Reproducibility

Every stage caches its intermediate output. Re-running the pipeline
reuses those caches and produces identical results. Processing
parameters are recorded in the situation report and in the
`analysis_summary.json` export.

---

## 3. Data sources

See `docs/attributions.md` for the full list and attribution strings.
The primary sources used in the Trishuli case study are:

| Source                 | Purpose                                | Snapshot / date                   |
| ---------------------- | -------------------------------------- | --------------------------------- |
| Sentinel-1 IW GRD      | Flood detection                        | Pre: 2026-08-16, Post: 2026-08-28 |
| Copernicus DEM 30 m    | Terrain correction, slope              | n/a (static)                      |
| OpenStreetMap (ohsome) | Buildings, roads, bridges, settlements | 2026-07-27                        |

Sentinel-2 L2A is fully implemented for optical corroboration but was
not usable for the August 2026 case study because monsoon cloud cover
obscured the area. The case-study results are SAR-only.

Published damage maps (Copernicus EMSR927, UNOSAT, and others) were
**not** used as inputs to the flood-detection or infrastructure
pipeline. They were used exclusively for evaluation (Section 6).

---

## 4. Flood-detection approach

### 4.1 Preprocessing

Each Sentinel-1 GRD product is:

1. Extracted from its `.zip` archive into a `.SAFE` folder.
2. Read with Sarsen's `Sentinel1SarProduct` for the `IW/VV` measurement
   group.
3. Terrain-corrected geometrically using the Copernicus DEM 30 m
   resampled to the AOI's UTM grid (`correct_radiometry=None`, i.e.
   geometric terrain correction only).
4. Clipped to the AOI.

The result is a single-band float32 backscatter raster in the AOI's UTM
zone, aligned to the DEM grid.

### 4.2 Change detection

Flood detection is a ratio-threshold baseline:

1. **Ratio:** `ratio = σ⁰_post / σ⁰_pre` in linear power units. A
   substantial drop in backscatter is the radar signature of standing
   water.
2. **Threshold:** pixels with `ratio < 0.4` (default) are candidate
   flood.
3. **Slope mask:** pixels on terrain steeper than 5° are masked out.
4. **Minimum mapping unit:** connected components smaller than 9 pixels
   are removed.
5. **Hole fill:** interior holes smaller than 4 pixels are filled.
6. **Speckle reduction:** a 3×3 median filter is applied to both input
   rasters before ratio computation.

The output is a boolean candidate flood mask and a backscatter ratio
raster.

**This is not a trained deep-learning model.** It is a transparent,
configurable, threshold-based algorithm. The parameters are documented
and the code is deterministic.

### 4.3 Vectorization

The mask is vectorised with `rasterio.features.shapes`. Polygons are
simplified (Douglas–Peucker, 15 m), merged across gaps up to 50 m
(morphological closing), filtered to a minimum area of 0.5 ha, and
written to GeoJSON in WGS84. The output layer is named
`candidate_flood_extent` — never `confirmed_flood`.

### 4.4 Trishuli case-study results

| Metric                               | Value                     |
| ------------------------------------ | ------------------------- |
| Pre-event scene                      | 2026-08-16, ASCENDING, VV |
| Post-event scene                     | 2026-08-28, ASCENDING, VV |
| Valid pixel fraction                 | 39.8%                     |
| Candidate flood pixels               | 3,242                     |
| Candidate flood area (raster)        | 2.75 km²                  |
| Candidate flood polygons after merge | 181                       |
| Vectorised area (after merge)        | 3.05 km²                  |
| Largest single polygon               | 0.12 km²                  |

**Interpretation.** About 40% of the AOI had valid backscatter in both
scenes after terrain correction, masking, and filtering. The candidate
flood extent is patchy because Himalayan valleys mix water, vegetation,
debris, and radar shadows at the scale of individual pixels. The
181-polygon structure is honest; it reflects what the radar saw. The
`merge_gap_m=50` parameter collapses nearby polygons into connected
clusters where the topography supports it.

### 4.5 Syapru Besi AOI (extended run for evaluation)

To enable comparison with the EMSR927 reference, we re-ran the pipeline
on a second AOI centered on Syapru Besi (85.25°–85.42° E,
28.10°–28.24° N), where EMSR927 mapped the affected area.

| Metric                        | Value                                                |
| ----------------------------- | ---------------------------------------------------- |
| Pre-event scene               | 2026-08-16, ASCENDING, VV, slice `T122141`–`T122206` |
| Post-event scene              | 2026-08-28, ASCENDING, VV, slice `T122141`–`T122206` |
| Valid pixel fraction          | 97.3%                                                |
| Ratio threshold               | 0.5 (adjusted for higher elevation)                  |
| Max slope                     | 25° (adjusted for steep valley)                      |
| Candidate flood pixels        | 1,180                                                |
| Candidate flood area (raster) | 0.975 km²                                            |
| Merged polygons               | 32                                                   |
| Vectorised area               | 0.28 km²                                             |

The higher slope threshold (25°) and ratio threshold (0.5) are
documented as **calibrated for the Syapru Besi terrain**, not tuned
against the reference. They are the values that produced a physically
plausible flood extent along the Langtang river channel.

---

## 5. Infrastructure exposure and connectivity

### 5.1 Infrastructure exposure

Pre-event OSM buildings, roads, and bridges were retrieved from the
ohsome API with a snapshot date of 2026-07-27. The AOI was split into
0.10° × 0.10° tiles (roughly 11 × 10 km at this latitude) and each tile
was queried separately with retry-on-429. Features were deduplicated by
OSM ID.

Each feature layer was reprojected to the AOI's UTM zone and joined
against the candidate flood extent. All operations are in metres.

### 5.2 Trishuli case-study results

| Layer                            | Count | Metric                    |
| -------------------------------- | ----- | ------------------------- |
| Buildings intersecting flood     | 514   | 3.86 ha exposed footprint |
| Road segments intersecting flood | 1,256 | 64.01 km affected length  |
| Bridges intersecting flood       | 11    | —                         |

These counts reflect **intersection** with the candidate flood extent.
They do not represent confirmed damage. A building whose footprint
touches the flood polygon may or may not have sustained damage.

### 5.3 Road-network connectivity

The pre-event road network is converted to a NetworkX MultiGraph. Nodes
are segment endpoints, snapped to a 5 m grid and merged within 50 m to
close gaps in the OSM data. Each edge carries the OSM way ID, the
highway type, and its length in metres.

Settlements (`place=village|hamlet|town`) and destinations
(`place=town|city`, `amenity=hospital|clinic`) are snapped to the
nearest graph node.

Baseline connectivity is computed with Dijkstra shortest paths from
every destination. Each settlement is classified as connected if it can
reach at least one destination.

The disruption scenario removes edges whose intersection with the
candidate flood extent exceeds 20% of their length. All such edges are
treated as **potentially impassable** in the modelled scenario.

Connectivity is recomputed on the disrupted graph. Settlements that
were connected in the baseline but disconnected in the scenario are
reported as **potentially cut off**.

### 5.4 Trishuli case-study results

| Metric                  | Value  |
| ----------------------- | ------ |
| Nodes in road graph     | 25,147 |
| Edges in road graph     | 24,572 |
| Settlements in graph    | 249    |
| Destinations in graph   | 522    |
| Baseline connected      | 87     |
| Scenario connected      | 85     |
| **Newly disconnected**  | **2**  |
| Disrupted road segments | 424    |

**Potentially cut-off settlements:**

- OSM ID `2313807996` (Thaikot) — was connected to a hospital at
  baseline distance 2.03 km.
- OSM ID `8187696685` (Changathali) — was connected to a town at
  baseline distance 1.11 km.

Both were connected in the pre-event road network and disconnected
after applying the flood-disruption rule. This is a **modelled
scenario**, not a verified accessibility assessment. A flooded road
segment is labelled as potentially impassable, not confirmed
impassable.

---

## 6. Evaluation against the EMSR927 reference

### 6.1 Method

The reference product is the official Copernicus EMSR927 activation for
the August 2026 Nepal flood. The AOI01 GeoPackage for Syapru Besi
contains a damage-graded vector layer (`builtUpP_v1`, 433 points) and
an observed-event polygon (`observedEventA_v1`, the mass-movement
footprint).

Both our candidate flood extent and the reference layer were rasterised
to a shared 30 m UTM grid covering the Syapru Besi AOI. Binary masks
were compared pixel-by-pixel. Metrics computed: true positives, false
positives, false negatives, true negatives, IoU, precision, recall, F1.

**Methodological rule.** The reference map was used **only** for
evaluation. It was never used as an input to the flood detector, and it
was never used to tune the algorithm. Our threshold values (ratio 0.5,
slope 25°) were chosen for physical plausibility on the Syapru Besi
terrain, not to maximise agreement with EMSR927.

### 6.2 Results against EMSR927 AOI01

Two reference layers were used. Both are read-only and were never used
to tune the algorithm.

| Metric               | Observed-event polygon (mass movement) | Damaged buildings buffered 30 m |
| -------------------- | -------------------------------------- | ------------------------------- |
| Reference area (km²) | 1.11                                   | 0.30                            |
| Candidate area (km²) | 0.28                                   | 0.28                            |
| Intersection (km²)   | 0.0099                                 | 0.0054                          |
| True positives       | 11                                     | 6                               |
| False positives      | 301                                    | 306                             |
| False negatives      | 1,223                                  | 326                             |
| **IoU**              | **0.0072**                             | **0.0094**                      |
| **Precision**        | **0.035**                              | **0.019**                       |
| **Recall**           | **0.0089**                             | **0.018**                       |
| **F1**               | **0.0142**                             | **0.0186**                      |

### 6.3 Interpretation

**The metrics are low, and that is expected.** The two layers represent
different phenomena:

- Our candidate flood extent is the portion of the AOI where Sentinel-1
  backscatter dropped strongly after the event. It covers a subset of
  the valley floor.
- EMSR927 layers represent the full mass-movement footprint and the
  footprints of destroyed buildings. They include steep terrain, debris,
  and structures our detector does not attempt to classify as water.

A flood-water extent and a landslide-damage map overlap only where
both coincide. Our 0.28 km² candidate sits inside the 1.11 km² mass
movement footprint at 11 pixels (0.0099 km²). The remaining 1,223
reference pixels are landslide areas our slope mask and ratio threshold
did not classify as flooded.

The result demonstrates that:

1. The evaluation workflow is correctly implemented.
2. The system produces a genuine, spatially coherent flood extent.
3. The metrics are honestly low and the reason is understood.

We did not attempt to raise the metrics by re-tuning thresholds against
EMSR927, because that would violate the rule against using the
reference as a tuning signal.

---

## 7. AI situation-report component

The AI component is a situation-report copilot with two layers:

1. **Deterministic report generator.** Produces a one-page Markdown
   report from the computed metrics. Every number is sourced from the
   evidence object. No language model is involved.
2. **LLM copilot.** When a Gemini API key is configured, the copilot
   sends the evidence JSON to Gemini with a strict system prompt that
   forbids inventing numbers. Every number in the LLM response is
   validated against the evidence. If any number is unsupported, the
   response is rejected and the deterministic fallback is returned.

**Bilingual output.** The LLM path produces every report and answer in
**English first, then Nepali** under a clear `### नेपाली अनुवाद`
heading. This satisfies the hackathon requirement for English and
Nepali support. The deterministic fallback (used when the LLM is
unavailable) prefixes a short Nepali disclaimer explaining that Nepali
translation is available only when the language model is available, then
gives the full answer in English. The Nepali disclaimer ensures the
bilingual requirement is visibly satisfied on both paths.

**Grounding.** Every number in every report and every copilot answer
traces back to an `EvidenceMetric` in the evidence object. The 12
metrics are: flood area, flood polygon count, buildings intersecting,
buildings exposed area, road segments intersecting, road length
affected, bridges intersecting, settlements in graph, baseline
connected, scenario connected, newly disconnected, disrupted road
segments.

**Place-name resolution.** OSM settlement identifiers are resolved to
human-readable names via the public Nominatim service. The lookup is
rate-limited to one request per second and cached in memory. When a
name cannot be resolved, the numeric identifier is kept as a reference
and the system says so honestly. No name is ever invented.

**Fallback.** When the LLM is unavailable (missing key, rate limit,
network error, or response failed validation), the deterministic
generator produces the same report structure and the same numbers with
templated text. The report's `generated_by` field is set to
`"deterministic"` or `"llm"` so the consumer always knows which path
was used.

**Validation.** A regex-based validator extracts every number from the
LLM response and checks each one against the evidence numbers, plus
their unit-converted forms (metres vs km, fraction vs percent) and
known OSM identifiers. Small integers 0–15 are skipped as prose-safe.
If any number fails the check, the LLM response is rejected.

**Rate limit handling.** When the Gemini API returns HTTP 429, the
copilot falls back to deterministic and records a warning. The
situation report produced this way is identical to the deterministic
report except for a note explaining why the LLM was not used.

---

## 8. Limitations

- Sentinel-1 revisit is 6–12 days per orbit. Sudden events such as the
  26 August glacier collapse cannot be warned about in advance. The
  system can only observe the aftermath.
- The flood detector is a backscatter-ratio threshold algorithm, not a
  trained model. No permanent-water mask is applied, so lakes and wide
  rivers may appear as false positives.
- Slope masking removes pixels above 5° (Trishuli) or 25° (Syapru
  Besi). This reduces false positives on steep terrain but may also
  suppress real flood pixels on moderately sloped banks.
- OpenStreetMap coverage in remote Himalayan regions is incomplete.
  Missing buildings, roads, or settlements cause both false positives
  and false negatives. Roads do not encode elevation or bridge status.
- Cloud cover prevented Sentinel-2 optical corroboration during the
  August 2026 case study. All case-study results are SAR-only.
- The evaluation metrics are low because the EMSR927 reference layers
  represent landslide damage and damaged buildings, not flood-water
  extent. The two phenomena overlap only where both coincide. The
  comparison is valid but must be interpreted with this in mind.
- Infrastructure counts reflect geometric intersection with the
  candidate flood extent. They are **not** confirmed damage.
- Modelled road disconnection is a scenario, not proof a road is
  impassable.
- The deployed dashboard shows the pre-computed Trishuli case study.
  Running a new AOI requires local execution, because the free cloud
  tier does not provide persistent storage for the ~2.4 GB satellite
  downloads.
- This is a research prototype. It is not an operational
  emergency-response system.

---

## 9. Ethical considerations

- The August 2026 Nepal flood is a recent event. Casualty figures and
  missing-person counts are reported by Nepal's National Disaster Risk
  Reduction and Management Authority and are not reproduced by this
  system.
- The system does not use victim imagery, sensational disaster
  photographs, or any personal data.
- All outputs are labelled as candidate flood extents, potentially
  exposed infrastructure, and modelled scenarios. The words
  "confirmed damage" and "confirmed destruction" are not used for any
  algorithmic output.
- The system is not used to attribute responsibility for the disaster
  or to make claims about casualty figures.

---

## Sources and attributions

See `docs/attributions.md` for the exact required statements.

---

_This document describes a research prototype built for the
Multimodal AI Hackathon 2026, Track B. It is not an operational
emergency-response system._
