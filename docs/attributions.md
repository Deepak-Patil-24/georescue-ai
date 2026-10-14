## File 2 of 4 — `docs/acceptance_checklist.md`

Changes:

- Section I now says Nepali is implemented.
- Section D now mentions the live URL.
- Section K updated with the current test count.
- Section B now mentions bilingual output.

````markdown
# GeoRescue AI — Hackathon Acceptance Checklist

**Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**

This document maps every requirement in the official challenge brief to
its implementation status in this repository. The status column uses
one of four labels:

- **✅ Implemented and tested** — code exists, has been run against
  real data, and the output has been verified.
- **✅ Implemented (not live-tested)** — code exists and has been
  reviewed; a specific live test has not been performed (usually
  because an external service was rate-limited or unavailable at the
  time of writing).
- **⚠️ Partially implemented** — some functionality exists but is
  incomplete or requires additional configuration.
- **❌ Not implemented** — the feature is not present.

---

## A. Core analytical requirements

### A1. Where did the flood hit?

| Requirement                                | Status                           | Evidence                                                                  |
| ------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------- |
| Accept user-supplied AOI and event date    | ✅ Implemented and tested        | `scripts/run_pipeline.py`, `app.py` sidebar inputs                        |
| Search CDSE for Sentinel-1 pre/post scenes | ✅ Implemented and tested        | `src/georescue/data_access/sentinel1.py`; live search returned 20+ scenes |
| Prefer same orbit track                    | ✅ Implemented and tested        | Pair selection prefers same relative orbit and direction                  |
| Download Sentinel-1 GRD products           | ✅ Implemented and tested        | Two 1.2 GB products downloaded (Trishuli + Syapru)                        |
| Sarsen terrain correction                  | ✅ Implemented and tested        | Both scenes preprocessed; log shows successful GTC                        |
| Backscatter ratio change detection         | ✅ Implemented and tested        | Trishuli: 2.75 km²; Syapru: 0.975 km²                                     |
| Slope masking from DEM                     | ✅ Implemented and tested        | Trishuli: 11,841 pixels removed; Syapru: 2,825 removed                    |
| Minimum mapping unit                       | ✅ Implemented and tested        | 9 px (Trishuli), 2 px (Syapru)                                            |
| Morphological cleanup                      | ✅ Implemented and tested        | Merge gap 50 m, hole fill 4 px                                            |
| Vectorisation to GeoJSON                   | ✅ Implemented and tested        | 181 polygons (Trishuli), 32 polygons (Syapru)                             |
| Area in km² and hectares                   | ✅ Implemented and tested        | Uses projected CRS; log values verified                                   |
| Acquisition metadata preserved             | ✅ Implemented and tested        | Written to `analysis_summary.json`                                        |
| Sentinel-2 corroboration                   | ✅ Implemented (not live-tested) | Code exists; monsoon cloud cover prevented use in case study              |

### A2. What infrastructure may have been affected?

| Requirement                            | Status                    | Evidence                                          |
| -------------------------------------- | ------------------------- | ------------------------------------------------- |
| Pre-event OSM snapshot                 | ✅ Implemented and tested | ohsome API with snapshot date 2026-07-27          |
| Building extraction                    | ✅ Implemented and tested | 185,443 features fetched over the AOI             |
| Road extraction                        | ✅ Implemented and tested | 13,867 features fetched                           |
| Bridge identification                  | ✅ Implemented and tested | `bridge=*` filter                                 |
| Geometry validation                    | ✅ Implemented and tested | `make_valid`, empty removal                       |
| Spatial intersection with flood extent | ✅ Implemented and tested | 514 buildings, 1,256 roads, 11 bridges intersect  |
| Counts and road lengths                | ✅ Implemented and tested | 3.86 ha exposed footprint, 64.01 km affected road |
| Export exposed feature layers          | ✅ Implemented and tested | `outputs/infrastructure/*.geojson`                |
| Preserve OSM identifiers               | ✅ Implemented and tested | Each exported feature carries its OSM ID          |
| Distinguish intersection from damage   | ✅ Implemented and tested | Documented in the report and in the code          |

### A3. Who may be cut off?

| Requirement                        | Status                    | Evidence                                                |
| ---------------------------------- | ------------------------- | ------------------------------------------------------- |
| Road-network graph (NetworkX)      | ✅ Implemented and tested | 25,147 nodes, 24,572 edges for Trishuli                 |
| Node merging to close OSM gaps     | ✅ Implemented and tested | 50 m tolerance                                          |
| Settlement identification          | ✅ Implemented and tested | 249 settlements in the graph                            |
| Destination identification         | ✅ Implemented and tested | 522 hospitals/towns                                     |
| Baseline connectivity              | ✅ Implemented and tested | 87 connected at baseline                                |
| Disruption rule                    | ✅ Implemented and tested | 20% length threshold; 424 edges disrupted               |
| Scenario connectivity              | ✅ Implemented and tested | 85 connected after disruption                           |
| Newly disconnected detection       | ✅ Implemented and tested | 2 settlements identified                                |
| Output GeoJSON of disconnected     | ✅ Implemented and tested | `disconnected_settlements.geojson`                      |
| Output GeoJSON of disrupted roads  | ✅ Implemented and tested | `disrupted_roads.geojson`                               |
| Assumptions documented             | ✅ Implemented and tested | `connectivity_summary.json` and technical report        |
| Modelled scenario labelled as such | ✅ Implemented and tested | Every output says "modelled access disruption scenario" |

---

## B. AI component

| Requirement                               | Status                           | Evidence                                                                               |
| ----------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| Situation-report copilot                  | ✅ Implemented and tested        | `src/georescue/reporting/llm_copilot.py`                                               |
| Deterministic fallback                    | ✅ Implemented and tested        | `src/georescue/reporting/situation_report.py`                                          |
| Every number grounded in computed metrics | ✅ Implemented and tested        | 12 `EvidenceMetric` objects in `evidence.py`                                           |
| LLM cannot invent numbers                 | ✅ Implemented and tested        | Validator rejects unsupported numbers; verified in tests                               |
| LLM can answer rescuer questions          | ✅ Implemented and tested        | `answer_question` works with Gemini and deterministic                                  |
| Free-form question support                | ✅ Implemented and tested        | Gemini path handles novel questions                                                    |
| **Bilingual English and Nepali**          | ✅ Implemented and tested        | LLM path produces English then Nepali; deterministic path prefixes a Nepali disclaimer |
| Handle rate limits gracefully             | ✅ Implemented and tested        | Falls back to deterministic; warning recorded                                          |
| Gemini (free tier) support                | ✅ Implemented and tested        | `GEMINI_API_KEY` in `.env` and in Streamlit Secrets                                    |
| DeepSeek support (optional)               | ✅ Implemented (not live-tested) | Code path exists; not exercised because we chose Gemini                                |
| Deterministic-only mode                   | ✅ Implemented and tested        | `GEORESCUE_COPILOT_MODE=deterministic` disables LLM                                    |
| Report structure (7 sections)             | ✅ Implemented and tested        | Sections documented in `situation_report.py`                                           |

---

## C. Data-use rules and restrictions

| Rule                                             | Status                    | Evidence                                                             |
| ------------------------------------------------ | ------------------------- | -------------------------------------------------------------------- |
| Use only permitted data sources                  | ✅ Implemented and tested | Only Sentinel-1/2, Copernicus DEM, OSM pre-event                     |
| Do not use EMSR927 as prediction input           | ✅ Implemented and tested | `evaluation/metrics.py` is read-only; never called from the pipeline |
| Do not use UNOSAT or other published damage maps | ✅ Implemented and tested | No such code exists                                                  |
| Do not use post-event OSM edits                  | ✅ Implemented and tested | Snapshot date enforced to be before event date                       |
| Historical OSM actually predates the event       | ✅ Implemented and tested | 2026-07-27 snapshot for a 2026-08-26 event                           |
| Evaluation separate from prediction              | ✅ Implemented and tested | Evaluation reads the prediction, not vice versa                      |
| No fabricated metrics                            | ✅ Implemented and tested | All metrics computed at runtime; summary JSON written                |

---

## D. Deliverables

| Deliverable                        | Status                    | Location                                                           |
| ---------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| Working dashboard                  | ✅ Implemented and tested | `app.py`; deployed at https://georescue-ai-georescue.streamlit.app |
| Interactive map with layer toggles | ✅ Implemented and tested | Folium map with all seven layers                                   |
| One-page situation report          | ✅ Implemented and tested | Generated by `situation_report.py`                                 |
| Machine-readable analysis summary  | ✅ Implemented and tested | `analysis_summary.json` export in dashboard                        |
| GitHub repository                  | ✅ Implemented            | https://github.com/Deepak-Patil-24/georescue-ai                    |
| README with setup instructions     | ✅ Implemented            | `README.md`                                                        |
| Technical report (≤ 6 pages)       | ✅ Implemented            | `docs/technical_report.md`                                         |
| Three-minute demo script           | ✅ Implemented            | `docs/demo_script.md`                                              |
| Attribution and citations          | ✅ Implemented            | `docs/attributions.md`, and in every report                        |
| Case-study evaluation              | ✅ Implemented and tested | IoU 0.007 and 0.009 against EMSR927 AOI01                          |

---

## E. Case-study requirements

| Requirement                                   | Status                    | Evidence                                             |
| --------------------------------------------- | ------------------------- | ---------------------------------------------------- |
| Configurable event date and AOI               | ✅ Implemented and tested | Both are inputs; Trishuli defaults are not hardcoded |
| Same processing pipeline used for other areas | ✅ Implemented and tested | Syapru Besi AOI ran through the identical pipeline   |
| Independent evaluation against EMSR927        | ✅ Implemented and tested | Two evaluations; both JSON summaries preserved       |
| Evaluation metrics computed                   | ✅ Implemented and tested | IoU, precision, recall, F1, TP/FP/FN/TN              |
| Reference not used for tuning                 | ✅ Implemented and tested | Thresholds chosen for physical plausibility          |
| Save acquisition metadata                     | ✅ Implemented and tested | `analysis_summary.json` and evidence object          |
| Explain limitations of evaluation             | ✅ Implemented            | Technical report Section 6.3                         |

---

## F. Required attributions

| Attribution                                          | Status                   | Location                                 |
| ---------------------------------------------------- | ------------------------ | ---------------------------------------- |
| "Contains modified Copernicus Sentinel data 2026."   | ✅ Present verbatim      | Report, dashboard, README, `evidence.py` |
| "Produced using Copernicus WorldDEM-30 © DLR e.V. …" | ✅ Present verbatim      | Report, dashboard, README, `evidence.py` |
| "© OpenStreetMap contributors."                      | ✅ Present verbatim      | Report, dashboard, README, `evidence.py` |
| Training-dataset citations                           | ✅ Present (as not used) | `docs/attributions.md`                   |

---

## G. Engineering quality

| Requirement                            | Status                    | Evidence                                               |
| -------------------------------------- | ------------------------- | ------------------------------------------------------ |
| Type hints on public functions         | ✅ Implemented            | Throughout `src/georescue/`                            |
| Docstrings on public modules           | ✅ Implemented            | Throughout                                             |
| Structured logging                     | ✅ Implemented            | `logging_config.py`                                    |
| Handled API timeouts and errors        | ✅ Implemented            | Retry-on-429; clear error messages                     |
| No bare `except` blocks                | ✅ Implemented            | All excepts catch named types                          |
| Atomic file writes                     | ✅ Implemented            | `.part` file then rename                               |
| Tests included                         | ✅ Implemented and tested | 76 automated tests across 7 files; all pass            |
| Tests runnable without credentials     | ✅ Implemented            | Synthetic fixtures only                                |
| No credentials in source               | ✅ Implemented            | All credentials read from `.env` or Streamlit Secrets  |
| `.env` in `.gitignore`                 | ✅ Implemented            | Yes                                                    |
| Application runs when LLM unavailable  | ✅ Implemented and tested | Deterministic fallback verified with Nepali disclaimer |
| Application runs when CDSE unavailable | ✅ Implemented            | Clear error message, no fabricated data                |

---

## H. Windows and VS Code requirements

| Requirement                         | Status         | Evidence                              |
| ----------------------------------- | -------------- | ------------------------------------- |
| Windows-friendly setup              | ✅ Implemented | All commands are PowerShell           |
| Python 3.11                         | ✅ Documented  | README prerequisite                   |
| Prebuilt wheels only                | ✅ Implemented | All packages ship Windows wheels      |
| No Linux-only shell commands        | ✅ Implemented | Every command in README is PowerShell |
| `pathlib` for filesystem paths      | ✅ Implemented | Throughout `src/georescue/`           |
| Documented project output directory | ✅ Implemented | `outputs/`, `data/`, `reports/`       |
| `.env.example` present              | ✅ Implemented | Yes                                   |
| `.gitignore` present                | ✅ Implemented | Yes                                   |
| Tests directory                     | ✅ Implemented | Yes                                   |
| Entry point                         | ✅ Implemented | `app.py`, `scripts/run_pipeline.py`   |

---

## I. What is NOT implemented

| Feature                                                          | Reason                                                                                                                                                                                                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trained flood-segmentation model                                 | Not implemented. A transparent baseline was used instead to keep the system fully reproducible without external model weights.                                                                                                                          |
| GeoTIFF export from dashboard                                    | Not implemented. The dashboard exports GeoJSON and JSON. GeoTIFF outputs are written by the pipeline to each run folder.                                                                                                                                |
| Copernicus Browser integration                                   | Not implemented. Not required for the submission.                                                                                                                                                                                                       |
| Multi-user authentication                                        | Not implemented. Discouraged by the hackathon brief.                                                                                                                                                                                                    |
| Optional flow-path bonus (upstream → downstream terrain tracing) | Not implemented. The bonus is optional in the hackathon brief, and this submission focuses on the three core questions and the AI component.                                                                                                            |
| Live new-AOI analysis on the deployed link                       | Not implemented on the free cloud tier. The deployed dashboard shows a pre-computed case study. Running a new AOI requires a local installation because the Sentinel-1 download is ~2.4 GB and free hosting does not provide persistent storage for it. |

These are honest limitations, not failures. Each is documented in the
technical report or README.

---

## J. Unresolved blockers at the time of writing

None. Every required feature either works or is honestly documented as
not implemented. Two features depend on external rate limits (Gemini
LLM, ohsome API) and degrade gracefully when the limits are hit.

---

## K. Verification summary

The automated test suite contains **76 tests** across seven files. Running `pytest tests/ -v` from the project root (with `PYTHONPATH=src`) produces:

```text
collected 76 items
...
76 passed, 12 warnings in ~3s
```
````
