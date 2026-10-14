# GeoRescue AI

**Flood damage mapping and rescue intelligence from satellite data.**

Submission for the **Multimodal AI Hackathon 2026 — Track B: Mapping
Flood Damage from Space**.

> ⚠️ **Research prototype — not an operational emergency-response system.**
>
> GeoRescue AI is an educational and research tool. It does not replace
> official emergency services, and its outputs must not be used as the sole
> basis for life-safety decisions. See [Limitations](#limitations-summary)
> below.

---

## Submission links

| Resource                 | Link                                                         |
| ------------------------ | ------------------------------------------------------------ |
| **Live dashboard**       | https://georescue-ai-georescue.streamlit.app                 |
| **Demo video**           | https://youtu.be/1xIeccUQCPc                              |
| **Technical report**     | [docs/technical_report.md](docs/technical_report.md)         |
| **Attributions**         | [docs/attributions.md](docs/attributions.md)                 |
| **Acceptance checklist** | [docs/acceptance_checklist.md](docs/acceptance_checklist.md) |
| **Demo script**          | [docs/demo_script.md](docs/demo_script.md)                   |

> The deployed dashboard shows the pre-computed Trishuli case study.
> Running a new area or date requires the full pipeline locally — see
> [Running on a new AOI](#running-on-a-new-aoi).

---

## What it does

Given an area of interest (bounding box or drawn polygon) and an event
date, GeoRescue AI answers three disaster-response questions:

1. **Where did the flood hit?**
   Compares pre-event and post-event **Sentinel-1** SAR observations to map
   candidate flood extent. Uses **Sentinel-2** optical imagery for
   corroboration when clouds permit.

2. **What infrastructure may have been affected?**
   Intersects the mapped flood extent with **pre-event OpenStreetMap**
   buildings, roads, and bridges, and reports counts, lengths, and OSM
   identifiers.

3. **Who may be cut off?**
   Builds a road-network graph from the same pre-event OSM snapshot and
   identifies settlements whose road connection to the nearest town or
   hospital is lost under a configurable flood-disruption scenario.

An optional **situation-report copilot** (LLM-backed, with a deterministic
fallback) drafts a one-page report and answers rescuer questions — grounded
strictly on metrics computed in Python, never invented by the language
model.

---

## Case-study results (Trishuli, 26 August 2026)

| Metric                              | Value                           |
| ----------------------------------- | ------------------------------- |
| Candidate flood extent (raster)     | 2.75 km²                        |
| Candidate flood extent (vectorised) | 3.05 km², 181 polygons          |
| Buildings intersecting flood        | 514 (3.86 ha exposed footprint) |
| Road segments intersecting flood    | 1,256 (64.01 km affected)       |
| Bridges intersecting flood          | 11                              |
| Settlements in road graph           | 249                             |
| Baseline connected                  | 87                              |
| Scenario connected                  | 85                              |
| **Settlements newly disconnected**  | **2**                           |

See [docs/technical_report.md](docs/technical_report.md) for the full
case study and evaluation.

---

## Quick start (Windows 11, VS Code, Python 3.11)

### Prerequisites

- Windows 10 or 11
- Python 3.11 (verify with `python --version`)
- Git (optional but recommended)
- ~10 GB free disk space (Sentinel-1 products are large)
- Internet connection for data downloads

### Step 1 — Get the code

```powershell
git clone <your-repo-url> GeoRescue-AI
cd GeoRescue-AI
```

If you do not use Git, download the project as a ZIP and extract it, then
`cd` into the folder.

### Step 2 — Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution policy error,
run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies ship prebuilt Windows wheels. No GDAL compilation is
required.

### Step 4 — Configure environment variables

```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code and fill in the following.

**Required for live satellite data:**
`CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`.
Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
Under _User settings → OAuth clients_, create a new client with grant type
_Client Credentials_. Copy the Client ID and Client Secret into `.env`.

**Required for historical OSM:**
`OHSOME_API_KEY`.
Register free at [account.heigit.org](https://account.heigit.org). Copy the
API key into `.env`.

**Optional for the LLM copilot:**
`GEMINI_API_KEY` (free tier).
Register at [aistudio.google.com](https://aistudio.google.com). Copy the API
key into `.env`. Without this key, the copilot runs in deterministic mode.

**Optional for LLM fallback:**
`DEEPSEEK_API_KEY` (paid).
If both Gemini and DeepSeek are unset, the copilot is deterministic.

### Step 5 — Verify the environment

```powershell
$env:PYTHONPATH = "src"
python scripts/check_environment.py
```

Expected: no errors, all required packages import successfully.

### Step 6 — Run the pipeline (case study)

Download the two case-study Sentinel-1 scenes:

```powershell
python scripts/download_case_study.py
```

This downloads two ~1.2 GB products from CDSE into `data/raw/sentinel1/`.
It takes 15–30 minutes.

Then run the pipeline:

```powershell
python scripts/run_pipeline.py
```

This executes terrain correction, alignment, slope computation, and flood
detection. It takes 1–3 minutes for the small Syapru Besi AOI and around
30 seconds for the Trishuli AOI (after the first run, all stages are
cached).

### Step 7 — Launch the dashboard

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```text
GeoRescue-AI/
├── app.py                     # Streamlit dashboard entry point
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   └── settings.py            # User-facing defaults
├── src/georescue/
│   ├── config.py
│   ├── logging_config.py
│   ├── models.py              # Pydantic schemas
│   ├── validation.py
│   ├── data_access/           # CDSE, Sentinel-1/2, OSM, DEM clients
│   ├── processing/            # SAR preprocessing, flood detection, vectorisation
│   ├── analysis/              # Infrastructure exposure, connectivity, terrain
│   ├── reporting/             # Evidence, situation report, LLM copilot
│   └── evaluation/            # Reference-map comparison
├── tests/
├── scripts/
│   ├── check_environment.py
│   ├── download_case_study.py
│   ├── run_pipeline.py
│   ├── inspect_emsr927.py
│   └── inspect_flood_extent.py
├── data/                      # Downloaded and intermediate (gitignored)
├── outputs/                   # Analysis outputs (gitignored)
├── reports/                   # Situation reports (gitignored)
└── docs/                      # Technical report, demo script, attributions
```

---

## Running on a new AOI

The pipeline works for any AOI and any event date. To run a fresh analysis:

**1. Edit the AOI and event date** in `scripts/run_pipeline.py` (the
configuration block at the top of the file).

**2. Edit the AOI** in `scripts/download_case_study.py` to match.

**3. Download the scenes:**

```powershell
python scripts/download_case_study.py
```

**4. Run the pipeline:**

```powershell
python scripts/run_pipeline.py
```

**5. Vectorise the mask:**

```powershell
python -c "from pathlib import Path; from georescue.processing.vectorization import vectorize_flood_mask, write_geojson, reproject_geojson_to_wgs84; g = vectorize_flood_mask(Path('data/processed/candidate_flood_mask.tif'), merge_gap_m=50.0); w = reproject_geojson_to_wgs84(g, g['properties']['crs']); write_geojson(w, Path('data/processed/candidate_flood_extent.geojson')); print('wrote candidate_flood_extent.geojson')"
```

**6. Run the infrastructure analysis:**

```powershell
python -c "from datetime import date; from pathlib import Path; from georescue.models import BoundingBox; from georescue.analysis.infrastructure import analyze_infrastructure_exposure; b = BoundingBox(min_lon=..., min_lat=..., max_lon=..., max_lat=...); analyze_infrastructure_exposure(Path('data/processed/candidate_flood_extent.geojson'), b, osm_snapshot=date(2026, 7, 27), event_date=date(2026, 8, 26), buffer_m=0.0)"
```

**7. Run the connectivity analysis:**

```powershell
python -c "from datetime import date; from pathlib import Path; from georescue.models import BoundingBox; from georescue.analysis.connectivity import analyze_connectivity; b = BoundingBox(min_lon=..., min_lat=..., max_lon=..., max_lat=...); analyze_connectivity(Path('data/processed/candidate_flood_extent.geojson'), b, osm_snapshot=date(2026, 7, 27), event_date=date(2026, 8, 26), disruption_fraction=0.20)"
```

Restart the dashboard to see the new results.

> **Note:** `data/raw/sentinel1/` may contain scenes from previous runs.
> The `download_case_study.py` script re-downloads only what is missing.

---

## Evaluation against a reference map

To compare the candidate flood extent against a reference (for example the
official Copernicus EMSR927 product), place the reference file at
`data/processed/reference.gpkg` (or any path) and run:

```powershell
python -c "from pathlib import Path; from georescue.models import BoundingBox; from georescue.evaluation.metrics import evaluate_flood_extent; b = BoundingBox(min_lon=..., min_lat=..., max_lon=..., max_lat=...); result = evaluate_flood_extent(Path('data/processed/candidate_flood_extent.geojson'), Path('data/processed/reference.gpkg'), aoi_bbox=b, reference_layer='<layer-name>', resolution_m=30.0); import json; print(json.dumps(result, indent=2))"
```

> **Critical:** the reference map is used only for evaluation. It must
> **never** be fed into the flood-detection or infrastructure pipeline. See
> the hackathon rules in the technical report.

---

## Testing

Run the automated test suite:

```powershell
$env:PYTHONPATH = "src"
pytest tests/ -v
```

Expected: **76 passed**.

---

## Configuration

All configuration is via `.env` (see `.env.example`). The most relevant
settings:

| Variable                               | Default | Purpose                        |
| -------------------------------------- | ------- | ------------------------------ |
| `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | —       | Sentinel-1/2 download          |
| `OHSOME_API_KEY`                       | —       | Historical OSM                 |
| `GEMINI_API_KEY`                       | —       | Optional LLM copilot           |
| `GEORESCUE_COPILOT_MODE`               | `auto`  | `auto`, `deterministic`, `llm` |
| `GEORESCUE_MAX_AOI_SQKM`               | `5000`  | Maximum AOI size               |
| `GEORESCUE_LOG_LEVEL`                  | `INFO`  | Logging                        |

---

## Data sources and attributions

GeoRescue AI uses only the data sources permitted by the hackathon rules:

- **Sentinel-1 and Sentinel-2** from the
  [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu).
- **Copernicus DEM 30 m** (WorldDEM-30) from the
  [AWS Open Data bucket](https://registry.opendata.aws/copernicus-dem/).
- **OpenStreetMap** pre-event snapshot via the
  [ohsome API](https://api.ohsome.org).

**Copernicus EMSR927 and other published damage maps are NOT used as
prediction inputs.** They may be used only for evaluation, clearly
separated from the analysis pipeline.

### Required attributions

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and
  © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by
  the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors.

See [docs/attributions.md](docs/attributions.md) for the complete list.

---

## Limitations (summary)

- Sentinel-1 revisit is roughly 6–12 days per orbit; sudden events such as
  the 26 August 2026 glacier collapse cannot be warned about in advance.
- The baseline flood detector uses backscatter change detection with a
  configurable threshold. It is **not** a trained deep-learning model.
- Detected change is **candidate** flood extent, not confirmed damage.
- Infrastructure intersecting the flood polygon is **potentially exposed**,
  not confirmed destroyed.
- Modelled road disconnection is a **scenario**, not proof a road is
  impassable.
- Cloud cover limits Sentinel-2 corroboration; SAR-only results are normal
  during monsoon conditions.
- OpenStreetMap coverage in remote Himalayan regions is incomplete.
- The Syapru Besi evaluation against EMSR927 returns low IoU (0.007 and
  0.009) because the two layers represent different phenomena: a flood
  extent and a landslide-damage footprint.

See [docs/technical_report.md](docs/technical_report.md), Section 8, for
the full limitations.

---

## Licence

- **Source code:** MIT.
- **OSM-derived outputs**, if published, must remain under ODbL.
- **Sentinel and Copernicus DEM data** are distributed under the Copernicus
  free, full and open data policy.
