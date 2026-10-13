# Attributions and Citations

This document lists every data source, service, and algorithm used by
GeoRescue AI, together with the attribution statements required by the
hackathon and the licence terms of each source.

## Required attribution statements

These statements must appear verbatim in the dashboard, in every
situation report, and in the technical report. They are also embedded in
the code (see `src/georescue/reporting/evidence.py`, constants
`ATTRIBUTION_SENTINEL`, `ATTRIBUTION_DEM`, `ATTRIBUTION_OSM`).

> Contains modified Copernicus Sentinel data 2026.

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus
> Defence and Space GmbH 2014–2018 provided under COPERNICUS by the
> European Union and ESA; all rights reserved.

> © OpenStreetMap contributors.

## Primary data sources

### Copernicus Sentinel-1

Sentinel-1 Interferometric Wide swath Ground Range Detected (IW GRD)
products at 10 m pixel spacing. Acquired through the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).

- Licence: Copernicus free, full and open data policy.
- Products used in the case study:
  - `S1D_IW_GRDH_1SDV_20260816T122141_20260816T122206_004151_007980_B091_COG.SAFE`
  - `S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4_01B4.SAFE`

### Copernicus DEM 30 m (WorldDEM-30)

1 arc-second digital elevation model used for terrain correction and
slope masking. Retrieved from the
[AWS Open Data Copernicus DEM bucket](https://registry.opendata.aws/copernicus-dem/).

- Licence: see the attribution statement above.

### OpenStreetMap (historical snapshot)

Pre-event buildings, roads, bridges, settlements, and destinations for
the area of interest. Retrieved through the
[ohsome API v2](https://api.ohsome.org) with an explicit snapshot date
of 27 July 2026, which is the latest documented snapshot before the
26 August 2026 event.

- Licence: ODbL. Any published derived product must remain under ODbL.
- Attribution: "© OpenStreetMap contributors."

### Sentinel-2 L2A (optional corroboration)

Sentinel-2 Level-2A surface reflectance. Used for optical corroboration
when cloud cover permits. During the August 2026 event, monsoon cloud
cover prevented useful optical imagery, so the case-study results are
**SAR-only**. Sentinel-2 acquisition is fully implemented in the code
and will be used when conditions allow.

- Licence: Copernicus free, full and open data policy.

## Services used

### Copernicus Data Space Ecosystem (CDSE)

- STAC and OData catalogue for Sentinel-1 and Sentinel-2 scene search.
- OAuth 2.0 Client Credentials for authentication.
- Product download.

Registration is free. The application reads credentials from
`CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in `.env`.

### ohsome API v2

Historical OpenStreetMap extraction service, provided by HeiGIT.

- Public service with a free API key.
- Endpoint: `https://api.heigit.org/ohsome-api/v2/extraction/features.parquet`
- Authentication: `Authorization` header with the API key.
- Attribution: "© OpenStreetMap contributors."

### Copernicus DEM on AWS

Public S3 bucket, no credentials required. Used to download 1° × 1°
DEM tiles and mosaic them for the AOI.

## Software libraries

GeoRescue AI uses the following open-source libraries. Each is used
under its published licence.

| Library                      | Purpose                          |
| ---------------------------- | -------------------------------- |
| `rasterio`                   | Raster I/O and reprojection      |
| `rioxarray`                  | xarray integration with rasterio |
| `geopandas`                  | Vector operations                |
| `shapely`                    | Planar geometry operations       |
| `fiona`                      | Vector file I/O                  |
| `pyproj`                     | Coordinate transformations       |
| `networkx`                   | Road-network graph analysis      |
| `numpy`, `scipy`, `pandas`   | Numerical and tabular operations |
| `sarsen`                     | Sentinel-1 terrain correction    |
| `streamlit`                  | Dashboard framework              |
| `folium`, `streamlit-folium` | Interactive mapping              |
| `pydantic`                   | Data validation                  |
| `requests`                   | HTTP client                      |

## Reference datasets used only for evaluation

The following reference product was used **only** to evaluate the
system's output. It was **never** fed into the flood-detection pipeline.
The competition rules prohibit using published damage maps as inputs to
the algorithm.

### Copernicus Emergency Management Service — EMSR927

Activation EMSR927 for the August 2026 Nepal flood. Product used:

- AOI01 = Syapru Besi.
- Delineation product: observed-event polygon (`observedEventA_v1`)
  representing the mass-movement footprint, and built-up points with
  damage grading (`builtUpP_v1`).
- Source: [Copernicus Emergency Management Service mapping portal](https://mapping.emergency.copernicus.eu/).
- Licence: free of charge; attribution: "European Union, Copernicus
  Emergency Management Service data."

**Used for evaluation only.** See `docs/technical_report.md`, Section 6.

## Method references

The following published datasets were permitted by the hackathon as
training resources but were **not used** in this submission because we
implemented a transparent, non-trained, threshold-based detector. They
are cited here for completeness and honesty about what was not used.

- **Kuro Siwo** — Bountos, N. I., et al. (2024). "Kuro Siwo: A Global
  Multi-Modal Dataset for Flood Segmentation in the Era of SAR
  Foundation Models." NeurIPS 2024.
- **Sen1Floods11** — Bonafilia, D., et al. (2020). "Sen1Floods11: A
  Georeferenced Dataset to Train and Test Deep Learning Flood
  Algorithms." CVPR Workshops 2020.

If, in a future version, the baseline threshold detector is replaced by
a trained model, the training data and the model weights will be
documented here with their licences and citations.

## What we did not use

The following are mentioned because the hackathon rules refer to them.
None of them were used as inputs to the flood-detection or
infrastructure-analysis pipeline.

- Published EMSR927 flood or damage maps as algorithm inputs — **not
  used**.
- UNOSAT damage or flood maps — **not used**.
- Any other published disaster-damage map — **not used**.
- Post-event OpenStreetMap edits — **not used**. All OSM data is a
  pre-event snapshot dated 2026-07-27, before the 2026-08-26 event.

## Reproduction

To reproduce the results in the technical report:

1. Follow `README.md` to install dependencies and configure `.env`.
2. Run `scripts/download_case_study.py` with the Syapru Besi AOI to
   download the two Sentinel-1 slices.
3. Run `scripts/run_pipeline.py` to preprocess and detect.
4. Follow the vectorization and evaluation commands in
   `docs/demo_script.md` to produce the flood polygons and the
   evaluation JSON.

The evaluation JSON files are preserved at
`outputs/evaluation/evaluation_summary_observed_event.json` and
`outputs/evaluation/evaluation_summary_buildings.json`.
