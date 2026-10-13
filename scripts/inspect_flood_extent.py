"""Inspect the candidate flood extent GeoJSON.

Prints summary statistics and per-feature details. Reads from
``data/processed/candidate_flood_extent.geojson`` by default; pass a
different path as the first argument to override.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/processed/candidate_flood_extent.geojson"
    )
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)

    features = g.get("features", [])
    print(f"File: {path}")
    print(f"Feature count: {len(features)}")
    print(f"Collection properties: {json.dumps(g.get('properties', {}), indent=2)}")

    if not features:
        return 0

    areas = sorted(
        (f["properties"]["area_sqkm"] for f in features),
        reverse=True,
    )
    total = sum(areas)
    print(f"\nTotal area: {total:.6f} km² ({total * 100:.2f} ha)")
    print(f"Largest 10 polygon areas (km²): {[round(a, 6) for a in areas[:10]]}")
    print(f"Smallest 5 polygon areas (km²): {[round(a, 6) for a in areas[-5:]]}")
    print(f"Median area: {sorted(areas)[len(areas) // 2]:.6f} km²")

    biggest = max(features, key=lambda f: f["properties"]["area_sqkm"])
    print(f"\nBiggest polygon:")
    print(f"  id: {biggest['id']}")
    print(f"  area km²: {biggest['properties']['area_sqkm']}")
    print(f"  area ha: {biggest['properties']['area_hectares']}")
    print(f"  geometry type: {biggest['geometry']['type']}")

    coords = biggest["geometry"]["coordinates"]
    if biggest["geometry"]["type"] == "Polygon":
        outer = coords[0]
    else:
        outer = coords[0][0]
    print(f"  first 3 coords: {outer[:3]}")


if __name__ == "__main__":
    sys.exit(main())