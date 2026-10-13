"""List all layers and their contents in the EMSR927 GeoPackage.

This is a one-off inspection tool. It does not modify the GeoPackage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fiona
import geopandas as gpd


def main() -> int:
    path = Path("data/processed/emr927_aoi01_gra.gpkg")
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    layers = fiona.listlayers(str(path))
    print(f"GeoPackage: {path.name}")
    print(f"Layers found: {len(layers)}\n")

    for layer in layers:
        print(f"--- {layer} ---")
        try:
            gdf = gpd.read_file(path, layer=layer)
        except Exception as exc:
            print(f"  (could not read: {exc})\n")
            continue

        print(f"  Features: {len(gdf)}")
        print(f"  CRS: {gdf.crs}")

        # Handle None geometries safely
        geom_types = []
        for g in gdf.geometry:
            if g is None:
                continue
            geom_types.append(g.geom_type)
        if geom_types:
            print(f"  Geometry types: {sorted(set(geom_types))}")
        else:
            print(f"  Geometry types: (none)")

        non_geom_cols = [c for c in gdf.columns if c != "geometry"]
        print(f"  Columns ({len(non_geom_cols)}): {non_geom_cols[:10]}"
              + (" ..." if len(non_geom_cols) > 10 else ""))

        if not gdf.empty:
            sample = gdf.iloc[0]
            for col in non_geom_cols[:5]:
                val = sample[col]
                if isinstance(val, str) and len(val) > 60:
                    val = val[:60] + "..."
                print(f"    {col}: {val!r}")

        # Bounds with None-geometry protection
        non_null = gdf[gdf.geometry.notna()]
        if not non_null.empty and gdf.crs is not None:
            if gdf.crs.to_epsg() != 4326:
                non_null = non_null.to_crs(4326)
            bounds = non_null.total_bounds
            print(f"  Bounds (WGS84): "
                  f"lon=[{bounds[0]:.4f}, {bounds[2]:.4f}] "
                  f"lat=[{bounds[1]:.4f}, {bounds[3]:.4f}]")

        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())