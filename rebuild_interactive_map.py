"""
rebuild_interactive_map.py
───────────────────────────
Rebuilds outputs/interactive_map.html from saved files, adding the three
freshwater layers (JRC occurrence, JRC seasonality, HydroLAKES) alongside
the existing detection layers.

Run from the project root:
    python rebuild_interactive_map.py
"""

from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — needed for .rio accessor
import geopandas as gpd
import rasterio

import config
from data.known_sites import get_known_sites
from processing.terrain import compute_multidirectional_hillshade
from processing.vegetation import compute_ndvi, compute_ndvi_anomaly
from processing.sar import compute_sar_anomaly, compute_combined_sar_anomaly
from visualize.interactive import build_interactive_map

WATER_DIR = config.RAW_DATA_DIR / "water_layers"


def _load_tif_as_da(path: Path, name: str) -> xr.DataArray:
    """Load a single-band GeoTIFF as a named xarray DataArray."""
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    da.name = name
    return da


def _load_composite_score() -> xr.DataArray:
    path = config.OPTIMIZED_SCORE_PATH
    if not path.exists():
        path = config.COMPOSITE_SCORE_PATH
    print(f"  Loading composite score from {path.name} …")
    return _load_tif_as_da(path, "composite_score")


def _load_hillshade() -> xr.DataArray | None:
    dem_path = config.DEM_PATH
    if not dem_path.exists():
        print("  DEM not found — skipping hillshade.")
        return None
    print("  Computing hillshade from SRTM DEM …")
    dem = _load_tif_as_da(dem_path, "dem")
    return compute_multidirectional_hillshade(dem)


def _load_ndvi_anomaly() -> xr.DataArray | None:
    s2_path = config.S2_COMPOSITE_PATH
    if not s2_path.exists():
        print("  Sentinel-2 composite not found — skipping NDVI anomaly.")
        return None
    print("  Computing NDVI anomaly from S2 composite …")
    s2 = xr.open_dataset(s2_path)
    b04 = s2.get("B04")
    b08 = s2.get("B08")
    if b04 is None or b08 is None:
        print("  B04/B08 bands not found in S2 composite.")
        return None
    ndvi = compute_ndvi(b04, b08)
    return compute_ndvi_anomaly(ndvi)


def _load_sar_anomaly() -> xr.DataArray | None:
    s1_path = config.S1_COMPOSITE_PATH
    if not s1_path.exists():
        print("  Sentinel-1 composite not found — skipping SAR anomaly.")
        return None
    print("  Computing SAR anomaly from S1 composite …")
    s1 = xr.open_dataset(s1_path)
    vh = s1.get("VH") if "VH" in s1 else s1.get("vh")
    vv = s1.get("VV") if "VV" in s1 else s1.get("vv")
    if vh is None or vv is None:
        print("  VH/VV bands not found in S1 composite.")
        return None
    sar_anoms = compute_sar_anomaly(vv, vh)
    return compute_combined_sar_anomaly(
        sar_anoms.get("vv_anomaly"), sar_anoms.get("vh_anomaly"), None
    )


def _load_water_layers():
    print("  Loading JRC occurrence …")
    occ  = _load_tif_as_da(WATER_DIR / "jrc_occurrence.tif",  "jrc_occurrence")
    print("  Loading JRC seasonality …")
    seas = _load_tif_as_da(WATER_DIR / "jrc_seasonality.tif", "jrc_seasonality")
    print("  Loading HydroLAKES …")
    lakes = gpd.read_file(WATER_DIR / "hydrolakes_aoi.gpkg")
    if lakes.crs is None or str(lakes.crs).upper() != "EPSG:4326":
        lakes = lakes.to_crs("EPSG:4326")
    return occ, seas, lakes


if __name__ == "__main__":
    print("─── Loading layers ───")
    score    = _load_composite_score()
    hillshade = _load_hillshade()
    ndvi      = _load_ndvi_anomaly()
    sar       = _load_sar_anomaly()
    sites     = get_known_sites(csv_path=config.KNOWN_SITES_CSV, target_crs=config.CRS)

    candidates = None
    if config.CANDIDATES_GEOJSON_PATH.exists():
        print("  Loading candidate sites …")
        candidates = gpd.read_file(config.CANDIDATES_GEOJSON_PATH)

    jrc_occ, jrc_seas, lakes = _load_water_layers()

    print("\n─── Building interactive map ───")
    build_interactive_map(
        hillshade=hillshade,
        ndvi_anomaly=ndvi,
        sar_anomaly=sar,
        score=score,
        sites_gdf=sites,
        candidates_gdf=candidates,
        jrc_occurrence=jrc_occ,
        jrc_seasonality=jrc_seas,
        hydrolakes_gdf=lakes,
    )
