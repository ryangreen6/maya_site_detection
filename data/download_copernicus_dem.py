"""download_copernicus_dem.py — Download Copernicus GLO-30 DEM from Planetary Computer.

The Copernicus DEM has ~4 m vertical RMSE vs SRTM's ~16 m, and is assembled
from TanDEM-X acquisitions at consistent orbital geometry, which reduces
systematic striping artefacts.  Like SRTM it still measures the canopy surface
in forested terrain, but the higher vertical accuracy improves slope and TPI
derivatives — useful as a secondary terrain signal alongside GEDI.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers .rio accessor

import config


def get_copernicus_dem(
    bbox: tuple[float, float, float, float] = config.AOI_BBOX_WGS84,
    output_path: Path = config.COPERNICUS_DEM_PATH,
    target_crs: str = config.CRS,
    force_download: bool = False,
) -> Optional[xr.DataArray]:
    """Download and mosaic Copernicus GLO-30 DEM tiles from Planetary Computer.

    Searches the ``cop-dem-glo-30`` STAC collection, opens each tile at native
    resolution, mosaics them, reprojects to the project CRS, and saves a
    GeoTIFF cache.

    Args:
        bbox: WGS84 bounding box (west, south, east, north).
        output_path: Path to write the merged GeoTIFF.
        target_crs: EPSG string for the output projection.
        force_download: Re-download even if cache exists.

    Returns:
        DEM DataArray in *target_crs*, or None on failure.
    """
    if output_path.exists() and not force_download:
        print(f"[cop_dem] Using cached Copernicus DEM: {output_path}")
        da = rioxarray.open_rasterio(str(output_path), masked=True).squeeze("band", drop=True)
        da = da.rename("cop_dem")
        print(f"[cop_dem] Loaded: shape={da.values.shape}, CRS={da.rio.crs}")
        return da

    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("[cop_dem] pystac_client and planetary_computer are required.")
        return None

    west, south, east, north = bbox
    print(f"[cop_dem] Searching cop-dem-glo-30 tiles for bbox {bbox} ...")

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=[west, south, east, north],
        )
        items = list(search.items())
        print(f"[cop_dem] Found {len(items)} tiles.")
        if not items:
            print("[cop_dem] No tiles found.")
            return None
    except Exception as exc:
        print(f"[cop_dem] STAC search failed: {exc}")
        return None

    # Open all tiles and mosaic
    tile_arrays: list[xr.DataArray] = []
    for item in items:
        try:
            # Asset key is 'data' in cop-dem-glo-30
            asset = item.assets.get("data")
            if asset is None:
                # Fallback: try first available asset
                asset = next(iter(item.assets.values()))
            href = asset.href
            tile = rioxarray.open_rasterio(href, masked=True).squeeze("band", drop=True)
            # Clip to AOI immediately to reduce memory
            try:
                tile = tile.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326")
            except Exception:
                pass
            tile_arrays.append(tile.astype(np.float32))
        except Exception as exc:
            print(f"[cop_dem] Failed to open tile {item.id}: {exc}")
            continue

    if not tile_arrays:
        print("[cop_dem] No tiles successfully loaded.")
        return None

    print(f"[cop_dem] Mosaicking {len(tile_arrays)} tiles ...")
    try:
        from rioxarray.merge import merge_arrays
        merged = merge_arrays(tile_arrays)
    except Exception as exc:
        print(f"[cop_dem] Mosaic failed: {exc}")
        return None

    # Reproject to project CRS
    try:
        if merged.rio.crs is None:
            merged = merged.rio.write_crs("EPSG:4326")
        print(f"[cop_dem] Reprojecting to {target_crs} ...")
        dem = merged.rio.reproject(target_crs)
    except Exception as exc:
        print(f"[cop_dem] Reprojection failed: {exc}")
        return None

    dem = dem.rename("cop_dem")

    # Save cache
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dem.rio.to_raster(str(output_path))
        print(f"[cop_dem] Saved to {output_path}. Shape: {dem.values.shape}")
    except Exception as exc:
        print(f"[cop_dem] Could not save: {exc}")

    return dem
