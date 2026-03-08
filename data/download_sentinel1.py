"""download_sentinel1.py — Search and retrieve Sentinel-1 GRD imagery from
Microsoft Planetary Computer.

Builds a multi-temporal median composite of VV and VH polarization bands,
clips to the AOI, and reprojects to the project CRS.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import config

S1_COLLECTION = "sentinel-1-grd"


def _open_catalog():
    """Open the Planetary Computer STAC catalog with signed-URL modifier.

    Returns:
        pystac_client.Client instance, or None if connection fails.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError as exc:
        print(f"[download_sentinel1] Missing dependency: {exc}. "
              "Install pystac-client and planetary-computer.")
        return None

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        return catalog
    except Exception as exc:
        print(f"[download_sentinel1] Could not connect to Planetary Computer: {exc}")
        return None


def search_sentinel1_scenes(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
) -> list:
    """Search Planetary Computer for Sentinel-1 GRD scenes.

    Searches both SENTINEL-1A and SENTINEL-1B acquisitions. No platform
    filter is applied — the collection itself constrains results to S1 GRD.

    Args:
        bbox: AOI bounding box (west, south, east, north) in WGS84.
        date_range: Tuple of ISO date strings (start, end).

    Returns:
        List of pystac Item objects, or an empty list on failure.
    """
    catalog = _open_catalog()
    if catalog is None:
        return []

    start, end = date_range
    print(
        f"[download_sentinel1] Searching {S1_COLLECTION} "
        f"{start} → {end} ..."
    )

    try:
        search = catalog.search(
            collections=[S1_COLLECTION],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
        )
        items = list(search.items())
    except Exception as exc:
        print(f"[download_sentinel1] Search failed: {exc}")
        return []

    if not items:
        print("[download_sentinel1] No Sentinel-1 scenes found.")
        return []

    # Group by relative orbit number so that each orbital track is represented.
    # S1 IW swaths are ~250 km wide; for a ~270 km × 190 km AOI we need at
    # least 2–3 distinct tracks to achieve full spatial coverage.  Taking only
    # chronological scenes risks selecting many scenes from the same swath.
    from collections import defaultdict
    orbit_groups: dict[int, list] = defaultdict(list)
    for item in items:
        orbit = item.properties.get("sat:relative_orbit", 0)
        orbit_groups[orbit].append(item)

    # From each orbit, take up to 4 scenes (temporal spread for speckle averaging)
    selected = []
    for orbit, orbit_items in orbit_groups.items():
        selected.extend(orbit_items[:4])

    print(
        f"[download_sentinel1] Found {len(items)} scenes across "
        f"{len(orbit_groups)} relative orbits; selected {len(selected)} "
        f"for compositing (up to 4 per orbit)."
    )
    return selected


def _resolve_polarization_asset(item, polarization: str) -> Optional[str]:
    """Find the href for a given polarization band in a STAC item.

    Sentinel-1 GRD items on Planetary Computer use asset keys like 'VV',
    'VH', 'vv', 'vh', or embed polarization in the key name. This function
    tries several naming conventions.

    Args:
        item: A signed pystac Item.
        polarization: Polarization string, e.g. 'VV' or 'VH'.

    Returns:
        URL string for the asset href, or None if not found.
    """
    candidates = [
        polarization,
        polarization.lower(),
        f"measurement_{polarization.lower()}",
        f"{polarization.lower()}-iw",
    ]
    for key in candidates:
        if key in item.assets:
            return item.assets[key].href

    # Fallback: scan all asset keys for the polarization string
    for key, asset in item.assets.items():
        if polarization.lower() in key.lower():
            return asset.href

    return None


def _load_polarization_band(
    item,
    polarization: str,
    bbox_wgs84: Optional[tuple[float, float, float, float]] = None,
) -> Optional[xr.DataArray]:
    """Load a single Sentinel-1 polarization band from a STAC item.

    Args:
        item: A signed pystac Item.
        polarization: 'VV' or 'VH'.
        bbox_wgs84: Optional AOI bounding box for immediate clipping.

    Returns:
        xarray DataArray (float32, dB scale) or None on failure.
    """
    href = _resolve_polarization_asset(item, polarization)
    if href is None:
        print(
            f"[download_sentinel1] '{polarization}' asset not found in item "
            f"{item.id}."
        )
        return None

    try:
        import rioxarray
        da = rioxarray.open_rasterio(href, masked=True, overview_level=2).squeeze(
            "band", drop=True
        )
        # Clip to AOI immediately to keep memory footprint small
        if bbox_wgs84 is not None:
            from pyproj import Transformer
            west, south, east, north = bbox_wgs84
            native_crs = da.rio.crs
            if native_crs is not None and str(native_crs).upper() != "EPSG:4326":
                tf = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
                x_min, y_min = tf.transform(west, south)
                x_max, y_max = tf.transform(east, north)
            else:
                x_min, y_min, x_max, y_max = west, south, east, north
            try:
                da = da.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
            except Exception:
                pass  # Non-fatal
        # Convert linear scale to dB if values look like linear backscatter
        data = da.values
        finite = data[np.isfinite(data)]
        if len(finite) > 0 and np.nanmedian(finite) < 10:
            da = 10.0 * np.log10(da.clip(min=1e-10))
        return da.astype(np.float32)
    except Exception as exc:
        print(f"[download_sentinel1] Failed to load {polarization} from {href}: {exc}")
        return None


def build_s1_composite(
    items: list,
    polarizations: list[str] = ["VV", "VH"],
    max_scenes: int = 40,
    bbox_wgs84: Optional[tuple[float, float, float, float]] = None,
) -> Optional[dict[str, xr.DataArray]]:
    """Build a multi-temporal median composite for each SAR polarization.

    Loads up to *max_scenes* items at overview_level=2 (~40m resolution),
    clips each to the AOI, stacks along a 'time' dimension, and computes a
    pixel-wise median to suppress speckle and temporal noise.

    Args:
        items: List of signed pystac Sentinel-1 GRD Items.
        polarizations: List of polarization band keys to composite.
        max_scenes: Maximum number of scenes to process.
        bbox_wgs84: AOI bounding box for immediate per-scene clipping.

    Returns:
        Dict of polarization → median DataArray, or None if nothing loaded.
    """
    scenes_to_use = items[:max_scenes]
    print(
        f"[download_sentinel1] Building SAR median composite from "
        f"{len(scenes_to_use)} scenes ..."
    )

    per_pol_stacks: dict[str, list[xr.DataArray]] = {p: [] for p in polarizations}

    for idx, item in enumerate(scenes_to_use):
        for pol in polarizations:
            da = _load_polarization_band(item, pol, bbox_wgs84=bbox_wgs84)
            if da is not None:
                per_pol_stacks[pol].append(da)
        print(f"[download_sentinel1]   Scene {idx + 1}/{len(scenes_to_use)} loaded.")

    composites: dict[str, xr.DataArray] = {}
    for pol, stack in per_pol_stacks.items():
        if not stack:
            print(f"[download_sentinel1] No valid data for polarization {pol}.")
            continue
        try:
            reference = stack[0]
            aligned = [reference]
            for da in stack[1:]:
                try:
                    aligned.append(da.rio.reproject_match(reference))
                except Exception:
                    aligned.append(da)
            stacked = xr.concat(aligned, dim="time")
            composites[pol] = stacked.median(dim="time", skipna=True)
        except Exception as exc:
            print(f"[download_sentinel1] Compositing failed for {pol}: {exc}")

    if not composites:
        print("[download_sentinel1] SAR composite is empty.")
        return None

    print("[download_sentinel1] SAR composite complete.")
    return composites


def get_sentinel1_bands(
    bbox: tuple[float, float, float, float] = config.AOI_BBOX_WGS84,
    date_range: tuple[str, str] = config.S1_DATE_RANGE,
    target_crs: str = config.CRS,
    cache_path: Path = config.S1_COMPOSITE_PATH,
    force_download: bool = False,
) -> Optional[dict[str, xr.DataArray]]:
    """Top-level function: search, composite, reproject, and return S1 bands.

    Returns a dictionary with keys 'VV' and 'VH', each containing a
    multi-temporal median composite DataArray in the project CRS. Loads
    from a NetCDF cache if available.

    Args:
        bbox: AOI bounding box in WGS84 (west, south, east, north).
        date_range: Tuple of ISO date strings (start, end).
        target_crs: Target projected CRS string.
        cache_path: Path to save/load a NetCDF cache.
        force_download: Re-download even if cache exists.

    Returns:
        Dict of polarization name → xarray DataArray, or None on failure.
    """
    if cache_path.exists() and not force_download:
        print(f"[download_sentinel1] Loading cached SAR composite from {cache_path}.")
        try:
            ds = xr.open_dataset(cache_path)
            return {var: ds[var] for var in ds.data_vars}
        except Exception as exc:
            print(f"[download_sentinel1] Cache read failed ({exc}); re-downloading.")

    items = search_sentinel1_scenes(bbox, date_range)
    if not items:
        return None

    composites = build_s1_composite(items, bbox_wgs84=bbox)
    if composites is None:
        return None

    # Reproject to project CRS and clip to AOI
    reprojected: dict[str, xr.DataArray] = {}
    for pol, da in composites.items():
        try:
            reprojected[pol] = da.rio.reproject(target_crs)
        except Exception as exc:
            print(f"[download_sentinel1] Reprojection failed for {pol}: {exc}")
            reprojected[pol] = da

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(reprojected)
        ds.to_netcdf(cache_path)
        print(f"[download_sentinel1] SAR composite cached to {cache_path}.")
    except Exception as exc:
        print(f"[download_sentinel1] Could not save cache: {exc}")

    return reprojected
