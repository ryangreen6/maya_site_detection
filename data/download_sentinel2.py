"""download_sentinel2.py — Search and retrieve Sentinel-2 L2A imagery from
Microsoft Planetary Computer.

Builds a cloud-masked median composite from the least-cloudy scenes in the
requested date range, returning Bands B04, B07, B08, and the SCL mask as
xarray DataArrays clipped to the AOI.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import config

# Planetary Computer collections
S2_COLLECTION = "sentinel-2-l2a"

# SCL classes to mask out: cloud shadows (3), medium cloud (8),
# high cloud (9), cirrus (10)
CLOUD_SCL_CLASSES = [3, 8, 9, 10]


def _open_catalog():
    """Open the Planetary Computer STAC catalog with signed-URL modifier.

    Returns:
        pystac_client.Client instance, or None if import/connection fails.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError as exc:
        print(f"[download_sentinel2] Missing dependency: {exc}. "
              "Install pystac-client and planetary-computer.")
        return None

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        return catalog
    except Exception as exc:
        print(f"[download_sentinel2] Could not connect to Planetary Computer: {exc}")
        return None


def search_sentinel2_scenes(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    cloud_threshold: int,
) -> list:
    """Search Planetary Computer for Sentinel-2 L2A scenes matching criteria.

    Args:
        bbox: AOI bounding box (west, south, east, north) in WGS84.
        date_range: Tuple of ISO date strings (start, end).
        cloud_threshold: Maximum scene-level cloud cover percentage.

    Returns:
        List of pystac Item objects sorted by ascending cloud cover,
        or an empty list if the search fails or returns nothing.
    """
    catalog = _open_catalog()
    if catalog is None:
        return []

    start, end = date_range
    print(
        f"[download_sentinel2] Searching {S2_COLLECTION} "
        f"{start} → {end}, cloud ≤ {cloud_threshold}% ..."
    )

    try:
        search = catalog.search(
            collections=[S2_COLLECTION],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": cloud_threshold}},
        )
        items = list(search.items())
    except Exception as exc:
        print(f"[download_sentinel2] Search failed: {exc}")
        return []

    if not items:
        print("[download_sentinel2] No scenes found matching criteria.")
        return []

    # Group by MGRS tile so we pick the best scene from *each* tile rather
    # than taking the N globally least-cloudy scenes (which could all come
    # from one tile on the edge of the AOI).
    from collections import defaultdict
    tile_groups: dict[str, list] = defaultdict(list)
    for item in items:
        tile_id = item.properties.get("s2:mgrs_tile", item.id)
        tile_groups[tile_id].append(item)

    selected = []
    for tile_id, tile_items in tile_groups.items():
        tile_items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        selected.append(tile_items[0])  # least-cloudy scene for this tile

    selected.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    n_tiles = len(tile_groups)
    print(
        f"[download_sentinel2] Found {len(items)} scenes across {n_tiles} MGRS tiles; "
        f"selected 1 best scene per tile ({len(selected)} total)."
    )
    return selected


def _clip_to_bbox(
    da: xr.DataArray,
    bbox_wgs84: tuple[float, float, float, float],
) -> xr.DataArray:
    """Clip a DataArray to a WGS84 bounding box, reprojecting the bbox as needed.

    Converts the WGS84 bbox to the DataArray's native CRS before clipping,
    so the clip is applied in the correct coordinate space.

    Args:
        da: Input DataArray with a valid CRS set.
        bbox_wgs84: Bounding box (west, south, east, north) in WGS84.

    Returns:
        Clipped DataArray, or the original if clipping fails.
    """
    west, south, east, north = bbox_wgs84
    try:
        native_crs = da.rio.crs
        if native_crs is not None and str(native_crs).upper() != "EPSG:4326":
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
            x_min, y_min = tf.transform(west, south)
            x_max, y_max = tf.transform(east, north)
        else:
            x_min, y_min, x_max, y_max = west, south, east, north
        return da.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
    except Exception:
        return da  # Clip failure is non-fatal; use full scene


def _load_band(
    item,
    band_name: str,
    overview_level: int = 2,
    bbox_wgs84: Optional[tuple[float, float, float, float]] = None,
) -> Optional[xr.DataArray]:
    """Load a single band from a STAC item, clipped to the AOI at reduced resolution.

    Opens the COG at the requested overview level (~4× downsampled at level 2,
    giving ~40 m effective resolution for 10 m bands) then immediately clips to
    the AOI bounding box to minimise in-memory footprint.

    Args:
        item: A signed pystac Item.
        band_name: Asset key (e.g., 'B04', 'SCL').
        overview_level: COG overview level (0 = first/coarsest overview;
                        None = full resolution). Default 2 ≈ 40 m for 10 m bands.
        bbox_wgs84: AOI bounding box (west, south, east, north) in WGS84 used
                    to clip the scene immediately after opening.

    Returns:
        Clipped xarray DataArray or None if the asset is unavailable or load fails.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        print("[download_sentinel2] rioxarray is not installed.")
        return None

    if band_name not in item.assets:
        print(f"[download_sentinel2] Band '{band_name}' not in item assets.")
        return None

    href = item.assets[band_name].href
    try:
        import rioxarray
        da = rioxarray.open_rasterio(
            href,
            masked=True,
            overview_level=overview_level,
        ).squeeze("band", drop=True)

        if bbox_wgs84 is not None:
            da = _clip_to_bbox(da, bbox_wgs84)

        return da
    except Exception as exc:
        print(f"[download_sentinel2] Failed to load {band_name} from {href}: {exc}")
        return None


def _apply_cloud_mask(
    band: xr.DataArray,
    scl: xr.DataArray,
    cloud_classes: list[int] = CLOUD_SCL_CLASSES,
) -> xr.DataArray:
    """Apply a cloud mask derived from the SCL layer to a band DataArray.

    Pixels whose SCL class is in *cloud_classes* are set to NaN.

    Args:
        band: Band DataArray to mask.
        scl: Scene Classification Layer DataArray (same spatial extent).
        cloud_classes: List of SCL integer class values to mask.

    Returns:
        Masked xarray DataArray with cloud pixels set to NaN.
    """
    try:
        scl_resampled = scl.rio.reproject_match(band)
    except Exception:
        # If reprojection fails, skip masking rather than crashing
        return band

    mask = xr.zeros_like(scl_resampled, dtype=bool)
    for cls in cloud_classes:
        mask = mask | (scl_resampled == cls)

    return band.where(~mask)


def build_s2_composite(
    items: list,
    bands: list[str] = ["B02", "B03", "B04", "B07", "B08"],
    max_scenes: int = 30,
    overview_level: int = 2,
    bbox_wgs84: Optional[tuple[float, float, float, float]] = None,
) -> Optional[dict[str, xr.DataArray]]:
    """Build a median composite from a list of Sentinel-2 STAC items.

    Loads up to *max_scenes* least-cloudy scenes at a reduced COG overview
    level, clips each immediately to the AOI, applies the SCL cloud mask,
    stacks along a 'time' dimension, and reduces to a per-pixel median.
    Loading at overview_level=2 reduces each 10 m band to ~40 m resolution,
    cutting per-scene RAM by ~16× versus full resolution.

    Args:
        items: Sorted list of signed pystac Items (least cloudy first).
        bands: List of band asset keys to load.
        max_scenes: Maximum number of scenes to load. Default 8 keeps total
                    in-memory footprint manageable on consumer hardware.
        overview_level: COG overview level for loading (2 ≈ 40 m for 10 m bands).
        bbox_wgs84: AOI bounding box for immediate per-scene clipping.

    Returns:
        Dictionary mapping band name → median-composite DataArray,
        or None if compositing fails.
    """
    scenes_to_use = items[:max_scenes]
    print(f"[download_sentinel2] Building median composite from "
          f"{len(scenes_to_use)} scenes (overview_level={overview_level}) ...")

    per_band_stacks: dict[str, list[xr.DataArray]] = {b: [] for b in bands}

    for idx, item in enumerate(scenes_to_use):
        scl = _load_band(item, "SCL", overview_level=overview_level, bbox_wgs84=bbox_wgs84)
        for band_name in bands:
            da = _load_band(item, band_name, overview_level=overview_level, bbox_wgs84=bbox_wgs84)
            if da is None:
                continue
            if scl is not None:
                da = _apply_cloud_mask(da, scl)
            da = da.astype(np.float32)
            per_band_stacks[band_name].append(da)

        print(f"[download_sentinel2]   Scene {idx + 1}/{len(scenes_to_use)} loaded "
              f"({item.properties.get('eo:cloud_cover', '?'):.1f}% cloud).")

    composites: dict[str, xr.DataArray] = {}
    for band_name, stack in per_band_stacks.items():
        if not stack:
            print(f"[download_sentinel2] No valid data for band {band_name}.")
            continue
        try:
            if len(stack) == 1:
                composites[band_name] = stack[0]
            else:
                # Mosaic tiles across their full spatial union using merge_arrays,
                # which preserves each tile's extent rather than clipping all to
                # the first tile (the bug with reproject_match as reference).
                from rioxarray.merge import merge_arrays
                composites[band_name] = merge_arrays(stack, nodata=np.nan)
        except Exception as exc:
            print(f"[download_sentinel2] Failed to composite band {band_name}: {exc}")

    if not composites:
        print("[download_sentinel2] Composite is empty — no bands loaded.")
        return None

    print("[download_sentinel2] Composite complete.")
    return composites


def reproject_to_project_crs(
    composites: dict[str, xr.DataArray],
    target_crs: str = config.CRS,
) -> dict[str, xr.DataArray]:
    """Reproject all composite bands to the project CRS.

    Args:
        composites: Dictionary of band name → DataArray.
        target_crs: EPSG string for the target CRS.

    Returns:
        Dictionary with the same keys, values reprojected.
    """
    reprojected: dict[str, xr.DataArray] = {}
    for band_name, da in composites.items():
        try:
            reprojected[band_name] = da.rio.reproject(target_crs)
        except Exception as exc:
            print(f"[download_sentinel2] Reprojection failed for {band_name}: {exc}")
            reprojected[band_name] = da
    return reprojected


def get_sentinel2_bands(
    bbox: tuple[float, float, float, float] = config.SITE_CORE_BBOX,
    date_range: tuple[str, str] = config.S2_DATE_RANGE,
    cloud_threshold: int = config.S2_CLOUD_THRESHOLD,
    target_crs: str = config.CRS,
    cache_path: Path = config.S2_COMPOSITE_PATH,
    force_download: bool = False,
) -> Optional[dict[str, xr.DataArray]]:
    """Top-level function: search, composite, and return Sentinel-2 bands.

    Checks for a cached NetCDF composite before downloading. Returns a
    dictionary of cloud-masked, median-composited DataArrays for bands
    B04, B07, and B08, reprojected to the project CRS.

    Args:
        bbox: AOI bounding box in WGS84 (west, south, east, north).
        date_range: Tuple of ISO date strings (start, end).
        cloud_threshold: Maximum scene cloud cover percentage.
        target_crs: Target projected CRS string.
        cache_path: Path to save/load a NetCDF cache of the composite.
        force_download: Re-download even if cache exists.

    Returns:
        Dict of band name → xarray DataArray, or None on failure.
    """
    if cache_path.exists() and not force_download:
        print(f"[download_sentinel2] Loading cached composite from {cache_path}.")
        try:
            ds = xr.open_dataset(cache_path)
            return {var: ds[var] for var in ds.data_vars}
        except Exception as exc:
            print(f"[download_sentinel2] Cache read failed ({exc}); re-downloading.")

    items = search_sentinel2_scenes(bbox, date_range, cloud_threshold)
    if not items:
        return None

    composites = build_s2_composite(items, bbox_wgs84=bbox)
    if composites is None:
        return None

    composites = reproject_to_project_crs(composites, target_crs)

    # Align all bands to B04's grid so they share the same coordinates in the
    # NetCDF. Without this, xarray.Dataset creates a union coordinate system
    # where bands at different native resolutions have non-overlapping valid
    # pixels, making multi-band arithmetic produce all-NaN results.
    # Align all bands to B04's grid (10 m native → ~40 m at OL2).
    # B02, B03, B04 are all 10 m native; B07 is 20 m; B08 is 10 m.
    # reproject_match ensures they all share the same coordinate grid so
    # multi-band arithmetic doesn't produce all-NaN from misaligned coords.
    if "B04" in composites:
        reference_band = composites["B04"]
        for band_name in list(composites.keys()):
            if band_name == "B04":
                continue
            try:
                composites[band_name] = composites[band_name].rio.reproject_match(
                    reference_band
                )
            except Exception as exc:
                print(f"[download_sentinel2] Could not align {band_name} to B04 grid: {exc}")

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(composites)
        ds.to_netcdf(cache_path)
        print(f"[download_sentinel2] Composite cached to {cache_path}.")
    except Exception as exc:
        print(f"[download_sentinel2] Could not save cache: {exc}")

    return composites
