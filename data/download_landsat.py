"""download_landsat.py — Download Landsat Collection 2 Level-2 thermal data.

Retrieves the Surface Temperature band (lwir11 / Band 10) from Landsat 8 and 9
via Microsoft Planetary Computer.  Stone and masonry structures have markedly
different thermal inertia than surrounding soil and vegetation: they heat up
more slowly during the day and cool more slowly at night, producing a distinct
thermal anomaly over buried or surface Maya architecture.

Landsat Collection 2 Level-2 ST scale factors:
    surface_temperature = DN × 0.00341802 + 149.0  (Kelvin)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

import config


def get_landsat_thermal(
    bbox: tuple[float, float, float, float] = config.SITE_CORE_BBOX,
    date_range: tuple[str, str] = config.LANDSAT_DATE_RANGE,
    cloud_threshold: int = config.LANDSAT_CLOUD_THRESHOLD,
    target_crs: str = config.CRS,
    cache_path: Path = config.LANDSAT_COMPOSITE_PATH,
    force_download: bool = False,
    max_scenes: int = 12,
) -> Optional[xr.DataArray]:
    """Composite Landsat thermal (ST) scenes into a mean surface-temperature layer.

    Args:
        bbox: WGS84 bounding box (west, south, east, north).
        date_range: (start, end) date strings for the search.
        cloud_threshold: Maximum scene cloud cover % to accept.
        target_crs: EPSG string for the output CRS.
        cache_path: Path to a NetCDF cache file.
        force_download: Ignore the cache and re-download.
        max_scenes: Maximum number of scenes to composite.

    Returns:
        Mean surface temperature DataArray (float32, Kelvin) in *target_crs*,
        or None if no data could be retrieved.
    """
    if cache_path.exists() and not force_download:
        print(f"[landsat] Using cached thermal composite: {cache_path}")
        try:
            ds = xr.open_dataset(cache_path)
            da = ds["lwir11"]
            if da.rio.crs is None:
                da = da.rio.write_crs(target_crs)
            print(f"[landsat] Loaded: shape={da.values.shape}")
            return da
        except Exception as exc:
            print(f"[landsat] Cache load failed ({exc}); re-downloading.")

    try:
        import pystac_client
        import planetary_computer
        from pyproj import Transformer
    except ImportError:
        print("[landsat] pystac_client, planetary_computer, and pyproj are required.")
        return None

    west, south, east, north = bbox
    print(f"[landsat] Searching landsat-c2-l2 scenes ({date_range[0]} – {date_range[1]}) ...")

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=[west, south, east, north],
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": cloud_threshold}},
            sortby="datetime",
        )
        items = list(search.items())
        print(f"[landsat] Found {len(items)} scenes; using up to {max_scenes}.")
        items = items[:max_scenes]
        if not items:
            print("[landsat] No scenes found.")
            return None
    except Exception as exc:
        print(f"[landsat] STAC search failed: {exc}")
        return None

    # Build AOI bbox in projected CRS for clipping
    try:
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        x_min, y_min = transformer.transform(west, south)
        x_max, y_max = transformer.transform(east, north)
    except Exception:
        x_min, y_min, x_max, y_max = None, None, None, None

    composites: list[np.ndarray] = []
    reference_da: Optional[xr.DataArray] = None

    for i, item in enumerate(items):
        try:
            asset = item.assets.get("lwir11")
            if asset is None:
                print(f"[landsat] Scene {i+1}: no lwir11 asset; skipping.")
                continue

            da = rioxarray.open_rasterio(
                asset.href, masked=True, overview_level=2
            ).squeeze("band", drop=True)

            # Reproject to project CRS
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:32616")  # Landsat scenes often in UTM
            da = da.rio.reproject(target_crs)

            # Clip to AOI
            if x_min is not None:
                try:
                    da = da.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
                except Exception:
                    pass

            # Apply Collection 2 Level-2 ST scale factors → Kelvin
            arr = da.values.astype(np.float32)
            arr = np.where(arr > 0, arr * 0.00341802 + 149.0, np.nan)

            if reference_da is None:
                reference_da = da
                composites.append(arr)
            else:
                # Align to reference grid before stacking
                temp_da = xr.DataArray(
                    arr, coords=da.coords, dims=da.dims
                ).rio.write_crs(target_crs)
                aligned = temp_da.rio.reproject_match(reference_da)
                composites.append(aligned.values.astype(np.float32))

            print(f"[landsat] Scene {i+1}/{len(items)} loaded.")
        except Exception as exc:
            print(f"[landsat] Scene {i+1} failed: {exc}")
            continue

    if not composites or reference_da is None:
        print("[landsat] No valid thermal scenes composited.")
        return None

    # Align all arrays to common shape (take minimum shape)
    min_shape = min(a.shape for a in composites)
    trimmed = [a[:min_shape[0], :min_shape[1]] for a in composites]
    stack = np.stack(trimmed, axis=0)
    mean_temp = np.nanmean(stack, axis=0).astype(np.float32)

    # Trim reference coordinates to match
    ref_arr = reference_da.values
    coords_x = reference_da.coords["x"].values[:min_shape[1]]
    coords_y = reference_da.coords["y"].values[:min_shape[0]]

    result = xr.DataArray(
        mean_temp,
        coords={"y": coords_y, "x": coords_x},
        dims=["y", "x"],
        name="lwir11",
    ).rio.write_crs(target_crs)

    valid_count = int(np.sum(np.isfinite(mean_temp)))
    print(
        f"[landsat] Composite from {len(composites)} scenes. "
        f"Temp range: [{np.nanmin(mean_temp):.1f}, {np.nanmax(mean_temp):.1f}] K. "
        f"{valid_count:,} valid pixels."
    )

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_dataset(name="lwir11").to_netcdf(str(cache_path))
        print(f"[landsat] Thermal composite saved to {cache_path}.")
    except Exception as exc:
        print(f"[landsat] Could not save cache: {exc}")

    return result
