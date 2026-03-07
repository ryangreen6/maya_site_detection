"""gedi_terrain.py — Interpolate GEDI ground shots onto a regular grid.

GEDI L2A shots are sparse along-track measurements (~60 m spacing along each
beam, ~600 m between adjacent beams).  This module interpolates those shots
onto a regular raster grid matching the DEM extent and resolution, then
computes a Local Relief Model (LRM) from the interpolated ground surface.

Because GEDI measures actual ground elevation through the canopy, the
resulting LRM reveals platform-scale micro-topography (~1–5 m) that SRTM
and Copernicus DEMs cannot detect under dense jungle cover.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from scipy.ndimage import gaussian_filter

import config


def load_gedi_shots(
    shots_path: Path = config.GEDI_SHOTS_PATH,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load GEDI ground shots CSV into numpy arrays.

    Args:
        shots_path: Path to CSV with columns (latitude, longitude, elev_ground).

    Returns:
        Tuple (lats, lons, elevs) as float64 arrays, or None on failure.
    """
    if not shots_path.exists():
        print(f"[gedi_terrain] Shots file not found: {shots_path}")
        return None
    try:
        import csv
        lats, lons, elevs = [], [], []
        with open(shots_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                elevs.append(float(row["elev_ground"]))
        print(f"[gedi_terrain] Loaded {len(lats):,} GEDI shots from {shots_path}.")
        return (
            np.array(lats, dtype=np.float64),
            np.array(lons, dtype=np.float64),
            np.array(elevs, dtype=np.float64),
        )
    except Exception as exc:
        print(f"[gedi_terrain] Failed to load shots: {exc}")
        return None


def interpolate_gedi_to_grid(
    shots_path: Path = config.GEDI_SHOTS_PATH,
    reference_da: Optional[xr.DataArray] = None,
    target_crs: str = config.CRS,
    raster_path: Path = config.GEDI_RASTER_PATH,
    force_recompute: bool = False,
    max_gap_m: float = config.GEDI_MAX_GAP_M,
) -> Optional[xr.DataArray]:
    """Interpolate sparse GEDI shots onto the reference raster grid.

    Projects WGS84 GEDI shot coordinates to *target_crs*, then uses
    linear triangulation (scipy ``griddata``) to interpolate ground elevation
    onto the reference grid.  Pixels farther than *max_gap_m* from any shot
    are masked to NaN so that large data-gap regions do not receive unreliable
    extrapolated values.

    Args:
        shots_path: CSV file produced by download_gedi.get_gedi_shots().
        reference_da: DataArray whose grid (extent, resolution, CRS) defines
                      the output raster.  If None, tries to load the cached
                      raster directly.
        target_crs: Project CRS (must match reference_da).
        raster_path: Path to save / load the interpolated GeoTIFF.
        force_recompute: Ignore cache and recompute.
        max_gap_m: Pixels with no GEDI shot within this distance (metres) are
                   set to NaN.

    Returns:
        Interpolated ground elevation DataArray (float32) or None.
    """
    if raster_path.exists() and not force_recompute:
        print(f"[gedi_terrain] Using cached GEDI raster: {raster_path}")
        da = rioxarray.open_rasterio(str(raster_path), masked=True).squeeze("band", drop=True)
        da = da.rename("gedi_ground_elev")
        if da.rio.crs is None:
            da = da.rio.write_crs(target_crs)
        print(f"[gedi_terrain] Loaded: shape={da.values.shape}")
        return da

    if reference_da is None:
        print("[gedi_terrain] No reference DataArray provided and no cached raster found.")
        return None

    shot_data = load_gedi_shots(shots_path)
    if shot_data is None:
        return None
    lats, lons, elevs = shot_data

    # Project shot coordinates to target CRS
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        shot_x, shot_y = transformer.transform(lons, lats)
    except Exception as exc:
        print(f"[gedi_terrain] Coordinate projection failed: {exc}")
        return None

    # Build output grid from reference DataArray
    grid_x = reference_da.coords["x"].values.astype(np.float64)
    grid_y = reference_da.coords["y"].values.astype(np.float64)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    print(
        f"[gedi_terrain] Interpolating {len(shot_x):,} shots onto "
        f"{grid_xx.shape[0]}×{grid_xx.shape[1]} grid ..."
    )

    try:
        from scipy.interpolate import griddata
        points = np.column_stack([shot_x, shot_y])
        grid_elev = griddata(
            points,
            elevs,
            (grid_xx, grid_yy),
            method="linear",
        ).astype(np.float32)
    except Exception as exc:
        print(f"[gedi_terrain] Interpolation failed: {exc}")
        return None

    # Mask pixels too far from any shot
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([shot_x, shot_y]))
        flat_pts = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
        dist, _ = tree.query(flat_pts, k=1, workers=-1)
        dist_grid = dist.reshape(grid_xx.shape).astype(np.float32)
        gap_mask = dist_grid > max_gap_m
        grid_elev[gap_mask] = np.nan
        coverage_pct = 100.0 * float(np.sum(~gap_mask)) / gap_mask.size
        print(
            f"[gedi_terrain] Coverage within {max_gap_m:.0f} m gap mask: "
            f"{coverage_pct:.1f}% of grid pixels."
        )
    except Exception as exc:
        print(f"[gedi_terrain] Gap masking failed ({exc}); keeping all interpolated values.")

    result = xr.DataArray(
        grid_elev,
        coords={"y": grid_y, "x": grid_x},
        dims=["y", "x"],
        name="gedi_ground_elev",
    ).rio.write_crs(target_crs)

    # Save GeoTIFF cache
    try:
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        result.rio.to_raster(str(raster_path))
        print(f"[gedi_terrain] GEDI raster saved to {raster_path}.")
    except Exception as exc:
        print(f"[gedi_terrain] Could not save raster: {exc}")

    return result


def compute_gedi_lrm(
    gedi_elev: xr.DataArray,
    gaussian_sigma: float = config.LRM_GAUSSIAN_SIGMA,
) -> Optional[xr.DataArray]:
    """Compute a Local Relief Model from GEDI ground elevation.

    Subtracts a Gaussian low-pass trend surface from the GEDI ground
    elevation to isolate micro-topographic features (Maya platforms,
    mounds, causeways) at the 1–15 m scale.

    Args:
        gedi_elev: Interpolated GEDI ground elevation DataArray.
        gaussian_sigma: Gaussian sigma in pixels for the trend surface.

    Returns:
        GEDI LRM DataArray (float32), or None on failure.
    """
    if gedi_elev is None:
        return None
    try:
        arr = gedi_elev.values.astype(np.float32)
        nan_mask = ~np.isfinite(arr)

        fill_val = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
        arr_filled = np.where(nan_mask, fill_val, arr)

        trend = gaussian_filter(arr_filled, sigma=gaussian_sigma).astype(np.float32)
        lrm = arr_filled - trend
        lrm[nan_mask] = np.nan

        result = xr.DataArray(
            lrm,
            coords=gedi_elev.coords,
            dims=gedi_elev.dims,
            name="gedi_relief",
        )
        crs = gedi_elev.rio.crs
        if crs is not None:
            result = result.rio.write_crs(crs)
        else:
            result = result.rio.write_crs(config.CRS)

        n_valid = int(np.sum(np.isfinite(lrm)))
        print(
            f"[gedi_terrain] GEDI LRM computed (sigma={gaussian_sigma}). "
            f"Range: [{float(np.nanmin(lrm)):.2f}, {float(np.nanmax(lrm)):.2f}] m. "
            f"{n_valid:,} valid pixels."
        )
        return result

    except Exception as exc:
        print(f"[gedi_terrain] GEDI LRM computation failed: {exc}")
        return None
