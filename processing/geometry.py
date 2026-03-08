"""geometry.py — Compute east-facing horizon sightline score from the DEM.

Maya E-Group ceremonial complexes orient a tall western pyramid toward the
eastern horizon so that observers can track solstice and equinox sunrises
from a fixed vantage point. This module scores each DEM pixel for three
simultaneously-required conditions:

  1. Elevated above its surroundings (positive large-scale TPI).
  2. East-facing aspect (terrain slope faces toward sunrise azimuth).
  3. Open eastern horizon (no ridge blocks the view within horizon_km).

The aspect score uses cos²(azimuth − 90°), which peaks at due east (90°)
but smoothly rewards azimuths up to ~30° north or south — covering the
full solstice sunrise range at Guatemala's latitude (~17°N ≈ ±26° of east).
"""

from typing import Optional

import numpy as np
import xarray as xr

import config


def compute_east_sightline(
    dem: xr.DataArray,
    tpi: Optional[xr.DataArray] = None,
    horizon_km: float = config.EAST_SIGHTLINE_HORIZON_KM,
) -> Optional[xr.DataArray]:
    """Score pixels for east-facing elevation with an open eastern horizon.

    Maya E-Group ceremonial complexes place a tall western pyramid with a
    clear sightline to the eastern horizon, where three low eastern platforms
    mark solstice and equinox sunrise. This layer rewards pixels that are:

      1. Elevated above their surroundings (positive TPI).
      2. Facing east (terrain aspect close to 90° from north).
      3. Have an unobstructed eastern horizon (low maximum elevation angle
         looking east within *horizon_km* kilometres).

    The three sub-scores are multiplied so that all three conditions must be
    met simultaneously; a west-facing mound or one blocked by a ridge to its
    east scores near zero.

    Args:
        dem: DEM DataArray in a projected metric CRS (e.g. UTM metres).
        tpi: Large-scale TPI DataArray on the same grid (optional).
             If None, the elevation score component is set to 1 everywhere.
        horizon_km: Search distance in km for the eastward horizon angle.

    Returns:
        East-sightline score DataArray (float32, values in [0, 1]), or None.
    """
    if dem is None:
        return None
    try:
        elev = dem.values.astype(np.float64)
        nan_mask = ~np.isfinite(elev)

        # Pixel spacing in metres from UTM coordinate arrays
        y_vals = dem.coords["y"].values
        x_vals = dem.coords["x"].values
        pixel_size_y = abs(float(np.mean(np.diff(y_vals))))
        pixel_size_x = abs(float(np.mean(np.diff(x_vals))))

        # --- 1. Aspect score (east-facing) --------------------------------
        # np.gradient axis-0 = rows (increases southward in GIS rasters),
        # axis-1 = cols (increases eastward).  Convert to geographic
        # northward/eastward slope components before computing bearing.
        dy_row, dx_col = np.gradient(elev)
        dy_geo = -dy_row / pixel_size_y   # slope per metre northward
        dx_geo = dx_col / pixel_size_x    # slope per metre eastward

        # Geographic aspect: 0° = north, 90° = east, 180° = south, 270° = west
        aspect_deg = np.degrees(np.arctan2(dx_geo, dy_geo)) % 360.0
        east_diff_rad = np.radians(aspect_deg - 90.0)
        # cos²(diff) gives max=1 at due east, covers ±26° solstice range at 0.8+
        cos_diff = np.cos(east_diff_rad)
        aspect_score = np.where(cos_diff > 0, cos_diff ** 2, 0.0)

        # --- 2. Eastward horizon angle ------------------------------------
        # For each pixel, find the maximum elevation angle of any pixel to
        # its east within horizon_km.  Low angle = open horizon = high score.
        k_max = max(1, int(horizon_km * 1000.0 / pixel_size_x))
        fill = np.where(nan_mask, np.nanmean(elev) if np.any(np.isfinite(elev)) else 0.0, elev)
        horizon_angle = np.full_like(fill, -90.0)

        for k in range(1, k_max + 1):
            east_elev = np.empty_like(fill)
            east_elev[:, :-k] = fill[:, k:]
            east_elev[:, -k:] = np.nan
            angle = np.degrees(np.arctan2(east_elev - fill, k * pixel_size_x))
            horizon_angle = np.where(np.isfinite(angle),
                                     np.maximum(horizon_angle, angle),
                                     horizon_angle)

        # Map 0° (flat/open) → 1.0, 45° (blocked) → 0.0; ignore negative angles
        horizon_clipped = np.clip(horizon_angle, 0.0, 45.0)
        horizon_score = 1.0 - horizon_clipped / 45.0

        # --- 3. Elevation score (elevated above surroundings) -------------
        if tpi is not None:
            tpi_vals = tpi.values.astype(np.float64)
            tpi_pos = np.clip(tpi_vals, 0.0, None)
            tpi_max = np.nanmax(tpi_pos)
            tpi_score = tpi_pos / tpi_max if tpi_max > 0 else np.zeros_like(elev)
        else:
            tpi_score = np.ones_like(elev)

        # --- Combine (all three must be satisfied) ------------------------
        east_sl = (aspect_score * horizon_score * tpi_score).astype(np.float32)
        east_sl[nan_mask] = np.nan

        result = xr.DataArray(
            east_sl, coords=dem.coords, dims=dem.dims, name="east_sightline"
        )
        if dem.rio.crs is not None:
            result = result.rio.write_crs(dem.rio.crs)

        n_high = int(np.sum(east_sl > 0.5))
        print(f"[geometry] East sightline computed (horizon={horizon_km} km). "
              f"{n_high:,} pixels score > 0.5.")
        return result

    except Exception as exc:
        print(f"[geometry] East sightline computation failed: {exc}")
        return None
