"""thermal.py — Compute thermal inertia anomaly from Landsat surface temperature.

Stone and fired-clay masonry have thermal inertia ~3–5× higher than moist
soil.  This means buried or exposed Maya structures:
  - Heat up more slowly during the day (appear cool in daytime thermal imagery).
  - Retain heat longer at night (appear warm in pre-dawn imagery).

From a single-composite mean temperature layer we compute a *local* thermal
anomaly — the deviation of each pixel from its neighbourhood mean — which
highlights pixels with unusual thermal behaviour independent of regional
temperature gradients.  High positive anomaly = warmer than surroundings
(nighttime signature of high-inertia stone); we output absolute deviation
so both warm and cool anomalies are preserved as a high score.
"""

from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

import config


def compute_thermal_anomaly(
    thermal_da: xr.DataArray,
    window_px: int = 31,
) -> Optional[xr.DataArray]:
    """Compute local thermal anomaly as a z-score relative to a moving window.

    For each pixel, subtracts the neighbourhood mean (uniform filter over
    *window_px* × *window_px* pixels) and divides by the neighbourhood
    standard deviation.  The result highlights locally anomalous thermal
    pixels regardless of regional temperature trends.

    Args:
        thermal_da: Surface temperature DataArray (Kelvin or Celsius).
        window_px: Side length of the moving-window neighbourhood in pixels.
                   At ~30 m Landsat resolution, 31 px ≈ 930 m window.

    Returns:
        Local thermal z-score DataArray (float32), or None on failure.
    """
    if thermal_da is None:
        return None

    try:
        arr = thermal_da.values.astype(np.float32)
        nan_mask = ~np.isfinite(arr)

        # Fill NaN with local mean for filter computation
        fill_val = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
        arr_filled = np.where(nan_mask, fill_val, arr)

        # Local mean
        local_mean = uniform_filter(arr_filled, size=window_px, mode="reflect").astype(np.float32)

        # Local std via E[X²] - E[X]²
        local_mean_sq = uniform_filter(arr_filled ** 2, size=window_px, mode="reflect")
        local_var = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
        local_std = np.sqrt(local_var).astype(np.float32)

        # Z-score
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(local_std > 0, (arr - local_mean) / local_std, 0.0)

        z = z.astype(np.float32)
        z[nan_mask] = np.nan

        result = xr.DataArray(
            z,
            coords=thermal_da.coords,
            dims=thermal_da.dims,
            name="thermal_anomaly",
        )
        if thermal_da.rio.crs is not None:
            result = result.rio.write_crs(thermal_da.rio.crs)
        elif thermal_da.rio.crs is None:
            result = result.rio.write_crs(config.CRS)

        n_valid = int(np.sum(np.isfinite(z)))
        print(
            f"[thermal] Anomaly computed (window={window_px}px). "
            f"Z-score range: [{float(np.nanmin(z)):.2f}, {float(np.nanmax(z)):.2f}]. "
            f"{n_valid:,} valid pixels."
        )
        return result

    except Exception as exc:
        print(f"[thermal] Thermal anomaly computation failed: {exc}")
        return None
