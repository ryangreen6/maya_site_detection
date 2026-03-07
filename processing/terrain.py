"""terrain.py — Compute topographic derivatives from a DEM.

All computations are performed on numpy arrays derived from xarray DataArrays.
CRS and spatial metadata are preserved by copying attributes and coordinates
from the input DataArray. Functions return xarray DataArrays.
"""

from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pixel_size(dem: xr.DataArray) -> tuple[float, float]:
    """Extract pixel size in projected units from a DataArray's transform.

    Args:
        dem: Input DEM DataArray with spatial coordinates.

    Returns:
        Tuple (dx, dy) representing pixel width and height in map units.
    """
    x = dem.coords.get("x", dem.coords.get("lon", None))
    y = dem.coords.get("y", dem.coords.get("lat", None))
    if x is not None and len(x) > 1:
        dx = float(abs(x[1] - x[0]))
    else:
        dx = 30.0  # Default SRTM 30m

    if y is not None and len(y) > 1:
        dy = float(abs(y[1] - y[0]))
    else:
        dy = 30.0

    return dx, dy


def _like(dem: xr.DataArray, data: np.ndarray, name: str) -> xr.DataArray:
    """Wrap a numpy array in a DataArray with the same coords/dims as dem.

    Args:
        dem: Template DataArray for coordinates and dimensions.
        data: Numpy array of the same shape as dem.
        name: Name to assign to the resulting DataArray.

    Returns:
        xarray DataArray with spatial metadata copied from dem.
    """
    da = xr.DataArray(data, coords=dem.coords, dims=dem.dims, name=name)
    if dem.rio.crs is not None:
        da = da.rio.write_crs(dem.rio.crs)
    return da


# ---------------------------------------------------------------------------
# Hillshade
# ---------------------------------------------------------------------------

def compute_hillshade_single(
    dem: xr.DataArray,
    azimuth: float,
    altitude: float = float(config.HILLSHADE_ALTITUDE),
) -> np.ndarray:
    """Compute hillshade for a single sun azimuth and altitude angle.

    Uses the standard ESRI hillshade formula based on slope and aspect
    derived via central-difference gradient.

    Args:
        dem: DEM DataArray in projected coordinates.
        azimuth: Sun azimuth in degrees clockwise from north (0–360).
        altitude: Sun elevation angle above horizon in degrees (0–90).

    Returns:
        Hillshade array (0–255 float) of the same shape as dem.
    """
    elev = dem.values.astype(np.float64)
    dx, dy = _get_pixel_size(dem)

    # Compute gradients using central differences
    dzdx = np.gradient(elev, dx, axis=1)
    dzdy = np.gradient(elev, dy, axis=0)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(-dzdy, dzdx)

    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)

    hillshade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = np.clip(hillshade, 0.0, 1.0) * 255.0
    return hillshade


def compute_multidirectional_hillshade(
    dem: xr.DataArray,
    azimuths: list[int] = config.HILLSHADE_AZIMUTHS,
    altitude: int = config.HILLSHADE_ALTITUDE,
) -> Optional[xr.DataArray]:
    """Compute a multi-directional hillshade by averaging across azimuths.

    Illuminates the terrain from all cardinal and intercardinal directions at
    a low sun angle to maximally reveal subtle topographic anomalies consistent
    with buried archaeological structures.

    Args:
        dem: Input DEM as an xarray DataArray.
        azimuths: List of sun azimuth angles in degrees (clockwise from north).
        altitude: Sun elevation angle in degrees above the horizon.

    Returns:
        Multi-directional hillshade DataArray (0–255), or None on error.
    """
    if dem is None:
        print("[terrain] DEM is None; cannot compute hillshade.")
        return None
    try:
        stacked = np.stack(
            [compute_hillshade_single(dem, az, altitude) for az in azimuths],
            axis=0,
        )
        md_hillshade = stacked.mean(axis=0)
        result = _like(dem, md_hillshade, name="hillshade")
        print("[terrain] Multi-directional hillshade computed.")
        return result
    except Exception as exc:
        print(f"[terrain] Hillshade computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

def compute_slope(dem: xr.DataArray) -> Optional[xr.DataArray]:
    """Compute terrain slope in degrees using central-difference gradients.

    Args:
        dem: Input DEM DataArray in projected coordinates.

    Returns:
        Slope DataArray in degrees, or None on error.
    """
    if dem is None:
        return None
    try:
        elev = dem.values.astype(np.float64)
        dx, dy = _get_pixel_size(dem)
        dzdx = np.gradient(elev, dx, axis=1)
        dzdy = np.gradient(elev, dy, axis=0)
        slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        result = _like(dem, slope_deg, name="slope")
        print("[terrain] Slope computed.")
        return result
    except Exception as exc:
        print(f"[terrain] Slope computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Topographic Position Index (TPI)
# ---------------------------------------------------------------------------

def compute_tpi(
    dem: xr.DataArray,
    radius: int,
) -> Optional[xr.DataArray]:
    """Compute Topographic Position Index at a given neighborhood radius.

    TPI = elevation − mean(elevation in circular neighborhood of radius r).
    Positive TPI indicates ridges / raised features; negative indicates
    valleys or depressions — both relevant to Maya platform detection.

    Args:
        dem: Input DEM DataArray.
        radius: Neighborhood radius in pixels for the mean filter.

    Returns:
        TPI DataArray (same units as DEM elevation), or None on error.
    """
    if dem is None:
        return None
    try:
        elev = dem.values.astype(np.float64)
        nan_mask = ~np.isfinite(elev)
        # uniform_filter does not handle NaN; fill with local mean before filtering
        fill_value = np.nanmean(elev) if np.any(np.isfinite(elev)) else 0.0
        filled = np.where(nan_mask, fill_value, elev)
        kernel_size = 2 * radius + 1
        local_mean = uniform_filter(filled, size=kernel_size, mode="reflect")
        tpi = elev - local_mean
        tpi[nan_mask] = np.nan
        name = f"tpi_r{radius}"
        result = _like(dem, tpi, name=name)
        print(f"[terrain] TPI (radius={radius}) computed.")
        return result
    except Exception as exc:
        print(f"[terrain] TPI (radius={radius}) failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Local Relief Model (LRM)
# ---------------------------------------------------------------------------

def compute_lrm(
    dem: xr.DataArray,
    sigma: float = config.LRM_GAUSSIAN_SIGMA,
) -> Optional[xr.DataArray]:
    """Compute the Local Relief Model by removing regional topographic trend.

    Subtracts a Gaussian-smoothed version of the DEM from the original.
    The result isolates micro-topographic features — such as Maya platforms,
    plazas, and causeways — from the underlying regional terrain.

    Args:
        dem: Input DEM DataArray.
        sigma: Standard deviation in pixels for the Gaussian low-pass filter.

    Returns:
        LRM DataArray (in DEM elevation units), or None on error.
    """
    if dem is None:
        return None
    try:
        elev = dem.values.astype(np.float64)
        # Fill NaNs with local mean for the smoothing pass
        nan_mask = np.isnan(elev)
        filled = np.where(nan_mask, np.nanmean(elev), elev)
        smoothed = gaussian_filter(filled, sigma=sigma, mode="reflect")
        lrm = elev - smoothed
        lrm[nan_mask] = np.nan
        result = _like(dem, lrm, name="lrm")
        print(f"[terrain] LRM (sigma={sigma}) computed.")
        return result
    except Exception as exc:
        print(f"[terrain] LRM computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Terrain Ruggedness Index (TRI)
# ---------------------------------------------------------------------------

def compute_tri(dem: xr.DataArray) -> Optional[xr.DataArray]:
    """Compute the Terrain Ruggedness Index (Riley et al. 1999).

    TRI = mean absolute difference between each pixel and its 8 neighbours.
    Archaeological sites often exhibit anomalously low ruggedness (flat
    plazas) or high ruggedness (pyramidal structures) relative to the
    surrounding jungle terrain.

    Args:
        dem: Input DEM DataArray.

    Returns:
        TRI DataArray (in DEM elevation units), or None on error.
    """
    if dem is None:
        return None
    try:
        elev = dem.values.astype(np.float64)

        # Vectorized 8-neighbor TRI: shift the array in each compass direction
        # and accumulate the absolute difference from centre — orders of magnitude
        # faster than generic_filter for large arrays.
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
        abs_diff_sum = np.zeros_like(elev)
        valid_count = np.zeros_like(elev)
        for dr, dc in shifts:
            neighbour = np.roll(np.roll(elev, dr, axis=0), dc, axis=1)
            diff = np.abs(elev - neighbour)
            abs_diff_sum += np.where(np.isfinite(diff), diff, 0.0)
            valid_count += np.isfinite(diff).astype(np.float64)

        tri_arr = np.where(valid_count > 0, abs_diff_sum / valid_count, np.nan)
        result = _like(dem, tri_arr, name="tri")
        print("[terrain] TRI computed.")
        return result
    except Exception as exc:
        print(f"[terrain] TRI computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_terrain_derivatives(
    dem: xr.DataArray,
    tpi_small_radius: int = config.TPI_SMALL_RADIUS,
    tpi_large_radius: int = config.TPI_LARGE_RADIUS,
    lrm_sigma: float = config.LRM_GAUSSIAN_SIGMA,
    hillshade_azimuths: list[int] = config.HILLSHADE_AZIMUTHS,
    hillshade_altitude: int = config.HILLSHADE_ALTITUDE,
) -> dict[str, Optional[xr.DataArray]]:
    """Compute all terrain derivatives and return them in a dictionary.

    Orchestrates computation of hillshade, slope, TPI at two scales, LRM,
    and TRI from a single DEM input.

    Args:
        dem: Input DEM DataArray in projected coordinates.
        tpi_small_radius: Pixel radius for small-scale TPI.
        tpi_large_radius: Pixel radius for large-scale TPI.
        lrm_sigma: Gaussian sigma for LRM computation.
        hillshade_azimuths: Sun azimuth list for multi-directional hillshade.
        hillshade_altitude: Sun elevation angle in degrees.

    Returns:
        Dictionary with keys: 'hillshade', 'slope', 'tpi_small', 'tpi_large',
        'lrm', 'tri'. Values are DataArrays or None if computation failed.
    """
    return {
        "hillshade": compute_multidirectional_hillshade(
            dem, azimuths=hillshade_azimuths, altitude=hillshade_altitude
        ),
        "slope": compute_slope(dem),
        "tpi_small": compute_tpi(dem, radius=tpi_small_radius),
        "tpi_large": compute_tpi(dem, radius=tpi_large_radius),
        "lrm": compute_lrm(dem, sigma=lrm_sigma),
        "tri": compute_tri(dem),
    }
