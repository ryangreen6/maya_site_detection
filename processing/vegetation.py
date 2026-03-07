"""vegetation.py — Compute vegetation indices and anomaly maps from Sentinel-2.

Computes NDVI, NDRE, per-pixel z-score anomaly maps, and a persistent
anomaly layer that distinguishes chronic structural canopy stress (consistent
with buried archaeology) from seasonal drought or cloud artefacts.
"""

from typing import Optional

import numpy as np
import xarray as xr

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_ratio(
    numerator: xr.DataArray,
    denominator: xr.DataArray,
    name: str,
) -> Optional[xr.DataArray]:
    """Compute a band ratio, masking division-by-zero as NaN.

    Args:
        numerator: Numerator DataArray.
        denominator: Denominator DataArray.
        name: Name for the resulting DataArray.

    Returns:
        Ratio DataArray with NaN where denominator is zero, or None on error.
    """
    try:
        num = numerator.astype(np.float32)
        den = denominator.astype(np.float32)
        ratio = num / den.where(den != 0)
        return ratio.rename(name)
    except Exception as exc:
        print(f"[vegetation] Failed to compute ratio '{name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def compute_ndvi(
    b08: xr.DataArray,
    b04: xr.DataArray,
) -> Optional[xr.DataArray]:
    """Compute the Normalized Difference Vegetation Index (NDVI).

    NDVI = (B08 − B04) / (B08 + B04). Values range from −1 to 1; dense
    healthy vegetation is typically above 0.6 in the humid tropics.

    Args:
        b08: NIR band (Sentinel-2 B08) as a DataArray.
        b04: Red band (Sentinel-2 B04) as a DataArray.

    Returns:
        NDVI DataArray clipped to [−1, 1], or None on error.
    """
    if b08 is None or b04 is None:
        print("[vegetation] Cannot compute NDVI: missing band(s).")
        return None
    ndvi = _safe_ratio(b08 - b04, b08 + b04, name="ndvi")
    if ndvi is not None:
        ndvi = ndvi.clip(-1.0, 1.0)
        print("[vegetation] NDVI computed.")
    return ndvi


def compute_ndre(
    b07: xr.DataArray,
    b04: xr.DataArray,
) -> Optional[xr.DataArray]:
    """Compute the Normalized Difference Red Edge Index (NDRE).

    NDRE = (B07 − B04) / (B07 + B04). Red edge indices are more sensitive
    to subtle canopy chlorophyll stress than broadband NDVI, making them
    valuable for detecting archaeology-driven vegetation anomalies.

    Args:
        b07: Red Edge band (Sentinel-2 B07) as a DataArray.
        b04: Red band (Sentinel-2 B04) as a DataArray.

    Returns:
        NDRE DataArray clipped to [−1, 1], or None on error.
    """
    if b07 is None or b04 is None:
        print("[vegetation] Cannot compute NDRE: missing band(s).")
        return None
    # B07 (20 m native) and B04 (10 m native) may have different spatial grids
    # after compositing, even if their shapes match. xarray arithmetic aligns on
    # coordinates and produces all-NaN when grids differ. We reproject B07 onto
    # B04's grid and then compute on numpy arrays to avoid coordinate alignment.
    try:
        # CRS is often lost on NetCDF round-trip; restore from config since all
        # bands are reprojected to the project CRS before caching.
        project_crs = config.CRS
        if b04.rio.crs is None:
            b04 = b04.rio.write_crs(project_crs)
        if b07.rio.crs is None:
            b07 = b07.rio.write_crs(project_crs)
        b07 = b07.rio.reproject_match(b04)
    except Exception as exc:
        print(f"[vegetation] B07 reproject_match failed ({exc}); skipping NDRE.")
        return None

    # Compute on numpy arrays to avoid xarray coordinate-alignment NaN propagation
    b07_vals = b07.values.astype(np.float32)
    b04_vals = b04.values.astype(np.float32)
    denom = b07_vals + b04_vals
    ndre_arr = np.where(denom != 0, (b07_vals - b04_vals) / denom, np.nan).clip(-1.0, 1.0)

    ndre = xr.DataArray(ndre_arr, coords=b04.coords, dims=b04.dims, name="ndre")
    if b04.rio.crs is not None:
        ndre = ndre.rio.write_crs(b04.rio.crs)
    print("[vegetation] NDRE computed.")
    return ndre


# ---------------------------------------------------------------------------
# Single-composite anomaly
# ---------------------------------------------------------------------------

def compute_ndvi_anomaly(
    ndvi: xr.DataArray,
    ndvi_stack: Optional[list[xr.DataArray]] = None,
) -> Optional[xr.DataArray]:
    """Compute an NDVI anomaly map as a z-score relative to a multi-date stack.

    If a *ndvi_stack* of multiple time-period NDVI arrays is provided, the
    z-score is computed from that stack (preferred). Otherwise, a spatial
    z-score is computed across the single scene using the local mean and
    standard deviation.

    Args:
        ndvi: Single-composite NDVI DataArray (primary layer).
        ndvi_stack: Optional list of NDVI DataArrays from different time
                    periods. If provided, anomaly is temporal z-score.

    Returns:
        Anomaly DataArray (z-score, dimensionless), or None on error.
    """
    if ndvi is None:
        print("[vegetation] Cannot compute NDVI anomaly: NDVI is None.")
        return None

    try:
        if ndvi_stack and len(ndvi_stack) > 1:
            # Temporal z-score: align all scenes to the same grid first
            reference = ndvi_stack[0]
            aligned = [reference]
            for da in ndvi_stack[1:]:
                try:
                    aligned.append(da.rio.reproject_match(reference))
                except Exception:
                    aligned.append(da)

            stacked = xr.concat(aligned, dim="time")
            mean = stacked.mean(dim="time", skipna=True)
            std = stacked.std(dim="time", skipna=True)
            std = std.where(std > 0)  # Avoid division by zero
            anomaly = (ndvi - mean) / std
            print("[vegetation] Temporal NDVI anomaly (z-score) computed.")
        else:
            # Spatial z-score across the single image
            vals = ndvi.values.astype(np.float32)
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            if std_val == 0:
                print("[vegetation] NDVI has zero variance; anomaly will be zeros.")
                std_val = 1.0
            anomaly_arr = (vals - mean_val) / std_val
            anomaly = xr.DataArray(
                anomaly_arr, coords=ndvi.coords, dims=ndvi.dims, name="ndvi_anomaly"
            )
            if ndvi.rio.crs is not None:
                anomaly = anomaly.rio.write_crs(ndvi.rio.crs)
            print("[vegetation] Spatial NDVI anomaly (z-score) computed.")

        return anomaly.rename("ndvi_anomaly")
    except Exception as exc:
        print(f"[vegetation] NDVI anomaly computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Persistent anomaly
# ---------------------------------------------------------------------------

def compute_persistent_anomaly(
    ndvi_stack: list[xr.DataArray],
    z_threshold: float = -1.0,
    min_occurrence_fraction: float = 0.5,
) -> Optional[xr.DataArray]:
    """Identify pixels that are persistently anomalously low across time periods.

    A persistent low-NDVI anomaly — one that appears in more than half of all
    available time periods — is unlikely to be caused by a single drought or
    cloud artefact. It is instead consistent with a chronic structural effect
    on vegetation rooting depth, such as a buried stone structure.

    Args:
        ndvi_stack: List of NDVI DataArrays from different time periods.
                    All must cover roughly the same spatial extent.
        z_threshold: Z-score below which a pixel is considered anomalously low.
        min_occurrence_fraction: Fraction of time periods in which a pixel
                                  must be anomalously low to be flagged.

    Returns:
        Persistent anomaly DataArray (values 0–1, proportion of periods with
        anomalous NDVI), or None if the stack is too small or fails.
    """
    if not ndvi_stack or len(ndvi_stack) < 2:
        print("[vegetation] Need at least 2 NDVI scenes for persistent anomaly.")
        return None

    try:
        reference = ndvi_stack[0]
        aligned = [reference]
        for da in ndvi_stack[1:]:
            try:
                aligned.append(da.rio.reproject_match(reference))
            except Exception:
                aligned.append(da)

        stacked = xr.concat(aligned, dim="time")
        mean = stacked.mean(dim="time", skipna=True)
        std = stacked.std(dim="time", skipna=True)
        std = std.where(std > 0)

        # z-score each time slice
        z_stacked = (stacked - mean) / std

        # Count how often each pixel is below the threshold
        low_count = (z_stacked < z_threshold).sum(dim="time").astype(np.float32)
        total_valid = stacked.notnull().sum(dim="time").astype(np.float32)
        persistence = (low_count / total_valid.where(total_valid > 0)).rename(
            "persistent_ndvi_anomaly"
        )

        if reference.rio.crs is not None:
            persistence = persistence.rio.write_crs(reference.rio.crs)

        print(
            f"[vegetation] Persistent anomaly computed across {len(ndvi_stack)} scenes."
        )
        return persistence
    except Exception as exc:
        print(f"[vegetation] Persistent anomaly computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_vegetation_layers(
    bands: Optional[dict[str, xr.DataArray]],
    ndvi_stack: Optional[list[xr.DataArray]] = None,
) -> dict[str, Optional[xr.DataArray]]:
    """Compute all vegetation layers from Sentinel-2 band composites.

    Args:
        bands: Dictionary with keys 'B04', 'B07', 'B08' as DataArrays.
               If None or bands are missing, affected layers will be None.
        ndvi_stack: Optional list of NDVI DataArrays across time periods
                    for temporal anomaly computation.

    Returns:
        Dictionary with keys: 'ndvi', 'ndre', 'ndvi_anomaly',
        'persistent_anomaly'. Values are DataArrays or None.
    """
    if bands is None:
        print("[vegetation] No band data provided; all vegetation layers will be None.")
        return {
            "ndvi": None,
            "ndre": None,
            "ndvi_anomaly": None,
            "persistent_anomaly": None,
        }

    b04 = bands.get("B04")
    b07 = bands.get("B07")
    b08 = bands.get("B08")

    ndvi = compute_ndvi(b08, b04)
    ndre = compute_ndre(b07, b04)
    ndvi_anomaly = compute_ndvi_anomaly(ndvi, ndvi_stack)
    persistent = compute_persistent_anomaly(ndvi_stack) if ndvi_stack else None

    return {
        "ndvi": ndvi,
        "ndre": ndre,
        "ndvi_anomaly": ndvi_anomaly,
        "persistent_anomaly": persistent,
    }
