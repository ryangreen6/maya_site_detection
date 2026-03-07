"""sar.py — Compute SAR-derived anomaly layers from Sentinel-1 backscatter.

Computes the VH/VV cross-polarization ratio, per-polarization z-score
anomaly maps, and a cross-polarization anomaly that flags pixels with
structurally anomalous subsurface moisture or vegetation architecture.
"""

from typing import Optional

import numpy as np
import xarray as xr

import config


def _zscore_array(arr: np.ndarray) -> np.ndarray:
    """Compute a spatial z-score, masking NaN values.

    Args:
        arr: Input numpy array (may contain NaN).

    Returns:
        Z-score array of the same shape. Where arr is NaN, output is NaN.
    """
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _like(
    template: xr.DataArray, data: np.ndarray, name: str
) -> xr.DataArray:
    """Wrap a numpy array in a DataArray matching a template's coordinates.

    Args:
        template: DataArray whose spatial metadata will be copied.
        data: Numpy array of the same shape.
        name: Name for the output DataArray.

    Returns:
        xarray DataArray with coordinates and CRS from template.
    """
    da = xr.DataArray(data, coords=template.coords, dims=template.dims, name=name)
    if template.rio.crs is not None:
        da = da.rio.write_crs(template.rio.crs)
    return da


# ---------------------------------------------------------------------------
# VH/VV ratio
# ---------------------------------------------------------------------------

def compute_vh_vv_ratio(
    vv: xr.DataArray,
    vh: xr.DataArray,
) -> Optional[xr.DataArray]:
    """Compute the VH/VV backscatter ratio in dB space.

    The cross-polarization ratio is sensitive to vegetation structure and
    soil moisture. Anomalously high VH/VV values can indicate dense canopy
    or disturbed subsurface conditions consistent with buried masonry.

    Args:
        vv: VV polarization DataArray (values expected in dB).
        vh: VH polarization DataArray (values expected in dB).

    Returns:
        VH/VV ratio DataArray in dB (VH − VV), or None on error.
    """
    if vv is None or vh is None:
        print("[sar] Cannot compute VH/VV ratio: missing band(s).")
        return None
    try:
        # In dB space, ratio = difference
        ratio = (vh.astype(np.float32) - vv.astype(np.float32)).rename(
            "vh_vv_ratio"
        )
        if vv.rio.crs is not None:
            ratio = ratio.rio.write_crs(vv.rio.crs)
        print("[sar] VH/VV ratio computed.")
        return ratio
    except Exception as exc:
        print(f"[sar] VH/VV ratio computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# SAR anomaly maps
# ---------------------------------------------------------------------------

def compute_sar_anomaly(
    vv: Optional[xr.DataArray],
    vh: Optional[xr.DataArray],
) -> dict[str, Optional[xr.DataArray]]:
    """Compute spatial z-score anomaly maps for VV and VH polarizations.

    Pixels with anomalously low or high backscatter relative to the scene
    mean indicate unusual surface or subsurface conditions. Anomalies
    consistent with archaeological masonry typically appear as slightly
    elevated VV returns from hard reflectors (stone) or as depressed VH
    from areas with reduced double-bounce scattering (cleared plazas).

    Args:
        vv: VV polarization composite DataArray (dB).
        vh: VH polarization composite DataArray (dB).

    Returns:
        Dictionary with keys 'vv_anomaly' and 'vh_anomaly', each a
        z-score DataArray or None if the input was None.
    """
    result: dict[str, Optional[xr.DataArray]] = {
        "vv_anomaly": None,
        "vh_anomaly": None,
    }

    for pol_name, da in [("vv_anomaly", vv), ("vh_anomaly", vh)]:
        if da is None:
            print(f"[sar] Skipping {pol_name}: input is None.")
            continue
        try:
            arr = da.values.astype(np.float32)
            z = _zscore_array(arr)
            result[pol_name] = _like(da, z, name=pol_name)
            print(f"[sar] {pol_name} computed.")
        except Exception as exc:
            print(f"[sar] {pol_name} computation failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Cross-polarization anomaly
# ---------------------------------------------------------------------------

def compute_cross_pol_anomaly(
    vh_vv_ratio: xr.DataArray,
    z_threshold: float = 2.0,
) -> Optional[xr.DataArray]:
    """Flag pixels where VH/VV ratio deviates significantly from the spatial mean.

    A high absolute z-score of the VH/VV ratio indicates that the local
    polarimetric scattering mechanism differs from the surrounding terrain.
    This can reflect anomalous subsurface moisture retention (soil over
    stone) or unusual vegetation structure over buried architecture.

    Args:
        vh_vv_ratio: VH/VV ratio DataArray (dB difference).
        z_threshold: Absolute z-score threshold for flagging anomalous pixels.

    Returns:
        Cross-polarization anomaly DataArray (z-score of the ratio),
        or None on error.
    """
    if vh_vv_ratio is None:
        print("[sar] Cannot compute cross-pol anomaly: VH/VV ratio is None.")
        return None
    try:
        arr = vh_vv_ratio.values.astype(np.float32)
        z = _zscore_array(arr)
        anomaly = _like(vh_vv_ratio, z, name="cross_pol_anomaly")
        n_flagged = int(np.sum(np.abs(z) > z_threshold))
        print(
            f"[sar] Cross-pol anomaly computed. "
            f"{n_flagged} pixels exceed |z| > {z_threshold}."
        )
        return anomaly
    except Exception as exc:
        print(f"[sar] Cross-pol anomaly computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Combined SAR anomaly score
# ---------------------------------------------------------------------------

def compute_combined_sar_anomaly(
    vv_anomaly: Optional[xr.DataArray],
    vh_anomaly: Optional[xr.DataArray],
    cross_pol_anomaly: Optional[xr.DataArray],
) -> Optional[xr.DataArray]:
    """Fuse VV, VH, and cross-polarization anomalies into a single SAR score.

    Takes the mean of all available anomaly layers. Pixels that are
    anomalous across multiple SAR signals receive the highest scores.

    Args:
        vv_anomaly: VV polarization z-score DataArray or None.
        vh_anomaly: VH polarization z-score DataArray or None.
        cross_pol_anomaly: Cross-polarization ratio z-score DataArray or None.

    Returns:
        Combined SAR anomaly DataArray (mean z-score), or None if no inputs.
    """
    layers = [l for l in [vv_anomaly, vh_anomaly, cross_pol_anomaly] if l is not None]
    if not layers:
        print("[sar] No SAR anomaly layers available for combination.")
        return None

    try:
        reference = layers[0]
        aligned = [reference]
        for da in layers[1:]:
            try:
                aligned.append(da.rio.reproject_match(reference))
            except Exception:
                aligned.append(da)

        stacked = xr.concat(aligned, dim="layer")
        combined = stacked.mean(dim="layer", skipna=True).rename("sar_anomaly")
        if reference.rio.crs is not None:
            combined = combined.rio.write_crs(reference.rio.crs)
        print("[sar] Combined SAR anomaly computed.")
        return combined
    except Exception as exc:
        print(f"[sar] Combined SAR anomaly failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_sar_layers(
    s1_bands: Optional[dict[str, xr.DataArray]],
) -> dict[str, Optional[xr.DataArray]]:
    """Compute all SAR-derived layers from a Sentinel-1 band dictionary.

    Args:
        s1_bands: Dictionary with keys 'VV' and 'VH' as DataArrays.
                  If None or keys are missing, affected layers will be None.

    Returns:
        Dictionary with keys: 'vh_vv_ratio', 'vv_anomaly', 'vh_anomaly',
        'cross_pol_anomaly', 'sar_anomaly'. Values are DataArrays or None.
    """
    if s1_bands is None:
        print("[sar] No SAR band data; all SAR layers will be None.")
        return {
            "vh_vv_ratio": None,
            "vv_anomaly": None,
            "vh_anomaly": None,
            "cross_pol_anomaly": None,
            "sar_anomaly": None,
        }

    vv = s1_bands.get("VV")
    vh = s1_bands.get("VH")

    vh_vv_ratio = compute_vh_vv_ratio(vv, vh)
    anomalies = compute_sar_anomaly(vv, vh)
    cross_pol = compute_cross_pol_anomaly(vh_vv_ratio)
    combined = compute_combined_sar_anomaly(
        anomalies["vv_anomaly"], anomalies["vh_anomaly"], cross_pol
    )

    return {
        "vh_vv_ratio": vh_vv_ratio,
        "vv_anomaly": anomalies["vv_anomaly"],
        "vh_anomaly": anomalies["vh_anomaly"],
        "cross_pol_anomaly": cross_pol,
        "sar_anomaly": combined,
    }
