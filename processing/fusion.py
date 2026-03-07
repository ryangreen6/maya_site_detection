"""fusion.py — Fuse multi-source anomaly layers into a composite site score.

Normalizes each input layer to [0, 1], reprojects all layers to a common
grid, applies a configurable weighted sum, and optionally optimizes the
weights using known site locations. Saves the result as a GeoTIFF.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import geopandas as gpd

import config


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_layer(
    da: xr.DataArray,
    invert: bool = False,
) -> Optional[xr.DataArray]:
    """Normalize a DataArray to the [0, 1] range using min-max scaling.

    NaN values are preserved through normalization. If the input has zero
    range (constant values), returns a zero-filled array.

    Args:
        da: Input DataArray to normalize.
        invert: If True, invert the output so that low values → 1.
                Useful for layers where low values indicate archaeological
                anomaly (e.g., negative NDVI anomaly = vegetation stress).

    Returns:
        Normalized DataArray in [0, 1], or None if input is None.
    """
    if da is None:
        return None
    try:
        arr = da.values.astype(np.float32)
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if vmax == vmin:
            normalized = np.zeros_like(arr)
        else:
            normalized = (arr - vmin) / (vmax - vmin)
        if invert:
            normalized = 1.0 - normalized
        result = xr.DataArray(
            normalized, coords=da.coords, dims=da.dims, name=da.name
        )
        if da.rio.crs is not None:
            result = result.rio.write_crs(da.rio.crs)
        return result
    except Exception as exc:
        print(f"[fusion] Normalization failed for '{da.name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------

def align_layers_to_reference(
    layers: dict[str, Optional[xr.DataArray]],
    reference_key: str,
) -> dict[str, Optional[xr.DataArray]]:
    """Reproject and resample all layers to match a reference layer's grid.

    Uses rioxarray.reproject_match() to align resolution, extent, and CRS.
    Layers that are None are passed through unchanged.

    Args:
        layers: Dictionary of layer name → DataArray (some may be None).
        reference_key: Key in *layers* whose grid all others will match.

    Returns:
        Dictionary with all non-None layers aligned to the reference grid.
    """
    reference = layers.get(reference_key)
    if reference is None:
        # Try to find any non-None layer as reference
        for key, da in layers.items():
            if da is not None:
                reference = da
                reference_key = key
                print(
                    f"[fusion] Reference '{config.__name__}' was None; "
                    f"using '{key}' as reference grid."
                )
                break

    if reference is None:
        print("[fusion] All layers are None; cannot align.")
        return layers

    aligned: dict[str, Optional[xr.DataArray]] = {}
    for name, da in layers.items():
        if da is None:
            aligned[name] = None
            continue
        if name == reference_key:
            aligned[name] = da
            continue
        try:
            # CRS is lost on NetCDF round-trip; restore from config before
            # reproject_match so rioxarray can compute the transform.
            if da.rio.crs is None:
                da = da.rio.write_crs(config.CRS)
            if reference.rio.crs is None:
                reference = reference.rio.write_crs(config.CRS)
            aligned[name] = da.rio.reproject_match(reference)
        except Exception as exc:
            print(f"[fusion] reproject_match failed for '{name}': {exc}. "
                  "Using unaligned layer.")
            aligned[name] = da

    return aligned


# ---------------------------------------------------------------------------
# Weighted fusion
# ---------------------------------------------------------------------------

def weighted_sum(
    layers: dict[str, Optional[xr.DataArray]],
    weights: dict[str, float],
) -> Optional[xr.DataArray]:
    """Compute a weighted sum of normalized anomaly layers.

    Only layers present in both *layers* and *weights* are included.
    Weights are re-normalized to sum to 1 over the available layers.

    Args:
        layers: Dict of layer name → normalized DataArray (values in [0, 1]).
        weights: Dict of layer name → relative weight.

    Returns:
        Composite score DataArray (values in [0, 1]), or None if no
        valid layers are present.
    """
    available = {
        k: da
        for k, da in layers.items()
        if da is not None and k in weights and weights[k] > 0
    }

    if not available:
        print("[fusion] No valid layers with positive weights for fusion.")
        return None

    total_weight = sum(weights[k] for k in available)
    if total_weight == 0:
        print("[fusion] All weights are zero.")
        return None

    reference = next(iter(available.values()))
    ref_shape = reference.values.shape

    # NaN-safe per-pixel weighted sum: at each pixel, sum over only the layers
    # that have valid (non-NaN) data, renormalizing weights accordingly.
    # This allows layers with different spatial extents (DEM vs S2 vs SAR) to
    # each contribute wherever they have coverage.
    weighted_total = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)

    for name, da in available.items():
        w = weights[name] / total_weight
        print(f"[fusion]   Layer '{name}' weight: {w:.4f}")
        arr = da.values.astype(np.float32)
        if arr.shape != ref_shape:
            print(f"[fusion] Shape mismatch for '{name}': {arr.shape} vs {ref_shape}; skipping layer.")
            continue
        valid = np.isfinite(arr)
        weighted_total = np.where(valid, weighted_total + arr * w, weighted_total)
        weight_sum = np.where(valid, weight_sum + w, weight_sum)

    # Normalize by accumulated weights; pixels with no coverage remain NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        composite_arr = np.where(weight_sum > 0, weighted_total / weight_sum, np.nan)

    composite = xr.DataArray(
        composite_arr.astype(np.float32),
        coords=reference.coords,
        dims=reference.dims,
        name="composite_score",
    )
    if reference.rio.crs is not None:
        composite = composite.rio.write_crs(reference.rio.crs)

    print(
        f"[fusion] Composite score range: "
        f"[{np.nanmin(composite_arr):.4f}, {np.nanmax(composite_arr):.4f}]"
    )
    return composite


# ---------------------------------------------------------------------------
# Score extraction at point locations
# ---------------------------------------------------------------------------

def extract_scores_at_points(
    score: xr.DataArray,
    sites_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Sample the composite score at known site point locations.

    Args:
        score: Composite score DataArray (values in [0, 1]).
        sites_gdf: GeoDataFrame of site points in the same CRS as score.

    Returns:
        1D numpy array of sampled score values (NaN where outside extent).
    """
    scores: list[float] = []
    arr = score.values.astype(np.float32)
    x_coords = score.coords["x"].values
    y_coords = score.coords["y"].values

    for geom in sites_gdf.geometry:
        px, py = geom.x, geom.y
        # Find nearest pixel
        xi = int(np.argmin(np.abs(x_coords - px)))
        yi = int(np.argmin(np.abs(y_coords - py)))
        if 0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]:
            scores.append(float(arr[yi, xi]))
        else:
            scores.append(np.nan)

    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def optimize_weights(
    layers: dict[str, Optional[xr.DataArray]],
    sites_gdf: gpd.GeoDataFrame,
    initial_weights: dict[str, float] = config.FUSION_WEIGHTS,
    n_background_samples: int = config.N_RANDOM_NEGATIVES,
    random_seed: int = config.RANDOM_SEED,
) -> dict[str, float]:
    """Optimize fusion weights to maximize site scores relative to background.

    Uses scipy.optimize.minimize (L-BFGS-B) with a simplex weight constraint
    to find weights that maximize the mean score at known sites minus the
    mean score at random background pixels.

    Args:
        layers: Dict of normalized DataArrays for each layer key.
        sites_gdf: GeoDataFrame of known site locations.
        initial_weights: Starting weight dictionary.
        n_background_samples: Number of random background points to sample.
        random_seed: Random seed for background point sampling.

    Returns:
        Optimized weight dictionary with the same keys as initial_weights.
        Falls back to initial_weights if optimization fails.
    """
    from scipy.optimize import minimize

    # Filter to layers that are available and have a weight entry
    valid_keys = [
        k for k in initial_weights if layers.get(k) is not None
    ]

    if len(valid_keys) < 2:
        print("[fusion] Not enough layers for optimization; using config weights.")
        return initial_weights

    # Align all layers to a common reference grid before optimizing so that
    # weighted_sum receives arrays of identical shape.
    ref_key = next(k for k in valid_keys if layers.get(k) is not None)
    layers = align_layers_to_reference(layers, reference_key=ref_key)

    # Sample background scores across a reference layer
    reference = next(l for l in layers.values() if l is not None)
    arr_shape = reference.values.shape
    rng = np.random.default_rng(random_seed)
    bg_rows = rng.integers(0, arr_shape[0], n_background_samples)
    bg_cols = rng.integers(0, arr_shape[1], n_background_samples)

    def _objective(w_vec: np.ndarray) -> float:
        """Negative margin: -(mean_site_score − mean_bg_score)."""
        weights = {k: max(float(w_vec[i]), 0.0) for i, k in enumerate(valid_keys)}
        total = sum(weights.values())
        if total == 0:
            return 0.0
        weights = {k: v / total for k, v in weights.items()}

        composite = weighted_sum(layers, weights)
        if composite is None:
            return 0.0

        site_scores = extract_scores_at_points(composite, sites_gdf)
        site_scores = site_scores[np.isfinite(site_scores)]

        arr = composite.values.astype(np.float32)
        bg_scores = arr[bg_rows, bg_cols].astype(np.float32)
        bg_scores = bg_scores[np.isfinite(bg_scores)]

        if len(site_scores) == 0 or len(bg_scores) == 0:
            return 0.0

        margin = float(np.mean(site_scores)) - float(np.mean(bg_scores))
        return -margin  # Minimize negative margin

    x0 = np.array([initial_weights.get(k, 0.2) for k in valid_keys], dtype=np.float64)
    bounds = [(0.0, 1.0)] * len(valid_keys)

    print(f"[fusion] Optimizing weights for layers: {valid_keys} ...")
    try:
        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        opt_w = np.maximum(result.x, 0.0)
        total = opt_w.sum()
        if total == 0:
            raise ValueError("All optimized weights are zero.")
        opt_w = opt_w / total
        optimized = {k: float(opt_w[i]) for i, k in enumerate(valid_keys)}
        # Fill in any missing keys with zero
        full_weights = {k: optimized.get(k, 0.0) for k in initial_weights}
        print(f"[fusion] Optimized weights: {full_weights}")
        return full_weights
    except Exception as exc:
        print(f"[fusion] Weight optimization failed ({exc}); using config weights.")
        return initial_weights


# ---------------------------------------------------------------------------
# Main fusion pipeline
# ---------------------------------------------------------------------------

def fuse_layers(
    tpi: Optional[xr.DataArray],
    lrm: Optional[xr.DataArray],
    ndvi_anomaly: Optional[xr.DataArray],
    sar_anomaly: Optional[xr.DataArray],
    geometric: Optional[xr.DataArray],
    east_sightline: Optional[xr.DataArray] = None,
    cop_tpi: Optional[xr.DataArray] = None,
    ndvi_dry: Optional[xr.DataArray] = None,
    thermal: Optional[xr.DataArray] = None,
    gedi_relief: Optional[xr.DataArray] = None,
    weights: dict[str, float] = config.FUSION_WEIGHTS,
    output_path: Path = config.COMPOSITE_SCORE_PATH,
) -> Optional[xr.DataArray]:
    """Fuse all anomaly layers into a normalized composite site probability score.

    Normalizes each layer, aligns all to the LRM/TPI grid, applies the
    weighted sum, saves the result as a GeoTIFF, and returns the composite.

    Args:
        tpi: Topographic Position Index DataArray (large-scale, SRTM).
        lrm: Local Relief Model DataArray (SRTM).
        ndvi_anomaly: NDVI anomaly z-score DataArray.
        sar_anomaly: Combined SAR anomaly DataArray.
        geometric: Lineament density DataArray.
        east_sightline: East-facing elevated feature with open eastern horizon.
        cop_tpi: TPI computed from Copernicus DEM.
        ndvi_dry: Dry-season NDVI anomaly z-score.
        thermal: Landsat thermal anomaly z-score.
        gedi_relief: Local Relief Model from GEDI ground elevation.
        weights: Layer weights (keys must match layer names used internally).
        output_path: Path to save the GeoTIFF composite score.

    Returns:
        Normalized composite score DataArray, or None if fusion fails.
    """
    raw_layers: dict[str, Optional[xr.DataArray]] = {
        "tpi": tpi,
        "lrm": lrm,
        "ndvi": ndvi_anomaly,
        "sar": sar_anomaly,
        "geometric": geometric,
        "east_sightline": east_sightline,
        "cop_tpi": cop_tpi,
        "ndvi_dry": ndvi_dry,
        "thermal": thermal,
        "gedi_relief": gedi_relief,
    }

    # Layers where low values indicate anomaly → invert so high score = anomaly
    invert_layers = {"ndvi", "ndvi_dry"}

    print("[fusion] Normalizing layers ...")
    normalized: dict[str, Optional[xr.DataArray]] = {}
    for name, da in raw_layers.items():
        if da is None:
            normalized[name] = None
            continue
        invert = name in invert_layers
        norm = normalize_layer(da, invert=invert)
        if norm is not None:
            norm = norm.rename(name)
        normalized[name] = norm

    print("[fusion] Aligning layers to common grid ...")
    # Use LRM as reference (same resolution as DEM), fallback to TPI
    ref_key = "lrm" if normalized.get("lrm") is not None else "tpi"
    aligned = align_layers_to_reference(normalized, reference_key=ref_key)

    print("[fusion] Computing weighted composite ...")
    composite = weighted_sum(aligned, weights)
    if composite is None:
        return None

    # Final normalization to [0, 1]
    composite = normalize_layer(composite)
    if composite is None:
        return None
    composite = composite.rename("composite_score")

    # Save to GeoTIFF
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite.rio.to_raster(str(output_path))
        print(f"[fusion] Composite score saved to {output_path}.")
    except Exception as exc:
        print(f"[fusion] Could not save composite to {output_path}: {exc}")

    return composite
