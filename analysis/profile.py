"""profile.py — Profile known Maya sites across all computed layers.

Extracts layer values at known site locations and at random background pixels,
then computes per-layer discrimination statistics (Cohen's d, Mann-Whitney U)
to identify which remote-sensing signals most strongly distinguish site from
non-site terrain.  Results inform fusion weight choices.
"""

from typing import Optional

import numpy as np
import xarray as xr
import geopandas as gpd

import config


def _sample_layer_at_points(
    da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Extract values from a DataArray at GeoDataFrame point locations.

    Args:
        da: DataArray with x/y coordinate dimensions (projected CRS).
        gdf: GeoDataFrame of point geometries in the same CRS.

    Returns:
        1-D float32 array of sampled values (NaN where outside extent).
    """
    arr = da.values.astype(np.float32)
    x_coords = da.coords["x"].values
    y_coords = da.coords["y"].values
    out = []
    for geom in gdf.geometry:
        xi = int(np.argmin(np.abs(x_coords - geom.x)))
        yi = int(np.argmin(np.abs(y_coords - geom.y)))
        if 0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]:
            out.append(float(arr[yi, xi]))
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _sample_layer_at_random(
    da: xr.DataArray,
    n: int,
    exclude_gdf: Optional[gpd.GeoDataFrame],
    rng: np.random.Generator,
    exclusion_radius_px: int = 5,
) -> np.ndarray:
    """Sample a DataArray at random valid pixel locations.

    Pixels within *exclusion_radius_px* of any known site are excluded to
    avoid accidentally sampling a site as background.

    Args:
        da: DataArray to sample.
        n: Number of background samples.
        exclude_gdf: Known site GeoDataFrame (same CRS); may be None.
        rng: numpy Generator for reproducibility.
        exclusion_radius_px: Pixel buffer around known sites to exclude.

    Returns:
        1-D float32 array of background values.
    """
    arr = da.values.astype(np.float32)
    x_coords = da.coords["x"].values
    y_coords = da.coords["y"].values

    # Build set of pixel indices to exclude around known sites
    excluded: set[tuple[int, int]] = set()
    if exclude_gdf is not None:
        r = exclusion_radius_px
        for geom in exclude_gdf.geometry:
            xi = int(np.argmin(np.abs(x_coords - geom.x)))
            yi = int(np.argmin(np.abs(y_coords - geom.y)))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    excluded.add((yi + dy, xi + dx))

    valid_rows, valid_cols = np.where(np.isfinite(arr))
    if len(valid_rows) == 0:
        return np.full(n, np.nan, dtype=np.float32)

    samples = []
    attempts = 0
    max_attempts = n * 20
    idx = rng.integers(0, len(valid_rows), size=max_attempts)
    for i in idx:
        r, c = int(valid_rows[i]), int(valid_cols[i])
        if (r, c) not in excluded:
            samples.append(float(arr[r, c]))
        if len(samples) >= n:
            break

    return np.array(samples, dtype=np.float32)


def profile_layer_discrimination(
    layers: dict[str, Optional[xr.DataArray]],
    sites_gdf: gpd.GeoDataFrame,
    n_background: int = 2000,
    random_seed: int = config.RANDOM_SEED,
) -> None:
    """Print a ranked table of per-layer discrimination power at known sites.

    For each layer, computes:
      - Mean and std at known sites and at random background pixels.
      - Cohen's d effect size (site − background) / pooled std.
      - Mann-Whitney U p-value (one-sided: sites > background).
      - Direction: whether sites score higher (+) or lower (-) than background.

    Args:
        layers: Dict of layer name → DataArray. None values are skipped.
        sites_gdf: GeoDataFrame of known site point locations.
        n_background: Number of random background pixels to sample.
        random_seed: Seed for reproducibility.
    """
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(random_seed)
    results = []

    for name, da in layers.items():
        if da is None:
            continue

        site_vals = _sample_layer_at_points(da, sites_gdf)
        bg_vals = _sample_layer_at_random(da, n_background, sites_gdf, rng)

        sv = site_vals[np.isfinite(site_vals)]
        bv = bg_vals[np.isfinite(bg_vals)]

        if len(sv) == 0 or len(bv) == 0:
            results.append((name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "?"))
            continue

        s_mean, s_std = float(np.mean(sv)), float(np.std(sv))
        b_mean, b_std = float(np.mean(bv)), float(np.std(bv))

        pooled_std = np.sqrt((s_std ** 2 + b_std ** 2) / 2.0)
        cohens_d = (s_mean - b_mean) / pooled_std if pooled_std > 0 else 0.0

        # Two-sided Mann-Whitney; derive one-sided p based on direction
        try:
            direction = "higher" if s_mean >= b_mean else "lower"
            alt = "greater" if direction == "higher" else "less"
            _, p = mannwhitneyu(sv, bv, alternative=alt)
        except Exception:
            p = np.nan
            direction = "?"

        results.append((name, s_mean, s_std, b_mean, b_std, cohens_d, p, direction))

    # Sort by |Cohen's d| descending
    results.sort(key=lambda x: abs(x[5]) if np.isfinite(x[5]) else 0, reverse=True)

    print()
    print("=" * 80)
    print("  LAYER DISCRIMINATION PROFILE  (known sites vs. random background)")
    print("=" * 80)
    print(f"  {'Layer':<20} {'Site μ':>8} {'Site σ':>8} {'BG μ':>8} {'BG σ':>8} "
          f"{'Cohen d':>9} {'p-val':>8}  Direction")
    print("-" * 80)
    for name, sm, ss, bm, bs, d, p, direction in results:
        sig = "*" if (np.isfinite(p) and p < 0.05) else " "
        print(f"  {name:<20} {sm:>8.3f} {ss:>8.3f} {bm:>8.3f} {bs:>8.3f} "
              f"{d:>9.3f} {p:>8.4f}{sig}  sites {direction}")
    print("=" * 80)
    print("  * p < 0.05 (Mann-Whitney one-sided)")
    print()
