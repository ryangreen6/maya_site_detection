"""statistics.py — Compute summary statistics for the composite score and
individual layers.

Produces score distribution summaries, per-layer contribution statistics,
candidate counts by score decile, and comparisons between known site scores
and random background. Results are saved to CSV.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import config


def score_distribution_stats(
    score: xr.DataArray,
    label: str = "full_aoi",
) -> pd.DataFrame:
    """Compute descriptive statistics for a score distribution.

    Args:
        score: Score DataArray (any value range, NaN values excluded).
        label: Label string identifying this distribution in the output.

    Returns:
        Single-row DataFrame with statistics: label, count, mean, std,
        min, p25, median, p75, p90, p95, max.
    """
    if score is None:
        return pd.DataFrame()
    arr = score.values.astype(np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return pd.DataFrame()

    row = {
        "label": label,
        "n_pixels": len(arr),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }
    return pd.DataFrame([row])


def site_vs_background_stats(
    score: xr.DataArray,
    sites_gdf: gpd.GeoDataFrame,
    n_background: int = config.N_RANDOM_NEGATIVES,
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Compare score distributions at known sites versus random background.

    Args:
        score: Composite score DataArray.
        sites_gdf: GeoDataFrame of known site locations.
        n_background: Number of random background locations to sample.
        random_seed: Random seed for reproducible background sampling.

    Returns:
        DataFrame with two rows: one for 'known_sites' and one for
        'background', with the same statistics as score_distribution_stats.
    """
    if score is None or sites_gdf is None or sites_gdf.empty:
        print("[statistics] Missing score or sites data for comparison.")
        return pd.DataFrame()

    from analysis.validate import extract_site_scores, sample_background_scores

    site_scores = extract_site_scores(score, sites_gdf)
    bg_scores = sample_background_scores(
        score, sites_gdf, n_samples=n_background, random_seed=random_seed
    )

    site_da = xr.DataArray(site_scores[np.isfinite(site_scores)])
    bg_da = xr.DataArray(bg_scores[np.isfinite(bg_scores)])

    df_site = score_distribution_stats(site_da, label="known_sites")
    df_bg = score_distribution_stats(bg_da, label="background")

    return pd.concat([df_site, df_bg], ignore_index=True)


def layer_contribution_stats(
    candidates_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Summarize per-layer score contributions across candidate sites.

    Examines the 'layer_*' columns in the candidates GeoDataFrame (produced
    by analysis/candidates.py) and reports mean, std, min, max per layer.

    Args:
        candidates_gdf: GeoDataFrame of candidate sites with 'layer_*' columns.

    Returns:
        DataFrame with one row per layer showing contribution statistics,
        or an empty DataFrame if no layer columns are found.
    """
    if candidates_gdf is None or candidates_gdf.empty:
        print("[statistics] No candidate sites for layer contribution stats.")
        return pd.DataFrame()

    layer_cols = [c for c in candidates_gdf.columns if c.startswith("layer_")]
    if not layer_cols:
        print("[statistics] No layer columns found in candidates GeoDataFrame.")
        return pd.DataFrame()

    rows: list[dict] = []
    for col in layer_cols:
        vals = candidates_gdf[col].dropna().values.astype(np.float32)
        if len(vals) == 0:
            continue
        layer_name = col.replace("layer_", "")
        rows.append(
            {
                "layer": layer_name,
                "n_candidates": len(vals),
                "mean_contribution": round(float(np.mean(vals)), 6),
                "std_contribution": round(float(np.std(vals)), 6),
                "min_contribution": round(float(np.min(vals)), 6),
                "max_contribution": round(float(np.max(vals)), 6),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("mean_contribution", ascending=False)
    return df.reset_index(drop=True)


def candidates_by_score_decile(
    candidates_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Count candidate sites by score decile.

    Args:
        candidates_gdf: GeoDataFrame of candidate sites with a 'mean_score' column.

    Returns:
        DataFrame with columns: decile, score_min, score_max, n_candidates.
    """
    if candidates_gdf is None or candidates_gdf.empty:
        return pd.DataFrame()
    if "mean_score" not in candidates_gdf.columns:
        print("[statistics] 'mean_score' column missing from candidates.")
        return pd.DataFrame()

    scores = candidates_gdf["mean_score"].dropna().values
    if len(scores) == 0:
        return pd.DataFrame()

    decile_edges = np.percentile(scores, np.arange(0, 110, 10))
    rows: list[dict] = []
    for i in range(len(decile_edges) - 1):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi)
        rows.append(
            {
                "decile": i + 1,
                "score_min": round(float(lo), 4),
                "score_max": round(float(hi), 4),
                "n_candidates": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_all_statistics(
    score: Optional[xr.DataArray],
    sites_gdf: Optional[gpd.GeoDataFrame],
    candidates_gdf: Optional[gpd.GeoDataFrame],
    n_background: int = config.N_RANDOM_NEGATIVES,
    random_seed: int = config.RANDOM_SEED,
    output_path: Path = config.STATISTICS_CSV_PATH,
) -> pd.DataFrame:
    """Compute and save all summary statistics to a CSV file.

    Combines full-AOI score distribution, site vs. background comparison,
    per-layer contribution stats, and candidate counts by decile into a
    single output CSV with section labels.

    Args:
        score: Composite score DataArray.
        sites_gdf: GeoDataFrame of known site locations.
        candidates_gdf: GeoDataFrame of extracted candidate sites.
        n_background: Number of random background samples.
        random_seed: Random seed for background sampling.
        output_path: Path to save the CSV statistics file.

    Returns:
        Combined statistics DataFrame.
    """
    all_frames: list[pd.DataFrame] = []

    # 1. Full AOI distribution
    print("[statistics] Computing AOI score distribution ...")
    df_aoi = score_distribution_stats(score, label="full_aoi")
    if not df_aoi.empty:
        df_aoi.insert(0, "section", "score_distribution")
        all_frames.append(df_aoi)

    # 2. Site vs background comparison
    print("[statistics] Computing site vs. background comparison ...")
    df_compare = site_vs_background_stats(
        score, sites_gdf, n_background, random_seed
    )
    if not df_compare.empty:
        df_compare.insert(0, "section", "site_vs_background")
        all_frames.append(df_compare)

    # 3. Per-layer contribution
    print("[statistics] Computing per-layer contribution statistics ...")
    df_layers = layer_contribution_stats(candidates_gdf)
    if not df_layers.empty:
        df_layers.insert(0, "section", "layer_contributions")
        all_frames.append(df_layers)

    # 4. Candidates by decile
    print("[statistics] Computing candidate counts by score decile ...")
    df_deciles = candidates_by_score_decile(candidates_gdf)
    if not df_deciles.empty:
        df_deciles.insert(0, "section", "candidates_by_decile")
        all_frames.append(df_deciles)

    if not all_frames:
        print("[statistics] No statistics computed.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"[statistics] Statistics saved to {output_path}.")
    except Exception as exc:
        print(f"[statistics] Could not save CSV: {exc}")

    return combined
