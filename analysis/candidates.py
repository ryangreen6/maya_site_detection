"""candidates.py — Extract discrete candidate Maya site locations from the
composite score map.

Thresholds the score surface, identifies connected components, filters by
minimum cluster size, computes per-candidate attributes (centroid, scores,
area, layer contributions), and returns a GeoDataFrame exported to GeoJSON.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from scipy.ndimage import label
from shapely.geometry import Point

import config


def threshold_score(
    score: xr.DataArray,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
) -> Optional[np.ndarray]:
    """Apply a threshold to the composite score to produce a binary mask.

    Args:
        score: Composite score DataArray (values in [0, 1]).
        threshold: Score value above which pixels are candidate sites.

    Returns:
        Boolean numpy array (True = candidate pixel), or None on error.
    """
    if score is None:
        print("[candidates] Score array is None; cannot threshold.")
        return None
    try:
        arr = score.values.astype(np.float32)
        mask = arr >= threshold
        n = int(mask.sum())
        print(
            f"[candidates] {n} pixels ({100*n/mask.size:.2f}%) exceed "
            f"threshold {threshold}."
        )
        return mask
    except Exception as exc:
        print(f"[candidates] Thresholding failed: {exc}")
        return None


def label_components(
    mask: np.ndarray,
    min_size: int = config.MIN_CANDIDATE_CLUSTER_SIZE,
) -> tuple[np.ndarray, int]:
    """Label connected components in a binary mask and filter by minimum size.

    Args:
        mask: Boolean array of above-threshold pixels.
        min_size: Minimum cluster size in pixels; smaller clusters are removed.

    Returns:
        Tuple of (labeled array, number of valid clusters). The labeled array
        has integer cluster IDs (0 = background).
    """
    labeled, n_raw = label(mask)
    # Filter small clusters
    valid_label = 0
    filtered = np.zeros_like(labeled)
    for cluster_id in range(1, n_raw + 1):
        pixels = labeled == cluster_id
        if pixels.sum() >= min_size:
            valid_label += 1
            filtered[pixels] = valid_label

    print(
        f"[candidates] {n_raw} raw clusters → {valid_label} clusters "
        f"≥ {min_size} pixels after size filter."
    )
    return filtered, valid_label


def _pixel_area_ha(score: xr.DataArray) -> float:
    """Compute the area of one pixel in hectares from the DataArray's CRS.

    Assumes projected coordinates in metres.

    Args:
        score: DataArray with projected spatial coordinates.

    Returns:
        Area of one pixel in hectares (float).
    """
    x = score.coords.get("x", score.coords.get("lon", None))
    y = score.coords.get("y", score.coords.get("lat", None))
    if x is not None and len(x) > 1:
        dx = float(abs(x[1] - x[0]))
    else:
        dx = 30.0
    if y is not None and len(y) > 1:
        dy = float(abs(y[1] - y[0]))
    else:
        dy = 30.0
    return (dx * dy) / 10_000.0  # m² → ha


def extract_candidate_attributes(
    score: xr.DataArray,
    labeled: np.ndarray,
    n_clusters: int,
    layer_arrays: Optional[dict[str, Optional[xr.DataArray]]] = None,
) -> gpd.GeoDataFrame:
    """Compute attributes for each candidate cluster and return a GeoDataFrame.

    For each labeled cluster, computes centroid coordinates, mean and max
    score, area in hectares, and which input layers contributed most to
    the high composite score.

    Args:
        score: Composite score DataArray used for score extraction.
        labeled: Labeled array from label_components().
        n_clusters: Number of valid clusters.
        layer_arrays: Optional dictionary of normalized input layer DataArrays
                      keyed by layer name, used to compute per-layer contribution.

    Returns:
        GeoDataFrame with columns: cluster_id, geometry (centroid), mean_score,
        max_score, area_ha, top_layer, plus per-layer mean score columns.
    """
    score_arr = score.values.astype(np.float32)
    x_coords = score.coords["x"].values
    y_coords = score.coords["y"].values
    pixel_ha = _pixel_area_ha(score)

    records: list[dict] = []

    for cid in range(1, n_clusters + 1):
        pixels = labeled == cid
        row_idx, col_idx = np.where(pixels)

        mean_score = float(np.nanmean(score_arr[pixels]))
        max_score = float(np.nanmax(score_arr[pixels]))
        area_ha = float(pixels.sum()) * pixel_ha

        # Centroid in projected coordinates
        cx = float(np.mean(x_coords[col_idx]))
        cy = float(np.mean(y_coords[row_idx]))

        record: dict = {
            "cluster_id": cid,
            "geometry": Point(cx, cy),
            "mean_score": round(mean_score, 4),
            "max_score": round(max_score, 4),
            "area_ha": round(area_ha, 4),
        }

        # Per-layer contribution scores
        if layer_arrays:
            layer_scores: dict[str, float] = {}
            for lname, lda in layer_arrays.items():
                if lda is None:
                    continue
                try:
                    # Align layer to score grid if shapes differ
                    if lda.shape != score.shape:
                        lda_aligned = lda.rio.reproject_match(score)
                    else:
                        lda_aligned = lda
                    larr = lda_aligned.values.astype(np.float32)
                    layer_scores[lname] = float(np.nanmean(larr[pixels]))
                except Exception:
                    layer_scores[lname] = float("nan")
            for lname, lscore in layer_scores.items():
                record[f"layer_{lname}"] = round(lscore, 4)
            if layer_scores:
                top = max(layer_scores, key=lambda k: layer_scores[k])
                record["top_layer"] = top
            else:
                record["top_layer"] = "unknown"
        else:
            record["top_layer"] = "unknown"

        records.append(record)

    if not records:
        print("[candidates] No candidate clusters extracted.")
        return gpd.GeoDataFrame(
            columns=["cluster_id", "geometry", "mean_score", "max_score", "area_ha"],
            crs=score.rio.crs,
        )

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=score.rio.crs)
    print(f"[candidates] Extracted {len(gdf)} candidate sites.")
    return gdf


def extract_candidates(
    score: xr.DataArray,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
    min_size: int = config.MIN_CANDIDATE_CLUSTER_SIZE,
    layer_arrays: Optional[dict[str, Optional[xr.DataArray]]] = None,
    output_path: Path = config.CANDIDATES_GEOJSON_PATH,
) -> Optional[gpd.GeoDataFrame]:
    """Full pipeline to extract candidate Maya site detections from a score map.

    Thresholds the score, labels connected components, filters by size,
    computes attributes, exports to GeoJSON, and returns a GeoDataFrame.

    Args:
        score: Composite score DataArray (values in [0, 1]).
        threshold: Score threshold for candidate pixel selection.
        min_size: Minimum cluster size in pixels.
        layer_arrays: Optional dict of normalized input layer DataArrays for
                      per-layer contribution reporting.
        output_path: Path to save the candidate GeoJSON file.

    Returns:
        GeoDataFrame of candidate sites, or None on failure.
    """
    if score is None:
        print("[candidates] Score is None; cannot extract candidates.")
        return None

    mask = threshold_score(score, threshold)
    if mask is None or mask.sum() == 0:
        print("[candidates] No pixels above threshold.")
        return None

    labeled, n_clusters = label_components(mask, min_size)
    if n_clusters == 0:
        print("[candidates] No clusters meet the minimum size requirement.")
        return None

    gdf = extract_candidate_attributes(score, labeled, n_clusters, layer_arrays)
    if gdf is None or gdf.empty:
        return None

    # Export GeoJSON
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(output_path), driver="GeoJSON")
        print(f"[candidates] Candidates saved to {output_path}.")
    except Exception as exc:
        print(f"[candidates] Could not save GeoJSON: {exc}")

    return gdf


def consolidate_candidates(
    candidates_gdf: gpd.GeoDataFrame,
    distance_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Merge candidate sites within *distance_m* metres of each other.

    Buffers each candidate centroid by half the distance, unions overlapping
    buffers, and returns the centroid of each merged group. Reduces thousands
    of raw clusters to a more interpretable set of distinct candidate locations.

    Args:
        candidates_gdf: GeoDataFrame of candidate centroids (Point geometry).
        distance_m: Maximum centre-to-centre distance for merging (metres).

    Returns:
        GeoDataFrame of consolidated candidate Points (same CRS).
    """
    if candidates_gdf is None or candidates_gdf.empty:
        return candidates_gdf

    from shapely.ops import unary_union

    buffered = candidates_gdf.geometry.buffer(distance_m / 2.0)
    union = unary_union(buffered)
    parts = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    consolidated = gpd.GeoDataFrame(
        {"geometry": [p.centroid for p in parts]},
        crs=candidates_gdf.crs,
    )
    print(
        f"[candidates] Consolidated {len(candidates_gdf)} raw candidates → "
        f"{len(consolidated)} sites (merge radius {distance_m:.0f} m)."
    )
    return consolidated
