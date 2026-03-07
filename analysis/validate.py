"""validate.py — Evaluate detection performance against known Maya site locations.

Extracts composite scores at known site locations, samples random background
points as negative examples, computes the ROC curve and AUC, and reports
detection rate and false positive rate at the configured threshold.
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import config


def sample_background_scores(
    score: xr.DataArray,
    sites_gdf: gpd.GeoDataFrame,
    n_samples: int = config.N_RANDOM_NEGATIVES,
    random_seed: int = config.RANDOM_SEED,
    exclusion_radius_px: int = 5,
) -> np.ndarray:
    """Sample composite scores at random background locations.

    Avoids sampling within *exclusion_radius_px* pixels of any known site
    to prevent known sites from contaminating the negative class.

    Args:
        score: Composite score DataArray.
        sites_gdf: GeoDataFrame of known sites (same CRS as score).
        n_samples: Number of background samples to draw.
        random_seed: Random seed for reproducibility.
        exclusion_radius_px: Pixel radius around known sites to exclude.

    Returns:
        1D numpy array of background score values.
    """
    arr = score.values.astype(np.float32)
    x_coords = score.coords["x"].values
    y_coords = score.coords["y"].values

    # Build exclusion mask around known site pixels
    exclusion = np.zeros(arr.shape, dtype=bool)
    for geom in sites_gdf.geometry:
        xi = int(np.argmin(np.abs(x_coords - geom.x)))
        yi = int(np.argmin(np.abs(y_coords - geom.y)))
        r = exclusion_radius_px
        y0, y1 = max(0, yi - r), min(arr.shape[0], yi + r + 1)
        x0, x1 = max(0, xi - r), min(arr.shape[1], xi + r + 1)
        exclusion[y0:y1, x0:x1] = True

    valid_mask = (~exclusion) & np.isfinite(arr)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        print("[validate] No valid background pixels found.")
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(random_seed)
    n = min(n_samples, len(valid_rows))
    idx = rng.choice(len(valid_rows), size=n, replace=False)
    bg_scores = arr[valid_rows[idx], valid_cols[idx]]
    return bg_scores.astype(np.float32)


def extract_site_scores(
    score: xr.DataArray,
    sites_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Extract composite score values at each known site location.

    Args:
        score: Composite score DataArray.
        sites_gdf: GeoDataFrame of known site points (same CRS as score).

    Returns:
        1D numpy array of score values at site locations.
    """
    arr = score.values.astype(np.float32)
    x_coords = score.coords["x"].values
    y_coords = score.coords["y"].values

    scores: list[float] = []
    for geom in sites_gdf.geometry:
        xi = int(np.argmin(np.abs(x_coords - geom.x)))
        yi = int(np.argmin(np.abs(y_coords - geom.y)))
        if 0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]:
            val = float(arr[yi, xi])
            scores.append(val)
        else:
            scores.append(np.nan)

    return np.array(scores, dtype=np.float32)


def compute_roc(
    site_scores: np.ndarray,
    bg_scores: np.ndarray,
) -> dict:
    """Compute the ROC curve and AUC from site and background scores.

    Args:
        site_scores: Score values at known site locations (positive class).
        bg_scores: Score values at random background locations (negative class).

    Returns:
        Dictionary with keys: 'fpr' (array), 'tpr' (array), 'thresholds'
        (array), 'auc' (float). Returns empty dict on failure.
    """
    try:
        from sklearn.metrics import roc_curve, auc

        site_valid = site_scores[np.isfinite(site_scores)]
        bg_valid = bg_scores[np.isfinite(bg_scores)]

        if len(site_valid) == 0 or len(bg_valid) == 0:
            print("[validate] Not enough valid scores for ROC computation.")
            return {}

        y_true = np.concatenate(
            [np.ones(len(site_valid)), np.zeros(len(bg_valid))]
        )
        y_score = np.concatenate([site_valid, bg_valid])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }
    except ImportError:
        print("[validate] scikit-learn is required for ROC computation.")
        return {}
    except Exception as exc:
        print(f"[validate] ROC computation failed: {exc}")
        return {}


def compute_detection_metrics(
    site_scores: np.ndarray,
    bg_scores: np.ndarray,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
) -> dict[str, float]:
    """Compute detection rate and false positive rate at a fixed threshold.

    Args:
        site_scores: Score values at known site locations.
        bg_scores: Score values at random background locations.
        threshold: Composite score threshold for positive detection.

    Returns:
        Dictionary with keys: 'detection_rate', 'false_positive_rate',
        'n_sites', 'n_detected', 'n_background', 'n_false_positives'.
    """
    site_valid = site_scores[np.isfinite(site_scores)]
    bg_valid = bg_scores[np.isfinite(bg_scores)]

    n_sites = len(site_valid)
    n_detected = int((site_valid >= threshold).sum())
    n_bg = len(bg_valid)
    n_fp = int((bg_valid >= threshold).sum())

    dr = n_detected / n_sites if n_sites > 0 else 0.0
    fpr = n_fp / n_bg if n_bg > 0 else 0.0

    return {
        "detection_rate": round(dr, 4),
        "false_positive_rate": round(fpr, 4),
        "n_sites": n_sites,
        "n_detected": n_detected,
        "n_background": n_bg,
        "n_false_positives": n_fp,
    }


def run_validation(
    score: xr.DataArray,
    sites_gdf: gpd.GeoDataFrame,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
    n_background: int = config.N_RANDOM_NEGATIVES,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[Optional[pd.DataFrame], dict]:
    """Run the full validation pipeline and return a report DataFrame and ROC data.

    Extracts site and background scores, computes ROC curve, detection rate,
    and false positive rate. Prints a summary to console.

    Args:
        score: Composite score DataArray (values in [0, 1]).
        sites_gdf: GeoDataFrame of known Maya site locations (same CRS as score).
        threshold: Score threshold for binary detection.
        n_background: Number of random background samples.
        random_seed: Seed for reproducible sampling.

    Returns:
        Tuple of (validation_report DataFrame, roc_data dict).
        The DataFrame has one row per known site with columns: site_name,
        latitude, longitude, score, detected.
        Returns (None, {}) on failure.
    """
    if score is None:
        print("[validate] Score is None; cannot run validation.")
        return None, {}

    if sites_gdf is None or sites_gdf.empty:
        print("[validate] No known sites for validation.")
        return None, {}

    print("[validate] Extracting scores at known site locations ...")
    site_scores = extract_site_scores(score, sites_gdf)

    print(f"[validate] Sampling {n_background} background locations ...")
    bg_scores = sample_background_scores(
        score, sites_gdf, n_samples=n_background, random_seed=random_seed
    )

    roc_data = compute_roc(site_scores, bg_scores)
    metrics = compute_detection_metrics(site_scores, bg_scores, threshold)

    # Build per-site report
    report_rows: list[dict] = []
    wgs84 = sites_gdf.to_crs("EPSG:4326") if sites_gdf.crs else sites_gdf
    for i, (_, row) in enumerate(sites_gdf.iterrows()):
        s = float(site_scores[i]) if i < len(site_scores) else np.nan
        report_rows.append(
            {
                "site_name": row.get("site_name", f"site_{i}"),
                "longitude": wgs84.geometry.iloc[i].x,
                "latitude": wgs84.geometry.iloc[i].y,
                "composite_score": round(s, 4) if np.isfinite(s) else np.nan,
                "detected": bool(s >= threshold) if np.isfinite(s) else False,
            }
        )

    report_df = pd.DataFrame(report_rows)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Known sites evaluated      : {metrics['n_sites']}")
    print(f"  Sites detected (≥{threshold}) : {metrics['n_detected']} "
          f"({metrics['detection_rate']*100:.1f}%)")
    print(f"  Background samples         : {metrics['n_background']}")
    print(f"  False positives            : {metrics['n_false_positives']} "
          f"({metrics['false_positive_rate']*100:.1f}%)")
    if "auc" in roc_data:
        print(f"  ROC AUC                    : {roc_data['auc']:.4f}")
    print("=" * 60 + "\n")

    return report_df, roc_data
