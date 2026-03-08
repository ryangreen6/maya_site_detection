"""composite.py — Generate composite score visualizations and diagnostic figures.

Produces:
  1. Composite score map with known sites and candidate detections overlaid.
  2. Six-panel figure showing all input layers plus the composite score.
  3. Scatter plot: known site scores vs. background scores with threshold.
  4. ROC curve plot.

All figures saved as PNG to the outputs/maps directory.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import xarray as xr
import geopandas as gpd

import config
from visualize.maps import (
    _add_north_arrow,
    _add_scale_bar,
    _raster_to_display,
    _overlay_sites,
    _save_fig,
)


# ---------------------------------------------------------------------------
# 1. Composite score map
# ---------------------------------------------------------------------------

def plot_composite_score(
    score: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    candidates_gdf: Optional[gpd.GeoDataFrame] = None,
    hillshade: Optional[xr.DataArray] = None,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
    output_path: Path = config.STATIC_MAPS_DIR / "composite_score.png",
) -> None:
    """Plot the composite site probability score over a DEM hillshade.

    The DEM hillshade is drawn as a grayscale base layer; the composite
    score is overlaid with 60% opacity. Known sites are labeled by name;
    consolidated candidate detections are shown as stars.

    Args:
        score: Composite score DataArray (values in [0, 1]).
        sites_gdf: GeoDataFrame of known Maya site locations.
        candidates_gdf: Pre-consolidated GeoDataFrame of candidate detections.
        hillshade: Optional hillshade DataArray for the background layer.
        threshold: Composite score threshold; used to draw a contour line.
        output_path: Path to save the PNG.
    """
    if score is None:
        print("[composite] Score is None; skipping composite map.")
        return

    arr, extent = _raster_to_display(score)
    fig, ax = plt.subplots(figsize=(14, 11))

    # --- Background: DEM hillshade ---
    if hillshade is not None:
        try:
            hs_arr, hs_extent = _raster_to_display(hillshade)
            ax.imshow(
                hs_arr, cmap="gray", extent=hs_extent,
                origin="upper", vmin=0, vmax=255, zorder=1,
            )
        except Exception:
            pass

    # --- Composite score overlay (semi-transparent) ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "site_prob", ["#00000000", "#FFF5EB80", "#FD8D3C", "#D94701", "#7F0000"]
    )
    # Mask low-scoring pixels to keep them transparent over the hillshade
    display_arr = np.where(arr >= threshold, arr, np.nan)
    im = ax.imshow(
        display_arr, cmap=cmap, extent=extent,
        origin="upper", vmin=threshold, vmax=1.0,
        alpha=0.75, zorder=2,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Composite Score (≥{threshold})", fontsize=10)

    # --- Candidate sites (consolidated) ---
    if candidates_gdf is not None and not candidates_gdf.empty:
        try:
            cxs = candidates_gdf.geometry.x.values
            cys = candidates_gdf.geometry.y.values
            ax.scatter(
                cxs, cys, c="yellow", s=120, marker="*",
                edgecolors="#333333", linewidths=0.7, zorder=6,
                label=f"Candidate sites (n={len(candidates_gdf)})",
            )
        except Exception:
            pass

    # --- Known sites: markers + name labels ---
    if sites_gdf is not None and not sites_gdf.empty:
        try:
            xs = sites_gdf.geometry.x.values
            ys = sites_gdf.geometry.y.values
            ax.scatter(
                xs, ys, c="dodgerblue", s=60, marker="o",
                edgecolors="white", linewidths=1.2, zorder=7,
                label=f"Known Maya sites (n={len(sites_gdf)})",
            )
            name_col = "site_name" if "site_name" in sites_gdf.columns else sites_gdf.columns[0]
            for x, y, name in zip(xs, ys, sites_gdf[name_col]):
                ax.annotate(
                    str(name),
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    zorder=8,
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="black")
                    ],
                )
        except Exception as exc:
            print(f"[composite] Site label error: {exc}")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title(
        "Composite Site Probability — Northern Petén, Guatemala\n"
        "GEDI ground LRM · NDVI anomaly · SAR backscatter",
        fontsize=12,
    )
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Six-panel figure
# ---------------------------------------------------------------------------

def plot_layer_panel(
    hillshade: Optional[xr.DataArray],
    lrm: Optional[xr.DataArray],
    ndvi_anomaly: Optional[xr.DataArray],
    sar_anomaly: Optional[xr.DataArray],
    geometric: Optional[xr.DataArray],
    score: Optional[xr.DataArray],
    output_path: Path = config.STATIC_MAPS_DIR / "layer_panel.png",
) -> None:
    """Plot all five input layers plus the composite score in a 2×3 panel figure.

    Args:
        hillshade: Multi-directional hillshade DataArray.
        lrm: Local Relief Model DataArray.
        ndvi_anomaly: NDVI anomaly DataArray.
        sar_anomaly: SAR anomaly DataArray.
        geometric: Lineament density DataArray.
        score: Composite score DataArray.
        output_path: Path to save the PNG.
    """
    panels = [
        (hillshade, "Hillshade", "gray", None, None),
        (lrm, "Local Relief Model", "RdBu_r", "symmetric", None),
        (ndvi_anomaly, "NDVI Anomaly (z-score)", "RdYlGn", "symmetric", None),
        (sar_anomaly, "SAR Anomaly (z-score)", "PuOr", "symmetric", None),
        (geometric, "Lineament Density", "hot_r", None, None),
        (score, "Composite Score", None, None, "site_prob"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "site_prob", ["white", "#FFF5EB", "#FD8D3C", "#D94701", "#7F0000"]
    )

    for i, (da, title, cmap_name, stretch, cmap_custom) in enumerate(panels):
        ax = axes[i]
        if da is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            continue

        arr, extent = _raster_to_display(da)

        if cmap_custom == "site_prob":
            cmap = score_cmap
            vmin, vmax = 0, 1
        elif stretch == "symmetric":
            vmax = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))
            vmin = -vmax
            cmap = cmap_name
        else:
            vmin, vmax = None, None
            cmap = cmap_name or "viridis"

        im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Maya Site Detection — Input Layers and Composite Score", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Score distribution scatter plot
# ---------------------------------------------------------------------------

def plot_score_scatter(
    site_scores: np.ndarray,
    bg_scores: np.ndarray,
    threshold: float = config.COMPOSITE_SCORE_THRESHOLD,
    output_path: Path = config.STATIC_MAPS_DIR / "score_scatter.png",
) -> None:
    """Scatter / jitter plot of known site scores vs. background scores.

    Overlays a horizontal dashed line at the detection threshold.

    Args:
        site_scores: 1D array of composite scores at known site locations.
        bg_scores: 1D array of composite scores at random background locations.
        threshold: Detection threshold score (drawn as a horizontal line).
        output_path: Path to save the PNG.
    """
    if len(site_scores) == 0 and len(bg_scores) == 0:
        print("[composite] No score data for scatter plot.")
        return

    rng = np.random.default_rng(42)
    n_bg_plot = min(len(bg_scores), 500)
    bg_subset = bg_scores[rng.choice(len(bg_scores), n_bg_plot, replace=False)] \
        if len(bg_scores) > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(bg_subset) > 0:
        bg_x = rng.uniform(0.7, 1.3, len(bg_subset))
        ax.scatter(bg_x, bg_subset, alpha=0.3, color="steelblue", s=10,
                   label=f"Background (n={len(bg_subset)})")

    if len(site_scores) > 0:
        site_valid = site_scores[np.isfinite(site_scores)]
        site_x = rng.uniform(1.7, 2.3, len(site_valid))
        ax.scatter(site_x, site_valid, alpha=0.7, color="tomato", s=25,
                   label=f"Known sites (n={len(site_valid)})")

    ax.axhline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"Threshold = {threshold}")

    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["Background", "Known Sites"])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylabel("Composite Score")
    ax.set_title("Score Distribution: Known Sites vs. Background")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 4. ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    roc_data: dict,
    output_path: Path = config.STATIC_MAPS_DIR / "roc_curve.png",
) -> None:
    """Plot the ROC curve with AUC annotation.

    Args:
        roc_data: Dictionary from analysis.validate.compute_roc() with keys
                  'fpr', 'tpr', 'thresholds', 'auc'.
        output_path: Path to save the PNG.
    """
    if not roc_data or "fpr" not in roc_data:
        print("[composite] No ROC data available; skipping ROC plot.")
        return

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc_val = roc_data.get("auc", float("nan"))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Detection Rate)")
    ax.set_title("ROC Curve — Maya Site Detection")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_all_composite_figures(
    score: Optional[xr.DataArray],
    hillshade: Optional[xr.DataArray],
    lrm: Optional[xr.DataArray],
    ndvi_anomaly: Optional[xr.DataArray],
    sar_anomaly: Optional[xr.DataArray],
    geometric: Optional[xr.DataArray],
    sites_gdf: Optional[gpd.GeoDataFrame],
    candidates_gdf: Optional[gpd.GeoDataFrame],
    site_scores: Optional[np.ndarray],
    bg_scores: Optional[np.ndarray],
    roc_data: Optional[dict],
    output_dir: Path = config.STATIC_MAPS_DIR,
) -> None:
    """Generate all composite visualizations and save to output_dir.

    Args:
        score: Composite score DataArray.
        hillshade: Hillshade DataArray.
        lrm: LRM DataArray.
        ndvi_anomaly: NDVI anomaly DataArray.
        sar_anomaly: SAR anomaly DataArray.
        geometric: Lineament density DataArray.
        sites_gdf: GeoDataFrame of known sites.
        candidates_gdf: GeoDataFrame of candidate detections.
        site_scores: Array of composite scores at known sites.
        bg_scores: Array of composite scores at background locations.
        roc_data: Dictionary from compute_roc().
        output_dir: Directory where PNG files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Consolidate raw candidate clusters into distinct site locations
    from analysis.candidates import consolidate_candidates
    consolidated = consolidate_candidates(candidates_gdf, distance_m=500.0)

    plot_composite_score(
        score, sites_gdf, consolidated, hillshade=hillshade,
        output_path=output_dir / "composite_score.png"
    )
    plot_layer_panel(
        hillshade, lrm, ndvi_anomaly, sar_anomaly, geometric, score,
        output_path=output_dir / "layer_panel.png"
    )
    if site_scores is not None and bg_scores is not None:
        plot_score_scatter(
            site_scores, bg_scores,
            output_path=output_dir / "score_scatter.png"
        )
    if roc_data:
        plot_roc_curve(roc_data, output_path=output_dir / "roc_curve.png")

    print("[composite] All composite figures generated.")
