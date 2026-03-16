"""
map_water_layers.py
────────────────────
Generates publication-quality maps for the three freshwater datasets,
matching the style of visualize/maps.py (white bg, UTM 16N, sites overlay,
north arrow, scale bar, 200 dpi).

Outputs → outputs/maps/
  jrc_occurrence.png
  jrc_seasonality.png
  hydrolakes.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform

import config
from data.known_sites import get_known_sites

# ── Paths ─────────────────────────────────────────────────────────────────────
WATER_DIR  = config.RAW_DATA_DIR / "water_layers"
OUT_DIR    = config.STATIC_MAPS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = config.CRS   # EPSG:32616


# ── Helpers (mirrors visualize/maps.py) ───────────────────────────────────────
def _add_north_arrow(ax, x=0.95, y=0.95):
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - 0.07),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=12, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )


def _add_scale_bar(ax, x_min, x_max, bar_fraction=0.2, y_frac=0.05, x_frac=0.05):
    try:
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        bar_len_m = round((x_max - x_min) * bar_fraction / 5000) * 5000
        if bar_len_m == 0:
            return
        x0 = xlim[0] + (xlim[1] - xlim[0]) * x_frac
        y0 = ylim[0] + (ylim[1] - ylim[0]) * y_frac
        ax.plot([x0, x0 + bar_len_m], [y0, y0], color="black", lw=3, solid_capstyle="butt")
        ax.text(x0 + bar_len_m / 2, y0 + (ylim[1] - ylim[0]) * 0.015,
                f"{bar_len_m/1000:.0f} km", ha="center", va="bottom",
                fontsize=9, color="black")
    except Exception:
        pass


def _overlay_sites(ax, sites_gdf, label="Known Maya sites", color="red"):
    if sites_gdf is None or sites_gdf.empty:
        return
    ax.scatter(sites_gdf.geometry.x, sites_gdf.geometry.y,
               c=color, s=40, marker="o", edgecolors="white",
               linewidths=0.6, zorder=5, label=label)
    for _, row in sites_gdf.iterrows():
        ax.annotate(
            row["site_name"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=6.5, color="black", fontweight="bold",
            zorder=6,
        )


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[water maps] Saved {path}")


def _reproject_raster(src_path: Path) -> tuple[np.ndarray, list, rasterio.crs.CRS]:
    """Reproject a GeoTIFF to TARGET_CRS, return (array, extent, crs)."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        data = np.empty((height, width), dtype=np.float32)
        nodata = src.nodata if src.nodata is not None else 255
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest,
            dst_nodata=np.nan,
        )
        data[data == nodata] = np.nan
        # Compute projected extent
        left   = transform.c
        top    = transform.f
        right  = left + transform.a * width
        bottom = top  + transform.e * height
        extent = [left, right, bottom, top]
    return data, extent


# ── Load known sites in UTM 16N ───────────────────────────────────────────────
sites = get_known_sites(csv_path=config.KNOWN_SITES_CSV, target_crs=TARGET_CRS)


# ── 1. JRC Occurrence ─────────────────────────────────────────────────────────
def map_jrc_occurrence():
    data, extent = _reproject_raster(WATER_DIR / "jrc_occurrence.tif")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="Blues", extent=extent, origin="upper",
                   vmin=0, vmax=100, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Surface water occurrence (%)", fontsize=10)

    _overlay_sites(ax, sites)
    if sites is not None and not sites.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])

    ax.set_title(
        "JRC Global Surface Water — Occurrence (1984–2021)\n"
        "Northern Petén, Guatemala",
        fontsize=13,
    )
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save(fig, "jrc_occurrence.png")


# ── 2. JRC Seasonality (GLWD proxy) ───────────────────────────────────────────
def map_jrc_seasonality():
    data, extent = _reproject_raster(WATER_DIR / "jrc_seasonality.tif")

    # Custom colormap: white=never, green=seasonal/aguadas, blue=permanent
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "seas",
        [
            (0.00, "#f8f9fa"),   # never wet
            (0.01, "#b7e4c7"),   # 1 month
            (0.25, "#52b788"),   # seasonal wetlands / aguadas
            (0.70, "#2d6a4f"),   # semi-permanent
            (1.00, "#1e6091"),   # permanent (12 months)
        ],
    )
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper",
                   vmin=0, vmax=12, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                      ticks=[0, 3, 6, 9, 12])
    cb.set_label("Months/year with surface water", fontsize=10)
    cb.ax.set_yticklabels(["0 (never)", "3", "6", "9", "12 (permanent)"])

    legend_patches = [
        mpatches.Patch(color="#f8f9fa",  label="Never wet",                  ec="gray", lw=0.5),
        mpatches.Patch(color="#b7e4c7",  label="Seasonal  1–3 mo (aguadas/wetlands)"),
        mpatches.Patch(color="#52b788",  label="Semi-permanent  4–8 mo"),
        mpatches.Patch(color="#1e6091",  label="Permanent  9–12 mo (lakes/rivers)"),
        mpatches.Patch(color="red",      label="Known Maya sites", ec="white", lw=0.6),
    ]
    _overlay_sites(ax, sites)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])

    ax.set_title(
        "JRC Global Surface Water — Seasonality\n"
        "Northern Petén, Guatemala  ·  GLWD-equivalent wetland proxy",
        fontsize=13,
    )
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save(fig, "jrc_seasonality.png")


# ── 3. HydroLAKES ─────────────────────────────────────────────────────────────
def map_hydrolakes():
    lakes = gpd.read_file(WATER_DIR / "hydrolakes_aoi.gpkg")
    lakes = lakes.to_crs(TARGET_CRS)

    # Get extent from sites bounds + small buffer
    bounds = lakes.total_bounds          # minx, miny, maxx, maxy
    buf    = 20_000                      # 20 km padding
    x_min, y_min = bounds[0] - buf, bounds[1] - buf
    x_max, y_max = bounds[2] + buf, bounds[3] + buf

    # Size lakes by volume for colour gradient (Hylak_id / Vol_total column)
    vol_col = "Vol_total" if "Vol_total" in lakes.columns else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#e8f4f8")   # light water-blue background

    if vol_col:
        lakes_sorted = lakes.sort_values(vol_col, ascending=True)
        vmax = float(np.nanpercentile(lakes_sorted[vol_col].fillna(0), 98))
        sm   = plt.cm.ScalarMappable(
            cmap="Blues_r",
            norm=mcolors.LogNorm(vmin=max(0.001, lakes_sorted[vol_col].min()), vmax=max(1, vmax))
        )
        for _, row in lakes_sorted.iterrows():
            vol   = row[vol_col] if row[vol_col] > 0 else 0.001
            color = sm.to_rgba(vol)
            gpd.GeoDataFrame([row], crs=TARGET_CRS).plot(
                ax=ax, color=color, edgecolor="#1e6091", linewidth=0.4, alpha=0.85
            )
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Lake volume (10⁶ m³, log scale)", fontsize=10)
    else:
        lakes.plot(ax=ax, color="#4a90d9", edgecolor="#1e6091", linewidth=0.4, alpha=0.85)

    # Label the largest lakes by area
    if "Lake_name" in lakes.columns:
        area_thresh = lakes.geometry.area.quantile(0.90)
        for _, row in lakes[lakes.geometry.area >= area_thresh].iterrows():
            name = str(row.get("Lake_name", "")).strip()
            if name and name.lower() not in ("nan", ""):
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                ax.text(cx, cy, name, fontsize=7, ha="center", va="center",
                        fontweight="bold", color="white",
                        path_effects=[
                            __import__("matplotlib.patheffects", fromlist=["withStroke"])
                            .withStroke(linewidth=2, foreground="black")
                        ])

    _overlay_sites(ax, sites)
    if sites is not None and not sites.empty:
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _add_north_arrow(ax)
    _add_scale_bar(ax, x_min, x_max)

    ax.set_title(
        f"HydroLAKES v1.0 — {len(lakes)} lakes\n"
        "Northern Petén, Guatemala",
        fontsize=13,
    )
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    ax.grid(color="white", linestyle="--", linewidth=0.4, alpha=0.5)
    _save(fig, "hydrolakes.png")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating water layer maps …")
    map_jrc_occurrence()
    map_jrc_seasonality()
    map_hydrolakes()
    print(f"\nDone. Maps saved to {OUT_DIR}")
