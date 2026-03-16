"""
fetch_water_layers.py
─────────────────────
Downloads and visualises three freshwater datasets for the northern Petén AOI:
  1. JRC Global Surface Water — occurrence + seasonality (via Planetary Computer)
  2. HydroLAKES v1.0          — lake polygons  (direct zip download, ~820 MB)
  3. JRC GSW Seasonality       — used as GLWD-equivalent wetland map
       (GLWD v2 GeoTIFF is 900+ MB; JRC seasonality at 30 m is a better proxy)

Outputs → data/raw/water_layers/
  jrc_occurrence.tif
  jrc_seasonality.tif
  hydrolakes_aoi.gpkg
  viz_jrc_occurrence.png
  viz_jrc_seasonality.png
  viz_hydrolakes.png
"""

from pathlib import Path
import urllib.request, zipfile, io, sys, requests

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import from_bounds
from shapely.geometry import box
import pystac_client
import planetary_computer as pc

# ── Config ───────────────────────────────────────────────────────────────────
AOI  = (-91.5, 16.5, -89.0, 18.2)   # west, south, east, north
W, S, E, N = AOI
OUT  = Path("data/raw/water_layers")
OUT.mkdir(parents=True, exist_ok=True)

HYDROLAKES_URL = (
    "https://data.hydrosheds.org/file/HydroLAKES/"
    "HydroLAKES_polys_v10_shp.zip"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  JRC Global Surface Water via Planetary Computer
# ─────────────────────────────────────────────────────────────────────────────
def fetch_jrc():
    print("\n─── JRC Global Surface Water (Planetary Computer) ───")
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = cat.search(
        collections=["jrc-gsw"],
        bbox=AOI,
    )
    items = list(search.items())
    print(f"  Found {len(items)} JRC-GSW tile(s) covering AOI")

    for asset_name, out_fname in [("occurrence", "jrc_occurrence.tif"),
                                   ("seasonality", "jrc_seasonality.tif")]:
        out_path = OUT / out_fname
        if out_path.exists():
            print(f"  {out_fname} already exists, skipping download")
            continue

        # Collect per-tile assets, merge, clip
        tile_paths = []
        for item in items:
            signed = pc.sign(item)
            href   = signed.assets[asset_name].href
            tile_paths.append(href)

        # Open all tiles via VSICURL and merge
        srcs   = [rasterio.open(p) for p in tile_paths]
        merged, transform = merge(srcs, bounds=(W, S, E, N))
        meta = srcs[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": merged.shape[1],
            "width":  merged.shape[2],
            "transform": transform,
            "compress": "lzw",
        })
        for s in srcs:
            s.close()

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(merged)
        print(f"  Saved {out_fname}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("  JRC done.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HydroLAKES
# ─────────────────────────────────────────────────────────────────────────────
def fetch_hydrolakes():
    print("\n─── HydroLAKES v1.0 ───")
    out_gpkg = OUT / "hydrolakes_aoi.gpkg"
    if out_gpkg.exists():
        print("  hydrolakes_aoi.gpkg already exists, skipping download")
        return

    zip_path = OUT / "HydroLAKES_polys_v10_shp.zip"
    if not zip_path.exists():
        print(f"  Downloading HydroLAKES (~820 MB) …")
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        with requests.get(HYDROLAKES_URL, headers=headers, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    pct   = done / total * 100 if total else 0
                    print(f"\r  {done/1e6:6.0f} MB / {total/1e6:.0f} MB  ({pct:.0f}%)", end="", flush=True)
        print()
    else:
        print("  Zip already cached.")

    print("  Extracting and clipping to AOI …")
    aoi_geom = box(W, S, E, N)
    with zipfile.ZipFile(zip_path) as zf:
        # Find the .shp file name
        shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
        tmp_dir  = OUT / "_hl_tmp"
        tmp_dir.mkdir(exist_ok=True)
        zf.extractall(tmp_dir)

    gdf = gpd.read_file(tmp_dir / shp_name, bbox=(W, S, E, N))
    gdf = gdf[gdf.intersects(aoi_geom)].copy()
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"  Saved hydrolakes_aoi.gpkg  ({len(gdf)} lakes in AOI)")

    # Clean up extracted files (keep the zip for re-use)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────
def viz_jrc_occurrence():
    path = OUT / "jrc_occurrence.tif"
    if not path.exists():
        print("  Skipping occurrence viz — file missing")
        return
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")
    cmap = plt.cm.Blues
    cmap.set_bad("#1a1d27")
    im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap,
                   vmin=0, vmax=100, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Occurrence (%)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    ax.set_title("JRC Global Surface Water — Occurrence\nNorthern Petén, Guatemala",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3148")
    fig.tight_layout()
    out = OUT / "viz_jrc_occurrence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out.name}")


def viz_jrc_seasonality():
    path = OUT / "jrc_seasonality.tif"
    if not path.exists():
        print("  Skipping seasonality viz — file missing")
        return
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata if src.nodata is not None else 255
        data[data == nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # 0 = never water, 1–11 = seasonal, 12 = permanent
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")

    # Custom colormap: dark = never water, green = seasonal, blue = permanent
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "seas", [(0, "#1a1d27"), (0.08, "#2d6a4f"), (0.5, "#52b788"), (1.0, "#1e6091")]
    )
    cmap.set_bad("#1a1d27")
    im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap,
                   vmin=0, vmax=12, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                        ticks=[0, 3, 6, 9, 12])
    cbar.set_label("Months/year wet", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    patches = [
        mpatches.Patch(color="#1a1d27",  label="Never wet (0 mo)"),
        mpatches.Patch(color="#2d6a4f",  label="Seasonal  (1–3 mo)  → aguadas/wetlands"),
        mpatches.Patch(color="#52b788",  label="Semi-perm (4–8 mo)"),
        mpatches.Patch(color="#1e6091",  label="Permanent (9–12 mo) → lakes/rivers"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7.5,
              facecolor="#13161f", edgecolor="#2d3148",
              labelcolor="white", framealpha=0.85)
    ax.set_title("JRC Global Surface Water — Seasonality (GLWD proxy)\nNorthern Petén, Guatemala",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3148")
    fig.tight_layout()
    out = OUT / "viz_jrc_seasonality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out.name}")


def viz_hydrolakes():
    path = OUT / "hydrolakes_aoi.gpkg"
    if not path.exists():
        print("  Skipping HydroLAKES viz — file missing")
        return
    gdf = gpd.read_file(path)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")
    ax.set_xlim(W, E); ax.set_ylim(S, N)

    if not gdf.empty:
        # Size lakes by area for colour scale
        areas = gdf.geometry.area  # approx in deg²
        gdf.plot(ax=ax, color="#1e6091", edgecolor="#60a5fa", linewidth=0.5,
                 alpha=0.85)
        # Label the biggest ones
        if "Lake_name" in gdf.columns:
            big = gdf[gdf.geometry.area > gdf.geometry.area.quantile(0.85)]
            for _, row in big.iterrows():
                name = str(row.get("Lake_name", "")).strip()
                if name and name.lower() != "nan":
                    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                    ax.text(cx, cy, name, fontsize=6, color="white",
                            ha="center", va="center",
                            fontweight="bold", alpha=0.9)
    else:
        ax.text((W+E)/2, (S+N)/2, "No HydroLAKES features in AOI",
                ha="center", va="center", color="#94a3b8", fontsize=12)

    ax.set_title(f"HydroLAKES v1.0 — {len(gdf)} lakes\nNorthern Petén, Guatemala",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3148")
    # Grid
    ax.grid(color="#2d3148", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out = OUT / "viz_hydrolakes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetch_jrc()
    fetch_hydrolakes()

    print("\n─── Generating visualisations ───")
    viz_jrc_occurrence()
    viz_jrc_seasonality()
    viz_hydrolakes()
    print("\nDone. Outputs in", OUT)
