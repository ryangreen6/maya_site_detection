"""raw_data_preview.py — Quick visual sanity-check of all cached raw data.

Run directly:
    python visualize/raw_data_preview.py

Saves PNG thumbnails to outputs/maps/raw_preview/ showing:
  - SRTM DEM with hillshade
  - Sentinel-2 true-colour (B04/B03/B02) and false-colour NDVI (B08/B04)
  - Sentinel-1 VV and VH backscatter
  - Known site locations overlaid on each
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
import rioxarray  # noqa: F401

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.known_sites import get_known_sites, filter_sites_to_bbox


OUT_DIR = config.STATIC_MAPS_DIR / "raw_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sites_in_array_coords(da: xr.DataArray, sites_gdf):
    """Return pixel (col, row) positions for each known site within the array."""
    x_coords = da.coords["x"].values
    y_coords = da.coords["y"].values
    xs, ys = [], []
    for geom in sites_gdf.geometry:
        xi = int(np.argmin(np.abs(x_coords - geom.x)))
        yi = int(np.argmin(np.abs(y_coords - geom.y)))
        if 0 <= yi < len(y_coords) and 0 <= xi < len(x_coords):
            xs.append(xi)
            ys.append(yi)
    return xs, ys


def _norm(arr: np.ndarray, plow: float = 2, phigh: float = 98) -> np.ndarray:
    """Percentile-stretch an array to [0, 1] for display."""
    arr = arr.astype(np.float32)
    vmin = np.nanpercentile(arr, plow)
    vmax = np.nanpercentile(arr, phigh)
    if vmax == vmin:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def _save(fig, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_dem(sites_gdf) -> None:
    """SRTM DEM — elevation + simple hillshade overlay."""
    if not config.DEM_PATH.exists():
        print("  [skip] DEM not found.")
        return
    print("Plotting DEM ...")
    da = rioxarray.open_rasterio(str(config.DEM_PATH), masked=True).squeeze("band", drop=True)

    elev = da.values.astype(np.float32)
    # Simple hillshade using gradient
    dy, dx = np.gradient(np.nan_to_num(elev))
    azimuth = np.radians(315)
    altitude = np.radians(45)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    hs = np.clip(hs, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    img = ax.imshow(_norm(elev), cmap="terrain", origin="upper")
    fig.colorbar(img, ax=ax, fraction=0.03, label="Elevation (normalized)")
    xs, ys = _sites_in_array_coords(da, sites_gdf)
    ax.scatter(xs, ys, c="red", s=30, marker="^", zorder=5, label="Known sites")
    ax.set_title("SRTM DEM — Elevation")
    ax.set_xlabel("Column pixel"); ax.set_ylabel("Row pixel")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.imshow(_norm(elev), cmap="terrain", origin="upper", alpha=0.6)
    ax.imshow(hs, cmap="gray", origin="upper", alpha=0.5)
    ax.scatter(xs, ys, c="red", s=30, marker="^", zorder=5)
    ax.set_title("SRTM DEM — Hillshade overlay")
    ax.set_xlabel("Column pixel")

    fig.suptitle(
        f"SRTM 30m DEM  |  shape: {elev.shape}  |  "
        f"elev range: [{np.nanmin(elev):.0f}, {np.nanmax(elev):.0f}] m",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, "01_dem")


def plot_sentinel2(sites_gdf) -> None:
    """Sentinel-2 composite — true-colour and NDVI false-colour."""
    if not config.S2_COMPOSITE_PATH.exists():
        print("  [skip] Sentinel-2 composite not found.")
        return
    print("Plotting Sentinel-2 ...")
    ds = xr.open_dataset(config.S2_COMPOSITE_PATH)

    bands_present = list(ds.data_vars)
    print(f"  S2 bands in cache: {bands_present}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # True colour: B04 (red), B03 (green), B02 (blue)
    ax = axes[0]
    has_tc = all(b in ds for b in ["B04", "B03", "B02"])
    if has_tc:
        r = _norm(ds["B04"].values.astype(np.float32))
        g = _norm(ds["B03"].values.astype(np.float32))
        b = _norm(ds["B02"].values.astype(np.float32))
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0).clip(0, 1)
        ax.imshow(rgb, origin="upper")
        ax.set_title("Sentinel-2 True Colour (B4/B3/B2)")
    else:
        ax.set_title("True colour N/A")
        ax.text(0.5, 0.5, "Bands B02/B03/B04\nnot all available",
                ha="center", va="center", transform=ax.transAxes)

    da_ref = ds[bands_present[0]]
    xs, ys = _sites_in_array_coords(da_ref, sites_gdf)
    ax.scatter(xs, ys, c="red", s=30, marker="^", zorder=5, label="Known sites")
    ax.set_xlabel("Column pixel"); ax.set_ylabel("Row pixel")
    ax.legend(fontsize=8)

    # NDVI false colour: B08 (NIR), B04 (red)
    ax = axes[1]
    has_ndvi = "B08" in ds and "B04" in ds
    if has_ndvi:
        nir = ds["B08"].values.astype(np.float32)
        red = ds["B04"].values.astype(np.float32)
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(denom != 0, (nir - red) / denom, np.nan).clip(-1, 1)
        img = ax.imshow(ndvi, cmap="RdYlGn", origin="upper", vmin=-0.5, vmax=1.0)
        fig.colorbar(img, ax=ax, fraction=0.03, label="NDVI")
        ax.set_title("Sentinel-2 NDVI")
    else:
        ax.set_title("NDVI N/A (need B04 + B08)")

    ax.scatter(xs, ys, c="red", s=30, marker="^", zorder=5)
    ax.set_xlabel("Column pixel")

    shape = da_ref.values.shape
    fig.suptitle(f"Sentinel-2 L2A composite  |  shape: {shape}", fontsize=10)
    plt.tight_layout()
    _save(fig, "02_sentinel2")


def plot_sentinel1(sites_gdf) -> None:
    """Sentinel-1 composite — VV and VH backscatter in dB."""
    if not config.S1_COMPOSITE_PATH.exists():
        print("  [skip] Sentinel-1 composite not found.")
        return
    print("Plotting Sentinel-1 ...")
    ds = xr.open_dataset(config.S1_COMPOSITE_PATH)
    bands_present = list(ds.data_vars)
    print(f"  S1 bands in cache: {bands_present}")

    vv = ds.get("vv", ds.get("VV"))
    vh = ds.get("vh", ds.get("VH"))
    if vv is None and vh is None:
        print("  [skip] No VV or VH bands found in S1 cache.")
        return

    da_ref = vv if vv is not None else vh
    xs, ys = _sites_in_array_coords(da_ref, sites_gdf)

    n_panels = (vv is not None) + (vh is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    for band, label in [(vv, "VV"), (vh, "VH")]:
        if band is None:
            continue
        arr = band.values.astype(np.float32)
        # Convert linear to dB if values look like linear power (most > 0.001)
        if np.nanmedian(arr[arr > 0]) < 10:
            with np.errstate(divide="ignore", invalid="ignore"):
                arr_db = np.where(arr > 0, 10 * np.log10(arr), np.nan)
        else:
            arr_db = arr  # already dB

        ax = axes[panel]
        img = ax.imshow(_norm(arr_db, plow=5, phigh=95), cmap="gray", origin="upper")
        fig.colorbar(img, ax=ax, fraction=0.03, label=f"{label} backscatter (normalized)")
        ax.scatter(xs, ys, c="red", s=30, marker="^", zorder=5, label="Known sites")
        ax.set_title(f"Sentinel-1 {label} backscatter")
        ax.set_xlabel("Column pixel")
        if panel == 0:
            ax.set_ylabel("Row pixel")
        ax.legend(fontsize=8)
        panel += 1

    shape = da_ref.values.shape
    fig.suptitle(f"Sentinel-1 GRD composite  |  shape: {shape}", fontsize=10)
    plt.tight_layout()
    _save(fig, "03_sentinel1")


def plot_known_sites_summary(sites_gdf) -> None:
    """Table-style figure listing all known sites with coordinates."""
    print("Plotting known sites summary ...")
    fig, ax = plt.subplots(figsize=(10, 0.4 * (len(sites_gdf) + 2)))
    ax.axis("off")

    rows = []
    for _, row in sites_gdf.iterrows():
        rows.append([
            row.get("site_name", "?"),
            f"{row.geometry.y:.4f}",
            f"{row.geometry.x:.4f}",
            row.get("source", ""),
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Site name", "Lat (UTM northing)", "Lon (UTM easting)", "Source"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width([0, 1, 2, 3])
    ax.set_title(f"Known Maya sites ({len(sites_gdf)} total, projected CRS: {config.CRS})",
                 pad=12, fontsize=11)
    plt.tight_layout()
    _save(fig, "00_known_sites")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nRaw data preview — outputs will be saved to: {OUT_DIR}\n")

    # Load known sites in project CRS for overlay
    sites_gdf = get_known_sites(csv_path=config.KNOWN_SITES_CSV, target_crs=config.CRS)
    sites_gdf = filter_sites_to_bbox(sites_gdf)
    print(f"Loaded {len(sites_gdf)} known sites for overlay.\n")

    plot_known_sites_summary(sites_gdf)
    plot_dem(sites_gdf)
    plot_sentinel2(sites_gdf)
    plot_sentinel1(sites_gdf)

    print(f"\nAll previews saved to: {OUT_DIR}")
    print("Open the PNG files to inspect the raw data.")


if __name__ == "__main__":
    main()
