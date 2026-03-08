"""raw_data_preview.py — Quick visual sanity-check of all cached raw data.

Run directly:
    python visualize/raw_data_preview.py

Saves PNG thumbnails to outputs/maps/raw_preview/:
  00_coverage_overview  — AOI + all raster footprints + known sites
  01_dem                — SRTM elevation + hillshade with sites overlaid
  02_sentinel2          — S2 false-colour (B08/B07/B04) + NDVI with coverage note
  03_sentinel1          — S1 VV and VH backscatter with in-extent sites
  04_known_sites        — Table of all 15 known sites
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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
# CRS-aware helpers
# ---------------------------------------------------------------------------

def _transform_sites(sites_gdf, target_crs: str):
    """Return (xs, ys, names) of sites projected into target_crs."""
    from pyproj import Transformer
    src_crs = sites_gdf.crs.to_epsg()
    if src_crs is None:
        src_crs = 32616  # fallback
    transformer = Transformer.from_crs(f"EPSG:{src_crs}", target_crs, always_xy=True)
    xs, ys, names = [], [], []
    for _, row in sites_gdf.iterrows():
        x, y = transformer.transform(row.geometry.x, row.geometry.y)
        xs.append(x)
        ys.append(y)
        names.append(row.get("site_name", "?"))
    return np.array(xs), np.array(ys), names


def _pixel_coords(da: xr.DataArray, xs_crs, ys_crs):
    """Convert CRS coordinates to pixel (col, row) indices clipped to array."""
    x_arr = da.coords["x"].values
    y_arr = da.coords["y"].values
    cols, rows, mask = [], [], []
    for x, y in zip(xs_crs, ys_crs):
        c = int(np.argmin(np.abs(x_arr - x)))
        r = int(np.argmin(np.abs(y_arr - y)))
        # Accept only if within 5% of array edge AND within actual extent
        in_x = float(x_arr.min()) <= x <= float(x_arr.max())
        in_y = min(float(y_arr.min()), float(y_arr.max())) <= y <= max(float(y_arr.min()), float(y_arr.max()))
        mask.append(in_x and in_y)
        cols.append(c)
        rows.append(r)
    return np.array(cols), np.array(rows), np.array(mask)


def _norm(arr: np.ndarray, plow: float = 2, phigh: float = 98) -> np.ndarray:
    """Percentile-stretch a float array to [0, 1] for display."""
    arr = arr.copy().astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    vmin = np.percentile(valid, plow)
    vmax = np.percentile(valid, phigh)
    if vmax == vmin:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def _save(fig, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _raster_bbox_wgs84(da: xr.DataArray) -> tuple:
    """Return (west, south, east, north) of a DataArray in WGS84."""
    from pyproj import Transformer
    crs = da.rio.crs
    if crs is None:
        crs_str = "EPSG:4326"
    else:
        crs_str = crs.to_string()

    x_arr = da.coords["x"].values
    y_arr = da.coords["y"].values
    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    y_min, y_max = float(y_arr.min()), float(y_arr.max())

    if "4326" in crs_str:
        return x_min, y_min, x_max, y_max

    t = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_min, y_min, y_max, y_max]
    lons, lats = t.transform(corners_x, corners_y)
    return min(lons), min(lats), max(lons), max(lats)


# ---------------------------------------------------------------------------
# Plot 0: AOI + coverage overview (in WGS84 degrees)
# ---------------------------------------------------------------------------

def plot_coverage_overview(sites_gdf) -> None:
    """Show AOI rectangle, each dataset's footprint, and site locations."""
    print("Plotting coverage overview ...")

    # Site coordinates in WGS84
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    site_lons, site_lats = t.transform(
        [r.geometry.x for _, r in sites_gdf.iterrows()],
        [r.geometry.y for _, r in sites_gdf.iterrows()],
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # AOI rectangle
    west, south, east, north = config.AOI_BBOX_WGS84
    aoi = mpatches.Rectangle(
        (west, south), east - west, north - south,
        linewidth=2, edgecolor="black", facecolor="lightyellow", alpha=0.4, label="AOI"
    )
    ax.add_patch(aoi)

    colors = {"DEM (SRTM)": "steelblue", "Sentinel-2": "forestgreen", "Sentinel-1": "darkorange"}

    # Dataset footprints
    datasets = [
        ("DEM (SRTM)", config.DEM_PATH),
        ("Sentinel-2", config.S2_COMPOSITE_PATH),
        ("Sentinel-1", config.S1_COMPOSITE_PATH),
    ]
    for label, path in datasets:
        if not path.exists():
            continue
        try:
            if path.suffix == ".tif":
                da = rioxarray.open_rasterio(str(path), masked=True).squeeze("band", drop=True)
            else:
                ds = xr.open_dataset(str(path))
                key = [k for k in ds.data_vars if k not in ("spatial_ref",)][0]
                da = ds[key]
                if da.rio.crs is None:
                    da = da.rio.write_crs(config.CRS)
            bw, bs, be, bn = _raster_bbox_wgs84(da)
            rect = mpatches.Rectangle(
                (bw, bs), be - bw, bn - bs,
                linewidth=1.5, edgecolor=colors[label],
                facecolor=colors[label], alpha=0.2, label=f"{label} coverage"
            )
            ax.add_patch(rect)
            ax.plot(
                [bw, be, be, bw, bw], [bs, bs, bn, bn, bs],
                color=colors[label], linewidth=1.5,
            )
        except Exception as exc:
            print(f"  Could not get extent for {label}: {exc}")

    # Known sites
    ax.scatter(site_lons, site_lats, c="red", s=60, marker="^",
               zorder=10, label="Known sites", edgecolors="darkred", linewidths=0.5)
    for lon, lat, name in zip(site_lons, site_lats,
                               [r.get("site_name", "?") for _, r in sites_gdf.iterrows()]):
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color="darkred")

    ax.set_xlim(west - 0.2, east + 0.2)
    ax.set_ylim(south - 0.2, north + 0.3)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Dataset Coverage vs Known Maya Sites", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))

    plt.tight_layout()
    _save(fig, "00_coverage_overview")


# ---------------------------------------------------------------------------
# Plot 1: SRTM DEM
# ---------------------------------------------------------------------------

def plot_dem(sites_gdf) -> None:
    if not config.DEM_PATH.exists():
        print("  [skip] DEM not found.")
        return
    print("Plotting DEM ...")
    da = rioxarray.open_rasterio(str(config.DEM_PATH), masked=True).squeeze("band", drop=True)
    elev = da.values.astype(np.float32)

    # Hillshade
    dy, dx = np.gradient(np.nan_to_num(elev))
    azimuth, altitude = np.radians(315), np.radians(45)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dx, dy)
    hs = (np.sin(altitude) * np.cos(slope)
          + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    hs = np.clip(hs, 0, 1)

    # Site coords transformed to DEM's CRS (WGS84 degrees)
    dem_crs = da.rio.crs.to_string() if da.rio.crs else "EPSG:4326"
    xs_d, ys_d, names = _transform_sites(sites_gdf, dem_crs)
    cols, rows, in_extent = _pixel_coords(da, xs_d, ys_d)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    img = ax.imshow(_norm(elev), cmap="terrain", origin="upper")
    fig.colorbar(img, ax=ax, fraction=0.03, label="Elevation (2–98th pct stretch)")
    if in_extent.any():
        ax.scatter(cols[in_extent], rows[in_extent], c="red", s=40, marker="^",
                   zorder=5, label=f"Known sites (n={in_extent.sum()})")
    ax.set_title("SRTM DEM — Elevation")
    ax.set_xlabel("Column pixel"); ax.set_ylabel("Row pixel")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.imshow(_norm(elev), cmap="terrain", origin="upper", alpha=0.7)
    ax.imshow(hs, cmap="gray", origin="upper", alpha=0.4)
    if in_extent.any():
        ax.scatter(cols[in_extent], rows[in_extent], c="red", s=40, marker="^", zorder=5)
        for c, r, nm in zip(cols[in_extent], rows[in_extent], np.array(names)[in_extent]):
            ax.annotate(nm, (c, r), xytext=(3, 3), textcoords="offset points",
                        fontsize=5, color="red")
    ax.set_title("SRTM DEM — Hillshade overlay")
    ax.set_xlabel("Column pixel")

    out_n = (~in_extent).sum()
    fig.suptitle(
        f"SRTM 30m DEM | shape: {elev.shape} | CRS: {dem_crs}\n"
        f"Elev range: [{np.nanmin(elev):.0f}, {np.nanmax(elev):.0f}] m | "
        f"{in_extent.sum()}/15 sites in extent ({out_n} outside)",
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "01_dem")


# ---------------------------------------------------------------------------
# Plot 2: Sentinel-2
# ---------------------------------------------------------------------------

def plot_sentinel2(sites_gdf) -> None:
    if not config.S2_COMPOSITE_PATH.exists():
        print("  [skip] Sentinel-2 composite not found.")
        return
    print("Plotting Sentinel-2 ...")
    ds = xr.open_dataset(config.S2_COMPOSITE_PATH)
    bands = list(ds.data_vars)
    da_ref = ds[bands[0]]
    if da_ref.rio.crs is None:
        da_ref = da_ref.rio.write_crs(config.CRS)

    s2_crs = da_ref.rio.crs.to_string()
    xs_s, ys_s, names = _transform_sites(sites_gdf, s2_crs)
    cols, rows, in_extent = _pixel_coords(da_ref, xs_s, ys_s)
    n_in = int(in_extent.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: False colour using available bands
    ax = axes[0]
    # Best false-colour with available bands: B08 (NIR)=R, B07 (RedEdge)=G, B04 (Red)=B
    has_fc = all(b in ds for b in ["B08", "B07", "B04"])
    has_tc = all(b in ds for b in ["B04", "B03", "B02"])
    if has_fc:
        r = _norm(ds["B08"].values.astype(np.float32))
        g = _norm(ds["B07"].values.astype(np.float32))
        b = _norm(ds["B04"].values.astype(np.float32))
        rgb = np.nan_to_num(np.stack([r, g, b], axis=-1)).clip(0, 1)
        ax.imshow(rgb, origin="upper")
        ax.set_title("S2 False Colour (B08=R, B07=G, B04=B)\nVegetation=red; bare ground=green-blue")
    elif has_tc:
        r = _norm(ds["B04"].values.astype(np.float32))
        g = _norm(ds["B03"].values.astype(np.float32))
        b = _norm(ds["B02"].values.astype(np.float32))
        rgb = np.nan_to_num(np.stack([r, g, b], axis=-1)).clip(0, 1)
        ax.imshow(rgb, origin="upper")
        ax.set_title("S2 True Colour (B4/B3/B2)")
    else:
        ax.imshow(_norm(ds[bands[0]].values.astype(np.float32)), cmap="gray", origin="upper")
        ax.set_title(f"S2 — {bands[0]} (only band available for display)")

    if n_in > 0:
        ax.scatter(cols[in_extent], rows[in_extent], c="red", s=40, marker="^", zorder=5)
    ax.set_xlabel("Column pixel"); ax.set_ylabel("Row pixel")

    # Panel 2: NDVI
    ax = axes[1]
    if "B08" in ds and "B04" in ds:
        nir = ds["B08"].values.astype(np.float32)
        red = ds["B04"].values.astype(np.float32)
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(denom != 0, (nir - red) / denom, np.nan).clip(-1, 1)
        img = ax.imshow(ndvi, cmap="RdYlGn", origin="upper", vmin=-0.3, vmax=0.9)
        fig.colorbar(img, ax=ax, fraction=0.03, label="NDVI")
        ax.set_title("S2 NDVI (B08/B04)")
    else:
        ax.set_title("NDVI N/A")
    if n_in > 0:
        ax.scatter(cols[in_extent], rows[in_extent], c="red", s=40, marker="^", zorder=5,
                   label=f"{n_in} sites in extent")
        ax.legend(fontsize=8)
    ax.set_xlabel("Column pixel")

    valid_pct = 100 * int(np.sum(np.isfinite(ds[bands[0]].values))) / ds[bands[0]].values.size
    bbox_w84 = _raster_bbox_wgs84(da_ref)
    fig.suptitle(
        f"Sentinel-2 L2A composite | bands: {bands} | shape: {ds[bands[0]].values.shape}\n"
        f"Coverage: lon [{bbox_w84[0]:.2f}°, {bbox_w84[2]:.2f}°]  "
        f"lat [{bbox_w84[1]:.2f}°, {bbox_w84[3]:.2f}°] | "
        f"{valid_pct:.0f}% valid pixels | {n_in}/15 known sites in footprint",
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "02_sentinel2")


# ---------------------------------------------------------------------------
# Plot 3: Sentinel-1
# ---------------------------------------------------------------------------

def plot_sentinel1(sites_gdf) -> None:
    if not config.S1_COMPOSITE_PATH.exists():
        print("  [skip] Sentinel-1 composite not found.")
        return
    print("Plotting Sentinel-1 ...")
    ds = xr.open_dataset(config.S1_COMPOSITE_PATH)

    vv = ds.get("vv") or ds.get("VV")
    vh = ds.get("vh") or ds.get("VH")
    da_ref = vv if vv is not None else vh
    if da_ref is None:
        print("  [skip] No VV/VH found in S1 cache.")
        return
    if da_ref.rio.crs is None:
        da_ref = da_ref.rio.write_crs(config.CRS)

    s1_crs = da_ref.rio.crs.to_string()
    xs_s, ys_s, names = _transform_sites(sites_gdf, s1_crs)
    cols, rows, in_extent = _pixel_coords(da_ref, xs_s, ys_s)
    n_in = int(in_extent.sum())

    n_panels = (vv is not None) + (vh is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    for band, label in [(vv, "VV"), (vh, "VH")]:
        if band is None:
            continue
        arr = band.values.astype(np.float32)
        # Convert to dB if linear
        if np.nanmedian(arr[arr > 0]) < 10:
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(arr > 0, 10 * np.log10(arr), np.nan)

        ax = axes[panel]
        img = ax.imshow(_norm(arr, plow=5, phigh=95), cmap="gray", origin="upper")
        fig.colorbar(img, ax=ax, fraction=0.03, label=f"{label} (normalized dB)")

        # In-extent sites (filled red)
        if in_extent.any():
            ax.scatter(cols[in_extent], rows[in_extent], c="red", s=50, marker="^",
                       zorder=5, label=f"In coverage ({n_in})", edgecolors="darkred", linewidths=0.5)
            for c, r, nm in zip(cols[in_extent], rows[in_extent], np.array(names)[in_extent]):
                ax.annotate(nm, (c, r), xytext=(3, 3), textcoords="offset points",
                            fontsize=5.5, color="red")

        out_n = (~in_extent).sum()
        ax.set_title(f"Sentinel-1 {label} backscatter\n{n_in}/15 sites in footprint, {out_n} outside")
        ax.set_xlabel("Column pixel")
        if panel == 0:
            ax.set_ylabel("Row pixel")
        ax.legend(fontsize=8)
        panel += 1

    bbox_w84 = _raster_bbox_wgs84(da_ref)
    fig.suptitle(
        f"Sentinel-1 GRD composite | shape: {da_ref.values.shape}\n"
        f"Coverage: lon [{bbox_w84[0]:.2f}°, {bbox_w84[2]:.2f}°]  "
        f"lat [{bbox_w84[1]:.2f}°, {bbox_w84[3]:.2f}°]",
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "03_sentinel1")


# ---------------------------------------------------------------------------
# Plot 4: Known sites table
# ---------------------------------------------------------------------------

def plot_known_sites_table(sites_gdf) -> None:
    print("Plotting known sites table ...")
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)

    rows = []
    for _, row in sites_gdf.iterrows():
        lon, lat = t.transform(row.geometry.x, row.geometry.y)
        rows.append([
            row.get("site_name", "?"),
            f"{lat:.4f}°N",
            f"{abs(lon):.4f}°W",
            f"{row.geometry.x:.0f}",
            f"{row.geometry.y:.0f}",
            row.get("source", ""),
        ])

    fig, ax = plt.subplots(figsize=(13, 0.4 * (len(rows) + 2) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Site", "Latitude", "Longitude", "UTM Easting", "UTM Northing", "Source"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5])
    ax.set_title(
        f"Known Maya sites — {len(rows)} total  |  CRS: {config.CRS}  |  "
        f"AOI: {config.AOI_BBOX_WGS84}",
        pad=14, fontsize=10,
    )
    plt.tight_layout()
    _save(fig, "04_known_sites")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nRaw data preview — saving to: {OUT_DIR}\n")
    sites_gdf = get_known_sites(csv_path=config.KNOWN_SITES_CSV, target_crs=config.CRS)
    sites_gdf = filter_sites_to_bbox(sites_gdf)
    print(f"Loaded {len(sites_gdf)} known sites.\n")

    plot_coverage_overview(sites_gdf)
    plot_dem(sites_gdf)
    plot_sentinel2(sites_gdf)
    plot_sentinel1(sites_gdf)
    plot_known_sites_table(sites_gdf)

    print(f"\nAll previews saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
