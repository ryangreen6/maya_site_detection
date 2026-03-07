"""maps.py — Generate publication-quality static maps for each analysis layer.

Produces one PNG per layer (hillshade, LRM, NDVI anomaly, SAR anomaly,
lineament density) with known site locations overlaid. Each map includes
a colorbar, north arrow, scale bar, and title. Uses matplotlib with
optional contextily basemap tiles.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import xarray as xr
import geopandas as gpd

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_north_arrow(ax: plt.Axes, x: float = 0.95, y: float = 0.95) -> None:
    """Draw a simple north arrow in axes coordinates.

    Args:
        ax: Matplotlib Axes object.
        x: Horizontal position (axes fraction, right-aligned).
        y: Vertical position (axes fraction, top-aligned).
    """
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.07),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )


def _add_scale_bar(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    bar_fraction: float = 0.2,
    y_frac: float = 0.05,
    x_frac: float = 0.05,
) -> None:
    """Draw a simple linear scale bar in projected map units (metres).

    Args:
        ax: Matplotlib Axes.
        x_min: Left edge of the map extent in projected units.
        x_max: Right edge of the map extent in projected units.
        bar_fraction: Fraction of map width for the scale bar.
        y_frac: Bottom position as a fraction of the axes height.
        x_frac: Left position as a fraction of the axes width.
    """
    try:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        bar_len_m = (x_max - x_min) * bar_fraction
        # Round to nearest 5 km
        bar_len_m = round(bar_len_m / 5000) * 5000
        if bar_len_m == 0:
            return
        bar_len_km = bar_len_m / 1000

        x0 = xlim[0] + (xlim[1] - xlim[0]) * x_frac
        y0 = ylim[0] + (ylim[1] - ylim[0]) * y_frac

        ax.plot(
            [x0, x0 + bar_len_m],
            [y0, y0],
            color="black",
            lw=3,
            solid_capstyle="butt",
        )
        ax.text(
            x0 + bar_len_m / 2,
            y0 + (ylim[1] - ylim[0]) * 0.015,
            f"{bar_len_km:.0f} km",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    except Exception:
        pass  # Scale bar is cosmetic; don't crash the pipeline


def _raster_to_display(da: xr.DataArray) -> tuple[np.ndarray, list]:
    """Extract numpy array and extent [left, right, bottom, top] from a DataArray.

    Args:
        da: Spatial DataArray with 'x' and 'y' coordinates.

    Returns:
        Tuple of (2D numpy array, [xmin, xmax, ymin, ymax]).
    """
    arr = da.values.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    x = da.coords["x"].values
    y = da.coords["y"].values
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    return arr, extent


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a figure to disk with standard export settings.

    Args:
        fig: Matplotlib Figure to save.
        path: Output file path (PNG).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[maps] Saved {path.name} → {path}")


def _overlay_sites(
    ax: plt.Axes,
    sites_gdf: Optional[gpd.GeoDataFrame],
    label: str = "Known sites",
    color: str = "red",
    marker: str = "o",
    size: int = 40,
) -> None:
    """Overlay known site points on a map axes.

    Args:
        ax: Matplotlib Axes.
        sites_gdf: GeoDataFrame of site locations.
        label: Legend label for the site markers.
        color: Marker fill colour.
        marker: Marker symbol.
        size: Marker size.
    """
    if sites_gdf is None or sites_gdf.empty:
        return
    try:
        xs = sites_gdf.geometry.x.values
        ys = sites_gdf.geometry.y.values
        ax.scatter(
            xs,
            ys,
            c=color,
            s=size,
            marker=marker,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
            label=label,
        )
    except Exception as exc:
        print(f"[maps] Could not overlay sites: {exc}")


# ---------------------------------------------------------------------------
# Individual layer maps
# ---------------------------------------------------------------------------

def map_hillshade(
    hillshade: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: Path = config.STATIC_MAPS_DIR / "hillshade.png",
) -> None:
    """Produce a multi-directional hillshade map with known sites overlaid.

    Args:
        hillshade: Hillshade DataArray (values 0–255).
        sites_gdf: Optional GeoDataFrame of known site locations.
        output_path: Path to save the PNG.
    """
    if hillshade is None:
        print("[maps] Hillshade is None; skipping map.")
        return

    arr, extent = _raster_to_display(hillshade)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap="gray", extent=extent, origin="upper", vmin=0, vmax=255)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Hillshade (0–255)")
    _overlay_sites(ax, sites_gdf)
    if sites_gdf is not None and not sites_gdf.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title("Multi-directional Hillshade — Northern Petén, Guatemala", fontsize=13)
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


def map_lrm(
    lrm: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: Path = config.STATIC_MAPS_DIR / "lrm.png",
) -> None:
    """Produce a Local Relief Model map with known sites overlaid.

    Uses a diverging colormap centred at zero to distinguish raised
    features (positive LRM) from depressions (negative LRM).

    Args:
        lrm: LRM DataArray in elevation units.
        sites_gdf: Optional GeoDataFrame of known site locations.
        output_path: Path to save the PNG.
    """
    if lrm is None:
        print("[maps] LRM is None; skipping map.")
        return

    arr, extent = _raster_to_display(lrm)
    vmax = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        arr, cmap="RdBu_r", extent=extent, origin="upper",
        vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="LRM (m)")
    _overlay_sites(ax, sites_gdf)
    if sites_gdf is not None and not sites_gdf.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title("Local Relief Model — Northern Petén, Guatemala", fontsize=13)
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


def map_ndvi_anomaly(
    ndvi_anomaly: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: Path = config.STATIC_MAPS_DIR / "ndvi_anomaly.png",
) -> None:
    """Produce an NDVI anomaly map with a diverging colormap centred at zero.

    Negative values (vegetation stress) appear in warm tones; positive
    values (healthy vegetation) appear in cool tones.

    Args:
        ndvi_anomaly: NDVI anomaly DataArray (z-score).
        sites_gdf: Optional GeoDataFrame of known site locations.
        output_path: Path to save the PNG.
    """
    if ndvi_anomaly is None:
        print("[maps] NDVI anomaly is None; skipping map.")
        return

    arr, extent = _raster_to_display(ndvi_anomaly)
    vmax = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        arr, cmap="RdYlGn", extent=extent, origin="upper",
        vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="NDVI Anomaly (z-score)")
    _overlay_sites(ax, sites_gdf)
    if sites_gdf is not None and not sites_gdf.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title("NDVI Anomaly — Northern Petén, Guatemala", fontsize=13)
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


def map_sar_anomaly(
    sar_anomaly: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: Path = config.STATIC_MAPS_DIR / "sar_anomaly.png",
) -> None:
    """Produce a SAR backscatter anomaly map.

    Args:
        sar_anomaly: SAR anomaly DataArray (z-score).
        sites_gdf: Optional GeoDataFrame of known site locations.
        output_path: Path to save the PNG.
    """
    if sar_anomaly is None:
        print("[maps] SAR anomaly is None; skipping map.")
        return

    arr, extent = _raster_to_display(sar_anomaly)
    vmax = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        arr, cmap="PuOr", extent=extent, origin="upper",
        vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="SAR Anomaly (z-score)")
    _overlay_sites(ax, sites_gdf)
    if sites_gdf is not None and not sites_gdf.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title("SAR Backscatter Anomaly — Northern Petén, Guatemala", fontsize=13)
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


def map_lineament_density(
    lineament_density: xr.DataArray,
    sites_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: Path = config.STATIC_MAPS_DIR / "lineament_density.png",
) -> None:
    """Produce a geometric lineament density map.

    Args:
        lineament_density: Lineament density DataArray (line pixels per window).
        sites_gdf: Optional GeoDataFrame of known site locations.
        output_path: Path to save the PNG.
    """
    if lineament_density is None:
        print("[maps] Lineament density is None; skipping map.")
        return

    arr, extent = _raster_to_display(lineament_density)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap="hot_r", extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Lineament Density")
    _overlay_sites(ax, sites_gdf)
    if sites_gdf is not None and not sites_gdf.empty:
        ax.legend(loc="lower right", fontsize=9)
    _add_north_arrow(ax)
    _add_scale_bar(ax, extent[0], extent[1])
    ax.set_title(
        "Geometric Lineament Density — Northern Petén, Guatemala", fontsize=13
    )
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_all_layer_maps(
    hillshade: Optional[xr.DataArray],
    lrm: Optional[xr.DataArray],
    ndvi_anomaly: Optional[xr.DataArray],
    sar_anomaly: Optional[xr.DataArray],
    lineament_density: Optional[xr.DataArray],
    sites_gdf: Optional[gpd.GeoDataFrame],
    output_dir: Path = config.STATIC_MAPS_DIR,
) -> None:
    """Generate all five individual layer maps and save to output_dir.

    Args:
        hillshade: Multi-directional hillshade DataArray.
        lrm: Local Relief Model DataArray.
        ndvi_anomaly: NDVI anomaly DataArray.
        sar_anomaly: SAR anomaly DataArray.
        lineament_density: Lineament density DataArray.
        sites_gdf: GeoDataFrame of known Maya site locations.
        output_dir: Directory where PNG files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    map_hillshade(hillshade, sites_gdf, output_dir / "hillshade.png")
    map_lrm(lrm, sites_gdf, output_dir / "lrm.png")
    map_ndvi_anomaly(ndvi_anomaly, sites_gdf, output_dir / "ndvi_anomaly.png")
    map_sar_anomaly(sar_anomaly, sites_gdf, output_dir / "sar_anomaly.png")
    map_lineament_density(lineament_density, sites_gdf, output_dir / "lineament_density.png")
    print("[maps] All layer maps generated.")
