"""interactive.py — Build a Folium interactive HTML map of the detection results.

Generates a multi-layer interactive map with toggleable raster overlays
(hillshade, NDVI anomaly, SAR anomaly, composite score), known site markers,
and candidate detection markers. Saved to outputs/interactive_map.html.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import geopandas as gpd

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _array_to_png_bytes(
    arr: np.ndarray,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.7,
) -> bytes:
    """Convert a 2D numpy array to PNG bytes using matplotlib.

    Args:
        arr: 2D float array (may contain NaN).
        cmap_name: Matplotlib colormap name.
        vmin: Minimum value for colormap scaling (auto if None).
        vmax: Maximum value for colormap scaling (auto if None).
        alpha: Opacity level for transparent pixels (NaN → transparent).

    Returns:
        PNG image as bytes.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if len(finite) > 0 else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if len(finite) > 0 else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if cmap_name == "site_prob":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "site_prob", ["white", "#FFF5EB", "#FD8D3C", "#D94701", "#7F0000"]
        )
    elif cmap_name == "water_seas":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "water_seas",
            [(0.00, "#ffffff"), (0.01, "#b7e4c7"), (0.25, "#52b788"),
             (0.70, "#2d6a4f"), (1.00, "#1e6091")],
        )
    else:
        cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm(np.where(np.isfinite(arr), arr, np.nan)))
    # Set NaN pixels to transparent
    rgba[~np.isfinite(arr), 3] = 0.0
    rgba[np.isfinite(arr), 3] = alpha

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    from PIL import Image  # pillow
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _da_to_folium_image_overlay(
    da: xr.DataArray,
    cmap_name: str,
    layer_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.65,
):
    """Convert a DataArray to a Folium ImageOverlay.

    Reprojects the DataArray to WGS84 for folium, converts to a PNG,
    and wraps in a folium.ImageOverlay with the given layer name.

    Args:
        da: Input DataArray (any projected CRS).
        cmap_name: Matplotlib colormap name or 'site_prob'.
        layer_name: Name shown in the folium layer control.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        alpha: Raster overlay opacity.

    Returns:
        folium.ImageOverlay object, or None on failure.
    """
    try:
        import folium
        import base64

        # Reproject to WGS84 for folium
        da_wgs = da.rio.reproject("EPSG:4326")
        arr = da_wgs.values.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]

        x = da_wgs.coords["x"].values
        y = da_wgs.coords["y"].values
        bounds = [[float(y.min()), float(x.min())], [float(y.max()), float(x.max())]]

        png_bytes = _array_to_png_bytes(arr, cmap_name, vmin, vmax, alpha)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        url = f"data:image/png;base64,{b64}"

        overlay = folium.raster_layers.ImageOverlay(
            image=url,
            bounds=bounds,
            name=layer_name,
            opacity=1.0,  # Transparency handled in the image itself
            show=False,
        )
        return overlay
    except Exception as exc:
        print(f"[interactive] Could not create image overlay for '{layer_name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Main interactive map builder
# ---------------------------------------------------------------------------

def build_interactive_map(
    hillshade: Optional[xr.DataArray],
    ndvi_anomaly: Optional[xr.DataArray],
    sar_anomaly: Optional[xr.DataArray],
    score: Optional[xr.DataArray],
    sites_gdf: Optional[gpd.GeoDataFrame],
    candidates_gdf: Optional[gpd.GeoDataFrame],
    output_path: Path = config.INTERACTIVE_MAP_PATH,
    jrc_occurrence: Optional[xr.DataArray] = None,
    jrc_seasonality: Optional[xr.DataArray] = None,
    hydrolakes_gdf: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Build and save a Folium interactive HTML map with all detection layers.

    Creates a multi-layer Folium map with:
      - Multi-directional hillshade as a raster overlay
      - NDVI anomaly as a raster overlay
      - SAR anomaly as a raster overlay
      - Composite score as a raster overlay (white → dark red)
      - JRC surface water occurrence as a raster overlay (optional)
      - JRC surface water seasonality as a raster overlay (optional)
      - HydroLAKES lake polygons as a vector layer (optional)
      - Known Maya sites as circle markers with site name/source popups
      - Candidate detections as star markers with score/area/layer popups
      - Layer control widget for independent toggling of all layers

    Args:
        hillshade: Multi-directional hillshade DataArray.
        ndvi_anomaly: NDVI anomaly DataArray.
        sar_anomaly: SAR anomaly DataArray.
        score: Composite score DataArray (values in [0, 1]).
        sites_gdf: GeoDataFrame of known Maya site locations.
        candidates_gdf: GeoDataFrame of candidate detections.
        output_path: Path to save the HTML file.
        jrc_occurrence: JRC surface water occurrence DataArray (0–100%).
        jrc_seasonality: JRC surface water seasonality DataArray (0–12 months).
        hydrolakes_gdf: GeoDataFrame of HydroLAKES lake polygons (WGS84).
    """
    try:
        import folium
    except ImportError:
        print("[interactive] folium is not installed. Cannot generate interactive map.")
        return

    # Determine map centre from config bbox
    west, south, east, north = config.AOI_BBOX_WGS84
    centre_lat = (south + north) / 2
    centre_lon = (west + east) / 2

    print("[interactive] Building Folium map ...")
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
    )

    # Add satellite basemap option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        show=False,
    ).add_to(m)

    # ---------------------------------------------------------------------------
    # Raster overlays
    # ---------------------------------------------------------------------------
    raster_specs = [
        (hillshade, "gray", "Hillshade", None, None, 0.55),
        (ndvi_anomaly, "RdYlGn", "NDVI Anomaly", None, None, 0.65),
        (sar_anomaly, "PuOr", "SAR Anomaly", None, None, 0.65),
        (score, "site_prob", "Composite Score", 0.0, 1.0, 0.70),
    ]

    for da, cmap, name, vmin, vmax, alpha in raster_specs:
        if da is None:
            print(f"[interactive] Skipping '{name}' overlay: data is None.")
            continue
        overlay = _da_to_folium_image_overlay(da, cmap, name, vmin, vmax, alpha)
        if overlay is not None:
            overlay.add_to(m)
            print(f"[interactive] Added '{name}' raster overlay.")

    # ---------------------------------------------------------------------------
    # Known site markers
    # ---------------------------------------------------------------------------
    if sites_gdf is not None and not sites_gdf.empty:
        sites_layer = folium.FeatureGroup(name="Known Maya Sites", show=True)

        try:
            sites_wgs = sites_gdf.to_crs("EPSG:4326")
        except Exception:
            sites_wgs = sites_gdf

        for _, row in sites_wgs.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            site_name = row.get("site_name", "Unknown")
            source = row.get("source", "unknown")
            popup_html = (
                f"<b>{site_name}</b><br>"
                f"Lat: {lat:.4f}, Lon: {lon:.4f}<br>"
                f"Source: {source}"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="blue",
                fill=True,
                fill_color="royalblue",
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=site_name,
            ).add_to(sites_layer)

        sites_layer.add_to(m)
        print(f"[interactive] Added {len(sites_wgs)} known site markers.")

    # ---------------------------------------------------------------------------
    # Candidate detection markers
    # ---------------------------------------------------------------------------
    if candidates_gdf is not None and not candidates_gdf.empty:
        cand_layer = folium.FeatureGroup(name="Candidate Detections", show=True)

        try:
            cand_wgs = candidates_gdf.to_crs("EPSG:4326")
        except Exception:
            cand_wgs = candidates_gdf

        for _, row in cand_wgs.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            mean_score = row.get("mean_score", float("nan"))
            max_score = row.get("max_score", float("nan"))
            area_ha = row.get("area_ha", float("nan"))
            top_layer = row.get("top_layer", "unknown")
            cid = row.get("cluster_id", "?")

            popup_html = (
                f"<b>Candidate #{cid}</b><br>"
                f"Mean score: {mean_score:.3f}<br>"
                f"Max score: {max_score:.3f}<br>"
                f"Area: {area_ha:.2f} ha<br>"
                f"Top layer: {top_layer}"
            )

            # Color candidates by score: yellow (low) → orange → red (high)
            if np.isfinite(mean_score):
                r = int(min(255, 100 + 155 * mean_score))
                g = int(max(0, 200 - 200 * mean_score))
                b = 0
                color = f"#{r:02X}{g:02X}{b:02X}"
            else:
                color = "orange"

            # Use a DivIcon star marker
            star_html = (
                f'<div style="font-size:20px; color:{color}; '
                f'text-shadow: 0 0 3px black;">&#9733;</div>'
            )
            icon = folium.DivIcon(html=star_html, icon_size=(24, 24),
                                  icon_anchor=(12, 12))
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Candidate #{cid} (score={mean_score:.3f})",
            ).add_to(cand_layer)

        cand_layer.add_to(m)
        print(f"[interactive] Added {len(cand_wgs)} candidate markers.")

    # ---------------------------------------------------------------------------
    # Water layer raster overlays
    # ---------------------------------------------------------------------------
    water_raster_specs = [
        (jrc_occurrence,  "Blues",    "JRC Surface Water — Occurrence (%)",  0.0, 100.0, 0.65),
        (jrc_seasonality, "water_seas", "JRC Surface Water — Seasonality (months/yr)", 0.0, 12.0, 0.65),
    ]
    for da, cmap, name, vmin, vmax, alpha in water_raster_specs:
        if da is None:
            continue
        overlay = _da_to_folium_image_overlay(da, cmap, name, vmin, vmax, alpha)
        if overlay is not None:
            overlay.add_to(m)
            print(f"[interactive] Added '{name}' raster overlay.")

    # HydroLAKES vector layer
    if hydrolakes_gdf is not None and not hydrolakes_gdf.empty:
        lakes_layer = folium.FeatureGroup(name="HydroLAKES", show=False)
        try:
            lakes_wgs = hydrolakes_gdf.to_crs("EPSG:4326")
        except Exception:
            lakes_wgs = hydrolakes_gdf

        vol_col = "Vol_total" if "Vol_total" in lakes_wgs.columns else None
        for _, row in lakes_wgs.iterrows():
            name_val = str(row.get("Lake_name", "")).strip()
            area_ha  = row.get("Lake_area", float("nan"))
            vol      = row.get(vol_col, float("nan")) if vol_col else float("nan")
            popup_html = (
                f"<b>{name_val if name_val and name_val != 'nan' else 'Unnamed lake'}</b><br>"
                f"Area: {area_ha:,.1f} km²<br>"
                + (f"Volume: {vol:,.2f} × 10⁶ m³<br>" if np.isfinite(vol) else "")
            )
            import json
            folium.GeoJson(
                data=row.geometry.__geo_interface__,
                style_function=lambda _: {
                    "fillColor": "#1e6091",
                    "color": "#60a5fa",
                    "weight": 0.8,
                    "fillOpacity": 0.55,
                },
                tooltip=name_val if name_val and name_val != "nan" else "Lake",
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(lakes_layer)

        lakes_layer.add_to(m)
        print(f"[interactive] Added HydroLAKES layer ({len(lakes_wgs)} polygons).")

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        print(f"[interactive] Interactive map saved to {output_path}.")
    except Exception as exc:
        print(f"[interactive] Could not save HTML map: {exc}")
