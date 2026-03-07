"""known_sites.py — Load and provide known Maya site reference locations.

Reads from an optional CSV file (columns: site_name, latitude, longitude,
source) and falls back to a hardcoded list of well-documented northern
Petén sites if the CSV is not found. Returns a GeoDataFrame in the
project CRS.
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import config

# ---------------------------------------------------------------------------
# Hardcoded fallback: major well-documented northern Petén Maya sites
# Coordinates are WGS84 decimal degrees (approximate centroids).
# Sources: PACUNAM LiDAR Initiative, ASTER, published site reports.
# ---------------------------------------------------------------------------
FALLBACK_SITES: list[dict] = [
    {
        "site_name": "Tikal",
        "latitude": 17.2220,
        "longitude": -89.6237,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Uaxactún",
        "latitude": 17.3936,
        "longitude": -89.6333,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "El Mirador",
        "latitude": 17.7567,
        "longitude": -89.9186,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Nakbé",
        "latitude": 17.6589,
        "longitude": -89.8386,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "La Muralla",
        "latitude": 17.4500,
        "longitude": -89.8500,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Río Azul",
        "latitude": 17.7500,
        "longitude": -89.3833,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Yaxhá",
        "latitude": 17.0750,
        "longitude": -89.4033,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Naranjo",
        "latitude": 17.1708,
        "longitude": -89.3631,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Xultún",
        "latitude": 17.4833,
        "longitude": -89.6000,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "San Bartolo",
        "latitude": 17.5083,
        "longitude": -89.5500,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Calakmul",
        "latitude": 18.1042,
        "longitude": -89.8161,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "La Danta (El Mirador complex)",
        "latitude": 17.7650,
        "longitude": -89.9250,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Tintal",
        "latitude": 17.6700,
        "longitude": -90.0700,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Wakná (Güiro)",
        "latitude": 17.6900,
        "longitude": -89.7200,
        "source": "hardcoded_fallback",
    },
    {
        "site_name": "Dos Lagunas",
        "latitude": 17.6083,
        "longitude": -89.7167,
        "source": "hardcoded_fallback",
    },
]


def load_sites_from_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load known site coordinates from a CSV file.

    Expects columns: site_name, latitude, longitude, source.
    Latitude and longitude must be numeric WGS84 decimal degrees.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        pandas DataFrame with the required columns, or None if loading fails.
    """
    if not csv_path.exists():
        print(f"[known_sites] CSV not found at {csv_path}.")
        return None

    required_cols = {"site_name", "latitude", "longitude", "source"}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[known_sites] Failed to read CSV: {exc}")
        return None

    missing = required_cols - set(df.columns)
    if missing:
        print(f"[known_sites] CSV missing required columns: {missing}")
        return None

    # Coerce coordinate columns to numeric; drop rows with invalid coords
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[known_sites] Dropped {n_dropped} rows with invalid coordinates.")

    if df.empty:
        print("[known_sites] CSV has no valid coordinate rows.")
        return None

    print(f"[known_sites] Loaded {len(df)} sites from {csv_path}.")
    return df[["site_name", "latitude", "longitude", "source"]].reset_index(drop=True)


def dataframe_to_geodataframe(
    df: pd.DataFrame,
    target_crs: str = config.CRS,
) -> gpd.GeoDataFrame:
    """Convert a DataFrame with lat/lon columns to a projected GeoDataFrame.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns in WGS84.
        target_crs: EPSG string for the output CRS.

    Returns:
        GeoDataFrame with Point geometries projected to *target_crs*.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    try:
        gdf = gdf.to_crs(target_crs)
    except Exception as exc:
        print(f"[known_sites] CRS reprojection failed: {exc}. Keeping WGS84.")
    return gdf


def get_known_sites(
    csv_path: Path = config.KNOWN_SITES_CSV,
    target_crs: str = config.CRS,
    use_fallback: bool = True,
) -> gpd.GeoDataFrame:
    """Load known Maya site locations and return a projected GeoDataFrame.

    Attempts to read from *csv_path* first. If the file does not exist or
    fails to load and *use_fallback* is True, uses the hardcoded site list.
    If both sources are available, they are merged with CSV taking precedence
    for duplicate site names.

    Args:
        csv_path: Path to an optional CSV file with site coordinates.
        target_crs: EPSG string for the output CRS.
        use_fallback: If True, include fallback hardcoded sites when CSV is
                      missing or empty.

    Returns:
        GeoDataFrame of known Maya sites projected to *target_crs*.
    """
    csv_df = load_sites_from_csv(csv_path)
    fallback_df = pd.DataFrame(FALLBACK_SITES) if use_fallback else None

    if csv_df is not None and fallback_df is not None:
        # Merge: keep all CSV rows; add fallback sites not already in CSV
        csv_names = set(csv_df["site_name"].str.lower())
        extra = fallback_df[
            ~fallback_df["site_name"].str.lower().isin(csv_names)
        ]
        combined = pd.concat([csv_df, extra], ignore_index=True)
        print(
            f"[known_sites] Combined {len(csv_df)} CSV sites + "
            f"{len(extra)} fallback sites = {len(combined)} total."
        )
    elif csv_df is not None:
        combined = csv_df
    elif fallback_df is not None:
        print(f"[known_sites] Using {len(fallback_df)} hardcoded fallback sites.")
        combined = fallback_df
    else:
        print("[known_sites] No site data available.")
        combined = pd.DataFrame(
            columns=["site_name", "latitude", "longitude", "source"]
        )

    return dataframe_to_geodataframe(combined, target_crs)


def filter_sites_to_bbox(
    sites_gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float] = config.AOI_BBOX_WGS84,
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame of sites to those within the study area bbox.

    Args:
        sites_gdf: GeoDataFrame of known sites (any CRS).
        bbox: Bounding box in WGS84 (west, south, east, north).

    Returns:
        Filtered GeoDataFrame containing only sites inside the bbox.
    """
    west, south, east, north = bbox
    try:
        wgs84 = sites_gdf.to_crs("EPSG:4326")
    except Exception:
        wgs84 = sites_gdf

    mask = (
        (wgs84.geometry.x >= west)
        & (wgs84.geometry.x <= east)
        & (wgs84.geometry.y >= south)
        & (wgs84.geometry.y <= north)
    )
    filtered = sites_gdf[mask].reset_index(drop=True)
    print(
        f"[known_sites] {len(filtered)}/{len(sites_gdf)} sites fall within AOI bbox."
    )
    return filtered
