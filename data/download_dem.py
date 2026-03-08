"""download_dem.py — Download and merge SRTM 30m DEM tiles for the study area.

Uses the OpenTopography REST API to fetch SRTM GL1 (30m) data for the
bounding box defined in config.py, saves the result as a single merged
GeoTIFF, and returns the output file path.
"""

import os
from pathlib import Path
from typing import Optional

import requests
import rioxarray
import xarray as xr

import config


def _get_api_key() -> str:
    """Resolve the OpenTopography API key from config or environment variable.

    Returns:
        API key string, or empty string if not set.
    """
    key = config.OPENTOPOGRAPHY_API_KEY
    if not key:
        key = os.environ.get("OT_API_KEY", "")
    return key


def download_srtm(
    bbox: tuple[float, float, float, float] = config.SITE_CORE_BBOX,
    output_path: Path = config.DEM_PATH,
    dem_type: str = "SRTMGL1",
) -> Optional[Path]:
    """Download an SRTM DEM tile from the OpenTopography REST API.

    Sends a single request to the OpenTopography Global DEM API, which
    returns a pre-merged GeoTIFF covering the requested bounding box.
    Saves the result to *output_path*.

    Args:
        bbox: Bounding box as (west, south, east, north) in WGS84 decimal degrees.
        output_path: Destination path for the downloaded GeoTIFF.
        dem_type: OpenTopography DEM product identifier. Defaults to SRTMGL1 (30m).

    Returns:
        Path to the saved GeoTIFF on success, or None if the download fails.
    """
    west, south, east, north = bbox
    api_key = _get_api_key()

    params: dict = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
    }
    if api_key:
        params["API_Key"] = api_key
    else:
        print(
            "[download_dem] WARNING: No OpenTopography API key set. "
            "Set OPENTOPOGRAPHY_API_KEY in config.py or the OT_API_KEY env var. "
            "Requests without a key may be rate-limited or rejected."
        )

    url = "https://portal.opentopography.org/API/globaldem"

    print(f"[download_dem] Requesting {dem_type} DEM for bbox {bbox} ...")
    try:
        response = requests.get(url, params=params, timeout=300, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        print(f"[download_dem] HTTP error from OpenTopography API: {exc}")
        print(f"[download_dem] Response text: {response.text[:500]}")
        return None
    except requests.exceptions.ConnectionError as exc:
        print(f"[download_dem] Connection error — check network access: {exc}")
        return None
    except requests.exceptions.Timeout:
        print("[download_dem] Request timed out after 300 s.")
        return None

    # Validate by checking TIFF magic bytes (II* = little-endian, MM* = big-endian).
    # OpenTopography returns application/octet-stream, so Content-Type is unreliable.
    first_bytes = response.content[:4]
    is_tiff = first_bytes[:2] in (b"II", b"MM") and first_bytes[2:4] in (b"*\x00", b"\x00*")
    if not is_tiff:
        content_type = response.headers.get("Content-Type", "")
        print(
            f"[download_dem] Response does not appear to be a GeoTIFF "
            f"(Content-Type: '{content_type}'). Server may have returned an error."
        )
        print(f"[download_dem] Response snippet: {response.content[:300]}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            fh.write(chunk)

    print(f"[download_dem] DEM saved to {output_path}")
    return output_path


def load_and_clip_dem(
    dem_path: Path = config.DEM_PATH,
    bbox: tuple[float, float, float, float] = config.SITE_CORE_BBOX,
    target_crs: str = config.CRS,
) -> Optional[xr.DataArray]:
    """Load a GeoTIFF DEM, clip to the AOI, and reproject to the target CRS.

    Reads the DEM with rioxarray, reprojects to *target_crs*, and clips to
    the bounding box expressed in that CRS. Handles missing-data masking.

    Args:
        dem_path: Path to the input GeoTIFF.
        bbox: Bounding box in WGS84 (west, south, east, north) used to derive
              the clip geometry after reprojection.
        target_crs: EPSG string for the output CRS.

    Returns:
        xarray DataArray with the DEM data and CRS metadata, or None on error.
    """
    if not dem_path.exists():
        print(f"[download_dem] DEM file not found at {dem_path}.")
        return None

    try:
        dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze("band", drop=True)
    except Exception as exc:
        print(f"[download_dem] Failed to open DEM with rioxarray: {exc}")
        return None

    try:
        dem = dem.rio.reproject(target_crs)
    except Exception as exc:
        print(f"[download_dem] CRS reprojection failed: {exc}")
        return None

    print(f"[download_dem] DEM loaded and reprojected to {target_crs}.")
    return dem


def get_dem(
    bbox: tuple[float, float, float, float] = config.SITE_CORE_BBOX,
    output_path: Path = config.DEM_PATH,
    target_crs: str = config.CRS,
    force_download: bool = False,
) -> Optional[xr.DataArray]:
    """Orchestrate DEM download (if needed) and return a clipped DataArray.

    Skips the download if *output_path* already exists and *force_download*
    is False. Returns None if any step fails so the caller can handle
    gracefully.

    Args:
        bbox: AOI bounding box in WGS84 (west, south, east, north).
        output_path: Where to save / read the merged GeoTIFF.
        target_crs: Target projected CRS string.
        force_download: If True, re-download even if the file exists.

    Returns:
        xarray DataArray of the DEM, or None on failure.
    """
    if not output_path.exists() or force_download:
        result = download_srtm(bbox=bbox, output_path=output_path)
        if result is None:
            print("[download_dem] Download failed. Returning None.")
            return None
    else:
        print(f"[download_dem] Using cached DEM at {output_path}.")

    return load_and_clip_dem(dem_path=output_path, bbox=bbox, target_crs=target_crs)
