"""download_gedi.py — Download NASA GEDI L2A ground elevation shots.

GEDI (Global Ecosystem Dynamics Investigation) is a full-waveform lidar
instrument on the ISS that measures the vertical structure of forests.
The L2A product includes ``elev_lowestmode`` — the elevation of the lowest
detected return, which represents the actual ground surface underneath the
jungle canopy.  This is the key advantage over SRTM/Copernicus DEM: GEDI
sees through the tree canopy and can detect the 1–5 m topographic relief
of Maya platforms and pyramids that optical and radar DEMs cannot resolve.

Credentials
-----------
Set these two environment variables before running:

    export EARTHDATA_USERNAME=your_username
    export EARTHDATA_PASSWORD=your_password

Add them to ~/.zshrc (or ~/.bash_profile) and run ``source ~/.zshrc``
so they persist across terminal sessions.  No credentials are stored in code.
"""

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np

import config


def _check_credentials() -> tuple[str, str]:
    """Read NASA Earthdata credentials from environment variables.

    Returns:
        (username, password) tuple.

    Raises:
        RuntimeError: If either variable is missing or empty.
    """
    user = os.environ.get("EARTHDATA_USERNAME", "").strip()
    pwd = os.environ.get("EARTHDATA_PASSWORD", "").strip()
    if not user or not pwd:
        raise RuntimeError(
            "NASA Earthdata credentials not found in environment.\n"
            "  1. Open ~/.zshrc in a text editor.\n"
            "  2. Add these two lines:\n"
            "       export EARTHDATA_USERNAME=your_username\n"
            "       export EARTHDATA_PASSWORD=your_password\n"
            "  3. Save and run:  source ~/.zshrc\n"
            "  4. Re-run the pipeline in the same terminal."
        )
    return user, pwd


def get_gedi_shots(
    bbox: tuple[float, float, float, float] = config.AOI_BBOX_WGS84,
    cache_path: Path = config.GEDI_SHOTS_PATH,
    force_download: bool = False,
    max_granules: int = 30,
    min_sensitivity: float = config.GEDI_MIN_SENSITIVITY,
) -> Optional[Path]:
    """Download GEDI L2A ground elevation shots and cache as CSV.

    Searches for GEDI02_A (version 002) granules covering the AOI using the
    ``earthaccess`` library, downloads each HDF5 granule, extracts quality-
    filtered ground elevation shots from all beams, and writes them to a CSV
    with columns (latitude, longitude, elev_ground).

    Args:
        bbox: WGS84 bounding box (west, south, east, north).
        cache_path: Output CSV path for the extracted shots.
        force_download: Re-download even if cache exists.
        max_granules: Maximum number of granules to download.
        min_sensitivity: Minimum beam sensitivity to accept (0–1).

    Returns:
        Path to the CSV file on success, None on failure.
    """
    if cache_path.exists() and not force_download:
        import csv as _csv
        with open(cache_path) as f:
            n = sum(1 for _ in f) - 1  # subtract header
        print(f"[gedi] Using cached GEDI shots: {cache_path} ({n:,} shots)")
        return cache_path

    try:
        import earthaccess
        import h5py
    except ImportError:
        print(
            "[gedi] Required packages missing. Install with:\n"
            "    pip install earthaccess h5py"
        )
        return None

    # Authenticate using environment variables
    try:
        _check_credentials()
        earthaccess.login(strategy="environment")
        print("[gedi] NASA Earthdata authentication successful.")
    except RuntimeError as exc:
        print(f"[gedi] {exc}")
        return None
    except Exception as exc:
        print(f"[gedi] Authentication failed: {exc}")
        return None

    west, south, east, north = bbox
    print(f"[gedi] Searching GEDI02_A granules for bbox {bbox} ...")

    try:
        results = earthaccess.search_data(
            short_name="GEDI02_A",
            version="002",
            bounding_box=(west, south, east, north),
            count=max_granules,
        )
        n_found = len(results)
        print(f"[gedi] Found {n_found} granules (using up to {max_granules}).")
        if n_found == 0:
            print("[gedi] No granules found for AOI.")
            return None
    except Exception as exc:
        print(f"[gedi] Granule search failed: {exc}")
        return None

    # Download HDF5 files
    download_dir = cache_path.parent / "h5_files"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gedi] Downloading {len(results)} granules to {download_dir} ...")
    try:
        downloaded_paths = earthaccess.download(results, local_path=str(download_dir))
        print(f"[gedi] Downloaded {len(downloaded_paths)} files.")
    except Exception as exc:
        print(f"[gedi] Download failed: {exc}")
        return None

    # Extract shots from each HDF5 file
    all_lats: list[float] = []
    all_lons: list[float] = []
    all_elevs: list[float] = []

    h5_files = list(download_dir.glob("*.h5"))
    print(f"[gedi] Extracting shots from {len(h5_files)} HDF5 files ...")

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                # Iterate over all BEAM groups in the file
                beam_keys = [k for k in f.keys() if k.startswith("BEAM")]
                for beam in beam_keys:
                    grp = f[beam]
                    try:
                        lat = grp["lat_lowestmode"][:]
                        lon = grp["lon_lowestmode"][:]
                        elev = grp["elev_lowestmode"][:]
                        quality = grp["quality_flag"][:]
                        sensitivity = grp["sensitivity"][:]
                    except KeyError:
                        continue  # Some beams may lack these datasets

                    # Quality filter: only accept high-confidence ground shots
                    mask = (
                        (quality == 1)
                        & (sensitivity >= min_sensitivity)
                        & (lat >= south) & (lat <= north)
                        & (lon >= west) & (lon <= east)
                        & np.isfinite(elev)
                        & np.isfinite(lat)
                        & np.isfinite(lon)
                    )
                    n_accepted = int(mask.sum())
                    if n_accepted > 0:
                        all_lats.extend(lat[mask].tolist())
                        all_lons.extend(lon[mask].tolist())
                        all_elevs.extend(elev[mask].tolist())

        except Exception as exc:
            print(f"[gedi] Failed to read {h5_path.name}: {exc}")
            continue

    if not all_lats:
        print("[gedi] No valid GEDI shots extracted after quality filtering.")
        return None

    print(f"[gedi] Extracted {len(all_lats):,} quality ground shots.")

    # Write CSV cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latitude", "longitude", "elev_ground"])
            for lat, lon, elev in zip(all_lats, all_lons, all_elevs):
                writer.writerow([f"{lat:.6f}", f"{lon:.6f}", f"{elev:.3f}"])
        print(f"[gedi] Ground shots saved to {cache_path}")
        return cache_path
    except Exception as exc:
        print(f"[gedi] Failed to write CSV: {exc}")
        return None
