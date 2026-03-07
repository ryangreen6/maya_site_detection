"""config.py — Central configuration for the Maya site detection pipeline.

All user-defined parameters live here. Import this module in all other
modules to access paths, thresholds, and weights rather than hardcoding.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Study area — northern Petén, Guatemala
# Bounding box in WGS84 (EPSG:4326): (west, south, east, north)
# Covers the northern Petén department including Tikal, El Mirador, Uaxactún
# ---------------------------------------------------------------------------
AOI_BBOX_WGS84: tuple[float, float, float, float] = (-91.5, 16.5, -89.0, 18.2)  # Extended north to include Calakmul (18.1°N)

# Target projected CRS for all analysis layers (UTM Zone 16N)
CRS: str = "EPSG:32616"

# ---------------------------------------------------------------------------
# API keys — set these before running downloads
# OpenTopography: https://portal.opentopography.org/requestApiKey
# Planetary Computer: no key required (public access)
# ---------------------------------------------------------------------------
OPENTOPOGRAPHY_API_KEY: str = "de526e9767dfb8a5a28db9b54c36fde7"  # Set your key here or via env var OT_API_KEY

# ---------------------------------------------------------------------------
# Sentinel-2 parameters
# ---------------------------------------------------------------------------
S2_DATE_RANGE: tuple[str, str] = ("2022-01-01", "2023-12-31")
S2_CLOUD_THRESHOLD: int = 20  # Maximum scene-level cloud cover percentage (0–100)

# ---------------------------------------------------------------------------
# Sentinel-1 parameters
# ---------------------------------------------------------------------------
S1_DATE_RANGE: tuple[str, str] = ("2022-01-01", "2023-12-31")

# ---------------------------------------------------------------------------
# Directory structure — all relative to the project root
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
DEM_DIR: Path = RAW_DATA_DIR / "dem"
S2_DIR: Path = RAW_DATA_DIR / "sentinel2"
S1_DIR: Path = RAW_DATA_DIR / "sentinel1"
STATIC_MAPS_DIR: Path = OUTPUT_DIR / "maps"

# Derived file paths
DEM_PATH: Path = DEM_DIR / "srtm_merged.tif"
S2_COMPOSITE_PATH: Path = S2_DIR / "s2_composite.nc"
S1_COMPOSITE_PATH: Path = S1_DIR / "s1_composite.nc"
COMPOSITE_SCORE_PATH: Path = OUTPUT_DIR / "composite_score.tif"
OPTIMIZED_SCORE_PATH: Path = OUTPUT_DIR / "composite_score_optimized.tif"
CANDIDATES_GEOJSON_PATH: Path = OUTPUT_DIR / "candidate_sites.geojson"
STATISTICS_CSV_PATH: Path = OUTPUT_DIR / "statistics.csv"
INTERACTIVE_MAP_PATH: Path = OUTPUT_DIR / "interactive_map.html"

# ---------------------------------------------------------------------------
# Known Maya sites CSV
# Expected columns: site_name, latitude, longitude, source
# ---------------------------------------------------------------------------
KNOWN_SITES_CSV: Path = BASE_DIR / "data" / "known_sites.csv"

# ---------------------------------------------------------------------------
# Layer fusion weights
# Keys must match the layer names used in processing/fusion.py.
# Values are relative; they will be normalized to sum to 1 internally.
# ---------------------------------------------------------------------------
FUSION_WEIGHTS: dict[str, float] = {
    "tpi": 0.25,
    "lrm": 0.25,
    "ndvi": 0.20,
    "sar": 0.15,
    "geometric": 0.15,
}

# ---------------------------------------------------------------------------
# Candidate extraction thresholds
# ---------------------------------------------------------------------------
COMPOSITE_SCORE_THRESHOLD: float = 0.65   # Normalized score (0–1)
MIN_CANDIDATE_CLUSTER_SIZE: int = 9       # Minimum cluster size in pixels

# ---------------------------------------------------------------------------
# Terrain processing parameters
# ---------------------------------------------------------------------------
HILLSHADE_AZIMUTHS: list[int] = [0, 45, 90, 135, 180, 225, 270, 315]
HILLSHADE_ALTITUDE: int = 20     # Sun elevation angle in degrees
TPI_SMALL_RADIUS: int = 3        # Small-scale TPI neighborhood radius (pixels)
TPI_LARGE_RADIUS: int = 15       # Large-scale TPI neighborhood radius (pixels)
LRM_GAUSSIAN_SIGMA: float = 50.0 # Gaussian sigma in pixels for LRM low-pass filter

# ---------------------------------------------------------------------------
# Geometry / lineament detection parameters
# ---------------------------------------------------------------------------
CANNY_SIGMA: float = 2.0          # Gaussian blur sigma for Canny edge detection
HOUGH_THRESHOLD: int = 10         # Minimum votes in Hough accumulator
LINEAMENT_DENSITY_RADIUS: int = 15  # Neighborhood radius for density map (pixels)
CARDINAL_AZIMUTHS: list[int] = [0, 45, 90, 135]  # Target orientations in degrees

# ---------------------------------------------------------------------------
# Validation parameters
# ---------------------------------------------------------------------------
N_RANDOM_NEGATIVES: int = 500  # Number of random background points for ROC
RANDOM_SEED: int = 42
