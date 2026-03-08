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

# Tight bbox enclosing all 15 known sites + ~17 km buffer on each edge.
# All remote-sensing downloads use this bbox so every dataset covers the
# site-dense area (Tintal → Naranjo E–W, Yaxhá → Calakmul N–S).
SITE_CORE_BBOX: tuple[float, float, float, float] = (-90.22, 16.93, -89.21, 18.25)

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

# Dry season (Mar–May) maximises thermal/vegetation stress over buried stone
S2_DRY_SEASON_DATE_RANGE: tuple[str, str] = ("2023-03-01", "2023-05-31")

# ---------------------------------------------------------------------------
# Sentinel-1 parameters
# ---------------------------------------------------------------------------
S1_DATE_RANGE: tuple[str, str] = ("2022-01-01", "2023-12-31")

# ---------------------------------------------------------------------------
# Landsat Collection 2 Level-2 thermal parameters
# ---------------------------------------------------------------------------
LANDSAT_DATE_RANGE: tuple[str, str] = ("2022-01-01", "2023-12-31")
LANDSAT_CLOUD_THRESHOLD: int = 20

# ---------------------------------------------------------------------------
# GEDI L2A ground elevation parameters
# ---------------------------------------------------------------------------
# NASA Earthdata credentials — set EARTHDATA_USERNAME and EARTHDATA_PASSWORD
# in your shell environment (e.g., ~/.zshrc) before running the GEDI step.
GEDI_MIN_SENSITIVITY: float = 0.95   # Minimum beam sensitivity to accept a shot
GEDI_MAX_GAP_M: float = 500.0        # Max interpolation gap in metres (NaN beyond)

# ---------------------------------------------------------------------------
# Directory structure — all relative to the project root
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
DEM_DIR: Path = RAW_DATA_DIR / "dem"
S2_DIR: Path = RAW_DATA_DIR / "sentinel2"
S2_DRY_DIR: Path = RAW_DATA_DIR / "sentinel2_dry"
S1_DIR: Path = RAW_DATA_DIR / "sentinel1"
COPERNICUS_DEM_DIR: Path = RAW_DATA_DIR / "copernicus_dem"
LANDSAT_DIR: Path = RAW_DATA_DIR / "landsat"
GEDI_DIR: Path = RAW_DATA_DIR / "gedi"
STATIC_MAPS_DIR: Path = OUTPUT_DIR / "maps"

# Derived file paths
DEM_PATH: Path = DEM_DIR / "srtm_merged.tif"
COPERNICUS_DEM_PATH: Path = COPERNICUS_DEM_DIR / "cop_dem_merged.tif"
S2_COMPOSITE_PATH: Path = S2_DIR / "s2_composite.nc"
S2_DRY_COMPOSITE_PATH: Path = S2_DRY_DIR / "s2_dry_composite.nc"
S1_COMPOSITE_PATH: Path = S1_DIR / "s1_composite.nc"
LANDSAT_COMPOSITE_PATH: Path = LANDSAT_DIR / "landsat_thermal.nc"
GEDI_SHOTS_PATH: Path = GEDI_DIR / "gedi_shots.csv"
GEDI_RASTER_PATH: Path = GEDI_DIR / "gedi_ground_elev.tif"
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
    # --- Strongly discriminating layers (Cohen d > 0.5 from profiling) ---
    # GEDI canopy-penetrating ground LRM — highest discriminator (d=1.24)
    "gedi_relief": 0.45,
    # NDVI anomaly — sites consistently higher than background (d=0.85)
    "ndvi": 0.35,
    # SAR backscatter anomaly — moderate signal (d=0.60)
    "sar": 0.20,
    # --- Weakly/non-discriminating layers — zeroed out ---
    # tpi, lrm, cop_tpi, ndvi_dry, geometric, east_sightline, thermal
    # all have |Cohen d| < 0.4 and are excluded to avoid diluting the signal.
    "tpi": 0.0,
    "lrm": 0.0,
    "cop_tpi": 0.0,
    "ndvi_dry": 0.0,
    "geometric": 0.0,
    "east_sightline": 0.0,
    "thermal": 0.0,
}

# ---------------------------------------------------------------------------
# Candidate extraction thresholds
# ---------------------------------------------------------------------------
COMPOSITE_SCORE_THRESHOLD: float = 0.55   # Normalized score (0–1)
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
# East sightline parameters
# ---------------------------------------------------------------------------
EAST_SIGHTLINE_HORIZON_KM: float = 1.0  # Distance east over which to compute horizon angle

# ---------------------------------------------------------------------------
# Validation parameters
# ---------------------------------------------------------------------------
N_RANDOM_NEGATIVES: int = 500  # Number of random background points for ROC
RANDOM_SEED: int = 42
