"""main.py — Top-level orchestration script for the Maya site detection pipeline.

Runs the full pipeline from data download through composite scoring,
candidate extraction, validation, statistics, and visualization.

Usage:
    python main.py                    # Full pipeline
    python main.py --dry-run          # Load config and sites only; skip downloads

The --dry-run flag is useful for testing the pipeline structure and
verifying imports without triggering any API calls.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Maya Archaeological Site Detection — multi-source remote sensing pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load config and known sites only. Skip all data downloads, "
            "processing, and visualization. Useful for testing imports."
        ),
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help=(
            "Skip API downloads and use cached data files. "
            "Fails gracefully if caches do not exist."
        ),
    )
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        default=True,
        help="Run weight optimization using known site locations (default: True).",
    )
    parser.add_argument(
        "--no-optimize-weights",
        action="store_false",
        dest="optimize_weights",
        help="Skip weight optimization; use config weights only.",
    )
    return parser.parse_args()


def _step(message: str) -> None:
    """Print a formatted pipeline step progress message.

    Args:
        message: Description of the current pipeline step.
    """
    print(f"\n{'─' * 60}")
    print(f"  {message}")
    print(f"{'─' * 60}")


def _create_output_dirs() -> None:
    """Create all output directories defined in config if they do not exist."""
    import config
    for d in [
        config.RAW_DATA_DIR,
        config.OUTPUT_DIR,
        config.DEM_DIR,
        config.COPERNICUS_DEM_DIR,
        config.S2_DIR,
        config.S2_DRY_DIR,
        config.S1_DIR,
        config.LANDSAT_DIR,
        config.GEDI_DIR,
        config.STATIC_MAPS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the full Maya site detection pipeline.

    Executes all stages in order: download → process → fuse → analyse →
    visualize. Each stage handles None inputs gracefully and prints clear
    status messages so partial runs remain informative.
    """
    args = parse_args()

    # ── Step 1: Configuration ────────────────────────────────────────────────
    _step("Step 1/14 — Loading configuration")
    import config
    print(f"  AOI bbox (WGS84)  : {config.AOI_BBOX_WGS84}")
    print(f"  Target CRS        : {config.CRS}")
    print(f"  Output directory  : {config.OUTPUT_DIR}")
    _create_output_dirs()

    if args.dry_run:
        _step("DRY RUN — Skipping all downloads and processing")
        print("  Imports and configuration verified successfully.")

        # Still load known sites to validate that path
        _step("Step 4/14 — Loading known sites (dry run)")
        from data.known_sites import get_known_sites, filter_sites_to_bbox
        sites_gdf = get_known_sites()
        sites_gdf = filter_sites_to_bbox(sites_gdf)
        print(f"  {len(sites_gdf)} known sites loaded.")
        print("\nDry run complete. Pipeline structure is valid.")
        return

    # ── Step 2: Download DEM ─────────────────────────────────────────────────
    _step("Step 2/14 — Downloading SRTM DEM")
    from data.download_dem import get_dem
    dem = get_dem(
        bbox=config.AOI_BBOX_WGS84,
        output_path=config.DEM_PATH,
        target_crs=config.CRS,
        force_download=not args.skip_downloads and not config.DEM_PATH.exists(),
    )
    if dem is None:
        print("  WARNING: DEM unavailable. Terrain derivatives will be skipped.")

    # ── Step 3: Download Sentinel-2 ──────────────────────────────────────────
    _step("Step 3a/14 — Downloading Sentinel-2 L2A composite")
    from data.download_sentinel2 import get_sentinel2_bands
    s2_bands = get_sentinel2_bands(
        bbox=config.AOI_BBOX_WGS84,
        date_range=config.S2_DATE_RANGE,
        cloud_threshold=config.S2_CLOUD_THRESHOLD,
        target_crs=config.CRS,
        cache_path=config.S2_COMPOSITE_PATH,
        force_download=not args.skip_downloads and not config.S2_COMPOSITE_PATH.exists(),
    )
    if s2_bands is None:
        print("  WARNING: Sentinel-2 data unavailable. Vegetation layers will be skipped.")

    # ── Step 3b: Download Sentinel-1 ─────────────────────────────────────────
    _step("Step 3b/14 — Downloading Sentinel-1 GRD composite")
    from data.download_sentinel1 import get_sentinel1_bands
    s1_bands = get_sentinel1_bands(
        bbox=config.AOI_BBOX_WGS84,
        date_range=config.S1_DATE_RANGE,
        target_crs=config.CRS,
        cache_path=config.S1_COMPOSITE_PATH,
        force_download=not args.skip_downloads and not config.S1_COMPOSITE_PATH.exists(),
    )
    if s1_bands is None:
        print("  WARNING: Sentinel-1 data unavailable. SAR layers will be skipped.")

    # ── Step 3c: Download Copernicus DEM ─────────────────────────────────────
    _step("Step 3c/14 — Downloading Copernicus GLO-30 DEM")
    from data.download_copernicus_dem import get_copernicus_dem
    cop_dem = get_copernicus_dem(
        bbox=config.AOI_BBOX_WGS84,
        output_path=config.COPERNICUS_DEM_PATH,
        target_crs=config.CRS,
        force_download=not args.skip_downloads and not config.COPERNICUS_DEM_PATH.exists(),
    )
    if cop_dem is None:
        print("  WARNING: Copernicus DEM unavailable. cop_tpi layer will be skipped.")

    # ── Step 3d: Download dry-season Sentinel-2 ───────────────────────────────
    _step("Step 3d/14 — Downloading dry-season Sentinel-2 (Mar–May)")
    from data.download_sentinel2 import get_sentinel2_bands
    s2_dry_bands = get_sentinel2_bands(
        bbox=config.AOI_BBOX_WGS84,
        date_range=config.S2_DRY_SEASON_DATE_RANGE,
        cloud_threshold=config.S2_CLOUD_THRESHOLD,
        target_crs=config.CRS,
        cache_path=config.S2_DRY_COMPOSITE_PATH,
        force_download=not args.skip_downloads and not config.S2_DRY_COMPOSITE_PATH.exists(),
    )
    if s2_dry_bands is None:
        print("  WARNING: Dry-season S2 unavailable. ndvi_dry layer will be skipped.")

    # ── Step 3e: Download Landsat thermal ────────────────────────────────────
    _step("Step 3e/14 — Downloading Landsat thermal composite")
    from data.download_landsat import get_landsat_thermal
    landsat_thermal_raw = get_landsat_thermal(
        bbox=config.AOI_BBOX_WGS84,
        date_range=config.LANDSAT_DATE_RANGE,
        cloud_threshold=config.LANDSAT_CLOUD_THRESHOLD,
        target_crs=config.CRS,
        cache_path=config.LANDSAT_COMPOSITE_PATH,
        force_download=not args.skip_downloads and not config.LANDSAT_COMPOSITE_PATH.exists(),
    )
    if landsat_thermal_raw is None:
        print("  WARNING: Landsat thermal unavailable. thermal layer will be skipped.")

    # ── Step 3f: Download GEDI ground elevation shots ─────────────────────────
    _step("Step 3f/14 — Downloading GEDI L2A ground elevation shots")
    from data.download_gedi import get_gedi_shots
    gedi_shots_path = get_gedi_shots(
        bbox=config.AOI_BBOX_WGS84,
        cache_path=config.GEDI_SHOTS_PATH,
        force_download=not args.skip_downloads and not config.GEDI_SHOTS_PATH.exists(),
    )
    if gedi_shots_path is None:
        print("  WARNING: GEDI shots unavailable. gedi_relief layer will be skipped.")

    # ── Step 4: Load known sites ─────────────────────────────────────────────
    _step("Step 4/14 — Loading known Maya site locations")
    from data.known_sites import get_known_sites, filter_sites_to_bbox
    sites_gdf = get_known_sites(csv_path=config.KNOWN_SITES_CSV, target_crs=config.CRS)
    sites_gdf = filter_sites_to_bbox(sites_gdf)
    print(f"  {len(sites_gdf)} known sites available for validation.")

    # ── Step 5: Terrain derivatives ──────────────────────────────────────────
    _step("Step 5/14 — Computing terrain derivatives from DEM")
    from processing.terrain import compute_all_terrain_derivatives
    terrain = compute_all_terrain_derivatives(dem) if dem is not None else {
        "hillshade": None, "slope": None,
        "tpi_small": None, "tpi_large": None,
        "lrm": None, "tri": None,
    }

    hillshade = terrain["hillshade"]
    lrm = terrain["lrm"]
    tpi_large = terrain["tpi_large"]

    # ── Step 5b: Copernicus DEM terrain derivatives ───────────────────────────
    cop_tpi = None
    if cop_dem is not None:
        _step("Step 5b/14 — Computing Copernicus DEM terrain derivatives")
        from processing.terrain import compute_tpi
        cop_tpi = compute_tpi(cop_dem, radius=config.TPI_LARGE_RADIUS)
        if cop_tpi is not None:
            import xarray as xr
            cop_tpi = cop_tpi.rename("cop_tpi")

    # ── Step 6: Vegetation indices and anomalies ─────────────────────────────
    _step("Step 6/14 — Computing vegetation indices and anomalies")
    from processing.vegetation import compute_all_vegetation_layers
    veg_layers = compute_all_vegetation_layers(s2_bands, ndvi_stack=None)

    ndvi = veg_layers["ndvi"]
    ndvi_anomaly = veg_layers["ndvi_anomaly"]

    # ── Step 6b: Dry-season vegetation anomaly ────────────────────────────────
    ndvi_dry = None
    if s2_dry_bands is not None:
        _step("Step 6b/14 — Computing dry-season NDVI anomaly")
        dry_veg = compute_all_vegetation_layers(s2_dry_bands, ndvi_stack=None)
        ndvi_dry = dry_veg.get("ndvi_anomaly")

    # ── Step 7: SAR anomalies ─────────────────────────────────────────────────
    _step("Step 7/14 — Computing SAR backscatter anomalies")
    from processing.sar import compute_all_sar_layers
    sar_layers = compute_all_sar_layers(s1_bands)
    sar_anomaly = sar_layers["sar_anomaly"]

    # ── Step 7b: Landsat thermal anomaly ──────────────────────────────────────
    thermal_anomaly = None
    if landsat_thermal_raw is not None:
        _step("Step 7b/14 — Computing Landsat thermal anomaly")
        from processing.thermal import compute_thermal_anomaly
        thermal_anomaly = compute_thermal_anomaly(landsat_thermal_raw)

    # ── Step 8: Geometric lineament detection ────────────────────────────────
    _step("Step 8/14 — Detecting geometric lineament features")
    from processing.geometry import compute_geometric_anomaly, compute_east_sightline
    lineament_density = compute_geometric_anomaly(lrm=lrm, hillshade=hillshade)
    east_sightline = compute_east_sightline(dem, tpi=tpi_large)

    # ── Step 8b: GEDI ground elevation interpolation and LRM ─────────────────
    gedi_relief = None
    if gedi_shots_path is not None:
        _step("Step 8b/14 — Interpolating GEDI ground shots onto DEM grid")
        from processing.gedi_terrain import interpolate_gedi_to_grid, compute_gedi_lrm
        gedi_elev = interpolate_gedi_to_grid(
            shots_path=gedi_shots_path,
            reference_da=lrm if lrm is not None else tpi_large,
            target_crs=config.CRS,
            raster_path=config.GEDI_RASTER_PATH,
            force_recompute=not args.skip_downloads and not config.GEDI_RASTER_PATH.exists(),
        )
        if gedi_elev is not None:
            gedi_relief = compute_gedi_lrm(gedi_elev)

    # ── Step 8c: Profile known sites across all layers ───────────────────────
    _step("Step 8c/14 — Profiling known sites across all layers")
    from analysis.profile import profile_layer_discrimination
    profile_layer_discrimination(
        layers={
            "tpi":            tpi_large,
            "lrm":            lrm,
            "cop_tpi":        cop_tpi,
            "ndvi_anomaly":   ndvi_anomaly,
            "ndvi_dry":       ndvi_dry,
            "sar_anomaly":    sar_anomaly,
            "thermal":        thermal_anomaly,
            "lineament":      lineament_density,
            "east_sightline": east_sightline,
            "gedi_relief":    gedi_relief,
        },
        sites_gdf=sites_gdf,
    )

    # ── Step 9a: Fuse with config weights ────────────────────────────────────
    _step("Step 9a/14 — Fusing layers with config weights")
    from processing.fusion import fuse_layers
    composite_score = fuse_layers(
        tpi=tpi_large,
        lrm=lrm,
        ndvi_anomaly=ndvi_anomaly,
        sar_anomaly=sar_anomaly,
        geometric=lineament_density,
        east_sightline=east_sightline,
        cop_tpi=cop_tpi,
        ndvi_dry=ndvi_dry,
        thermal=thermal_anomaly,
        gedi_relief=gedi_relief,
        weights=config.FUSION_WEIGHTS,
        output_path=config.COMPOSITE_SCORE_PATH,
    )

    # ── Step 9b: Optimize weights and re-fuse ────────────────────────────────
    optimized_score = composite_score  # Default to config-weight result
    optimized_weights = config.FUSION_WEIGHTS

    if args.optimize_weights and composite_score is not None and not sites_gdf.empty:
        _step("Step 9b/14 — Optimizing fusion weights using known sites")
        from processing.fusion import optimize_weights, normalize_layer

        # Prepare normalized layers for optimization
        norm_layers = {
            "tpi":            normalize_layer(tpi_large),
            "lrm":            normalize_layer(lrm),
            "ndvi":           normalize_layer(ndvi_anomaly, invert=True),
            "sar":            normalize_layer(sar_anomaly),
            "geometric":      normalize_layer(lineament_density),
            "east_sightline": normalize_layer(east_sightline),
            "cop_tpi":        normalize_layer(cop_tpi),
            "ndvi_dry":       normalize_layer(ndvi_dry, invert=True),
            "thermal":        normalize_layer(thermal_anomaly),
            "gedi_relief":    normalize_layer(gedi_relief),
        }

        optimized_weights = optimize_weights(
            layers=norm_layers,
            sites_gdf=sites_gdf,
            initial_weights=config.FUSION_WEIGHTS,
        )

        optimized_score = fuse_layers(
            tpi=tpi_large,
            lrm=lrm,
            ndvi_anomaly=ndvi_anomaly,
            sar_anomaly=sar_anomaly,
            geometric=lineament_density,
            east_sightline=east_sightline,
            cop_tpi=cop_tpi,
            ndvi_dry=ndvi_dry,
            thermal=thermal_anomaly,
            gedi_relief=gedi_relief,
            weights=optimized_weights,
            output_path=config.OPTIMIZED_SCORE_PATH,
        )
        if optimized_score is not None:
            print("  Using optimized-weight score for downstream steps.")
    else:
        _step("Step 9b/14 — Skipping weight optimization")

    # Use the best available score for downstream steps
    best_score = optimized_score if optimized_score is not None else composite_score

    # ── Step 10: Extract candidate sites ─────────────────────────────────────
    _step("Step 10/14 — Extracting candidate site detections")
    from analysis.candidates import extract_candidates

    layer_arrays_for_candidates = {
        "tpi": tpi_large,
        "lrm": lrm,
        "ndvi": ndvi_anomaly,
        "sar": sar_anomaly,
        "geometric": lineament_density,
    }

    candidates_gdf = extract_candidates(
        score=best_score,
        threshold=config.COMPOSITE_SCORE_THRESHOLD,
        min_size=config.MIN_CANDIDATE_CLUSTER_SIZE,
        layer_arrays=layer_arrays_for_candidates,
        output_path=config.CANDIDATES_GEOJSON_PATH,
    )
    n_cand = len(candidates_gdf) if candidates_gdf is not None else 0
    print(f"  {n_cand} candidate sites extracted.")

    # ── Step 11: Validation ───────────────────────────────────────────────────
    _step("Step 11/14 — Validating against known site locations")
    from analysis.validate import run_validation, extract_site_scores, sample_background_scores

    validation_report, roc_data = run_validation(
        score=best_score,
        sites_gdf=sites_gdf,
        threshold=config.COMPOSITE_SCORE_THRESHOLD,
        n_background=config.N_RANDOM_NEGATIVES,
        random_seed=config.RANDOM_SEED,
    )

    # Collect arrays for scatter plots
    site_scores = extract_site_scores(best_score, sites_gdf) \
        if best_score is not None and not sites_gdf.empty else None
    bg_scores = sample_background_scores(best_score, sites_gdf) \
        if best_score is not None and not sites_gdf.empty else None

    # ── Step 12: Summary statistics ──────────────────────────────────────────
    _step("Step 12/14 — Computing summary statistics")
    from analysis.statistics import compute_all_statistics
    stats_df = compute_all_statistics(
        score=best_score,
        sites_gdf=sites_gdf,
        candidates_gdf=candidates_gdf,
        n_background=config.N_RANDOM_NEGATIVES,
        random_seed=config.RANDOM_SEED,
        output_path=config.STATISTICS_CSV_PATH,
    )

    # ── Step 13: Static layer maps ────────────────────────────────────────────
    _step("Step 13a/14 — Generating static layer maps")
    from visualize.maps import generate_all_layer_maps
    generate_all_layer_maps(
        hillshade=hillshade,
        lrm=lrm,
        ndvi_anomaly=ndvi_anomaly,
        sar_anomaly=sar_anomaly,
        lineament_density=lineament_density,
        sites_gdf=sites_gdf,
        output_dir=config.STATIC_MAPS_DIR,
    )

    # ── Step 13b: Composite figures ───────────────────────────────────────────
    _step("Step 13b/14 — Generating composite and diagnostic figures")
    from visualize.composite import generate_all_composite_figures
    generate_all_composite_figures(
        score=best_score,
        hillshade=hillshade,
        lrm=lrm,
        ndvi_anomaly=ndvi_anomaly,
        sar_anomaly=sar_anomaly,
        geometric=lineament_density,
        sites_gdf=sites_gdf,
        candidates_gdf=candidates_gdf,
        site_scores=site_scores,
        bg_scores=bg_scores,
        roc_data=roc_data,
        output_dir=config.STATIC_MAPS_DIR,
    )

    # ── Step 14: Interactive HTML map ─────────────────────────────────────────
    _step("Step 14/14 — Generating interactive HTML map")
    from visualize.interactive import build_interactive_map
    build_interactive_map(
        hillshade=hillshade,
        ndvi_anomaly=ndvi_anomaly,
        sar_anomaly=sar_anomaly,
        score=best_score,
        sites_gdf=sites_gdf,
        candidates_gdf=candidates_gdf,
        output_path=config.INTERACTIVE_MAP_PATH,
    )

    # ── Pipeline complete ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Composite score     : {config.COMPOSITE_SCORE_PATH}")
    print(f"  Candidate sites     : {config.CANDIDATES_GEOJSON_PATH}")
    print(f"  Statistics          : {config.STATISTICS_CSV_PATH}")
    print(f"  Static maps         : {config.STATIC_MAPS_DIR}")
    print(f"  Interactive map     : {config.INTERACTIVE_MAP_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
