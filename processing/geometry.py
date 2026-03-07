"""geometry.py — Detect rectilinear geometric patterns in terrain layers.

Applies Canny edge detection and the Hough line transform to the LRM and
multi-directional hillshade to identify straight linear features inconsistent
with natural terrain. Returns a lineament density raster as a DataArray.

The target azimuths (0°, 45°, 90°, 135°) correspond to cardinal and diagonal
orientations consistent with Maya site planning conventions.
"""

from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

import config


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array to uint8 range [0, 255] for edge detection.

    Args:
        arr: Input float array (may contain NaN).

    Returns:
        uint8 array with NaN positions set to 0.
    """
    arr = arr.astype(np.float32)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0.0
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - vmin) / (vmax - vmin)
    return (norm * 255).astype(np.uint8)


def detect_edges(
    layer: xr.DataArray,
    sigma: float = config.CANNY_SIGMA,
) -> Optional[np.ndarray]:
    """Apply Canny edge detection to a raster layer.

    Args:
        layer: Input DataArray (hillshade or LRM) to detect edges in.
        sigma: Gaussian blur sigma controlling edge detection scale.

    Returns:
        Boolean numpy array (True = edge pixel), or None on error.
    """
    if layer is None:
        print("[geometry] Cannot detect edges: layer is None.")
        return None
    try:
        from skimage.feature import canny

        arr = _normalize_to_uint8(layer.values.astype(np.float32))
        # skimage canny expects float image in [0, 1]
        arr_f = arr.astype(np.float64) / 255.0
        edges = canny(arr_f, sigma=sigma)
        return edges
    except ImportError:
        print("[geometry] scikit-image is required for Canny edge detection.")
        return None
    except Exception as exc:
        print(f"[geometry] Edge detection failed: {exc}")
        return None


def detect_hough_lines(
    edges: np.ndarray,
    threshold: int = config.HOUGH_THRESHOLD,
    target_azimuths: list[int] = config.CARDINAL_AZIMUTHS,
    azimuth_tolerance_deg: float = 10.0,
) -> Optional[np.ndarray]:
    """Apply the Hough line transform and return a binary line-pixel mask.

    Detects dominant straight-line features in the edge map and filters to
    those oriented near the target azimuths (cardinal and diagonal).

    Args:
        edges: Boolean edge array from detect_edges().
        threshold: Minimum Hough accumulator votes to accept a line.
        target_azimuths: List of target line orientations in degrees (0–180).
        azimuth_tolerance_deg: Angle tolerance (degrees) around each target.

    Returns:
        Boolean numpy array marking pixels on accepted Hough lines,
        or None on error.
    """
    if edges is None:
        return None
    try:
        from skimage.transform import hough_line, hough_line_peaks

        h, theta, d = hough_line(edges)
        _, angles, dists = hough_line_peaks(h, theta, d, threshold=threshold)

        line_mask = np.zeros(edges.shape, dtype=bool)

        for angle, dist in zip(angles, dists):
            # Convert angle from radians to degrees; map to 0–180
            angle_deg = np.degrees(angle) % 180.0

            # Check if this line is near any target azimuth
            near_target = False
            for target in target_azimuths:
                diff = abs(angle_deg - (target % 180))
                diff = min(diff, 180 - diff)
                if diff <= azimuth_tolerance_deg:
                    near_target = True
                    break

            if not near_target:
                continue

            # Draw line pixels on the mask
            rows, cols = np.ogrid[: edges.shape[0], : edges.shape[1]]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            if abs(sin_a) > abs(cos_a):
                col_indices = np.arange(edges.shape[1])
                row_indices = np.round(
                    (dist - col_indices * cos_a) / (sin_a + 1e-10)
                ).astype(int)
                valid = (row_indices >= 0) & (row_indices < edges.shape[0])
                line_mask[row_indices[valid], col_indices[valid]] = True
            else:
                row_indices = np.arange(edges.shape[0])
                col_indices = np.round(
                    (dist - row_indices * sin_a) / (cos_a + 1e-10)
                ).astype(int)
                valid = (col_indices >= 0) & (col_indices < edges.shape[1])
                line_mask[row_indices[valid], col_indices[valid]] = True

        return line_mask
    except ImportError:
        print("[geometry] scikit-image is required for Hough line transform.")
        return None
    except Exception as exc:
        print(f"[geometry] Hough transform failed: {exc}")
        return None


def compute_lineament_density(
    line_mask: np.ndarray,
    radius: int = config.LINEAMENT_DENSITY_RADIUS,
    template: Optional[xr.DataArray] = None,
) -> Optional[xr.DataArray]:
    """Count detected line pixels per neighborhood to create a density surface.

    High lineament density in a small area indicates a concentration of
    linear features at consistent orientations, which is a strong signal
    of rectilinear architectural layout.

    Args:
        line_mask: Boolean array of accepted Hough line pixels.
        radius: Neighborhood radius in pixels for the counting window.
        template: DataArray to copy spatial coordinates and CRS from.

    Returns:
        Lineament density DataArray (float32), or None on error.
    """
    if line_mask is None:
        return None
    try:
        kernel_size = 2 * radius + 1
        density = uniform_filter(
            line_mask.astype(np.float32), size=kernel_size, mode="reflect"
        )

        if template is not None:
            da = xr.DataArray(
                density,
                coords=template.coords,
                dims=template.dims,
                name="lineament_density",
            )
            if template.rio.crs is not None:
                da = da.rio.write_crs(template.rio.crs)
        else:
            da = xr.DataArray(density, name="lineament_density")

        print(
            f"[geometry] Lineament density computed (radius={radius}). "
            f"Max density: {float(density.max()):.4f}"
        )
        return da
    except Exception as exc:
        print(f"[geometry] Lineament density computation failed: {exc}")
        return None


def compute_geometric_anomaly(
    lrm: Optional[xr.DataArray],
    hillshade: Optional[xr.DataArray],
    canny_sigma: float = config.CANNY_SIGMA,
    hough_threshold: int = config.HOUGH_THRESHOLD,
    lineament_radius: int = config.LINEAMENT_DENSITY_RADIUS,
    target_azimuths: list[int] = config.CARDINAL_AZIMUTHS,
) -> Optional[xr.DataArray]:
    """Full pipeline to compute the geometric lineament anomaly layer.

    Runs edge detection and Hough line analysis on both the LRM and hillshade
    layers, combines the resulting line masks, and computes a lineament
    density surface.

    Args:
        lrm: Local Relief Model DataArray (isolates micro-topography).
        hillshade: Multi-directional hillshade DataArray.
        canny_sigma: Gaussian blur sigma for Canny edge detection.
        hough_threshold: Minimum Hough accumulator votes.
        lineament_radius: Neighborhood radius for the density map.
        target_azimuths: List of target orientations in degrees.

    Returns:
        Lineament density DataArray, or None if no layers are available.
    """
    template = lrm if lrm is not None else hillshade
    if template is None:
        print("[geometry] Both LRM and hillshade are None; cannot compute geometry.")
        return None

    combined_mask: Optional[np.ndarray] = None

    for layer_name, layer in [("LRM", lrm), ("hillshade", hillshade)]:
        if layer is None:
            continue
        edges = detect_edges(layer, sigma=canny_sigma)
        if edges is None:
            continue
        lines = detect_hough_lines(
            edges,
            threshold=hough_threshold,
            target_azimuths=target_azimuths,
        )
        if lines is None:
            continue
        print(
            f"[geometry] {layer_name}: {int(lines.sum())} Hough line pixels detected."
        )
        if combined_mask is None:
            combined_mask = lines.copy()
        else:
            # Only combine if shapes match; reproject otherwise
            if combined_mask.shape == lines.shape:
                combined_mask = combined_mask | lines
            else:
                combined_mask = combined_mask | lines[
                    : combined_mask.shape[0], : combined_mask.shape[1]
                ]

    if combined_mask is None:
        print("[geometry] No line features detected.")
        return None

    return compute_lineament_density(combined_mask, radius=lineament_radius, template=template)
