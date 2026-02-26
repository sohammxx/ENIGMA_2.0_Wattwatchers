"""
landslide_detector.py
=====================
Detect mass wasting (landslide) regions using:
  1. Slope map from DTM (steep areas = candidates)
  2. Texture entropy on OHRC image (rough = disturbed ground)
  3. Morphological cleanup
  4. Region properties filtering (size, elongation)
  5. Source point extraction (highest elevation point)
"""

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from skimage import morphology, measure, filters
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")


def compute_slope(dem: np.ndarray, pixel_res_m: float = 5.0) -> np.ndarray:
    """
    Compute slope in degrees from a DEM.
    Handles flat or nearly-flat DEMs gracefully.
    """
    if np.nanstd(dem) < 0.1:
        # Flat terrain — return zeros
        return np.zeros_like(dem, dtype=np.float32)

    gy, gx = np.gradient(dem, pixel_res_m)
    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2)) * (180.0 / np.pi)
    slope = np.nan_to_num(slope, nan=0.0)
    return slope.astype(np.float32)


def compute_texture_entropy(img: np.ndarray, disk_radius: int = 5) -> np.ndarray:
    """
    Compute local entropy as texture measure.
    High entropy = textually rough = potentially disturbed material.
    """
    # rank.entropy needs uint8 input
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    tex = rank_entropy(img_u8, disk(disk_radius))
    return tex.astype(np.float32)


def compute_roughness(dem: np.ndarray, window: int = 7) -> np.ndarray:
    """Local standard deviation of DEM = roughness."""
    def local_std(values):
        return np.std(values)
    roughness = generic_filter(dem, local_std, size=window)
    return roughness.astype(np.float32)


def build_candidate_mask(slope: np.ndarray, texture: np.ndarray,
                         cfg: dict) -> np.ndarray:
    """
    Combine slope + texture to create binary candidate mask.
    """
    slope_mask = slope > cfg['slope_threshold_deg']
    texture_thresh = np.percentile(texture, cfg['texture_percentile'])
    texture_mask = texture > texture_thresh

    candidate = slope_mask & texture_mask

    # Morphological cleanup
    close_r = cfg['morphology_close_radius']
    open_r = cfg['morphology_open_radius']
    if close_r > 0:
        candidate = morphology.binary_closing(candidate, disk(close_r))
    if open_r > 0:
        candidate = morphology.binary_opening(candidate, disk(open_r))

    candidate = morphology.remove_small_objects(
        candidate, min_size=cfg['min_area_pixels'])
    candidate = morphology.remove_small_holes(
        candidate, area_threshold=cfg['min_area_pixels'] // 2)

    return candidate.astype(bool)


def extract_landslide_regions(candidate: np.ndarray, dem: np.ndarray,
                              img: np.ndarray, cfg: dict,
                              pixel_res_m: float = 0.3) -> List[Dict]:
    """
    Label connected regions in candidate mask.
    Filter by area and elongation.
    Extract source point (max elevation in region).
    
    Returns list of dicts with region properties.
    """
    labels = measure.label(candidate)
    regions = measure.regionprops(labels, intensity_image=dem)

    landslides = []
    for region in regions:
        area_px = region.area
        area_m2 = area_px * (pixel_res_m ** 2)

        # Size filter
        if area_px < cfg['min_area_pixels']:
            continue
        if area_px > cfg['max_area_pixels']:
            continue

        # Elongation filter — landslides are elongated
        if region.minor_axis_length == 0:
            continue
        elongation = region.minor_axis_length / region.major_axis_length
        if elongation > cfg['elongation_ratio']:
            continue  # Too circular — skip (likely a crater)

        # Find source point: pixel with maximum elevation in region
        coords = region.coords  # (N, 2) array of (row, col)
        elevations = dem[coords[:, 0], coords[:, 1]]
        max_idx = np.argmax(elevations)
        source_row, source_col = coords[max_idx]

        # Centroid
        cen_row, cen_col = region.centroid

        # Bounding box
        min_row, min_col, max_row, max_col = region.bbox

        landslides.append({
            'source_row': int(source_row),
            'source_col': int(source_col),
            'centroid_row': float(cen_row),
            'centroid_col': float(cen_col),
            'area_px': int(area_px),
            'area_m2': round(area_m2, 1),
            'elongation': round(elongation, 3),
            'major_axis_m': round(region.major_axis_length * pixel_res_m, 1),
            'minor_axis_m': round(region.minor_axis_length * pixel_res_m, 1),
            'bbox': (min_row, min_col, max_row, max_col),
            'label': int(region.label),
            'orientation_deg': round(float(region.orientation) * 180 / np.pi, 1),
            'coords': coords   # keep for later (age classification, viz)
        })

    return landslides


def detect_landslides(img: np.ndarray, dem: np.ndarray,
                      cfg: dict, pixel_res_m: float = 0.3) -> List[Dict]:
    """
    Full landslide detection pipeline for a single tile.
    
    Parameters
    ----------
    img : np.ndarray     — uint8 grayscale OHRC image tile
    dem : np.ndarray     — float32 DTM elevation tile (same shape as img)
    cfg : dict           — landslide_detection section of config
    pixel_res_m : float  — meters per pixel
    
    Returns list of landslide dicts (see extract_landslide_regions).
    """
    # Reuse pixel_res_m as dtm_res for slope if DEM is same resolution
    slope = compute_slope(dem, pixel_res_m=pixel_res_m)
    texture = compute_texture_entropy(img, disk_radius=cfg['texture_disk_radius'])
    candidate = build_candidate_mask(slope, texture, cfg)
    landslides = extract_landslide_regions(candidate, dem, img, cfg, pixel_res_m)
    return landslides


if __name__ == "__main__":
    from preprocess import generate_synthetic_data

    img, dem, profile, transform = generate_synthetic_data(800)

    cfg = {
        'slope_threshold_deg': 12.0,
        'texture_percentile': 75,
        'texture_disk_radius': 5,
        'min_area_pixels': 300,
        'max_area_pixels': 5_000_000,
        'elongation_ratio': 0.65,
        'morphology_close_radius': 3,
        'morphology_open_radius': 2,
    }

    ls = detect_landslides(img, dem, cfg)
    print(f"Detected {len(ls)} landslide regions")
    for l in ls[:3]:
        print(f"  Source: ({l['source_row']}, {l['source_col']}), "
              f"Area: {l['area_m2']} m², Elongation: {l['elongation']}")
    print("Landslide detector module OK.")
