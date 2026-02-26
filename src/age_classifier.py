"""
age_classifier.py
=================
Classify detected landslides (and boulders near them) as 'recent' or 'ancient'
using a freshness score derived from:
  1. Edge sharpness along region boundary (high = fresh)
  2. Slope roughness inside region (high = fresh)
  3. Albedo contrast relative to surroundings (high = fresh)

Score is 0–1. Above threshold → 'recent'. Below → 'ancient'.
"""

import numpy as np
import cv2
from skimage import filters, measure
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")


def _boundary_gradient(img: np.ndarray, coords: np.ndarray,
                        boundary_width: int = 3) -> float:
    """
    Compute mean gradient magnitude along region boundary pixels.
    
    Parameters
    ----------
    img : np.ndarray   — full grayscale image
    coords : np.ndarray — (N,2) array of region pixel coordinates (row, col)
    boundary_width     — pixels inward from edge to sample
    
    Returns mean Sobel gradient magnitude along boundary.
    """
    h, w = img.shape
    # Create region mask
    mask = np.zeros((h, w), dtype=bool)
    # Clip coords to image bounds
    rows = np.clip(coords[:, 0], 0, h - 1)
    cols = np.clip(coords[:, 1], 0, w - 1)
    mask[rows, cols] = True

    # Erode to find interior; boundary = mask XOR eroded
    from skimage.morphology import erosion, disk
    interior = erosion(mask, disk(boundary_width))
    boundary_mask = mask & ~interior

    if not np.any(boundary_mask):
        boundary_mask = mask  # fallback

    # Sobel gradient
    img_f = img.astype(float)
    grad = filters.sobel(img_f)

    boundary_vals = grad[boundary_mask]
    surrounding_mask = np.zeros_like(mask)
    # 10px ring around region
    from skimage.morphology import dilation
    dilated = dilation(mask, disk(10))
    surrounding_mask = dilated & ~mask
    surrounding_vals = grad[surrounding_mask] if np.any(surrounding_mask) else grad.flatten()

    mean_boundary = float(np.mean(boundary_vals)) if len(boundary_vals) > 0 else 0.0
    mean_surround = float(np.mean(surrounding_vals)) if len(surrounding_vals) > 0 else 1.0

    # Ratio: how much sharper is boundary vs surroundings
    ratio = mean_boundary / (mean_surround + 1e-6)
    return mean_boundary, ratio


def _slope_roughness(dem: np.ndarray, coords: np.ndarray,
                     pixel_res_m: float = 0.3) -> float:
    """Std dev of slope within region = roughness indicator."""
    from landslide_detector import compute_slope
    slope = compute_slope(dem, pixel_res_m)
    h, w = slope.shape
    rows = np.clip(coords[:, 0], 0, h - 1)
    cols = np.clip(coords[:, 1], 0, w - 1)
    region_slope = slope[rows, cols]
    return float(np.std(region_slope))


def _albedo_contrast(img: np.ndarray, coords: np.ndarray) -> float:
    """
    Mean albedo inside region vs 20px-wide surrounding ring.
    Fresh deposits often have different albedo than surroundings.
    """
    h, w = img.shape
    mask = np.zeros((h, w), dtype=bool)
    # Clip coords to image bounds
    rows = np.clip(coords[:, 0], 0, h - 1)
    cols = np.clip(coords[:, 1], 0, w - 1)
    mask[rows, cols] = True
    from skimage.morphology import dilation, disk
    dilated = dilation(mask, disk(20))
    surround = dilated & ~mask

    inner_mean = float(np.mean(img[mask].astype(float)))
    outer_mean = float(np.mean(img[surround].astype(float))) if np.any(surround) else inner_mean
    return abs(inner_mean - outer_mean)


def compute_freshness_score(img: np.ndarray, dem: np.ndarray,
                            landslide: Dict, cfg: dict,
                            pixel_res_m: float = 0.3) -> Dict:
    """
    Compute a freshness score (0–1) and classify as 'recent' or 'ancient'.
    
    Adds keys to landslide dict:
      gradient_boundary, gradient_ratio, slope_roughness,
      albedo_contrast, freshness_score, age_class
    
    Returns updated landslide dict.
    """
    coords = landslide['coords']

    # 1. Boundary sharpness
    grad_mean, grad_ratio = _boundary_gradient(img, coords)

    # 2. Slope roughness
    roughness = _slope_roughness(dem, coords, pixel_res_m)

    # 3. Albedo contrast
    albedo_c = _albedo_contrast(img, coords)

    # --- Normalize each metric to 0–1 ---
    # Gradient ratio: fresh if > multiplier (e.g., 1.8)
    mult = cfg.get('fresh_gradient_multiplier', 1.8)
    grad_score = min(1.0, grad_ratio / (mult * 2))

    # Roughness: fresh if > threshold (e.g., 4 deg)
    rough_thresh = cfg.get('fresh_roughness_threshold', 4.0)
    rough_score = min(1.0, roughness / (rough_thresh * 2))

    # Albedo contrast: normalize by typical range (0–50 greyscale units)
    albedo_score = min(1.0, albedo_c / 50.0)

    # Combined freshness (weighted)
    freshness = 0.5 * grad_score + 0.3 * rough_score + 0.2 * albedo_score

    threshold = cfg.get('fresh_score_threshold', 0.55)
    age_class = 'recent' if freshness >= threshold else 'ancient'

    ls = landslide.copy()
    ls.update({
        'gradient_boundary': round(grad_mean, 4),
        'gradient_ratio': round(grad_ratio, 4),
        'slope_roughness_deg': round(roughness, 3),
        'albedo_contrast': round(albedo_c, 2),
        'freshness_score': round(freshness, 4),
        'age_class': age_class
    })
    return ls


def classify_all(img: np.ndarray, dem: np.ndarray,
                 landslides: List[Dict], cfg: dict,
                 pixel_res_m: float = 0.3) -> List[Dict]:
    """Run age classification on all detected landslides."""
    classified = []
    for ls in landslides:
        classified.append(
            compute_freshness_score(img, dem, ls, cfg, pixel_res_m)
        )
    return classified


def tag_boulders_near_landslides(boulders: List[Dict],
                                 landslides: List[Dict],
                                 proximity_px: int = 50) -> List[Dict]:
    """
    Tag boulders that fall within proximity_px of any landslide region.
    Adds 'near_landslide' and 'landslide_age' fields to boulder dicts.
    """
    # Build set of all landslide pixels
    ls_pixels = {}  # (row,col) → age_class
    for ls in landslides:
        age = ls.get('age_class', 'unknown')
        for (r, c) in ls.get('coords', []):
            ls_pixels[(r, c)] = age

    updated = []
    for b in boulders:
        bx, by = b['x_pixel'], b['y_pixel']
        nearby = False
        nearby_age = 'none'
        # Check within proximity box (fast approximation)
        for dr in range(-proximity_px, proximity_px, 5):
            for dc in range(-proximity_px, proximity_px, 5):
                key = (by + dr, bx + dc)
                if key in ls_pixels:
                    nearby = True
                    nearby_age = ls_pixels[key]
                    break
            if nearby:
                break
        b2 = b.copy()
        b2['near_landslide'] = nearby
        b2['landslide_age'] = nearby_age
        updated.append(b2)
    return updated


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from preprocess import generate_synthetic_data
    from landslide_detector import detect_landslides

    img, dem, profile, transform = generate_synthetic_data(800)

    ls_cfg = {
        'slope_threshold_deg': 12.0, 'texture_percentile': 75,
        'texture_disk_radius': 5, 'min_area_pixels': 300,
        'max_area_pixels': 5_000_000, 'elongation_ratio': 0.65,
        'morphology_close_radius': 3, 'morphology_open_radius': 2,
    }
    age_cfg = {
        'fresh_gradient_multiplier': 1.8,
        'fresh_roughness_threshold': 4.0,
        'fresh_score_threshold': 0.55
    }

    landslides = detect_landslides(img, dem, ls_cfg)
    classified = classify_all(img, dem, landslides, age_cfg)
    recent = sum(1 for l in classified if l['age_class'] == 'recent')
    print(f"Classified {len(classified)} landslides: {recent} recent, "
          f"{len(classified)-recent} ancient")
    print("Age classifier module OK.")
