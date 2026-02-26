"""
boulder_detector.py
===================
Detect boulders in OHRC images using:
  1. Circular Hough Transform (cv2.HoughCircles)
  2. Laplacian-of-Gaussian blob detection (skimage)
  3. Merge results, remove duplicates, filter by contrast
"""

import numpy as np
import cv2
from skimage.feature import blob_log
from skimage.filters import gaussian
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


def _local_contrast(img: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Compute contrast: mean inside circle vs surrounding annulus."""
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = img[dist < r]
    outer = img[(dist >= r) & (dist < r * 2.5)]
    if len(inner) == 0 or len(outer) == 0:
        return 0.0
    return abs(float(np.mean(inner)) - float(np.mean(outer)))


def detect_boulders_hough(img_gray: np.ndarray, cfg: dict) -> List[Tuple[int, int, int]]:
    """
    Run Circular Hough Transform.
    Returns list of (cx, cy, radius) in pixels.
    """
    # Blur to reduce noise — important for stable Hough
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=cfg['hough_dp'],
        minDist=cfg['hough_min_dist'],
        param1=cfg['hough_param1'],
        param2=cfg['hough_param2'],
        minRadius=cfg['hough_min_radius'],
        maxRadius=cfg['hough_max_radius']
    )

    if circles is None:
        return []

    result = []
    for (x, y, r) in np.round(circles[0]).astype(int):
        contrast = _local_contrast(img_gray.astype(float), x, y, r)
        if contrast >= cfg['contrast_threshold']:
            result.append((int(x), int(y), int(r)))
    return result


def detect_boulders_blob(img_gray: np.ndarray, cfg: dict) -> List[Tuple[int, int, int]]:
    """
    Run LoG blob detection.
    Returns list of (cx, cy, radius) in pixels.
    """
    # Normalize to 0-1 for blob_log
    img_norm = img_gray.astype(float) / 255.0
    img_norm = gaussian(img_norm, sigma=1)

    blobs = blob_log(
        img_norm,
        min_sigma=cfg['blob_min_sigma'],
        max_sigma=cfg['blob_max_sigma'],
        num_sigma=cfg['blob_num_sigma'],
        threshold=cfg['blob_threshold']
    )

    result = []
    for blob in blobs:
        y, x, sigma = blob
        r = int(np.round(sigma * np.sqrt(2)))
        x, y = int(x), int(y)
        contrast = _local_contrast(img_gray.astype(float), x, y, r)
        if contrast >= cfg['contrast_threshold']:
            result.append((x, y, r))
    return result


def _deduplicate(detections: List[Tuple[int, int, int]],
                 min_dist: int = 15) -> List[Tuple[int, int, int]]:
    """Remove detections that overlap significantly. Keep larger radius."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: -d[2])  # largest first
    kept = []
    for d in detections:
        cx, cy, cr = d
        too_close = False
        for k in kept:
            kx, ky, kr = k
            dist = np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
            if dist < max(cr, kr):
                too_close = True
                break
        if not too_close:
            kept.append(d)
    return kept


def detect_boulders(img: np.ndarray, cfg: dict,
                    pixel_res_m: float = 0.3) -> List[Dict]:
    """
    Main boulder detection function.
    
    Parameters
    ----------
    img : np.ndarray  — uint8 grayscale image tile
    cfg : dict        — boulder_detection section of config
    pixel_res_m : float — meters per pixel
    
    Returns
    -------
    List of dicts with keys: x_pixel, y_pixel, radius_px, diameter_m, contrast
    """
    method = cfg.get('method', 'hybrid')
    detections = []

    if method in ('hough', 'hybrid'):
        hough_dets = detect_boulders_hough(img, cfg)
        detections.extend(hough_dets)

    if method in ('blob', 'hybrid'):
        blob_dets = detect_boulders_blob(img, cfg)
        detections.extend(blob_dets)

    # Deduplicate merged detections
    detections = _deduplicate(detections)

    # Filter by physical size
    min_r = cfg['min_diameter_m'] / (2 * pixel_res_m)
    max_r = cfg['max_diameter_m'] / (2 * pixel_res_m)
    detections = [(x, y, r) for x, y, r in detections if min_r <= r <= max_r]

    # Build output records
    results = []
    for (x, y, r) in detections:
        diameter_m = r * 2 * pixel_res_m
        contrast = _local_contrast(img.astype(float), x, y, r)
        results.append({
            'x_pixel': x,
            'y_pixel': y,
            'radius_px': r,
            'diameter_m': round(diameter_m, 2),
            'contrast': round(contrast, 2)
        })

    return results


if __name__ == "__main__":
    # Quick sanity test
    rng = np.random.default_rng(0)
    test_img = rng.integers(80, 120, (500, 500), dtype=np.uint8)
    # Draw a bright circle
    cv2.circle(test_img, (250, 250), 20, 200, -1)
    cv2.circle(test_img, (100, 100), 8, 210, -1)

    cfg = {
        'method': 'hybrid', 'hough_dp': 1, 'hough_min_dist': 15,
        'hough_param1': 50, 'hough_param2': 25, 'hough_min_radius': 3,
        'hough_max_radius': 80, 'blob_min_sigma': 2, 'blob_max_sigma': 30,
        'blob_num_sigma': 10, 'blob_threshold': 0.05,
        'min_diameter_m': 1.0, 'max_diameter_m': 50.0, 'contrast_threshold': 15
    }

    boulders = detect_boulders(test_img, cfg)
    print(f"Detected {len(boulders)} boulders: {boulders[:3]}")
    print("Boulder detector module OK.")
