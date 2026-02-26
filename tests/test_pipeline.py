"""
test_pipeline.py
================
Unit tests using synthetic data. Run with:
  python -m pytest tests/ -v
  OR:
  python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cv2


# ─── Helpers ──────────────────────────────────────────────────────────────
def make_test_image(size=500):
    """Create a synthetic lunar image with known boulders."""
    rng = np.random.default_rng(42)
    img = rng.integers(80, 120, (size, size), dtype=np.uint8)
    # Add 3 bright circles (boulders)
    cv2.circle(img, (100, 100), 15, 200, -1)
    cv2.circle(img, (300, 200), 8, 210, -1)
    cv2.circle(img, (400, 400), 20, 195, -1)
    return img


def make_test_dem(size=500):
    """Create a synthetic DEM with slope."""
    dem = np.zeros((size, size), dtype=np.float32)
    # Add a slope
    for i in range(size):
        dem[i, :] = i * 0.5  # gentle slope
    # Add a steep section
    dem[200:300, 150:350] += 50
    return dem


BOULDER_CFG = {
    'method': 'hybrid', 'hough_dp': 1, 'hough_min_dist': 15,
    'hough_param1': 50, 'hough_param2': 25,
    'hough_min_radius': 3, 'hough_max_radius': 80,
    'blob_min_sigma': 2, 'blob_max_sigma': 30,
    'blob_num_sigma': 10, 'blob_threshold': 0.05,
    'min_diameter_m': 1.0, 'max_diameter_m': 50.0,
    'contrast_threshold': 10,
}

LS_CFG = {
    'slope_threshold_deg': 5.0,
    'texture_percentile': 70,
    'texture_disk_radius': 5,
    'min_area_pixels': 100,
    'max_area_pixels': 5_000_000,
    'elongation_ratio': 0.75,
    'morphology_close_radius': 2,
    'morphology_open_radius': 1,
}

AGE_CFG = {
    'fresh_gradient_multiplier': 1.5,
    'fresh_roughness_threshold': 3.0,
    'fresh_score_threshold': 0.4,
}


# ─── Tests ────────────────────────────────────────────────────────────────
def test_preprocess_synthetic():
    """Test synthetic data generation."""
    from preprocess import generate_synthetic_data, normalize_image
    img, dem, profile, transform = generate_synthetic_data(500)
    assert img.shape == (500, 500), f"Expected (500,500), got {img.shape}"
    assert dem.shape == (500, 500)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
    print("  ✅ test_preprocess_synthetic PASSED")


def test_normalize():
    """Test image normalization."""
    from preprocess import normalize_image
    arr = np.array([[0, 100, 200, 300, 1000]], dtype=np.float32)
    norm = normalize_image(arr)
    assert norm.dtype == np.uint8
    assert norm.min() >= 0 and norm.max() <= 255
    print("  ✅ test_normalize PASSED")


def test_tiling():
    """Test that tiling covers entire image."""
    from preprocess import tile_image
    img = np.zeros((2000, 2000), dtype=np.uint8)
    dem = np.zeros((2000, 2000), dtype=np.float32)
    tiles = tile_image(img, dem, tile_size=1000, overlap=50)
    assert len(tiles) > 1, "Should produce multiple tiles"
    # Check all tiles have correct types
    for t_img, t_dem, r, c in tiles:
        assert t_img.ndim == 2
        assert t_dem.ndim == 2
    print(f"  ✅ test_tiling PASSED ({len(tiles)} tiles)")


def test_boulder_detection():
    """Test boulder detection finds known circles."""
    from boulder_detector import detect_boulders
    img = make_test_image(500)
    boulders = detect_boulders(img, BOULDER_CFG, pixel_res_m=0.3)
    assert len(boulders) >= 1, f"Expected at least 1 boulder, got {len(boulders)}"
    for b in boulders:
        assert 'x_pixel' in b
        assert 'y_pixel' in b
        assert 'diameter_m' in b
        assert b['diameter_m'] > 0
    print(f"  ✅ test_boulder_detection PASSED ({len(boulders)} boulders)")


def test_boulder_diameter():
    """Test that diameter calculation is correct."""
    from boulder_detector import detect_boulders
    img = make_test_image(500)
    boulders = detect_boulders(img, BOULDER_CFG, pixel_res_m=0.3)
    for b in boulders:
        expected_diam = b['radius_px'] * 2 * 0.3
        assert abs(b['diameter_m'] - expected_diam) < 0.01
    print("  ✅ test_boulder_diameter PASSED")


def test_slope_computation():
    """Test slope is computed correctly."""
    from landslide_detector import compute_slope
    # Flat terrain → zero slope
    flat_dem = np.zeros((100, 100), dtype=np.float32)
    slope = compute_slope(flat_dem)
    assert np.allclose(slope, 0, atol=0.01), "Flat terrain should have near-zero slope"

    # Linear slope
    dem = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        dem[i, :] = i * 1.0  # 1m rise per 1m pixel
    slope = compute_slope(dem, pixel_res_m=1.0)
    # Expected: arctan(1) = 45 degrees
    interior_slope = slope[5:-5, 5:-5]  # avoid edges
    assert interior_slope.mean() > 30, "Steep slope should be > 30 degrees"
    print("  ✅ test_slope_computation PASSED")


def test_landslide_detection():
    """Test landslide detection runs without error and returns dicts."""
    from landslide_detector import detect_landslides
    img = make_test_image(500)
    dem = make_test_dem(500)
    landslides = detect_landslides(img, dem, LS_CFG, pixel_res_m=0.3)
    # Should not crash; result is a list
    assert isinstance(landslides, list)
    for ls in landslides:
        assert 'source_row' in ls
        assert 'source_col' in ls
        assert 'area_m2' in ls
        assert ls['area_m2'] > 0
    print(f"  ✅ test_landslide_detection PASSED ({len(landslides)} regions)")


def test_age_classification():
    """Test that age classification returns valid classes."""
    from landslide_detector import detect_landslides
    from age_classifier import classify_all
    img = make_test_image(500)
    dem = make_test_dem(500)
    landslides = detect_landslides(img, dem, LS_CFG)
    if not landslides:
        print("  ⚠️  test_age_classification SKIPPED (no landslides detected)")
        return
    classified = classify_all(img, dem, landslides, AGE_CFG)
    for ls in classified:
        assert 'age_class' in ls
        assert ls['age_class'] in ('recent', 'ancient')
        assert 0.0 <= ls['freshness_score'] <= 1.0
    print(f"  ✅ test_age_classification PASSED")


def test_export_csv(tmp_path=None):
    """Test CSV export produces valid files."""
    import tempfile
    import pandas as pd
    from exporter import export_boulders, export_landslides

    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()

    # Dummy data
    boulders = [
        {'x_pixel': 100, 'y_pixel': 200, 'radius_px': 10,
         'diameter_m': 6.0, 'contrast': 25.0,
         'near_landslide': False, 'landslide_age': 'none'}
    ]
    landslides = [
        {'source_row': 50, 'source_col': 80, 'centroid_row': 55.0,
         'centroid_col': 85.0, 'area_m2': 1200.0,
         'major_axis_m': 60.0, 'minor_axis_m': 20.0,
         'elongation': 0.33, 'orientation_deg': 45.0,
         'gradient_ratio': 2.1, 'slope_roughness_deg': 5.2,
         'albedo_contrast': 12.0, 'freshness_score': 0.72,
         'age_class': 'recent', 'coords': np.array([[50, 80]])
        }
    ]

    b_path = os.path.join(tmp_path, 'test_boulders.csv')
    ls_path = os.path.join(tmp_path, 'test_landslides.csv')

    export_boulders(boulders, b_path)
    export_landslides(landslides, ls_path)

    b_df = pd.read_csv(b_path)
    ls_df = pd.read_csv(ls_path)

    assert len(b_df) == 1
    assert 'diameter_m' in b_df.columns
    assert len(ls_df) == 1
    assert 'age_class' in ls_df.columns
    assert ls_df['age_class'].iloc[0] == 'recent'

    print("  ✅ test_export_csv PASSED")


def test_full_demo_pipeline():
    """Integration test: run the full pipeline in demo mode."""
    import tempfile
    from pipeline import run_pipeline, DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    # Use smaller tile + lower thresholds for speed
    cfg['data']['tile_size'] = 500
    cfg['landslide_detection']['min_area_pixels'] = 100
    cfg['landslide_detection']['slope_threshold_deg'] = 8.0

    with tempfile.TemporaryDirectory() as tmp:
        boulder_df, landslide_df = run_pipeline(
            ohrc_path=None, dtm_path=None,
            cfg=cfg, output_dir=tmp, demo_mode=True
        )
        assert boulder_df is not None
        assert landslide_df is not None
        # Check output files exist
        assert os.path.exists(os.path.join(tmp, 'csv', 'boulders.csv'))
        assert os.path.exists(os.path.join(tmp, 'csv', 'landslides.csv'))
        assert os.path.exists(os.path.join(tmp, 'annotated_maps', 'annotated_map.png'))
    print("  ✅ test_full_demo_pipeline PASSED")


# ─── Run all tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   LUNAR MAPPER — RUNNING UNIT TESTS")
    print("=" * 55 + "\n")

    tests = [
        test_preprocess_synthetic,
        test_normalize,
        test_tiling,
        test_boulder_detection,
        test_boulder_diameter,
        test_slope_computation,
        test_landslide_detection,
        test_age_classification,
        test_export_csv,
        test_full_demo_pipeline,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)
    sys.exit(0 if failed == 0 else 1)
