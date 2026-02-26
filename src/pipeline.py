"""
pipeline.py
===========
Master runner for the Lunar Mapper pipeline.

Usage:
  # With real data:
  python src/pipeline.py --ohrc data/raw/ohrc_image.tif --dtm data/raw/dtm.tif

  # Demo mode (synthetic data, no real files needed):
  python src/pipeline.py --demo

  # Custom config:
  python src/pipeline.py --ohrc ... --dtm ... --config config/config.yaml
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# Make sure src/ is in path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocess import (load_image, normalize_image, resample_dtm_to_ohrc,
                        generate_synthetic_data, tile_image)
from boulder_detector import detect_boulders
from landslide_detector import detect_landslides
from age_classifier import classify_all, tag_boulders_near_landslides
from exporter import export_boulders, export_landslides, export_summary
from visualizer import create_annotated_map, create_detail_panels


# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'data': {
        'pixel_resolution_m': 0.3,
        'dtm_resolution_m': 5.0,
        'tile_size': 1000,
        'tile_overlap': 50,
    },
    'boulder_detection': {
        'method': 'hybrid',
        'hough_dp': 1, 'hough_min_dist': 15,
        'hough_param1': 50, 'hough_param2': 25,
        'hough_min_radius': 3, 'hough_max_radius': 80,
        'blob_min_sigma': 2, 'blob_max_sigma': 30,
        'blob_num_sigma': 10, 'blob_threshold': 0.05,
        'min_diameter_m': 1.0, 'max_diameter_m': 50.0,
        'contrast_threshold': 20,
    },
    'landslide_detection': {
        'slope_threshold_deg': 12.0,
        'texture_percentile': 75,
        'texture_disk_radius': 5,
        'min_area_pixels': 300,
        'max_area_pixels': 5_000_000,
        'elongation_ratio': 0.65,
        'morphology_close_radius': 3,
        'morphology_open_radius': 2,
    },
    'age_classification': {
        'fresh_gradient_multiplier': 1.8,
        'fresh_roughness_threshold': 4.0,
        'fresh_score_threshold': 0.55,
    },
    'visualization': {
        'boulder_color': [255, 50, 50],
        'landslide_color': [50, 150, 255],
        'source_color': [0, 255, 100],
        'recent_color': [255, 200, 0],
        'ancient_color': [180, 180, 180],
        'dpi': 150,
        'figsize': [20, 20],
    },
    'output': {
        'save_annotated_png': True,
        'save_geotiff_overlay': False,
        'save_per_tile': False,
    }
}


def load_config(config_path: str) -> dict:
    """Load YAML config, falling back to defaults for missing keys."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        # Deep merge with defaults
        cfg = DEFAULT_CONFIG.copy()
        for section, values in user_cfg.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
        return cfg
    return DEFAULT_CONFIG.copy()


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(ohrc_path: str, dtm_path: str, cfg: dict,
                 output_dir: str = 'outputs', demo_mode: bool = False):
    """
    Full end-to-end pipeline.
    
    Returns (boulders_df, landslides_df)
    """
    t_start = time.time()
    print("\n" + "=" * 60)
    print("   🌕  LUNAR MAPPER — STARTING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load / generate data ─────────────────────────────────────
    print("\n[1/6] Loading data...")
    if demo_mode:
        img, dem, profile, transform = generate_synthetic_data(size=1500)
        pixel_res = cfg['data']['pixel_resolution_m']
    else:
        print(f"  Loading OHRC: {ohrc_path}")
        img_raw, img_profile, transform, crs = load_image(ohrc_path)
        img = normalize_image(img_raw)

        print(f"  Loading DTM:  {dtm_path}")
        dem_raw, dem_profile, dtm_transform, _ = load_image(dtm_path)

        print("  Co-registering DTM to OHRC resolution...")
        dem = resample_dtm_to_ohrc(dem_raw, dem_profile, img_profile)

        profile = img_profile
        pixel_res = cfg['data']['pixel_resolution_m']

    print(f"  Image shape: {img.shape}, DEM shape: {dem.shape}")
    print(f"  Pixel resolution: {pixel_res} m")

    # ── Step 2: Tile ──────────────────────────────────────────────────────
    tile_size = cfg['data']['tile_size']
    overlap = cfg['data']['tile_overlap']
    tiles = tile_image(img, dem, tile_size=tile_size, overlap=overlap)
    print(f"\n[2/6] Split into {len(tiles)} tiles ({tile_size}px, {overlap}px overlap)")

    # ── Step 3: Detect boulders + landslides per tile ─────────────────────
    print("\n[3/6] Running detection across all tiles...")
    all_boulders = []
    all_landslides = []

    for tile_img, tile_dem, row_off, col_off in tqdm(tiles, desc="  Tiles"):
        # Boulder detection
        boulders = detect_boulders(tile_img, cfg['boulder_detection'], pixel_res)
        # Adjust pixel coords to global
        for b in boulders:
            b['x_pixel'] += col_off
            b['y_pixel'] += row_off
        all_boulders.extend(boulders)

        # Landslide detection
        landslides = detect_landslides(tile_img, tile_dem,
                                        cfg['landslide_detection'], pixel_res)
        # Adjust coords to global
        for ls in landslides:
            ls['source_row'] += row_off
            ls['source_col'] += col_off
            ls['centroid_row'] += row_off
            ls['centroid_col'] += col_off
            if ls.get('coords') is not None:
                ls['coords'] = ls['coords'] + np.array([[row_off, col_off]])
            # Store tile-local refs for age classification (still valid as-is
            # because classify_all uses the tile image+dem via coords)
            ls['_tile_img'] = tile_img
            ls['_tile_dem'] = tile_dem
        all_landslides.extend(landslides)

    print(f"  Raw detections → Boulders: {len(all_boulders)}, "
          f"Landslides: {len(all_landslides)}")

    # Deduplicate boulders from overlapping tiles
    # (Simple approach: if two boulders within 15px of each other, keep larger)
    seen = {}
    dedup_boulders = []
    for b in sorted(all_boulders, key=lambda x: -x['diameter_m']):
        key = (b['x_pixel'] // 15, b['y_pixel'] // 15)
        if key not in seen:
            seen[key] = True
            dedup_boulders.append(b)
    all_boulders = dedup_boulders

    # ── Step 4: Age classification ─────────────────────────────────────────
    print("\n[4/6] Classifying landslide ages (recent vs ancient)...")
    classified_landslides = []
    for ls in tqdm(all_landslides, desc="  Classifying"):
        tile_img = ls.pop('_tile_img', img)
        tile_dem = ls.pop('_tile_dem', dem)
        # Adjust coords back to tile-local for classification
        coords_global = ls['coords']
        row_off = coords_global[:, 0].min() - (coords_global[:, 0].min() % tile_size)
        col_off = coords_global[:, 1].min() - (coords_global[:, 1].min() % tile_size)
        ls_local = ls.copy()
        ls_local['coords'] = coords_global - np.array([[row_off, col_off]])
        ls_local['source_row'] = ls['source_row'] - row_off
        ls_local['source_col'] = ls['source_col'] - col_off

        from age_classifier import compute_freshness_score
        classified = compute_freshness_score(
            tile_img, tile_dem, ls_local, cfg['age_classification'], pixel_res
        )
        # Restore global coords
        classified['coords'] = coords_global
        classified['source_row'] = ls['source_row']
        classified['source_col'] = ls['source_col']
        classified['centroid_row'] = ls['centroid_row']
        classified['centroid_col'] = ls['centroid_col']
        classified_landslides.append(classified)

    # Tag boulders near landslides
    all_boulders = tag_boulders_near_landslides(all_boulders, classified_landslides)

    n_recent = sum(1 for l in classified_landslides if l.get('age_class') == 'recent')
    print(f"  → {n_recent} recent, {len(classified_landslides)-n_recent} ancient landslides")

    # ── Step 5: Export CSVs ────────────────────────────────────────────────
    print("\n[5/6] Exporting results...")
    csv_dir = os.path.join(output_dir, 'csv')

    boulder_csv = os.path.join(csv_dir, 'boulders.csv')
    boulder_df = export_boulders(all_boulders, boulder_csv, transform)

    landslide_csv = os.path.join(csv_dir, 'landslides.csv')
    landslide_df = export_landslides(classified_landslides, landslide_csv, transform)

    summary_path = os.path.join(output_dir, 'reports', 'summary.txt')
    export_summary(boulder_df, landslide_df, summary_path)

    # ── Step 6: Visualize ─────────────────────────────────────────────────
    print("\n[6/6] Creating annotated maps...")

    # For large images, use a subsampled version for visualization
    viz_scale = 1
    if img.shape[0] > 3000 or img.shape[1] > 3000:
        viz_scale = 4
        viz_img = img[::viz_scale, ::viz_scale]
    else:
        viz_img = img

    # Scale boulder/landslide coords for visualization
    def scale_boulders(boulders, s):
        scaled = []
        for b in boulders:
            b2 = b.copy()
            b2['x_pixel'] = b['x_pixel'] // s
            b2['y_pixel'] = b['y_pixel'] // s
            b2['radius_px'] = max(1, b['radius_px'] // s)
            scaled.append(b2)
        return scaled

    def scale_landslides(landslides, s):
        scaled = []
        for ls in landslides:
            ls2 = ls.copy()
            ls2['source_row'] = ls['source_row'] // s
            ls2['source_col'] = ls['source_col'] // s
            ls2['centroid_row'] = ls['centroid_row'] // s
            ls2['centroid_col'] = ls['centroid_col'] // s
            if ls.get('coords') is not None:
                ls2['coords'] = ls['coords'] // s
            scaled.append(ls2)
        return scaled

    viz_boulders = scale_boulders(all_boulders, viz_scale)
    viz_landslides = scale_landslides(classified_landslides, viz_scale)

    map_dir = os.path.join(output_dir, 'annotated_maps')
    create_annotated_map(
        viz_img, viz_boulders, viz_landslides,
        cfg['visualization'],
        output_path=os.path.join(map_dir, 'annotated_map.png'),
        title="Lunar Mapper — Chandrayaan-2 OHRC Detection"
    )

    create_detail_panels(
        viz_img, viz_boulders, viz_landslides,
        cfg['visualization'],
        output_path=os.path.join(map_dir, 'detail_panels.png')
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  📁 All outputs in: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}\n")

    return boulder_df, landslide_df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Lunar Mapper — Automated boulder & landslide detection"
    )
    parser.add_argument('--ohrc', type=str, help='Path to OHRC GeoTIFF image')
    parser.add_argument('--dtm', type=str, help='Path to DTM GeoTIFF')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--demo', action='store_true',
                        help='Run on synthetic data (no real files needed)')
    args = parser.parse_args()

    # Validate
    if not args.demo:
        if not args.ohrc or not args.dtm:
            print("ERROR: Provide --ohrc and --dtm paths, or use --demo mode")
            parser.print_help()
            sys.exit(1)
        if not os.path.exists(args.ohrc):
            print(f"ERROR: OHRC file not found: {args.ohrc}")
            sys.exit(1)
        if not os.path.exists(args.dtm):
            print(f"ERROR: DTM file not found: {args.dtm}")
            sys.exit(1)

    cfg = load_config(args.config)

    run_pipeline(
        ohrc_path=args.ohrc,
        dtm_path=args.dtm,
        cfg=cfg,
        output_dir=args.output,
        demo_mode=args.demo
    )


if __name__ == "__main__":
    main()
