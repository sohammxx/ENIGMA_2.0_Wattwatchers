"""
exporter.py
===========
Export detection results to CSV with geographic coordinates.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Optional
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def pixels_to_geo(row: int, col: int, transform) -> tuple:
    """Convert pixel (row, col) to geographic (lon, lat) using rasterio transform."""
    try:
        lon, lat = rasterio.transform.xy(transform, row, col)
        return float(lon), float(lat)
    except Exception:
        return float(col), float(row)


def export_boulders(boulders: List[Dict], output_path: str,
                    transform=None, row_offset: int = 0, col_offset: int = 0):
    """
    Save boulder detections to CSV.
    Adds global pixel coordinates (offset from tile) and geographic coords.
    """
    if not boulders:
        df = pd.DataFrame(columns=[
            'x_pixel_global', 'y_pixel_global', 'radius_px',
            'diameter_m', 'contrast', 'lon', 'lat',
            'near_landslide', 'landslide_age'
        ])
        df.to_csv(output_path, index=False)
        print(f"  No boulders detected. Empty CSV saved: {output_path}")
        return df

    rows = []
    for b in boulders:
        global_col = b['x_pixel'] + col_offset
        global_row = b['y_pixel'] + row_offset

        lon, lat = (None, None)
        if transform is not None:
            lon, lat = pixels_to_geo(global_row, global_col, transform)

        rows.append({
            'x_pixel_global': global_col,
            'y_pixel_global': global_row,
            'radius_px': b['radius_px'],
            'diameter_m': b['diameter_m'],
            'contrast': b.get('contrast', 0),
            'lon': lon,
            'lat': lat,
            'near_landslide': b.get('near_landslide', False),
            'landslide_age': b.get('landslide_age', 'none')
        })

    df = pd.DataFrame(rows)

    # Remove duplicates that may arise from tile overlaps
    df = df.drop_duplicates(subset=['x_pixel_global', 'y_pixel_global'])
    df = df.sort_values('diameter_m', ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} boulders → {output_path}")
    return df


def export_landslides(landslides: List[Dict], output_path: str,
                      transform=None, row_offset: int = 0, col_offset: int = 0):
    """
    Save landslide detections to CSV.
    """
    if not landslides:
        df = pd.DataFrame(columns=[
            'source_col_global', 'source_row_global', 'source_lon', 'source_lat',
            'centroid_col_global', 'centroid_row_global',
            'area_m2', 'major_axis_m', 'minor_axis_m',
            'elongation', 'orientation_deg',
            'gradient_ratio', 'slope_roughness_deg',
            'albedo_contrast', 'freshness_score', 'age_class'
        ])
        df.to_csv(output_path, index=False)
        print(f"  No landslides detected. Empty CSV saved: {output_path}")
        return df

    rows = []
    for ls in landslides:
        src_row = ls['source_row'] + row_offset
        src_col = ls['source_col'] + col_offset
        cen_row = ls['centroid_row'] + row_offset
        cen_col = ls['centroid_col'] + col_offset

        src_lon, src_lat = (None, None)
        if transform is not None:
            src_lon, src_lat = pixels_to_geo(src_row, src_col, transform)

        rows.append({
            'source_col_global': src_col,
            'source_row_global': src_row,
            'source_lon': src_lon,
            'source_lat': src_lat,
            'centroid_col_global': cen_col,
            'centroid_row_global': cen_row,
            'area_m2': ls.get('area_m2', 0),
            'major_axis_m': ls.get('major_axis_m', 0),
            'minor_axis_m': ls.get('minor_axis_m', 0),
            'elongation': ls.get('elongation', 0),
            'orientation_deg': ls.get('orientation_deg', 0),
            'gradient_ratio': ls.get('gradient_ratio', 0),
            'slope_roughness_deg': ls.get('slope_roughness_deg', 0),
            'albedo_contrast': ls.get('albedo_contrast', 0),
            'freshness_score': ls.get('freshness_score', 0),
            'age_class': ls.get('age_class', 'unknown')
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('area_m2', ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} landslides → {output_path}")
    return df


def export_summary(boulders_df: pd.DataFrame, landslides_df: pd.DataFrame,
                   output_path: str):
    """Write a plain-text summary report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "=" * 60,
        "       LUNAR MAPPER — DETECTION SUMMARY",
        "=" * 60,
        "",
        "BOULDERS",
        "-" * 30,
        f"  Total detected         : {len(boulders_df)}",
    ]

    if len(boulders_df) > 0 and 'diameter_m' in boulders_df.columns:
        lines += [
            f"  Diameter range (m)     : {boulders_df['diameter_m'].min():.1f} – {boulders_df['diameter_m'].max():.1f}",
            f"  Median diameter (m)    : {boulders_df['diameter_m'].median():.1f}",
        ]
        if 'near_landslide' in boulders_df.columns:
            near = boulders_df['near_landslide'].sum()
            lines.append(f"  Near landslides        : {near}")

    lines += [
        "",
        "LANDSLIDES",
        "-" * 30,
        f"  Total detected         : {len(landslides_df)}",
    ]

    if len(landslides_df) > 0:
        if 'age_class' in landslides_df.columns:
            recent = (landslides_df['age_class'] == 'recent').sum()
            ancient = (landslides_df['age_class'] == 'ancient').sum()
            lines += [
                f"  Recent                 : {recent}",
                f"  Ancient                : {ancient}",
            ]
        if 'area_m2' in landslides_df.columns:
            lines += [
                f"  Area range (m²)        : {landslides_df['area_m2'].min():.0f} – {landslides_df['area_m2'].max():.0f}",
                f"  Median freshness score : {landslides_df['freshness_score'].median():.3f}",
            ]

    lines += ["", "=" * 60]

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"  Summary saved → {output_path}")
    print(report)
    return report


if __name__ == "__main__":
    print("Exporter module OK.")
