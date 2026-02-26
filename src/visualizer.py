"""
visualizer.py
=============
Generate annotated maps overlaying detected boulders and landslides
on the OHRC image. Saves as PNG (and optionally GeoTIFF).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works in scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrow
import cv2
import os
from typing import List, Dict, Optional


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale uint8 to RGB for colored overlays."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    return img


def draw_boulders_cv2(canvas: np.ndarray, boulders: List[Dict],
                      cfg: dict) -> np.ndarray:
    """Draw boulder circles on OpenCV BGR canvas."""
    recent_color = tuple(reversed(cfg['boulder_color']))  # BGR
    for b in boulders:
        cx, cy, r = b['x_pixel'], b['y_pixel'], b['radius_px']
        cv2.circle(canvas, (cx, cy), r, recent_color, 1, cv2.LINE_AA)
        # Tiny dot at center
        cv2.circle(canvas, (cx, cy), 1, (255, 255, 100), -1)
    return canvas


def draw_landslides_cv2(canvas: np.ndarray, landslides: List[Dict],
                        cfg: dict) -> np.ndarray:
    """Draw landslide contours and source points on OpenCV BGR canvas."""
    ls_color = tuple(reversed(cfg['landslide_color']))
    recent_ls_color = tuple(reversed(cfg['recent_color']))
    ancient_ls_color = tuple(reversed(cfg['ancient_color']))
    src_color = tuple(reversed(cfg['source_color']))

    h, w = canvas.shape[:2]
    for ls in landslides:
        coords = ls.get('coords', np.zeros((0, 2), dtype=int))
        age = ls.get('age_class', 'unknown')

        color = recent_ls_color if age == 'recent' else ancient_ls_color

        # Draw boundary pixels
        if len(coords) > 0:
            from skimage.morphology import erosion, disk
            mask = np.zeros((h, w), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            interior = erosion(mask, disk(2))
            boundary = mask & ~interior
            by, bx = np.where(boundary)
            for (px, py) in zip(bx, by):
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = color

        # Draw source point as crosshair
        sr, sc = ls['source_row'], ls['source_col']
        if 0 <= sr < h and 0 <= sc < w:
            cv2.drawMarker(canvas, (sc, sr), src_color,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=2)
    return canvas


def create_annotated_map(img: np.ndarray, boulders: List[Dict],
                         landslides: List[Dict], cfg: dict,
                         output_path: str, title: str = "Lunar Mapper — Detection Output"):
    """
    Full annotated map using matplotlib for high quality output.
    
    Parameters
    ----------
    img         : uint8 grayscale image
    boulders    : list of boulder dicts
    landslides  : list of (classified) landslide dicts
    cfg         : visualization config section
    output_path : where to save PNG
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Build RGB canvas via OpenCV for pixel-level ops
    canvas = _to_rgb(img).copy()

    # Draw with cv2 first
    canvas = draw_landslides_cv2(canvas, landslides, cfg)
    canvas = draw_boulders_cv2(canvas, boulders, cfg)

    # Now render with matplotlib for quality + legend
    figsize = cfg.get('figsize', [20, 20])
    dpi = cfg.get('dpi', 150)
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] // 2 + 2))

    # Left: original image
    axes[0].imshow(img, cmap='gray', interpolation='nearest')
    axes[0].set_title("Original OHRC Image", fontsize=12)
    axes[0].axis('off')

    # Right: annotated
    axes[1].imshow(canvas, interpolation='nearest')
    axes[1].set_title("Detected Features", fontsize=12)
    axes[1].axis('off')

    # Legend
    legend_handles = [
        mpatches.Patch(color=np.array(cfg['boulder_color']) / 255,
                       label=f'Boulders (n={len(boulders)})'),
        mpatches.Patch(color=np.array(cfg['recent_color']) / 255,
                       label=f'Recent Landslides'),
        mpatches.Patch(color=np.array(cfg['ancient_color']) / 255,
                       label=f'Ancient Landslides'),
        mpatches.Patch(color=np.array(cfg['source_color']) / 255,
                       label='Landslide Source Points'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.85)

    # Stats text
    n_recent = sum(1 for l in landslides if l.get('age_class') == 'recent')
    n_ancient = len(landslides) - n_recent
    stats = (f"Boulders: {len(boulders)} | "
             f"Landslides: {len(landslides)} (Recent: {n_recent}, Ancient: {n_ancient})")
    fig.suptitle(f"{title}\n{stats}", fontsize=11, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  Annotated map saved → {output_path}")


def create_detail_panels(img: np.ndarray, boulders: List[Dict],
                         landslides: List[Dict], cfg: dict,
                         output_path: str):
    """
    Create a detail panel: show the 3 largest boulders and 3 largest landslides
    as zoomed crops with annotations.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    h, w = img.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Detection Details — Top Boulders & Landslides", fontsize=14)

    # Top 3 boulders
    sorted_b = sorted(boulders, key=lambda x: -x['diameter_m'])[:3]
    for i, b in enumerate(sorted_b):
        cx, cy, r = b['x_pixel'], b['y_pixel'], b['radius_px']
        pad = max(r * 4, 30)
        r1, r2 = max(0, cy - pad), min(h, cy + pad)
        c1, c2 = max(0, cx - pad), min(w, cx + pad)
        crop = img[r1:r2, c1:c2]
        axes[0, i].imshow(crop, cmap='gray')
        circle = Circle((cx - c1, cy - r1), r,
                         color=np.array(cfg['boulder_color']) / 255, fill=False, lw=2)
        axes[0, i].add_patch(circle)
        axes[0, i].set_title(f"Boulder #{i+1}: Ø{b['diameter_m']:.1f}m", fontsize=10)
        axes[0, i].axis('off')

    # Fill remaining boulder panels if fewer than 3
    for i in range(len(sorted_b), 3):
        axes[0, i].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        axes[0, i].axis('off')

    # Top 3 landslides
    sorted_ls = sorted(landslides, key=lambda x: -x.get('area_m2', 0))[:3]
    for i, ls in enumerate(sorted_ls):
        coords = ls.get('coords', np.zeros((0, 2), dtype=int))
        if len(coords) > 0:
            rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
            cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
            pad = 20
            r1, r2 = max(0, rmin - pad), min(h, rmax + pad)
            c1, c2 = max(0, cmin - pad), min(w, cmax + pad)
            crop = img[r1:r2, c1:c2]
            axes[1, i].imshow(crop, cmap='gray')
            age = ls.get('age_class', '?')
            color = (np.array(cfg['recent_color'] if age == 'recent'
                              else cfg['ancient_color']) / 255)
            axes[1, i].set_title(
                f"Landslide #{i+1}: {ls.get('area_m2', 0):.0f}m² [{age}]",
                fontsize=10, color=color
            )
        else:
            axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[1, i].axis('off')

    for i in range(len(sorted_ls), 3):
        axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.get('dpi', 120), bbox_inches='tight')
    plt.close(fig)
    print(f"  Detail panels saved → {output_path}")


if __name__ == "__main__":
    print("Visualizer module OK.")
