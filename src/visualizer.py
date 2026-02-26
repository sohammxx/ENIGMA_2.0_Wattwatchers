"""
visualizer.py
=============
Generate annotated maps overlaying detected boulders and landslides
on the OHRC image using semi-transparent colored overlays.
Detections are blended directly on top of the original image.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works in scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import cv2
import os
from typing import List, Dict, Optional


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale uint8 to RGB for colored overlays."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    return img.copy()


def _blend(canvas: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend overlay onto canvas using alpha transparency.
    Only blends pixels where overlay is non-zero (i.e. where we drew something).
    canvas and overlay are both uint8 RGB arrays of the same shape.
    alpha = 0.0 -> fully transparent, 1.0 -> fully opaque
    """
    mask = np.any(overlay > 0, axis=-1)
    blended = canvas.copy()
    blended[mask] = (
        (1 - alpha) * canvas[mask].astype(float) +
        alpha * overlay[mask].astype(float)
    ).astype(np.uint8)
    return blended


def draw_landslide_fills(img_rgb: np.ndarray, landslides: List[Dict],
                         cfg: dict, fill_alpha: float = 0.45,
                         boundary_alpha: float = 0.9) -> np.ndarray:
    """
    Draw semi-transparent filled regions + solid boundary for each landslide.
    - Interior pixels: blended with region color at fill_alpha opacity
    - Boundary pixels: drawn at boundary_alpha for a crisp visible edge
    - Source point: bright crosshair marker
    """
    from skimage.morphology import erosion, disk

    canvas = img_rgb.copy()
    h, w = canvas.shape[:2]

    recent_color  = np.array(cfg['recent_color'],  dtype=np.uint8)
    ancient_color = np.array(cfg['ancient_color'], dtype=np.uint8)

    for ls in landslides:
        coords = ls.get('coords', np.zeros((0, 2), dtype=int))
        if len(coords) == 0:
            continue

        age   = ls.get('age_class', 'unknown')
        color = recent_color if age == 'recent' else ancient_color

        rows = np.clip(coords[:, 0], 0, h - 1)
        cols = np.clip(coords[:, 1], 0, w - 1)

        # Interior fill (semi-transparent)
        mask = np.zeros((h, w), dtype=bool)
        mask[rows, cols] = True
        fill_overlay = np.zeros_like(canvas)
        fill_overlay[mask] = color
        canvas = _blend(canvas, fill_overlay, fill_alpha)

        # Boundary (more opaque, crisp edge)
        interior = erosion(mask, disk(2))
        boundary = mask & ~interior
        if not np.any(boundary):
            boundary = mask
        boundary_overlay = np.zeros_like(canvas)
        boundary_overlay[boundary] = color
        canvas = _blend(canvas, boundary_overlay, boundary_alpha)

        # Source point crosshair
        sr, sc = int(ls['source_row']), int(ls['source_col'])
        if 0 <= sr < h and 0 <= sc < w:
            src_bgr = tuple(int(v) for v in reversed(cfg['source_color']))
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.drawMarker(canvas_bgr, (sc, sr), src_bgr,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2, line_type=cv2.LINE_AA)
            canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    return canvas


def draw_boulders_overlay(img_rgb: np.ndarray, boulders: List[Dict],
                           cfg: dict, circle_alpha: float = 0.85) -> np.ndarray:
    """
    Draw boulder circles with semi-transparent filled discs + solid ring.
    - Filled disc: blended at low alpha so original image shows through
    - Circle outline: drawn at higher alpha for visibility
    - Center dot: fully solid
    """
    canvas = img_rgb.copy()
    h, w = canvas.shape[:2]
    boulder_color = np.array(cfg['boulder_color'], dtype=np.uint8)

    for b in boulders:
        cx, cy, r = int(b['x_pixel']), int(b['y_pixel']), int(b['radius_px'])

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Semi-transparent disc fill
        disc = dist <= r
        fill_overlay = np.zeros_like(canvas)
        fill_overlay[disc] = boulder_color
        canvas = _blend(canvas, fill_overlay, 0.25)

        # Solid circle outline
        ring = (dist >= r - 1) & (dist <= r + 1)
        ring_overlay = np.zeros_like(canvas)
        ring_overlay[ring] = boulder_color
        canvas = _blend(canvas, ring_overlay, circle_alpha)

        # Center dot
        dot = dist <= 1.5
        if np.any(dot):
            canvas[dot] = [255, 255, 100]

    return canvas


def create_annotated_map(img: np.ndarray, boulders: List[Dict],
                         landslides: List[Dict], cfg: dict,
                         output_path: str,
                         title: str = "Lunar Mapper - Detection Output"):
    """
    Creates a single annotated map with detections blended semi-transparently
    directly on top of the original OHRC image.
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True
    )

    # Convert original grayscale to RGB
    base_rgb = _to_rgb(img)

    # Draw landslide fills first (under boulders)
    annotated = draw_landslide_fills(base_rgb, landslides, cfg,
                                     fill_alpha=0.45, boundary_alpha=0.9)

    # Draw boulder circles on top
    annotated = draw_boulders_overlay(annotated, boulders, cfg,
                                      circle_alpha=0.85)

    figsize = cfg.get('figsize', [20, 20])
    dpi     = cfg.get('dpi', 150)

    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    ax.imshow(annotated, interpolation='nearest')
    ax.axis('off')

    n_recent  = sum(1 for l in landslides if l.get('age_class') == 'recent')
    n_ancient = len(landslides) - n_recent
    stats = (f"Boulders: {len(boulders)}  |  "
             f"Landslides: {len(landslides)}  "
             f"(Recent: {n_recent}, Ancient: {n_ancient})")
    fig.suptitle(f"{title}\n{stats}", fontsize=13, color='white',
                 fontweight='bold', y=0.98)

    legend_handles = [
        mpatches.Patch(color=np.array(cfg['boulder_color'])  / 255,
                       label=f"Boulders (n={len(boulders)})"),
        mpatches.Patch(color=np.array(cfg['recent_color'])   / 255,
                       label=f"Recent Landslides (n={n_recent})"),
        mpatches.Patch(color=np.array(cfg['ancient_color'])  / 255,
                       label=f"Ancient Landslides (n={n_ancient})"),
        mpatches.Patch(color=np.array(cfg['source_color'])   / 255,
                       label="Landslide Source Points"),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.75,
               facecolor='#1a1a1a', labelcolor='white',
               edgecolor='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  Annotated map saved -> {output_path}")


def create_detail_panels(img: np.ndarray, boulders: List[Dict],
                         landslides: List[Dict], cfg: dict,
                         output_path: str):
    """
    Create a detail panel: show the 3 largest boulders and 3 largest landslides
    as zoomed crops with semi-transparent overlay annotations.
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True
    )
    h, w = img.shape
    base_rgb = _to_rgb(img)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')
    fig.suptitle("Detection Details - Top Boulders & Landslides",
                 fontsize=14, color='white', fontweight='bold')

    # Top 3 boulders
    sorted_b = sorted(boulders, key=lambda x: -x['diameter_m'])[:3]
    for i, b in enumerate(sorted_b):
        cx, cy, r = int(b['x_pixel']), int(b['y_pixel']), int(b['radius_px'])
        pad = max(r * 4, 30)
        r1, r2 = max(0, cy - pad), min(h, cy + pad)
        c1, c2 = max(0, cx - pad), min(w, cx + pad)

        crop_base = base_rgb[r1:r2, c1:c2]
        b_local = b.copy()
        b_local['x_pixel'] = cx - c1
        b_local['y_pixel'] = cy - r1
        crop_annotated = draw_boulders_overlay(crop_base, [b_local], cfg,
                                               circle_alpha=0.9)

        axes[0, i].imshow(crop_annotated, interpolation='nearest')
        axes[0, i].set_title(f"Boulder #{i+1}: Diam {b['diameter_m']:.1f}m",
                              fontsize=10, color='white')
        axes[0, i].set_facecolor('black')
        axes[0, i].axis('off')

    for i in range(len(sorted_b), 3):
        axes[0, i].text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=14, color='gray')
        axes[0, i].set_facecolor('black')
        axes[0, i].axis('off')

    # Top 3 landslides
    sorted_ls = sorted(landslides, key=lambda x: -x.get('area_m2', 0))[:3]
    for i, ls in enumerate(sorted_ls):
        coords = ls.get('coords', np.zeros((0, 2), dtype=int))
        if len(coords) > 0:
            rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
            cmin, cmax = int(coords[:, 1].min()), int(coords[:, 1].max())
            pad = 20
            r1, r2 = max(0, rmin - pad), min(h, rmax + pad)
            c1, c2 = max(0, cmin - pad), min(w, cmax + pad)

            crop_base = base_rgb[r1:r2, c1:c2]
            ls_local = ls.copy()
            ls_local['coords']       = coords - np.array([[r1, c1]])
            ls_local['source_row']   = ls['source_row']   - r1
            ls_local['source_col']   = ls['source_col']   - c1
            ls_local['centroid_row'] = ls['centroid_row'] - r1
            ls_local['centroid_col'] = ls['centroid_col'] - c1

            crop_annotated = draw_landslide_fills(
                crop_base, [ls_local], cfg, fill_alpha=0.45, boundary_alpha=0.9
            )

            age   = ls.get('age_class', '?')
            color = (np.array(cfg['recent_color']  if age == 'recent'
                              else cfg['ancient_color']) / 255)
            axes[1, i].imshow(crop_annotated, interpolation='nearest')
            axes[1, i].set_title(
                f"Landslide #{i+1}: {ls.get('area_m2', 0):.0f} m2  [{age}]",
                fontsize=10, color=color
            )
        else:
            axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=14, color='gray')
            axes[1, i].set_facecolor('black')
        axes[1, i].axis('off')

    for i in range(len(sorted_ls), 3):
        axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=14, color='gray')
        axes[1, i].set_facecolor('black')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.get('dpi', 120),
                bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  Detail panels saved -> {output_path}")


if __name__ == "__main__":
    print("Visualizer module OK.")
