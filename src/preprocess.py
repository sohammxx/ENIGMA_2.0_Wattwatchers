"""
preprocess.py
=============
Load OHRC image + DTM, co-register (resample DTM to OHRC resolution),
and split into manageable tiles for processing.
"""

import numpy as np
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠️  rasterio not installed. Install with: pip install rasterio")
    print("    Real GeoTIFF loading disabled; --demo mode works fine.")

import warnings
warnings.filterwarnings("ignore")


def load_image(path: str):
    """Load a GeoTIFF. Returns (array, profile, transform, crs)."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to load GeoTIFF files. "
                           "Run: pip install rasterio")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to 0-255 uint8, handling NaN/NoData."""
    img = img.copy()
    img[np.isnan(img)] = np.nanmedian(img)
    mn, mx = np.nanpercentile(img, 2), np.nanpercentile(img, 98)
    img = np.clip(img, mn, mx)
    img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return img


def resample_dtm_to_ohrc(dtm, dtm_profile, ohrc_profile):
    """Resample DTM to match OHRC image dimensions."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required for resampling.")
    oh_h = ohrc_profile['height']
    oh_w = ohrc_profile['width']
    if dtm.shape == (oh_h, oh_w):
        return dtm
    resampled = np.zeros((oh_h, oh_w), dtype=np.float32)
    reproject(
        source=dtm,
        destination=resampled,
        src_transform=dtm_profile['transform'],
        src_crs=dtm_profile.get('crs', 'EPSG:4326'),
        dst_transform=ohrc_profile['transform'],
        dst_crs=ohrc_profile.get('crs', 'EPSG:4326'),
        resampling=Resampling.bilinear
    )
    return resampled


def _make_dummy_transform(size, res=0.3):
    """Create a simple affine transform for demo data."""
    class FakeTransform:
        def __init__(self, res): self.res = res
    return FakeTransform(res)


def generate_synthetic_data(size: int = 2000):
    """
    Generate synthetic lunar-like image + DTM for testing without real data.
    Returns (img_uint8, dem_float32, dummy_profile, dummy_transform).
    """
    from scipy.ndimage import gaussian_filter

    print("  [DEMO] Generating synthetic lunar terrain...")

    rng = np.random.default_rng(42)

    # --- DEM ---
    dem = gaussian_filter(rng.random((size, size)) * 200, sigma=80)
    for _ in range(15):
        cx, cy = rng.integers(100, size - 100, 2)
        r = rng.integers(20, 120)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dem[dist < r] -= (r - dist[dist < r]) * 0.8
    dem = dem.astype(np.float32)

    # --- Image ---
    img = normalize_image(dem).astype(np.float32)
    img += rng.normal(0, 8, img.shape)

    # Synthetic boulders
    for _ in range(80):
        bx, by = rng.integers(20, size - 20, 2)
        br = rng.integers(3, 18)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
        img[dist < br] += 60
        img[((xx - bx) > 0) & ((xx - bx) < br + 5) & (np.abs(yy - by) < br)] -= 30

    # Synthetic landslides
    for _ in range(8):
        sx, sy = rng.integers(100, size - 200, 2)
        length = rng.integers(100, 400)
        width = rng.integers(10, 40)
        angle = rng.uniform(30, 150)
        for t in range(length):
            tx = int(sx + t * np.cos(np.radians(angle)))
            ty = int(sy + t * np.sin(np.radians(angle)))
            if 0 <= tx < size and 0 <= ty < size:
                img[max(0, ty - width // 2):min(size, ty + width // 2),
                    max(0, tx - 3):min(size, tx + 3)] += 40
        dem[sy:sy + width, sx:sx + length // 2] += 15

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Build fake rasterio-compatible objects
    class FakeTransform:
        def __init__(self, size, res=0.3):
            self.c = 0; self.f = 0; self.a = res; self.e = -res
            self._size = size; self._res = res
    
    transform = FakeTransform(size)
    profile = {
        'driver': 'GTiff', 'dtype': 'uint8', 'width': size, 'height': size,
        'count': 1, 'crs': 'EPSG:4326', 'transform': transform
    }

    return img, dem, profile, transform


def tile_image(img: np.ndarray, dem: np.ndarray,
               tile_size: int = 1000, overlap: int = 50):
    """Split image + DEM into overlapping tiles."""
    h, w = img.shape
    tiles = []
    r = 0
    while r < h:
        c = 0
        while c < w:
            r_end = min(r + tile_size, h)
            c_end = min(c + tile_size, w)
            tiles.append((img[r:r_end, c:c_end], dem[r:r_end, c:c_end], r, c))
            c += tile_size - overlap
        r += tile_size - overlap
    return tiles


if __name__ == "__main__":
    img, dem, profile, transform = generate_synthetic_data(1000)
    print(f"Synthetic data — Image: {img.shape}, DEM: {dem.shape}")
    print("Preprocess module OK.")
