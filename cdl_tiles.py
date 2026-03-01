"""Local CDL tile renderer.

Renders 256×256 PNG tiles from the 2024 USDA CDL GeoTIFF (EPSG:5070)
into Web Mercator (EPSG:3857) XYZ tile coordinates for Leaflet.

Flask endpoint: GET /api/cdl/tile/<z>/<x>/<y>.png?code=<int|all|Total>

  code=all     → full CDL with standard NASS colors (all categories)
  code=Total   → only cropland codes highlighted
  code=<int>   → only that single CDL code; all other pixels transparent
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform as warp_transform
from rasterio.transform import from_bounds
from rasterio.windows import Window

logger = logging.getLogger(__name__)

CDL_PATH       = Path("/Volumes/T7/Weather_Models/data/cdl_2024_10m/2024_10m_cdls.tif")
TILE_CACHE_DIR = Path("/Volumes/T7/Weather_Models/data/cdl_tiles")
TILE_SIZE      = 256
HALF_WORLD     = 20037508.3427892  # half Earth circumference in metres (EPSG:3857)

# ── Official NASS CDL colors (R, G, B) ────────────────────────────────────────
CDL_RGB: dict[int, tuple[int, int, int]] = {
    1:   (255, 211,   0),  # Corn
    2:   (255,  37,  37),  # Cotton
    3:   (255, 158,   9),  # Rice
    4:   (255, 127, 127),  # Sorghum
    5:   ( 38, 112,   0),  # Soybeans
    6:   (255, 255,   0),  # Sunflower
    10:  (255, 211, 127),  # Peanuts
    11:  (  0, 175,  76),  # Tobacco
    12:  (255, 211,   0),  # Sweet Corn
    13:  (255, 211,   0),  # Pop/Orn Corn
    14:  (  0, 168, 232),  # Mint
    21:  (137,  96,  84),  # Barley
    22:  (214, 158, 188),  # Durum Wheat
    23:  (214, 158, 188),  # Spring Wheat
    24:  (165, 112,   0),  # Winter Wheat
    25:  (209, 180, 139),  # Other Small Grains
    26:  (165, 112,   0),  # Dbl WinWht/Soybeans
    27:  (209, 255, 115),  # Rye
    28:  (115, 178, 255),  # Oats
    29:  (245, 202, 122),  # Millet
    30:  (138, 113,  57),  # Speltz
    31:  (210, 232,   0),  # Canola
    32:  (219, 227, 255),  # Flaxseed
    33:  (255, 255,   0),  # Safflower
    34:  (159, 159, 159),  # Rape Seed
    35:  (137,  96,  84),  # Mustard
    36:  (168,   0, 232),  # Alfalfa
    37:  (168,   0, 232),  # Other Hay/Non Alfalfa
    38:  (  0, 168,  56),  # Camelina
    39:  (110,  73,   0),  # Buckwheat
    41:  (230,   0, 169),  # Sugarbeets
    42:  (165, 245, 122),  # Dry Beans
    43:  (  0, 174,  87),  # Potatoes
    44:  (115, 223, 255),  # Other Crops
    45:  (170, 255,   0),  # Sugarcane
    46:  (127,  95, 127),  # Sweet Potatoes
    47:  (147, 204, 147),  # Misc Vegs & Fruits
    48:  ( 43, 127,  95),  # Watermelons
    49:  (255, 170,   0),  # Onions
    50:  (181, 207, 143),  # Cucumbers
    51:  (204, 191, 163),  # Chick Peas
    52:  (255,   0, 255),  # Lentils
    53:  (255, 136,   0),  # Peas
    54:  (255, 102, 102),  # Tomatoes
    55:  (255, 115,   0),  # Caneberries
    56:  ( 95,  39,   0),  # Hops
    57:  (  0, 168, 232),  # Herbs
    58:  (127, 201, 127),  # Clover/Wildflowers
    59:  (217, 184, 196),  # Sod/Grass Seed
    60:  (209, 255, 115),  # Switchgrass
    61:  (204, 191, 163),  # Fallow/Idle Cropland
    66:  (230,   0, 169),  # Cherries
    67:  (181, 207, 143),  # Peaches
    68:  (204, 153, 102),  # Apples
    69:  (112, 168,   0),  # Grapes
    70:  (  0, 175,  76),  # Christmas Trees
    71:  (  0, 255, 255),  # Other Tree Crops
    72:  (255, 102, 119),  # Citrus
    74:  (137,  96,  84),  # Pecans
    75:  (214, 158, 188),  # Almonds
    76:  (165, 232,  10),  # Walnuts
    77:  (165, 165, 255),  # Pears
    92:  ( 72, 112, 235),  # Aquaculture
    111: ( 72, 112, 235),  # Open Water
    121: (204, 184, 170),  # Developed/Open Space
    122: (204,   0,   0),  # Developed/Low Intensity
    123: (153,   0,   0),  # Developed/Med Intensity
    124: (102,   0,   0),  # Developed/High Intensity
    131: (178, 173, 163),  # Barren
    141: (104, 170,  99),  # Deciduous Forest
    142: ( 28,  99,  48),  # Evergreen Forest
    143: (181, 201, 142),  # Mixed Forest
    152: (204, 186, 124),  # Shrubland
    176: (232, 255, 191),  # Grassland/Pasture
    190: (127, 201, 127),  # Herbaceous Wetlands
    195: (147, 204, 147),  # Woody Wetlands
    204: (214, 158, 188),  # Pistachios
    205: (165, 112,   0),  # Triticale
    206: (255, 102, 102),  # Carrots
    207: (  0, 175,  76),  # Asparagus
    208: (255, 160, 255),  # Garlic
    209: (255, 158,   9),  # Cantaloupes
    210: ( 38, 112,   0),  # Prunes
    211: ( 38, 112,   0),  # Olives
    212: (255, 158,   9),  # Oranges
    213: (165, 245, 122),  # Honeydew Melons
    214: ( 38, 112,   0),  # Broccoli
    215: ( 38, 112,   0),  # Avocados
    216: (255, 102, 102),  # Peppers
    217: (230,   0, 169),  # Pomegranates
    218: (255, 181, 200),  # Nectarines
    219: ( 38, 112,   0),  # Greens
    220: ( 38, 112,   0),  # Plums
    221: (255, 102, 102),  # Strawberries
    222: (255, 170,   0),  # Squash
    223: (255, 181, 200),  # Apricots
    224: (165, 245, 122),  # Vetch
    225: (165, 112,   0),  # Dbl WinWht/Corn
    226: (115, 178, 255),  # Dbl Oats/Corn
    227: ( 38, 112,   0),  # Lettuce
    228: (165, 112,   0),  # Dbl Triticale/Corn
    229: (255, 102,   0),  # Pumpkins
    236: (165, 112,   0),  # Dbl WinWht/Sorghum
    237: (137,  96,  84),  # Dbl Barley/Corn
    238: (255,  37,  37),  # Dbl WinWht/Cotton
    239: ( 38, 112,   0),  # Dbl Soybeans/Cotton
    240: ( 38, 112,   0),  # Dbl Soybeans/Oats
    241: (255, 211,   0),  # Dbl Corn/Soybeans
    242: (230,   0, 169),  # Blueberries
    243: ( 38, 112,   0),  # Cabbage
    244: ( 38, 112,   0),  # Cauliflower
    245: ( 38, 112,   0),  # Celery
    246: ( 38, 112,   0),  # Radishes
    247: ( 38, 112,   0),  # Turnips
    248: (168,   0, 232),  # Eggplant
    249: ( 38, 112,   0),  # Gourds
    250: (255,  37,  37),  # Cranberries
    254: (165, 112,   0),  # Dbl WinWht/Sunflower
}

# ── Build per-channel lookup tables (index = uint8 CDL code → byte value) ─────
_R_LUT = np.zeros(256, dtype=np.uint8)
_G_LUT = np.zeros(256, dtype=np.uint8)
_B_LUT = np.zeros(256, dtype=np.uint8)
_A_LUT = np.zeros(256, dtype=np.uint8)  # default alpha = 0 (transparent)

for _code, (_r, _g, _b) in CDL_RGB.items():
    if 0 <= _code <= 255:
        _R_LUT[_code] = _r
        _G_LUT[_code] = _g
        _B_LUT[_code] = _b
        _A_LUT[_code] = 255  # fully opaque for any known code


# ── PNG encoding ───────────────────────────────────────────────────────────────
try:
    from PIL import Image as _PIL_Image
    import io as _io

    def _encode_png(rgba: np.ndarray) -> bytes:
        """Encode (4, H, W) uint8 RGBA array → PNG bytes via Pillow."""
        img = _PIL_Image.fromarray(rgba.transpose(1, 2, 0), mode='RGBA')
        buf = _io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

except ImportError:
    from rasterio.io import MemoryFile as _MemoryFile

    def _encode_png(rgba: np.ndarray) -> bytes:
        """Encode (4, H, W) uint8 RGBA array → PNG bytes via rasterio MemoryFile."""
        with _MemoryFile() as memfile:
            with memfile.open(driver='PNG', dtype='uint8',
                              width=rgba.shape[2], height=rgba.shape[1],
                              count=4) as dst:
                dst.write(rgba)
            return memfile.read()


_TRANSPARENT_TILE: bytes | None = None


def _transparent_tile() -> bytes:
    """Return a cached 256×256 fully-transparent PNG."""
    global _TRANSPARENT_TILE
    if _TRANSPARENT_TILE is None:
        _TRANSPARENT_TILE = _encode_png(np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=np.uint8))
    return _TRANSPARENT_TILE


# ── Tile math ──────────────────────────────────────────────────────────────────
def _tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) in EPSG:3857 for XYZ tile (z, x, y)."""
    n = 2 ** z
    tile_m = 2 * HALF_WORLD / n
    x_min = -HALF_WORLD + x * tile_m
    x_max = x_min + tile_m
    y_max = HALF_WORLD - y * tile_m
    y_min = y_max - tile_m
    return x_min, y_min, x_max, y_max


def _overview_level(z: int) -> int:
    """Map Leaflet zoom level to rasterio overview_level index (0=finest, 8=coarsest).

    CDL overview factors: [2, 4, 8, 16, 32, 64, 128, 256, 512]
    z=2  → level 8 (5 km/px)   z=4  → level 6 (1.3 km/px)
    z=6  → level 4 (320 m/px)  z=8  → level 2 (80 m/px)
    z=10 → level 0 (20 m/px)
    """
    return max(0, min(8, 10 - z))


# ── Main render function ───────────────────────────────────────────────────────
def render_tile(z: int, x: int, y: int, code: str = 'all') -> bytes:
    """Render a 256×256 RGBA PNG for XYZ tile (z, x, y).

    code='all'     → all CDL categories with standard colors
    code='Total'   → only cropland codes highlighted (is_cropland=True)
    code='<int>'   → only that CDL pixel value visible; rest transparent
    """
    # ── Disk cache ────────────────────────────────────────────────────────────
    cache_path = TILE_CACHE_DIR / code / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    # ── Validate CDL file availability ────────────────────────────────────────
    if not CDL_PATH.exists():
        logger.warning("CDL raster not found: %s", CDL_PATH)
        return _transparent_tile()

    # ── Warp CDL into tile CRS / extent ───────────────────────────────────────
    bounds = _tile_bounds_3857(z, x, y)
    dst_transform = from_bounds(*bounds, TILE_SIZE, TILE_SIZE)
    dst_crs = CRS.from_epsg(3857)
    ov_level = _overview_level(z)

    dst_data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    try:
        with rasterio.open(CDL_PATH, overview_level=ov_level) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
    except Exception as exc:
        logger.warning("CDL tile warp failed z=%d x=%d y=%d: %s", z, x, y, exc)
        return _transparent_tile()

    # If tile is entirely background (all zeros), skip coloring
    if not dst_data.any():
        return _transparent_tile()

    # ── Colorize via LUT ──────────────────────────────────────────────────────
    rgba = np.stack([
        _R_LUT[dst_data],
        _G_LUT[dst_data],
        _B_LUT[dst_data],
        _A_LUT[dst_data],
    ])

    # ── Apply code filter ─────────────────────────────────────────────────────
    if code == 'all':
        pass  # keep all known-code pixels opaque

    elif code == 'Total':
        # Show only cropland codes (is_cropland=True in drought_crops.CDL_CATEGORIES)
        try:
            from drought_crops import CROPLAND_CODES
            cropland_mask = np.isin(dst_data, list(CROPLAND_CODES))
        except Exception:
            cropland_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        rgba[3] = np.where(cropland_mask, rgba[3], np.uint8(0))

    else:
        try:
            icode = int(code)
            match = dst_data == icode
            rgba[3] = np.where(match, rgba[3], np.uint8(0))
        except ValueError:
            logger.warning("Invalid CDL code param: %r", code)

    # ── Save to disk cache ────────────────────────────────────────────────────
    png_bytes = _encode_png(rgba)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(png_bytes)
    except Exception as exc:
        logger.warning("CDL tile cache write failed: %s", exc)

    return png_bytes


# ── Point sampling ─────────────────────────────────────────────────────────────
def _code_name(code: int) -> str | None:
    """Return human-readable CDL category name, or None for background."""
    if code == 0:
        return None
    try:
        from drought_crops import CDL_CATEGORIES
        info = CDL_CATEGORIES.get(code)
        if info:
            return info[0]  # (name, group, is_cropland)
    except Exception:
        pass
    return f"CDL code {code}"


def sample_pixel(lat: float, lng: float) -> dict:
    """Return CDL code and category name for a WGS84 (lat, lng) coordinate.

    Uses overview level 2 (~80 m/pixel) for fast single-pixel sampling.
    Returns {"code": <int|None>, "name": <str|None>}.
    """
    if not CDL_PATH.exists():
        return {"code": None, "name": None}
    try:
        with rasterio.open(CDL_PATH, overview_level=2) as src:
            # Reproject WGS84 → CDL native CRS (EPSG:5070)
            xs, ys = warp_transform('EPSG:4326', src.crs, [lng], [lat])
            row, col = src.index(xs[0], ys[0])
            if not (0 <= row < src.height and 0 <= col < src.width):
                return {"code": None, "name": None}
            val = int(src.read(1, window=Window(col, row, 1, 1))[0, 0])
        return {"code": val, "name": _code_name(val)}
    except Exception as exc:
        logger.debug("CDL sample_pixel failed lat=%.4f lng=%.4f: %s", lat, lng, exc)
        return {"code": None, "name": None}


# ── Storm crop impact ──────────────────────────────────────────────────────────

# Assumed impact radii in km for each hazard type:
#   torn (tornado)  — 5 km:   immediate damage path + secondary debris zone
#   hail            — 10 km:  typical severe-hail swath from one supercell
#   wind            — 17.5 km: straight-line wind / squall-line corridor
#   fire            — 0.5 km: buffer around each VIIRS 375 m detection pixel
STORM_IMPACT_RADII_KM: dict[str, float] = {
    "torn": 2.5,
    "hail": 5.0,
    "wind": 8.75,
    "fire": 0.5,
}

# Use overview level 2 (~80 m/px in EPSG:5070) — same level used by sample_pixel(),
# confirmed to return correct categorical CDL codes. 1 pixel ≈ 1.58 acres.
# Level 4 (~320 m/px) was tried but overview resampling corrupts categorical codes.
_IMPACT_OVERVIEW_LEVEL = 2


def compute_storm_impact(
    reports_by_type: dict[str, list[dict]],
    radii_km: dict[str, float] | None = None,
) -> dict[str, dict[int, float]]:
    """
    Sample CDL within a circular buffer around each storm report and
    accumulate pixel counts per CDL code for each hazard type.

    Args:
        reports_by_type: {'torn': [{lat, lon, ...}], 'hail': [...], 'wind': [...]}
        radii_km:        Per-type radius overrides; defaults to STORM_IMPACT_RADII_KM.

    Returns:
        {haz_type: {cdl_code (int): acres (float)}}
        Acreage may double-count overlapping storm impact zones.
    """
    if radii_km is None:
        radii_km = STORM_IMPACT_RADII_KM

    results: dict[str, dict[int, float]] = {k: {} for k in reports_by_type}

    if not CDL_PATH.exists():
        logger.warning("CDL raster not found for storm impact: %s", CDL_PATH)
        return results

    try:
        with rasterio.open(CDL_PATH, overview_level=_IMPACT_OVERVIEW_LEVEL) as src:
            pixel_area_m2 = abs(src.res[0]) * abs(src.res[1])
            pixel_acres = pixel_area_m2 / 4046.856  # m² → acres
            tfm = src.transform       # affine: x = c + (col+0.5)*a, y = f + (row+0.5)*e

            for haz_type, reports in reports_by_type.items():
                radius_m = radii_km.get(haz_type, 10.0) * 1000.0
                code_pixels: dict[int, int] = {}

                # Batch-project all report centres in one pyproj call (much faster
                # than one call per report, especially for fire with ~3000 points).
                valid: list[tuple[float, float]] = []
                for rep in reports:
                    try:
                        valid.append((float(rep["lat"]), float(rep["lon"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                if not valid:
                    results[haz_type] = {}
                    continue
                lats_b, lons_b = zip(*valid)
                xs_all, ys_all = warp_transform(
                    "EPSG:4326", src.crs, list(lons_b), list(lats_b)
                )

                for cx, cy in zip(xs_all, ys_all):

                    # Bounding box in native CRS
                    x_lo, x_hi = cx - radius_m, cx + radius_m
                    y_lo, y_hi = cy - radius_m, cy + radius_m

                    # Convert to raster row/col (y_hi → top row, y_lo → bottom row)
                    row_top, col_left  = src.index(x_lo, y_hi)
                    row_bot, col_right = src.index(x_hi, y_lo)

                    row_top   = max(0, row_top)
                    col_left  = max(0, col_left)
                    row_bot   = min(src.height, row_bot)
                    col_right = min(src.width,  col_right)

                    if row_top >= row_bot or col_left >= col_right:
                        continue

                    win = Window(col_left, row_top,
                                 col_right - col_left, row_bot - row_top)
                    data = src.read(1, window=win)

                    # Pixel-centre coordinates in native CRS
                    c_arr = np.arange(col_left, col_right)
                    r_arr = np.arange(row_top,  row_bot)
                    cc, rr = np.meshgrid(c_arr, r_arr)
                    px = tfm.c + (cc + 0.5) * tfm.a
                    py = tfm.f + (rr + 0.5) * tfm.e

                    # Circular mask
                    mask = ((px - cx) ** 2 + (py - cy) ** 2) <= radius_m ** 2

                    # Accumulate pixel counts by CDL code
                    unique, counts = np.unique(data[mask], return_counts=True)
                    for code, cnt in zip(unique.tolist(), counts.tolist()):
                        if code > 0:  # 0 = background / outside US
                            code_pixels[code] = code_pixels.get(code, 0) + cnt

                results[haz_type] = {
                    code: cnt * pixel_acres
                    for code, cnt in code_pixels.items()
                }

    except Exception as exc:
        logger.error("compute_storm_impact failed: %s", exc, exc_info=True)

    return results
