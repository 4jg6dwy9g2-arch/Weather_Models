"""Drought-Crop Exposure Time Series.

Intersects USDM weekly drought polygons (D1–D4) with the 2024 USDA CDL
10m raster to compute acres of every CDL land-cover category exposed to drought.

Cache file: /Volumes/T7/Weather_Models/data/drought_crop_timeseries.json

USDM polygons are cumulative: DM=1 covers all D1+ area, DM=2 covers D2+ area, etc.
"""

import json
import logging
import requests
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FILE = Path("/Volumes/T7/Weather_Models/data/drought_crop_timeseries.json")
CDL_PATH   = Path("/Volumes/T7/Weather_Models/data/cdl_2024_10m/2024_10m_cdls.tif")

# Use overview level 4 (decimation factor 32) → ~320m pixels for performance.
# Full 10m raster is 13 GB; overview is ~130 MB and processes in ~1 s per date.
# 320m × 320m pixel = 102,400 m²; 1 acre = 4,047 m²
CDL_OVERVIEW_LEVEL = 4
CDL_PIXEL_SIZE_M   = 320.0
PIXEL_ACRES        = CDL_PIXEL_SIZE_M ** 2 / 4047.0   # ≈ 25.3 acres/pixel

# CDL code → (display name, dropdown group, counts_as_cropland)
CDL_CATEGORIES: dict[int, tuple[str, str, bool]] = {
    # ── Major Field Crops ────────────────────────────────────────────────────
    1:   ("Corn",                    "Major Field Crops", True),
    5:   ("Soybeans",                "Major Field Crops", True),
    2:   ("Cotton",                  "Major Field Crops", True),
    4:   ("Sorghum",                 "Major Field Crops", True),
    3:   ("Rice",                    "Major Field Crops", True),
    6:   ("Sunflower",               "Major Field Crops", True),
    10:  ("Peanuts",                 "Major Field Crops", True),
    11:  ("Tobacco",                 "Major Field Crops", True),
    12:  ("Sweet Corn",              "Major Field Crops", True),
    # ── Small Grains ────────────────────────────────────────────────────────
    24:  ("Winter Wheat",            "Small Grains",      True),
    23:  ("Spring Wheat",            "Small Grains",      True),
    22:  ("Durum Wheat",             "Small Grains",      True),
    21:  ("Barley",                  "Small Grains",      True),
    28:  ("Oats",                    "Small Grains",      True),
    27:  ("Rye",                     "Small Grains",      True),
    29:  ("Millet",                  "Small Grains",      True),
    31:  ("Canola",                  "Small Grains",      True),
    205: ("Triticale",               "Small Grains",      True),
    25:  ("Other Small Grains",      "Small Grains",      True),
    # ── Hay & Forage ────────────────────────────────────────────────────────
    36:  ("Alfalfa",                 "Hay & Forage",      True),
    37:  ("Other Hay/Non Alfalfa",   "Hay & Forage",      True),
    58:  ("Clover/Wildflowers",      "Hay & Forage",      True),
    59:  ("Sod/Grass Seed",          "Hay & Forage",      True),
    60:  ("Switchgrass",             "Hay & Forage",      True),
    # ── Legumes ─────────────────────────────────────────────────────────────
    42:  ("Dry Beans",               "Legumes",           True),
    53:  ("Peas",                    "Legumes",           True),
    51:  ("Chick Peas",              "Legumes",           True),
    52:  ("Lentils",                 "Legumes",           True),
    # ── Fruits & Nuts ───────────────────────────────────────────────────────
    75:  ("Almonds",                 "Fruits & Nuts",     True),
    69:  ("Grapes",                  "Fruits & Nuts",     True),
    76:  ("Walnuts",                 "Fruits & Nuts",     True),
    68:  ("Apples",                  "Fruits & Nuts",     True),
    74:  ("Pecans",                  "Fruits & Nuts",     True),
    72:  ("Citrus",                  "Fruits & Nuts",     True),
    66:  ("Cherries",                "Fruits & Nuts",     True),
    67:  ("Peaches",                 "Fruits & Nuts",     True),
    77:  ("Pears",                   "Fruits & Nuts",     True),
    204: ("Pistachios",              "Fruits & Nuts",     True),
    212: ("Oranges",                 "Fruits & Nuts",     True),
    215: ("Avocados",                "Fruits & Nuts",     True),
    242: ("Blueberries",             "Fruits & Nuts",     True),
    250: ("Cranberries",             "Fruits & Nuts",     True),
    221: ("Strawberries",            "Fruits & Nuts",     True),
    210: ("Prunes",                  "Fruits & Nuts",     True),
    211: ("Olives",                  "Fruits & Nuts",     True),
    217: ("Pomegranates",            "Fruits & Nuts",     True),
    218: ("Nectarines",              "Fruits & Nuts",     True),
    220: ("Plums",                   "Fruits & Nuts",     True),
    223: ("Apricots",                "Fruits & Nuts",     True),
    # ── Vegetables & Root Crops ─────────────────────────────────────────────
    43:  ("Potatoes",                "Vegetables",        True),
    45:  ("Sugarcane",               "Vegetables",        True),
    41:  ("Sugarbeets",              "Vegetables",        True),
    46:  ("Sweet Potatoes",          "Vegetables",        True),
    54:  ("Tomatoes",                "Vegetables",        True),
    49:  ("Onions",                  "Vegetables",        True),
    50:  ("Cucumbers",               "Vegetables",        True),
    48:  ("Watermelons",             "Vegetables",        True),
    47:  ("Misc Vegs & Fruits",      "Vegetables",        True),
    209: ("Cantaloupes",             "Vegetables",        True),
    213: ("Honeydew Melons",         "Vegetables",        True),
    214: ("Broccoli",                "Vegetables",        True),
    216: ("Peppers",                 "Vegetables",        True),
    219: ("Greens",                  "Vegetables",        True),
    222: ("Squash",                  "Vegetables",        True),
    227: ("Lettuce",                 "Vegetables",        True),
    229: ("Pumpkins",                "Vegetables",        True),
    206: ("Carrots",                 "Vegetables",        True),
    207: ("Asparagus",               "Vegetables",        True),
    208: ("Garlic",                  "Vegetables",        True),
    243: ("Cabbage",                 "Vegetables",        True),
    244: ("Cauliflower",             "Vegetables",        True),
    245: ("Celery",                  "Vegetables",        True),
    246: ("Radishes",                "Vegetables",        True),
    247: ("Turnips",                 "Vegetables",        True),
    248: ("Eggplant",                "Vegetables",        True),
    249: ("Gourds",                  "Vegetables",        True),
    # ── Other Agricultural ───────────────────────────────────────────────────
    61:  ("Fallow/Idle Cropland",    "Other Agricultural", True),
    44:  ("Other Crops",             "Other Agricultural", True),
    55:  ("Caneberries",             "Other Agricultural", True),
    56:  ("Hops",                    "Other Agricultural", True),
    57:  ("Herbs",                   "Other Agricultural", True),
    70:  ("Christmas Trees",         "Other Agricultural", True),
    71:  ("Other Tree Crops",        "Other Agricultural", True),
    30:  ("Speltz",                  "Other Agricultural", True),
    32:  ("Flaxseed",                "Other Agricultural", True),
    33:  ("Safflower",               "Other Agricultural", True),
    34:  ("Rape Seed",               "Other Agricultural", True),
    35:  ("Mustard",                 "Other Agricultural", True),
    38:  ("Camelina",                "Other Agricultural", True),
    39:  ("Buckwheat",               "Other Agricultural", True),
    224: ("Vetch",                   "Other Agricultural", True),
    26:  ("Dbl WinWht/Soybeans",     "Other Agricultural", True),
    225: ("Dbl WinWht/Corn",         "Other Agricultural", True),
    226: ("Dbl Oats/Corn",           "Other Agricultural", True),
    228: ("Dbl Triticale/Corn",      "Other Agricultural", True),
    236: ("Dbl WinWht/Sorghum",      "Other Agricultural", True),
    237: ("Dbl Barley/Corn",         "Other Agricultural", True),
    238: ("Dbl WinWht/Cotton",       "Other Agricultural", True),
    239: ("Dbl Soybeans/Cotton",     "Other Agricultural", True),
    240: ("Dbl Soybeans/Oats",       "Other Agricultural", True),
    241: ("Dbl Corn/Soybeans",       "Other Agricultural", True),
    254: ("Dbl WinWht/Sunflower",    "Other Agricultural", True),
    13:  ("Pop/Orn Corn",            "Other Agricultural", True),
    14:  ("Mint",                    "Other Agricultural", True),
    # ── Land Cover ───────────────────────────────────────────────────────────
    176: ("Grassland/Pasture",       "Land Cover",        False),
    152: ("Shrubland",               "Land Cover",        False),
    141: ("Deciduous Forest",        "Land Cover",        False),
    142: ("Evergreen Forest",        "Land Cover",        False),
    143: ("Mixed Forest",            "Land Cover",        False),
    131: ("Barren",                  "Land Cover",        False),
    190: ("Herbaceous Wetlands",     "Land Cover",        False),
    195: ("Woody Wetlands",          "Land Cover",        False),
    111: ("Open Water",              "Land Cover",        False),
    121: ("Developed/Open Space",    "Developed",         False),
    122: ("Developed/Low Intensity", "Developed",         False),
    123: ("Developed/Med Intensity", "Developed",         False),
    124: ("Developed/High Intensity","Developed",         False),
    92:  ("Aquaculture",             "Other",             False),
}

# Codes that count toward "Total Cropland"
CROPLAND_CODES: set[int] = {c for c, (_, _, is_crop) in CDL_CATEGORIES.items() if is_crop}

# Fast lookup: CDL pixel value → is it a tracked code?
_TRACKED_CODES: set[int] = set(CDL_CATEGORIES.keys())


def get_usdm_tuesdays(start_date: str = "2024-01-02") -> list[str]:
    """Return YYYYMMDD strings for every Tuesday from start_date through today."""
    d = date.fromisoformat(start_date)
    today = date.today()
    tuesdays = []
    while d <= today:
        tuesdays.append(d.strftime("%Y%m%d"))
        d += timedelta(days=7)
    return tuesdays


def fetch_usdm_for_date(date_str: str) -> dict | None:
    """Fetch USDM GeoJSON for a given YYYYMMDD string."""
    url = f"https://droughtmonitor.unl.edu/data/json/usdm_{date_str}.json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch USDM for %s: %s", date_str, e)
        return None


def compute_exposure_for_date(date_str: str, geojson: dict, cdl_path: Path) -> dict:
    """Compute per-CDL-code acres exposed to each drought level for one USDM snapshot.

    Returns {"D1": {"1": acres, "5": acres, ..., "Total": cropland_acres},
             "D2": {...}, "D3": {...}, "D4": {...}}
    Keys for individual codes are stringified ints (JSON-safe).
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom

    features_by_dm: dict[int, dict] = {}
    for feat in geojson.get("features", []):
        dm = feat.get("properties", {}).get("DM")
        if dm is not None and 1 <= dm <= 4:
            features_by_dm[dm] = feat.get("geometry")

    result: dict[str, dict] = {}

    with rasterio.open(cdl_path, overview_level=CDL_OVERVIEW_LEVEL) as src:
        src_crs = src.crs.to_string()

        for dm_int in [1, 2, 3, 4]:
            level = f"D{dm_int}"
            empty: dict = {"Total": 0}
            geom = features_by_dm.get(dm_int)
            if geom is None:
                result[level] = empty
                continue

            try:
                geom_proj = transform_geom("EPSG:4326", src_crs, geom)
            except Exception as e:
                logger.warning("transform_geom failed for %s DM=%d: %s", date_str, dm_int, e)
                result[level] = empty
                continue

            try:
                out_image, _ = rio_mask(
                    src, [geom_proj], crop=True, all_touched=False, nodata=0
                )
            except Exception as e:
                logger.warning("rio_mask failed for %s DM=%d: %s", date_str, dm_int, e)
                result[level] = empty
                continue

            pixels = out_image[0].ravel()
            pixels = pixels[pixels > 0]
            if pixels.size == 0:
                result[level] = empty
                continue

            codes, counts = np.unique(pixels, return_counts=True)
            level_data: dict = {}
            cropland_total = 0.0
            for code, count in zip(codes.tolist(), counts.tolist()):
                if code not in _TRACKED_CODES:
                    continue
                acres = count * PIXEL_ACRES
                level_data[str(code)] = round(acres)
                if code in CROPLAND_CODES:
                    cropland_total += acres

            level_data["Total"] = round(cropland_total)
            result[level] = level_data

    return result


def compute_code_acres(cdl_path: Path = CDL_PATH) -> dict[str, int]:
    """Return total US acres per tracked CDL code (stringified keys) from overview raster."""
    import rasterio
    with rasterio.open(cdl_path, overview_level=CDL_OVERVIEW_LEVEL) as src:
        data = src.read(1).ravel()
    codes, counts = np.unique(data, return_counts=True)
    return {
        str(int(c)): round(int(n) * PIXEL_ACRES)
        for c, n in zip(codes, counts)
        if int(c) in _TRACKED_CODES
    }


def compute_total_cropland_acres(cdl_path: Path = CDL_PATH) -> int:
    """Count total US cropland acres (excludes non-cropland categories)."""
    import rasterio
    with rasterio.open(cdl_path, overview_level=CDL_OVERVIEW_LEVEL) as src:
        data = src.read(1).ravel()
    crop_pixels = int(np.isin(data, list(CROPLAND_CODES)).sum())
    return round(crop_pixels * PIXEL_ACRES)


def build_timeseries(
    cdl_path: Path = CDL_PATH,
    start_date: str = "2024-01-02",
    progress_cb=None,
) -> dict:
    """Build the full time-series cache, skipping dates already computed.

    Saves incrementally after each date. Returns the full time-series dict.
    """
    data = load_timeseries() or {"dates": [], "by_date": {}}

    # Rebuild code_info whenever the category list may have changed
    if "code_info" not in data:
        if progress_cb:
            progress_cb("Computing per-code US acreage from CDL (one-time)...")
        code_acres = compute_code_acres(cdl_path)
        data["code_info"] = {
            str(code): {
                "name":        info[0],
                "group":       info[1],
                "is_cropland": info[2],
                "total_acres": code_acres.get(str(code), 0),
            }
            for code, info in CDL_CATEGORIES.items()
        }
        data["total_cropland_acres"] = compute_total_cropland_acres(cdl_path)
        if progress_cb:
            progress_cb(f"  Total cropland: {data['total_cropland_acres']/1e6:.0f}M acres")

    tuesdays = get_usdm_tuesdays(start_date)
    already_iso = set(data.get("dates", []))

    to_compute = []
    for t in tuesdays:
        iso = f"{t[:4]}-{t[4:6]}-{t[6:]}"
        if iso not in already_iso:
            to_compute.append((t, iso))

    if progress_cb:
        progress_cb(
            f"Need to compute {len(to_compute)} weeks "
            f"({len(already_iso)} already cached)"
        )

    for i, (date_str, iso_date) in enumerate(to_compute):
        if progress_cb:
            progress_cb(f"[{i+1}/{len(to_compute)}] {iso_date} — fetching USDM...")

        geojson = fetch_usdm_for_date(date_str)
        if geojson is None:
            if progress_cb:
                progress_cb(f"  SKIP {iso_date} (fetch failed)")
            continue

        if progress_cb:
            progress_cb(f"  Computing exposure...")

        try:
            exposure = compute_exposure_for_date(date_str, geojson, cdl_path)
        except Exception as e:
            logger.error("compute_exposure_for_date failed for %s: %s", iso_date, e)
            if progress_cb:
                progress_cb(f"  ERROR {iso_date}: {e}")
            continue

        data["by_date"][iso_date] = exposure
        if iso_date not in data["dates"]:
            data["dates"].append(iso_date)

        data["dates"] = sorted(data["dates"])
        data["computed_at"] = datetime.now(timezone.utc).isoformat()
        save_timeseries(data)

        if progress_cb:
            d1_total = exposure.get("D1", {}).get("Total", 0)
            progress_cb(f"  Done: D1+ cropland total = {d1_total/1e6:.1f}M acres")

    return data


def load_timeseries() -> dict | None:
    """Load drought_crop_timeseries.json or return None if missing/unreadable."""
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load timeseries cache: %s", e)
        return None


def save_timeseries(data: dict) -> None:
    """Write data to CACHE_FILE atomically via a temp file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(CACHE_FILE)
