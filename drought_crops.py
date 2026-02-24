"""Drought-Crop Exposure Time Series.

Intersects USDM weekly drought polygons (D1–D4) with the 2024 USDA CDL
10m raster to compute acreage of each major crop type exposed to drought.

Cache file: /Volumes/T7/Weather_Models/data/drought_crop_timeseries.json

USDM polygons are cumulative: DM=1 covers all D1+ area, DM=2 covers D2+ area, etc.
So the "D1+" chart series comes directly from the DM=1 polygon.
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
CDL_PIXEL_SIZE_M   = 320.0          # approximate (actual: 319.999…)
PIXEL_ACRES        = CDL_PIXEL_SIZE_M ** 2 / 4047.0   # ≈ 25.3 acres/pixel

CDL_CROP_GROUPS = {
    "Corn":        [1],
    "Soybeans":    [5],
    "Wheat":       [22, 23, 24, 26, 27],        # durum, spring, winter, rye, millet
    "Oats":        [28],
    "Cotton":      [2],
    "Sorghum":     [4],
    "Hay/Alfalfa": [36, 37],
    "Peas":        [53],              # dry peas (CDL 53); chickpeas=51, lentils=52 stay in Other
    "Dry Beans":   [42],              # dry beans incl. fava, kidney, navy, pinto (no fava-specific code)
    "Other Crops": [
        3, 6, 10, 11, 12, 13, 14, 21, 25, 29, 30, 31, 32, 33, 34, 35,
        41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55,
        56, 57, 58, 59, 60, 61, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77,
        *range(204, 255),
    ],
    "Pasture":     [176],             # CDL 176 = Grassland/Pasture (not counted in cropland total)
}

# Groups that count toward the "Total Cropland" figure (excludes Pasture)
CROPLAND_GROUPS = {k for k in CDL_CROP_GROUPS if k != "Pasture"}

# Build reverse lookup: CDL pixel code → group name
_CODE_TO_GROUP: dict[int, str] = {}
for _grp, _codes in CDL_CROP_GROUPS.items():
    for _c in _codes:
        _CODE_TO_GROUP[_c] = _grp


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
    """Fetch USDM GeoJSON for a given YYYYMMDD string.

    Returns GeoJSON FeatureCollection or None on error.
    """
    url = f"https://droughtmonitor.unl.edu/data/json/usdm_{date_str}.json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch USDM for %s: %s", date_str, e)
        return None


def compute_exposure_for_date(date_str: str, geojson: dict, cdl_path: Path) -> dict:
    """Compute crop acres exposed to each drought level for one USDM snapshot.

    USDM polygons are cumulative: DM=1 covers all D1+ area.
    Returns {"D1": {"Corn": acres, ..., "Total": acres}, "D2": {...}, "D3": {...}, "D4": {...}}
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom

    # Build per-DM-level geometry dict (DM property: 1=D1, 2=D2, 3=D3, 4=D4)
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
            empty = {k: 0 for k in CDL_CROP_GROUPS}
            empty["Total"] = 0

            geom = features_by_dm.get(dm_int)
            if geom is None:
                result[level] = empty
                continue

            # Reproject geometry from WGS84 → CDL CRS (EPSG:5070)
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
            # Exclude background (0) and nodata pixels
            pixels = pixels[pixels > 0]

            if pixels.size == 0:
                result[level] = empty
                continue

            # Count pixels per CDL code, then group into named categories
            codes, counts = np.unique(pixels, return_counts=True)
            group_acres: dict[str, float] = {k: 0.0 for k in CDL_CROP_GROUPS}
            for code, count in zip(codes.tolist(), counts.tolist()):
                grp = _CODE_TO_GROUP.get(code)
                if grp:
                    group_acres[grp] += count * PIXEL_ACRES

            cropland_total = sum(group_acres[k] for k in CROPLAND_GROUPS)
            result[level] = {k: round(v) for k, v in group_acres.items()}
            result[level]["Total"] = round(cropland_total)

    return result


def compute_total_cropland_acres(cdl_path: Path = CDL_PATH) -> int:
    """Count total US cropland acres (excludes Pasture) using the overview raster."""
    import rasterio
    cropland_codes = [c for grp, codes in CDL_CROP_GROUPS.items()
                      if grp in CROPLAND_GROUPS for c in codes]
    with rasterio.open(cdl_path, overview_level=CDL_OVERVIEW_LEVEL) as src:
        data = src.read(1).ravel()
    crop_pixels = int(np.isin(data, cropland_codes).sum())
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

    # Compute total cropland once if not already stored
    if "total_cropland_acres" not in data:
        if progress_cb:
            progress_cb("Computing total US cropland acres from CDL...")
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
            progress_cb(f"  Computing crop exposure...")

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

        # Save incrementally after each date
        data["dates"] = sorted(data["dates"])
        data["computed_at"] = datetime.now(timezone.utc).isoformat()
        save_timeseries(data)

        if progress_cb:
            d1_total = exposure.get("D1", {}).get("Total", 0)
            progress_cb(f"  Done: D1+ Total = {d1_total/1e6:.1f}M acres")

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
