"""Livestock Drought Exposure Time Series.

For each weekly USDM date, computes how many head of cattle, hogs, and
chickens (layers) are in US counties designated D1+/D2+/D3+/D4 drought.

Method:
  - NASS 2022 Census of Agriculture county-level livestock inventories
  - USDM weekly GeoJSON polygons (cumulative per DM level, same as drought_crops.py)
  - Census 2020 population-weighted county centroids for spatial join
  - County centroid point-in-polygon → county drought designation (highest DM)
  - Sum livestock head counts by drought level for each week

Cache: /Volumes/T7/Weather_Models/data/livestock_drought_timeseries.json
"""

import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from drought_crops import fetch_usdm_for_date, get_usdm_tuesdays

logger = logging.getLogger(__name__)

CACHE_FILE            = Path("/Volumes/T7/Weather_Models/data/livestock_drought_timeseries.json")
NASS_LIVESTOCK_CACHE  = Path("/Volumes/T7/Weather_Models/data/nass_livestock_county.json")
COUNTY_CENTROIDS_CACHE = Path("/Volumes/T7/Weather_Models/data/county_centroids.json")

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# (commodity_desc, class_desc_or_None, series display name)
LIVESTOCK_SERIES = [
    ("CATTLE",   "INCL CALVES",  "Cattle"),
    ("CATTLE",   "COWS, MILK",   "Dairy Cows"),
    ("HOGS",     None,           "Hogs"),
    ("CHICKENS", "LAYERS",       "Chickens, Layers"),
    ("CHICKENS", "BROILERS",     "Chickens, Broilers"),
]

COUNTY_CENTROIDS_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
)


# ── API key ────────────────────────────────────────────────────────────────────
def _nass_api_key() -> str:
    key = os.environ.get("NASS_API_KEY", "")
    if key:
        return key
    # Fall back to reading .env in the same directory
    try:
        env_path = Path(__file__).parent / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("NASS_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


# ── NASS livestock county data ─────────────────────────────────────────────────
def fetch_nass_livestock_county(force: bool = False) -> dict:
    """Fetch 2022 Census county-level livestock inventories from NASS QuickStats.

    Returns:
      {
        "by_county": {fips: {"Cattle": N, "Hogs": N, "Chickens, Layers": N}},
        "totals":    {"Cattle": N, "Hogs": N, "Chickens, Layers": N}
      }

    Results are cached to NASS_LIVESTOCK_CACHE (2022 Census data never changes).
    Pass force=True to re-fetch even if cache exists.
    """
    if not force and NASS_LIVESTOCK_CACHE.exists():
        logger.info("Loading NASS livestock cache from disk")
        return json.loads(NASS_LIVESTOCK_CACHE.read_text())

    api_key = _nass_api_key()
    if not api_key:
        raise RuntimeError(
            "NASS_API_KEY not found. Set it in the .env file or environment."
        )

    by_county: dict[str, dict] = {}

    for commodity, class_desc, series_name in LIVESTOCK_SERIES:
        logger.info("Fetching NASS %s county inventory...", series_name)
        params: dict = {
            "key":               api_key,
            "source_desc":       "CENSUS",
            "year":              "2022",
            "agg_level_desc":    "COUNTY",
            "sector_desc":       "ANIMALS & PRODUCTS",
            "commodity_desc":    commodity,
            "statisticcat_desc": "INVENTORY",
            "unit_desc":         "HEAD",
            "domain_desc":       "TOTAL",   # one record per county; excludes size-class breakdowns
            "format":            "JSON",
        }
        if class_desc:
            params["class_desc"] = class_desc

        resp = requests.get(NASS_API_URL, params=params, timeout=120)
        resp.raise_for_status()
        records = resp.json().get("data", [])
        logger.info("  %d records returned for %s", len(records), series_name)

        for rec in records:
            state_ansi  = rec.get("state_ansi", "").strip().zfill(2)
            county_code = rec.get("county_code", "").strip()

            # Skip state-level aggregates and invalid entries
            if not county_code or county_code in ("998", "999", "000"):
                continue

            fips = state_ansi + county_code.zfill(3)

            # Parse value — may contain commas ("1,234") or be suppressed "(D)"
            val_str = rec.get("Value", "").replace(",", "").strip()
            if not val_str or val_str.startswith("("):
                continue
            try:
                val = int(float(val_str))
            except ValueError:
                continue

            if fips not in by_county:
                by_county[fips] = {}
            # If multiple sub-items for the same species (e.g. several class breakdowns),
            # keep the largest value as the most inclusive total.
            by_county[fips][series_name] = max(
                by_county[fips].get(series_name, 0), val
            )

    # National totals
    totals = {s[2]: 0 for s in LIVESTOCK_SERIES}
    for county_data in by_county.values():
        for name in totals:
            totals[name] += county_data.get(name, 0)

    output = {"by_county": by_county, "totals": totals}
    NASS_LIVESTOCK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    NASS_LIVESTOCK_CACHE.write_text(json.dumps(output))
    logger.info(
        "NASS livestock data cached: %d counties. Totals: %s",
        len(by_county),
        {k: f"{v/1e6:.1f}M" for k, v in totals.items()},
    )
    return output


# ── County centroids ───────────────────────────────────────────────────────────
def fetch_county_centroids(force: bool = False) -> dict:
    """Download 2020 Census population-weighted county centroids.

    Returns {fips: [lat, lng]} (lat/lng as floats).
    Cached to COUNTY_CENTROIDS_CACHE permanently (centroids don't change often).
    """
    if not force and COUNTY_CENTROIDS_CACHE.exists():
        return json.loads(COUNTY_CENTROIDS_CACHE.read_text())

    logger.info("Downloading Census county centroids from %s", COUNTY_CENTROIDS_URL)
    resp = requests.get(COUNTY_CENTROIDS_URL, timeout=30)
    resp.raise_for_status()

    centroids: dict[str, list] = {}
    # File has a UTF-8 BOM on the first line; utf-8-sig strips it automatically
    reader = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))
    for row in reader:
        statefp  = row.get("STATEFP",  "").strip().zfill(2)
        countyfp = row.get("COUNTYFP", "").strip().zfill(3)
        fips = statefp + countyfp
        try:
            lat = float(row["LATITUDE"])
            lng = float(row["LONGITUDE"])
            centroids[fips] = [lat, lng]
        except (ValueError, KeyError):
            continue

    COUNTY_CENTROIDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COUNTY_CENTROIDS_CACHE.write_text(json.dumps(centroids))
    logger.info("County centroids cached: %d counties", len(centroids))
    return centroids


# ── Spatial join ───────────────────────────────────────────────────────────────
def compute_county_drought_levels(usdm_geojson: dict, centroids: dict) -> dict:
    """Determine drought level (0–4) for each county using point-in-polygon.

    USDM polygons are cumulative: DM=1 covers all D1+ area, DM=4 covers only D4.
    A county's drought level = the highest DM polygon that contains its centroid.

    Uses shapely 2.x vectorized contains_xy() for fast batch testing.

    Returns {fips: dm_level} where dm_level 0 = no drought.
    """
    from shapely import contains_xy
    from shapely.geometry import shape

    fips_list = list(centroids.keys())
    lngs = np.array([centroids[f][1] for f in fips_list], dtype=np.float64)
    lats = np.array([centroids[f][0] for f in fips_list], dtype=np.float64)
    county_dm = np.zeros(len(fips_list), dtype=np.int8)

    # Process DM levels in ascending order; higher wins (cumulative)
    for feature in sorted(
        usdm_geojson.get("features", []),
        key=lambda f: f["properties"].get("DM", 0),
    ):
        dm = feature["properties"].get("DM")
        if dm not in (1, 2, 3, 4):
            continue
        try:
            geom = shape(feature["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception as exc:
            logger.warning("Skipping invalid USDM geometry DM=%d: %s", dm, exc)
            continue

        # Vectorized point-in-polygon for all ~3200 county centroids at once
        inside = contains_xy(geom, lngs, lats)
        county_dm = np.where(inside, np.maximum(county_dm, dm), county_dm)

    return dict(zip(fips_list, county_dm.tolist()))


# ── Per-date exposure ──────────────────────────────────────────────────────────
def compute_exposure_for_date(
    date_str: str,
    usdm_geojson: dict,
    livestock: dict,
    centroids: dict,
) -> dict:
    """Compute livestock head in D1+/D2+/D3+/D4 drought for one USDM date.

    Returns {"D1": {"Cattle": N, "Hogs": N, ...}, "D2": {...}, "D3": {...}, "D4": {...}}
    """
    county_dm = compute_county_drought_levels(usdm_geojson, centroids)
    by_county = livestock["by_county"]
    species_names = [s[2] for s in LIVESTOCK_SERIES]

    result = {}
    for dl in ("D1", "D2", "D3", "D4"):
        min_dm = int(dl[1])
        totals = {s: 0 for s in species_names}
        for fips, dm in county_dm.items():
            if dm >= min_dm and fips in by_county:
                for species in species_names:
                    totals[species] += by_county[fips].get(species, 0)
        result[dl] = totals

    return result


# ── Cache I/O ──────────────────────────────────────────────────────────────────
def load_timeseries() -> dict | None:
    """Load cached livestock drought time series, or None if missing."""
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception as exc:
        logger.error("Failed to load livestock drought cache: %s", exc)
        return None


def save_timeseries(data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data))


# ── Main build function ────────────────────────────────────────────────────────
def build_timeseries(
    start_date: str = "2024-01-02",
    progress_cb=None,
) -> dict:
    """Build or incrementally update the livestock drought time series.

    Downloads NASS county livestock data once (cached), then fetches USDM
    GeoJSON for each Tuesday that is not already in the cache.

    Returns the full time series dict.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        else:
            logger.info(msg)

    livestock = fetch_nass_livestock_county()
    log(
        f"NASS livestock: {len(livestock['by_county'])} counties. "
        + "  ".join(f"{s}: {v/1e6:.1f}M head" for s, v in livestock["totals"].items())
    )

    centroids = fetch_county_centroids()
    log(f"County centroids: {len(centroids)} counties")

    data = load_timeseries() or {
        "dates":          [],
        "by_date":        {},
        "species":        [s[2] for s in LIVESTOCK_SERIES],
        "species_totals": livestock["totals"],
        "computed_at":    None,
    }

    cached = set(data["dates"])
    tuesdays = get_usdm_tuesdays(start_date)
    new_count = 0

    for date_str in tuesdays:
        iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if iso in cached:
            continue

        geojson = fetch_usdm_for_date(date_str)
        if not geojson:
            log(f"[{iso}] USDM not available, skipping.")
            continue

        exposure = compute_exposure_for_date(date_str, geojson, livestock, centroids)
        data["by_date"][iso] = exposure
        data["dates"] = sorted(set(data["dates"]) | {iso})
        data["computed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_timeseries(data)
        new_count += 1

        d2_cattle = exposure["D2"]["Cattle"]
        log(f"[{iso}] D2 Cattle: {d2_cattle/1e6:.1f}M  D2 Hogs: {exposure['D2']['Hogs']/1e6:.1f}M")

    log(f"Livestock drought cache: {len(data['dates'])} weeks total ({new_count} new).")
    return data
