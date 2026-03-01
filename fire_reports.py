"""NASA FIRMS VIIRS active fire detection fetching and caching.

Data source: NASA FIRMS — no API key required for rolling CSV files.
  CONUS 24h: https://firms.modaps.eosdis.nasa.gov/data/active_fire/
             noaa-20-viirs-c2/csv/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv
  CONUS 7d:  …_7d.csv  (used to extract individual-day data for the timeseries)

Satellite: NOAA-20 VIIRS (375 m resolution)
Confidence levels: low / nominal / high — we keep nominal and high only.
FRP filter: detections below MIN_FRP (10 MW) are dropped to reduce small agricultural burns.

Cache: /Volumes/T7/Weather_Models/data/fire_reports.json
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_FILE = Path("/Volumes/T7/Weather_Models/data/fire_reports.json")

_FIRMS_BASE = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire"
    "/noaa-20-viirs-c2/csv"
)
FIRMS_24H_URL = f"{_FIRMS_BASE}/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv"
FIRMS_7D_URL  = f"{_FIRMS_BASE}/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_7d.csv"

# Low-confidence detections have high false-positive rate (gas flares, sun glint, etc.)
CONFIDENCE_INCLUDE = {"nominal", "high"}

# Drop detections below this Fire Radiative Power threshold to filter small agricultural burns
MIN_FRP = 10.0  # MW

# Continental US bounding box (excludes Hawaii, Alaska, Puerto Rico)
_CONUS_LAT = (24.5, 49.5)
_CONUS_LON = (-124.8, -66.9)


def _parse_rows(rows: list[dict]) -> list[dict]:
    """Normalise FIRMS CSV rows into detection dicts."""
    detections = []
    for row in rows:
        try:
            conf = row.get("confidence", "").strip().lower()
            if conf not in CONFIDENCE_INCLUDE:
                continue
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if not (_CONUS_LAT[0] <= lat <= _CONUS_LAT[1] and _CONUS_LON[0] <= lon <= _CONUS_LON[1]):
                continue
            acq_date = row.get("acq_date", "").strip()   # YYYY-MM-DD
            acq_time = row.get("acq_time", "").strip()   # HHMM UTC
            frp = float(row.get("frp", 0) or 0)
            if frp < MIN_FRP:
                continue
            iso_time = None
            if acq_date and len(acq_time) == 4 and acq_time.isdigit():
                iso_time = f"{acq_date}T{acq_time[:2]}:{acq_time[2:]}:00Z"
            detections.append({
                "lat":        lat,
                "lon":        lon,
                "acq_date":   acq_date,
                "acq_time":   acq_time,
                "iso_time":   iso_time,
                "frp":        frp,
                "confidence": conf,
                "daynight":   row.get("daynight", "").strip(),
            })
        except (ValueError, KeyError):
            continue
    return detections


def sync_fire_reports() -> dict:
    """Fetch FIRMS VIIRS 24h CONUS fire detections and write to DATA_FILE."""
    try:
        resp = requests.get(FIRMS_24H_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error("FIRMS fetch failed: %s", e)
        return {"success": False, "error": str(e)}

    reader = csv.DictReader(io.StringIO(resp.text))
    detections = _parse_rows(list(reader))

    fetched_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "fetched_at":      fetched_at,
        "detection_count": len(detections),
        "detections":      detections,
    }
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(payload))
    logger.info("Fire reports synced: %d detections", len(detections))
    return {"success": True, "detection_count": len(detections), "fetched_at": fetched_at}


def fetch_fire_for_date(date_str: str) -> list[dict]:
    """Return fire detections for a single calendar date (YYYYMMDD).

    Fetches the 7-day rolling FIRMS CSV and filters to the target date.
    Works for dates within the last ~7 days; returns [] for older dates.
    """
    target = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"  # YYYY-MM-DD
    try:
        resp = requests.get(FIRMS_7D_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.debug("FIRMS 7d fetch failed: %s", e)
        return []
    reader = csv.DictReader(io.StringIO(resp.text))
    return _parse_rows(
        [row for row in reader if row.get("acq_date", "").strip() == target]
    )


def load_fire_reports() -> dict | None:
    """Return cached fire detections dict, or None if unavailable."""
    if not DATA_FILE.exists():
        return None
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load fire reports cache: %s", e)
        return None
