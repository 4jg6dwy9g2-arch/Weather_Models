"""
US Drought Monitor data fetching and caching.

Data source: droughtmonitor.unl.edu (updates every Thursday)
Cache file: /Volumes/T7/Weather_Models/data/drought_monitor.json
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_FILE = Path("/Volumes/T7/Weather_Models/data/drought_monitor.json")
DROUGHT_URL = "https://droughtmonitor.unl.edu/data/json/usdm_current.json"


def sync_drought_monitor() -> dict:
    """Fetch latest GeoJSON from Drought Monitor, save to DATA_FILE.

    Returns dict with keys: success, date, feature_count (or error).
    """
    try:
        resp = requests.get(DROUGHT_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(data))

        feature_count = len(data.get("features", []))
        logger.info("Drought Monitor sync: %d features", feature_count)
        return {"success": True, "feature_count": feature_count}

    except Exception as e:
        logger.error("Drought Monitor sync failed: %s", e)
        return {"success": False, "error": str(e)}


def load_drought_data() -> dict | None:
    """Return cached data as {geojson, synced_at}, or None if file missing/unreadable.

    synced_at is the file mtime as an ISO date string (YYYY-MM-DD).
    """
    if not DATA_FILE.exists():
        return None
    try:
        geojson = json.loads(DATA_FILE.read_text())
        mtime = DATA_FILE.stat().st_mtime
        synced_at = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        return {"geojson": geojson, "synced_at": synced_at}
    except Exception as e:
        logger.error("Failed to load drought monitor cache: %s", e)
        return None
