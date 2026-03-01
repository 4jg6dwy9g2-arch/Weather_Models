"""
NWS/SPC Local Storm Reports fetching and caching.

Data source: NOAA Storm Prediction Center
  Today:      https://www.spc.noaa.gov/climo/reports/today_{type}.csv
  Historical: https://www.spc.noaa.gov/climo/reports/YYMMDD_rpts_{type}.csv

Types: torn (tornadoes), hail, wind
Cache: /Volumes/T7/Weather_Models/data/storm_reports.json
"""

import csv
import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_FILE = Path("/Volumes/T7/Weather_Models/data/storm_reports.json")
SPC_BASE = "https://www.spc.noaa.gov/climo/reports"
REPORT_TYPES = ["torn", "hail", "wind"]
DAYS_BACK = 2  # fetch today + 2 previous SPC days (~72 h of coverage)


def _fetch_csv(url: str) -> list[dict]:
    """Fetch one SPC CSV and return list of row dicts. Returns [] on any error."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text), skipinitialspace=True)
        return [row for row in reader if any(v.strip() for v in row.values())]
    except Exception as e:
        logger.debug("SPC fetch skipped %s: %s", url, e)
        return []


def _parse_rows(rows: list[dict], report_type: str, date_str: str) -> list[dict]:
    """Normalise SPC CSV rows into report dicts.

    date_str: YYYYMMDD
    """
    reports = []
    for row in rows:
        try:
            lat = float(row.get("Lat") or 0)
            lon = float(row.get("Lon") or 0)
            if lat == 0 and lon == 0:
                continue

            if report_type == "torn":
                magnitude = row.get("F_Scale", "").strip()
            elif report_type == "hail":
                magnitude = row.get("Size", "").strip()
            else:
                magnitude = row.get("Speed", "").strip()

            time_str = row.get("Time", "").strip()  # HHMM
            # Build ISO datetime string for easy frontend filtering
            if len(time_str) == 4 and time_str.isdigit():
                iso_time = (
                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    f"T{time_str[:2]}:{time_str[2:]}:00Z"
                )
            else:
                iso_time = None

            reports.append({
                "type":      report_type,
                "date":      date_str,
                "time":      time_str,
                "iso_time":  iso_time,
                "magnitude": magnitude,
                "location":  row.get("Location", "").strip(),
                "county":    row.get("County", "").strip(),
                "state":     row.get("State", "").strip(),
                "lat":       lat,
                "lon":       lon,
                "comments":  row.get("Comments", "").strip(),
            })
        except (ValueError, KeyError):
            continue
    return reports


def sync_storm_reports() -> dict:
    """Fetch recent SPC storm reports and write to DATA_FILE.

    Returns dict with keys: success, report_count, fetched_at (or error).
    """
    now = datetime.now(timezone.utc)
    all_reports = []

    # Current SPC day ("today" files are updated live throughout the day)
    today_str = now.strftime("%Y%m%d")
    for rtype in REPORT_TYPES:
        url = f"{SPC_BASE}/today_{rtype}.csv"
        rows = _fetch_csv(url)
        all_reports.extend(_parse_rows(rows, rtype, today_str))

    # Previous SPC days
    for days_ago in range(1, DAYS_BACK + 1):
        day = now - timedelta(days=days_ago)
        date_str   = day.strftime("%Y%m%d")   # YYYYMMDD for storage
        date_short = day.strftime("%y%m%d")   # YYMMDD for SPC URL
        for rtype in REPORT_TYPES:
            url = f"{SPC_BASE}/{date_short}_rpts_{rtype}.csv"
            rows = _fetch_csv(url)
            all_reports.extend(_parse_rows(rows, rtype, date_str))

    fetched_at = now.isoformat()
    payload = {
        "fetched_at":   fetched_at,
        "report_count": len(all_reports),
        "reports":      all_reports,
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(payload))

    logger.info("Storm reports synced: %d reports", len(all_reports))
    return {"success": True, "report_count": len(all_reports), "fetched_at": fetched_at}


def fetch_reports_for_date(date_str: str) -> dict[str, list[dict]]:
    """Fetch SPC storm reports for a single calendar date (YYYYMMDD).

    Returns {'torn': [...], 'hail': [...], 'wind': [...]} — may be empty lists
    if no reports exist or the fetch fails.
    """
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y%m%d")
    result: dict[str, list[dict]] = {t: [] for t in REPORT_TYPES}
    for rtype in REPORT_TYPES:
        if date_str == today_str:
            url = f"{SPC_BASE}/today_{rtype}.csv"
        else:
            date_short = date_str[2:]  # YYYYMMDD → YYMMDD
            url = f"{SPC_BASE}/{date_short}_rpts_{rtype}.csv"
        rows = _fetch_csv(url)
        result[rtype] = _parse_rows(rows, rtype, date_str)
    return result


def load_storm_reports() -> dict | None:
    """Return cached storm reports dict, or None if unavailable."""
    if not DATA_FILE.exists():
        return None
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load storm reports cache: %s", e)
        return None
