"""Storm crop impact timeseries — precomputed daily CDL acreage within storm buffers.

Each calendar day's SPC reports are processed independently, so a storm report
is counted in exactly one bucket (the day it occurred). No double-counting
across days.

Data structure stored in DATA_FILE:
{
  "computed_at": "2026-02-26T12:00:00Z",
  "days_back":   90,
  "radii_km":    {"torn": 5.0, "hail": 20.0, "wind": 35.0},
  "dates":       ["2025-11-28", ..., "2026-02-25"],   # ISO YYYY-MM-DD, sorted
  "by_date": {
    "2025-11-28": {
      "torn":   {"1": 50000, "5": 30000, "Total": 85000},  # acres per CDL code
      "hail":   {"1": 200000, "Total": 400000},
      "wind":   {"24": 80000, "Total": 200000},
      "counts": {"torn": 5, "hail": 23, "wind": 67}         # report count per type
    },
    ...
  },
  "code_info": {
    "1": {"name": "Corn", "group": "Major Field Crops", "is_cropland": true},
    ...
  }
}

Cache: /Volumes/T7/Weather_Models/data/storm_crop_timeseries.json
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_FILE = Path("/Volumes/T7/Weather_Models/data/storm_crop_timeseries.json")
DAYS_BACK  = 90   # rolling window to keep in the cache


def _date_range(days_back: int) -> list[date]:
    """Dates from `days_back` days ago up through today, inclusive."""
    today = date.today()
    return [today - timedelta(days=i) for i in range(days_back - 1, -1, -1)]


def _compute_day(date_str: str) -> dict:
    """Fetch and sample CDL for one calendar date (YYYY-MM-DD).

    Returns {
        'torn': {str(cdl_code): acres, 'Total': cropland_acres},
        'hail': {...},
        'wind': {...},
        'fire': {...},
        'counts': {'torn': N, 'hail': N, 'wind': N, 'fire': N},
    }
    """
    from cdl_tiles import compute_storm_impact, STORM_IMPACT_RADII_KM
    from drought_crops import CROPLAND_CODES
    from storm_reports import fetch_reports_for_date
    from fire_reports import fetch_fire_for_date

    ds_spc = date_str.replace("-", "")          # YYYYMMDD

    reports_by_type = fetch_reports_for_date(ds_spc)
    fire_detections  = fetch_fire_for_date(ds_spc)

    storm_counts = {t: len(reports_by_type.get(t, [])) for t in ("torn", "hail", "wind")}
    counts = {**storm_counts, "fire": len(fire_detections)}

    # Build combined dict for a single CDL pass
    all_reports: dict[str, list] = {**reports_by_type, "fire": fire_detections}
    # Drop types with no reports to avoid unnecessary work
    all_reports = {k: v for k, v in all_reports.items() if v}

    day: dict = {"counts": counts}

    if all_reports:
        impact = compute_storm_impact(all_reports, STORM_IMPACT_RADII_KM)
    else:
        impact = {}

    for haz in ("torn", "hail", "wind", "fire"):
        haz_data: dict[str, int] = {}
        cropland_total = 0
        for code, acres in impact.get(haz, {}).items():
            rounded = round(acres)
            haz_data[str(code)] = rounded
            if int(code) in CROPLAND_CODES:
                cropland_total += rounded
        haz_data["Total"] = cropland_total
        day[haz] = haz_data

    return day


def load_timeseries() -> dict | None:
    """Return cached timeseries dict, or None if unavailable."""
    if not DATA_FILE.exists():
        return None
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load storm crop timeseries: %s", e)
        return None


def _save(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data))


def update_timeseries(
    days_back: int = DAYS_BACK,
    progress_cb=None,
) -> dict:
    """Incrementally update the timeseries cache.

    - Computes only dates not yet in the cache.
    - Always recomputes today (reports may still arrive throughout the day).
    - Trims entries older than `days_back` days.

    Returns {'success': True, 'added': N, 'refreshed': N, 'total_dates': N}.
    """
    from cdl_tiles import STORM_IMPACT_RADII_KM
    from drought_crops import CDL_CATEGORIES

    # Load existing cache
    existing = {}
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text())
        except Exception as e:
            logger.warning("Could not read existing timeseries (%s); rebuilding.", e)
    by_date: dict = existing.get("by_date", {})

    # Build code_info from CDL_CATEGORIES
    code_info = {
        str(code): {"name": name, "group": group, "is_cropland": is_crop}
        for code, (name, group, is_crop) in CDL_CATEGORIES.items()
    }

    today = date.today()
    target_dates = _date_range(days_back)
    added = 0
    refreshed = 0

    total = len(target_dates)
    for i, d in enumerate(target_dates):
        ds = d.isoformat()
        is_today = (d == today)
        already_done = ds in by_date

        if already_done and not is_today:
            continue

        if progress_cb:
            progress_cb(f"[{i+1}/{total}] Storm crop impact for {ds}…")

        try:
            day_data = _compute_day(ds)
        except Exception as e:
            logger.error("storm_crop_timeseries: failed for %s: %s", ds, e)
            if progress_cb:
                progress_cb(f"  ERROR {ds}: {e}")
            continue

        by_date[ds] = day_data
        if already_done:
            refreshed += 1
        else:
            added += 1

        # Save incrementally so a crash doesn't lose everything
        cutoff = (today - timedelta(days=days_back)).isoformat()
        trimmed = {k: v for k, v in by_date.items() if k >= cutoff}
        _save({
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "days_back":   days_back,
            "radii_km":    {k: float(v) for k, v in STORM_IMPACT_RADII_KM.items()},
            "dates":       sorted(trimmed.keys()),
            "by_date":     trimmed,
            "code_info":   code_info,
        })
        if progress_cb:
            c = day_data.get("counts", {})
            t = day_data.get("torn",  {}).get("Total", 0)
            h = day_data.get("hail",  {}).get("Total", 0)
            w = day_data.get("wind",  {}).get("Total", 0)
            f = day_data.get("fire",  {}).get("Total", 0)
            progress_cb(
                f"  torn={c.get('torn',0)} rpts ({t/1e3:.0f}k crop-ac)  "
                f"hail={c.get('hail',0)} rpts ({h/1e3:.0f}k crop-ac)  "
                f"wind={c.get('wind',0)} rpts ({w/1e3:.0f}k crop-ac)  "
                f"fire={c.get('fire',0)} detections ({f/1e3:.0f}k crop-ac)"
            )

    # Final trim and save (handles case where nothing was computed)
    cutoff = (today - timedelta(days=days_back)).isoformat()
    by_date = {k: v for k, v in by_date.items() if k >= cutoff}
    final = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "days_back":   days_back,
        "radii_km":    {k: float(v) for k, v in STORM_IMPACT_RADII_KM.items()},
        "dates":       sorted(by_date.keys()),
        "by_date":     by_date,
        "code_info":   code_info,
    }
    _save(final)

    logger.info(
        "Storm crop timeseries: +%d new, %d refreshed, %d total dates",
        added, refreshed, len(final["dates"]),
    )
    return {
        "success":     True,
        "added":       added,
        "refreshed":   refreshed,
        "total_dates": len(final["dates"]),
    }
