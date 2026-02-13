"""
Weather Forecast Comparison Flask App
Compare GFS and ECMWF AIFS model forecasts.
"""

from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import logging
import json
from functools import lru_cache
import time
import queue
import threading
import os
import math
import hmac
import hashlib
import collections
import re

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

import requests

from gfs import GFSModel
from ecmwf_aifs import AIFSModel, AIFS_VARIABLES
from ecmwf_ifs import IFSModel, IFS_VARIABLES
import weatherlink
from nws_forecast import (
    get_grid_point,
    fetch_hourly_forecast,
    fetch_aqi_forecast,
    fetch_wind_gusts,
    rate_running_conditions,
    calculate_dew_point as nws_calculate_dew_point,
    is_daylight as nws_is_daylight,
    IDEAL_TEMP_RANGE
)

NWS_API = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(Weather Models App, contact@example.com)",
    "Accept": "application/geo+json"
}

NWS_CACHE_PATH = Path(__file__).resolve().parent / "data" / "nws_forecast_cache.json"
VERIF_TS_CACHE_PATH = Path(__file__).resolve().parent / "data" / "verification_time_series_cache.json"
VERIF_LEAD_CACHE_PATH = Path(__file__).resolve().parent / "data" / "verification_lead_time_cache.json"
COCORAHs_CACHE_PATH = Path(__file__).resolve().parent / "data" / "cocorahs_daily_cache.json"
BIAS_HISTORY_CACHE_PATH = Path(__file__).resolve().parent / "data" / "bias_history_cache.json"
import asos
import rossby_waves
import nws_batch

try:
    import analog_metrics as _analog_metrics
    _HAS_ANALOG_METRICS = True
except Exception as _e:
    _analog_metrics = None  # type: ignore
    _HAS_ANALOG_METRICS = False
    logger = logging.getLogger(__name__)
    logger.warning("analog_metrics import failed – falling back to Pearson: %s", _e)

try:
    from astral import LocationInfo, moon
    from astral.sun import sun
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - optional dependency
    LocationInfo = None
    moon = None
    sun = None
    ZoneInfo = None
from pangu_integration import pangu_bp, cleanup_database, load_runs_db

# Single JSON file for storing all forecast data
FORECASTS_FILE = Path(__file__).parent / "data" / "forecasts.json"

# ASOS forecast trends storage file (last 2 runs with 6-hourly data)
ASOS_TRENDS_FILE = Path(__file__).parent / "data" / "asos_forecast_trends.json"

# ERA5 analog forecast history (tracks predictions over time)
ANALOG_HISTORY_FILE = Path(__file__).parent / "data" / "analog_forecast_history.json"
# Full cached result from the most recent auto-sync (allows instant UI load)
ANALOG_LATEST_FILE = Path(__file__).parent / "data" / "analog_latest_result.json"
SOM_CACHE_FILE = Path(__file__).parent / "data" / "era5_som_cache_6x6_30y_anom.npz"
SOM_SKILL_FILE = Path(__file__).parent / "data" / "som_cluster_skill.json"

_SOM_CACHE: dict | None = None
_SOM_SKILL_CACHE: dict = {}
_SOM_ANOM_CACHE: dict | None = None

# ---------------------------------------------------------------------------
# ERA5 analog caches (populated lazily, persist for the lifetime of the process)
# ---------------------------------------------------------------------------
# Structure: {cache_key: {'climatology': dict, 'ds': xr.Dataset, ...}}
# cache_key = frozenset of ERA5 file paths + overlap bounds
_CLIMATOLOGY_CACHE: dict = {}

# ---------------------------------------------------------------------------
# In-memory cache for forecasts.json (invalidated when file mtime changes)
# ---------------------------------------------------------------------------
_forecasts_db_cache: dict | None = None
_forecasts_db_mtime: float | None = None

# EOF cache (loaded/built in background thread)
_EOF_CACHE: dict | None = None
_EOF_CACHE_READY = threading.Event()

if load_dotenv:
    load_dotenv()

# WeatherLink API Credentials (Davis Weather Station)
WEATHERLINK_API_KEY = os.getenv("WEATHERLINK_API_KEY")
WEATHERLINK_API_SECRET = os.getenv("WEATHERLINK_API_SECRET")
WEATHERLINK_STATION_ID = os.getenv("WEATHERLINK_STATION_ID", "117994")
COCORAHs_STATION_ID = os.getenv("COCORAHs_STATION_ID", "VA-FX-121")
COCORAHs_ACIS_SID = os.getenv("COCORAHs_ACIS_SID")

# Default location: Fairfax, VA (matches Workout_Data)
DEFAULT_LAT = 38.8419
DEFAULT_LON = -77.3091

# Forecast retention window (days)
FORECAST_RETENTION_DAYS = 20

# EPA AQI color scale
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "USG": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}

try:
    import weather_data_local as local_weather_data
except Exception:
    local_weather_data = None


def load_forecasts_db():
    """Load the forecasts database from JSON file, with in-memory mtime cache."""
    global _forecasts_db_cache, _forecasts_db_mtime
    if not FORECASTS_FILE.exists():
        return {}
    try:
        current_mtime = FORECASTS_FILE.stat().st_mtime
        if _forecasts_db_cache is not None and _forecasts_db_mtime == current_mtime:
            return _forecasts_db_cache
        with open(FORECASTS_FILE) as f:
            data = json.load(f)
        _forecasts_db_cache = migrate_db_format(data)
        _forecasts_db_mtime = current_mtime
        return _forecasts_db_cache
    except Exception as e:
        logger.warning(f"Error loading forecasts.json: {e}")
        return {}


def load_nws_cache():
    """Load cached NWS forecast data."""
    if NWS_CACHE_PATH.exists():
        try:
            with open(NWS_CACHE_PATH, "r") as f:
                data = json.load(f)
                # Normalize legacy format into runs list
                if isinstance(data, dict) and "runs" not in data and "forecast" in data:
                    data = {
                        "runs": [data]
                    }
                return data
        except Exception as e:
            logger.warning(f"Error loading NWS cache: {e}")
    return {}


def load_cocorahs_cache():
    """Load cached CoCoRaHS daily precipitation data."""
    if COCORAHs_CACHE_PATH.exists():
        try:
            with open(COCORAHs_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading CoCoRaHS cache: {e}")
    return {}


def save_cocorahs_cache(payload: dict):
    """Save CoCoRaHS daily precipitation cache."""
    try:
        with open(COCORAHs_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving CoCoRaHS cache: {e}")


def _normalize_cocorahs_station_id(station_id: str) -> str | None:
    """
    Normalize a CoCoRaHS station ID like "VA-FX-121" -> "VAFX0121".
    Returns None if format isn't recognized.
    """
    if not station_id:
        return None
    cleaned = station_id.strip().upper()
    # Normalize any non-alphanumeric separators to a dash
    cleaned = re.sub(r"[^A-Z0-9]+", "-", cleaned).strip("-")

    match = re.match(r"^([A-Z]{2})-([A-Z]{2})-(\d+)$", cleaned)
    if not match:
        match = re.match(r"^([A-Z]{2})([A-Z]{2})(\d+)$", cleaned)
    if not match:
        return None
    state, county, num = match.groups()
    return f"{state}{county}{int(num):04d}"


def _parse_cocorahs_value(val):
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    if val in ("M", "NA", "", None, "--", "**", "*"):
        return None
    if val == "T":
        return 0.001
    try:
        return float(val)
    except Exception:
        return None


def fetch_cocorahs_daily_precip(station_id: str, start_date: str, end_date: str) -> dict:
    """
    Fetch daily precipitation from ACIS for a CoCoRaHS station.

    Returns:
        Dict of date -> precip (inches)
    """
    # Cache lookup
    cache = load_cocorahs_cache()
    station_cache = cache.get(station_id, {})
    cached_data = station_cache.get("data", {})
    last_fetched = station_cache.get("last_fetched")

    # Check if cache covers requested range and is recent (<12 hours)
    def _covers_range():
        try:
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
        except Exception:
            return False
        d = start_dt
        while d <= end_dt:
            if d.isoformat() not in cached_data:
                return False
            d += timedelta(days=1)
        return True

    if last_fetched and _covers_range():
        try:
            last_dt = datetime.fromisoformat(last_fetched)
            if datetime.now(timezone.utc) - last_dt < timedelta(hours=12):
                return cached_data
        except Exception:
            pass

    # Fetch from ACIS - try multiple station id formats
    normalized = _normalize_cocorahs_station_id(station_id)
    sid_candidates = [station_id]
    if COCORAHs_ACIS_SID:
        sid_candidates.insert(0, COCORAHs_ACIS_SID)
    if normalized and normalized not in sid_candidates:
        # ACIS CoCoRaHS type code is 10 (8-char id like VAFX0121)
        sid_candidates.append(normalized)
        sid_candidates.append(f"{normalized} 10")
        sid_candidates.append(f"US1{normalized}")

    data = None
    for sid in sid_candidates:
        params = {
            "sid": sid,
            "sdate": start_date,
            "edate": end_date,
            "elems": [{"name": "pcpn", "interval": "dly", "duration": "dly", "add": "f"}],
            "meta": "name"
        }
        try:
            response = requests.post("https://data.rcc-acis.org/StnData", json=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if payload and not payload.get("error") and payload.get("data"):
                data = payload
                break
        except Exception as e:
            logger.warning(f"Error fetching CoCoRaHS data for {sid}: {e}")
            continue

    if not data:
        logger.warning(f"No CoCoRaHS data returned for {station_id} ({sid_candidates})")
        return cached_data

    results = {}
    for row in data.get("data", []):
        if not row or len(row) < 2:
            continue
        date_str = row[0]
        val = row[1]
        results[date_str] = _parse_cocorahs_value(val)

    # Merge into cache
    cached_data.update(results)
    cache[station_id] = {
        "last_fetched": datetime.now(timezone.utc).isoformat(),
        "data": cached_data
    }
    save_cocorahs_cache(cache)
    return cached_data


def get_asos_data_span_days() -> int:
    """Return the span of available ASOS run data in days."""
    db = asos.load_asos_forecasts_db()
    runs = db.get("runs", {})
    if not runs:
        return 0
    times = []
    for run_id in runs.keys():
        try:
            dt = datetime.fromisoformat(run_id)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            times.append(dt)
        except Exception:
            continue
    if not times:
        return 0
    span = max(times) - min(times)
    return int(span.total_seconds() / 86400)


def save_nws_cache(payload: dict):
    """Save NWS forecast data to cache."""
    try:
        with open(NWS_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving NWS cache: {e}")


def load_verif_ts_cache() -> dict:
    if VERIF_TS_CACHE_PATH.exists():
        try:
            with open(VERIF_TS_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading verification time series cache: {e}")
    return {"entries": {}}


def save_verif_ts_cache(payload: dict):
    try:
        with open(VERIF_TS_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving verification time series cache: {e}")


def load_verif_lead_cache() -> dict:
    if VERIF_LEAD_CACHE_PATH.exists():
        try:
            with open(VERIF_LEAD_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading verification lead-time cache: {e}")
    return {"entries": {}}


def save_verif_lead_cache(payload: dict):
    try:
        with open(VERIF_LEAD_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving verification lead-time cache: {e}")


def load_bias_history_cache() -> dict:
    if BIAS_HISTORY_CACHE_PATH.exists():
        try:
            with open(BIAS_HISTORY_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading bias history cache: {e}")
    return {"entries": {}}


def save_bias_history_cache(payload: dict):
    try:
        with open(BIAS_HISTORY_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving bias history cache: {e}")


def _get_verif_ts_source_mtimes() -> dict:
    def _mtime(path: Path):
        try:
            return path.stat().st_mtime
        except Exception:
            return None

    return {
        "forecasts": _mtime(FORECASTS_FILE),
        "nws": _mtime(NWS_CACHE_PATH),
        "cocorahs": _mtime(COCORAHs_CACHE_PATH)
    }


def precompute_verif_time_series_cache(location_name: str, configs: list[dict]):
    """
    Precompute verification time series cache entries for fast first-load.
    configs: list of dicts with keys: variable, lead_time, days_back
    """
    cache = load_verif_ts_cache()
    entries = cache.get("entries", {})
    source_mtimes = _get_verif_ts_source_mtimes()

    for cfg in configs:
        variable = cfg.get("variable", "temp")
        lead_time = int(cfg.get("lead_time", 24))
        days_back = int(cfg.get("days_back", 30))
        cache_key = f"{location_name}|{variable}|{lead_time}|{days_back}"
        result = calculate_verification_time_series(location_name, variable, lead_time, days_back)
        if "error" in result:
            continue
        entries[cache_key] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mtimes": source_mtimes,
            "time_series": result
        }

    cache["entries"] = entries
    save_verif_ts_cache(cache)


def load_asos_trends_db():
    """Load the ASOS forecast trends database from JSON file."""
    if ASOS_TRENDS_FILE.exists():
        try:
            with open(ASOS_TRENDS_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading asos_forecast_trends.json: {e}")
    return {"runs": {}, "stations": {}}


def save_asos_trends_db(db: dict):
    """Save the ASOS forecast trends database to JSON file."""
    try:
        with open(ASOS_TRENDS_FILE, 'w') as f:
            json.dump(db, f, indent=2)
        logger.info(f"Saved ASOS trends to {ASOS_TRENDS_FILE}")
    except Exception as e:
        logger.error(f"Error saving ASOS trends database: {e}")


def store_asos_trend_run(init_time: datetime, model: str, station_forecasts: dict):
    """
    Store a single model run with 6-hourly forecast data for trend visualization.
    Keeps only the last 2 runs in the database.

    Args:
        init_time: Model initialization time
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
        station_forecasts: Dict mapping station_id to forecast data
            Each station dict has: temps, mslps, precips (lists aligned with 6-hourly forecast_hours)
    """
    db = load_asos_trends_db()

    # Update stations dict (always keep current)
    stations = asos.get_stations_dict()
    db["stations"] = stations

    # Forecast hours for trends: every 6 hours from 0 to 360
    forecast_hours = list(range(0, 361, 6))

    # Create run entry if needed
    run_key = init_time.isoformat()
    if run_key not in db["runs"]:
        db["runs"][run_key] = {
            "forecast_hours": forecast_hours,
        }

    # Store this model's forecasts
    db["runs"][run_key][model.lower()] = station_forecasts

    # Keep only the last 2 runs
    sorted_runs = sorted(db["runs"].keys())
    if len(sorted_runs) > 2:
        # Remove oldest runs
        for old_run in sorted_runs[:-2]:
            del db["runs"][old_run]
            logger.info(f"Removed old trend run: {old_run}")

    # Save
    save_asos_trends_db(db)
    logger.info(f"Stored {model.upper()} trend forecasts for {len(station_forecasts)} stations at {run_key}")


def load_analog_history():
    """Load the analog forecast history from JSON file."""
    if ANALOG_HISTORY_FILE.exists():
        try:
            with open(ANALOG_HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading analog_forecast_history.json: {e}")
    return {"predictions": []}


def save_analog_prediction(target_date: str, analog_precip: float, analog_temp: float,
                           climatology_precip: float, climatology_temp: float,
                           top_analogs: list, avg_correlation: float):
    """
    Save an analog forecast prediction to the history file.
    Keeps all predictions indefinitely for long-term analysis.

    Args:
        target_date: Date being analyzed (analog match date)
        analog_precip: Predicted 14-day precipitation
        analog_temp: Predicted 14-day temperature
        climatology_precip: Climatological normal precipitation
        climatology_temp: Climatological normal temperature
        top_analogs: List of top analog dates with correlations
        avg_correlation: Average pattern similarity of top analogs (0-1)
    """
    history = load_analog_history()

    prediction = {
        "prediction_date": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date,
        "analog_precip": analog_precip,
        "analog_temp": analog_temp,
        "climatology_precip": climatology_precip,
        "climatology_temp": climatology_temp,
        "avg_correlation": avg_correlation,
        "top_analogs": top_analogs[:5]  # Keep top 5 analogs
    }

    history["predictions"].append(prediction)

    # Keep all predictions indefinitely for long-term analysis
    # No cutoff applied

    try:
        with open(ANALOG_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved analog prediction for {target_date} (avg correlation: {avg_correlation:.3f})")
    except Exception as e:
        logger.error(f"Error saving analog prediction: {e}")


def run_analog_sync(gfs_init_time=None) -> dict:
    """
    Compute historical analog pattern matching and persist results.

    Called automatically from run_master_sync() after each new GFS fetch.
    Skips recomputation if the latest cached result already covers this GFS
    initialization time.

    Returns a dict with keys:
      status: 'computed' | 'cached' | 'skipped' | 'error'
      avg_precip, avg_temp, avg_correlation, wave_number  (on 'computed')
      message / error  (on other statuses)
    """
    import pandas as pd
    import numpy as np
    import xarray as xr

    # Dedup check: skip if we already have a cached result for this exact GFS run.
    if gfs_init_time is not None and ANALOG_LATEST_FILE.exists():
        try:
            with open(ANALOG_LATEST_FILE) as f:
                cached = json.load(f)
            cached_init = cached.get('computed_for_init')
            if cached_init and pd.Timestamp(cached_init) == pd.Timestamp(str(gfs_init_time)):
                return {'status': 'cached', 'message': f'Already computed for {gfs_init_time}'}
        except Exception:
            pass  # If check fails, proceed with fresh computation

    try:
        core = _find_analogs_core(top_n=10, method='composite')
    except Exception as e:
        return {'status': 'error', 'error': f'Pattern matching failed: {e}'}

    top_analogs = core['top_analogs']
    init_time = core['init_time']
    current_date_str = core['current_date_str']
    current_wave_num = core['current_wave_num']

    weather_path = Path("/Volumes/T7/Weather_Models/era5/Fairfax/reanalysis-era5-single-levels-timeseries-sfc1zs15i59.nc")
    if not weather_path.exists():
        return {'status': 'skipped', 'message': 'ERA5 Fairfax weather data not available (drive not mounted?)'}

    avg_precip = avg_temp = climatology_precip = climatology_temp = None
    try:
        weather_ds = xr.open_dataset(weather_path)

        for analog in top_analogs:
            try:
                analog_date = pd.Timestamp(analog['date'])
                end_date = analog_date + pd.Timedelta(days=14)
                precip_m = float(weather_ds['tp'].sel(valid_time=slice(analog_date, end_date)).sum().values)
                analog['precip_14d'] = round(precip_m * 39.3701, 2)
                temp_k = float(weather_ds['t2m'].sel(valid_time=slice(analog_date, end_date)).mean().values)
                analog['temp_14d'] = round((temp_k - 273.15) * 9/5 + 32, 1)
            except Exception as e:
                logger.warning(f"Analog sync: could not get outcomes for {analog['date']}: {e}")
                analog['precip_14d'] = None
                analog['temp_14d'] = None

        precip_values = [a['precip_14d'] for a in top_analogs if a.get('precip_14d') is not None]
        avg_precip = round(np.mean(precip_values), 2) if precip_values else None
        temp_values = [a['temp_14d'] for a in top_analogs if a.get('temp_14d') is not None]
        avg_temp = round(np.mean(temp_values), 1) if temp_values else None

        # Climatology normals (1940–2025, same calendar window)
        clim_precip_list: list = []
        clim_temp_list: list = []
        try:
            for year in range(1940, 2026):
                try:
                    start = pd.Timestamp(year=year, month=init_time.month, day=init_time.day)
                    end = start + pd.Timedelta(days=14)
                    cp = float(weather_ds['tp'].sel(valid_time=slice(start, end)).sum().values)
                    if not np.isnan(cp):
                        clim_precip_list.append(cp * 39.3701)
                    ct = float(weather_ds['t2m'].sel(valid_time=slice(start, end)).mean().values)
                    if not np.isnan(ct):
                        clim_temp_list.append((ct - 273.15) * 9/5 + 32)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Analog sync: climatology error: {e}")

        climatology_precip = round(np.mean(clim_precip_list), 2) if clim_precip_list else None
        climatology_temp = round(np.mean(clim_temp_list), 1) if clim_temp_list else None

    except Exception as e:
        logger.warning(f"Analog sync: ERA5 weather data error: {e}")

    if avg_precip is None or avg_temp is None:
        return {'status': 'skipped', 'message': 'Insufficient ERA5 weather data for outcome calculation'}

    avg_correlation = (sum(a['correlation'] for a in top_analogs) / len(top_analogs)) if top_analogs else 0

    save_analog_prediction(
        target_date=current_date_str,
        analog_precip=avg_precip,
        analog_temp=avg_temp,
        climatology_precip=climatology_precip or 0,
        climatology_temp=climatology_temp or 0,
        top_analogs=[{'date': a['date'], 'correlation': a['correlation']} for a in top_analogs],
        avg_correlation=avg_correlation,
    )

    # Cache the full result so the UI can display it instantly on page load.
    cached_result = {
        'success': True,
        'current_date': current_date_str,
        'current_wave_number': current_wave_num,
        'method': 'composite',
        'analogs': top_analogs,
        'avg_precip_14d': avg_precip,
        'avg_temp_14d': avg_temp,
        'climatology_precip_14d': climatology_precip,
        'climatology_temp_14d': climatology_temp,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'computed_for_init': str(gfs_init_time) if gfs_init_time is not None else current_date_str,
    }
    try:
        with open(ANALOG_LATEST_FILE, 'w') as f:
            json.dump(cached_result, f, indent=2)
    except Exception as e:
        logger.warning(f"Analog sync: could not write latest result cache: {e}")

    logger.info(
        f"Analog sync complete: {current_date_str}, "
        f"precip={avg_precip}\", temp={avg_temp}°F, corr={avg_correlation:.3f}"
    )
    return {
        'status': 'computed',
        'target_date': current_date_str,
        'avg_precip': avg_precip,
        'avg_temp': avg_temp,
        'avg_correlation': avg_correlation,
        'wave_number': current_wave_num,
    }


def fetch_nws_forecast_cache(hours_ahead: int = 168):
    """Fetch NWS forecast and cache temperature and precip data for verification overlays."""
    grid_id, grid_x, grid_y, forecast_url = get_grid_point(DEFAULT_LAT, DEFAULT_LON)
    forecast = fetch_hourly_forecast(forecast_url)

    now = datetime.now().astimezone()
    cutoff = now + timedelta(hours=hours_ahead)

    result = []
    for hour in forecast:
        hour_time = datetime.fromisoformat(hour["datetime"])
        if hour_time > cutoff:
            break
        result.append({
            "datetime": hour["datetime"],
            "temperature": hour["temperature"],
            "precip_mm": hour.get("rain_amount_mm", 0) or 0
        })

    snapshot = {
        "success": True,
        "location": {"lat": DEFAULT_LAT, "lon": DEFAULT_LON, "grid": grid_id},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "hours_ahead": hours_ahead,
        "forecast": result
    }
    # Append snapshot to cache history (same retention as model runs)
    cache = load_nws_cache()
    runs = cache.get("runs", []) if isinstance(cache, dict) else []
    runs.append(snapshot)

    cutoff = datetime.now(timezone.utc) - timedelta(days=FORECAST_RETENTION_DAYS)
    pruned = []
    for run in runs:
        try:
            fetched_at = datetime.fromisoformat(run.get("fetched_at"))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            if fetched_at >= cutoff:
                pruned.append(run)
        except Exception:
            continue

    payload = {"runs": pruned}
    save_nws_cache(payload)
    return snapshot


def migrate_db_format(data: dict) -> dict:
    """
    Migrate from old flat format to new historical runs format.

    Old format:
    {
      "Fairfax, VA": {
        "fetched_at": "...",
        "gfs": {...},
        "aifs": {...}
      }
    }

    New format:
    {
      "Fairfax, VA": {
        "runs": {
          "2026-01-27T06:00:00": {
            "fetched_at": "...",
            "gfs": {...},
            "aifs": {...},
            "observed": {...}
          }
        },
        "latest_run": "2026-01-27T06:00:00"
      }
    }
    """
    migrated = {}

    for location, loc_data in data.items():
        # Check if already in new format
        if "runs" in loc_data:
            migrated[location] = loc_data
            continue

        # Migrate old format
        gfs_init = loc_data.get("gfs", {}).get("init_time")
        if gfs_init:
            migrated[location] = {
                "runs": {
                    gfs_init: {
                        "fetched_at": loc_data.get("fetched_at"),
                        "gfs": loc_data.get("gfs"),
                        "aifs": loc_data.get("aifs"),
                        "observed": loc_data.get("observed"),
                        "verification": loc_data.get("verification")
                    }
                },
                "latest_run": gfs_init
            }
        else:
            # No valid data, skip
            migrated[location] = {"runs": {}, "latest_run": None}

    return migrated


def save_forecasts_db(data):
    """Save the forecasts database to JSON file."""
    with open(FORECASTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved forecasts to {FORECASTS_FILE}")


def _mean_series(series_list):
    if not series_list:
        return None
    base = series_list[0]
    mean = {"times": base.get("times", [])}
    for key, values in base.items():
        if key == "times":
            continue
        if not isinstance(values, list):
            continue
        length = len(values)
        sums = [0.0] * length
        count = 0
        for series in series_list:
            vals = series.get(key)
            if isinstance(vals, list) and len(vals) == length:
                for i in range(length):
                    v = vals[i]
                    if v is None:
                        continue
                    sums[i] += v
                count += 1
        if count > 0:
            mean[key] = [v / count for v in sums]
    return mean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)


@app.route('/api/cocorahs-debug')
def api_cocorahs_debug():
    """
    Debug CoCoRaHS ACIS lookup for a station.
    Returns the first payload that has data, plus a summary of attempts.
    """
    station_id = request.args.get('station_id', COCORAHs_STATION_ID)
    days_back = int(request.args.get('days_back', 10))
    now_utc = datetime.now(timezone.utc)
    local_now = weatherlink.utc_to_eastern(now_utc.replace(tzinfo=None))
    end_date = (local_now.date() - timedelta(days=1))
    start_date = end_date - timedelta(days=days_back - 1)

    normalized = _normalize_cocorahs_station_id(station_id)
    sid_candidates = [station_id]
    if COCORAHs_ACIS_SID:
        sid_candidates.insert(0, COCORAHs_ACIS_SID)
    if normalized and normalized not in sid_candidates:
        sid_candidates.append(normalized)
        sid_candidates.append(f"{normalized} 10")
        sid_candidates.append(f"US1{normalized}")

    attempts = []
    payload_with_data = None
    for sid in sid_candidates:
        params = {
            "sid": sid,
            "sdate": start_date.isoformat(),
            "edate": end_date.isoformat(),
            "elems": [{"name": "pcpn", "interval": "dly", "duration": "dly"}],
            "meta": "name"
        }
        try:
            response = requests.post("https://data.rcc-acis.org/StnData", json=params, timeout=30)
            status = response.status_code
            payload = response.json()
            has_data = bool(payload and payload.get("data"))
            attempts.append({
                "sid": sid,
                "status": status,
                "has_data": has_data,
                "error": payload.get("error") if isinstance(payload, dict) else None
            })
            if has_data and payload_with_data is None:
                payload_with_data = payload
        except Exception as e:
            attempts.append({"sid": sid, "status": None, "has_data": False, "error": str(e)})

    return jsonify({
        "success": True,
        "station_id": station_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sid_candidates": sid_candidates,
        "attempts": attempts,
        "sample_payload": payload_with_data
    })
app.register_blueprint(pangu_bp, url_prefix="/pangu")
cleanup_database()

# Queue for streaming sync logs to clients
sync_log_queues = []
sync_log_lock = threading.Lock()


class StreamingLogHandler(logging.Handler):
    """Custom log handler that sends messages to connected SSE clients."""

    def emit(self, record):
        try:
            msg = self.format(record)
            log_type = 'info'

            if record.levelno >= logging.ERROR:
                log_type = 'error'
            elif record.levelno >= logging.WARNING:
                log_type = 'warning'
            elif 'success' in msg.lower() or 'synced' in msg.lower():
                log_type = 'success'

            # Send to all connected clients
            with sync_log_lock:
                for q in sync_log_queues[:]:
                    try:
                        q.put_nowait({'message': msg, 'type': log_type})
                    except queue.Full:
                        sync_log_queues.remove(q)
        except Exception:
            pass


def broadcast_sync_log(message, log_type='info'):
    """Broadcast a custom message to all sync log listeners."""
    with sync_log_lock:
        for q in sync_log_queues[:]:
            try:
                q.put_nowait({'message': message, 'type': log_type})
            except queue.Full:
                sync_log_queues.remove(q)


# Add streaming handler to the root logger to capture all logs
streaming_handler = StreamingLogHandler()
streaming_handler.setLevel(logging.INFO)
streaming_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(streaming_handler)


class Variable:
    """Weather variable definition."""
    def __init__(self, name, display_name, units, herbie_search, category, colormap, contour_levels, fill=True, level=None):
        self.name = name
        self.display_name = display_name
        self.units = units
        self.herbie_search = herbie_search
        self.category = category
        self.colormap = colormap
        self.contour_levels = contour_levels
        self.fill = fill
        self.level = level


class Region:
    """Geographic region definition."""
    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds


def get_aqi_color(aqi: int) -> str:
    """Return EPA color for an AQI value."""
    if aqi <= 50:
        return AQI_COLORS["Good"]
    elif aqi <= 100:
        return AQI_COLORS["Moderate"]
    elif aqi <= 150:
        return AQI_COLORS["USG"]
    elif aqi <= 200:
        return AQI_COLORS["Unhealthy"]
    elif aqi <= 300:
        return AQI_COLORS["Very Unhealthy"]
    else:
        return AQI_COLORS["Hazardous"]


def get_aqi_category(aqi: int) -> str:
    """Return EPA category name for an AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "USG"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def calculate_sunrise_sunset(lat: float, lon: float, date: datetime) -> tuple:
    """
    Calculate sunrise and sunset times for a given location and date.
    Returns (sunrise, sunset) as datetime objects in local time.
    """
    day_of_year = date.timetuple().tm_yday
    lat_rad = math.radians(lat)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    decl_rad = math.radians(declination)
    cos_hour_angle = -math.tan(lat_rad) * math.tan(decl_rad)
    cos_hour_angle = max(-1, min(1, cos_hour_angle))
    hour_angle = math.degrees(math.acos(cos_hour_angle))

    # Approximate local solar noon; UTC offset for Eastern (rough)
    utc_offset = -5
    solar_noon_utc = 12 - (lon / 15)
    solar_noon_local = solar_noon_utc + utc_offset

    sunrise_hour = solar_noon_local - (hour_angle / 15)
    sunset_hour = solar_noon_local + (hour_angle / 15)

    sunrise = date.replace(
        hour=int(sunrise_hour),
        minute=int((sunrise_hour % 1) * 60),
        second=0,
        microsecond=0,
    )
    sunset = date.replace(
        hour=int(sunset_hour),
        minute=int((sunset_hour % 1) * 60),
        second=0,
        microsecond=0,
    )

    return sunrise, sunset


def is_daylight(dt: datetime, lat: float, lon: float) -> bool:
    """Check if a given datetime is during daylight hours."""
    sunrise, sunset = calculate_sunrise_sunset(lat, lon, dt)
    return sunrise <= dt.replace(tzinfo=None) <= sunset


def generate_weatherlink_signature(parameters: dict) -> str:
    """Generate HMAC SHA-256 signature for WeatherLink API."""
    parameters = collections.OrderedDict(sorted(parameters.items()))
    api_secret = parameters.pop("api-secret")

    data = ""
    for key in parameters:
        data = data + key + str(parameters[key])

    signature = hmac.new(
        api_secret.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return signature


def fetch_current_weather() -> dict:
    """Fetch current conditions from WeatherLink API."""
    if not WEATHERLINK_API_KEY or not WEATHERLINK_API_SECRET:
        raise RuntimeError("Missing WeatherLink API credentials")

    parameters = {
        "api-key": WEATHERLINK_API_KEY,
        "api-secret": WEATHERLINK_API_SECRET,
        "station-id": WEATHERLINK_STATION_ID,
        "t": int(time.time()),
    }

    signature = generate_weatherlink_signature(parameters.copy())

    url = (
        f"https://api.weatherlink.com/v2/current/{WEATHERLINK_STATION_ID}"
        f"?api-key={WEATHERLINK_API_KEY}"
        f"&api-signature={signature}"
        f"&t={parameters['t']}"
    )

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"WeatherLink API Error: {response.status_code} - {response.text}")

    return response.json()


def parse_current_weather(json_data: dict) -> dict:
    """Parse WeatherLink current conditions JSON into a clean dict."""
    result = {
        "timestamp": None,
        "temp_f": None,
        "dew_point_f": None,
        "humidity": None,
        "heat_index_f": None,
        "wind_speed_mph": None,
        "wind_gust_mph": None,
        "wind_direction": None,
        "wind_direction_deg": None,
        "solar_rad": None,
        "rain_today_in": None,
        "rain_rate_in": None,
        "barometer_mb": None,
        "aqi": None,
        "pm25": None,
        "inside_temp_f": None,
        "inside_humidity": None,
        "soil_temp_5in": None,
        "soil_temp_10in": None,
        "soil_temp_20in": None,
        "soil_moisture_5in": None,
        "soil_moisture_10in": None,
        "soil_moisture_20in": None,
    }

    if not json_data or "sensors" not in json_data:
        return result

    def wind_dir_to_compass(degrees):
        if degrees is None:
            return "--"
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        ]
        idx = round(degrees / 22.5) % 16
        return directions[idx]

    for sensor in json_data["sensors"]:
        sensor_type = sensor.get("sensor_type")

        for record in sensor.get("data", []):
            ts = record.get("ts")
            if ts and (result["timestamp"] is None or ts > result["timestamp"]):
                result["timestamp"] = ts

            if sensor_type == 243:  # Inside Temp/Hum (from AirLink)
                result["inside_temp_f"] = record.get("temp_in", record.get("temp_in_last"))
                result["inside_humidity"] = record.get("hum_in", record.get("hum_in_last"))

            elif sensor_type == 242:  # Barometer
                bar_inhg = record.get("bar_sea_level")
                if bar_inhg is not None:
                    result["barometer_mb"] = round(bar_inhg * 33.8639, 1)

            elif sensor_type == 45:  # ISS (main weather station)
                result["temp_f"] = record.get("temp", record.get("temp_last"))
                result["dew_point_f"] = record.get("dew_point", record.get("dew_point_last"))
                result["humidity"] = record.get("hum", record.get("hum_last"))
                result["heat_index_f"] = record.get("heat_index", record.get("heat_index_last"))
                result["wind_speed_mph"] = record.get("wind_speed_avg_last_10_min", record.get("wind_speed_avg"))
                result["wind_gust_mph"] = record.get("wind_speed_hi_last_10_min", record.get("wind_speed_hi"))
                wind_deg = record.get("wind_dir_scalar_avg_last_10_min", record.get("wind_dir_of_prevail"))
                result["wind_direction_deg"] = wind_deg
                result["wind_direction"] = wind_dir_to_compass(wind_deg)
                result["solar_rad"] = record.get("solar_rad")
                result["rain_today_in"] = record.get("rainfall_daily_in", record.get("rainfall_in"))
                result["rain_rate_in"] = record.get("rain_rate_hi_in", record.get("rain_rate_last_in"))

            elif sensor_type == 323:  # Outside AirLink
                result["aqi"] = record.get("aqi_val")
                result["pm25"] = record.get("pm_2p5")

            elif sensor_type == 56:  # Soil/Leaf station
                result["soil_temp_10in"] = record.get("temp_1")
                result["soil_temp_5in"] = record.get("temp_2")
                result["soil_temp_20in"] = record.get("temp_4")
                result["soil_moisture_10in"] = record.get("moist_soil_1")
                result["soil_moisture_5in"] = record.get("moist_soil_2")
                result["soil_moisture_20in"] = record.get("moist_soil_4")

    return result


def calculate_rain_24h() -> Optional[float]:
    """Calculate the last 24-hour precipitation total from local CSV data."""
    if not local_weather_data:
        return None

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    try:
        records = local_weather_data.get_historical_data(start_time, end_time)
    except Exception as e:
        logger.warning(f"Rain 24h calculation failed: {e}")
        return None

    if not records:
        return None

    def parse_dt(val):
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None

    filtered = []
    for r in records:
        dt = parse_dt(r.get("datetime"))
        if dt is not None:
            filtered.append((dt, r.get("rain")))

    if not filtered:
        return None

    filtered.sort(key=lambda x: x[0])

    total = 0.0
    prev_val = None
    for _, rain_val in filtered:
        if rain_val is None:
            continue
        if prev_val is None:
            prev_val = rain_val
            continue
        diff = rain_val - prev_val
        if diff < 0:
            diff = rain_val
        total += max(diff, 0.0)
        prev_val = rain_val

    return round(total, 2)




# GFS variable definitions
TEMP_2M_GFS = Variable(
    name="t2m",
    display_name="2m Temperature",
    units="F",
    herbie_search=":TMP:2 m above ground:",
    category="surface",
    colormap="RdYlBu_r",
    contour_levels=list(range(-40, 120, 5))
)

PRECIP_GFS = Variable(
    name="precip",
    display_name="Total Precipitation",
    units="in",
    herbie_search=":APCP:",
    category="surface",
    colormap="Blues",
    contour_levels=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
)

MSLP_GFS = Variable(
    name="mslp",
    display_name="Mean Sea Level Pressure",
    units="mb",
    herbie_search=":PRMSL:mean sea level:",
    category="surface",
    colormap="coolwarm",
    contour_levels=list(range(960, 1060, 4))
)

Z500_GFS = Variable(
    name="z500",
    display_name="500 hPa Geopotential Height",
    units="dm",
    herbie_search=":HGT:500 mb:",
    category="upper_air",
    colormap="viridis",
    contour_levels=list(range(480, 600, 6))
)


# Preset locations (bounds aligned to 0.25 degree grid)
LOCATIONS = {
    "Fairfax, VA": (-77.5, -77.0, 38.5, 39.0),
}

# CONUS region for ASOS station extraction (covers continental US)
CONUS_BOUNDS = (-130.0, -60.0, 20.0, 55.0)  # west, east, south, north

# Lead times for ASOS verification (hours)
ASOS_LEAD_TIMES = [6, 12, 24, 48, 72, 120, 168]

# Cache for verification results (1 hour TTL)
_verification_cache = {}
_verification_cache_time = {}



def fetch_gfs_data(region, forecast_hours, init_hour: Optional[int] = None):
    """Fetch temperature, precipitation, and MSLP data from GFS.

    Note: GFS precipitation is already per-interval (e.g., 0-6h, 6-12h).

    Args:
        region: Geographic region to fetch
        forecast_hours: List of forecast hours
        init_hour: Optional specific init hour (0, 6, 12, or 18). If None, gets the latest available.
    """
    gfs_model = GFSModel()

    if init_hour is not None:
        init_time = gfs_model.get_init_time_for_hour(init_hour)
    else:
        init_time = gfs_model.get_latest_init_time()

    temps = []
    precips = []
    mslps = []
    times = []

    for hour in forecast_hours:
        # Temperature
        try:
            temp_data = gfs_model.fetch_data(TEMP_2M_GFS, init_time, hour, region)
            temps.append(float(temp_data.values.mean()))
        except Exception as e:
            logger.warning(f"GFS temp fetch failed for F{hour}: {e}")
            temps.append(None)

        # Precipitation (6-hour interval totals)
        try:
            precip_data = gfs_model.fetch_data(PRECIP_GFS, init_time, hour, region)
            precips.append(float(precip_data.values.mean()))
        except Exception as e:
            logger.warning(f"GFS precip fetch failed for F{hour}: {e}")
            precips.append(None)

        # MSLP
        try:
            mslp_data = gfs_model.fetch_data(MSLP_GFS, init_time, hour, region)
            mslps.append(float(mslp_data.values.mean()))
        except Exception as e:
            logger.warning(f"GFS MSLP fetch failed for F{hour}: {e}")
            mslps.append(None)

        times.append((init_time + timedelta(hours=hour)).isoformat())

    # Calculate wave number forecast at 24-hour intervals for 15 days
    z500_waves = None  # Current wave number (F000)
    z500_field = None  # Store F000 z500 field for analog matching
    z500_waves_forecast = {
        "times": [],
        "wave_numbers": [],
        "amplitudes": [],
        "variance_explained": []
    }

    # Calculate at 24-hour intervals up to 360 hours (15 days)
    wave_forecast_hours = list(range(0, 361, 24))
    global_region = Region("Global", (-180, 180, 20, 70))

    for fhr in wave_forecast_hours:
        try:
            logger.info(f"Calculating Rossby wave number for GFS F{fhr:03d}")
            z500_data = gfs_model.fetch_data(Z500_GFS, init_time, fhr, global_region)
            wave_metrics = rossby_waves.calculate_wave_number(z500_data, latitude=55.0)

            # Store forecast time series
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(wave_metrics.get('wave_number'))
            # Get amplitude and variance of dominant wave
            amplitudes = wave_metrics.get('top_3_amplitudes', [])
            variance = wave_metrics.get('top_3_variance', [])
            z500_waves_forecast["amplitudes"].append(amplitudes[0] if amplitudes else None)
            z500_waves_forecast["variance_explained"].append(variance[0] if variance else None)

            # Store F000 as current wave number and save full field
            if fhr == 0:
                z500_waves = wave_metrics
                # Save the full z500 field for analog matching (F000 only)
                z500_field = {
                    'values': z500_data.values.tolist(),
                    'latitude': z500_data.latitude.values.tolist(),
                    'longitude': z500_data.longitude.values.tolist()
                }
                amp_str = f"{amplitudes[0]:.1f}m" if amplitudes else "N/A"
                var_str = f"{variance[0]:.1f}%" if variance else "N/A"
                logger.info(f"GFS wave number: {wave_metrics.get('wave_number')} (amplitude: {amp_str}, variance: {var_str})")

        except Exception as e:
            logger.warning(f"Wave analysis failed for GFS F{fhr:03d}: {e}")
            # Add None values to maintain array alignment
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(None)
            z500_waves_forecast["amplitudes"].append(None)
            z500_waves_forecast["variance_explained"].append(None)

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat(),
        "z500_waves": z500_waves,
        "z500_waves_forecast": z500_waves_forecast,
        "z500_field": z500_field
    }


def fetch_aifs_data(region, forecast_hours, init_hour: Optional[int] = None):
    """Fetch temperature, precipitation, and MSLP data from ECMWF AIFS.

    Note: AIFS precipitation is cumulative from init time, so we convert
    to 6-hour interval totals to match GFS behavior.

    Args:
        region: Geographic region to fetch
        forecast_hours: List of forecast hours
        init_hour: Optional specific init hour (0, 6, 12, or 18). If None, gets the latest available.
    """
    aifs_model = AIFSModel()

    if init_hour is not None:
        init_time = aifs_model.get_init_time_for_hour(init_hour)
    else:
        init_time = aifs_model.get_latest_init_time()

    temp_var = AIFS_VARIABLES["t2m"]
    precip_var = AIFS_VARIABLES["tp"]
    mslp_var = AIFS_VARIABLES["mslp"]

    temps = []
    cumulative_precips = []  # Store cumulative first, then convert
    mslps = []
    times = []

    for hour in forecast_hours:
        # Temperature
        try:
            temp_data = aifs_model.fetch_data(temp_var, init_time, hour, region)
            temps.append(float(temp_data.values.mean()))
        except Exception as e:
            logger.warning(f"AIFS temp fetch failed for F{hour}: {e}")
            temps.append(None)

        # Precipitation (cumulative from init - will convert to interval below)
        try:
            precip_data = aifs_model.fetch_data(precip_var, init_time, hour, region)
            cumulative_precips.append(float(precip_data.values.mean()))
        except Exception as e:
            logger.warning(f"AIFS precip fetch failed for F{hour}: {e}")
            cumulative_precips.append(None)

        # MSLP
        try:
            mslp_data = aifs_model.fetch_data(mslp_var, init_time, hour, region)
            mslps.append(float(mslp_data.values.mean()))
        except Exception as e:
            logger.warning(f"AIFS MSLP fetch failed for F{hour}: {e}")
            mslps.append(None)

        times.append((init_time + timedelta(hours=hour)).isoformat())

    # Convert cumulative precipitation to 6-hour interval totals
    precips = []
    for i, cum in enumerate(cumulative_precips):
        if cum is None:
            precips.append(None)
        elif i == 0:
            # First interval is just the cumulative value (0 to first hour)
            precips.append(cum)
        else:
            prev_cum = cumulative_precips[i - 1]
            if prev_cum is not None:
                interval = cum - prev_cum
                # Ensure non-negative (rounding errors can cause tiny negatives)
                precips.append(max(0.0, interval))
            else:
                precips.append(None)

    # Calculate wave number forecast at 24-hour intervals for 15 days
    z500_waves = None  # Current wave number (F000)
    z500_waves_forecast = {
        "times": [],
        "wave_numbers": [],
        "amplitudes": [],
        "variance_explained": []
    }

    # Calculate at 24-hour intervals up to 360 hours (15 days)
    wave_forecast_hours = list(range(0, 361, 24))
    global_region = Region("Global", (-180, 180, 20, 70))
    z500_var = AIFS_VARIABLES["z500"]

    for fhr in wave_forecast_hours:
        try:
            logger.info(f"Calculating Rossby wave number for AIFS F{fhr:03d}")
            z500_data = aifs_model.fetch_data(z500_var, init_time, fhr, global_region)
            wave_metrics = rossby_waves.calculate_wave_number(z500_data, latitude=55.0)

            # Store forecast time series
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(wave_metrics.get('wave_number'))
            # Get amplitude and variance of dominant wave
            amplitudes = wave_metrics.get('top_3_amplitudes', [])
            variance = wave_metrics.get('top_3_variance', [])
            z500_waves_forecast["amplitudes"].append(amplitudes[0] if amplitudes else None)
            z500_waves_forecast["variance_explained"].append(variance[0] if variance else None)

            # Store F000 as current wave number
            if fhr == 0:
                z500_waves = wave_metrics
                amp_str = f"{amplitudes[0]:.1f}m" if amplitudes else "N/A"
                var_str = f"{variance[0]:.1f}%" if variance else "N/A"
                logger.info(f"AIFS wave number: {wave_metrics.get('wave_number')} (amplitude: {amp_str}, variance: {var_str})")

        except Exception as e:
            logger.warning(f"Wave analysis failed for AIFS F{fhr:03d}: {e}")
            # Add None values to maintain array alignment
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(None)
            z500_waves_forecast["amplitudes"].append(None)
            z500_waves_forecast["variance_explained"].append(None)

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat(),
        "z500_waves": z500_waves,
        "z500_waves_forecast": z500_waves_forecast
    }


def fetch_ifs_data(region, forecast_hours, init_hour: Optional[int] = None):
    """Fetch temperature, precipitation, and MSLP data from ECMWF IFS.

    Note: IFS precipitation is cumulative from init time, so we convert
    to 6-hour interval totals to match GFS behavior.

    Args:
        region: Geographic region to fetch
        forecast_hours: List of forecast hours
        init_hour: Optional specific init hour (0, 6, 12, or 18). If None, gets the latest available.
    """
    ifs_model = IFSModel()

    if init_hour is not None:
        init_time = ifs_model.get_init_time_for_hour(init_hour)
    else:
        init_time = ifs_model.get_latest_init_time()

    temp_var = IFS_VARIABLES["t2m"]
    precip_var = IFS_VARIABLES["tp"]
    mslp_var = IFS_VARIABLES["mslp"]

    temps = []
    cumulative_precips = []
    mslps = []
    times = []

    # IFS range depends on init time: 00Z/12Z go to 240h, 06Z/18Z go to 144h
    if init_time.hour in [0, 12]:
        max_ifs_hour = 240  # 10 days for main synoptic runs
    else:
        max_ifs_hour = 144  # 6 days for intermediate runs
    ifs_hours = [h for h in forecast_hours if h <= max_ifs_hour]

    for hour in ifs_hours:
        # Temperature
        try:
            temp_data = ifs_model.fetch_data(temp_var, init_time, hour, region)
            temps.append(float(temp_data.values.mean()))
        except Exception as e:
            logger.warning(f"IFS temp fetch failed for F{hour}: {e}")
            temps.append(None)

        # Precipitation (cumulative from init)
        try:
            precip_data = ifs_model.fetch_data(precip_var, init_time, hour, region)
            cumulative_precips.append(float(precip_data.values.mean()))
        except Exception as e:
            logger.warning(f"IFS precip fetch failed for F{hour}: {e}")
            cumulative_precips.append(None)

        # MSLP
        try:
            mslp_data = ifs_model.fetch_data(mslp_var, init_time, hour, region)
            mslps.append(float(mslp_data.values.mean()))
        except Exception as e:
            logger.warning(f"IFS MSLP fetch failed for F{hour}: {e}")
            mslps.append(None)

        times.append((init_time + timedelta(hours=hour)).isoformat())

    # Convert cumulative precipitation to 6-hour interval totals
    precips = []
    for i, cum in enumerate(cumulative_precips):
        if cum is None:
            precips.append(None)
        elif i == 0:
            precips.append(cum)
        else:
            prev_cum = cumulative_precips[i - 1]
            if prev_cum is not None:
                interval = cum - prev_cum
                precips.append(max(0.0, interval))
            else:
                precips.append(None)

    # Pad with None for hours beyond IFS range
    while len(temps) < len(forecast_hours):
        temps.append(None)
        precips.append(None)
        mslps.append(None)
        times.append(None)

    # Calculate wave number forecast at 24-hour intervals (IFS only goes to 240 hours)
    z500_waves = None  # Current wave number (F000)
    z500_waves_forecast = {
        "times": [],
        "wave_numbers": [],
        "amplitudes": [],
        "variance_explained": []
    }

    # Calculate at 24-hour intervals (range depends on init time)
    wave_max_hour = 240 if init_time.hour in [0, 12] else 144
    wave_forecast_hours = list(range(0, wave_max_hour + 1, 24))
    global_region = Region("Global", (-180, 180, 20, 70))
    z500_var = IFS_VARIABLES["z500"]

    for fhr in wave_forecast_hours:
        try:
            logger.info(f"Calculating Rossby wave number for IFS F{fhr:03d}")
            z500_data = ifs_model.fetch_data(z500_var, init_time, fhr, global_region)
            wave_metrics = rossby_waves.calculate_wave_number(z500_data, latitude=55.0)

            # Store forecast time series
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(wave_metrics.get('wave_number'))
            # Get amplitude and variance of dominant wave
            amplitudes = wave_metrics.get('top_3_amplitudes', [])
            variance = wave_metrics.get('top_3_variance', [])
            z500_waves_forecast["amplitudes"].append(amplitudes[0] if amplitudes else None)
            z500_waves_forecast["variance_explained"].append(variance[0] if variance else None)

            # Store F000 as current wave number
            if fhr == 0:
                z500_waves = wave_metrics
                amp_str = f"{amplitudes[0]:.1f}m" if amplitudes else "N/A"
                var_str = f"{variance[0]:.1f}%" if variance else "N/A"
                logger.info(f"IFS wave number: {wave_metrics.get('wave_number')} (amplitude: {amp_str}, variance: {var_str})")

        except Exception as e:
            logger.warning(f"Wave analysis failed for IFS F{fhr:03d}: {e}")
            # Add None values to maintain array alignment
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(None)
            z500_waves_forecast["amplitudes"].append(None)
            z500_waves_forecast["variance_explained"].append(None)

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat(),
        "z500_waves": z500_waves,
        "z500_waves_forecast": z500_waves_forecast
    }


def fetch_observations(forecast_times: list[str], location_name: str) -> dict:
    """
    Fetch observed data matching forecast times.
    Only available for Fairfax, VA location.

    Args:
        forecast_times: List of ISO format datetime strings
        location_name: Location name (observations only available for Fairfax, VA)

    Returns:
        Dict with keys: temps, mslps, times (or empty dict if not available)
    """
    # Observations only available for Fairfax, VA
    if location_name != "Fairfax, VA":
        return {}

    try:
        # Fetch any missing WeatherLink data first
        weatherlink.fetch_missing_data(silent=True)

        # Get observations for forecast times
        return weatherlink.get_observations_for_forecast_times(forecast_times)
    except Exception as e:
        logger.warning(f"Error fetching observations: {e}")
        return {}


def calculate_verification_metrics(forecast_values: list, observed_values: list, forecast_times: list[str]) -> dict:
    """
    Calculate verification metrics (MAE, bias) for forecasts vs observations.
    Only considers past times where both forecast and observation exist.

    Args:
        forecast_values: List of forecast values
        observed_values: List of observed values
        forecast_times: List of ISO format datetime strings

    Returns:
        Dict with mae, bias, count (number of valid pairs)
    """
    now = datetime.now(timezone.utc)
    errors = []

    for i, time_str in enumerate(forecast_times):
        # Only verify past times
        try:
            valid_time = datetime.fromisoformat(time_str)
        except (ValueError, TypeError):
            continue

        if valid_time >= now:
            continue

        # Check if both forecast and observation exist
        if i >= len(forecast_values) or i >= len(observed_values):
            continue

        fcst = forecast_values[i]
        obs = observed_values[i]

        if fcst is not None and obs is not None:
            errors.append(fcst - obs)

    if not errors:
        return {"mae": None, "bias": None, "count": 0}

    mae = sum(abs(e) for e in errors) / len(errors)
    bias = sum(errors) / len(errors)

    return {"mae": round(mae, 2), "bias": round(bias, 2), "count": len(errors)}


def calculate_all_verification(gfs_data: dict, aifs_data: dict, observed: dict, ifs_data: dict = None) -> dict:
    """
    Calculate verification metrics for all variables.

    Returns:
        Dict with verification metrics for each model and variable
    """
    verification = {}

    if not observed or not observed.get("temps"):
        return verification

    forecast_times = gfs_data.get("times", [])

    # Temperature verification
    gfs_temp_metrics = calculate_verification_metrics(
        gfs_data.get("temps", []),
        observed.get("temps", []),
        forecast_times
    )
    aifs_temp_metrics = calculate_verification_metrics(
        aifs_data.get("temps", []),
        observed.get("temps", []),
        forecast_times
    )

    # MSLP verification
    gfs_mslp_metrics = calculate_verification_metrics(
        gfs_data.get("mslps", []),
        observed.get("mslps", []),
        forecast_times
    )
    aifs_mslp_metrics = calculate_verification_metrics(
        aifs_data.get("mslps", []),
        observed.get("mslps", []),
        forecast_times
    )

    verification = {
        "gfs_temp_mae": gfs_temp_metrics["mae"],
        "gfs_temp_bias": gfs_temp_metrics["bias"],
        "aifs_temp_mae": aifs_temp_metrics["mae"],
        "aifs_temp_bias": aifs_temp_metrics["bias"],
        "gfs_mslp_mae": gfs_mslp_metrics["mae"],
        "gfs_mslp_bias": gfs_mslp_metrics["bias"],
        "aifs_mslp_mae": aifs_mslp_metrics["mae"],
        "aifs_mslp_bias": aifs_mslp_metrics["bias"],
        "temp_count": gfs_temp_metrics["count"],
        "mslp_count": gfs_mslp_metrics["count"]
    }

    # Add IFS verification if available
    if ifs_data:
        ifs_temp_metrics = calculate_verification_metrics(
            ifs_data.get("temps", []),
            observed.get("temps", []),
            forecast_times
        )
        ifs_mslp_metrics = calculate_verification_metrics(
            ifs_data.get("mslps", []),
            observed.get("mslps", []),
            forecast_times
        )
        verification["ifs_temp_mae"] = ifs_temp_metrics["mae"]
        verification["ifs_temp_bias"] = ifs_temp_metrics["bias"]
        verification["ifs_mslp_mae"] = ifs_mslp_metrics["mae"]
        verification["ifs_mslp_bias"] = ifs_mslp_metrics["bias"]

    return verification


def calculate_lead_time_verification(location_name: str, days_back: Optional[int] = None, use_cumulative: bool = True) -> dict:
    """
    Calculate verification metrics grouped by lead time (forecast hour).
    Aggregates data across all historical runs.

    Returns:
        Dict with structure:
        {
            "lead_times": [6, 12, 18, ...],  # hours
            "lead_times_days": [0.25, 0.5, 0.75, ...],  # days
            "gfs_temp_mae": [1.2, 1.5, ...],
            "gfs_temp_bias": [0.3, 0.5, ...],
            "aifs_temp_mae": [1.1, 1.4, ...],
            "aifs_temp_bias": [0.2, 0.4, ...],
            "gfs_mslp_mae": [...],
            "gfs_mslp_bias": [...],
            "aifs_mslp_mae": [...],
            "aifs_mslp_bias": [...],
            "sample_counts": [10, 10, ...],  # number of forecast-obs pairs per lead time
            "run_count": 5  # total runs included
        }
    """
    db = load_forecasts_db()

    if location_name not in db:
        return {"error": "Location not found"}

    runs = db[location_name].get("runs", {})
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back) if days_back else None

    # Collect sums/counts by lead time (supports cumulative stats)
    errors_by_lead_time = {}

    runs_with_obs = 0
    obs_by_time = {}

    # Build WeatherLink observation lookup for all valid times across runs
    time_keys = set()
    for run_id, run_data in runs.items():
        gfs_data = run_data.get("gfs", {})
        init_time_str = gfs_data.get("init_time")
        if not init_time_str:
            continue
        try:
            init_time = datetime.fromisoformat(init_time_str)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_times = gfs_data.get("times", [])
        for time_str in forecast_times:
            try:
                valid_time = datetime.fromisoformat(time_str)
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if valid_time >= now:
                continue
            if cutoff_date and valid_time < cutoff_date:
                continue
            time_keys.add(valid_time.astimezone(timezone.utc).isoformat())

    obs_lookup = {}
    if time_keys:
        sorted_time_keys = sorted(time_keys)
        obs_times = []
        for t in sorted_time_keys:
            try:
                dt = datetime.fromisoformat(t)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                obs_times.append(dt.replace(tzinfo=None))
            except Exception:
                obs_times.append(None)
        obs_data = weatherlink.get_observations(obs_times, times_are_utc=True)
        obs_temps = obs_data.get("temps", [])
        obs_mslps = obs_data.get("mslps", [])
        for idx, t in enumerate(sorted_time_keys):
            obs_lookup[t] = {
                "temp": obs_temps[idx] if idx < len(obs_temps) else None,
                "mslp": obs_mslps[idx] if idx < len(obs_mslps) else None
            }

    def _ensure_lt(lt):
        if lt not in errors_by_lead_time:
            errors_by_lead_time[lt] = {
                "gfs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "aifs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "ifs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "nws_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "gfs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "aifs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "ifs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0}
            }

    def _add(lt, key, err):
        _ensure_lt(lt)
        errors_by_lead_time[lt][key]["sum_abs"] += abs(err)
        errors_by_lead_time[lt][key]["sum"] += err
        errors_by_lead_time[lt][key]["count"] += 1

    # Seed with cumulative stats if present
    if use_cumulative:
        cumulative = db[location_name].get("cumulative_stats", {}).get("by_lead_time", {})
        for lt_str, stats in cumulative.items():
            try:
                lt = int(lt_str)
            except ValueError:
                continue
            _ensure_lt(lt)
            for key in ["gfs_temp", "aifs_temp", "ifs_temp", "gfs_mslp", "aifs_mslp", "ifs_mslp"]:
                s = stats.get(key, {})
                errors_by_lead_time[lt][key]["sum_abs"] += s.get("sum_abs", 0.0)
                errors_by_lead_time[lt][key]["sum"] += s.get("sum", 0.0)
                errors_by_lead_time[lt][key]["count"] += s.get("count", 0)

    for run_id, run_data in runs.items():
        gfs_data = run_data.get("gfs", {})
        aifs_data = run_data.get("aifs", {})
        ifs_data = run_data.get("ifs", {})

        init_time_str = gfs_data.get("init_time")
        if not init_time_str:
            continue

        try:
            init_time = datetime.fromisoformat(init_time_str)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        runs_with_obs += 1

        forecast_times = gfs_data.get("times", [])
        gfs_temps = gfs_data.get("temps", [])
        aifs_temps = aifs_data.get("temps", [])
        ifs_temps = ifs_data.get("temps", []) if ifs_data else []
        gfs_mslps = gfs_data.get("mslps", [])
        aifs_mslps = aifs_data.get("mslps", [])
        ifs_mslps = ifs_data.get("mslps", []) if ifs_data else []
        run_has_obs = False
        for i, time_str in enumerate(forecast_times):
            try:
                valid_time = datetime.fromisoformat(time_str)
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            # Only include past times (verified)
            if valid_time >= now:
                continue
            if cutoff_date and valid_time < cutoff_date:
                continue

            # Calculate lead time in hours
            lead_time_hours = int((valid_time - init_time).total_seconds() / 3600)

            _ensure_lt(lead_time_hours)

            time_key = valid_time.astimezone(timezone.utc).isoformat()
            obs = obs_lookup.get(time_key, {})
            obs_temp = obs.get("temp")
            obs_mslp = obs.get("mslp")
            if obs_temp is not None or obs_mslp is not None:
                run_has_obs = True

            # Cache observed temperature by valid time (UTC) for NWS comparisons
            if obs_temp is not None:
                obs_by_time[time_key] = obs_temp

            # Collect temperature errors
            if i < len(gfs_temps):
                gfs_temp = gfs_temps[i]
                if gfs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "gfs_temp", gfs_temp - obs_temp)

            if i < len(aifs_temps):
                aifs_temp = aifs_temps[i]
                if aifs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "aifs_temp", aifs_temp - obs_temp)

            if i < len(ifs_temps):
                ifs_temp = ifs_temps[i]
                if ifs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "ifs_temp", ifs_temp - obs_temp)

            # Collect MSLP errors
            if i < len(gfs_mslps):
                gfs_mslp = gfs_mslps[i]
                if gfs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "gfs_mslp", gfs_mslp - obs_mslp)

            if i < len(aifs_mslps):
                aifs_mslp = aifs_mslps[i]
                if aifs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "aifs_mslp", aifs_mslp - obs_mslp)

            if i < len(ifs_mslps):
                ifs_mslp = ifs_mslps[i]
                if ifs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "ifs_mslp", ifs_mslp - obs_mslp)

        if run_has_obs:
            runs_with_obs += 1

    # NWS verification (temperature only) using cached forecast + WeatherLink observations
    nws_cache = load_nws_cache()
    runs = nws_cache.get("runs", []) if isinstance(nws_cache, dict) else []
    for run in runs:
        if not run.get("forecast") or not run.get("fetched_at"):
            continue
        try:
            nws_init = datetime.fromisoformat(run["fetched_at"])
            if nws_init.tzinfo is None:
                # Assume local time if tz missing, then convert to UTC
                local_tz = datetime.now().astimezone().tzinfo
                nws_init = nws_init.replace(tzinfo=local_tz).astimezone(timezone.utc)
            else:
                nws_init = nws_init.astimezone(timezone.utc)
        except Exception:
            continue

        # Anchor NWS lead times to the previous 6-hour cycle (UTC) to align with other models
        nws_cycle_init = nws_init.replace(minute=0, second=0, microsecond=0)
        cycle_hour = (nws_cycle_init.hour // 6) * 6
        nws_cycle_init = nws_cycle_init.replace(hour=cycle_hour)

        for entry in run.get("forecast", []):
            try:
                valid_time = datetime.fromisoformat(entry["datetime"])
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            utc_time = valid_time.astimezone(timezone.utc)
            if utc_time >= now:
                continue
            if cutoff_date and utc_time < cutoff_date:
                continue

            # Only include 6-hourly verification times (UTC)
            if utc_time.hour % 6 != 0:
                continue

            lead_time_hours = int((utc_time - nws_cycle_init).total_seconds() / 3600)
            obs_temp = obs_by_time.get(utc_time.isoformat())
            temp_val = entry.get("temperature")
            if obs_temp is None or temp_val is None:
                continue

            _add(lead_time_hours, "nws_temp", temp_val - obs_temp)

    # Calculate statistics for each lead time
    lead_times = sorted(errors_by_lead_time.keys())

    result = {
        "lead_times": lead_times,
        "lead_times_days": [h / 24 for h in lead_times],
        "gfs_temp_mae": [],
        "gfs_temp_bias": [],
        "aifs_temp_mae": [],
        "aifs_temp_bias": [],
        "ifs_temp_mae": [],
        "ifs_temp_bias": [],
        "nws_temp_mae": [],
        "nws_temp_bias": [],
        "gfs_mslp_mae": [],
        "gfs_mslp_bias": [],
        "aifs_mslp_mae": [],
        "aifs_mslp_bias": [],
        "ifs_mslp_mae": [],
        "ifs_mslp_bias": [],
        "temp_sample_counts": [],
        "mslp_sample_counts": [],
        "run_count": runs_with_obs
    }

    for lt in lead_times:
        errors = errors_by_lead_time[lt]

        # Temperature
        gfs_temp_errors = errors["gfs_temp"]
        aifs_temp_errors = errors["aifs_temp"]

        if gfs_temp_errors["count"] > 0:
            result["gfs_temp_mae"].append(round(gfs_temp_errors["sum_abs"] / gfs_temp_errors["count"], 2))
            result["gfs_temp_bias"].append(round(gfs_temp_errors["sum"] / gfs_temp_errors["count"], 2))
        else:
            result["gfs_temp_mae"].append(None)
            result["gfs_temp_bias"].append(None)

        if aifs_temp_errors["count"] > 0:
            result["aifs_temp_mae"].append(round(aifs_temp_errors["sum_abs"] / aifs_temp_errors["count"], 2))
            result["aifs_temp_bias"].append(round(aifs_temp_errors["sum"] / aifs_temp_errors["count"], 2))
        else:
            result["aifs_temp_mae"].append(None)
            result["aifs_temp_bias"].append(None)

        ifs_temp_errors = errors["ifs_temp"]
        if ifs_temp_errors["count"] > 0:
            result["ifs_temp_mae"].append(round(ifs_temp_errors["sum_abs"] / ifs_temp_errors["count"], 2))
            result["ifs_temp_bias"].append(round(ifs_temp_errors["sum"] / ifs_temp_errors["count"], 2))
        else:
            result["ifs_temp_mae"].append(None)
            result["ifs_temp_bias"].append(None)

        nws_temp_errors = errors.get("nws_temp", None)
        if nws_temp_errors and nws_temp_errors["count"] > 0:
            result["nws_temp_mae"].append(round(nws_temp_errors["sum_abs"] / nws_temp_errors["count"], 2))
            result["nws_temp_bias"].append(round(nws_temp_errors["sum"] / nws_temp_errors["count"], 2))
        else:
            result["nws_temp_mae"].append(None)
            result["nws_temp_bias"].append(None)

        result["temp_sample_counts"].append(max(
            gfs_temp_errors["count"],
            aifs_temp_errors["count"],
            ifs_temp_errors["count"],
            nws_temp_errors["count"] if nws_temp_errors else 0
        ))

        # MSLP
        gfs_mslp_errors = errors["gfs_mslp"]
        aifs_mslp_errors = errors["aifs_mslp"]

        if gfs_mslp_errors["count"] > 0:
            result["gfs_mslp_mae"].append(round(gfs_mslp_errors["sum_abs"] / gfs_mslp_errors["count"], 2))
            result["gfs_mslp_bias"].append(round(gfs_mslp_errors["sum"] / gfs_mslp_errors["count"], 2))
        else:
            result["gfs_mslp_mae"].append(None)
            result["gfs_mslp_bias"].append(None)

        if aifs_mslp_errors["count"] > 0:
            result["aifs_mslp_mae"].append(round(aifs_mslp_errors["sum_abs"] / aifs_mslp_errors["count"], 2))
            result["aifs_mslp_bias"].append(round(aifs_mslp_errors["sum"] / aifs_mslp_errors["count"], 2))
        else:
            result["aifs_mslp_mae"].append(None)
            result["aifs_mslp_bias"].append(None)

        ifs_mslp_errors = errors["ifs_mslp"]
        if ifs_mslp_errors["count"] > 0:
            result["ifs_mslp_mae"].append(round(ifs_mslp_errors["sum_abs"] / ifs_mslp_errors["count"], 2))
            result["ifs_mslp_bias"].append(round(ifs_mslp_errors["sum"] / ifs_mslp_errors["count"], 2))
        else:
            result["ifs_mslp_mae"].append(None)
            result["ifs_mslp_bias"].append(None)

        result["mslp_sample_counts"].append(max(
            gfs_mslp_errors["count"],
            aifs_mslp_errors["count"],
            ifs_mslp_errors["count"]
        ))

    return result


def calculate_verification_time_series(location_name: str, variable: str = 'temp', lead_time_hours: int = 24, days_back: int = 30) -> dict:
    """
    Calculate verification time series (daily MAE and bias) for a location.

    Args:
        location_name: Location name (e.g., "Fairfax, VA")
        variable: Variable to verify ('temp' or 'mslp')
        lead_time_hours: Forecast lead time in hours
        days_back: Number of days to look back

    Returns:
        Dict with structure:
        {
            "dates": ["2026-01-20", "2026-01-21", ...],
            "gfs": {"mae": [2.1, 2.3, ...], "bias": [0.5, 0.3, ...], "counts": [1, 1, ...]},
            "aifs": {"mae": [2.0, 2.2, ...], "bias": [0.4, 0.2, ...], "counts": [1, 1, ...]},
            "ifs": {"mae": [1.9, 2.1, ...], "bias": [0.3, 0.1, ...], "counts": [1, 1, ...]}
        }
    """
    db = load_forecasts_db()

    if location_name not in db:
        return {"error": "Location not found"}

    runs = db[location_name].get("runs", {})
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back)

    # Variable mapping
    var_map = {
        'temp': ('temps', 'temps'),
        'mslp': ('mslps', 'mslps')
    }

    if variable not in var_map and variable != 'precip':
        return {"error": "Invalid variable"}

    if variable != 'precip':
        fcst_key, obs_key = var_map[variable]
        obs_lookup_key = 'temp' if variable == 'temp' else 'mslp'
    else:
        if location_name != "Fairfax, VA":
            return {"error": "Precip time series only available for Fairfax, VA"}

    # Collect sums/counts grouped by date
    # Key: date string (YYYY-MM-DD)
    errors_by_date = {}

    def _ensure_date(date_key):
        if date_key not in errors_by_date:
            errors_by_date[date_key] = {
                "gfs": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "aifs": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "ifs": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "nws": {"sum_abs": 0.0, "sum": 0.0, "count": 0}
            }

    def _add(date_key, model_key, err):
        _ensure_date(date_key)
        errors_by_date[date_key][model_key]["sum_abs"] += abs(err)
        errors_by_date[date_key][model_key]["sum"] += err
        errors_by_date[date_key][model_key]["count"] += 1

    # Seed from cached daily stats (keeps long-term trend after trimming)
    if variable != 'precip':
        cached_ts = db[location_name].get("cumulative_stats", {}).get("time_series", {})
        for date_key, model_data in cached_ts.items():
            try:
                date_obj = datetime.fromisoformat(date_key).date()
            except ValueError:
                continue
            if date_obj < cutoff_date.date():
                continue

            _ensure_date(date_key)
            model_key = None
            var_key = 'temp' if variable == 'temp' else 'mslp'

            for model_key in ["gfs", "aifs", "ifs"]:
                for lt_key, stats in model_data.get(model_key, {}).get(var_key, {}).items():
                    # Keep only the selected lead time
                    if int(lt_key) != lead_time_hours:
                        continue
                    errors_by_date[date_key][model_key]["sum_abs"] += stats.get("sum_abs", 0.0)
                    errors_by_date[date_key][model_key]["sum"] += stats.get("sum", 0.0)
                    errors_by_date[date_key][model_key]["count"] += stats.get("count", 0)

    if variable != 'precip':
        # Build WeatherLink observation lookup for valid times in window
        time_keys = set()
        for run_id, run_data in runs.items():
            gfs_data = run_data.get("gfs", {})
            init_time_str = gfs_data.get("init_time")
            if not init_time_str:
                continue
            try:
                init_time = datetime.fromisoformat(init_time_str)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            forecast_times = gfs_data.get("times", [])
            for time_str in forecast_times:
                try:
                    valid_time = datetime.fromisoformat(time_str)
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                if valid_time >= now or valid_time < cutoff_date:
                    continue

                lead_time = int((valid_time - init_time).total_seconds() / 3600)
                if lead_time != lead_time_hours:
                    continue

                time_keys.add(valid_time.astimezone(timezone.utc).isoformat())

        obs_lookup = {}
        if time_keys:
            sorted_time_keys = sorted(time_keys)
            obs_times = []
            for t in sorted_time_keys:
                try:
                    dt = datetime.fromisoformat(t)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    obs_times.append(dt.replace(tzinfo=None))
                except Exception:
                    obs_times.append(None)
            obs_data = weatherlink.get_observations(obs_times, times_are_utc=True)
            obs_temps = obs_data.get("temps", [])
            obs_mslps = obs_data.get("mslps", [])
            for idx, t in enumerate(sorted_time_keys):
                obs_lookup[t] = {
                    "temp": obs_temps[idx] if idx < len(obs_temps) else None,
                    "mslp": obs_mslps[idx] if idx < len(obs_mslps) else None
                }

    # Build observed lookup by valid time for NWS verification (temp only)
    obs_by_time = {}

    if variable != 'precip':
        for run_id, run_data in runs.items():
            gfs_data = run_data.get("gfs", {})
            aifs_data = run_data.get("aifs", {})
            ifs_data = run_data.get("ifs", {})

            init_time_str = gfs_data.get("init_time")
            if not init_time_str:
                continue

            try:
                init_time = datetime.fromisoformat(init_time_str)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Skip runs outside the time window
            if init_time < cutoff_date:
                continue

            forecast_times = gfs_data.get("times", [])
            gfs_values = gfs_data.get(fcst_key, [])
            aifs_values = aifs_data.get(fcst_key, [])
            ifs_values = ifs_data.get(fcst_key, []) if ifs_data else []
            for i, time_str in enumerate(forecast_times):
                try:
                    valid_time = datetime.fromisoformat(time_str)
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                # Only include past times
                if valid_time >= now:
                    continue
                if cutoff_date and valid_time < cutoff_date:
                    continue

                # Calculate lead time in hours
                lead_time = int((valid_time - init_time).total_seconds() / 3600)

                # Only include the requested lead time
                if lead_time != lead_time_hours:
                    continue

                time_key = valid_time.astimezone(timezone.utc).isoformat()
                obs_val = obs_lookup.get(time_key, {}).get(obs_lookup_key)
                if obs_val is not None:
                    obs_by_time[time_key] = obs_val

                # Use the date of the valid time for grouping
                date_key = valid_time.date().isoformat()
                _ensure_date(date_key)

                # Collect errors for each model
                if i < len(gfs_values):
                    gfs_val = gfs_values[i]
                    if gfs_val is not None and obs_val is not None:
                        _add(date_key, "gfs", gfs_val - obs_val)

                if i < len(aifs_values):
                    aifs_val = aifs_values[i]
                    if aifs_val is not None and obs_val is not None:
                        _add(date_key, "aifs", aifs_val - obs_val)

                if ifs_values and i < len(ifs_values):
                    ifs_val = ifs_values[i]
                    if ifs_val is not None and obs_val is not None:
                        _add(date_key, "ifs", ifs_val - obs_val)
    else:
        # Precip verification based on 12Z CoCoRaHS daily totals
        local_now = weatherlink.utc_to_eastern(now.replace(tzinfo=None))
        obs_end = local_now.date() - timedelta(days=1)
        start_date = (local_now.date() - timedelta(days=1)) - timedelta(days=days_back - 1)

        obs_data = fetch_cocorahs_daily_precip(
            COCORAHs_STATION_ID,
            start_date.isoformat(),
            obs_end.isoformat()
        )

        def compute_daily_total_for_run(model_data, day_end_utc):
            times = model_data.get("times", [])
            precips = model_data.get("precips", [])
            if not times or not precips:
                return None, 0
            window_start = day_end_utc - timedelta(hours=24)
            total = 0.0
            count = 0
            for i, time_str in enumerate(times):
                try:
                    valid_time = datetime.fromisoformat(time_str)
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue
                if window_start < valid_time <= day_end_utc:
                    if i < len(precips) and precips[i] is not None:
                        total += precips[i]
                        count += 1
            if count >= 3:
                return total, count
            return None, count

        def select_best_total_for_date(model_key, day_end_utc):
            best_total = None
            best_init = None
            for run_id, run_data in runs.items():
                model_data = run_data.get(model_key, {})
                init_time_str = model_data.get("init_time") or run_data.get("gfs", {}).get("init_time")
                if not init_time_str:
                    continue
                try:
                    init_time = datetime.fromisoformat(init_time_str)
                    if init_time.tzinfo is None:
                        init_time = init_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                # Check that this run provides the day_end_utc valid time at the selected lead time
                times = model_data.get("times", [])
                has_valid = False
                for t in times:
                    try:
                        valid_time = datetime.fromisoformat(t)
                        if valid_time.tzinfo is None:
                            valid_time = valid_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                    if valid_time == day_end_utc:
                        lead_time = int((valid_time - init_time).total_seconds() / 3600)
                        if lead_time == lead_time_hours:
                            has_valid = True
                        break
                if not has_valid:
                    continue

                total, count = compute_daily_total_for_run(model_data, day_end_utc)
                if total is None:
                    continue

                if best_init is None or init_time > best_init:
                    best_init = init_time
                    best_total = total
            return best_total

        # NWS precip totals from cached hourly forecast
        def compute_nws_total_for_run(run, day_end_utc):
            forecast = run.get("forecast", [])
            if not forecast:
                return None, 0
            window_start = day_end_utc - timedelta(hours=24)
            total_mm = 0.0
            count = 0
            for entry in forecast:
                try:
                    valid_time = datetime.fromisoformat(entry["datetime"])
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if window_start < valid_time <= day_end_utc:
                    val = entry.get("precip_mm")
                    if val is None:
                        continue
                    total_mm += val
                    count += 1
            if count >= 3:
                return total_mm / 25.4, count
            return None, count

        def select_best_nws_total(day_end_utc):
            nws_cache = load_nws_cache()
            runs = nws_cache.get("runs", []) if isinstance(nws_cache, dict) else []
            best_total = None
            best_init = None
            for run in runs:
                fetched_at = run.get("fetched_at")
                if not fetched_at:
                    continue
                try:
                    nws_init = datetime.fromisoformat(fetched_at)
                    if nws_init.tzinfo is None:
                        local_tz = datetime.now().astimezone().tzinfo
                        nws_init = nws_init.replace(tzinfo=local_tz).astimezone(timezone.utc)
                    else:
                        nws_init = nws_init.astimezone(timezone.utc)
                except Exception:
                    continue

                # Anchor NWS lead times to the previous 6-hour cycle (UTC)
                nws_cycle_init = nws_init.replace(minute=0, second=0, microsecond=0)
                cycle_hour = (nws_cycle_init.hour // 6) * 6
                nws_cycle_init = nws_cycle_init.replace(hour=cycle_hour)

                lead_time = int((day_end_utc - nws_cycle_init).total_seconds() / 3600)
                if lead_time != lead_time_hours:
                    continue

                total_in, count = compute_nws_total_for_run(run, day_end_utc)
                if total_in is None:
                    continue

                if best_init is None or nws_cycle_init > best_init:
                    best_init = nws_cycle_init
                    best_total = total_in
            return best_total

        d = start_date
        while d <= obs_end:
            date_key = d.isoformat()
            obs_val = obs_data.get(date_key)
            if obs_val is None:
                d += timedelta(days=1)
                continue
            _ensure_date(date_key)
            day_end = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
            gfs_total = select_best_total_for_date("gfs", day_end)
            aifs_total = select_best_total_for_date("aifs", day_end)
            ifs_total = select_best_total_for_date("ifs", day_end)
            nws_total = select_best_nws_total(day_end)

            if gfs_total is not None:
                _add(date_key, "gfs", gfs_total - obs_val)
            if aifs_total is not None:
                _add(date_key, "aifs", aifs_total - obs_val)
            if ifs_total is not None:
                _add(date_key, "ifs", ifs_total - obs_val)
            if nws_total is not None:
                _add(date_key, "nws", nws_total - obs_val)
            d += timedelta(days=1)

    # NWS verification (temperature only) using cached forecast + WeatherLink observations
    if variable == 'temp':
        nws_cache = load_nws_cache()
        runs = nws_cache.get("runs", []) if isinstance(nws_cache, dict) else []
        for run in runs:
            if not run.get("forecast") or not run.get("fetched_at"):
                continue
            try:
                nws_init = datetime.fromisoformat(run["fetched_at"])
                if nws_init.tzinfo is None:
                    local_tz = datetime.now().astimezone().tzinfo
                    nws_init = nws_init.replace(tzinfo=local_tz).astimezone(timezone.utc)
                else:
                    nws_init = nws_init.astimezone(timezone.utc)
            except Exception:
                continue

            # Anchor NWS lead times to the previous 6-hour cycle (UTC) to align with other models
            nws_cycle_init = nws_init.replace(minute=0, second=0, microsecond=0)
            cycle_hour = (nws_cycle_init.hour // 6) * 6
            nws_cycle_init = nws_cycle_init.replace(hour=cycle_hour)

            for entry in run.get("forecast", []):
                try:
                    valid_time = datetime.fromisoformat(entry["datetime"])
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                utc_time = valid_time.astimezone(timezone.utc)

                # Only include past times
                if utc_time >= now:
                    continue

                # Only include 6-hourly verification times (UTC)
                if utc_time.hour % 6 != 0:
                    continue

                # Calculate lead time in hours
                lead_time = int((utc_time - nws_cycle_init).total_seconds() / 3600)
                if lead_time != lead_time_hours:
                    continue

                temp_val = entry.get("temperature")
                obs_val = obs_by_time.get(utc_time.isoformat())
                if temp_val is None or obs_val is None:
                    continue

                date_key = utc_time.date().isoformat()
                _ensure_date(date_key)
                _add(date_key, "nws", temp_val - obs_val)

    # Calculate daily MAE and Bias
    dates = sorted(errors_by_date.keys())

    result = {
        "dates": dates,
        "gfs": {"mae": [], "bias": [], "counts": []},
        "aifs": {"mae": [], "bias": [], "counts": []},
        "ifs": {"mae": [], "bias": [], "counts": []}
    }
    if variable in ('temp', 'precip'):
        result["nws"] = {"mae": [], "bias": [], "counts": []}

    for date in dates:
        for model in ["gfs", "aifs", "ifs"]:
            stats = errors_by_date[date][model]
            if stats["count"] > 0:
                mae = stats["sum_abs"] / stats["count"]
                bias = stats["sum"] / stats["count"]
                result[model]["mae"].append(round(mae, 2))
                result[model]["bias"].append(round(bias, 2))
                result[model]["counts"].append(stats["count"])
            else:
                result[model]["mae"].append(None)
                result[model]["bias"].append(None)
                result[model]["counts"].append(0)

        if variable in ('temp', 'precip'):
            stats = errors_by_date[date]["nws"]
            if stats["count"] > 0:
                mae = stats["sum_abs"] / stats["count"]
                bias = stats["sum"] / stats["count"]
                result["nws"]["mae"].append(round(mae, 2))
                result["nws"]["bias"].append(round(bias, 2))
                result["nws"]["counts"].append(stats["count"])
            else:
                result["nws"]["mae"].append(None)
                result["nws"]["bias"].append(None)
                result["nws"]["counts"].append(0)

    return result


def save_forecast_data(location_name, gfs_data, aifs_data, observed=None, verification=None, ifs_data=None):
    """
    Save forecast data to the central JSON file.
    Stores each model run as a separate entry keyed by init_time.
    """
    db = load_forecasts_db()

    # Initialize location if needed
    if location_name not in db:
        db[location_name] = {"runs": {}, "latest_run": None, "cumulative_stats": {}}

    # Use GFS init_time as the run key (all models should have same init time)
    run_key = gfs_data.get("init_time")
    if not run_key:
        logger.warning("No init_time in GFS data, cannot save run")
        return str(FORECASTS_FILE)

    # Create run entry
    run_data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "gfs": gfs_data,
        "aifs": aifs_data
    }

    # Add IFS data if available
    if ifs_data:
        run_data["ifs"] = ifs_data

    # Add observed data if available
    if observed:
        run_data["observed"] = observed

    # Add verification metrics if available
    if verification:
        run_data["verification"] = verification

    # Store the run
    db[location_name]["runs"][run_key] = run_data
    db[location_name]["latest_run"] = run_key

    # Trim runs while preserving long-term stats
    prune_forecasts_db(db, location_name, retention_days=FORECAST_RETENTION_DAYS)

    save_forecasts_db(db)
    logger.info(f"Saved run {run_key} for {location_name} (total runs: {len(db[location_name]['runs'])})")
    return str(FORECASTS_FILE)


def _init_cumulative_stats(location_data: dict):
    if "cumulative_stats" not in location_data:
        location_data["cumulative_stats"] = {}
    stats = location_data["cumulative_stats"]
    if "by_lead_time" not in stats:
        stats["by_lead_time"] = {}
    if "time_series" not in stats:
        stats["time_series"] = {}
    if "total_runs" not in stats:
        stats["total_runs"] = 0
    if "generated_at" not in stats:
        stats["generated_at"] = None


def _accumulate_fairfax_stats_from_run(location_data: dict, run_data: dict):
    """
    Accumulate long-term verification stats from a run before it is trimmed.
    Stores sums and counts by lead time for temp/mslp and model.
    """
    _init_cumulative_stats(location_data)
    cumulative = location_data["cumulative_stats"]["by_lead_time"]

    observed = run_data.get("observed")
    if not observed or not observed.get("temps"):
        return

    gfs_data = run_data.get("gfs", {})
    aifs_data = run_data.get("aifs", {})
    ifs_data = run_data.get("ifs", {})

    init_time_str = gfs_data.get("init_time")
    if not init_time_str:
        return

    try:
        init_time = datetime.fromisoformat(init_time_str)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return

    forecast_times = gfs_data.get("times", [])
    gfs_temps = gfs_data.get("temps", [])
    aifs_temps = aifs_data.get("temps", [])
    ifs_temps = ifs_data.get("temps", []) if ifs_data else []
    gfs_mslps = gfs_data.get("mslps", [])
    aifs_mslps = aifs_data.get("mslps", [])
    ifs_mslps = ifs_data.get("mslps", []) if ifs_data else []
    obs_temps = observed.get("temps", [])
    obs_mslps = observed.get("mslps", [])

    for i, time_str in enumerate(forecast_times):
        try:
            valid_time = datetime.fromisoformat(time_str)
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        lead_time_hours = int((valid_time - init_time).total_seconds() / 3600)
        lt_key = str(lead_time_hours)
        if lt_key not in cumulative:
            cumulative[lt_key] = {
                "gfs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "aifs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "ifs_temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "gfs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "aifs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "ifs_mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0}
            }

        def _acc(model_key, fcst_vals, obs_vals):
            if i < len(fcst_vals) and i < len(obs_vals):
                fcst_val = fcst_vals[i]
                obs_val = obs_vals[i]
                if fcst_val is not None and obs_val is not None:
                    err = fcst_val - obs_val
                    cumulative[lt_key][model_key]["sum_abs"] += abs(err)
                    cumulative[lt_key][model_key]["sum"] += err
                    cumulative[lt_key][model_key]["count"] += 1

        _acc("gfs_temp", gfs_temps, obs_temps)
        _acc("aifs_temp", aifs_temps, obs_temps)
        if ifs_temps:
            _acc("ifs_temp", ifs_temps, obs_temps)

        _acc("gfs_mslp", gfs_mslps, obs_mslps)
        _acc("aifs_mslp", aifs_mslps, obs_mslps)
        if ifs_mslps:
            _acc("ifs_mslp", ifs_mslps, obs_mslps)


def _accumulate_fairfax_daily_stats_from_run(location_data: dict, run_data: dict):
    """
    Accumulate per-day MAE/bias stats from a run into the time_series cache.
    """
    _init_cumulative_stats(location_data)
    time_series = location_data["cumulative_stats"]["time_series"]

    observed = run_data.get("observed")
    if not observed or not observed.get("temps"):
        return

    gfs_data = run_data.get("gfs", {})
    aifs_data = run_data.get("aifs", {})
    ifs_data = run_data.get("ifs", {})

    init_time_str = gfs_data.get("init_time")
    if not init_time_str:
        return

    try:
        init_time = datetime.fromisoformat(init_time_str)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return

    forecast_times = gfs_data.get("times", [])
    gfs_temps = gfs_data.get("temps", [])
    aifs_temps = aifs_data.get("temps", [])
    ifs_temps = ifs_data.get("temps", []) if ifs_data else []
    gfs_mslps = gfs_data.get("mslps", [])
    aifs_mslps = aifs_data.get("mslps", [])
    ifs_mslps = ifs_data.get("mslps", []) if ifs_data else []
    obs_temps = observed.get("temps", [])
    obs_mslps = observed.get("mslps", [])

    def _ensure(date_key, model, var, lt_key):
        time_series.setdefault(date_key, {})
        time_series[date_key].setdefault(model, {})
        time_series[date_key][model].setdefault(var, {})
        time_series[date_key][model][var].setdefault(lt_key, {"sum_abs": 0.0, "sum": 0.0, "count": 0})

    def _add(date_key, model, var, lt_key, err):
        _ensure(date_key, model, var, lt_key)
        stats = time_series[date_key][model][var][lt_key]
        stats["sum_abs"] += abs(err)
        stats["sum"] += err
        stats["count"] += 1

    for i, time_str in enumerate(forecast_times):
        try:
            valid_time = datetime.fromisoformat(time_str)
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        date_key = valid_time.date().isoformat()
        lead_time_hours = int((valid_time - init_time).total_seconds() / 3600)
        lt_key = str(lead_time_hours)

        if i < len(gfs_temps) and i < len(obs_temps):
            fcst_val = gfs_temps[i]
            obs_val = obs_temps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "gfs", "temp", lt_key, fcst_val - obs_val)

        if i < len(aifs_temps) and i < len(obs_temps):
            fcst_val = aifs_temps[i]
            obs_val = obs_temps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "aifs", "temp", lt_key, fcst_val - obs_val)

        if ifs_temps and i < len(ifs_temps) and i < len(obs_temps):
            fcst_val = ifs_temps[i]
            obs_val = obs_temps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "ifs", "temp", lt_key, fcst_val - obs_val)

        if i < len(gfs_mslps) and i < len(obs_mslps):
            fcst_val = gfs_mslps[i]
            obs_val = obs_mslps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "gfs", "mslp", lt_key, fcst_val - obs_val)

        if i < len(aifs_mslps) and i < len(obs_mslps):
            fcst_val = aifs_mslps[i]
            obs_val = obs_mslps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "aifs", "mslp", lt_key, fcst_val - obs_val)

        if ifs_mslps and i < len(ifs_mslps) and i < len(obs_mslps):
            fcst_val = ifs_mslps[i]
            obs_val = obs_mslps[i]
            if fcst_val is not None and obs_val is not None:
                _add(date_key, "ifs", "mslp", lt_key, fcst_val - obs_val)


def prune_forecasts_db(db: dict, location_name: str, retention_days: int = 20):
    """
    Trim old runs while preserving long-term stats.
    """
    if location_name not in db:
        return

    location_data = db[location_name]
    _init_cumulative_stats(location_data)

    runs = location_data.get("runs", {})
    if not runs:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    to_delete = []

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if init_time < cutoff:
            _accumulate_fairfax_stats_from_run(location_data, run_data)
            _accumulate_fairfax_daily_stats_from_run(location_data, run_data)
            to_delete.append(run_key)

    for run_key in to_delete:
        runs.pop(run_key, None)

    if to_delete:
        location_data["cumulative_stats"]["total_runs"] += len(to_delete)
        location_data["cumulative_stats"]["generated_at"] = datetime.now(timezone.utc).isoformat()


def get_latest_init_times(init_hour: Optional[int] = None):
    """
    Get the latest available init times from both models without fetching full data.

    Args:
        init_hour: Optional specific init hour (0, 6, 12, or 18). If None, gets the latest available.
    """
    gfs_model = GFSModel()
    aifs_model = AIFSModel()

    if init_hour is not None:
        gfs_init = gfs_model.get_init_time_for_hour(init_hour)
        aifs_init = aifs_model.get_init_time_for_hour(init_hour)
    else:
        gfs_init = gfs_model.get_latest_init_time()
        aifs_init = aifs_model.get_latest_init_time()

    # Use GFS init time as the primary (GFS and AIFS usually run at same times)
    return gfs_init.isoformat(), aifs_init.isoformat()


def check_if_already_fetched(location_name, gfs_init, aifs_init):
    """Check if we already have data for these init times."""
    db = load_forecasts_db()

    if location_name not in db:
        return False, "No cached data for this location"

    runs = db[location_name].get("runs", {})

    # Check if this run already exists
    if gfs_init in runs:
        return True, f"Already have this run (GFS: {gfs_init}, AIFS: {aifs_init})"

    return False, f"New model run available (GFS: {gfs_init}, AIFS: {aifs_init})"


@app.route('/')
def dashboard():
    """Main page - current conditions."""
    return render_template('current.html')


@app.route('/forecasts')
def forecast_comparison():
    """Forecast comparison page with Fairfax and ASOS tabs."""
    return render_template('forecast_comparison.html')


@app.route('/dashboard')
def forecast_dashboard():
    """Fairfax verification dashboard."""
    return render_template(
        'dashboard.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA",
        forecast_days=15
    )


@app.route('/historical')
def historical_weather():
    """Historical weather page with date range queries and charts."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    if request.args.get('start'):
        try:
            start_date = datetime.strptime(request.args.get('start'), '%Y-%m-%d')
        except ValueError:
            pass

    if request.args.get('end'):
        try:
            end_date = datetime.strptime(request.args.get('end'), '%Y-%m-%d')
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            pass

    summary = {}
    if local_weather_data:
        try:
            summary = local_weather_data.get_period_summary(start_date, end_date)
        except Exception as e:
            logger.warning(f"Historical summary error: {e}")

    return render_template(
        'historical.html',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        summary=summary
    )


@app.route('/sync')
def sync():
    """Sync page - fetch new data from models."""
    return render_template(
        'sync.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA"
    )


@app.route('/current')
def current_conditions():
    """Current weather conditions page."""
    return render_template('current.html')


@app.route('/forecast')
def run_forecast_page():
    """Forecast comparison page with Fairfax and ASOS tabs."""
    return render_template('forecast_comparison.html')


@app.route('/run-forecast')
def running_conditions_page():
    """Run forecast page - best times to run based on weather."""
    return render_template('forecast.html')


@app.route('/api/forecast')
def api_forecast():
    """API endpoint to fetch forecast data."""
    location_name = request.args.get('location', 'Fairfax, VA')
    days = int(request.args.get('days', 15))
    force = request.args.get('force', 'false').lower() == 'true'

    if location_name not in LOCATIONS:
        return jsonify({"success": False, "error": f"Unknown location: {location_name}"})

    bounds = LOCATIONS[location_name]
    region = Region(location_name, bounds)

    try:
        # Check if we already have the latest data
        gfs_init, aifs_init = get_latest_init_times()
        already_fetched, message = check_if_already_fetched(location_name, gfs_init, aifs_init)

        if already_fetched and not force:
            logger.info(f"Using cached data for {location_name}: {message}")
            db = load_forecasts_db()
            loc_data = db[location_name]
            # Get the run data for this init time
            run_data = loc_data.get("runs", {}).get(gfs_init, {})
            return jsonify({
                "success": True,
                "location": location_name,
                "run_id": gfs_init,
                "gfs": run_data.get("gfs"),
                "aifs": run_data.get("aifs"),
                "ifs": run_data.get("ifs"),
                "observed": run_data.get("observed"),
                "verification": run_data.get("verification"),
                "fetched_at": run_data.get("fetched_at"),
                "cached": True,
                "message": message,
                "saved_to": str(FORECASTS_FILE),
                "total_runs": len(loc_data.get("runs", {}))
            })

        # Calculate forecast hours (6-hour intervals, max 360 for AIFS)
        max_hours = min(days * 24, 360)
        forecast_hours = list(range(0, max_hours + 1, 6))

        logger.info(f"Fetching forecast for {location_name}, {days} days - {message}")

        gfs_data = fetch_gfs_data(region, forecast_hours)
        aifs_data = fetch_aifs_data(region, forecast_hours)
        ifs_data = fetch_ifs_data(region, forecast_hours)

        # Fetch observations for Fairfax, VA
        observed = fetch_observations(gfs_data.get("times", []), location_name)

        # Calculate verification metrics
        verification = calculate_all_verification(gfs_data, aifs_data, observed, ifs_data)

        # Save to JSON file
        saved_file = save_forecast_data(location_name, gfs_data, aifs_data, observed, verification, ifs_data)

        return jsonify({
            "success": True,
            "location": location_name,
            "gfs": gfs_data,
            "aifs": aifs_data,
            "ifs": ifs_data,
            "observed": observed if observed else None,
            "verification": verification if verification else None,
            "cached": False,
            "message": "Fetched new data",
            "saved_to": saved_file
        })

    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/latest-forecast')
def api_latest_forecast():
    """Get the latest saved forecast for a location."""
    location_name = request.args.get('location', 'Fairfax, VA')

    db = load_forecasts_db()

    if location_name not in db:
        return jsonify({
            "success": False,
            "error": f"No saved forecast for {location_name}. Please sync data first."
        })

    try:
        loc_data = db[location_name]
        latest_run = loc_data.get("latest_run")

        if not latest_run or latest_run not in loc_data.get("runs", {}):
            return jsonify({
                "success": False,
                "error": f"No saved forecast for {location_name}. Please sync data first."
            })

        run_data = loc_data["runs"][latest_run]
        return jsonify({
            "success": True,
            "location": location_name,
            "run_id": latest_run,
            "fetched_at": run_data.get("fetched_at"),
            "gfs": run_data.get("gfs"),
            "aifs": run_data.get("aifs"),
            "ifs": run_data.get("ifs"),
            "observed": run_data.get("observed"),
            "verification": run_data.get("verification"),
            "total_runs": len(loc_data.get("runs", {}))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/saved-forecasts')
def api_saved_forecasts():
    """List all saved forecasts from the database."""
    db = load_forecasts_db()

    forecasts = []
    for location, loc_data in db.items():
        runs = loc_data.get("runs", {})
        for run_id, run_data in runs.items():
            forecasts.append({
                "location": location,
                "run_id": run_id,
                "fetched_at": run_data.get("fetched_at", "Unknown"),
                "gfs_init": run_data.get("gfs", {}).get("init_time"),
                "aifs_init": run_data.get("aifs", {}).get("init_time"),
                "ifs_init": run_data.get("ifs", {}).get("init_time") if run_data.get("ifs") else None,
                "has_observations": "observed" in run_data and run_data["observed"]
            })

    # Sort by run_id (init_time) descending
    forecasts.sort(key=lambda x: x.get("run_id", ""), reverse=True)

    return jsonify({"success": True, "forecasts": forecasts})


@app.route('/api/verification-by-lead-time')
def api_verification_by_lead_time():
    """
    Get verification metrics grouped by lead time.
    Shows how forecast accuracy degrades with increasing lead time.
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    period = request.args.get('period', 'all').lower()

    try:
        cache = load_verif_lead_cache()
        entries = cache.get("entries", {})
        cache_key = f"{location_name}|{period}"
        source_mtimes = _get_verif_ts_source_mtimes()

        cached = entries.get(cache_key)
        if cached:
            cached_sources = cached.get("source_mtimes", {})
            relevant_keys = ["forecasts"]
            if period == "monthly":
                # monthly uses recent obs data
                relevant_keys.append("forecasts")
            if period in ("all", "monthly"):
                relevant_keys.append("nws")

            if all(cached_sources.get(k) == source_mtimes.get(k) for k in relevant_keys):
                return jsonify({
                    "success": True,
                    "location": location_name,
                    "verification": cached.get("verification", {}),
                    "period": period
                })

        if period == 'monthly':
            result = calculate_lead_time_verification(location_name, days_back=30, use_cumulative=False)
        else:
            result = calculate_lead_time_verification(location_name, use_cumulative=True)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        entries[cache_key] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mtimes": source_mtimes,
            "verification": result
        }
        cache["entries"] = entries
        save_verif_lead_cache(cache)

        return jsonify({
            "success": True,
            "location": location_name,
            "verification": result,
            "period": period
        })
    except Exception as e:
        logger.error(f"Error calculating lead-time verification: {e}")
        return jsonify({"success": False, "error": str(e)})


def calculate_temp_bias_history(location_name: str, lead_time_hours: int = 24, days_back: int = 30) -> dict:
    """
    Calculate temperature bias history with observed values for visualization.
    Returns observed temperatures and model biases for each verification time.

    Args:
        location_name: Location name (e.g., "Fairfax, VA")
        lead_time_hours: Forecast lead time in hours
        days_back: Number of days to look back

    Returns:
        Dict with structure:
        {
            "dates": ["2026-01-20T12:00:00", "2026-01-20T18:00:00", ...],
            "observed": [32.5, 35.2, ...],  # Observed temperature for each time
            "gfs_bias": [1.2, -0.5, ...],    # GFS bias for each time
            "aifs_bias": [0.8, -0.3, ...],   # AIFS bias for each time
            "ifs_bias": [0.5, -0.1, ...]     # IFS bias for each time
        }
    """
    db = load_forecasts_db()

    if location_name not in db:
        return {"error": "Location not found"}

    runs = db[location_name].get("runs", {})
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back)

    # Collect candidate verification points (latest run per valid time)
    # Key: valid_time ISO string
    verification_points = {}

    for run_id, run_data in runs.items():
        gfs_data = run_data.get("gfs", {})
        aifs_data = run_data.get("aifs", {})
        ifs_data = run_data.get("ifs", {})

        init_time_str = gfs_data.get("init_time")
        if not init_time_str:
            continue

        try:
            init_time = datetime.fromisoformat(init_time_str)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_times = gfs_data.get("times", [])
        gfs_temps = gfs_data.get("temps", [])
        aifs_temps = aifs_data.get("temps", [])
        ifs_temps = ifs_data.get("temps", []) if ifs_data else []

        for i, time_str in enumerate(forecast_times):
            try:
                valid_time = datetime.fromisoformat(time_str)
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            # Only include past times within requested window
            if valid_time >= now or valid_time < cutoff_date:
                continue

            # Calculate lead time in hours
            lead_time = int((valid_time - init_time).total_seconds() / 3600)

            # Only include the requested lead time
            if lead_time != lead_time_hours:
                continue

            # Only include 6-hourly verification times to match model cycles
            if valid_time.hour % 6 != 0:
                continue

            # Use the valid time as the key and keep latest init_time
            time_key = valid_time.astimezone(timezone.utc).isoformat()
            existing = verification_points.get(time_key)
            if existing and existing.get("init_time") and existing["init_time"] >= init_time:
                continue

            verification_points[time_key] = {
                "init_time": init_time,
                "gfs_val": gfs_temps[i] if i < len(gfs_temps) else None,
                "aifs_val": aifs_temps[i] if i < len(aifs_temps) else None,
                "ifs_val": ifs_temps[i] if (ifs_temps and i < len(ifs_temps)) else None
            }

    # Sort by time and build result arrays
    sorted_times = sorted(verification_points.keys())

    # Fetch observed values for all times (WeatherLink)
    obs_times = []
    for t in sorted_times:
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            obs_times.append(dt.replace(tzinfo=None))
        except Exception:
            obs_times.append(None)

    obs_lookup = {}
    if obs_times:
        obs_data = weatherlink.get_observations(obs_times, times_are_utc=True)
        obs_vals = obs_data.get("temps", [])
        for idx, t in enumerate(sorted_times):
            if idx < len(obs_vals):
                obs_lookup[t] = obs_vals[idx]

    result = {
        "dates": sorted_times,
        "observed": [],
        "gfs_bias": [],
        "aifs_bias": [],
        "ifs_bias": []
    }

    for time_key in sorted_times:
        point = verification_points[time_key]
        obs_val = obs_lookup.get(time_key)
        if obs_val is None:
            result["observed"].append(None)
            result["gfs_bias"].append(None)
            result["aifs_bias"].append(None)
            result["ifs_bias"].append(None)
            continue

        result["observed"].append(round(obs_val, 1))
        result["gfs_bias"].append(round(point["gfs_val"] - obs_val, 2) if point["gfs_val"] is not None else None)
        result["aifs_bias"].append(round(point["aifs_val"] - obs_val, 2) if point["aifs_val"] is not None else None)
        result["ifs_bias"].append(round(point["ifs_val"] - obs_val, 2) if point["ifs_val"] is not None else None)

    return result


@app.route('/api/verification-time-series')
def api_verification_time_series():
    """
    Get verification time series (daily MAE and bias) for a location.
    Shows trends in forecast accuracy over time.
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    variable = request.args.get('variable', 'temp')
    lead_time = int(request.args.get('lead_time', 24))
    days_back = int(request.args.get('days_back', 30))

    try:
        cache = load_verif_ts_cache()
        entries = cache.get("entries", {})
        cache_key = f"{location_name}|{variable}|{lead_time}|{days_back}"

        source_mtimes = _get_verif_ts_source_mtimes()

        cached = entries.get(cache_key)
        if cached:
            cached_sources = cached.get("source_mtimes", {})
            # For temp/mslp, cocorahs is not relevant; for mslp, NWS is not relevant
            relevant_keys = ["forecasts"]
            if variable in ("temp", "precip"):
                relevant_keys.append("nws")
            if variable == "precip":
                relevant_keys.append("cocorahs")

            if all(cached_sources.get(k) == source_mtimes.get(k) for k in relevant_keys):
                return jsonify({
                    "success": True,
                    "location": location_name,
                    "variable": variable,
                    "lead_time": lead_time,
                    "days_back": days_back,
                    "time_series": cached.get("time_series", {})
                })

        result = calculate_verification_time_series(location_name, variable, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        entries[cache_key] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mtimes": source_mtimes,
            "time_series": result
        }
        cache["entries"] = entries
        save_verif_ts_cache(cache)

        return jsonify({
            "success": True,
            "location": location_name,
            "variable": variable,
            "lead_time": lead_time,
            "days_back": days_back,
            "time_series": result
        })
    except Exception as e:
        logger.error(f"Error calculating verification time series: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/pangu-latest")
def api_pangu_latest():
    """
    Return the latest Pangu run time series for Fairfax, VA (temp_surface and pressure_msl).
    If the latest run is an ensemble, return the mean across members.
    """
    location = request.args.get("location", "Fairfax, VA")
    if location.lower() not in {"fairfax, va", "fairfax"}:
        return jsonify({"success": False, "error": "Pangu overlay available only for Fairfax, VA"}), 200

    db = load_runs_db()
    runs = db.get("runs", [])
    if not runs:
        return jsonify({"success": False, "error": "No Pangu runs available"}), 200

    selected = None
    latest_ref = None
    try:
        forecast_db = load_forecasts_db()
        loc_data = forecast_db.get("Fairfax, VA", {})
        latest_key = loc_data.get("latest_run")
        if latest_key:
            latest_ref = datetime.fromisoformat(latest_key)
    except Exception:
        latest_ref = None
    for run in runs:
        if run.get("ensemble"):
            members = run.get("members", [])
            member_series = []
            for m in members:
                data = m.get("data", {})
                ts = data.get("time_series")
                if ts and ts.get("temp_surface") and ts.get("pressure_msl"):
                    member_series.append(ts)
            if member_series:
                selected = {
                    "run": run,
                    "time_series": _mean_series(member_series),
                    "ensemble": True,
                    "member_count": len(member_series),
                }
                break
        else:
            ts = run.get("data", {}).get("time_series")
            if ts and ts.get("temp_surface") and ts.get("pressure_msl"):
                selected = {
                    "run": run,
                    "time_series": ts,
                    "ensemble": False,
                    "member_count": 1,
                }
                break

    if not selected or not selected.get("time_series"):
        return jsonify({"success": False, "error": "No usable Pangu run data"}), 200

    run = selected["run"]
    # Filter out stale runs if too far from latest main-model init time
    if latest_ref:
        try:
            run_time = datetime.fromisoformat(f"{run.get('init_date')}T{run.get('init_time')}:00:00")
            if run_time.tzinfo is None:
                run_time = run_time.replace(tzinfo=timezone.utc)
            if latest_ref.tzinfo is None:
                latest_ref = latest_ref.replace(tzinfo=timezone.utc)
            if abs((run_time - latest_ref).total_seconds()) > 24 * 3600:
                return jsonify({"success": False, "error": "Pangu run too old"}), 200
        except Exception:
            pass
    ts = selected["time_series"]
    raw_times = ts.get("times", [])
    normalized_times = []
    for t in raw_times:
        if isinstance(t, str) and (t.endswith("Z") or "+" in t):
            normalized_times.append(t)
        elif isinstance(t, str) and "T" in t:
            normalized_times.append(f"{t}Z")
        else:
            normalized_times.append(t)

    return jsonify({
        "success": True,
        "init_date": run.get("init_date"),
        "init_time": run.get("init_time"),
        "run_id": run.get("run_id"),
        "ensemble": selected["ensemble"],
        "member_count": selected["member_count"],
        "times": normalized_times,
        "temp_surface": ts.get("temp_surface", []),
        "pressure_msl": ts.get("pressure_msl", [])
    })


@app.route('/api/temp-bias-history')
def api_temp_bias_history():
    """
    Get temperature bias history with observed values.
    Shows observed temperatures and model biases over time.
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    lead_time = int(request.args.get('lead_time', 24))
    days_back = int(request.args.get('days_back', 30))

    try:
        cache = load_bias_history_cache()
        entries = cache.get("entries", {})
        cache_key = f"temp|{location_name}|{lead_time}|{days_back}"
        source_mtimes = _get_verif_ts_source_mtimes()

        cached = entries.get(cache_key)
        if cached and cached.get("source_mtimes", {}).get("forecasts") == source_mtimes.get("forecasts"):
            return jsonify({
                "success": True,
                "location": location_name,
                "lead_time": lead_time,
                "days_back": days_back,
                "history": cached.get("history", {})
            })

        result = calculate_temp_bias_history(location_name, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        entries[cache_key] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mtimes": source_mtimes,
            "history": result
        }
        cache["entries"] = entries
        save_bias_history_cache(cache)

        return jsonify({
            "success": True,
            "location": location_name,
            "lead_time": lead_time,
            "days_back": days_back,
            "history": result
        })
    except Exception as e:
        logger.error(f"Error calculating temp bias history: {e}")
        return jsonify({"success": False, "error": str(e)})


def calculate_mslp_bias_history(location_name: str, lead_time_hours: int = 24, days_back: int = 30) -> dict:
    """
    Calculate MSLP bias history with observed values for visualization.
    Returns observed MSLP and model biases for each verification time.

    Args:
        location_name: Location name (e.g., "Fairfax, VA")
        lead_time_hours: Forecast lead time in hours
        days_back: Number of days to look back

    Returns:
        Dict with structure:
        {
            "dates": ["2026-01-20T12:00:00", "2026-01-20T18:00:00", ...],
            "observed": [1013.2, 1015.8, ...],  # Observed MSLP for each time
            "gfs_bias": [1.2, -0.5, ...],       # GFS bias for each time
            "aifs_bias": [0.8, -0.3, ...],      # AIFS bias for each time
            "ifs_bias": [0.5, -0.1, ...]        # IFS bias for each time
        }
    """
    db = load_forecasts_db()

    if location_name not in db:
        return {"error": "Location not found"}

    runs = db[location_name].get("runs", {})
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back)

    # Collect candidate verification points (latest run per valid time)
    # Key: valid_time ISO string
    verification_points = {}

    for run_id, run_data in runs.items():
        gfs_data = run_data.get("gfs", {})
        aifs_data = run_data.get("aifs", {})
        ifs_data = run_data.get("ifs", {})

        init_time_str = gfs_data.get("init_time")
        if not init_time_str:
            continue

        try:
            init_time = datetime.fromisoformat(init_time_str)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_times = gfs_data.get("times", [])
        gfs_mslps = gfs_data.get("mslps", [])
        aifs_mslps = aifs_data.get("mslps", [])
        ifs_mslps = ifs_data.get("mslps", []) if ifs_data else []

        for i, time_str in enumerate(forecast_times):
            try:
                valid_time = datetime.fromisoformat(time_str)
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            # Only include past times within requested window
            if valid_time >= now or valid_time < cutoff_date:
                continue

            # Calculate lead time in hours
            lead_time = int((valid_time - init_time).total_seconds() / 3600)

            # Only include the requested lead time
            if lead_time != lead_time_hours:
                continue

            # Only include 00Z and 12Z verification times (reduce clutter)
            if valid_time.hour not in [0, 12]:
                continue

            # Use the valid time as the key and keep latest init_time
            time_key = valid_time.astimezone(timezone.utc).isoformat()
            existing = verification_points.get(time_key)
            if existing and existing.get("init_time") and existing["init_time"] >= init_time:
                continue

            verification_points[time_key] = {
                "init_time": init_time,
                "gfs_val": gfs_mslps[i] if i < len(gfs_mslps) else None,
                "aifs_val": aifs_mslps[i] if i < len(aifs_mslps) else None,
                "ifs_val": ifs_mslps[i] if (ifs_mslps and i < len(ifs_mslps)) else None
            }

    # Sort by time and build result arrays
    sorted_times = sorted(verification_points.keys())

    # Fetch observed values for all times (WeatherLink)
    obs_times = []
    for t in sorted_times:
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            obs_times.append(dt.replace(tzinfo=None))
        except Exception:
            obs_times.append(None)

    obs_lookup = {}
    if obs_times:
        obs_data = weatherlink.get_observations(obs_times, times_are_utc=True)
        obs_vals = obs_data.get("mslps", [])
        for idx, t in enumerate(sorted_times):
            if idx < len(obs_vals):
                obs_lookup[t] = obs_vals[idx]

    result = {
        "dates": sorted_times,
        "observed": [],
        "gfs_bias": [],
        "aifs_bias": [],
        "ifs_bias": []
    }

    for time_key in sorted_times:
        point = verification_points[time_key]
        obs_val = obs_lookup.get(time_key)
        if obs_val is None:
            result["observed"].append(None)
            result["gfs_bias"].append(None)
            result["aifs_bias"].append(None)
            result["ifs_bias"].append(None)
            continue

        result["observed"].append(round(obs_val, 1))
        result["gfs_bias"].append(round(point["gfs_val"] - obs_val, 2) if point["gfs_val"] is not None else None)
        result["aifs_bias"].append(round(point["aifs_val"] - obs_val, 2) if point["aifs_val"] is not None else None)
        result["ifs_bias"].append(round(point["ifs_val"] - obs_val, 2) if point["ifs_val"] is not None else None)

    return result


@app.route('/api/mslp-bias-history')
def api_mslp_bias_history():
    """
    Get MSLP bias history with observed values.
    Shows observed MSLP and model biases over time.
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    lead_time = int(request.args.get('lead_time', 24))
    days_back = int(request.args.get('days_back', 30))

    try:
        cache = load_bias_history_cache()
        entries = cache.get("entries", {})
        cache_key = f"mslp|{location_name}|{lead_time}|{days_back}"
        source_mtimes = _get_verif_ts_source_mtimes()

        cached = entries.get(cache_key)
        if cached and cached.get("source_mtimes", {}).get("forecasts") == source_mtimes.get("forecasts"):
            return jsonify({
                "success": True,
                "location": location_name,
                "lead_time": lead_time,
                "days_back": days_back,
                "history": cached.get("history", {})
            })

        result = calculate_mslp_bias_history(location_name, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        entries[cache_key] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mtimes": source_mtimes,
            "history": result
        }
        cache["entries"] = entries
        save_bias_history_cache(cache)

        return jsonify({
            "success": True,
            "location": location_name,
            "lead_time": lead_time,
            "days_back": days_back,
            "history": result
        })
    except Exception as e:
        logger.error(f"Error calculating MSLP bias history: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/precip-daily-history')
def api_precip_daily_history():
    """
    Get daily precipitation history for Fairfax using CoCoRaHS (past) and latest model run (future only).
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    days_back = int(request.args.get('days_back', 30))
    future_days = int(request.args.get('future_days', 15))
    lead_time_hours = int(request.args.get('lead_time', 24))

    if location_name != "Fairfax, VA":
        return jsonify({"success": False, "error": "Precip history only available for Fairfax, VA"}), 400

    now_utc = datetime.now(timezone.utc)
    local_now = weatherlink.utc_to_eastern(now_utc.replace(tzinfo=None))
    end_date = local_now.date() + timedelta(days=future_days)
    start_date = (local_now.date() - timedelta(days=1)) - timedelta(days=days_back - 1)

    obs_end = local_now.date() - timedelta(days=1)
    obs_data = fetch_cocorahs_daily_precip(
        COCORAHs_STATION_ID,
        start_date.isoformat(),
        obs_end.isoformat()
    )

    # Build model daily totals from latest run only (future dates)
    db = load_forecasts_db()
    runs = db.get(location_name, {}).get("runs", {})
    latest_run_id = db.get(location_name, {}).get("latest_run")
    run_data = runs.get(latest_run_id, {}) if latest_run_id else {}
    has_ifs = any(rd.get("ifs") for rd in runs.values())

    gfs_data = run_data.get("gfs", {})
    aifs_data = run_data.get("aifs", {})
    ifs_data = run_data.get("ifs", {})

    times = gfs_data.get("times", [])
    gfs_precips = gfs_data.get("precips", [])
    aifs_precips = aifs_data.get("precips", [])
    ifs_precips = ifs_data.get("precips", []) if ifs_data else []

    model_daily = {}
    for i, time_str in enumerate(times):
        try:
            valid_time = datetime.fromisoformat(time_str)
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        local_dt = weatherlink.utc_to_eastern(valid_time.replace(tzinfo=None))
        date_key = local_dt.date()
        if date_key < local_now.date() or date_key > end_date:
            continue

        model_daily.setdefault(date_key, {"gfs": [], "aifs": [], "ifs": []})

        if i < len(gfs_precips) and gfs_precips[i] is not None:
            model_daily[date_key]["gfs"].append(gfs_precips[i])
        if i < len(aifs_precips) and aifs_precips[i] is not None:
            model_daily[date_key]["aifs"].append(aifs_precips[i])
        if ifs_precips and i < len(ifs_precips) and ifs_precips[i] is not None:
            model_daily[date_key]["ifs"].append(ifs_precips[i])

    model_totals = {}
    for date_key, vals in model_daily.items():
        model_totals[date_key] = {
            "gfs": sum(vals["gfs"]) if len(vals["gfs"]) >= 3 else None,
            "aifs": sum(vals["aifs"]) if len(vals["aifs"]) >= 3 else None,
            "ifs": sum(vals["ifs"]) if len(vals["ifs"]) >= 3 else None
        }

    # Build historical model daily totals (lead-time matched to 12Z) for bias only
    def compute_daily_total_for_run(model_data, day_end_utc):
        times = model_data.get("times", [])
        precips = model_data.get("precips", [])
        if not times or not precips:
            return None, 0
        window_start = day_end_utc - timedelta(hours=24)
        total = 0.0
        count = 0
        for i, time_str in enumerate(times):
            try:
                valid_time = datetime.fromisoformat(time_str)
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if window_start < valid_time <= day_end_utc:
                if i < len(precips) and precips[i] is not None:
                    total += precips[i]
                    count += 1
        if count >= 3:
            return total, count
        return None, count

    def select_best_total_for_date(model_key, day_end_utc):
        best_total = None
        best_init = None
        for run_id, run_data in runs.items():
            model_data = run_data.get(model_key, {})
            init_time_str = model_data.get("init_time") or run_data.get("gfs", {}).get("init_time")
            if not init_time_str:
                continue
            try:
                init_time = datetime.fromisoformat(init_time_str)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Check that this run provides the day_end_utc valid time at the selected lead time
            times = model_data.get("times", [])
            has_valid = False
            for t in times:
                try:
                    valid_time = datetime.fromisoformat(t)
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue
                if valid_time == day_end_utc:
                    lead_time = int((valid_time - init_time).total_seconds() / 3600)
                    if lead_time == lead_time_hours:
                        has_valid = True
                    break
            if not has_valid:
                continue

            total, count = compute_daily_total_for_run(model_data, day_end_utc)
            if total is None:
                continue

            if best_init is None or init_time > best_init:
                best_init = init_time
                best_total = total

        return best_total

    dates = []
    observed = []
    gfs = []
    aifs = []
    ifs = []
    gfs_bias = []
    aifs_bias = []
    ifs_bias = []

    d = start_date
    while d <= end_date:
        date_str = d.isoformat()
        dates.append(date_str)
        observed.append(obs_data.get(date_str))
        best_future = model_totals.get(d, {})
        gfs.append(best_future.get("gfs"))
        aifs.append(best_future.get("aifs"))
        ifs.append(best_future.get("ifs"))

        if d <= obs_end:
            obs_val = obs_data.get(date_str)
            day_end = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
            gfs_hist = select_best_total_for_date("gfs", day_end)
            aifs_hist = select_best_total_for_date("aifs", day_end)
            ifs_hist = select_best_total_for_date("ifs", day_end) if has_ifs else None
            gfs_bias.append(
                (gfs_hist - obs_val) if (gfs_hist is not None and obs_val is not None) else None
            )
            aifs_bias.append(
                (aifs_hist - obs_val) if (aifs_hist is not None and obs_val is not None) else None
            )
            ifs_bias.append(
                (ifs_hist - obs_val) if (ifs_hist is not None and obs_val is not None) else None
            )
        else:
            gfs_bias.append(None)
            aifs_bias.append(None)
            ifs_bias.append(None)
        d += timedelta(days=1)

    return jsonify({
        "success": True,
        "station_id": COCORAHs_STATION_ID,
        "dates": dates,
        "observed": observed,
        "gfs": gfs,
        "aifs": aifs,
        "ifs": ifs,
        "gfs_bias": gfs_bias,
        "aifs_bias": aifs_bias,
        "ifs_bias": ifs_bias
    })

@app.route('/api/wave-analysis')
def api_wave_analysis():
    """
    Get Rossby wave numbers for latest forecast run.

    Query params:
        - location: Location name (default: "Fairfax, VA")

    Returns:
        {
            "success": true,
            "location": "Fairfax, VA",
            "init_time": "2026-02-02T00:00:00",
            "gfs_waves": {...},
            "aifs_waves": {...},
            "ifs_waves": {...}
        }
    """
    location_name = request.args.get('location', 'Fairfax, VA')

    try:
        db = load_forecasts_db()
        if location_name not in db:
            return jsonify({"success": False, "error": "Location not found"})

        latest_run = db[location_name].get("latest_run")
        if not latest_run:
            return jsonify({"success": False, "error": "No forecast data"})

        run_data = db[location_name]["runs"][latest_run]

        return jsonify({
            "success": True,
            "location": location_name,
            "init_time": latest_run,
            "gfs_waves": run_data.get("gfs", {}).get("z500_waves"),
            "aifs_waves": run_data.get("aifs", {}).get("z500_waves"),
            "ifs_waves": run_data.get("ifs", {}).get("z500_waves")
        })
    except Exception as e:
        logger.error(f"Error fetching wave analysis: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/wave-time-series')
def api_wave_time_series():
    """
    Get historical time series of wave numbers from 00Z runs only.

    Note: Only 00Z model runs are included in the historical time series
    to maintain consistency in daily wave number tracking.

    Query params:
        - location: Location name (default: "Fairfax, VA")
        - days_back: Number of days to look back (default: 30)

    Returns:
        {
            "success": true,
            "location": "Fairfax, VA",
            "dates": ["2026-01-15T00:00:00", ...],
            "gfs": [5.5, 6.0, 4.5, ...],
            "aifs": [5.0, 5.5, 4.0, ...],
            "ifs": [5.5, 6.0, 4.5, ...]
        }
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    days_back = int(request.args.get('days_back', 30))

    try:
        db = load_forecasts_db()
        if location_name not in db:
            return jsonify({"success": False, "error": "Location not found"})

        runs = db[location_name].get("runs", {})
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=days_back)

        # Collect wave numbers from all runs
        dates = []
        gfs_waves = []
        aifs_waves = []
        ifs_waves = []

        for run_id, run_data in sorted(runs.items()):
            try:
                run_time = datetime.fromisoformat(run_id)
                if run_time.tzinfo is None:
                    run_time = run_time.replace(tzinfo=timezone.utc)

                if run_time < cutoff_date:
                    continue

                # Only include 00Z runs in historical data
                if run_time.hour != 0:
                    continue

                # Extract wave numbers (handle None case with 'or {}')
                gfs_wave = (run_data.get("gfs", {}).get("z500_waves") or {}).get("wave_number")
                aifs_wave = (run_data.get("aifs", {}).get("z500_waves") or {}).get("wave_number")
                ifs_wave = (run_data.get("ifs", {}).get("z500_waves") or {}).get("wave_number")

                # Only include if at least one model has data
                if gfs_wave or aifs_wave or ifs_wave:
                    dates.append(run_id)
                    gfs_waves.append(gfs_wave)
                    aifs_waves.append(aifs_wave)
                    ifs_waves.append(ifs_wave)

            except (ValueError, TypeError):
                continue

        return jsonify({
            "success": True,
            "location": location_name,
            "dates": dates,
            "gfs": gfs_waves,
            "aifs": aifs_waves,
            "ifs": ifs_waves
        })

    except Exception as e:
        logger.error(f"Error fetching wave time series: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/wave-forecast')
def api_wave_forecast():
    """
    Get wave number forecast for the latest run (15-day forecast of wave numbers).

    Query params:
        - location: Location name (default: "Fairfax, VA")

    Returns:
        {
            "success": true,
            "location": "Fairfax, VA",
            "init_time": "2026-02-02T00:00:00",
            "gfs_forecast": {
                "times": [...],
                "wave_numbers": [...]
            },
            "aifs_forecast": {...},
            "ifs_forecast": {...}
        }
    """
    location_name = request.args.get('location', 'Fairfax, VA')

    try:
        db = load_forecasts_db()
        if location_name not in db:
            return jsonify({"success": False, "error": "Location not found"})

        latest_run = db[location_name].get("latest_run")
        if not latest_run:
            return jsonify({"success": False, "error": "No forecast data"})

        run_data = db[location_name]["runs"][latest_run]

        return jsonify({
            "success": True,
            "location": location_name,
            "init_time": latest_run,
            "gfs_forecast": run_data.get("gfs", {}).get("z500_waves_forecast"),
            "aifs_forecast": run_data.get("aifs", {}).get("z500_waves_forecast"),
            "ifs_forecast": run_data.get("ifs", {}).get("z500_waves_forecast")
        })
    except Exception as e:
        logger.error(f"Error fetching wave forecast: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/wave-skill-correlation')
def api_wave_skill_correlation():
    """
    Analyze correlation between Rossby wave patterns and forecast errors.

    This endpoint calculates the relationship between wave characteristics
    (wave number, amplitude) and temperature forecast errors at different
    lead times using ASOS network mean verification.

    Query params:
        - location: Location name (default: "Fairfax, VA")
        - days_back: Number of days to analyze (default: 90)

    Returns:
        {
            "success": true,
            "current_wave_number": 5,
            "predictability_status": "MEDIUM",
            "confidence": "High",
            "expected_error_72h": 2.8,
            "expected_error_120h": 4.2,
            "wave_error_correlation": {
                "points_72h": [{x: 3, y: 2.1}, ...],
                "points_120h": [{x: 3, y: 3.5}, ...]
            },
            "amplitude_error_correlation": {
                "points_72h": [{x: 245, y: 2.1}, ...],
                "points_120h": [{x: 245, y: 3.5}, ...]
            }
        }
    """
    location_name = request.args.get('location', 'Fairfax, VA')
    days_back = int(request.args.get('days_back', 90))

    try:
        # Import ASOS module
        import asos

        # Load Fairfax forecast database (for wave data)
        fairfax_db = load_forecasts_db()
        if location_name not in fairfax_db:
            return jsonify({"success": False, "error": "Location not found"})

        # Load ASOS forecast database (for verification data)
        asos_db = asos.load_asos_forecasts_db()
        if not asos_db or "runs" not in asos_db:
            return jsonify({"success": False, "error": "ASOS verification data not available"})

        # Get cutoff date (timezone-aware)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Collect wave-error pairs
        wave_error_data_72h = []
        wave_error_data_120h = []
        amplitude_error_data_72h = []
        amplitude_error_data_120h = []

        # Loop through ASOS runs to get verification data
        for run_time_str, asos_run in asos_db["runs"].items():
            try:
                run_time = datetime.fromisoformat(run_time_str)
                if run_time.tzinfo is None:
                    run_time = run_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Skip runs outside the time window
            if run_time < cutoff_date:
                continue

            # Get wave data from Fairfax database for this same init time
            fairfax_runs = fairfax_db[location_name].get("runs", {})
            if run_time_str not in fairfax_runs:
                continue

            fairfax_run = fairfax_runs[run_time_str]
            gfs_waves = fairfax_run.get("gfs", {}).get("z500_waves")
            if not gfs_waves or gfs_waves.get("wave_number") is None:
                continue

            wave_number = gfs_waves.get("wave_number")
            amplitude = gfs_waves.get("top_3_amplitudes", [None])[0]

            # Calculate ASOS mean MAE for this run at 72h and 120h
            forecast_hours = asos_run.get("forecast_hours", [])

            # Calculate 72h error
            if 72 in forecast_hours:
                mae_72h = asos._calculate_asos_mean_mae(asos_db, run_time_str, 'gfs', 'temp', 72)
                if mae_72h is not None and mae_72h > 0:
                    wave_error_data_72h.append({"x": wave_number, "y": mae_72h})
                    if amplitude is not None:
                        amplitude_error_data_72h.append({"x": amplitude, "y": mae_72h})

            # Calculate 120h error
            if 120 in forecast_hours:
                mae_120h = asos._calculate_asos_mean_mae(asos_db, run_time_str, 'gfs', 'temp', 120)
                if mae_120h is not None and mae_120h > 0:
                    wave_error_data_120h.append({"x": wave_number, "y": mae_120h})
                    if amplitude is not None:
                        amplitude_error_data_120h.append({"x": amplitude, "y": mae_120h})

        # Calculate expected errors based on current wave pattern
        latest_run = fairfax_db[location_name].get("latest_run")
        current_wave_number = None
        expected_error_72h = None
        expected_error_120h = None
        predictability_status = "UNKNOWN"
        confidence = "Low"

        if latest_run and latest_run in fairfax_db[location_name]["runs"]:
            current_run = fairfax_db[location_name]["runs"][latest_run]
            current_waves = current_run.get("gfs", {}).get("z500_waves", {})
            current_wave_number = current_waves.get("wave_number")

            if current_wave_number is not None and wave_error_data_72h:
                # Calculate average error for this wave number (±1 wave number tolerance)
                errors_72h_for_wave = [
                    d["y"] for d in wave_error_data_72h
                    if abs(d["x"] - current_wave_number) <= 1
                ]
                errors_120h_for_wave = [
                    d["y"] for d in wave_error_data_120h
                    if abs(d["x"] - current_wave_number) <= 1
                ]

                if errors_72h_for_wave:
                    expected_error_72h = sum(errors_72h_for_wave) / len(errors_72h_for_wave)
                    confidence = "High" if len(errors_72h_for_wave) >= 10 else "Medium" if len(errors_72h_for_wave) >= 5 else "Low"

                if errors_120h_for_wave:
                    expected_error_120h = sum(errors_120h_for_wave) / len(errors_120h_for_wave)

                # Determine predictability status based on wave number
                if current_wave_number <= 4:
                    predictability_status = "HIGH"
                elif current_wave_number <= 6:
                    predictability_status = "MEDIUM"
                else:
                    predictability_status = "LOW"

        return jsonify({
            "success": True,
            "location": location_name,
            "days_analyzed": days_back,
            "sample_size": len(wave_error_data_72h),
            "current_wave_number": current_wave_number,
            "predictability_status": predictability_status,
            "confidence": confidence,
            "expected_error_72h": round(expected_error_72h, 1) if expected_error_72h else None,
            "expected_error_120h": round(expected_error_120h, 1) if expected_error_120h else None,
            "wave_error_correlation": {
                "points_72h": wave_error_data_72h,
                "points_120h": wave_error_data_120h
            },
            "amplitude_error_correlation": {
                "points_72h": amplitude_error_data_72h,
                "points_120h": amplitude_error_data_120h
            }
        })

    except Exception as e:
        logger.error(f"Error calculating wave-skill correlation: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/era5/wave-analysis')
def api_era5_wave_analysis():
    """
    Analyze Rossby waves from ERA5 reanalysis data.

    Query params:
        - start_date: Start date (YYYY-MM-DD), default: 30 days ago
        - end_date: End date (YYYY-MM-DD), default: today

    Returns wave numbers calculated from ERA5 500mb heights over time.
    """
    try:
        import xarray as xr
        from datetime import datetime, timedelta

        # Get date range
        end_date = request.args.get('end_date')
        start_date = request.args.get('start_date')

        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.fromisoformat(end_date)

        if not start_date:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = datetime.fromisoformat(start_date)

        # Load ERA5 data
        era5_files = list(Path("/Volumes/T7/Weather_Models/era5/global_500mb").glob("era5_z500_NH_*.nc"))

        if not era5_files:
            return jsonify({"success": False, "error": "No ERA5 data found. Please download data first."})

        # Load the most recent file or combine multiple files
        datasets = []
        for f in era5_files:
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not datasets:
            return jsonify({"success": False, "error": "Failed to load ERA5 datasets"})

        # Combine datasets
        ds = xr.concat(datasets, dim='time')
        ds = ds.sortby('time')

        # Filter to requested date range
        ds = ds.sel(time=slice(start_date, end_date))

        # Calculate wave numbers for each day
        wave_data = []

        for t in ds.time.values:
            try:
                # Get 500mb heights for this time (keep as DataArray with coordinates)
                z500 = ds['z'].sel(time=t, pressure_level=500)

                # Use rossby_waves module to calculate wave number (pass DataArray directly)
                wave_result = rossby_waves.calculate_wave_number(z500, latitude=55.0)

                wave_data.append({
                    "time": str(t),
                    "wave_number": wave_result.get("wave_number", 0),
                    "dominant_waves": wave_result.get("dominant_wave_numbers", []),
                    "variance_explained": wave_result.get("top_3_variance", {})
                })
            except Exception as e:
                logger.warning(f"Failed to calculate wave for {t}: {e}")
                continue

        return jsonify({
            "success": True,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "wave_data": wave_data
        })

    except Exception as e:
        logger.error(f"Error in ERA5 wave analysis: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/era5/500mb-map')
def api_era5_500mb_map():
    """
    Get 500mb height field for spatial mapping.

    Query params:
        - date: Date to fetch (YYYY-MM-DD), default: most recent

    Returns 500mb height grid data for mapping.
    """
    try:
        import xarray as xr
        from datetime import datetime

        # Get requested date
        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.fromisoformat(date_str)
        else:
            target_date = datetime.now()

        # Load ERA5 data
        era5_files = list(Path("/Volumes/T7/Weather_Models/era5/global_500mb").glob("era5_z500_NH_*.nc"))

        if not era5_files:
            return jsonify({"success": False, "error": "No ERA5 data found"})

        # Load datasets
        datasets = []
        for f in era5_files:
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not datasets:
            return jsonify({"success": False, "error": "Failed to load ERA5 datasets"})

        # Combine and sort
        ds = xr.concat(datasets, dim='time')
        ds = ds.sortby('time')

        # Check for pressure level coordinate
        pressure_coord = None
        for coord_name in ['pressure_level', 'level', 'plev', 'lev']:
            if coord_name in ds.coords or coord_name in ds.dims:
                pressure_coord = coord_name
                break

        if pressure_coord is None:
            return jsonify({"success": False, "error": "Could not find pressure level coordinate in dataset"})

        # Find nearest time to requested date
        nearest_time = ds.sel(time=target_date, method='nearest')

        # Get 500mb heights
        z500 = nearest_time['z'].sel(**{pressure_coord: 500})

        # Convert to dam (decameters) for traditional display
        z500_dam = z500 / 9.80665

        # Calculate wave number
        wave_result = rossby_waves.calculate_wave_number(z500, latitude=55.0)

        # Generate map image
        import warnings
        import warnings
        import warnings
        import warnings
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from io import BytesIO
        import base64
        import pandas as pd

        fig = plt.figure(figsize=(14, 8))
        # Rotate view to center North America (central_longitude=-100°)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-100))

        # Set extent for Northern Hemisphere
        ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Plot height field
        lons = z500.longitude.values
        lats = z500.latitude.values
        heights = z500_dam.values

        # Calculate climatological anomalies using 1990-2021 period
        # Get day of year for the target date
        target_doy = pd.Timestamp(nearest_time.time.values).dayofyear

        # Filter to 1990-2021 for climatology calculation
        clim_ds = ds.sel(time=slice('1990-01-01', '2021-12-31'))

        # Calculate climatology with 31-day window
        ds_with_doy = clim_ds.assign_coords(dayofyear=clim_ds.time.dt.dayofyear)
        window_days = [(target_doy + offset - 1) % 366 + 1 for offset in range(-15, 16)]
        mask = ds_with_doy.dayofyear.isin(window_days)

        if mask.sum() > 0:
            window_data = ds_with_doy.where(mask, drop=True)
            clim_z500 = window_data['z'].sel(**{pressure_coord: 500}).mean(dim='time')
            clim_z500_dam = clim_z500 / 9.80665
            anomalies = heights - clim_z500_dam.values
        else:
            # Fallback to zonal mean if climatology fails
            logger.warning(f"No climatology data for day {target_doy}, using zonal mean")
            zonal_mean = np.mean(heights, axis=1, keepdims=True)
            anomalies = heights - zonal_mean

        # Contour fill for anomalies
        anom_levels = np.arange(-30, 31, 2)  # Anomaly contours every 2 dam from -30 to +30
        cf = ax.contourf(lons, lats, anomalies, levels=anom_levels, cmap='RdBu_r',
                        transform=ccrs.PlateCarree(), extend='both')

        # Contour lines for raw heights
        height_levels = np.arange(460, 620, 6)  # Height contour every 60m (6 dam)
        cs = ax.contour(lons, lats, heights, levels=height_levels[::2], colors='black',
                       linewidths=0.8, transform=ccrs.PlateCarree(), alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

        # Add colorbar
        # Colorbar intentionally omitted to save space in modal

        # Title with top 3 wave numbers
        date_str = pd.Timestamp(nearest_time.time.values).strftime('%Y-%m-%d %H:%M UTC')
        wave_num = wave_result.get('wave_number', 0)
        top_waves = wave_result.get('dominant_wave_numbers', [])
        top_var = wave_result.get('top_3_variance', [])

        # Format wave number display with variance explained
        if len(top_waves) >= 3 and len(top_var) >= 3:
            wave_text = f"Wave {top_waves[0]} ({top_var[0]:.0f}%), Wave {top_waves[1]} ({top_var[1]:.0f}%), Wave {top_waves[2]} ({top_var[2]:.0f}%)"
        else:
            wave_text = f"Wave {wave_num}"

        plt.title(f'ERA5 500mb Height Anomalies (shaded) & Heights (contours) | {date_str}\nTop Wave Numbers: {wave_text}',
                 fontsize=12, fontweight='bold')

        # Convert to base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
        buffer.seek(0)
        img_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            "success": True,
            "date": str(nearest_time.time.values),
            "map_image": img_base64,
            "wave_number": wave_result.get("wave_number", 0),
            "dominant_waves": wave_result.get("dominant_wave_numbers", []),
            "variance_explained": wave_result.get("top_3_variance", {})
        })

    except Exception as e:
        logger.error(f"Error fetching 500mb map data: {e}")
        return jsonify({"success": False, "error": str(e)})


def _find_analogs_core(top_n: int = 10, method: str = 'composite') -> dict:
    """Shared pattern-matching core used by both analog endpoints.

    Loads GFS z500, ERA5 z500, builds climatology (cached), computes
    multi-metric similarity for every seasonal-window candidate, and
    returns the top_n most diverse analogs.

    Parameters
    ----------
    top_n:
        Number of analog dates to return (after diversity filtering).
    method:
        One of ``'pearson'``, ``'lat_pearson'``, or ``'composite'``.
        - ``'pearson'``     – legacy unweighted Pearson (exact backward compat)
        - ``'lat_pearson'`` – area-weighted Pearson only
        - ``'composite'``   – weighted combination of lat_pearson + grad_corr + rmse_sim [+ eof_sim]

    Returns
    -------
    dict with keys:
        ``init_time``, ``current_date_str``, ``current_wave_num``,
        ``ds``, ``pressure_coord``, ``lat_coord_name``, ``lon_coord_name``,
        ``era5_lats_overlap``, ``era5_lons_overlap``,
        ``current_z500``, ``current_anomaly``, ``climatology``,
        ``top_analogs``
    Raises on data errors (caller should catch and return JSON error).
    """
    import xarray as xr
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    # ------------------------------------------------------------------
    # 1. Get current GFS z500 — live-fetch latest available init (any cycle)
    # ------------------------------------------------------------------
    gfs_model = GFSModel()
    init_time = pd.Timestamp(gfs_model.get_latest_init_time())
    current_date_str = init_time.isoformat()

    logger.info(f"Analog matching: using GFS init {init_time.strftime('%Y-%m-%d %HZ')}")

    global_region = Region("Global", (-180, 180, 20, 70))

    try:
        z500_da = gfs_model.fetch_data(Z500_GFS, init_time, 0, global_region)
        current_z500_gfs = z500_da.values * 10.0  # dm → m (ERA5 stores z in metres)
        current_lats_gfs = z500_da.latitude.values
        current_lons_gfs = z500_da.longitude.values

        # Compute wave number live
        wave_result = rossby_waves.calculate_wave_number(z500_da, latitude=55.0)
        current_wave_num = wave_result.get('wave_number')

    except Exception as e:
        logger.warning(f"Live GFS z500 fetch failed ({e}); falling back to stored DB")
        # Fallback: read from forecasts.json
        db = load_forecasts_db()
        location_name = 'Fairfax, VA'

        if location_name not in db or not db[location_name].get('latest_run'):
            raise ValueError("No forecast data available and live GFS fetch failed")

        latest_run = db[location_name]['latest_run']
        run_data = db[location_name]['runs'].get(latest_run)

        if not run_data or 'gfs' not in run_data:
            raise ValueError("No GFS forecast data in latest run")

        gfs_data = run_data['gfs']
        z500_field_data = gfs_data.get('z500_field')

        if not z500_field_data:
            raise ValueError(
                "No z500 field available. Live GFS fetch failed and no cached z500 found. "
                "Please sync forecasts to cache z500 data."
            )

        init_time = pd.Timestamp(latest_run)
        current_date_str = init_time.isoformat()
        current_z500_gfs = np.array(z500_field_data['values']) * 10.0  # dm → m to match ERA5
        current_lats_gfs = np.array(z500_field_data['latitude'])
        current_lons_gfs = np.array(z500_field_data['longitude'])
        current_waves = gfs_data.get('z500_waves', {})
        current_wave_num = current_waves.get('wave_number')

    # ------------------------------------------------------------------
    # 2. Load ERA5 z500
    # ------------------------------------------------------------------
    era5_path = Path("/Volumes/T7/Weather_Models/era5/global_500mb")

    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 data directory not found: {era5_path}")

    era5_files = sorted(era5_path.glob("era5_z500_NH_*.nc"))

    if not era5_files:
        raise FileNotFoundError("No ERA5 data found")

    datasets = []
    for f in era5_files:
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not datasets:
        raise RuntimeError("Failed to load ERA5 datasets")

    ds = xr.concat(datasets, dim='time')
    ds = ds.sortby('time')

    pressure_coord = None
    for coord_name in ['pressure_level', 'level', 'plev', 'lev']:
        if coord_name in ds.coords or coord_name in ds.dims:
            pressure_coord = coord_name
            break

    if pressure_coord is None:
        raise RuntimeError("Could not find pressure level coordinate in dataset")

    lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    era5_lats = ds[lat_coord_name].values
    era5_lons = ds[lon_coord_name].values

    # Overlapping lat range
    overlap_lat_min = max(min(era5_lats), min(current_lats_gfs))
    overlap_lat_max = min(max(era5_lats), max(current_lats_gfs))

    lat_mask = (era5_lats >= overlap_lat_min) & (era5_lats <= overlap_lat_max)
    era5_lats_overlap = era5_lats[lat_mask]
    era5_lons_overlap = era5_lons

    # Regrid GFS to ERA5 grid.
    # Use assign_coords+sortby to convert 0→360 to -180→180 monotonically.
    # np.where() alone produces a non-monotonic array that makes interp() return NaN.
    gfs_da = xr.DataArray(
        current_z500_gfs,
        coords={'latitude': current_lats_gfs, 'longitude': current_lons_gfs},
        dims=['latitude', 'longitude']
    )
    if gfs_da.longitude.values.max() > 180:
        gfs_da = gfs_da.assign_coords(
            longitude=(((gfs_da.longitude + 180) % 360) - 180)
        ).sortby('longitude')
    current_z500 = gfs_da.interp(
        latitude=era5_lats_overlap,
        longitude=era5_lons_overlap,
        method='linear'
    ).values

    # ------------------------------------------------------------------
    # 3. Build / retrieve climatology (cached by (files, overlap))
    # ------------------------------------------------------------------
    cache_key = (
        tuple(str(f) for f in era5_files),
        round(float(overlap_lat_min), 2),
        round(float(overlap_lat_max), 2),
    )

    if cache_key not in _CLIMATOLOGY_CACHE:
        logger.info("Calculating climatology from ERA5 data (1990-2021) – will cache for this session …")
        clim_ds = ds.sel(
            time=slice('1990-01-01', '2021-12-31'),
            **{lat_coord_name: era5_lats_overlap, lon_coord_name: era5_lons_overlap}
        )
        ds_with_doy = clim_ds.assign_coords(dayofyear=clim_ds.time.dt.dayofyear)

        climatology: dict = {}
        for doy in range(1, 367):
            window_days = [(doy + offset - 1) % 366 + 1 for offset in range(-15, 16)]
            mask = ds_with_doy.dayofyear.isin(window_days)
            if mask.sum() > 0:
                window_data = ds_with_doy.where(mask, drop=True)
                clim_z500 = window_data['z'].sel(**{pressure_coord: 500}).mean(dim='time')
                climatology[doy] = clim_z500.values

        _CLIMATOLOGY_CACHE[cache_key] = climatology
        logger.info("Climatology cached (%d DoY entries)", len(climatology))
    else:
        climatology = _CLIMATOLOGY_CACHE[cache_key]
        logger.info("Using cached climatology (%d DoY entries)", len(climatology))

    # ------------------------------------------------------------------
    # 4. Current anomaly
    # ------------------------------------------------------------------
    current_doy = init_time.dayofyear
    if current_doy in climatology:
        current_clim = climatology[current_doy]
        current_anomaly = current_z500 - current_clim
    else:
        logger.warning(f"No climatology for day {current_doy}, using zonal mean")
        current_anomaly = current_z500 - np.mean(current_z500, axis=1, keepdims=True)

    current_flat = current_anomaly.flatten()

    # Pre-compute weights and lat_weights for composite / lat_pearson modes
    weights_flat = None
    if _HAS_ANALOG_METRICS and method in ('composite', 'lat_pearson'):
        w2d = _analog_metrics.build_lat_weights(era5_lats_overlap, current_anomaly.shape)
        weights_flat = w2d.flatten()

    # ------------------------------------------------------------------
    # 5. Bulk-load seasonal-window candidates
    # ------------------------------------------------------------------
    seasonal_window = 45

    # Build a mask of candidate times (pd already imported at top of function)
    all_times = ds.time.values
    candidate_mask = []
    for t in all_times:
        t_pd = pd.Timestamp(t)
        doy = t_pd.dayofyear
        diff = min(
            abs(doy - current_doy),
            abs(doy - current_doy + 365),
            abs(doy - current_doy - 365),
        )
        candidate_mask.append(diff <= seasonal_window)

    candidate_times = all_times[np.array(candidate_mask)]
    logger.info("Candidate analog times: %d of %d total", len(candidate_times), len(all_times))

    # Bulk load all candidate z500 slices in one xarray operation
    z500_bulk = ds['z'].sel(
        time=candidate_times,
        **{pressure_coord: 500, lat_coord_name: era5_lats_overlap, lon_coord_name: era5_lons_overlap}
    ).values  # shape: (n_candidates, n_lat, n_lon)

    # ------------------------------------------------------------------
    # 6. Score each candidate
    # ------------------------------------------------------------------
    analogs = []

    for i, t in enumerate(candidate_times):
        try:
            t_pd = pd.Timestamp(t)
            historical_doy = t_pd.dayofyear
            historical_z500 = z500_bulk[i]  # (n_lat, n_lon)

            if historical_doy in climatology:
                historical_anomaly = historical_z500 - climatology[historical_doy]
            else:
                historical_anomaly = historical_z500 - np.mean(historical_z500, axis=1, keepdims=True)

            historical_flat = historical_anomaly.flatten()

            # Basic finite check
            min_size = min(len(current_flat), len(historical_flat))
            if min_size == 0:
                continue

            cur_sub = current_flat[:min_size]
            hist_sub = historical_flat[:min_size]
            finite_mask = np.isfinite(cur_sub) & np.isfinite(hist_sub)
            n_finite = int(finite_mask.sum())

            if n_finite < 0.9 * min_size:
                continue

            # ---- Choose scoring method ----
            if method == 'pearson' or not _HAS_ANALOG_METRICS:
                # Legacy unweighted Pearson
                cur_f = cur_sub[finite_mask]
                hist_f = hist_sub[finite_mask]
                correlation, _ = pearsonr(cur_f, hist_f)
                record: dict = {
                    'date': str(t),
                    'correlation': float(correlation),
                    'composite_score': float(correlation),
                }

            elif method == 'lat_pearson':
                w_sub = (weights_flat[:min_size] if weights_flat is not None else np.ones(min_size))
                lat_pearson = _analog_metrics.weighted_pearson(cur_sub, hist_sub, w_sub)
                if not np.isfinite(lat_pearson):
                    continue
                record = {
                    'date': str(t),
                    'correlation': float(lat_pearson),
                    'composite_score': float(lat_pearson),
                }

            else:  # composite
                # Weighted Pearson
                w_sub = (weights_flat[:min_size] if weights_flat is not None else np.ones(min_size))
                lat_pearson = _analog_metrics.weighted_pearson(cur_sub, hist_sub, w_sub)

                # Gradient correlation
                grad_corr = _analog_metrics.gradient_correlation(current_anomaly, historical_anomaly)

                # RMSE similarity
                rmse_sim = _analog_metrics.rmse_similarity(cur_sub, hist_sub)

                # EOF similarity (if cache ready)
                eof_sim = None
                if _EOF_CACHE is not None:
                    # Find the matching index in the EOF cache times array
                    t_str = str(t)
                    eof_times = _EOF_CACHE.get('times', [])
                    idx_arr = np.where(eof_times == t_str)[0]
                    if len(idx_arr) > 0:
                        eof_sim = _analog_metrics.eof_similarity(current_anomaly, _EOF_CACHE, int(idx_arr[0]))

                composite = _analog_metrics.compute_composite_score(lat_pearson, grad_corr, rmse_sim, eof_sim)
                if not np.isfinite(composite):
                    continue

                record = {
                    'date': str(t),
                    'correlation': float(lat_pearson) if np.isfinite(lat_pearson) else 0.0,
                    'composite_score': float(composite),
                    'grad_corr': float(grad_corr) if np.isfinite(grad_corr) else None,
                    'rmse_sim': float(rmse_sim) if np.isfinite(rmse_sim) else None,
                }
                if eof_sim is not None:
                    record['eof_sim'] = float(eof_sim) if np.isfinite(eof_sim) else None

            # Wave number for point-analog display
            hist_z_da = ds['z'].sel(
                time=t,
                **{pressure_coord: 500, lat_coord_name: era5_lats_overlap, lon_coord_name: era5_lons_overlap}
            )
            wave_result = rossby_waves.calculate_wave_number(hist_z_da, latitude=55.0)
            record['wave_number'] = wave_result.get('wave_number')

            analogs.append(record)

        except Exception:
            continue

    # ------------------------------------------------------------------
    # 7. Sort and pick diverse top-N
    # ------------------------------------------------------------------
    analogs.sort(key=lambda x: x['composite_score'], reverse=True)
    logger.info("Found %d candidate analogs; selecting top %d with ≥7-day separation", len(analogs), top_n)

    top_analogs = []
    min_sep = 7

    for analog in analogs:
        analog_date = pd.Timestamp(analog['date'])
        too_close = any(
            abs((analog_date - pd.Timestamp(sel['date'])).days) < min_sep
            for sel in top_analogs
        )
        if not too_close:
            top_analogs.append(analog)
        if len(top_analogs) >= top_n:
            break

    return {
        'init_time': init_time,
        'current_date_str': current_date_str,
        'current_wave_num': current_wave_num,
        'ds': ds,
        'pressure_coord': pressure_coord,
        'lat_coord_name': lat_coord_name,
        'lon_coord_name': lon_coord_name,
        'era5_lats_overlap': era5_lats_overlap,
        'era5_lons_overlap': era5_lons_overlap,
        'current_z500': current_z500,
        'current_anomaly': current_anomaly,
        'climatology': climatology,
        'top_analogs': top_analogs,
        'method': method,
    }


@app.route('/api/era5/analog-compare-map')
def api_era5_analog_compare_map():
    """
    Generate a side-by-side comparison of a historical ERA5 500mb pattern
    (analog date) vs the current GFS 500mb pattern.

    Query params:
        - date: Analog date (YYYY-MM-DD)

    Returns base64 PNG comparison image.
    """
    try:
        import xarray as xr
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from io import BytesIO
        import base64
        import pandas as pd
        from datetime import datetime as _dt

        date_str = request.args.get('date')
        if not date_str:
            return jsonify({"success": False, "error": "date parameter required (YYYY-MM-DD)"})

        target_date = _dt.fromisoformat(date_str)

        # ---- Load ERA5 data ----
        era5_files = sorted(Path("/Volumes/T7/Weather_Models/era5/global_500mb").glob("era5_z500_NH_*.nc"))
        if not era5_files:
            return jsonify({"success": False, "error": "No ERA5 data found"})

        datasets = []
        for f in era5_files:
            try:
                datasets.append(xr.open_dataset(f))
            except Exception:
                pass

        if not datasets:
            return jsonify({"success": False, "error": "Failed to load ERA5 datasets"})

        ds = xr.concat(datasets, dim='time').sortby('time')

        pressure_coord = None
        for c in ['pressure_level', 'level', 'plev', 'lev']:
            if c in ds.coords or c in ds.dims:
                pressure_coord = c
                break
        if pressure_coord is None:
            return jsonify({"success": False, "error": "Could not find pressure level coordinate"})

        lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'

        # Select nearest ERA5 timestep to requested date
        analog_slice = ds.sel(time=target_date, method='nearest')
        analog_time  = pd.Timestamp(analog_slice.time.values)
        era5_z500    = analog_slice['z'].sel(**{pressure_coord: 500})
        era5_lats    = era5_z500[lat_coord].values
        era5_lons    = era5_z500[lon_coord].values
        era5_heights = era5_z500.values / 10.0  # m → dam

        # Climatology for ERA5 (shared between both panels)
        target_doy  = analog_time.dayofyear
        clim_ds     = ds.sel(time=slice('1990-01-01', '2021-12-31'))
        ds_doy      = clim_ds.assign_coords(dayofyear=clim_ds.time.dt.dayofyear)
        window_days = [(target_doy + o - 1) % 366 + 1 for o in range(-15, 16)]
        mask        = ds_doy.dayofyear.isin(window_days)
        if mask.sum() > 0:
            clim_z500_raw = ds_doy.where(mask, drop=True)['z'].sel(**{pressure_coord: 500}).mean(dim='time')
            clim_heights  = clim_z500_raw.values / 10.0  # m → dam
        else:
            clim_heights  = np.mean(era5_heights, axis=1, keepdims=True) * np.ones_like(era5_heights)

        era5_anom = era5_heights - clim_heights

        # Climatology for GFS panel (use current DOY)
        gfs_model   = GFSModel()
        gfs_init    = pd.Timestamp(gfs_model.get_latest_init_time())
        current_doy = gfs_init.dayofyear
        gfs_window  = [(current_doy + o - 1) % 366 + 1 for o in range(-15, 16)]
        gfs_mask    = ds_doy.dayofyear.isin(gfs_window)
        if gfs_mask.sum() > 0:
            gfs_clim_raw  = ds_doy.where(gfs_mask, drop=True)['z'].sel(**{pressure_coord: 500}).mean(dim='time')
            gfs_clim_vals = gfs_clim_raw.values / 10.0  # m → dam
        else:
            gfs_clim_vals = None  # fallback handled below

        # ---- Live-fetch GFS z500 ----
        global_region = Region("Global", (-180, 180, 20, 70))
        gfs_z500_da   = gfs_model.fetch_data(Z500_GFS, gfs_init.to_pydatetime(), 0, global_region)

        gfs_lats_raw = gfs_z500_da.latitude.values
        gfs_vals_raw = gfs_z500_da.values  # already in dm (fetch_data applies gpm→dm conversion)

        # Regrid GFS to ERA5 grid for anomaly computation.
        # Use assign_coords+sortby to ensure monotonic longitude for interp().
        # np.where() alone produces non-monotonic arrays that cause interp() to return NaN.
        gfs_da_tmp = xr.DataArray(
            gfs_vals_raw,
            coords={'latitude': gfs_lats_raw, 'longitude': gfs_z500_da.longitude.values},
            dims=['latitude', 'longitude'],
        )
        if gfs_da_tmp.longitude.values.max() > 180:
            gfs_da_tmp = gfs_da_tmp.assign_coords(
                longitude=(((gfs_da_tmp.longitude + 180) % 360) - 180)
            ).sortby('longitude')
        gfs_da_regrid = gfs_da_tmp.interp(latitude=era5_lats, longitude=era5_lons, method='linear')
        gfs_heights = gfs_da_regrid.values  # dm, no further division needed

        if gfs_clim_vals is not None:
            gfs_anom = gfs_heights - gfs_clim_vals
        else:
            gfs_anom = gfs_heights - np.nanmean(gfs_heights, axis=1, keepdims=True)

        # Wave numbers
        era5_wave = rossby_waves.calculate_wave_number(era5_z500, latitude=55.0)
        gfs_wave  = rossby_waves.calculate_wave_number(gfs_z500_da, latitude=55.0)

        # ---- Build figure ----
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                                  subplot_kw={'projection': proj})

        anom_levels  = np.arange(-30, 31, 2)
        height_levels = np.arange(460, 620, 6)
        cmap = 'RdBu_r'

        panel_data = [
            (axes[0], era5_heights, era5_anom,
             f"ERA5  |  {analog_time.strftime('%Y-%m-%d')}",
             era5_wave, era5_lons, era5_lats),
            (axes[1], gfs_heights, gfs_anom,
             f"GFS  |  {gfs_init.strftime('%Y-%m-%d %HZ')}  (current)",
             gfs_wave,  era5_lons, era5_lats),
        ]

        cf_last = None
        for ax, heights, anom, title, wave_res, lons, lats in panel_data:
            ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.5, color='gray',
                         alpha=0.4, linestyle='--')

            cf = ax.contourf(lons, lats, anom, levels=anom_levels, cmap=cmap,
                             transform=ccrs.PlateCarree(), extend='both')
            cs = ax.contour(lons, lats, heights, levels=height_levels[::2],
                            colors='black', linewidths=0.8,
                            transform=ccrs.PlateCarree(), alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

            wn  = wave_res.get('wave_number', '?')
            ax.set_title(f"{title}\nWave #{wn}", fontsize=11, fontweight='bold')
            cf_last = cf

        # Shared colorbar
        cbar = fig.colorbar(cf_last, ax=axes, orientation='horizontal',
                            pad=0.04, shrink=0.6, fraction=0.03)
        cbar.set_label('500mb Height Anomaly (dam)', fontsize=10)

        fig.suptitle('Pattern Comparison: Analog vs Current GFS', fontsize=13,
                     fontweight='bold', y=1.01)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        return jsonify({
            "success": True,
            "map_image": img_b64,
            "analog_date": analog_time.strftime('%Y-%m-%d'),
            "gfs_init":    gfs_init.strftime('%Y-%m-%d %HZ'),
            "analog_wave": era5_wave.get('wave_number'),
            "current_wave": gfs_wave.get('wave_number'),
        })

    except Exception as e:
        logger.error(f"Error generating analog comparison map: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/era5/analogs')
def api_era5_analogs():
    """
    Find historical analog dates with similar 500mb height patterns.

    Query params:
        - top_n:  Number of top analogs to return (default: 10)
        - method: 'composite' (default), 'lat_pearson', or 'pearson' (legacy)

    Returns:
        JSON with analog dates sorted by composite pattern similarity
    """
    try:
        import xarray as xr
        import numpy as np
        import pandas as pd

        top_n = int(request.args.get('top_n', 10))
        method = request.args.get('method', 'composite')

        # Run the shared pattern-matching core
        core = _find_analogs_core(top_n=top_n, method=method)

        top_analogs     = core['top_analogs']
        init_time       = core['init_time']
        current_date_str = core['current_date_str']
        current_wave_num = core['current_wave_num']

        # Load Fairfax weather data to calculate analog outcomes
        weather_path = Path("/Volumes/T7/Weather_Models/era5/Fairfax/reanalysis-era5-single-levels-timeseries-sfc1zs15i59.nc")

        if weather_path.exists():
            try:
                weather_ds = xr.open_dataset(weather_path)

                for analog in top_analogs:
                    try:
                        analog_date = pd.Timestamp(analog['date'])
                        end_date = analog_date + pd.Timedelta(days=14)

                        precip_subset = weather_ds['tp'].sel(valid_time=slice(analog_date, end_date))
                        total_precip_m = float(precip_subset.sum().values)
                        analog['precip_14d'] = round(total_precip_m * 39.3701, 2)

                        temp_subset = weather_ds['t2m'].sel(valid_time=slice(analog_date, end_date))
                        avg_temp_k = float(temp_subset.mean().values)
                        analog['temp_14d'] = round((avg_temp_k - 273.15) * 9/5 + 32, 1)

                    except Exception as e:
                        logger.warning(f"Could not get weather data for analog {analog['date']}: {e}")
                        analog['precip_14d'] = None
                        analog['temp_14d'] = None

                precip_values = [a['precip_14d'] for a in top_analogs if a.get('precip_14d') is not None]
                avg_precip = round(np.mean(precip_values), 2) if precip_values else None

                temp_values = [a['temp_14d'] for a in top_analogs if a.get('temp_14d') is not None]
                avg_temp = round(np.mean(temp_values), 1) if temp_values else None

                try:
                    current_month = init_time.month
                    current_day   = init_time.day
                    climatology_precip: list = []
                    climatology_temp: list = []

                    for year in range(1940, 2026):
                        try:
                            start_date = pd.Timestamp(year=year, month=current_month, day=current_day)
                            end_date   = start_date + pd.Timedelta(days=14)

                            cp_sum = float(weather_ds['tp'].sel(valid_time=slice(start_date, end_date)).sum().values)
                            if not np.isnan(cp_sum):
                                climatology_precip.append(cp_sum * 39.3701)

                            ct_mean = float(weather_ds['t2m'].sel(valid_time=slice(start_date, end_date)).mean().values)
                            if not np.isnan(ct_mean):
                                climatology_temp.append((ct_mean - 273.15) * 9/5 + 32)
                        except Exception:
                            continue

                    climatology_normal_precip = round(np.mean(climatology_precip), 2) if climatology_precip else None
                    climatology_normal_temp   = round(np.mean(climatology_temp), 1)   if climatology_temp   else None

                except Exception as e:
                    logger.warning(f"Could not calculate climatology: {e}")
                    climatology_normal_precip = None
                    climatology_normal_temp   = None

            except Exception as e:
                logger.warning(f"Could not load Fairfax weather data: {e}")
                avg_precip = avg_temp = None
                climatology_normal_precip = climatology_normal_temp = None
        else:
            logger.warning(f"Fairfax weather data not found at {weather_path}")
            avg_precip = avg_temp = None
            climatology_normal_precip = climatology_normal_temp = None

        # Save prediction to history
        if avg_precip is not None and avg_temp is not None:
            avg_correlation = (sum(a['correlation'] for a in top_analogs) / len(top_analogs)) if top_analogs else 0
            save_analog_prediction(
                target_date=current_date_str,
                analog_precip=avg_precip,
                analog_temp=avg_temp,
                climatology_precip=climatology_normal_precip or 0,
                climatology_temp=climatology_normal_temp or 0,
                top_analogs=[{'date': a['date'], 'correlation': a['correlation']} for a in top_analogs],
                avg_correlation=avg_correlation,
            )

        return jsonify({
            "success": True,
            "current_date": current_date_str,
            "current_wave_number": current_wave_num,
            "method": method,
            "analogs": top_analogs,
            "avg_precip_14d": avg_precip,
            "avg_temp_14d": avg_temp,
            "climatology_precip_14d": climatology_normal_precip,
            "climatology_temp_14d": climatology_normal_temp,
        })

    except Exception as e:
        logger.error(f"Error finding analogs: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


def load_conus_precip_for_dates(dates, days=14):
    """
    Load ERA5 CONUS precipitation for specified date ranges.

    Args:
        dates: List of pd.Timestamp objects
        days: Number of days to include after each date

    Returns:
        xarray.Dataset with precipitation data
    """
    import glob
    import xarray as xr
    from pathlib import Path

    conus_precip_dir = Path("/Volumes/T7/Weather_Models/era5/conus_daily_precip")

    # Find all needed files
    all_files = sorted(glob.glob(str(conus_precip_dir / "era5_tp_daily_CONUS_*_M*.nc")))

    if not all_files:
        raise FileNotFoundError(f"No CONUS precipitation files found in {conus_precip_dir}")

    # Load with xarray
    ds = xr.open_mfdataset(all_files, combine='by_coords', parallel=True, engine='netcdf4')

    return ds


def load_conus_temp_for_dates(dates, days=14):
    """
    Load ERA5 CONUS 2m temperature for specified date ranges.

    Args:
        dates: List of pd.Timestamp objects
        days: Number of days to include after each date

    Returns:
        xarray.Dataset with temperature data
    """
    import glob
    import xarray as xr
    from pathlib import Path

    conus_temp_dir = Path("/Volumes/T7/Weather_Models/era5/conus_daily_temp")

    # Find all needed files
    all_files = sorted(glob.glob(str(conus_temp_dir / "era5_t2m_daily_CONUS_*_M*.nc")))

    if not all_files:
        raise FileNotFoundError(f"No CONUS temperature files found in {conus_temp_dir}. Please download temperature data using: python3 era5_bulk_download.py --dataset temp --start-year 1990 --end-year 2020 --chunk-months 1")

    # Load with xarray
    ds = xr.open_mfdataset(all_files, combine='by_coords', parallel=True, engine='netcdf4')

    return ds


def load_conus_climatology(variable, cache={}):
    """
    Load and cache climatology files.

    Args:
        variable: 'precip' or 'temp'
        cache: Dictionary for caching loaded datasets

    Returns:
        xarray.Dataset with climatology data
    """
    import xarray as xr
    from pathlib import Path

    if variable in cache:
        return cache[variable]

    clim_dir = Path("/Volumes/T7/Weather_Models/era5")

    if variable == 'precip':
        clim_file = clim_dir / 'conus_precip_climatology_14day.nc'
    elif variable == 'temp':
        clim_file = clim_dir / 'conus_temp_climatology_14day.nc'
    else:
        raise ValueError(f"Unknown variable: {variable}")

    if not clim_file.exists():
        raise FileNotFoundError(
            f"Climatology file not found: {clim_file}. "
            f"Please compute climatology using: "
            f"python3 era5_bulk_download.py --dataset climatology --start-year 1990 --end-year 2020"
        )

    ds = xr.open_dataset(clim_file)
    cache[variable] = ds

    return ds


def compute_analog_grid(ds, dates, variable_name, days=14, aggregation='sum'):
    """
    Compute 14-day windows for all analogs and return averaged grid.

    Args:
        ds: xarray.Dataset with the variable data
        dates: List of analog dates (pd.Timestamp objects)
        variable_name: Name of variable in dataset ('tp' or 't2m')
        days: Number of days in window (default 14)
        aggregation: 'sum' for precipitation, 'mean' for temperature

    Returns:
        numpy array with shape (lat, lon) containing averaged analog values
    """
    import pandas as pd
    import numpy as np

    analog_grids = []

    for analog_date in dates:
        try:
            end_date = analog_date + pd.Timedelta(days=days - 1)  # 14 days inclusive

            # Select the time window
            window_data = ds[variable_name].sel(time=slice(analog_date, end_date))

            # Check if we have enough data
            if len(window_data.time) < 10:  # Require at least 10 days
                logger.warning(f"Insufficient data for analog date {analog_date}: only {len(window_data.time)} days")
                continue

            # Aggregate
            if aggregation == 'sum':
                analog_value = window_data.sum(dim='time')
            else:  # mean
                analog_value = window_data.mean(dim='time')

            analog_grids.append(analog_value.values)

        except Exception as e:
            logger.warning(f"Failed to process analog date {analog_date}: {e}")
            continue

    if not analog_grids:
        raise ValueError("No valid analog grids computed")

    # Average across all analogs
    avg_grid = np.mean(analog_grids, axis=0)

    return avg_grid


@app.route('/api/era5/analogs-conus')
def api_era5_analogs_conus():
    """
    Find historical analog dates (reuse existing logic) and compute
    14-day precipitation and temperature outcomes for entire CONUS grid.

    Query params:
        - top_n: Number of analogs (default 10)
        - variable: 'precip', 'temp', or 'both' (default 'both')

    Returns:
        JSON with gridded analog and climatology data
    """
    try:
        import xarray as xr
        import numpy as np
        import pandas as pd

        top_n = int(request.args.get('top_n', 10))
        variable_filter = request.args.get('variable', 'both')
        method = request.args.get('method', 'composite')

        # ========== STEP 1: PATTERN MATCHING (shared core) ==========
        core = _find_analogs_core(top_n=top_n, method=method)

        top_analogs      = core['top_analogs']
        init_time        = core['init_time']
        current_date_str = core['current_date_str']
        current_wave_num = core['current_wave_num']
        current_doy      = init_time.dayofyear

        analog_dates     = [pd.Timestamp(a['date']) for a in top_analogs]
        analog_date_strs = [a['date'] for a in top_analogs]

        # Build analog_correlations list for UI display
        analog_correlations = [
            {
                'date': a['date'],
                'correlation': a.get('correlation'),
                'composite_score': a.get('composite_score'),
            }
            for a in top_analogs
        ]

        # ========== STEP 2: LOAD CONUS DATA AND COMPUTE GRIDS ==========

        result = {
            "success": True,
            "current_date": current_date_str,
            "current_wave_number": current_wave_num,
            "method": method,
            "analog_dates": analog_date_strs,
            "analog_correlations": analog_correlations,
            "grid": {},
            "precip": None,
            "temp": None,
        }

        # Process precipitation if requested
        if variable_filter in ['precip', 'both']:
            try:
                logger.info("Loading CONUS precipitation data...")
                precip_ds = load_conus_precip_for_dates(analog_dates)

                logger.info("Computing precipitation analog grid...")
                avg_precip_grid = compute_analog_grid(precip_ds, analog_dates, 'tp', days=14, aggregation='sum')

                logger.info("Loading precipitation climatology...")
                precip_clim_ds = load_conus_climatology('precip')
                clim_precip = precip_clim_ds['precip_14d_clim'].sel(doy=current_doy).values

                # Calculate anomalies
                anomaly_precip = avg_precip_grid - clim_precip

                # Avoid division by zero, use small epsilon
                clim_precip_safe = np.where(clim_precip > 0.1, clim_precip, np.nan)
                anomaly_percent = 100 * (avg_precip_grid - clim_precip) / clim_precip_safe

                # Convert mm to inches (1 mm = 1/25.4 inches)
                avg_precip_grid_in = avg_precip_grid / 25.4
                clim_precip_in = clim_precip / 25.4
                anomaly_precip_in = anomaly_precip / 25.4

                # Store grid coordinates (same for both variables)
                if not result['grid']:
                    result['grid'] = {
                        'latitudes': precip_ds.latitude.values.tolist(),
                        'longitudes': precip_ds.longitude.values.tolist()
                    }

                result['precip'] = {
                    'analog_mm': avg_precip_grid.tolist(),
                    'climatology_mm': clim_precip.tolist(),
                    'anomaly_mm': anomaly_precip.tolist(),
                    'analog_in': avg_precip_grid_in.tolist(),
                    'climatology_in': clim_precip_in.tolist(),
                    'anomaly_in': anomaly_precip_in.tolist(),
                    'anomaly_percent': np.where(np.isfinite(anomaly_percent), anomaly_percent, 0).tolist()
                }

                logger.info(f"Precipitation grid computed: {avg_precip_grid.shape}")

                # Cleanup
                precip_ds.close()
                precip_clim_ds.close()

            except Exception as e:
                logger.error(f"Failed to process precipitation: {e}", exc_info=True)
                result['precip_error'] = str(e)

        # Process temperature if requested
        if variable_filter in ['temp', 'both']:
            try:
                logger.info("Loading CONUS temperature data...")
                temp_ds = load_conus_temp_for_dates(analog_dates)

                logger.info("Computing temperature analog grid...")
                avg_temp_grid = compute_analog_grid(temp_ds, analog_dates, 't2m', days=14, aggregation='mean')

                logger.info("Loading temperature climatology...")
                temp_clim_ds = load_conus_climatology('temp')
                clim_temp = temp_clim_ds['temp_14d_clim'].sel(doy=current_doy).values

                # Calculate anomalies
                anomaly_temp = avg_temp_grid - clim_temp

                # Convert Celsius to Fahrenheit (°F = °C * 9/5 + 32)
                avg_temp_grid_f = avg_temp_grid * 9/5 + 32
                clim_temp_f = clim_temp * 9/5 + 32
                anomaly_temp_f = anomaly_temp * 9/5  # Anomaly doesn't need +32 offset

                # For temperature, percent anomaly is less meaningful, but calculate anyway
                # Use absolute zero in Celsius (-273.15) offset to avoid division issues
                temp_absolute = avg_temp_grid + 273.15
                clim_absolute = clim_temp + 273.15
                anomaly_percent_temp = 100 * (temp_absolute - clim_absolute) / clim_absolute

                # Store grid coordinates if not already stored
                if not result['grid']:
                    result['grid'] = {
                        'latitudes': temp_ds.latitude.values.tolist(),
                        'longitudes': temp_ds.longitude.values.tolist()
                    }

                result['temp'] = {
                    'analog_c': avg_temp_grid.tolist(),
                    'climatology_c': clim_temp.tolist(),
                    'anomaly_c': anomaly_temp.tolist(),
                    'analog_f': avg_temp_grid_f.tolist(),
                    'climatology_f': clim_temp_f.tolist(),
                    'anomaly_f': anomaly_temp_f.tolist(),
                    'anomaly_percent': np.where(np.isfinite(anomaly_percent_temp), anomaly_percent_temp, 0).tolist()
                }

                logger.info(f"Temperature grid computed: {avg_temp_grid.shape}")

                # Cleanup
                temp_ds.close()
                temp_clim_ds.close()

            except Exception as e:
                logger.error(f"Failed to process temperature: {e}", exc_info=True)
                result['temp_error'] = str(e)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error building CONUS analog grid: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


def _load_era5_z500_last_n_years(n_years: int = 30):
    import xarray as xr
    import re
    from datetime import datetime

    era5_path = Path("/Volumes/T7/Weather_Models/era5/global_500mb")
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 data directory not found: {era5_path}")

    range_re = re.compile(r"(\d{4})-(\d{4})")
    year_re = re.compile(r"(\d{4})")
    end_year = datetime.now(timezone.utc).year
    start_year = end_year - n_years + 1

    era5_files = []
    for f in sorted(era5_path.glob("era5_z500_NH_*.nc")):
        match_range = range_re.search(f.name)
        if match_range:
            start = int(match_range.group(1))
            end = int(match_range.group(2))
            if end >= start_year:
                era5_files.append(f)
            continue
        match_year = year_re.search(f.name)
        if match_year:
            year = int(match_year.group(1))
            if year >= start_year:
                era5_files.append(f)

    if not era5_files:
        raise FileNotFoundError(f"No ERA5 data found for requested years >= {start_year}")

    datasets = []
    for f in era5_files:
        try:
            datasets.append(xr.open_dataset(f))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not datasets:
        raise RuntimeError("Failed to load ERA5 datasets")

    ds = xr.concat(datasets, dim='time')
    ds = ds.sortby('time')

    pressure_coord = None
    for coord_name in ['pressure_level', 'level', 'plev', 'lev']:
        if coord_name in ds.coords or coord_name in ds.dims:
            pressure_coord = coord_name
            break

    if pressure_coord is None:
        raise RuntimeError("Could not find pressure level coordinate in dataset")

    lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    return ds, pressure_coord, lat_coord_name, lon_coord_name


def _train_som(data: np.ndarray, rows: int, cols: int, n_iter: int = 2000, lr: float = 0.5):
    rng = np.random.default_rng(42)
    n_samples, n_features = data.shape
    weights = data[rng.choice(n_samples, rows * cols, replace=False)].reshape(rows, cols, n_features).copy()
    grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    sigma0 = max(rows, cols) / 2.0

    for t in range(n_iter):
        x = data[rng.integers(0, n_samples)]
        dists = ((weights - x) ** 2).sum(axis=2)
        bmu_index = np.unravel_index(np.argmin(dists), dists.shape)
        lr_t = lr * (1 - t / n_iter)
        sigma_t = max(0.5, sigma0 * (1 - t / n_iter))

        dist_sq = (grid_r - bmu_index[0]) ** 2 + (grid_c - bmu_index[1]) ** 2
        influence = np.exp(-dist_sq / (2 * sigma_t ** 2))
        weights += influence[..., None] * lr_t * (x - weights)

    return weights


def _assign_som_bmu(data: np.ndarray, weights: np.ndarray, batch_size: int = 256):
    rows, cols, n_features = weights.shape
    weights_flat = weights.reshape(rows * cols, n_features)
    bmu = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        dists = ((batch[:, None, :] - weights_flat[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(dists, axis=1)
        bmu.extend(idx.tolist())
    return np.array(bmu, dtype=np.int32)


def _load_or_build_era5_som(rows: int = 6, cols: int = 6, n_years: int = 30, stride: int = 6):
    global _SOM_CACHE
    if _SOM_CACHE is not None:
        return _SOM_CACHE

    if SOM_CACHE_FILE.exists():
        try:
            cached = np.load(SOM_CACHE_FILE, allow_pickle=True)
            meta = cached["meta"].item()
            if (meta.get("rows") == rows and meta.get("cols") == cols and
                    meta.get("n_years") == n_years and meta.get("stride") == stride and
                    meta.get("anomaly") is True):
                _SOM_CACHE = {
                    "rows": rows,
                    "cols": cols,
                    "dates": cached["dates"].tolist(),
                    "bmu": cached["bmu"],
                    "weights": cached["weights"],
                    "mean": cached["mean"],
                    "std": cached["std"],
                    "climatology": cached["climatology"],
                    "lats": cached["lats"],
                    "lons": cached["lons"],
                    "lon_mode": meta.get("lon_mode", "unknown"),
                    "meta": meta
                }
                return _SOM_CACHE
        except Exception as e:
            logger.warning(f"Failed to load SOM cache: {e}")

    ds, pressure_coord, lat_coord_name, lon_coord_name = _load_era5_z500_last_n_years(n_years)

    var_name = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]
    z500 = ds[var_name].sel({pressure_coord: 500})

    # Daily mean
    z500 = z500.resample(time="1D").mean()
    z500 = z500.dropna("time", how="all")

    # Downsample grid
    z500 = z500.isel({
        lat_coord_name: slice(None, None, stride),
        lon_coord_name: slice(None, None, stride)
    })

    # Build climatology (31-day window) and compute anomalies
    ds_with_doy = z500.assign_coords(dayofyear=z500.time.dt.dayofyear)
    climatology = {}
    for doy in range(1, 367):
        window_days = [(doy + offset - 1) % 366 + 1 for offset in range(-15, 16)]
        mask = ds_with_doy.dayofyear.isin(window_days)
        if mask.sum() > 0:
            window_data = ds_with_doy.where(mask, drop=True)
            climatology[doy] = window_data.mean(dim="time").values

    z500_vals = z500.values
    doys = z500["time"].dt.dayofyear.values
    anomalies = np.empty_like(z500_vals)
    for i, doy in enumerate(doys):
        clim = climatology.get(int(doy))
        if clim is None:
            zonal_mean = z500_vals[i].mean(axis=1, keepdims=True)
            anomalies[i] = z500_vals[i] - zonal_mean
        else:
            anomalies[i] = z500_vals[i] - clim

    dates = [np.datetime_as_string(d, unit="D") for d in z500["time"].values]
    data = anomalies.reshape(anomalies.shape[0], -1).astype(np.float32)

    # Standardize features
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    data = (data - mean) / std

    # Train SOM on subset for speed
    rng = np.random.default_rng(42)
    sample_size = min(2000, data.shape[0])
    sample_idx = rng.choice(data.shape[0], sample_size, replace=False)
    weights = _train_som(data[sample_idx], rows, cols, n_iter=2000, lr=0.5)

    bmu = _assign_som_bmu(data, weights, batch_size=256)

    lats = z500[lat_coord_name].values
    lons = z500[lon_coord_name].values
    lon_mode = "0_360" if np.nanmax(lons) > 180 else "-180_180"

    # Stack climatology to array for caching (366, lat, lon)
    clim_stack = np.full((366, lats.shape[0], lons.shape[0]), np.nan, dtype=np.float32)
    for doy in range(1, 367):
        if doy in climatology:
            clim_stack[doy - 1] = climatology[doy].astype(np.float32)

    meta = {
        "rows": rows,
        "cols": cols,
        "n_years": n_years,
        "stride": stride,
        "anomaly": True,
        "climatology_window_days": 31,
        "lon_mode": lon_mode,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    np.savez_compressed(
        SOM_CACHE_FILE,
        dates=np.array(dates, dtype=object),
        bmu=bmu,
        weights=weights.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        climatology=clim_stack,
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        meta=meta
    )

    _SOM_CACHE = {
        "rows": rows,
        "cols": cols,
        "dates": dates,
        "bmu": bmu,
        "weights": weights,
        "mean": mean,
        "std": std,
        "climatology": clim_stack,
        "lats": lats,
        "lons": lons,
        "lon_mode": lon_mode,
        "meta": meta
    }
    return _SOM_CACHE


def _compute_latest_gfs_som_cluster(som: dict) -> tuple[str | None, tuple[int, int] | None]:
    try:
        import xarray as xr
        db = load_forecasts_db()
        location_name = "Fairfax, VA"
        if location_name not in db or not db[location_name].get("latest_run"):
            return None, None

        latest_run = db[location_name]["latest_run"]
        run_data = db[location_name]["runs"].get(latest_run, {})
        gfs_data = run_data.get("gfs", {})
        z500_field = gfs_data.get("z500_field")
        if not z500_field:
            return None, None

        gfs_vals = np.array(z500_field["values"], dtype=np.float32) * 10.0  # dm -> m
        gfs_lats = np.array(z500_field["latitude"], dtype=np.float32)
        gfs_lons = np.array(z500_field["longitude"], dtype=np.float32)

        era5_lats = som["lats"]
        era5_lons = som["lons"]
        lon_mode = som.get("lon_mode", "-180_180")

        gfs_da = xr.DataArray(
            gfs_vals,
            coords={"latitude": gfs_lats, "longitude": gfs_lons},
            dims=["latitude", "longitude"]
        )

        if lon_mode == "0_360":
            if gfs_da.longitude.values.min() < 0:
                gfs_da = gfs_da.assign_coords(
                    longitude=(gfs_da.longitude % 360)
                ).sortby("longitude")
        else:
            if gfs_da.longitude.values.max() > 180:
                gfs_da = gfs_da.assign_coords(
                    longitude=(((gfs_da.longitude + 180) % 360) - 180)
                ).sortby("longitude")

        gfs_regrid = gfs_da.interp(
            latitude=era5_lats,
            longitude=era5_lons,
            method="linear"
        ).values

        init_time = datetime.fromisoformat(latest_run)
        doy = init_time.timetuple().tm_yday
        clim_stack = som["climatology"]
        clim = clim_stack[doy - 1] if 1 <= doy <= 366 else None

        if clim is None or np.isnan(clim).all():
            zonal_mean = np.nanmean(gfs_regrid, axis=1, keepdims=True)
            anomaly = gfs_regrid - zonal_mean
        else:
            anomaly = gfs_regrid - clim

        flat = anomaly.reshape(-1).astype(np.float32)
        mean = som["mean"]
        std = som["std"]
        std_safe = np.where(std == 0, 1.0, std)
        flat = (flat - mean) / std_safe
        if np.any(~np.isfinite(flat)):
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        weights = som["weights"]
        weights_flat = weights.reshape(weights.shape[0] * weights.shape[1], -1)
        dists = ((weights_flat - flat) ** 2).sum(axis=1)
        idx = int(np.argmin(dists))
        return latest_run.split("T")[0], (idx // som["cols"], idx % som["cols"])

    except Exception as e:
        logger.warning(f"Failed to compute GFS SOM cluster: {e}")
        return None, None


def _load_era5_anomaly_matrix_for_som(n_years: int = 30, stride: int = 6) -> dict:
    """
    Load ERA5 Z500 anomalies on the SOM grid and return anomalies + dates.
    Cached in-memory for reuse by SOM cluster analog lookups.
    """
    global _SOM_ANOM_CACHE
    if _SOM_ANOM_CACHE is not None:
        return _SOM_ANOM_CACHE

    ds, pressure_coord, lat_coord_name, lon_coord_name = _load_era5_z500_last_n_years(n_years)
    var_name = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]
    z500 = ds[var_name].sel({pressure_coord: 500})

    z500 = z500.resample(time="1D").mean()
    z500 = z500.dropna("time", how="all")

    z500 = z500.isel({
        lat_coord_name: slice(None, None, stride),
        lon_coord_name: slice(None, None, stride)
    })

    lats = z500[lat_coord_name].values
    lons = z500[lon_coord_name].values

    ds_with_doy = z500.assign_coords(dayofyear=z500.time.dt.dayofyear)
    climatology = {}
    for doy in range(1, 367):
        window_days = [(doy + offset - 1) % 366 + 1 for offset in range(-15, 16)]
        mask = ds_with_doy.dayofyear.isin(window_days)
        if mask.sum() > 0:
            window_data = ds_with_doy.where(mask, drop=True)
            climatology[doy] = window_data.mean(dim="time").values

    z500_vals = z500.values
    doys = z500["time"].dt.dayofyear.values
    anomalies = np.empty_like(z500_vals)
    for i, doy in enumerate(doys):
        clim = climatology.get(int(doy))
        if clim is None:
            zonal_mean = z500_vals[i].mean(axis=1, keepdims=True)
            anomalies[i] = z500_vals[i] - zonal_mean
        else:
            anomalies[i] = z500_vals[i] - clim

    dates = [np.datetime_as_string(d, unit="D") for d in z500["time"].values]

    _SOM_ANOM_CACHE = {
        "dates": dates,
        "anomalies": anomalies.astype(np.float32),
        "lats": lats,
        "lons": lons
    }
    return _SOM_ANOM_CACHE


def _compute_gfs_som_cluster_for_run(
    som: dict,
    run_time_str: str,
    forecasts_db: dict,
    location_name: str = "Fairfax, VA"
) -> tuple[int, int] | None:
    try:
        import xarray as xr

        if location_name not in forecasts_db:
            return None

        runs = forecasts_db[location_name].get("runs", {})
        run_data = runs.get(run_time_str)
        if run_data is None:
            # Try without timezone offset if needed
            alt = run_time_str.split("+")[0]
            run_data = runs.get(alt)
        if run_data is None:
            return None

        gfs_data = run_data.get("gfs", {})
        z500_field = gfs_data.get("z500_field")
        if not z500_field:
            return None

        gfs_vals = np.array(z500_field["values"], dtype=np.float32) * 10.0
        gfs_lats = np.array(z500_field["latitude"], dtype=np.float32)
        gfs_lons = np.array(z500_field["longitude"], dtype=np.float32)

        era5_lats = som["lats"]
        era5_lons = som["lons"]
        lon_mode = som.get("lon_mode", "-180_180")

        gfs_da = xr.DataArray(
            gfs_vals,
            coords={"latitude": gfs_lats, "longitude": gfs_lons},
            dims=["latitude", "longitude"]
        )

        if lon_mode == "0_360":
            if gfs_da.longitude.values.min() < 0:
                gfs_da = gfs_da.assign_coords(
                    longitude=(gfs_da.longitude % 360)
                ).sortby("longitude")
        else:
            if gfs_da.longitude.values.max() > 180:
                gfs_da = gfs_da.assign_coords(
                    longitude=(((gfs_da.longitude + 180) % 360) - 180)
                ).sortby("longitude")

        gfs_regrid = gfs_da.interp(
            latitude=era5_lats,
            longitude=era5_lons,
            method="linear"
        ).values

        init_time = datetime.fromisoformat(run_time_str.split("+")[0])
        doy = init_time.timetuple().tm_yday
        clim_stack = som["climatology"]
        clim = clim_stack[doy - 1] if 1 <= doy <= 366 else None

        if clim is None or np.isnan(clim).all():
            zonal_mean = np.nanmean(gfs_regrid, axis=1, keepdims=True)
            anomaly = gfs_regrid - zonal_mean
        else:
            anomaly = gfs_regrid - clim

        flat = anomaly.reshape(-1).astype(np.float32)
        mean = som["mean"]
        std = som["std"]
        std_safe = np.where(std == 0, 1.0, std)
        flat = (flat - mean) / std_safe
        if np.any(~np.isfinite(flat)):
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        weights = som["weights"]
        weights_flat = weights.reshape(weights.shape[0] * weights.shape[1], -1)
        dists = ((weights_flat - flat) ** 2).sum(axis=1)
        idx = int(np.argmin(dists))
        return (idx // som["cols"], idx % som["cols"])

    except Exception as e:
        logger.warning(f"Failed to compute GFS SOM cluster for {run_time_str}: {e}")
        return None


def _load_som_skill_store() -> dict:
    if SOM_SKILL_FILE.exists():
        try:
            with SOM_SKILL_FILE.open("r") as f:
                store = json.load(f)
                # Backward-compat: migrate old schema (no variable dimension)
                if "clusters" in store and store.get("meta", {}).get("variables") is None:
                    old_clusters = store.get("clusters", {})
                    migrated = {}
                    for cluster_key, lt_dict in old_clusters.items():
                        migrated[cluster_key] = {"precip": lt_dict}
                    store["clusters"] = migrated
                    old_station = store.get("cluster_station", {})
                    migrated_station = {}
                    for cluster_key, lt_dict in old_station.items():
                        migrated_station[cluster_key] = {"precip": lt_dict}
                    store["cluster_station"] = migrated_station
                    store.setdefault("meta", {})["variables"] = ["precip", "temp"]
                # Backward-compat: migrate processed_runs to include variable
                processed = store.get("processed_runs", {})
                if processed and "precip" not in processed:
                    store["processed_runs"] = {
                        "precip": processed,
                        "temp": {k: [] for k in processed.keys()}
                    }
                return store
        except Exception as e:
            logger.warning(f"Failed to load SOM skill store: {e}")
    return {
        "meta": {
            "lead_times": [24, 48, 72, 120],
            "rows": 6,
            "cols": 6,
            "variables": ["precip", "temp"],
            "updated_at": None
        },
        "clusters": {},
        "cluster_station": {},
        "stations": {},
        "processed_runs": {
            "precip": {
                "24": [],
                "48": [],
                "72": [],
                "120": []
            },
            "temp": {
                "24": [],
                "48": [],
                "72": [],
                "120": []
            }
        }
    }


def _save_som_skill_store(store: dict) -> None:
    try:
        SOM_SKILL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SOM_SKILL_FILE.open("w") as f:
            json.dump(store, f)
    except Exception as e:
        logger.warning(f"Failed to save SOM skill store: {e}")


def update_som_cluster_skill() -> dict:
    """
    Incrementally update SOM cluster skill using GFS F000 patterns and
    NWS precip MAE/bias across ASOS stations.
    """
    try:
        import asos

        som = _load_or_build_era5_som(rows=6, cols=6, n_years=30, stride=6)
        store = _load_som_skill_store()

        lead_times = store.get("meta", {}).get("lead_times", [24, 48, 72, 120])
        variables = store.get("meta", {}).get("variables", ["precip", "temp"])
        processed = store.get("processed_runs", {})
        clusters = store.get("clusters", {})
        cluster_station = store.get("cluster_station", {})

        asos_db = asos.load_asos_forecasts_db()
        if not asos_db or "runs" not in asos_db:
            return {"success": False, "error": "ASOS verification data not available"}

        forecasts_db = load_forecasts_db()
        if not store.get("stations"):
            store["stations"] = asos_db.get("stations", {})

        updates = 0
        for run_time_str in asos_db["runs"].keys():
            for var in variables:
                for lt in lead_times:
                    lt_key = str(lt)
                    if run_time_str in processed.get(var, {}).get(lt_key, []):
                        continue

                    station_errors = asos._calculate_asos_station_errors(
                        asos_db,
                        run_time_str,
                        "nws",
                        var,
                        lt
                    )
                    if not station_errors:
                        continue  # do not mark processed yet

                    cluster = _compute_gfs_som_cluster_for_run(som, run_time_str, forecasts_db)
                    if cluster is None:
                        continue

                    r, c = cluster
                    cluster_key = f"{r},{c}"
                    clusters.setdefault(cluster_key, {})
                    clusters[cluster_key].setdefault(var, {})
                    clusters[cluster_key][var].setdefault(lt_key, {"sum_abs": 0.0, "sum": 0.0, "count": 0, "runs": 0})

                    cluster_station.setdefault(cluster_key, {})
                    cluster_station[cluster_key].setdefault(var, {})
                    cluster_station[cluster_key][var].setdefault(lt_key, {})

                    sum_abs = 0.0
                    sum_err = 0.0
                    count = 0
                    for station_id, err in station_errors.items():
                        sum_abs += abs(err)
                        sum_err += err
                        count += 1

                        station_bucket = cluster_station[cluster_key][var][lt_key].setdefault(
                            station_id, {"sum_abs": 0.0, "sum": 0.0, "count": 0}
                        )
                        station_bucket["sum_abs"] += abs(err)
                        station_bucket["sum"] += err
                        station_bucket["count"] += 1

                    clusters[cluster_key][var][lt_key]["sum_abs"] += sum_abs
                    clusters[cluster_key][var][lt_key]["sum"] += sum_err
                    clusters[cluster_key][var][lt_key]["count"] += count
                    clusters[cluster_key][var][lt_key]["runs"] += 1

                    processed.setdefault(var, {}).setdefault(lt_key, []).append(run_time_str)
                    updates += 1

        store["clusters"] = clusters
        store["cluster_station"] = cluster_station
        store["processed_runs"] = processed
        store.setdefault("meta", {})["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_som_skill_store(store)

        return {"success": True, "updates": updates}

    except Exception as e:
        logger.error(f"Failed to update SOM cluster skill: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.route("/api/era5/som-skill")
def api_era5_som_skill():
    """
    Cluster ERA5 500mb heights with a SOM and relate clusters to NWS precip MAE/bias
    over the ASOS network.
    """
    lead_time = int(request.args.get("lead_time", 72))
    variable = request.args.get("variable", "precip")
    try:
        som = _load_or_build_era5_som(rows=6, cols=6, n_years=30, stride=6)
        rows = som["rows"]
        cols = som["cols"]
        store = _load_som_skill_store()
        clusters = store.get("clusters", {})

        nodes = []
        all_mae = []
        all_bias = []
        lt_key = str(lead_time)

        for r in range(rows):
            for c in range(cols):
                cluster_key = f"{r},{c}"
                node = {"r": r, "c": c}
                if variable == "both":
                    for var in ["precip", "temp"]:
                        stats = clusters.get(cluster_key, {}).get(var, {}).get(lt_key, {})
                        count = stats.get("count", 0)
                        runs = stats.get("runs", 0)
                        if count > 0:
                            mae_mean = stats.get("sum_abs", 0.0) / count
                            bias_mean = stats.get("sum", 0.0) / count
                            all_mae.append(mae_mean)
                            all_bias.append(bias_mean)
                        else:
                            mae_mean = None
                            bias_mean = None
                        node[f"{var}_mae_mean"] = round(mae_mean, 3) if mae_mean is not None else None
                        node[f"{var}_bias_mean"] = round(bias_mean, 3) if bias_mean is not None else None
                        node[f"{var}_count"] = int(count)
                        node[f"{var}_runs"] = int(runs)
                else:
                    stats = clusters.get(cluster_key, {}).get(variable, {}).get(lt_key, {})
                    count = stats.get("count", 0)
                    runs = stats.get("runs", 0)
                    if count > 0:
                        mae_mean = stats.get("sum_abs", 0.0) / count
                        bias_mean = stats.get("sum", 0.0) / count
                        all_mae.append(mae_mean)
                        all_bias.append(bias_mean)
                    else:
                        mae_mean = None
                        bias_mean = None
                    node.update({
                        "mae_mean": round(mae_mean, 3) if mae_mean is not None else None,
                        "bias_mean": round(bias_mean, 3) if bias_mean is not None else None,
                        "count": int(count),
                        "runs": int(runs)
                    })
                nodes.append(node)

        overall_mae = sum(all_mae) / len(all_mae) if all_mae else None
        overall_bias = sum(all_bias) / len(all_bias) if all_bias else None

        latest_gfs_date, latest_gfs_cluster = _compute_latest_gfs_som_cluster(som)

        result = {
            "success": True,
            "rows": rows,
            "cols": cols,
            "lead_time": lead_time,
            "metric": f"nws_{variable}",
            "nodes": nodes,
            "overall_mae": round(overall_mae, 3) if overall_mae is not None else None,
            "overall_bias": round(overall_bias, 3) if overall_bias is not None else None,
            "latest_gfs_date": latest_gfs_date,
            "latest_gfs_cluster": latest_gfs_cluster
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error building SOM skill analysis: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/era5/som-cluster-stations")
def api_era5_som_cluster_stations():
    """
    Return per-station MAE/bias for a given SOM cluster and lead time.
    """
    try:
        import warnings
        cluster = request.args.get("cluster", "")
        lead_time = request.args.get("lead_time", "72")
        variable = request.args.get("variable", "precip")
        if not cluster or "," not in cluster:
            return jsonify({"success": False, "error": "cluster is required as r,c"})
        r_str, c_str = cluster.split(",", 1)
        cluster_key = f"{int(r_str)},{int(c_str)}"
        lt_key = str(int(lead_time))

        store = _load_som_skill_store()
        stations_meta = store.get("stations", {})
        cluster_station = store.get("cluster_station", {})
        stats = cluster_station.get(cluster_key, {}).get(variable, {}).get(lt_key, {})

        points = []
        for station_id, s in stats.items():
            count = s.get("count", 0)
            if count <= 0:
                continue
            meta = stations_meta.get(station_id, {})
            mae = s.get("sum_abs", 0.0) / count
            bias = s.get("sum", 0.0) / count
            points.append({
                "station_id": station_id,
                "name": meta.get("name"),
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "mae": round(mae, 3),
                "bias": round(bias, 3),
                "count": int(count)
            })

        return jsonify({
            "success": True,
            "cluster": cluster_key,
            "lead_time": int(lt_key),
            "stations": points
        })

    except Exception as e:
        logger.error(f"Error loading SOM cluster stations: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/era5/som-cluster-analogs")
def api_era5_som_cluster_analogs():
    """
    Return top-N representative ERA5 anomaly dates for a SOM cluster using
    the composite analog score (pattern + gradient + rmse similarity).
    """
    try:
        cluster = request.args.get("cluster", "")
        top_n = int(request.args.get("top_n", 5))
        if not cluster or "," not in cluster:
            return jsonify({"success": False, "error": "cluster is required as r,c"})
        r_str, c_str = cluster.split(",", 1)
        r = int(r_str)
        c = int(c_str)

        som = _load_or_build_era5_som(rows=6, cols=6, n_years=30, stride=6)
        bmu = som["bmu"]
        rows = som["rows"]
        cols = som["cols"]
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return jsonify({"success": False, "error": "cluster out of range"})

        anom_cache = _load_era5_anomaly_matrix_for_som(n_years=30, stride=6)
        dates = anom_cache["dates"]
        anomalies = anom_cache["anomalies"]
        lats = anom_cache["lats"]

        # indices for this cluster
        cluster_idx = r * cols + c
        indices = np.where(bmu == cluster_idx)[0]
        if len(indices) == 0:
            return jsonify({"success": True, "cluster": f"{r},{c}", "analogs": []})

        # centroid anomaly for cluster
        cluster_anoms = anomalies[indices]
        centroid = np.nanmean(cluster_anoms, axis=0)

        # compute composite score for each date in cluster
        if not _HAS_ANALOG_METRICS:
            return jsonify({"success": False, "error": "analog_metrics not available"})

        w2d = _analog_metrics.build_lat_weights(lats, centroid.shape)
        weights_flat = w2d.flatten()

        results = []
        for idx in indices:
            sample = anomalies[idx]
            lat_pearson = _analog_metrics.weighted_pearson(
                centroid.flatten(),
                sample.flatten(),
                weights_flat
            )
            grad_corr = _analog_metrics.gradient_correlation(centroid, sample)
            rmse_sim = _analog_metrics.rmse_similarity(centroid.flatten(), sample.flatten())
            score = _analog_metrics.compute_composite_score(lat_pearson, grad_corr, rmse_sim)
            results.append({
                "date": dates[idx],
                "score": round(float(score), 4),
                "pattern": round(float(lat_pearson), 4),
                "circulation": round(float(grad_corr), 4),
                "amplitude": round(float(rmse_sim), 4)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({
            "success": True,
            "cluster": f"{r},{c}",
            "analogs": results[:max(1, top_n)]
        })

    except Exception as e:
        logger.error(f"Error loading SOM cluster analogs: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/era5/som-cluster-mean-map")
def api_era5_som_cluster_mean_map():
    """
    Render the mean 500mb height anomaly map for a SOM cluster.
    """
    try:
        cluster = request.args.get("cluster", "")
        if not cluster or "," not in cluster:
            return jsonify({"success": False, "error": "cluster is required as r,c"})
        r_str, c_str = cluster.split(",", 1)
        r = int(r_str)
        c = int(c_str)

        som = _load_or_build_era5_som(rows=6, cols=6, n_years=30, stride=6)
        bmu = som["bmu"]
        rows = som["rows"]
        cols = som["cols"]
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return jsonify({"success": False, "error": "cluster out of range"})

        anom_cache = _load_era5_anomaly_matrix_for_som(n_years=30, stride=6)
        anomalies = anom_cache["anomalies"]
        lats = anom_cache["lats"]
        lons = anom_cache["lons"]

        cluster_idx = r * cols + c
        indices = np.where(bmu == cluster_idx)[0]
        if len(indices) == 0:
            return jsonify({"success": False, "error": "No samples in cluster"})

        mean_anom = np.nanmean(anomalies[indices], axis=0) / 9.80665  # to dam

        import warnings
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from io import BytesIO
        import base64

        # Suppress occasional shapely runtime warnings from cartopy rendering
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-100))
        ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        anom_levels = np.arange(-30, 31, 2)
        cf = ax.contourf(lons, lats, mean_anom, levels=anom_levels, cmap='RdBu_r',
                         transform=ccrs.PlateCarree(), extend='both')
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('500mb Height Anomaly (dam)', fontsize=10)

        plt.title(f'SOM Cluster Mean 500mb Anomaly | C{r + 1}-{c + 1}', fontsize=12, fontweight='bold')

        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
        buffer.seek(0)
        img_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({"success": True, "cluster": f"{r},{c}", "map_image": img_base64})

    except Exception as e:
        logger.error(f"Error rendering SOM cluster mean map: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/era5/analog-history')
def api_era5_analog_history():
    """
    Get historical analog forecast predictions for charting.

    Returns:
        JSON with list of historical predictions including dates and values
    """
    try:
        history = load_analog_history()
        predictions = history.get("predictions", [])

        # Sort by prediction date
        predictions_sorted = sorted(predictions, key=lambda x: x["prediction_date"])

        return jsonify({
            "success": True,
            "predictions": predictions_sorted,
            "count": len(predictions_sorted)
        })

    except Exception as e:
        logger.error(f"Error loading analog history: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/era5/latest-analog-result')
def api_era5_latest_analog_result():
    """
    Return the most recently auto-computed analog prediction written by run_analog_sync().
    Used by the Historical Weather UI to pre-populate the analog card on page load
    without requiring a slow live recomputation.
    """
    try:
        if not ANALOG_LATEST_FILE.exists():
            return jsonify({'success': False, 'error': 'No cached analog result available yet'})
        with open(ANALOG_LATEST_FILE) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error loading latest analog result: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/runs')
def api_runs():
    """List all runs for a location with summary info."""
    location_name = request.args.get('location', 'Fairfax, VA')

    db = load_forecasts_db()

    if location_name not in db:
        return jsonify({"success": False, "error": "Location not found"})

    runs = db[location_name].get("runs", {})
    latest_run = db[location_name].get("latest_run")

    run_list = []
    for run_id, run_data in runs.items():
        run_list.append({
            "run_id": run_id,
            "fetched_at": run_data.get("fetched_at"),
            "has_observations": bool(run_data.get("observed")),
            "verification": run_data.get("verification"),
            "is_latest": run_id == latest_run
        })

    # Sort by run_id descending
    run_list.sort(key=lambda x: x["run_id"], reverse=True)

    return jsonify({
        "success": True,
        "location": location_name,
        "runs": run_list,
        "total_runs": len(run_list)
    })


@app.route('/verification')
def verification_page():
    """Model verification page with Fairfax + ASOS tabs."""
    return render_template('model_verification.html')


@app.route('/asos-verification')
def asos_verification_page():
    """ASOS verification dashboard page - shows forecast skill by lead time."""
    return render_template(
        'verification.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA"
    )


@app.route('/rossby')
def rossby_page():
    """Rossby Wave Analysis page - shows wave patterns and predictability."""
    return render_template('rossby.html')


@app.route('/single-run-bias')
def single_run_bias_page():
    """Single run bias map page - shows model bias for a specific forecast run."""
    return render_template('single_run_bias.html')


@app.route('/api/observations')
def api_observations():
    """
    API endpoint to fetch observation data for specific times.
    Only available for Fairfax, VA.
    """
    location_name = request.args.get('location', 'Fairfax, VA')

    if location_name != "Fairfax, VA":
        return jsonify({
            "success": False,
            "error": "Observations only available for Fairfax, VA"
        })

    # Get times from query parameter (comma-separated ISO format)
    times_param = request.args.get('times', '')

    if not times_param:
        return jsonify({
            "success": False,
            "error": "Missing 'times' parameter (comma-separated ISO format datetimes)"
        })

    times = [t.strip() for t in times_param.split(',')]

    try:
        # Fetch missing data first
        weatherlink.fetch_missing_data(silent=True)

        # Get observations
        observations = weatherlink.get_observations_for_forecast_times(times)

        return jsonify({
            "success": True,
            "location": location_name,
            "observations": observations
        })
    except Exception as e:
        logger.error(f"Error fetching observations: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/current-weather')
def api_current_weather():
    """API endpoint for current Davis weather station data."""
    try:
        json_data = fetch_current_weather()
        weather = parse_current_weather(json_data)

        weather["rain_24h_in"] = calculate_rain_24h()

        if local_weather_data:
            try:
                percentiles = local_weather_data.get_soil_moisture_percentiles(weather)
                weather.update(percentiles)
            except Exception as e:
                logger.warning(f"Soil moisture percentile error: {e}")
            try:
                today = datetime.now()
                summaries = local_weather_data.get_daily_summaries(
                    today.replace(hour=0, minute=0, second=0, microsecond=0),
                    today.replace(hour=23, minute=59, second=59, microsecond=0)
                )
                if summaries:
                    weather["temp_high_f"] = summaries[-1].get("temp_high")
                    weather["temp_low_f"] = summaries[-1].get("temp_low")
            except Exception as e:
                logger.warning(f"Daily high/low error: {e}")

        aqi = weather.get("aqi")
        aqi_category = get_aqi_category(int(aqi)) if aqi is not None else None
        aqi_color = get_aqi_color(int(aqi)) if aqi is not None else None

        timestamp = weather.get("timestamp")
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            weather["timestamp_formatted"] = dt.strftime("%I:%M %p")
            weather["timestamp_iso"] = dt.isoformat()
        else:
            weather["timestamp_formatted"] = "--"
            weather["timestamp_iso"] = None

        is_day = is_daylight(datetime.now(), DEFAULT_LAT, DEFAULT_LON)
        today = datetime.now()
        sunrise, sunset = calculate_sunrise_sunset(DEFAULT_LAT, DEFAULT_LON, today)
        almanac = {
            "sunrise": sunrise.strftime("%I:%M %p"),
            "sunset": sunset.strftime("%I:%M %p"),
            "normal_high": None,
            "normal_low": None,
            "solar_historical_avg": None,
            "moonrise": None,
            "moonset": None,
            "moon_phase": None,
        }
        if LocationInfo and sun and moon and ZoneInfo:
            try:
                location = LocationInfo(
                    name="Fairfax, VA",
                    region="US",
                    timezone="America/New_York",
                    latitude=DEFAULT_LAT,
                    longitude=DEFAULT_LON,
                )
                tz = ZoneInfo(location.timezone)
                sun_times = sun(location.observer, date=today.date(), tzinfo=tz)
                almanac["sunrise"] = sun_times["sunrise"].strftime("%I:%M %p")
                almanac["sunset"] = sun_times["sunset"].strftime("%I:%M %p")

                mr = moon.moonrise(location.observer, date=today.date(), tzinfo=tz)
                ms = moon.moonset(location.observer, date=today.date(), tzinfo=tz)
                if mr:
                    almanac["moonrise"] = mr.strftime("%I:%M %p")
                if ms:
                    almanac["moonset"] = ms.strftime("%I:%M %p")

                phase_value = moon.phase(today.date())
                # Map phase to name
                if phase_value is not None:
                    if phase_value < 1.84566:
                        phase_name = "New Moon"
                    elif phase_value < 5.53699:
                        phase_name = "Waxing Crescent"
                    elif phase_value < 9.22831:
                        phase_name = "First Quarter"
                    elif phase_value < 12.91963:
                        phase_name = "Waxing Gibbous"
                    elif phase_value < 16.61096:
                        phase_name = "Full Moon"
                    elif phase_value < 20.30228:
                        phase_name = "Waning Gibbous"
                    elif phase_value < 23.99361:
                        phase_name = "Last Quarter"
                    elif phase_value < 27.68493:
                        phase_name = "Waning Crescent"
                    else:
                        phase_name = "New Moon"
                    almanac["moon_phase"] = phase_name
            except Exception as e:
                logger.warning(f"Astral almanac error: {e}")
        if local_weather_data:
            try:
                climo = local_weather_data.get_climatology_for_date(today)
                if climo:
                    almanac["normal_high"] = climo.get("high_median")
                    almanac["normal_low"] = climo.get("low_median")
                summary = local_weather_data.get_period_summary(
                    today.replace(hour=0, minute=0, second=0, microsecond=0),
                    today.replace(hour=23, minute=59, second=59, microsecond=0)
                )
                if summary:
                    almanac["solar_historical_avg"] = summary.get("solar_rad_historical")
            except Exception as e:
                logger.warning(f"Climatology lookup error: {e}")

        return jsonify({
            "success": True,
            "weather": weather,
            "aqi_category": aqi_category,
            "aqi_color": aqi_color,
            "is_daylight": is_day,
            "almanac": almanac,
            "location": {
                "name": "Fairfax, VA",
                "lat": DEFAULT_LAT,
                "lon": DEFAULT_LON,
            },
        })

    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/historical')
def api_historical():
    """API endpoint for historical data with climatology comparison."""
    if not local_weather_data:
        return jsonify({'error': 'Historical data module not available'}), 500

    start_str = request.args.get('start')
    end_str = request.args.get('end')

    if not start_str or not end_str:
        return jsonify({'error': 'start and end parameters required'}), 400

    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
        end_date = end_date.replace(hour=23, minute=59, second=59)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    data = local_weather_data.get_daily_summaries_with_climo(start_date, end_date)
    summary = local_weather_data.get_period_summary(start_date, end_date)

    return jsonify({
        'daily': data.get('daily', []),
        'climo': data.get('climo', []),
        'anomalies': data.get('anomalies', []),
        'summary': summary
    })


# ============================================================================
# ASOS Station Verification
# ============================================================================

def extract_model_at_stations(model_data_array, stations):
    """
    Extract model values at station locations via bilinear interpolation.

    Args:
        model_data_array: xarray DataArray with lat/lon coordinates
        stations: List of station dicts with 'lat' and 'lon' keys

    Returns:
        List of interpolated values (same order as stations)
    """
    import xarray as xr

    values = []
    lat_name = 'latitude' if 'latitude' in model_data_array.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in model_data_array.coords else 'lon'

    # Get coordinate values
    lats = model_data_array[lat_name].values
    lons = model_data_array[lon_name].values

    # Check longitude convention
    lon_is_0_360 = lons.min() >= 0 and lons.max() > 180

    for station in stations:
        try:
            slat = station['lat']
            slon = station['lon']

            # Convert longitude if needed
            if lon_is_0_360 and slon < 0:
                slon = slon + 360

            # Check bounds
            if slat < lats.min() or slat > lats.max():
                values.append(None)
                continue
            if slon < lons.min() or slon > lons.max():
                values.append(None)
                continue

            # Interpolate
            val = model_data_array.interp(
                {lat_name: slat, lon_name: slon},
                method='linear'
            ).values

            # Handle scalar or array result
            if hasattr(val, 'item'):
                val = val.item()

            if np.isnan(val):
                values.append(None)
            else:
                values.append(float(val))

        except Exception as e:
            logger.warning(f"Interpolation failed for station {station.get('station_id', '?')}: {e}")
            values.append(None)

    return values


def fetch_asos_forecasts_for_model(model_name, forecast_hours, stations, init_hour: Optional[int] = None):
    """
    Fetch forecasts at all ASOS station locations for a model.

    Args:
        model_name: 'gfs', 'aifs', or 'ifs'
        forecast_hours: List of forecast hours
        stations: List of station dicts
        init_hour: Optional specific init hour (0, 6, 12, or 18). If None, gets the latest available.

    Returns:
        Tuple of (init_time, forecast_dict) where forecast_dict maps station_id to forecast data
    """
    import tempfile
    from pathlib import Path

    region = Region("CONUS", CONUS_BOUNDS)

    # Initialize model and get init time
    if model_name.lower() == 'gfs':
        model = GFSModel()
        temp_var = TEMP_2M_GFS
        mslp_var = MSLP_GFS
        precip_var = PRECIP_GFS
    elif model_name.lower() == 'aifs':
        model = AIFSModel()
        temp_var = AIFS_VARIABLES['t2m']
        mslp_var = AIFS_VARIABLES['mslp']
        precip_var = AIFS_VARIABLES['tp']
    elif model_name.lower() == 'ifs':
        model = IFSModel()
        temp_var = IFS_VARIABLES['t2m']
        mslp_var = IFS_VARIABLES['mslp']
        precip_var = IFS_VARIABLES['tp']
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Get init time
    if init_hour is not None:
        init_time = model.get_init_time_for_hour(init_hour)
    else:
        init_time = model.get_latest_init_time()

    # For IFS, limit to available hours (depends on init time)
    if model_name.lower() == 'ifs':
        # 00Z/12Z go to 240h, 06Z/18Z go to 144h
        max_ifs_hour = 240 if init_time.hour in [0, 12] else 144
        model_hours = [h for h in forecast_hours if h <= max_ifs_hour]
    else:
        model_hours = forecast_hours

    # Initialize result dict
    station_forecasts = {
        s['station_id']: {'temps': [], 'mslps': [], 'precips': []}
        for s in stations
    }

    # Fetch each forecast hour
    for i, hour in enumerate(model_hours, 1):
        progress_pct = int((i / len(model_hours)) * 100)
        broadcast_sync_log(f"  [{model_name.upper()}] Extracting F{hour:03d} ({i}/{len(model_hours)}, {progress_pct}%)", 'info')
        logger.info(f"Extracting {model_name.upper()} F{hour:03d} at ASOS stations...")

        # Fetch temp
        try:
            data = model.fetch_data(temp_var, init_time, hour, region)
            temp_vals = extract_model_at_stations(data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} temp F{hour} failed: {e}")
            broadcast_sync_log(f"  [{model_name.upper()}] WARNING: temp F{hour:03d} failed: {e}", 'warning')
            temp_vals = [None] * len(stations)

        # Fetch mslp
        try:
            data = model.fetch_data(mslp_var, init_time, hour, region)
            mslp_vals = extract_model_at_stations(data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} mslp F{hour} failed: {e}")
            broadcast_sync_log(f"  [{model_name.upper()}] WARNING: mslp F{hour:03d} failed: {e}", 'warning')
            mslp_vals = [None] * len(stations)

        # Fetch precip
        try:
            data = model.fetch_data(precip_var, init_time, hour, region)
            precip_vals = extract_model_at_stations(data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} precip F{hour} failed: {e}")
            broadcast_sync_log(f"  [{model_name.upper()}] WARNING: precip F{hour:03d} failed: {e}", 'warning')
            precip_vals = [None] * len(stations)

        # Store values
        for i, station in enumerate(stations):
            sid = station['station_id']
            station_forecasts[sid]['temps'].append(temp_vals[i])
            station_forecasts[sid]['mslps'].append(mslp_vals[i])
            station_forecasts[sid]['precips'].append(precip_vals[i])

    # Pad IFS with None for hours beyond its range
    if model_name.lower() == 'ifs':
        pad_count = len(forecast_hours) - len(model_hours)
        for sid in station_forecasts:
            station_forecasts[sid]['temps'].extend([None] * pad_count)
            station_forecasts[sid]['mslps'].extend([None] * pad_count)
            station_forecasts[sid]['precips'].extend([None] * pad_count)

    # Convert AIFS/IFS cumulative precipitation to interval totals (6-hour accumulation)
    if model_name.lower() in ['aifs', 'ifs']:
        for sid in station_forecasts:
            cumulative_precips = station_forecasts[sid]['precips']
            interval_precips = []

            for i, cum in enumerate(cumulative_precips):
                if cum is None:
                    interval_precips.append(None)
                elif i == 0:
                    # First interval is just the cumulative value (0 to first hour)
                    interval_precips.append(cum)
                else:
                    prev_cum = cumulative_precips[i - 1]
                    if prev_cum is not None:
                        interval = cum - prev_cum
                        # Ensure non-negative (rounding errors can cause tiny negatives)
                        interval_precips.append(max(0.0, interval))
                    else:
                        interval_precips.append(None)

            station_forecasts[sid]['precips'] = interval_precips

    return init_time, station_forecasts


@app.route('/api/asos/stations')
def api_asos_stations():
    """Get all ASOS stations."""
    try:
        stations = asos.fetch_all_stations()
        return jsonify({
            "success": True,
            "count": len(stations),
            "stations": [
                {
                    "station_id": s.station_id,
                    "name": s.name,
                    "lat": s.lat,
                    "lon": s.lon,
                    "state": s.state
                }
                for s in stations
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching ASOS stations: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/asos/station-forecast')
def api_asos_station_forecast():
    """Get forecast data for a specific ASOS station (all models)."""
    station_id = request.args.get('station_id')
    show_trends = request.args.get('trends', 'false').lower() == 'true'
    use_trends_db = request.args.get('use_trends_db', 'false').lower() == 'true'

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id parameter"}), 400

    try:
        # Use trends database if requested (for 6-hourly data), otherwise verification database
        if use_trends_db:
            db = load_asos_trends_db()
            logger.info(f"Using trends database for station {station_id}")
        else:
            db = asos.load_asos_forecasts_db()

        runs = db.get("runs", {})

        if not runs:
            return jsonify({"success": False, "error": "No forecast data available"}), 404

        # Get the latest run
        sorted_runs = sorted(runs.keys(), reverse=True)
        latest_run_key = sorted_runs[0]
        run_data = runs[latest_run_key]
        forecast_hours = run_data.get("forecast_hours", [])

        # Initialize time series
        init_time = datetime.fromisoformat(latest_run_key)
        times = [(init_time + timedelta(hours=h)).isoformat() for h in forecast_hours]

        # Extract forecast data for each model
        result = {
            "success": True,
            "station_id": station_id,
            "init_time": latest_run_key,
            "times": times,
            "show_trends": show_trends,
            "gfs": {"temps": [], "precips": []},
            "aifs": {"temps": [], "precips": []},
            "ifs": {"temps": [], "precips": []},
            "nws": {"temps": [], "precips": []}
        }

        if show_trends and len(sorted_runs) >= 2:
            # Calculate trends by comparing latest vs previous run
            previous_run_key = sorted_runs[1]
            previous_run_data = runs[previous_run_key]
            previous_init_time = datetime.fromisoformat(previous_run_key)

            result["previous_init_time"] = previous_run_key

            for model in ['gfs', 'aifs', 'ifs', 'nws']:
                latest_model_data = run_data.get(model, {})
                latest_station_data = latest_model_data.get(station_id, {})
                latest_temps = latest_station_data.get("temps", [])
                latest_precips = latest_station_data.get("precips", [])

                previous_model_data = previous_run_data.get(model, {})
                previous_station_data = previous_model_data.get(station_id, {})
                previous_hours = previous_run_data.get("forecast_hours", [])

                # Build lookup of previous forecast by valid time
                previous_by_valid_time = {}
                for i, prev_hour in enumerate(previous_hours):
                    valid_time = previous_init_time + timedelta(hours=prev_hour)
                    previous_temps = previous_station_data.get("temps", [])
                    previous_precips = previous_station_data.get("precips", [])

                    if i < len(previous_temps):
                        previous_by_valid_time[valid_time.isoformat()] = {
                            "temp": previous_temps[i],
                            "precip": previous_precips[i] if i < len(previous_precips) else None
                        }

                # Calculate trends for each forecast hour
                temp_trends = []
                precip_trends = []

                for i, hour in enumerate(forecast_hours):
                    valid_time = (init_time + timedelta(hours=hour)).isoformat()
                    latest_temp = latest_temps[i] if i < len(latest_temps) else None
                    latest_precip = latest_precips[i] if i < len(latest_precips) else None

                    # Find matching forecast from previous run
                    previous_forecast = previous_by_valid_time.get(valid_time)

                    if previous_forecast and latest_temp is not None and previous_forecast["temp"] is not None:
                        temp_trends.append(latest_temp - previous_forecast["temp"])
                    else:
                        temp_trends.append(None)

                    if previous_forecast and latest_precip is not None and previous_forecast["precip"] is not None:
                        precip_trends.append(latest_precip - previous_forecast["precip"])
                    else:
                        precip_trends.append(None)

                result[model]["temps"] = temp_trends
                result[model]["precips"] = precip_trends

        else:
            # Return absolute values (not trends)
            for model in ['gfs', 'aifs', 'ifs', 'nws']:
                model_data = run_data.get(model, {})
                station_data = model_data.get(station_id, {})

                result[model]["temps"] = station_data.get("temps", [None] * len(forecast_hours))
                result[model]["precips"] = station_data.get("precips", [None] * len(forecast_hours))

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting station forecast for {station_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/forecast-summary')
def api_asos_forecast_summary():
    """
    Get forecast summary for all ASOS stations at a specific lead time.
    Used for coloring map markers.

    Query params:
        variable: temp or precip
        model: gfs, aifs, ifs, or nws
        lead_time: forecast lead time in hours (default 24)
        trends: 'true' to show change from previous run (default 'false')
    """
    variable = request.args.get('variable', 'temp')
    model = request.args.get('model', 'gfs')
    lead_time = int(request.args.get('lead_time', 24))
    show_trends = request.args.get('trends', 'false').lower() == 'true'

    try:
        # Always use the regular forecast database for dynamic trend calculation
        db = asos.load_asos_forecasts_db()
        runs = db.get("runs", {})
        stations = db.get("stations", {})

        if not runs:
            return jsonify({"success": False, "error": "No forecast data available"}), 404

        # Get the latest run
        sorted_runs = sorted(runs.keys())
        latest_run_key = sorted_runs[-1]
        run_data = runs[latest_run_key]
        forecast_hours = run_data.get("forecast_hours", [])

        # Find the index for the requested lead time (snap to nearest if exact not available)
        if lead_time not in forecast_hours:
            if not forecast_hours:
                return jsonify({"success": False, "error": "No forecast hours available"}), 400
            lead_time = min(forecast_hours, key=lambda h: abs(h - lead_time))

        lead_time_index = forecast_hours.index(lead_time)

        # Get model data
        model_data = run_data.get(model, {})

        # If trends mode, get previous run and match by valid time
        prev_model_data = {}
        prev_lead_time_index = None
        if show_trends and len(sorted_runs) >= 2:
            prev_run_key = sorted_runs[-2]
            prev_run_data = runs[prev_run_key]
            prev_model_data = prev_run_data.get(model, {})
            prev_forecast_hours = prev_run_data.get("forecast_hours", [])

            latest_init = datetime.fromisoformat(latest_run_key)
            prev_init = datetime.fromisoformat(prev_run_key)
            time_diff_hours = int((latest_init - prev_init).total_seconds() / 3600)

            # Calculate valid time for the latest forecast
            valid_time_target = latest_init + timedelta(hours=lead_time)

            # Find the forecast hour in previous run that matches this valid time
            # prev_init + prev_lead_time = valid_time_target
            # prev_lead_time = valid_time_target - prev_init
            prev_lead_time_needed = int((valid_time_target - prev_init).total_seconds() / 3600)

            # Find this in the previous run's forecast hours
            if prev_lead_time_needed in prev_forecast_hours:
                prev_lead_time_index = prev_forecast_hours.index(prev_lead_time_needed)
                logger.info(
                    f"Trends mode: comparing {model.upper()} for valid time {valid_time_target.strftime('%m/%d %HZ')} | "
                    f"Latest: {latest_init.strftime('%m/%d %HZ')} F{lead_time:03d} | "
                    f"Previous: {prev_init.strftime('%m/%d %HZ')} F{prev_lead_time_needed:03d}"
                )
            else:
                logger.warning(
                    f"Trends mode: F{prev_lead_time_needed:03d} not available in previous run (has {prev_forecast_hours}), "
                    f"cannot calculate trends for {model.upper()} at lead time {lead_time}h"
                )

        # Extract forecast value for each station
        station_forecasts = []
        debug_count = 0
        trends_calculated = 0

        for station_id, station_info in stations.items():
            station_data = model_data.get(station_id, {})

            if variable == 'temp':
                temps = station_data.get('temps', [])
                latest_value = temps[lead_time_index] if lead_time_index < len(temps) else None
            elif variable == 'precip':
                precips = station_data.get('precips', [])
                latest_value = precips[lead_time_index] if lead_time_index < len(precips) else None
            else:
                latest_value = None

            # Calculate trend if requested (compare same VALID TIME from previous run)
            value = latest_value  # Default to absolute value

            if show_trends and prev_model_data and latest_value is not None and prev_lead_time_index is not None:
                prev_station_data = prev_model_data.get(station_id, {})

                # Compare forecasts for the SAME valid time (different lead times)
                if variable == 'temp':
                    prev_temps = prev_station_data.get('temps', [])
                    prev_value = prev_temps[prev_lead_time_index] if prev_lead_time_index < len(prev_temps) else None
                elif variable == 'precip':
                    prev_precips = prev_station_data.get('precips', [])
                    prev_value = prev_precips[prev_lead_time_index] if prev_lead_time_index < len(prev_precips) else None
                else:
                    prev_value = None

                # Calculate delta if both values exist
                if prev_value is not None:
                    value = latest_value - prev_value
                    trends_calculated += 1

                    # Debug logging for first few stations
                    if debug_count < 3:
                        logger.info(
                            f"Trend: {station_id} = {value:+.1f}{'°F' if variable == 'temp' else ' in'} | "
                            f"Latest: {latest_value:.1f} | Prev: {prev_value:.1f}"
                        )
                        debug_count += 1

            if value is not None:
                station_forecasts.append({
                    "station_id": station_id,
                    "lat": station_info.get('lat'),
                    "lon": station_info.get('lon'),
                    "name": station_info.get('name'),
                    "value": value
                })

        # Trends are available if we calculated any
        trends_available = show_trends and trends_calculated > 0

        if show_trends and not trends_available:
            logger.warning(f"No trends calculated for {model.upper()} at {lead_time}h")

        return jsonify({
            "success": True,
            "variable": variable,
            "model": model,
            "lead_time": lead_time,
            "init_time": latest_run_key,
            "trends": show_trends,
            "trends_available": trends_available,
            "count": len(station_forecasts),
            "stations": station_forecasts
        })

    except Exception as e:
        logger.error(f"Error getting ASOS forecast summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/verification-map')
def api_asos_verification_map():
    """
    Get verification data for the ASOS station map.

    Query params:
        variable: temp, mslp, or precip
        metric: mae or bias
        model: gfs, aifs, or ifs
        lead_time: forecast lead time in hours
    """
    variable = request.args.get('variable', 'temp')
    metric = request.args.get('metric', 'mae')
    model = request.args.get('model', 'gfs')
    lead_time = int(request.args.get('lead_time', 24))
    period = request.args.get('period', 'all')
    if period == 'monthly' and get_asos_data_span_days() < 30:
        period = 'all'

    # Check cache
    cache_key = f"{variable}_{metric}_{model}_{lead_time}_{period}"
    cache_ttl = 3600  # 1 hour

    if cache_key in _verification_cache:
        cache_time = _verification_cache_time.get(cache_key, 0)
        if time.time() - cache_time < cache_ttl:
            return jsonify(_verification_cache[cache_key])

    try:
        # Get verification data (all-time cache or recent window)
        if period == 'monthly':
            verification = asos.get_verification_data_from_monthly_cache(model, variable, lead_time)
        else:
            verification = asos.get_verification_data_from_cache(model, variable, lead_time)

        if not verification:
            return jsonify({
                "success": True,
                "message": "No verification data available yet. Sync forecasts and wait for valid times to pass.",
                "stations": [],
                "variable": variable,
                "metric": metric,
                "model": model,
                "lead_time": lead_time
            })

        # Format for map display
        stations = []
        values = []

        for station_id, data in verification.items():
            value = data.get(metric)
            if value is not None:
                stations.append({
                    "station_id": station_id,
                    "name": data.get('name', station_id),
                    "lat": data.get('lat'),
                    "lon": data.get('lon'),
                    "state": data.get('state', ''),
                    "value": value,
                    "count": data.get('count', 0)
                })
                values.append(value)

        # Calculate range for color scale
        if values:
            if metric == 'bias':
                # Symmetric scale for bias
                max_abs = max(abs(min(values)), abs(max(values)), 0.1)
                min_val = -max_abs
                max_val = max_abs
            else:
                # MAE: 0 to max
                min_val = 0
                max_val = max(values) if values else 1
        else:
            min_val = 0
            max_val = 1

        result = {
            "success": True,
            "stations": stations,
            "variable": variable,
            "metric": metric,
            "model": model,
            "lead_time": lead_time,
            "min_value": round(min_val, 2),
            "max_value": round(max_val, 2),
            "station_count": len(stations)
        }

        # Cache result
        _verification_cache[cache_key] = result
        _verification_cache_time[cache_key] = time.time()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting ASOS verification: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/run-forecast")
def api_run_forecast():
    """API endpoint for NWS forecast data with run scores."""
    hours_ahead = request.args.get("hours", 72, type=int)
    lat = request.args.get("lat", DEFAULT_LAT, type=float)
    lon = request.args.get("lon", DEFAULT_LON, type=float)

    try:
        grid_id, grid_x, grid_y, forecast_url = get_grid_point(lat, lon)
        forecast = fetch_hourly_forecast(forecast_url)
        gusts_by_hour = fetch_wind_gusts(grid_id, grid_x, grid_y)
        aqi_by_date = fetch_aqi_forecast(lat, lon)

        now = datetime.now().astimezone()
        cutoff = now + timedelta(hours=hours_ahead)

        result = []
        for hour in forecast:
            hour_time = datetime.fromisoformat(hour["datetime"])
            if hour_time > cutoff:
                break

            aqi = None
            if aqi_by_date:
                date_str = hour_time.strftime("%Y-%m-%d")
                aqi = aqi_by_date.get(date_str)

            score, reasons = rate_running_conditions(hour, aqi=aqi, lat=lat, lon=lon)

            hour_utc = hour_time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hour_key = hour_utc.isoformat()
            gust = gusts_by_hour.get(hour_key) if gusts_by_hour else None

            humidity = hour.get("humidity") or 50
            dew_point = nws_calculate_dew_point(hour["temperature"], humidity)

            daylight = nws_is_daylight(hour_time.replace(tzinfo=None), lat, lon)

            result.append({
                "datetime": hour["datetime"],
                "datetime_local": hour["datetime_local"],
                "temperature": hour["temperature"],
                "wind_speed": hour["wind_speed_mph"],
                "wind_gust": gust or hour["wind_speed_mph"],
                "wind_direction": hour["wind_direction"],
                "precipitation_chance": hour["precipitation_chance"],
                "humidity": humidity,
                "dew_point": dew_point,
                "short_forecast": hour["short_forecast"],
                "aqi": aqi,
                "score": score,
                "reasons": reasons,
                "is_daylight": daylight
            })

        best_times = sorted(result, key=lambda x: x["score"], reverse=True)[:10]

        return jsonify({
            "success": True,
            "location": {"lat": lat, "lon": lon, "grid": grid_id},
            "fetched_at": datetime.now().isoformat(),
            "hours_ahead": hours_ahead,
            "forecast": result,
            "best_times": best_times,
            "ideal_temp_range": IDEAL_TEMP_RANGE
        })

    except Exception as e:
        logger.error(f"Run forecast failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nws-forecast-cache")
def api_nws_forecast_cache():
    """Return cached NWS forecast data (refreshed during sync)."""
    data = load_nws_cache()
    if not data:
        return jsonify({"success": False, "error": "No NWS cache available"}), 200
    runs = data.get("runs", [])
    if not runs:
        return jsonify({"success": False, "error": "No NWS cache available"}), 200
    return jsonify(runs[-1])


@app.route('/api/asos/station-verification')
def api_asos_station_verification():
    """Get detailed verification for a single station across all lead times."""
    station_id = request.args.get('station_id')
    model = request.args.get('model') # Required model filter
    period = request.args.get('period', 'all')
    if period == 'monthly' and get_asos_data_span_days() < 30:
        period = 'all'

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id parameter"}), 400
    if not model:
        return jsonify({"success": False, "error": "Missing model parameter"}), 400

    try:
        if period == 'monthly':
            # get_station_detail_monthly returns a flat structure with pre-computed arrays
            raw_result = asos.get_station_detail_monthly(station_id, model, days_back=30)
            if "error" in raw_result:
                return jsonify({"success": False, "error": raw_result["error"]}), 404

            verification_data = {
                "lead_times": raw_result["lead_times"],
                "temp_mae": raw_result["temp_mae"],
                "temp_bias": raw_result["temp_bias"],
                "mslp_mae": raw_result["mslp_mae"],
                "mslp_bias": raw_result["mslp_bias"],
                "precip_mae": raw_result["precip_mae"],
                "precip_bias": raw_result["precip_bias"],
            }
        else:
            # Use the precomputed verification cache which uses composite observation matching.
            # This correctly finds temperature (from :56 METARs) even when MSLP-only
            # reports at :00 are the single-nearest observation.
            raw_result = asos.get_station_detail_from_cache(station_id, model)
            if "error" in raw_result:
                return jsonify({"success": False, "error": raw_result["error"]}), 404

            verification_data = {
                "lead_times": raw_result["lead_times"],
                "temp_mae": raw_result["temp_mae"],
                "temp_bias": raw_result["temp_bias"],
                "temp_count": raw_result.get("temp_count", []),
                "precip_mae": raw_result["precip_mae"],
                "precip_bias": raw_result["precip_bias"],
                "precip_count": raw_result.get("precip_count", []),
            }

        return jsonify({
            "success": True,
            "station_id": station_id,
            "station_name": raw_result["station"]["name"],
            "verification": verification_data,
            "period": period
        })

    except Exception as e:
        logger.error(f"Error getting station verification for {station_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/station-observations')
def api_asos_station_observations():
    """Return saved ASOS observations (temp/mslp) for a station."""
    station_id = request.args.get('station_id', '').upper()
    period = request.args.get('period', 'all').lower()
    if period == 'monthly' and get_asos_data_span_days() < 30:
        period = 'all'

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id"}), 400

    try:
        db = asos.load_asos_forecasts_db()
        station_obs = db.get("observations", {}).get(station_id, {})
        if not station_obs:
            return jsonify({"success": False, "error": "No observations found for this station"}), 404

        now = datetime.now(timezone.utc)
        cutoff = None
        if period == "monthly":
            cutoff = now - timedelta(days=30)

        times = []
        temps = []
        mslps = []
        for time_str, obs in station_obs.items():
            try:
                dt = datetime.fromisoformat(time_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if cutoff and dt < cutoff:
                continue

            times.append(dt.astimezone(timezone.utc).isoformat())
            temps.append(obs.get("temp"))
            mslps.append(obs.get("mslp"))

        zipped = sorted(zip(times, temps, mslps), key=lambda x: x[0])
        if not zipped:
            return jsonify({"success": False, "error": "No observations in selected period"}), 404

        times, temps, mslps = zip(*zipped)
        return jsonify({
            "success": True,
            "station_id": station_id,
            "period": period,
            "times": list(times),
            "temps": list(temps),
            "mslps": list(mslps)
        })
    except Exception as e:
        logger.error(f"Error fetching station observations: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/station-obs-timeseries')
def api_asos_station_obs_timeseries():
    """Return per-time observations and model bias for a station/lead time."""
    station_id = request.args.get('station_id', '').upper()
    model = request.args.get('model', 'gfs').lower()
    lead_time = int(request.args.get('lead_time', 24))
    period = request.args.get('period', 'all').lower()
    if period == 'monthly' and get_asos_data_span_days() < 30:
        period = 'all'

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id"}), 400

    try:
        db = asos.load_asos_forecasts_db()
        runs = db.get("runs", {})
        if not runs:
            return jsonify({"success": False, "error": "No ASOS forecast runs available"}), 404

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=30) if period == "monthly" else None

        times = []
        obs_temps = []
        obs_precips = []
        temp_bias = []
        precip_bias = []

        for run_key, run_data in runs.items():
            try:
                init_time = datetime.fromisoformat(run_key)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            forecast_hours = run_data.get("forecast_hours", [])
            if lead_time not in forecast_hours:
                continue
            idx = forecast_hours.index(lead_time)

            model_data = run_data.get(model, {})
            fcst = model_data.get(station_id)
            if not fcst:
                continue

            valid_time = init_time + timedelta(hours=lead_time)
            if valid_time >= now:
                continue
            if cutoff and valid_time < cutoff:
                continue

            # ASOS temps often report around :52; use wider window to match synoptic times.
            # Use a wider window for temperature (ASOS often reports at :52)
            station_obs = db.get("observations", {}).get(station_id, {})

            def _nearest_var(var_key, max_delta_minutes=90):
                best_val = None
                best_delta = timedelta(minutes=max_delta_minutes + 1)
                for obs_time_str, obs_data in station_obs.items():
                    try:
                        obs_time = datetime.fromisoformat(obs_time_str)
                    except Exception:
                        continue
                    delta = abs(obs_time - valid_time)
                    if delta > timedelta(minutes=max_delta_minutes):
                        continue
                    val = obs_data.get(var_key)
                    if val is not None and delta < best_delta:
                        best_delta = delta
                        best_val = val
                return best_val

            obs_temp = _nearest_var('temp', max_delta_minutes=90)
            # For precip, use 6-hr accumulation logic (synoptic times)
            obs = asos.get_stored_observation(db, station_id, valid_time, max_delta_minutes=70)
            obs_precip = obs.get("precip_6hr") if obs else None
            if not obs:
                continue

            fcst_temps = fcst.get("temps", [])
            fcst_precips = fcst.get("precips", [])
            fcst_temp = fcst_temps[idx] if idx < len(fcst_temps) else None
            fcst_precip = fcst_precips[idx] if idx < len(fcst_precips) else None


            times.append(valid_time.astimezone(timezone.utc).isoformat())
            obs_temps.append(obs_temp)
            obs_precips.append(obs_precip)
            temp_bias.append(
                (fcst_temp - obs_temp) if (fcst_temp is not None and obs_temp is not None) else None
            )
            precip_bias.append(
                (fcst_precip - obs_precip) if (fcst_precip is not None and obs_precip is not None) else None
            )

        if not times:
            return jsonify({"success": False, "error": "No observations available for this station/lead time"}), 404

        # Sort by time
        zipped = sorted(zip(times, obs_temps, obs_precips, temp_bias, precip_bias), key=lambda x: x[0])
        times, obs_temps, obs_precips, temp_bias, precip_bias = zip(*zipped)

        return jsonify({
            "success": True,
            "station_id": station_id,
            "model": model,
            "lead_time": lead_time,
            "period": period,
            "times": list(times),
            "obs_temps": list(obs_temps),
            "obs_precips": list(obs_precips),
            "temp_bias": list(temp_bias),
            "precip_bias": list(precip_bias)
        })
    except Exception as e:
        logger.error(f"Error building station obs time series: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/nws/qpf-debug')
def api_nws_qpf_debug():
    """
    Fetch raw NWS gridpoint QPF periods around a target time for a station.
    Query params:
      - station_id (ASOS station id)
      - target (ISO datetime, e.g. 2026-02-11T12:00:00Z)
      - hours (window radius, default 24)
    """
    station_id = request.args.get('station_id', '').upper()
    target_str = request.args.get('target')
    hours = int(request.args.get('hours', 24))

    if not station_id or not target_str:
        return jsonify({"success": False, "error": "Missing station_id or target"}), 400

    try:
        target_time = datetime.fromisoformat(target_str.replace("Z", "+00:00"))
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
    except Exception:
        return jsonify({"success": False, "error": "Invalid target datetime"}), 400

    stations = asos.get_stations_dict()
    st = stations.get(station_id)
    if not st:
        return jsonify({"success": False, "error": "Station not found"}), 404

    lat = st.get("lat")
    lon = st.get("lon")
    if lat is None or lon is None:
        return jsonify({"success": False, "error": "Station missing lat/lon"}), 400

    # Get grid point
    try:
        url = f"{NWS_API}/points/{lat:.4f},{lon:.4f}"
        resp = requests.get(url, headers=NWS_HEADERS, timeout=10)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        grid_id = props.get("gridId")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        if not grid_id:
            return jsonify({"success": False, "error": "No grid info returned"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Grid lookup failed: {e}"}), 500

    # Fetch gridpoint QPF values
    try:
        grid_url = f"{NWS_API}/gridpoints/{grid_id}/{grid_x},{grid_y}"
        resp = requests.get(grid_url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        qpf_values = props.get("quantitativePrecipitation", {}).get("values", [])
    except Exception as e:
        return jsonify({"success": False, "error": f"Grid QPF fetch failed: {e}"}), 500

    window_start = target_time - timedelta(hours=hours)
    window_end = target_time + timedelta(hours=hours)
    filtered = []

    for entry in qpf_values:
        valid_time = entry.get("validTime", "")
        if "/" in valid_time:
            start_str, dur_str = valid_time.split("/")
        else:
            start_str, dur_str = valid_time, "PT1H"
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except Exception:
            continue

        # Compute end from duration if possible
        end_dt = None
        try:
            match = re.search(r'PT(\\d+)H', dur_str)
            if match:
                end_dt = start_dt + timedelta(hours=int(match.group(1)))
        except Exception:
            end_dt = None

        if start_dt > window_end or (end_dt and end_dt < window_start):
            continue

        filtered.append({
            "validTime": valid_time,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat() if end_dt else None,
            "value_mm": entry.get("value")
        })

    return jsonify({
        "success": True,
        "station_id": station_id,
        "target": target_time.isoformat(),
        "window_hours": hours,
        "grid": {"id": grid_id, "x": grid_x, "y": grid_y},
        "qpf_values": filtered
    })


@app.route('/api/asos/verification-time-series')
def api_asos_verification_time_series():
    """Get time series of verification metrics for plotting trends over time."""
    variable = request.args.get('variable', 'temp')
    lead_time = int(request.args.get('lead_time', 24))
    days_back = int(request.args.get('days_back', 30))

    try:
        gfs_data = asos.get_verification_time_series_from_cache('gfs', variable, lead_time, days_back)
        aifs_data = asos.get_verification_time_series_from_cache('aifs', variable, lead_time, days_back)
        ifs_data = asos.get_verification_time_series_from_cache('ifs', variable, lead_time, days_back)
        nws_data = asos.get_verification_time_series_from_cache('nws', variable, lead_time, days_back)

        # Check for errors
        if "error" in gfs_data:
            return jsonify({"success": False, "error": gfs_data["error"]}), 404

        # Calculate daily winners based on MAE
        winner_counts = {"GFS": 0, "AIFS": 0, "IFS": 0, "NWS": 0, "Tie": 0}

        for i in range(len(gfs_data["mae"])):
            # Get MAE values for this date
            maes = []
            if gfs_data["mae"][i] is not None:
                maes.append(("GFS", gfs_data["mae"][i]))
            if aifs_data["mae"][i] is not None:
                maes.append(("AIFS", aifs_data["mae"][i]))
            if ifs_data["mae"][i] is not None:
                maes.append(("IFS", ifs_data["mae"][i]))
            if nws_data["mae"][i] is not None:
                maes.append(("NWS", nws_data["mae"][i]))

            # Find winner (lowest MAE)
            if maes:
                min_mae = min(m[1] for m in maes)
                winners = [m[0] for m in maes if m[1] == min_mae]
                if len(winners) == 1:
                    winner_counts[winners[0]] += 1
                else:
                    winner_counts["Tie"] += 1

        # Combine data (use GFS dates as the baseline)
        return jsonify({
            "success": True,
            "dates": gfs_data["dates"],
            "gfs": {
                "mae": gfs_data["mae"],
                "bias": gfs_data["bias"],
                "counts": gfs_data["counts"]
            },
            "aifs": {
                "mae": aifs_data["mae"],
                "bias": aifs_data["bias"],
                "counts": aifs_data["counts"]
            },
            "ifs": {
                "mae": ifs_data["mae"],
                "bias": ifs_data["bias"],
                "counts": ifs_data["counts"]
            },
            "nws": {
                "mae": nws_data["mae"],
                "bias": nws_data["bias"],
                "counts": nws_data["counts"]
            },
            "winner_counts": winner_counts,
            "variable": variable,
            "lead_time": lead_time,
            "days_back": days_back
        })

    except Exception as e:
        logger.error(f"Error getting ASOS verification time series: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/mean-verification')
def api_asos_mean_verification():
    """Get mean verification (MAE and Bias) across all stations by lead time for all models."""
    # The 'model' parameter is no longer taken here, as we fetch for all models.
    # Location is not directly used for mean across all stations.
    period = request.args.get('period', 'all').lower()
    if period == 'monthly' and get_asos_data_span_days() < 30:
        period = 'all'

    try:
        if period == 'monthly':
            gfs_results = asos.get_mean_verification_from_monthly_cache('gfs')
            aifs_results = asos.get_mean_verification_from_monthly_cache('aifs')
            ifs_results = asos.get_mean_verification_from_monthly_cache('ifs')
            nws_results = asos.get_mean_verification_from_monthly_cache('nws')
        else:
            gfs_results = asos.get_mean_verification_from_cache('gfs')
            aifs_results = asos.get_mean_verification_from_cache('aifs')
            ifs_results = asos.get_mean_verification_from_cache('ifs')
            nws_results = asos.get_mean_verification_from_cache('nws')

        # Check for errors from asos functions
        if "error" in gfs_results:
            return jsonify({"success": False, "error": gfs_results["error"]}), 404
        if "error" in aifs_results:
            return jsonify({"success": False, "error": aifs_results["error"]}), 404
        if "error" in ifs_results:
            return jsonify({"success": False, "error": ifs_results["error"]}), 404

        # Combine results and filter to only lead times with data
        # Assuming lead_times are the same for all models
        all_lead_times = gfs_results["lead_times"]

        # Filter to only include lead times where at least one model has non-null temp data
        filtered_indices = []
        for i, lt in enumerate(all_lead_times):
            has_data = (
                (gfs_results["temp_mae"][i] is not None) or
                (aifs_results["temp_mae"][i] is not None) or
                (ifs_results["temp_mae"][i] is not None) or
                (nws_results["temp_mae"][i] is not None)
            )
            if has_data:
                filtered_indices.append(i)

        # Helper to filter array by indices
        def filter_array(arr, indices):
            return [arr[i] for i in indices if i < len(arr)]

        lead_times = filter_array(all_lead_times, filtered_indices)

        combined_verification = {
            "lead_times": lead_times,
            "gfs_temp_mae": filter_array(gfs_results["temp_mae"], filtered_indices),
            "gfs_temp_bias": filter_array(gfs_results["temp_bias"], filtered_indices),
            "aifs_temp_mae": filter_array(aifs_results["temp_mae"], filtered_indices),
            "aifs_temp_bias": filter_array(aifs_results["temp_bias"], filtered_indices),
            "ifs_temp_mae": filter_array(ifs_results["temp_mae"], filtered_indices),
            "ifs_temp_bias": filter_array(ifs_results["temp_bias"], filtered_indices),
            "nws_temp_mae": filter_array(nws_results["temp_mae"], filtered_indices),
            "nws_temp_bias": filter_array(nws_results["temp_bias"], filtered_indices),
            "gfs_mslp_mae": filter_array(gfs_results["mslp_mae"], filtered_indices),
            "gfs_mslp_bias": filter_array(gfs_results["mslp_bias"], filtered_indices),
            "aifs_mslp_mae": filter_array(aifs_results["mslp_mae"], filtered_indices),
            "aifs_mslp_bias": filter_array(aifs_results["mslp_bias"], filtered_indices),
            "ifs_mslp_mae": filter_array(ifs_results["mslp_mae"], filtered_indices),
            "ifs_mslp_bias": filter_array(ifs_results["mslp_bias"], filtered_indices),
            "nws_mslp_mae": filter_array(nws_results["mslp_mae"], filtered_indices),
            "nws_mslp_bias": filter_array(nws_results["mslp_bias"], filtered_indices),
            "gfs_precip_mae": filter_array(gfs_results["precip_mae"], filtered_indices),
            "gfs_precip_bias": filter_array(gfs_results["precip_bias"], filtered_indices),
            "aifs_precip_mae": filter_array(aifs_results["precip_mae"], filtered_indices),
            "aifs_precip_bias": filter_array(aifs_results["precip_bias"], filtered_indices),
            "ifs_precip_mae": filter_array(ifs_results["precip_mae"], filtered_indices),
            "ifs_precip_bias": filter_array(ifs_results["precip_bias"], filtered_indices),
            "nws_precip_mae": filter_array(nws_results["precip_mae"], filtered_indices),
            "nws_precip_bias": filter_array(nws_results["precip_bias"], filtered_indices),
        }

        # Get cache timestamp
        window_start = None
        window_end = None
        if period == 'monthly':
            db = asos.load_asos_forecasts_db()
            cache_timestamp = db.get("cumulative_stats", {}).get("monthly_generated_at")
            if cache_timestamp:
                try:
                    end_dt = datetime.fromisoformat(cache_timestamp)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    start_dt = end_dt - timedelta(days=30)
                    window_start = start_dt.isoformat()
                    window_end = end_dt.isoformat()
                except Exception:
                    window_start = None
                    window_end = None
        else:
            cache = asos.load_verification_cache()
            cache_timestamp = cache.get("last_updated") if cache else None

        return jsonify({
            "success": True,
            "verification": combined_verification,
            "cache_timestamp": cache_timestamp,
            "period": period,
            "window_start": window_start,
            "window_end": window_end
        })

    except Exception as e:
        logger.error(f"Error getting mean ASOS verification: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def check_asos_model_exists(init_time, model_name):
    """Check if we already have ASOS forecasts for this model and init time."""
    db = asos.load_asos_forecasts_db()
    run_key = init_time.isoformat()
    runs = db.get("runs", {})

    if run_key not in runs:
        return False

    run_data = runs[run_key]
    model_data = run_data.get(model_name.lower())

    # Check if we have data for this model with a reasonable number of stations
    if model_data and len(model_data) > 100:
        return True

    return False


@app.route('/api/asos/sync')
def api_asos_sync():
    """
    Sync ASOS forecasts for the current model run.

    This extracts forecasts at all ASOS station locations and stores them
    in asos_forecasts.json for later verification. Skips models that have
    already been synced for the current init time.

    Query params:
        force: If 'true', force refresh even if data exists
        init_hour: Optional init hour (0, 6, 12, or 18). If not specified, uses latest available.
    """
    force = request.args.get('force', 'false').lower() == 'true'
    init_hour_param = request.args.get('init_hour')

    # Parse init_hour if provided
    init_hour = None
    if init_hour_param:
        try:
            init_hour = int(init_hour_param)
            if init_hour not in [0, 6, 12, 18]:
                return jsonify({
                    'success': False,
                    'error': f'Invalid init_hour {init_hour}. Must be 0, 6, 12, or 18.'
                })
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid init_hour parameter: {init_hour_param}'
            })

    try:
        # Get stations
        stations_list = asos.fetch_all_stations()
        stations = [
            {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
            for s in stations_list
        ]

        logger.info(f"Syncing ASOS forecasts for {len(stations)} stations...")

        # Define forecast hours (all 6-hour increments)
        forecast_hours = list(range(6, 361, 6))

        results = {}

        # Fetch GFS
        try:
            logger.info("Fetching GFS at ASOS stations...")
            gfs_init, gfs_forecasts = fetch_asos_forecasts_for_model('gfs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(gfs_init, 'gfs'):
                logger.info(f"GFS already synced for {gfs_init}, skipping")
                results['gfs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                asos.store_asos_forecasts(gfs_init, forecast_hours, 'gfs', gfs_forecasts)
                results['gfs'] = {'status': 'success', 'stations': len(gfs_forecasts)}
        except Exception as e:
            logger.error(f"GFS ASOS sync failed: {e}")
            results['gfs'] = {'status': 'error', 'error': str(e)}

        # Fetch AIFS
        try:
            logger.info("Fetching AIFS at ASOS stations...")
            aifs_init, aifs_forecasts = fetch_asos_forecasts_for_model('aifs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(aifs_init, 'aifs'):
                logger.info(f"AIFS already synced for {aifs_init}, skipping")
                results['aifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                asos.store_asos_forecasts(aifs_init, forecast_hours, 'aifs', aifs_forecasts)
                results['aifs'] = {'status': 'success', 'stations': len(aifs_forecasts)}
        except Exception as e:
            logger.error(f"AIFS ASOS sync failed: {e}")
            results['aifs'] = {'status': 'error', 'error': str(e)}

        # Fetch IFS
        try:
            logger.info("Fetching IFS at ASOS stations...")
            ifs_init, ifs_forecasts = fetch_asos_forecasts_for_model('ifs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(ifs_init, 'ifs'):
                logger.info(f"IFS already synced for {ifs_init}, skipping")
                results['ifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                asos.store_asos_forecasts(ifs_init, forecast_hours, 'ifs', ifs_forecasts)
                results['ifs'] = {'status': 'success', 'stations': len(ifs_forecasts)}
        except Exception as e:
            logger.error(f"IFS ASOS sync failed: {e}")
            results['ifs'] = {'status': 'error', 'error': str(e)}

        # Fetch NWS forecasts
        try:
            logger.info("Fetching NWS forecasts for ASOS stations...")
            nws_raw = nws_batch.fetch_nws_forecasts_batch_sync(stations)

            # Fetch WPC QPF to replace NWS precipitation (consistent 7-day national coverage)
            logger.info("Fetching WPC 5km QPF for precipitation...")
            try:
                wpc_precip = nws_batch.fetch_wpc_qpf_for_stations_sync(stations)
            except Exception as e:
                logger.warning(f"WPC QPF fetch failed, using NWS precipitation: {e}")
                wpc_precip = {}

            # Transform to ASOS format (temps/mslps/precips aligned with forecast_hours)
            nws_forecasts = nws_batch.transform_nws_to_asos_format(nws_raw, forecast_hours, gfs_init, wpc_precip=wpc_precip)

            # Use GFS init time for NWS (aligns verification timing)
            asos.store_asos_forecasts(gfs_init, forecast_hours, 'nws', nws_forecasts)

            success_count = sum(1 for f in nws_raw.values() if f is not None)
            results['nws'] = {'status': 'success', 'stations': success_count}
            logger.info(f"NWS forecasts synced for {success_count} stations")
        except Exception as e:
            logger.error(f"NWS ASOS sync failed: {e}")
            results['nws'] = {'status': 'error', 'error': str(e)}

        # Fetch observations for past valid times (always do this)
        try:
            logger.info("Fetching ASOS observations from IEM...")
            obs_count = asos.fetch_and_store_observations()
            results['observations'] = {'status': 'success', 'count': obs_count}
        except Exception as e:
            logger.error(f"ASOS observations fetch failed: {e}")
            results['observations'] = {'status': 'error', 'error': str(e)}

        # Fetch 6-hourly data for trend visualization (separate from verification)
        logger.info("=" * 60)
        logger.info("Fetching 6-hourly forecasts for trend visualization...")
        forecast_hours_trends = list(range(0, 361, 6))
        results['trends'] = {}

        # Fetch GFS trends
        try:
            logger.info("Fetching GFS 6-hourly trend data...")
            gfs_init_trends, gfs_trends = fetch_asos_forecasts_for_model('gfs', forecast_hours_trends, stations, init_hour)
            store_asos_trend_run(gfs_init_trends, 'gfs', gfs_trends)
            results['trends']['gfs'] = {'status': 'success', 'stations': len(gfs_trends)}
        except Exception as e:
            logger.error(f"GFS trend data fetch failed: {e}")
            results['trends']['gfs'] = {'status': 'error', 'error': str(e)}

        # Fetch AIFS trends
        try:
            logger.info("Fetching AIFS 6-hourly trend data...")
            aifs_init_trends, aifs_trends = fetch_asos_forecasts_for_model('aifs', forecast_hours_trends, stations, init_hour)
            store_asos_trend_run(aifs_init_trends, 'aifs', aifs_trends)
            results['trends']['aifs'] = {'status': 'success', 'stations': len(aifs_trends)}
        except Exception as e:
            logger.error(f"AIFS trend data fetch failed: {e}")
            results['trends']['aifs'] = {'status': 'error', 'error': str(e)}

        # Fetch IFS trends
        try:
            logger.info("Fetching IFS 6-hourly trend data...")
            ifs_init_trends, ifs_trends = fetch_asos_forecasts_for_model('ifs', forecast_hours_trends, stations, init_hour)
            store_asos_trend_run(ifs_init_trends, 'ifs', ifs_trends)
            results['trends']['ifs'] = {'status': 'success', 'stations': len(ifs_trends)}
        except Exception as e:
            logger.error(f"IFS trend data fetch failed: {e}")
            results['trends']['ifs'] = {'status': 'error', 'error': str(e)}

        # Fetch NWS trends
        try:
            logger.info("Fetching NWS 6-hourly trend data...")
            # Use the same NWS raw data but transform for 6-hourly hours
            nws_trends = nws_batch.transform_nws_to_asos_format(nws_raw, forecast_hours_trends, gfs_init, wpc_precip=wpc_precip)
            store_asos_trend_run(gfs_init, 'nws', nws_trends)
            results['trends']['nws'] = {'status': 'success', 'stations': len(nws_trends)}
        except Exception as e:
            logger.error(f"NWS trend data fetch failed: {e}")
            results['trends']['nws'] = {'status': 'error', 'error': str(e)}

        logger.info("Trend visualization data sync complete")

        return jsonify({
            "success": True,
            "message": f"Synced forecasts for {len(stations)} ASOS stations",
            "init_times": {
                "gfs": gfs_init.isoformat(),
                "aifs": aifs_init.isoformat(),
                "ifs": ifs_init.isoformat()
            },
            "forecast_hours": forecast_hours,
            "results": results
        })

    except Exception as e:
        logger.error(f"ASOS sync error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/asos/nws-resync')
def api_asos_nws_resync():
    """Rebuild NWS ASOS forecasts and verification cache."""
    try:
        stations_list = asos.fetch_all_stations()
        stations = [
            {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
            for s in stations_list
        ]

        gfs_init_str, _ = get_latest_init_times(None)
        gfs_init = datetime.fromisoformat(gfs_init_str)
        if gfs_init.tzinfo is None:
            gfs_init = gfs_init.replace(tzinfo=timezone.utc)
        forecast_hours = list(range(6, 361, 6))
        nws_forecast_hours = [h for h in forecast_hours if h <= 168]
        nws_raw = nws_batch.fetch_nws_forecasts_batch_sync(stations)
        try:
            wpc_precip = nws_batch.fetch_wpc_qpf_for_stations_sync(stations)
        except Exception as e:
            logger.warning(f"WPC QPF fetch failed, using NWS precipitation: {e}")
            wpc_precip = {}
        nws_forecasts = nws_batch.transform_nws_to_asos_format(nws_raw, nws_forecast_hours, gfs_init, wpc_precip=wpc_precip)
        asos.store_asos_forecasts(gfs_init, forecast_hours, 'nws', nws_forecasts)

        # Rebuild verification cache
        asos.precompute_verification_cache()

        return jsonify({
            "success": True,
            "message": "NWS ASOS forecasts refreshed",
            "init_time": gfs_init.isoformat()
        })
    except Exception as e:
        logger.error(f"NWS ASOS resync failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/sync-logs')
def api_sync_logs():
    """
    Server-Sent Events endpoint for streaming sync logs to the client.
    Clients connect to this endpoint to receive real-time log messages.
    """
    def event_stream():
        # Create a queue for this client
        q = queue.Queue(maxsize=100)

        with sync_log_lock:
            sync_log_queues.append(q)

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'message': 'Connected to sync log stream', 'type': 'info'})}\n\n"

            # Stream messages from the queue
            while True:
                try:
                    # Wait for message with timeout
                    msg = q.get(timeout=30)
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f": keepalive\n\n"
        except GeneratorExit:
            # Client disconnected
            with sync_log_lock:
                if q in sync_log_queues:
                    sync_log_queues.remove(q)

    return Response(event_stream(), mimetype='text/event-stream')


def run_master_sync(force: bool = False, init_hour: Optional[int] = None) -> dict:
    """Run the full master sync used by both API and launcher."""
    results = {
        'fairfax': {},
        'asos': {},
        'success': True,
        'errors': []
    }

    try:
        # 1. Sync Fairfax forecast data
        broadcast_sync_log("=" * 60, 'info')
        broadcast_sync_log("STARTING MASTER SYNC", 'info')
        if init_hour is not None:
            broadcast_sync_log(f"Using model init hour: {init_hour:02d}Z", 'info')
        broadcast_sync_log("=" * 60, 'info')
        broadcast_sync_log("Step 1/2: Syncing Fairfax Verification Location", 'info')
        logger.info("Master sync: Fetching Fairfax forecast data...")
        location = "Fairfax, VA"
        days = 15

        try:
            bounds = LOCATIONS[location]
            region = Region(location, bounds)

            # Check if we already have the latest data
            broadcast_sync_log("Checking for model init times...", 'info')
            gfs_init, aifs_init = get_latest_init_times(init_hour)
            already_fetched, message = check_if_already_fetched(location, gfs_init, aifs_init)

            if already_fetched and not force:
                broadcast_sync_log(f"Fairfax data already cached: {message}", 'info')
                logger.info(f"Fairfax data already cached: {message}")
                results['fairfax'] = {
                    'status': 'cached',
                    'message': message,
                    'init_time': gfs_init
                }
            else:
                # Fetch new data
                broadcast_sync_log(f"Fetching new forecast data for {location}...", 'info')
                forecast_hours = list(range(0, min(days * 24, 360) + 1, 6))

                broadcast_sync_log(f"Fetching GFS model data (init: {gfs_init})...", 'info')
                gfs_data = fetch_gfs_data(region, forecast_hours, init_hour)
                broadcast_sync_log("GFS data fetched successfully", 'success')

                broadcast_sync_log(f"Fetching ECMWF AIFS model data (init: {aifs_init})...", 'info')
                aifs_data = fetch_aifs_data(region, forecast_hours, init_hour)
                broadcast_sync_log("AIFS data fetched successfully", 'success')

                broadcast_sync_log("Fetching ECMWF IFS model data...", 'info')
                ifs_data = fetch_ifs_data(region, forecast_hours, init_hour)
                broadcast_sync_log("IFS data fetched successfully", 'success')

                # Update WeatherLink CSV files before fetching observations
                broadcast_sync_log("Updating WeatherLink CSV files...", 'info')
                try:
                    import sys
                    sys.path.insert(0, '/Users/kennypratt/Documents/Townhome_Weather/Weather_Data')
                    import fetch_weather_data as wl_fetcher
                    new_records = wl_fetcher.fetch_missing_data(silent=True)
                    if new_records > 0:
                        broadcast_sync_log(f"Updated WeatherLink data ({new_records} new records)", 'success')
                    else:
                        broadcast_sync_log("WeatherLink data already up to date", 'info')
                except Exception as e:
                    broadcast_sync_log(f"Warning: Could not update WeatherLink CSV: {e}", 'warning')

                broadcast_sync_log("Fetching WeatherLink observations...", 'info')
                observed = fetch_observations(gfs_data.get("times", []), location)
                if observed:
                    broadcast_sync_log("Observations fetched successfully", 'success')

                broadcast_sync_log("Calculating verification metrics...", 'info')
                verification = calculate_all_verification(gfs_data, aifs_data, observed, ifs_data)

                broadcast_sync_log("Saving forecast data to forecasts.json...", 'info')
                save_forecast_data(location, gfs_data, aifs_data, observed, verification, ifs_data)
                broadcast_sync_log("Fairfax forecast data saved successfully", 'success')

                # Auto-compute historical analog pattern matching for this new GFS run.
                # Skips silently if ERA5 drive is not mounted or result is already current.
                broadcast_sync_log("Computing historical analog pattern matching...", 'info')
                try:
                    analog_result = run_analog_sync(gfs_init_time=gfs_init)
                    if analog_result['status'] == 'computed':
                        broadcast_sync_log(
                            f"Analog prediction: {analog_result['avg_precip']:.2f}\" precip / "
                            f"{analog_result['avg_temp']:.1f}°F avg temp "
                            f"(pattern match: {analog_result['avg_correlation']:.0%})",
                            'success'
                        )
                    elif analog_result['status'] == 'cached':
                        broadcast_sync_log("Analog prediction already current", 'info')
                    else:
                        broadcast_sync_log(
                            f"Analog matching skipped: {analog_result.get('message', analog_result.get('error', ''))}",
                            'warning'
                        )
                except Exception as _ae:
                    broadcast_sync_log(f"Analog pattern matching failed: {_ae}", 'warning')
                    logger.warning(f"Analog sync failed during master sync: {_ae}", exc_info=True)

                results['fairfax'] = {
                    'status': 'synced',
                    'message': 'Fetched new forecast data',
                    'init_time': gfs_init
                }
                logger.info(f"Fairfax data synced: {gfs_init}")
        except Exception as e:
            broadcast_sync_log(f"Fairfax sync failed: {e}", 'error')
            logger.error(f"Fairfax sync failed: {e}")
            results['fairfax'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"Fairfax: {str(e)}")

        # Refresh NWS forecast cache once per sync
        try:
            broadcast_sync_log("Refreshing NWS forecast cache...", 'info')
            fetch_nws_forecast_cache(168)
            broadcast_sync_log("NWS forecast cache refreshed", 'success')
        except Exception as e:
            broadcast_sync_log(f"NWS cache refresh failed: {e}", 'warning')
            logger.warning(f"NWS cache refresh failed: {e}")

        # Refresh CoCoRaHS cache for last 30 days
        try:
            broadcast_sync_log("Refreshing CoCoRaHS daily precip cache (last 30 days)...", 'info')
            now_utc = datetime.now(timezone.utc)
            local_now = weatherlink.utc_to_eastern(now_utc.replace(tzinfo=None))
            obs_end = local_now.date() - timedelta(days=1)
            start_date = obs_end - timedelta(days=29)
            fetch_cocorahs_daily_precip(COCORAHs_STATION_ID, start_date.isoformat(), obs_end.isoformat())
            broadcast_sync_log("CoCoRaHS cache refreshed", 'success')
        except Exception as e:
            broadcast_sync_log(f"CoCoRaHS cache refresh failed: {e}", 'warning')
            logger.warning(f"CoCoRaHS cache refresh failed: {e}")

        # Precompute verification time series cache for fast first load
        try:
            broadcast_sync_log("Precomputing verification trends cache...", 'info')
            precompute_verif_time_series_cache("Fairfax, VA", [
                {"variable": "temp", "lead_time": 24, "days_back": 30},
                {"variable": "mslp", "lead_time": 24, "days_back": 30},
                {"variable": "precip", "lead_time": 24, "days_back": 30}
            ])
            broadcast_sync_log("Verification trends cache ready", 'success')
        except Exception as e:
            broadcast_sync_log(f"Verification trends cache failed: {e}", 'warning')
            logger.warning(f"Verification trends cache failed: {e}")

        # Precompute lead-time verification cache for table (monthly + all)
        try:
            broadcast_sync_log("Precomputing verification table cache...", 'info')
            lead_cache = load_verif_lead_cache()
            entries = lead_cache.get("entries", {})
            source_mtimes = _get_verif_ts_source_mtimes()

            for period in ["all", "monthly"]:
                if period == "monthly":
                    result = calculate_lead_time_verification("Fairfax, VA", days_back=30, use_cumulative=False)
                else:
                    result = calculate_lead_time_verification("Fairfax, VA", use_cumulative=True)
                if "error" in result:
                    continue
                cache_key = f"Fairfax, VA|{period}"
                entries[cache_key] = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_mtimes": source_mtimes,
                    "verification": result
                }

            lead_cache["entries"] = entries
            save_verif_lead_cache(lead_cache)
            broadcast_sync_log("Verification table cache ready", 'success')
        except Exception as e:
            broadcast_sync_log(f"Verification table cache failed: {e}", 'warning')
            logger.warning(f"Verification table cache failed: {e}")

        # Precompute bias history cache for fast first load
        try:
            broadcast_sync_log("Precomputing bias history cache...", 'info')
            bias_cache = load_bias_history_cache()
            bias_entries = bias_cache.get("entries", {})
            source_mtimes = _get_verif_ts_source_mtimes()

            for variable, calc_fn in [("temp", calculate_temp_bias_history), ("mslp", calculate_mslp_bias_history)]:
                for lead_time in [24]:
                    for days_back in [30]:
                        result = calc_fn("Fairfax, VA", lead_time, days_back)
                        if "error" in result:
                            continue
                        cache_key = f"{variable}|Fairfax, VA|{lead_time}|{days_back}"
                        bias_entries[cache_key] = {
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "source_mtimes": source_mtimes,
                            "history": result
                        }

            bias_cache["entries"] = bias_entries
            save_bias_history_cache(bias_cache)
            broadcast_sync_log("Bias history cache ready", 'success')
        except Exception as e:
            broadcast_sync_log(f"Bias history cache failed: {e}", 'warning')
            logger.warning(f"Bias history cache failed: {e}")

        # 2. Sync ASOS forecasts
        broadcast_sync_log("-" * 60, 'info')
        broadcast_sync_log("Step 2/2: Syncing ASOS Station Network", 'info')
        logger.info("Master sync: Fetching ASOS station forecasts...")
        try:
            broadcast_sync_log("Loading ASOS station list...", 'info')
            stations_list = asos.fetch_all_stations()
            stations = [
                {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
                for s in stations_list
            ]
            broadcast_sync_log(f"Found {len(stations)} ASOS stations across CONUS", 'info')

            # Define forecast hours for data fetching (all 6-hour increments)
            all_forecast_hours = list(range(6, 361, 6))  # F006, F012, F018, ..., F360

            # Define verification lead times (key hours for computing statistics)
            verification_lead_times = list(range(6, 25, 6)) + list(range(48, 361, 24))

            # Use all_forecast_hours for fetching model data
            forecast_hours = all_forecast_hours
            asos_results = {}

            # Fetch GFS
            gfs_init, gfs_forecasts = fetch_asos_forecasts_for_model('gfs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(gfs_init, 'gfs'):
                broadcast_sync_log(f"GFS ASOS data already synced for {gfs_init} - skipping", 'info')
                logger.info(f"ASOS GFS already synced for {gfs_init}")
                asos_results['gfs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                broadcast_sync_log(f"Extracting GFS forecasts at {len(stations)} ASOS stations...", 'info')
                asos.store_asos_forecasts(gfs_init, forecast_hours, 'gfs', gfs_forecasts)
                broadcast_sync_log(f"GFS ASOS data synced for {len(gfs_forecasts)} stations", 'success')
                asos_results['gfs'] = {'status': 'synced', 'stations': len(gfs_forecasts)}

            # Fetch AIFS
            aifs_init, aifs_forecasts = fetch_asos_forecasts_for_model('aifs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(aifs_init, 'aifs'):
                broadcast_sync_log(f"AIFS ASOS data already synced for {aifs_init} - skipping", 'info')
                logger.info(f"ASOS AIFS already synced for {aifs_init}")
                asos_results['aifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                broadcast_sync_log(f"Extracting AIFS forecasts at {len(stations)} ASOS stations...", 'info')
                asos.store_asos_forecasts(aifs_init, forecast_hours, 'aifs', aifs_forecasts)
                broadcast_sync_log(f"AIFS ASOS data synced for {len(aifs_forecasts)} stations", 'success')
                asos_results['aifs'] = {'status': 'synced', 'stations': len(aifs_forecasts)}

            # Fetch IFS
            ifs_init, ifs_forecasts = fetch_asos_forecasts_for_model('ifs', forecast_hours, stations, init_hour)

            if not force and check_asos_model_exists(ifs_init, 'ifs'):
                broadcast_sync_log(f"IFS ASOS data already synced for {ifs_init} - skipping", 'info')
                logger.info(f"ASOS IFS already synced for {ifs_init}")
                asos_results['ifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                broadcast_sync_log(f"Extracting IFS forecasts at {len(stations)} ASOS stations...", 'info')
                asos.store_asos_forecasts(ifs_init, forecast_hours, 'ifs', ifs_forecasts)
                broadcast_sync_log(f"IFS ASOS data synced for {len(ifs_forecasts)} stations", 'success')
                asos_results['ifs'] = {'status': 'synced', 'stations': len(ifs_forecasts)}

            # Fetch NWS forecasts
            try:
                broadcast_sync_log("Fetching NWS forecasts for ASOS stations (batch mode with rate limiting)...", 'info')
                nws_raw = nws_batch.fetch_nws_forecasts_batch_sync(stations)

                # Fetch WPC QPF to replace NWS precipitation (consistent 7-day national coverage)
                broadcast_sync_log("Fetching WPC 5km QPF for 7-day precipitation...", 'info')
                try:
                    wpc_precip = nws_batch.fetch_wpc_qpf_for_stations_sync(stations)
                except Exception as _wpc_exc:
                    broadcast_sync_log(f"WPC QPF fetch failed, using NWS precipitation: {_wpc_exc}", 'warning')
                    wpc_precip = {}

                # NWS forecasts only extend to ~7 days (168 hours), not 15 days
                # Filter forecast_hours to only include hours <= 168
                nws_forecast_hours = [h for h in forecast_hours if h <= 168]

                # Transform to ASOS format
                logger.info(f"Transforming NWS data with gfs_init={gfs_init} (type: {type(gfs_init)})")
                nws_forecasts = nws_batch.transform_nws_to_asos_format(nws_raw, nws_forecast_hours, gfs_init, wpc_precip=wpc_precip)

                # Use GFS init time for NWS forecasts (aligns verification timing)
                asos.store_asos_forecasts(gfs_init, forecast_hours, 'nws', nws_forecasts)

                success_count = sum(1 for f in nws_raw.values() if f is not None)
                broadcast_sync_log(f"NWS forecasts synced for {success_count}/{len(stations)} stations", 'success')
                asos_results['nws'] = {'status': 'synced', 'stations': success_count}
            except Exception as e:
                import traceback
                broadcast_sync_log(f"NWS forecast fetch failed: {e}", 'warning')
                logger.warning(f"NWS forecast fetch failed: {e}")
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                asos_results['nws'] = {'status': 'error', 'error': str(e)}

            # Fetch observations
            broadcast_sync_log("Fetching ASOS observations from Iowa Environmental Mesonet...", 'info')
            obs_count = asos.fetch_and_store_observations()
            broadcast_sync_log(f"Fetched {obs_count} ASOS observations", 'success')
            asos_results['observations'] = {'status': 'synced', 'count': obs_count}

            # Update SOM cluster skill after observations are refreshed
            try:
                broadcast_sync_log("Updating SOM cluster skill (GFS patterns + NWS precip)...", 'info')
                som_update = update_som_cluster_skill()
                if som_update.get("success"):
                    broadcast_sync_log(f"SOM skill updated ({som_update.get('updates', 0)} new samples)", 'success')
                else:
                    broadcast_sync_log(f"SOM skill update failed: {som_update.get('error', '')}", 'warning')
            except Exception as e:
                broadcast_sync_log(f"SOM skill update failed: {e}", 'warning')
                logger.warning(f"SOM skill update failed: {e}", exc_info=True)

            # Store 6-hourly forecasts for trend visualization (last 2 runs only)
            broadcast_sync_log("Storing 6-hourly forecast data for trend visualization...", 'info')
            try:
                store_asos_trend_run(gfs_init, 'gfs', gfs_forecasts)
                store_asos_trend_run(aifs_init, 'aifs', aifs_forecasts)
                store_asos_trend_run(ifs_init, 'ifs', ifs_forecasts)

                # NWS trends (if available)
                if asos_results.get('nws', {}).get('status') == 'synced':
                    nws_forecast_hours = [h for h in forecast_hours if h <= 168]
                    nws_trends = nws_batch.transform_nws_to_asos_format(nws_raw, nws_forecast_hours, gfs_init, wpc_precip=wpc_precip)
                    store_asos_trend_run(gfs_init, 'nws', nws_trends)

                broadcast_sync_log("Trend visualization data stored successfully", 'success')
                asos_results['trends'] = {'status': 'synced'}
            except Exception as e:
                broadcast_sync_log(f"Warning: Trend data storage failed: {e}", 'warning')
                logger.warning(f"Trend data storage failed: {e}")
                asos_results['trends'] = {'status': 'error', 'error': str(e)}

            broadcast_sync_log("Rebuilding monthly ASOS cache (last 30 days)...", 'info')
            asos.rebuild_monthly_station_cache(days_back=30)
            broadcast_sync_log("Monthly ASOS cache updated", 'success')

            results['asos'] = {
                'status': 'success',
                'models': asos_results,
                'station_count': len(stations)
            }
            broadcast_sync_log(f"ASOS sync complete: {len(stations)} stations processed", 'success')
            logger.info(f"ASOS sync complete: {len(stations)} stations")

        except Exception as e:
            broadcast_sync_log(f"ASOS sync failed: {e}", 'error')
            logger.error(f"ASOS sync failed: {e}")
            results['asos'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"ASOS: {str(e)}")

        # Check if any critical errors occurred
        if results['errors']:
            results['success'] = False
            broadcast_sync_log("=" * 60, 'error')
            broadcast_sync_log("SYNC COMPLETED WITH ERRORS", 'error')
            broadcast_sync_log("=" * 60, 'error')
        else:
            broadcast_sync_log("=" * 60, 'success')
            broadcast_sync_log("SYNC COMPLETED SUCCESSFULLY", 'success')
            broadcast_sync_log("=" * 60, 'success')

        return results

    except Exception as e:
        logger.error(f"Master sync error: {e}")
        return {
            'success': False,
            'error': str(e),
            'fairfax': results.get('fairfax', {}),
            'asos': results.get('asos', {})
        }


@app.route('/api/sync-all')
def api_sync_all():
    """
    Master sync endpoint that syncs everything:
    - Fairfax WeatherLink forecast data (GFS, AIFS, IFS)
    - ASOS station forecasts
    - ASOS observations
    """
    force = request.args.get('force', 'false').lower() == 'true'
    init_hour_param = request.args.get('init_hour')

    init_hour = None
    if init_hour_param:
        try:
            init_hour = int(init_hour_param)
            if init_hour not in [0, 6, 12, 18]:
                return jsonify({
                    'success': False,
                    'error': f'Invalid init_hour {init_hour}. Must be 0, 6, 12, or 18.'
                })
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid init_hour parameter: {init_hour_param}'
            })

    return jsonify(run_master_sync(force=force, init_hour=init_hour))


@app.route('/api/asos/status')
def api_asos_status():
    """Get status of ASOS forecasts database."""
    try:
        db = asos.load_asos_forecasts_db()
        stations = db.get("stations", {})
        runs = db.get("runs", {})

        run_list = []
        for run_id, run_data in runs.items():
            run_info = {
                "init_time": run_id,
                "fetched_at": run_data.get("fetched_at"),
                "forecast_hours": run_data.get("forecast_hours", []),
                "models": []
            }
            for model in ['gfs', 'aifs', 'ifs']:
                if model in run_data:
                    run_info["models"].append(model)
            run_list.append(run_info)

        # Sort by init time descending
        run_list.sort(key=lambda x: x["init_time"], reverse=True)

        # Get cumulative stats summary
        cumulative_summary = asos.get_cumulative_stats_summary()

        return jsonify({
            "success": True,
            "station_count": len(stations),
            "run_count": len(runs),
            "runs": run_list[:10],  # Last 10 runs
            "cumulative_stats": cumulative_summary
        })

    except Exception as e:
        logger.error(f"Error getting ASOS status: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/asos/available-runs')
def api_asos_available_runs():
    """Get list of available forecast runs for single run bias analysis."""
    try:
        # Load ASOS forecasts database
        db = asos.load_asos_forecasts_db()

        if not db or 'runs' not in db:
            return jsonify({"success": False, "error": "No forecast data available"})

        runs = db['runs']

        # Convert to sorted list (most recent first)
        run_list = [{"run_time": rt} for rt in sorted(runs.keys(), reverse=True)]

        return jsonify({
            "success": True,
            "runs": run_list[:50]  # Limit to 50 most recent runs
        })

    except Exception as e:
        logger.error(f"Error getting available runs: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/asos/available-valid-times')
def api_asos_available_valid_times():
    """Get list of available observation/valid times for single run bias analysis.
    Only returns synoptic times (00Z, 06Z, 12Z, 18Z) where we have observation data."""
    try:
        # Load ASOS forecasts database
        db = asos.load_asos_forecasts_db()

        if not db or 'observations' not in db:
            return jsonify({"success": False, "error": "No observation data available"})

        observations = db['observations']

        # Collect synoptic times (00Z, 06Z, 12Z, 18Z) where we have observations
        valid_times_set = set()

        for station_id, station_obs in observations.items():
            for obs_time_str in station_obs.keys():
                obs_time = datetime.fromisoformat(obs_time_str)

                # Only include synoptic times (00Z, 06Z, 12Z, 18Z)
                if obs_time.hour in [0, 6, 12, 18] and obs_time.minute <= 5:
                    # Round to exact synoptic hour
                    synoptic_time = obs_time.replace(minute=0, second=0, microsecond=0)
                    valid_times_set.add(synoptic_time.isoformat())

        # Convert to sorted list (most recent first)
        valid_times = sorted(valid_times_set, reverse=True)

        return jsonify({
            "success": True,
            "valid_times": valid_times[:100]  # Limit to 100 most recent synoptic times
        })

    except Exception as e:
        logger.error(f"Error getting available valid times: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/asos/single-run-bias')
def api_asos_single_run_bias():
    """
    Get bias data for a specific observation time and forecast lead time.

    Query parameters:
    - valid_time: ISO format datetime of the observation/valid time (required)
    - lead_time: Forecast lead time in hours (required)
    - variable: Variable to analyze ('temp', 'mslp', or 'precip', default 'temp')

    Example: valid_time=2026-02-07T12:00:00Z, lead_time=12
    This will use the forecast from 2026-02-07T00:00:00Z at F+12 hours
    """
    try:
        valid_time_str = request.args.get('valid_time')
        lead_time_str = request.args.get('lead_time')
        variable = request.args.get('variable', 'temp')

        if not valid_time_str or not lead_time_str:
            return jsonify({"success": False, "error": "Missing required parameters"})

        # Parse parameters
        valid_time = datetime.fromisoformat(valid_time_str.replace('Z', '+00:00'))
        lead_time = int(lead_time_str)

        # Calculate run time (valid_time - lead_time)
        run_time = valid_time - timedelta(hours=lead_time)

        # Load ASOS forecasts database
        db = asos.load_asos_forecasts_db()

        if not db or 'runs' not in db or 'stations' not in db or 'observations' not in db:
            return jsonify({"success": False, "error": "No forecast data available"})

        runs = db['runs']
        stations_meta = db['stations']
        observations = db['observations']

        # Check if this run exists
        run_key = run_time.isoformat()
        if run_key not in runs:
            return jsonify({"success": False, "error": f"No forecast data for run {run_time.strftime('%Y-%m-%d %HZ')}"})

        run_data = runs[run_key]

        # Get forecast hours for this run
        forecast_hours = run_data.get('forecast_hours', [])
        if lead_time not in forecast_hours:
            return jsonify({"success": False, "error": f"Lead time {lead_time}h not available for this run"})

        # Find the index of this lead time
        try:
            lead_idx = forecast_hours.index(lead_time)
        except ValueError:
            return jsonify({"success": False, "error": f"Lead time {lead_time}h not found"})

        # Variable name mapping
        var_name_map = {
            'temp': 'temps',
            'mslp': 'mslps',
            'precip': 'precips'
        }
        var_key = var_name_map.get(variable, variable + 's')

        # Collect bias data for all stations
        stations_data = {}
        all_biases = {'gfs': [], 'aifs': [], 'ifs': [], 'nws': []}

        # Helper function to get observation near valid time
        # Uses composite observation matching to handle ASOS stations that report
        # different variables at different times (MSLP every 5min, temp/precip hourly at :56)
        def get_observation(station_id, valid_time):
            if station_id not in observations:
                return None

            station_obs = observations[station_id]
            max_delta = timedelta(minutes=30)

            # Find nearest observation for each variable separately
            best = {
                'temp': {'value': None, 'delta': max_delta + timedelta(seconds=1)},
                'mslp': {'value': None, 'delta': max_delta + timedelta(seconds=1)},
                'precip': {'value': None, 'delta': max_delta + timedelta(seconds=1)}
            }

            # Search through all observations to find nearest for each variable
            for obs_time_str, obs_data in station_obs.items():
                try:
                    obs_time = datetime.fromisoformat(obs_time_str)
                    delta = abs(obs_time - valid_time)

                    if delta > max_delta:
                        continue

                    # Check each variable and update if this observation is closer
                    if obs_data.get('temp') is not None and delta < best['temp']['delta']:
                        best['temp']['value'] = obs_data['temp']
                        best['temp']['delta'] = delta

                    if obs_data.get('mslp') is not None and delta < best['mslp']['delta']:
                        best['mslp']['value'] = obs_data['mslp']
                        best['mslp']['delta'] = delta

                    if obs_data.get('precip') is not None and delta < best['precip']['delta']:
                        best['precip']['value'] = obs_data['precip']
                        best['precip']['delta'] = delta

                except (ValueError, AttributeError):
                    continue

            # Build composite observation from nearest source for each variable
            if all(best[v]['value'] is None for v in ['temp', 'mslp', 'precip']):
                return None

            return {
                'temp': best['temp']['value'],
                'mslp': best['mslp']['value'],
                'precip': best['precip']['value']
            }

        # Process each station
        for station_id, station_meta in stations_meta.items():
            station_info = {
                'name': station_meta['name'],
                'lat': station_meta['lat'],
                'lon': station_meta['lon']
            }

            # Get observation for this valid time
            obs = get_observation(station_id, valid_time)
            if not obs:
                continue

            # For precipitation, model forecasts are 6-hour accumulated totals.
            # Use calculate_6hr_precip_total to match the same accumulation window.
            if variable == 'precip':
                observed_value = asos.calculate_6hr_precip_total(db, station_id, valid_time)
                if observed_value is None:
                    continue
            else:
                if obs.get(variable) is None:
                    continue
                observed_value = obs[variable]

            # Get forecast for each model
            for model in ['gfs', 'aifs', 'ifs', 'nws']:
                if model not in run_data:
                    continue

                model_stations = run_data[model]
                if station_id not in model_stations:
                    continue

                station_fcst = model_stations[station_id]
                if var_key not in station_fcst:
                    continue

                forecast_values = station_fcst[var_key]
                if not forecast_values or len(forecast_values) <= lead_idx:
                    continue

                forecast_value = forecast_values[lead_idx]
                if forecast_value is None:
                    continue

                # Calculate bias (forecast - observed)
                bias = forecast_value - observed_value

                # Add to station data
                if station_id not in stations_data:
                    stations_data[station_id] = station_info.copy()

                stations_data[station_id][model] = {
                    'bias': bias,
                    'forecast': forecast_value,
                    'observed': observed_value
                }

                all_biases[model].append(bias)

        if not stations_data:
            return jsonify({"success": False, "error": "No matching data for this run and lead time"})

        # Calculate min/max for color scale (symmetric around 0 for bias)
        all_bias_values = []
        for model_biases in all_biases.values():
            all_bias_values.extend(model_biases)

        # Set color scale ranges based on variable
        if variable == 'temp':
            # Fixed scale for temperature for consistency across runs
            min_value = -10.0
            max_value = 10.0
        elif variable == 'precip':
            # Dynamic scale for precipitation (small values)
            if all_bias_values:
                max_abs = max(abs(min(all_bias_values)), abs(max(all_bias_values)))
                max_abs = max(max_abs, 0.5)  # Minimum ±0.5 inches
                min_value = -max_abs
                max_value = max_abs
            else:
                min_value = -1.0
                max_value = 1.0
        else:
            # Dynamic scale for pressure
            if all_bias_values:
                max_abs = max(abs(min(all_bias_values)), abs(max(all_bias_values)))
                max_abs = max(max_abs, 5.0)  # Minimum ±5 mb
                min_value = -max_abs
                max_value = max_abs
            else:
                min_value = -10.0
                max_value = 10.0

        # Calculate summary statistics for each model
        summary = {}
        for model in ['gfs', 'aifs', 'ifs', 'nws']:
            if all_biases[model]:
                biases = np.array(all_biases[model])
                summary[model] = {
                    'mean': float(np.mean(biases)),
                    'median': float(np.median(biases)),
                    'std': float(np.std(biases)),
                    'count': len(biases)
                }

        return jsonify({
            "success": True,
            "stations": stations_data,
            "min_value": min_value,
            "max_value": max_value,
            "summary": summary,
            "run_time": run_time.isoformat(),
            "valid_time": valid_time.isoformat(),
            "lead_time": lead_time,
            "variable": variable
        })

    except Exception as e:
        logger.error(f"Error getting single run bias: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the Flask server."""
    logger.info("Shutdown request received")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # Alternative shutdown method for production servers
        import os
        import signal
        os.kill(os.getpid(), signal.SIGINT)
    else:
        func()
    return jsonify({"success": True, "message": "Server shutting down..."})


def _eof_precompute_thread():
    """Background thread that builds/loads the EOF cache at startup."""
    global _EOF_CACHE

    try:
        if not _HAS_ANALOG_METRICS:
            logger.info("EOF pre-compute skipped: analog_metrics unavailable")
            return

        era5_path = Path("/Volumes/T7/Weather_Models/era5/global_500mb")
        if not era5_path.exists():
            logger.info("EOF pre-compute skipped: ERA5 path not found")
            return

        era5_files = sorted(era5_path.glob("era5_z500_NH_*.nc"))
        if not era5_files:
            logger.info("EOF pre-compute skipped: no ERA5 files")
            return

        # Try loading from disk first
        cache = _analog_metrics.load_eof_cache(era5_files=era5_files)

        if cache is None:
            logger.info("EOF cache stale or missing – rebuild not triggered automatically at startup")
            logger.info("Tip: POST /api/era5/rebuild-eof-cache to rebuild")
        else:
            _EOF_CACHE = cache
            logger.info("EOF cache loaded successfully (%d timesteps)", len(cache['times']))

    except Exception as e:
        logger.warning("EOF precompute thread error: %s", e)
    finally:
        _EOF_CACHE_READY.set()


# Start EOF precompute in background when module loads (non-blocking)
_eof_thread = threading.Thread(target=_eof_precompute_thread, daemon=True, name="eof-precompute")
_eof_thread.start()


@app.route('/api/era5/rebuild-eof-cache', methods=['POST'])
def api_rebuild_eof_cache():
    """Trigger an EOF cache rebuild in a background thread (long-running)."""
    global _EOF_CACHE

    if not _HAS_ANALOG_METRICS:
        return jsonify({"success": False, "error": "analog_metrics module not available"})

    def _rebuild():
        global _EOF_CACHE
        try:
            import xarray as xr
            import pandas as pd

            era5_path  = Path("/Volumes/T7/Weather_Models/era5/global_500mb")
            era5_files = sorted(era5_path.glob("era5_z500_NH_*.nc"))

            datasets = [xr.open_dataset(f) for f in era5_files]
            ds = xr.concat(datasets, dim='time').sortby('time')

            # Need climatology for anomaly calculation
            # Use _CLIMATOLOGY_CACHE if available; otherwise build a minimal one
            cache_key = (
                tuple(str(f) for f in era5_files),
            )
            # Find any cached climatology that uses these files
            matched_clim = None
            for k, v in _CLIMATOLOGY_CACHE.items():
                if k[0] == tuple(str(f) for f in era5_files):
                    matched_clim = v
                    break

            if matched_clim is None:
                logger.info("EOF rebuild: no climatology cache found; skipping EOF build")
                return

            pressure_coord = None
            for c in ['pressure_level', 'level', 'plev', 'lev']:
                if c in ds.coords or c in ds.dims:
                    pressure_coord = c
                    break

            _analog_metrics.build_eof_cache(ds, matched_clim, pressure_coord=pressure_coord)
            new_cache = _analog_metrics.load_eof_cache(force_rebuild=False)
            if new_cache:
                _EOF_CACHE = new_cache
                logger.info("EOF cache rebuilt and loaded")
        except Exception as e:
            logger.error("EOF rebuild failed: %s", e, exc_info=True)

    t = threading.Thread(target=_rebuild, daemon=True, name="eof-rebuild")
    t.start()
    return jsonify({"success": True, "message": "EOF cache rebuild started in background"})


if __name__ == '__main__':
    app.run(debug=False, port=5001)
