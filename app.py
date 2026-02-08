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

NWS_CACHE_PATH = Path(__file__).resolve().parent / "nws_forecast_cache.json"
import asos
import rossby_waves
import nws_batch

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
FORECASTS_FILE = Path(__file__).parent / "forecasts.json"

# ASOS forecast trends storage file (last 2 runs with 6-hourly data)
ASOS_TRENDS_FILE = Path(__file__).parent / "asos_forecast_trends.json"

# ERA5 analog forecast history (tracks predictions over time)
ANALOG_HISTORY_FILE = Path(__file__).parent / "analog_forecast_history.json"

if load_dotenv:
    load_dotenv()

# WeatherLink API Credentials (Davis Weather Station)
WEATHERLINK_API_KEY = os.getenv("WEATHERLINK_API_KEY")
WEATHERLINK_API_SECRET = os.getenv("WEATHERLINK_API_SECRET")
WEATHERLINK_STATION_ID = os.getenv("WEATHERLINK_STATION_ID", "117994")

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
    """Load the forecasts database from JSON file."""
    if FORECASTS_FILE.exists():
        try:
            with open(FORECASTS_FILE) as f:
                data = json.load(f)
                # Migrate old format to new format if needed
                return migrate_db_format(data)
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


def save_nws_cache(payload: dict):
    """Save NWS forecast data to cache."""
    try:
        with open(NWS_CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving NWS cache: {e}")


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


def fetch_nws_forecast_cache(hours_ahead: int = 168):
    """Fetch NWS forecast and cache only temperature data for verification overlays."""
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
            "temperature": hour["temperature"]
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

    if variable not in var_map:
        return {"error": "Invalid variable"}

    fcst_key, obs_key = var_map[variable]
    obs_lookup_key = 'temp' if variable == 'temp' else 'mslp'

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
    if variable == 'temp':
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

        if variable == 'temp':
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
        if period == 'monthly':
            result = calculate_lead_time_verification(location_name, days_back=30, use_cumulative=False)
        else:
            result = calculate_lead_time_verification(location_name, use_cumulative=True)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

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
        result = calculate_verification_time_series(location_name, variable, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

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
        result = calculate_temp_bias_history(location_name, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

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
        result = calculate_mslp_bias_history(location_name, lead_time, days_back)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

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
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('500mb Height Anomaly (dam)', fontsize=10)

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


@app.route('/api/era5/analogs')
def api_era5_analogs():
    """
    Find historical analog dates with similar 500mb height patterns using spatial correlation
    of height anomalies. Anomalies are calculated as deviation from the zonal mean to remove
    seasonal bias and focus on wave pattern similarity.

    Query params:
        - top_n: Number of top analogs to return (default: 10)

    Returns:
        JSON with list of analog dates sorted by spatial pattern similarity
    """
    try:
        import xarray as xr
        import numpy as np
        import pandas as pd
        from scipy.stats import pearsonr

        top_n = int(request.args.get('top_n', 10))

        # Load latest forecast data to get current pattern
        db = load_forecasts_db()
        location_name = 'Fairfax, VA'

        if location_name not in db or not db[location_name].get('latest_run'):
            return jsonify({"success": False, "error": "No forecast data available"})

        latest_run = db[location_name]['latest_run']
        run_data = db[location_name]['runs'].get(latest_run)

        if not run_data or 'gfs' not in run_data:
            return jsonify({"success": False, "error": "No GFS forecast data in latest run"})

        # Get initialization time (00Z)
        init_time = pd.Timestamp(latest_run)
        current_date_str = init_time.isoformat()

        # Get current z500 field
        gfs_data = run_data['gfs']
        z500_field_data = gfs_data.get('z500_field')

        if not z500_field_data:
            return jsonify({"success": False, "error": "No z500 field in current forecast. Please sync forecasts again to capture the field data."})

        # Extract current pattern (GFS data)
        current_z500_gfs = np.array(z500_field_data['values'])
        current_lats_gfs = np.array(z500_field_data['latitude'])
        current_lons_gfs = np.array(z500_field_data['longitude'])

        logger.info(f"GFS data contains NaN: {np.any(np.isnan(current_z500_gfs))}")
        logger.info(f"GFS lat range: {current_lats_gfs.min():.1f} to {current_lats_gfs.max():.1f}")
        logger.info(f"GFS lon range: {current_lons_gfs.min():.1f} to {current_lons_gfs.max():.1f}")

        # Convert GFS longitudes from 0/360 to -180/180 convention to match ERA5
        if current_lons_gfs.max() > 180:
            logger.info("Converting GFS longitudes from 0/360 to -180/180 convention")
            current_lons_gfs = np.where(current_lons_gfs > 180, current_lons_gfs - 360, current_lons_gfs)

        # Get wave number for display
        current_waves = gfs_data.get('z500_waves', {})
        current_wave_num = current_waves.get('wave_number')

        # Load ERA5 data
        era5_path = Path("/Volumes/T7/Weather_Models/era5/global_500mb")

        if not era5_path.exists():
            return jsonify({"success": False, "error": f"ERA5 data directory not found: {era5_path}"})

        era5_files = list(era5_path.glob("era5_z500_NH_*.nc"))

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

        # Get ERA5 coordinates for regridding
        era5_lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
        era5_lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values

        logger.info(f"ERA5 lat range: {era5_lats.min():.1f} to {era5_lats.max():.1f}")
        logger.info(f"ERA5 lon range: {era5_lons.min():.1f} to {era5_lons.max():.1f}")

        # Find overlapping latitude/longitude range to avoid NaNs
        # ERA5 latitudes (find overlap with GFS)
        era5_lat_min, era5_lat_max = min(era5_lats), max(era5_lats)
        gfs_lat_min, gfs_lat_max = min(current_lats_gfs), max(current_lats_gfs)
        overlap_lat_min = max(era5_lat_min, gfs_lat_min)
        overlap_lat_max = min(era5_lat_max, gfs_lat_max)

        # Filter ERA5 coordinates to overlapping region
        lat_mask = (era5_lats >= overlap_lat_min) & (era5_lats <= overlap_lat_max)
        era5_lats_overlap = era5_lats[lat_mask]

        # Use all longitudes (both should cover 0-360 or -180 to 180)
        era5_lons_overlap = era5_lons

        logger.info(f"Overlapping latitude range: {overlap_lat_min:.1f} to {overlap_lat_max:.1f}")
        logger.info(f"Regridding GFS data from {current_z500_gfs.shape} to ERA5 grid {(len(era5_lats_overlap), len(era5_lons_overlap))}")

        # Create xarray DataArray from GFS data
        gfs_da = xr.DataArray(
            current_z500_gfs,
            coords={'latitude': current_lats_gfs, 'longitude': current_lons_gfs},
            dims=['latitude', 'longitude']
        )

        # Interpolate to ERA5 grid (overlapping region only)
        current_z500 = gfs_da.interp(latitude=era5_lats_overlap, longitude=era5_lons_overlap, method='linear').values
        logger.info(f"Regridded GFS shape: {current_z500.shape}, contains NaN: {np.any(np.isnan(current_z500))}")

        # Calculate climatology (day-of-year mean using 31-day window)
        # Use 1990-2021 period for climatology (standard 30-year normal)
        logger.info("Calculating climatology from ERA5 data (1990-2021)...")

        # Filter to 1990-2021 and overlapping latitude range for climatology calculation
        lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_coord_name = 'longitude' if 'longitude' in ds.coords else 'lon'

        clim_ds = ds.sel(
            time=slice('1990-01-01', '2021-12-31'),
            **{lat_coord_name: era5_lats_overlap, lon_coord_name: era5_lons_overlap}
        )

        # Group by day of year and calculate mean
        ds_with_doy = clim_ds.assign_coords(dayofyear=clim_ds.time.dt.dayofyear)

        # Calculate climatology with 31-day smoothing window
        climatology = {}
        for doy in range(1, 367):  # 1-366 (leap years)
            # Get window of +/- 15 days around this day of year
            window_days = [(doy + offset - 1) % 366 + 1 for offset in range(-15, 16)]

            # Select all times matching these days of year
            mask = ds_with_doy.dayofyear.isin(window_days)
            if mask.sum() > 0:
                window_data = ds_with_doy.where(mask, drop=True)
                clim_z500 = window_data['z'].sel(**{pressure_coord: 500}).mean(dim='time')
                climatology[doy] = clim_z500.values

        # Calculate current pattern anomaly (relative to climatology)
        current_doy = init_time.dayofyear
        if current_doy in climatology:
            current_clim = climatology[current_doy]
            current_anomaly = current_z500 - current_clim
        else:
            logger.warning(f"No climatology for day {current_doy}, using zonal mean")
            current_zonal_mean = np.mean(current_z500, axis=1, keepdims=True)
            current_anomaly = current_z500 - current_zonal_mean

        current_flat = current_anomaly.flatten()
        logger.info(f"Current anomaly shape: {current_anomaly.shape}, finite values: {np.sum(np.isfinite(current_flat))}/{len(current_flat)}")

        # Calculate spatial correlations with all historical dates
        # Restrict to same season (±45 days) to avoid seasonal bias in weather outcomes
        analogs = []
        seasonal_window = 45  # days

        for t in ds.time.values:
            try:
                # Calculate anomaly (deviation from climatology for this date)
                t_pd = pd.Timestamp(t)
                historical_doy = t_pd.dayofyear

                # Filter to seasonal window: only use dates within ±45 days of current date
                current_doy = init_time.dayofyear
                # Handle year wraparound (e.g., Dec 31 to Jan 31)
                doy_diff = min(
                    abs(historical_doy - current_doy),
                    abs(historical_doy - current_doy + 365),
                    abs(historical_doy - current_doy - 365)
                )
                if doy_diff > seasonal_window:
                    continue  # Skip dates outside seasonal window

                # Get historical z500 (overlapping region only)
                z500 = ds['z'].sel(
                    time=t,
                    **{pressure_coord: 500, lat_coord_name: era5_lats_overlap, lon_coord_name: era5_lons_overlap}
                )
                historical_z500 = z500.values

                if historical_doy in climatology:
                    historical_clim = climatology[historical_doy]
                    historical_anomaly = historical_z500 - historical_clim
                else:
                    # Fallback to zonal mean if no climatology
                    historical_zonal_mean = np.mean(historical_z500, axis=1, keepdims=True)
                    historical_anomaly = historical_z500 - historical_zonal_mean

                historical_flat = historical_anomaly.flatten()

                # Skip if different sizes (shouldn't happen but just in case)
                min_size = min(len(current_flat), len(historical_flat))
                if min_size == 0:
                    continue

                current_subset = current_flat[:min_size]
                historical_subset = historical_flat[:min_size]

                # Calculate spatial correlation of climatological anomalies
                # Use only finite values from both arrays
                finite_mask = np.isfinite(current_subset) & np.isfinite(historical_subset)
                n_finite = np.sum(finite_mask)

                # Require at least 90% of points to be finite for valid correlation
                if n_finite > 0.9 * len(current_subset):
                    current_finite = current_subset[finite_mask]
                    historical_finite = historical_subset[finite_mask]

                    correlation, _ = pearsonr(current_finite, historical_finite)

                    # Calculate wave number for this date
                    wave_result = rossby_waves.calculate_wave_number(z500, latitude=55.0)

                    analogs.append({
                        'date': str(t),
                        'correlation': float(correlation),
                        'wave_number': wave_result.get('wave_number')
                    })

            except Exception as e:
                continue

        # Sort by correlation (highest first)
        analogs.sort(key=lambda x: x['correlation'], reverse=True)

        logger.info(f"Found {len(analogs)} total analogs")
        if len(analogs) > 0:
            logger.info(f"Top correlation: {analogs[0]['correlation']:.3f}")

        # Get top N analogs with minimum separation to avoid clustering
        # Ensure analogs are at least 7 days apart to get diverse patterns
        top_analogs = []
        min_separation_days = 7

        for analog in analogs:
            # Check if this analog is too close to any already selected
            analog_date = pd.Timestamp(analog['date'])
            too_close = False

            for selected in top_analogs:
                selected_date = pd.Timestamp(selected['date'])
                days_apart = abs((analog_date - selected_date).days)

                if days_apart < min_separation_days:
                    too_close = True
                    break

            # If not too close to any selected analog, add it
            if not too_close:
                top_analogs.append(analog)

            # Stop once we have enough
            if len(top_analogs) >= top_n:
                break

        logger.info(f"Selected {len(top_analogs)} diverse analogs (min {min_separation_days} days apart)")

        # Load Fairfax weather data to calculate analog outcomes
        weather_path = Path("/Volumes/T7/Weather_Models/era5/Fairfax/reanalysis-era5-single-levels-timeseries-sfc1zs15i59.nc")

        if weather_path.exists():
            try:
                weather_ds = xr.open_dataset(weather_path)

                for analog in top_analogs:
                    try:
                        analog_date = pd.Timestamp(analog['date'])
                        # Calculate 14 days after analog date
                        end_date = analog_date + pd.Timedelta(days=14)

                        # Select precipitation for next 14 days
                        precip_subset = weather_ds['tp'].sel(
                            valid_time=slice(analog_date, end_date)
                        )

                        # Sum precipitation (convert from m to inches: 1m = 39.3701 inches)
                        # ERA5 tp is in meters of water equivalent
                        total_precip_m = float(precip_subset.sum().values)
                        total_precip_in = total_precip_m * 39.3701
                        analog['precip_14d'] = round(total_precip_in, 2)

                        # Select temperature for next 14 days
                        temp_subset = weather_ds['t2m'].sel(
                            valid_time=slice(analog_date, end_date)
                        )

                        # Calculate average temperature (convert from Kelvin to Fahrenheit)
                        avg_temp_k = float(temp_subset.mean().values)
                        avg_temp_f = (avg_temp_k - 273.15) * 9/5 + 32
                        analog['temp_14d'] = round(avg_temp_f, 1)

                    except Exception as e:
                        logger.warning(f"Could not get weather data for analog {analog['date']}: {e}")
                        analog['precip_14d'] = None
                        analog['temp_14d'] = None

                # Calculate average precipitation across all analogs
                precip_values = [a['precip_14d'] for a in top_analogs if a.get('precip_14d') is not None]
                avg_precip = round(np.mean(precip_values), 2) if precip_values else None

                # Calculate average temperature across all analogs
                temp_values = [a['temp_14d'] for a in top_analogs if a.get('temp_14d') is not None]
                avg_temp = round(np.mean(temp_values), 1) if temp_values else None

                # Calculate climatological normals for current time of year
                try:
                    current_month = init_time.month
                    current_day = init_time.day

                    # Get all years of data for this approximate time of year
                    climatology_precip = []
                    climatology_temp = []

                    for year in range(1940, 2026):
                        try:
                            start_date = pd.Timestamp(year=year, month=current_month, day=current_day)
                            end_date = start_date + pd.Timedelta(days=14)

                            # Precipitation climatology
                            clim_precip = weather_ds['tp'].sel(
                                valid_time=slice(start_date, end_date)
                            )
                            total_precip_m = float(clim_precip.sum().values)
                            if not np.isnan(total_precip_m):
                                total_precip_in = total_precip_m * 39.3701
                                climatology_precip.append(total_precip_in)

                            # Temperature climatology
                            clim_temp = weather_ds['t2m'].sel(
                                valid_time=slice(start_date, end_date)
                            )
                            avg_temp_k = float(clim_temp.mean().values)
                            if not np.isnan(avg_temp_k):
                                avg_temp_f = (avg_temp_k - 273.15) * 9/5 + 32
                                climatology_temp.append(avg_temp_f)

                        except:
                            continue

                    climatology_normal_precip = round(np.mean(climatology_precip), 2) if climatology_precip else None
                    climatology_normal_temp = round(np.mean(climatology_temp), 1) if climatology_temp else None

                except Exception as e:
                    logger.warning(f"Could not calculate climatology: {e}")
                    climatology_normal_precip = None
                    climatology_normal_temp = None

            except Exception as e:
                logger.warning(f"Could not load Fairfax weather data: {e}")
                avg_precip = None
                avg_temp = None
                climatology_normal_precip = None
                climatology_normal_temp = None
        else:
            logger.warning(f"Fairfax weather data not found at {weather_path}")
            avg_precip = None
            avg_temp = None
            climatology_normal_precip = None
            climatology_normal_temp = None

        # Save prediction to history
        if avg_precip is not None and avg_temp is not None:
            # Calculate average correlation from top analogs
            avg_correlation = sum(a['correlation'] for a in top_analogs) / len(top_analogs) if top_analogs else 0

            save_analog_prediction(
                target_date=current_date_str,
                analog_precip=avg_precip,
                analog_temp=avg_temp,
                climatology_precip=climatology_normal_precip if climatology_normal_precip else 0,
                climatology_temp=climatology_normal_temp if climatology_normal_temp else 0,
                top_analogs=[{'date': a['date'], 'correlation': a['correlation']} for a in top_analogs],
                avg_correlation=avg_correlation
            )

        # Return top N
        return jsonify({
            "success": True,
            "current_date": current_date_str,
            "current_wave_number": current_wave_num,
            "analogs": top_analogs,
            "avg_precip_14d": avg_precip,
            "avg_temp_14d": avg_temp,
            "climatology_precip_14d": climatology_normal_precip,
            "climatology_temp_14d": climatology_normal_temp
        })

    except Exception as e:
        logger.error(f"Error finding analogs: {e}", exc_info=True)
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

        # Find the index for the requested lead time
        if lead_time not in forecast_hours:
            return jsonify({
                "success": False,
                "error": f"Lead time {lead_time}h not available. Available: {forecast_hours}"
            }), 400

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

    # Check cache
    cache_key = f"{variable}_{metric}_{model}_{lead_time}"
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

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id parameter"}), 400
    if not model:
        return jsonify({"success": False, "error": "Missing model parameter"}), 400

    try:
        if period == 'monthly':
            raw_result = asos.get_station_detail_monthly(station_id, model, days_back=30)
        else:
            raw_result = asos.get_station_detail(station_id, model)

        if "error" in raw_result:
            return jsonify({"success": False, "error": raw_result["error"]}), 404

        lead_times = raw_result.get("lead_times", [])
        data_by_lead_time = raw_result.get("data", {})

        temp_mae = []
        temp_bias = []
        mslp_mae = []
        mslp_bias = []

        for lt in lead_times:
            model_data = data_by_lead_time.get(lt, {}).get(model, {})
            
            temp_metrics = model_data.get("temp")
            temp_mae.append(temp_metrics["mae"] if temp_metrics else None)
            temp_bias.append(temp_metrics["bias"] if temp_metrics else None)

            mslp_metrics = model_data.get("mslp")
            mslp_mae.append(mslp_metrics["mae"] if mslp_metrics else None)
            mslp_bias.append(mslp_metrics["bias"] if mslp_metrics else None)
        
        verification_data = {
            "lead_times": lead_times,
            "temp_mae": temp_mae,
            "temp_bias": temp_bias,
            "mslp_mae": mslp_mae,
            "mslp_bias": mslp_bias,
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

        # Define forecast hours (6-hourly up to 24h, then 24-hourly to 15 days)
        forecast_hours = list(range(6, 25, 6)) + list(range(48, 361, 24))

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

            # Transform to ASOS format (temps/mslps/precips aligned with forecast_hours)
            nws_forecasts = nws_batch.transform_nws_to_asos_format(nws_raw, forecast_hours, gfs_init)

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
            nws_trends = nws_batch.transform_nws_to_asos_format(nws_raw, forecast_hours_trends, gfs_init)
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

                # NWS forecasts only extend to ~7 days (168 hours), not 15 days
                # Filter forecast_hours to only include hours <= 168
                nws_forecast_hours = [h for h in forecast_hours if h <= 168]

                # Transform to ASOS format
                logger.info(f"Transforming NWS data with gfs_init={gfs_init} (type: {type(gfs_init)})")
                nws_forecasts = nws_batch.transform_nws_to_asos_format(nws_raw, nws_forecast_hours, gfs_init)

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

            # Store 6-hourly forecasts for trend visualization (last 2 runs only)
            broadcast_sync_log("Storing 6-hourly forecast data for trend visualization...", 'info')
            try:
                store_asos_trend_run(gfs_init, 'gfs', gfs_forecasts)
                store_asos_trend_run(aifs_init, 'aifs', aifs_forecasts)
                store_asos_trend_run(ifs_init, 'ifs', ifs_forecasts)

                # NWS trends (if available)
                if asos_results.get('nws', {}).get('status') == 'synced':
                    nws_forecast_hours = [h for h in forecast_hours if h <= 168]
                    nws_trends = nws_batch.transform_nws_to_asos_format(nws_raw, nws_forecast_hours, gfs_init)
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


if __name__ == '__main__':
    app.run(debug=False, port=5001)
