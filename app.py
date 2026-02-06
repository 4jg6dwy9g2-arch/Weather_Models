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
        "fetched_at": datetime.now().isoformat(),
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

            # Store F000 as current wave number
            if fhr == 0:
                z500_waves = wave_metrics
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
        "z500_waves_forecast": z500_waves_forecast
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


def calculate_lead_time_verification(location_name: str) -> dict:
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

    # Collect sums/counts by lead time (supports cumulative stats)
    errors_by_lead_time = {}

    runs_with_obs = 0
    obs_by_time = {}

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
        observed = run_data.get("observed")
        if not observed or not observed.get("temps"):
            continue

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
        obs_temps = observed.get("temps", [])
        obs_mslps = observed.get("mslps", [])

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

            # Calculate lead time in hours
            lead_time_hours = int((valid_time - init_time).total_seconds() / 3600)

            _ensure_lt(lead_time_hours)

            # Cache observed temperature by valid time (UTC) for NWS comparisons
            if i < len(obs_temps):
                obs_temp = obs_temps[i]
                if obs_temp is not None:
                    obs_by_time[valid_time.astimezone(timezone.utc).isoformat()] = obs_temp

            # Collect temperature errors
            if i < len(gfs_temps) and i < len(obs_temps):
                gfs_temp = gfs_temps[i]
                obs_temp = obs_temps[i]
                if gfs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "gfs_temp", gfs_temp - obs_temp)

            if i < len(aifs_temps) and i < len(obs_temps):
                aifs_temp = aifs_temps[i]
                obs_temp = obs_temps[i]
                if aifs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "aifs_temp", aifs_temp - obs_temp)

            if i < len(ifs_temps) and i < len(obs_temps):
                ifs_temp = ifs_temps[i]
                obs_temp = obs_temps[i]
                if ifs_temp is not None and obs_temp is not None:
                    _add(lead_time_hours, "ifs_temp", ifs_temp - obs_temp)

            # Collect MSLP errors
            if i < len(gfs_mslps) and i < len(obs_mslps):
                gfs_mslp = gfs_mslps[i]
                obs_mslp = obs_mslps[i]
                if gfs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "gfs_mslp", gfs_mslp - obs_mslp)

            if i < len(aifs_mslps) and i < len(obs_mslps):
                aifs_mslp = aifs_mslps[i]
                obs_mslp = obs_mslps[i]
                if aifs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "aifs_mslp", aifs_mslp - obs_mslp)

            if i < len(ifs_mslps) and i < len(obs_mslps):
                ifs_mslp = ifs_mslps[i]
                obs_mslp = obs_mslps[i]
                if ifs_mslp is not None and obs_mslp is not None:
                    _add(lead_time_hours, "ifs_mslp", ifs_mslp - obs_mslp)

    # NWS verification (temperature only) using cached forecast + WeatherLink observations
    nws_cache = load_nws_cache()
    runs = nws_cache.get("runs", []) if isinstance(nws_cache, dict) else []
    for run in runs:
        if not run.get("forecast") or not run.get("fetched_at"):
            continue
        try:
            nws_init = datetime.fromisoformat(run["fetched_at"])
            if nws_init.tzinfo is None:
                nws_init = nws_init.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        for entry in run.get("forecast", []):
            try:
                valid_time = datetime.fromisoformat(entry["datetime"])
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            if valid_time >= now:
                continue

            # Only include 6-hourly verification times
            if valid_time.hour % 6 != 0:
                continue

            lead_time_hours = int((valid_time - nws_init).total_seconds() / 3600)
            obs_temp = obs_by_time.get(valid_time.astimezone(timezone.utc).isoformat())
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

    # Build observed lookup by valid time for NWS verification (temp only)
    obs_by_time = {}

    for run_id, run_data in runs.items():
        observed = run_data.get("observed")
        if not observed or not observed.get(obs_key):
            continue

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
        obs_values = observed.get(obs_key, [])

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

            # Calculate lead time in hours
            lead_time = int((valid_time - init_time).total_seconds() / 3600)

            # Only include the requested lead time
            if lead_time != lead_time_hours:
                continue

            # Cache observed values by valid time (UTC) for NWS comparisons
            if i < len(obs_values):
                obs_val = obs_values[i]
                if obs_val is not None:
                    obs_by_time[valid_time.astimezone(timezone.utc).isoformat()] = obs_val

            # Use the date of the valid time for grouping
            date_key = valid_time.date().isoformat()
            _ensure_date(date_key)

            # Collect errors for each model
            if i < len(gfs_values) and i < len(obs_values):
                gfs_val = gfs_values[i]
                obs_val = obs_values[i]
                if gfs_val is not None and obs_val is not None:
                    _add(date_key, "gfs", gfs_val - obs_val)

            if i < len(aifs_values) and i < len(obs_values):
                aifs_val = aifs_values[i]
                obs_val = obs_values[i]
                if aifs_val is not None and obs_val is not None:
                    _add(date_key, "aifs", aifs_val - obs_val)

            if ifs_values and i < len(ifs_values) and i < len(obs_values):
                ifs_val = ifs_values[i]
                obs_val = obs_values[i]
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
                    nws_init = nws_init.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            for entry in run.get("forecast", []):
                try:
                    valid_time = datetime.fromisoformat(entry["datetime"])
                    if valid_time.tzinfo is None:
                        valid_time = valid_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                # Only include past times
                if valid_time >= now:
                    continue

                # Only include 6-hourly verification times
                if valid_time.hour % 6 != 0:
                    continue

                # Calculate lead time in hours
                lead_time = int((valid_time - nws_init).total_seconds() / 3600)
                if lead_time != lead_time_hours:
                    continue

                temp_val = entry.get("temperature")
                obs_val = obs_by_time.get(valid_time.astimezone(timezone.utc).isoformat())
                if temp_val is None or obs_val is None:
                    continue

                date_key = valid_time.date().isoformat()
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

    try:
        result = calculate_lead_time_verification(location_name)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        return jsonify({
            "success": True,
            "location": location_name,
            "verification": result
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

    # Collect individual verification points
    # Key: valid_time ISO string
    verification_points = {}

    for run_id, run_data in runs.items():
        observed = run_data.get("observed")
        if not observed or not observed.get("temps"):
            continue

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
        gfs_temps = gfs_data.get("temps", [])
        aifs_temps = aifs_data.get("temps", [])
        ifs_temps = ifs_data.get("temps", []) if ifs_data else []
        obs_temps = observed.get("temps", [])

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

            # Calculate lead time in hours
            lead_time = int((valid_time - init_time).total_seconds() / 3600)

            # Only include the requested lead time
            if lead_time != lead_time_hours:
                continue

            # Only include 6-hourly verification times to match model cycles
            if valid_time.hour % 6 != 0:
                continue

            # Use the valid time as the key
            time_key = valid_time.astimezone(timezone.utc).isoformat()

            # Only store one verification per valid time (latest forecast run)
            if i < len(obs_temps) and obs_temps[i] is not None:
                obs_val = obs_temps[i]

                verification_points[time_key] = {
                    "observed": obs_val,
                    "gfs_bias": None,
                    "aifs_bias": None,
                    "ifs_bias": None
                }

                if i < len(gfs_temps) and gfs_temps[i] is not None:
                    verification_points[time_key]["gfs_bias"] = gfs_temps[i] - obs_val

                if i < len(aifs_temps) and aifs_temps[i] is not None:
                    verification_points[time_key]["aifs_bias"] = aifs_temps[i] - obs_val

                if ifs_temps and i < len(ifs_temps) and ifs_temps[i] is not None:
                    verification_points[time_key]["ifs_bias"] = ifs_temps[i] - obs_val

    # Sort by time and build result arrays
    sorted_times = sorted(verification_points.keys())

    result = {
        "dates": sorted_times,
        "observed": [],
        "gfs_bias": [],
        "aifs_bias": [],
        "ifs_bias": []
    }

    for time_key in sorted_times:
        point = verification_points[time_key]
        result["observed"].append(round(point["observed"], 1) if point["observed"] is not None else None)
        result["gfs_bias"].append(round(point["gfs_bias"], 2) if point["gfs_bias"] is not None else None)
        result["aifs_bias"].append(round(point["aifs_bias"], 2) if point["aifs_bias"] is not None else None)
        result["ifs_bias"].append(round(point["ifs_bias"], 2) if point["ifs_bias"] is not None else None)

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

    # Collect individual verification points
    # Key: valid_time ISO string
    verification_points = {}

    for run_id, run_data in runs.items():
        observed = run_data.get("observed")
        if not observed or not observed.get("mslps"):
            continue

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
        gfs_mslps = gfs_data.get("mslps", [])
        aifs_mslps = aifs_data.get("mslps", [])
        ifs_mslps = ifs_data.get("mslps", []) if ifs_data else []
        obs_mslps = observed.get("mslps", [])

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

            # Calculate lead time in hours
            lead_time = int((valid_time - init_time).total_seconds() / 3600)

            # Only include the requested lead time
            if lead_time != lead_time_hours:
                continue

            # Only include 00Z and 12Z verification times (reduce clutter)
            if valid_time.hour not in [0, 12]:
                continue

            # Use the valid time as the key
            time_key = valid_time.isoformat()

            # Only store one verification per valid time (latest forecast run)
            if i < len(obs_mslps) and obs_mslps[i] is not None:
                obs_val = obs_mslps[i]

                verification_points[time_key] = {
                    "observed": obs_val,
                    "gfs_bias": None,
                    "aifs_bias": None,
                    "ifs_bias": None
                }

                if i < len(gfs_mslps) and gfs_mslps[i] is not None:
                    verification_points[time_key]["gfs_bias"] = gfs_mslps[i] - obs_val

                if i < len(aifs_mslps) and aifs_mslps[i] is not None:
                    verification_points[time_key]["aifs_bias"] = aifs_mslps[i] - obs_val

                if ifs_mslps and i < len(ifs_mslps) and ifs_mslps[i] is not None:
                    verification_points[time_key]["ifs_bias"] = ifs_mslps[i] - obs_val

    # Sort by time and build result arrays
    sorted_times = sorted(verification_points.keys())

    result = {
        "dates": sorted_times,
        "observed": [],
        "gfs_bias": [],
        "aifs_bias": [],
        "ifs_bias": []
    }

    for time_key in sorted_times:
        point = verification_points[time_key]
        result["observed"].append(round(point["observed"], 1) if point["observed"] is not None else None)
        result["gfs_bias"].append(round(point["gfs_bias"], 2) if point["gfs_bias"] is not None else None)
        result["aifs_bias"].append(round(point["aifs_bias"], 2) if point["aifs_bias"] is not None else None)
        result["ifs_bias"].append(round(point["ifs_bias"], 2) if point["ifs_bias"] is not None else None)

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

        # Get cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

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
        latest_run = db[location_name].get("latest_run")
        current_wave_number = None
        expected_error_72h = None
        expected_error_120h = None
        predictability_status = "UNKNOWN"
        confidence = "Low"

        if latest_run and latest_run in db[location_name]["runs"]:
            current_run = db[location_name]["runs"][latest_run]
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

        try:
            # Fetch temperature
            temp_data = model.fetch_data(temp_var, init_time, hour, region)
            temp_vals = extract_model_at_stations(temp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} temp F{hour} failed: {e}")
            broadcast_sync_log(f"  [{model_name.upper()}] WARNING: temp F{hour:03d} failed: {e}", 'warning')
            temp_vals = [None] * len(stations)

        try:
            # Fetch MSLP
            mslp_data = model.fetch_data(mslp_var, init_time, hour, region)
            mslp_vals = extract_model_at_stations(mslp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} mslp F{hour} failed: {e}")
            broadcast_sync_log(f"  [{model_name.upper()}] WARNING: mslp F{hour:03d} failed: {e}", 'warning')
            mslp_vals = [None] * len(stations)

        try:
            # Fetch precip
            precip_data = model.fetch_data(precip_var, init_time, hour, region)
            precip_vals = extract_model_at_stations(precip_data, stations)
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

        # Check for errors
        if "error" in gfs_data:
            return jsonify({"success": False, "error": gfs_data["error"]}), 404

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
        else:
            gfs_results = asos.get_mean_verification_from_cache('gfs')
            aifs_results = asos.get_mean_verification_from_cache('aifs')
            ifs_results = asos.get_mean_verification_from_cache('ifs')

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
                (ifs_results["temp_mae"][i] is not None)
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
            "gfs_mslp_mae": filter_array(gfs_results["mslp_mae"], filtered_indices),
            "gfs_mslp_bias": filter_array(gfs_results["mslp_bias"], filtered_indices),
            "aifs_mslp_mae": filter_array(aifs_results["mslp_mae"], filtered_indices),
            "aifs_mslp_bias": filter_array(aifs_results["mslp_bias"], filtered_indices),
            "ifs_mslp_mae": filter_array(ifs_results["mslp_mae"], filtered_indices),
            "ifs_mslp_bias": filter_array(ifs_results["mslp_bias"], filtered_indices),
            "gfs_precip_mae": filter_array(gfs_results["precip_mae"], filtered_indices),
            "gfs_precip_bias": filter_array(gfs_results["precip_bias"], filtered_indices),
            "aifs_precip_mae": filter_array(aifs_results["precip_mae"], filtered_indices),
            "aifs_precip_bias": filter_array(aifs_results["precip_bias"], filtered_indices),
            "ifs_precip_mae": filter_array(ifs_results["precip_mae"], filtered_indices),
            "ifs_precip_bias": filter_array(ifs_results["precip_bias"], filtered_indices),
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

        # Fetch observations for past valid times (always do this)
        try:
            logger.info("Fetching ASOS observations from IEM...")
            obs_count = asos.fetch_and_store_observations()
            results['observations'] = {'status': 'success', 'count': obs_count}
        except Exception as e:
            logger.error(f"ASOS observations fetch failed: {e}")
            results['observations'] = {'status': 'error', 'error': str(e)}

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

            # Define forecast hours (6-hourly up to 24h, then 24-hourly to 15 days)
            forecast_hours = list(range(6, 25, 6)) + list(range(48, 361, 24))
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

            # Fetch observations
            broadcast_sync_log("Fetching ASOS observations from Iowa Environmental Mesonet...", 'info')
            obs_count = asos.fetch_and_store_observations()
            broadcast_sync_log(f"Fetched {obs_count} ASOS observations", 'success')
            asos_results['observations'] = {'status': 'synced', 'count': obs_count}

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
