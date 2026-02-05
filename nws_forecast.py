#!/usr/bin/env python3
"""
Lightweight NWS hourly forecast utilities for the Run Forecast tab.
Self-contained copy (no Workout_Data dependency).
"""

import os
import math
import requests
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Default location: Fairfax, VA
DEFAULT_LAT = 38.8419
DEFAULT_LON = -77.3091

# NWS API base URL
NWS_API = "https://api.weather.gov"

# AirNow API for AQI forecasts (optional)
AIRNOW_API_KEY = os.getenv("AQI_API_KEY")
AIRNOW_API = "https://www.airnowapi.org/aq/forecast/latLong"

# User agent required by NWS API
HEADERS = {
    "User-Agent": "(Weather Models App, contact@example.com)",
    "Accept": "application/geo+json"
}

# Running condition thresholds
IDEAL_TEMP_RANGE = (45, 65)  # Fahrenheit
GOOD_TEMP_RANGE = (35, 75)
MAX_WIND_IDEAL = 10  # mph


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

    # UTC offset for Eastern Time (approximate)
    utc_offset = -5
    solar_noon_utc = 12 - (lon / 15)
    solar_noon_local = solar_noon_utc + utc_offset

    sunrise_hour = solar_noon_local - (hour_angle / 15)
    sunset_hour = solar_noon_local + (hour_angle / 15)

    sunrise = date.replace(hour=int(sunrise_hour), minute=int((sunrise_hour % 1) * 60), second=0, microsecond=0)
    sunset = date.replace(hour=int(sunset_hour), minute=int((sunset_hour % 1) * 60), second=0, microsecond=0)

    return sunrise, sunset


def is_daylight(dt: datetime, lat: float, lon: float) -> bool:
    """Check if a given datetime is during daylight hours."""
    sunrise, sunset = calculate_sunrise_sunset(lat, lon, dt)
    return sunrise <= dt.replace(tzinfo=None) <= sunset


def get_grid_point(lat: float, lon: float) -> tuple:
    """Get NWS grid office and coordinates for a lat/lon."""
    url = f"{NWS_API}/points/{lat},{lon}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    props = data["properties"]
    return props["gridId"], props["gridX"], props["gridY"], props["forecastHourly"]


def fetch_wind_gusts(grid_id: str, grid_x: int, grid_y: int) -> dict:
    """
    Fetch wind gust data from NWS gridpoint API.
    Returns dict mapping ISO datetime strings to gust values in mph.
    """
    url = f"{NWS_API}/gridpoints/{grid_id}/{grid_x},{grid_y}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        gusts_by_hour = {}
        wind_gust_data = data.get("properties", {}).get("windGust", {})
        values = wind_gust_data.get("values", [])

        for entry in values:
            valid_time = entry.get("validTime", "")
            time_str = valid_time.split("/")[0] if "/" in valid_time else valid_time
            kmh = entry.get("value", 0)
            mph = int(round(kmh * 0.621371))

            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                hour_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
                gusts_by_hour[hour_key] = mph
            except (ValueError, AttributeError):
                continue

        return gusts_by_hour
    except Exception as e:
        print(f"Warning: Could not fetch wind gust data: {e}")
        return {}


def fetch_aqi_forecast(lat: float, lon: float) -> dict:
    """
    Fetch AQI forecast from AirNow API.
    Returns dict mapping date strings to AQI values.
    """
    if not AIRNOW_API_KEY:
        return {}

    aqi_by_date = {}
    try:
        for days_ahead in range(7):
            date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            url = (
                f"{AIRNOW_API}?format=application/json"
                f"&latitude={lat}&longitude={lon}"
                f"&date={date}&distance=25"
                f"&API_KEY={AIRNOW_API_KEY}"
            )

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue

            data = response.json()
            if not data:
                continue

            for forecast in data:
                forecast_date = forecast.get("DateForecast", "")
                aqi = forecast.get("AQI", -1)
                if aqi >= 0:
                    if forecast_date not in aqi_by_date or aqi > aqi_by_date[forecast_date]:
                        aqi_by_date[forecast_date] = aqi

    except Exception as e:
        print(f"Warning: Could not fetch AQI forecast: {e}")

    return aqi_by_date


def fetch_hourly_forecast(forecast_url: str) -> list:
    """Fetch hourly forecast data from NWS API."""
    response = requests.get(forecast_url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    periods = data["properties"]["periods"]

    forecast = []
    for period in periods:
        start_time = datetime.fromisoformat(period["startTime"].replace("Z", "+00:00"))
        wind_str = period.get("windSpeed", "0 mph")
        if "to" in wind_str:
            wind_speed = int(wind_str.split("to")[1].strip().split()[0])
        else:
            wind_speed = int(wind_str.split()[0]) if wind_str.split()[0].isdigit() else 0

        forecast.append({
            "datetime": start_time.isoformat(),
            "datetime_local": start_time.strftime("%a %m/%d %I:%M %p"),
            "temperature": period["temperature"],
            "temperature_unit": period["temperatureUnit"],
            "wind_speed_mph": wind_speed,
            "wind_direction": period.get("windDirection", ""),
            "precipitation_chance": period.get("probabilityOfPrecipitation", {}).get("value", 0) or 0,
            "rain_amount_mm": period.get("quantitativePrecipitation", {}).get("value") or 0,
            "snow_amount_cm": period.get("snowfallAmount", {}).get("value") or 0,
            "ice_amount_mm": period.get("iceAccumulation", {}).get("value") or 0,
            "humidity": period.get("relativeHumidity", {}).get("value"),
            "short_forecast": period["shortForecast"],
            "is_daytime": period["isDaytime"],
            "icon": period.get("icon", "")
        })

    return forecast


def calculate_dew_point(temp_f: float, humidity: float) -> float:
    """Approximate dew point from temperature and relative humidity."""
    if humidity is None or humidity <= 0:
        return None
    temp_c = (temp_f - 32) * 5/9
    dew_point_c = temp_c - ((100 - humidity) / 5)
    dew_point_f = dew_point_c * 9/5 + 32
    return round(dew_point_f, 1)


def rate_running_conditions(hour: dict, aqi: int = None, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> tuple:
    """Rate running conditions from 0-100. Returns (score, reasons)."""
    score = 100
    reasons = []

    temp = hour["temperature"]
    wind = hour["wind_speed_mph"]
    precip_chance = hour["precipitation_chance"]
    rain_mm = hour.get("rain_amount_mm", 0)
    snow_cm = hour.get("snow_amount_cm", 0)
    ice_mm = hour.get("ice_amount_mm", 0)
    short_forecast = hour["short_forecast"].lower()
    humidity = hour.get("humidity") or 50
    hour_time = datetime.fromisoformat(hour["datetime"])

    if aqi is not None:
        if aqi <= 50:
            pass
        elif aqi <= 100:
            score -= 10
            reasons.append(f"AQI {aqi} (Moderate)")
        elif aqi <= 150:
            score -= 25
            reasons.append(f"AQI {aqi} (USG)")
        elif aqi <= 200:
            score -= 40
            reasons.append(f"AQI {aqi} (Unhealthy)")
        else:
            score -= 60
            reasons.append(f"AQI {aqi} (Very Unhealthy)")

    if IDEAL_TEMP_RANGE[0] <= temp <= IDEAL_TEMP_RANGE[1]:
        reasons.append(f"Ideal temp ({temp}°F)")
    elif GOOD_TEMP_RANGE[0] <= temp <= GOOD_TEMP_RANGE[1]:
        score -= 15
        reasons.append(f"Cool ({temp}°F)" if temp < IDEAL_TEMP_RANGE[0] else f"Warm ({temp}°F)")
    elif (GOOD_TEMP_RANGE[0]-10.0) <= temp < GOOD_TEMP_RANGE[0]:
        score -= 30
        reasons.append(f"Cold ({temp}°F)")
    elif temp < (GOOD_TEMP_RANGE[0]-10.0):
        score -= 60
        reasons.append(f"Very Cold ({temp}°F)")
    else:
        score -= 30
        reasons.append(f"Hot ({temp}°F)")

    if wind <= MAX_WIND_IDEAL:
        pass
    elif wind <= 15:
        score -= 15
        reasons.append(f"Breezy ({wind} mph)")
    elif wind <= 20:
        score -= 50
        reasons.append(f"Windy ({wind} mph)")
    else:
        score -= 70
        reasons.append(f"Very windy ({wind} mph)")

    if rain_mm > 0.1 or "rain" in short_forecast or "showers" in short_forecast:
        if rain_mm > 2:
            score -= 80
            reasons.append(f"Heavy Rain ({round(rain_mm/25.4, 2)}\" likely)")
        elif rain_mm > 0.5:
            score -= 60
            reasons.append(f"Moderate Rain ({round(rain_mm/25.4, 2)}\" likely)")
        else:
            score -= 40
            reasons.append(f"Light Rain ({precip_chance}% chance)")
    elif snow_cm > 0.1 or "snow" in short_forecast:
        if snow_cm > 2:
            score -= 70
            reasons.append(f"Snow ({round(snow_cm/2.54, 1)}\" likely)")
        else:
            score -= 50
            reasons.append(f"Light Snow ({precip_chance}% chance)")
    elif ice_mm > 0.1 or "ice" in short_forecast or "freezing rain" in short_forecast:
        score -= 90
        reasons.append(f"Ice/Freezing Rain ({precip_chance}% chance)")
    elif precip_chance > 20:
        score -= 30
        reasons.append(f"Precip Chance ({precip_chance}%)")

    dew_point = calculate_dew_point(temp, humidity)
    if dew_point is not None:
        if dew_point < 55:
            pass
        elif dew_point < 60:
            score -= 5
            reasons.append(f"Dew point {dew_point:.0f}°F")
        elif dew_point < 65:
            score -= 15
            reasons.append(f"Muggy (DP {dew_point:.0f}°F)")
        elif dew_point < 70:
            score -= 25
            reasons.append(f"Humid (DP {dew_point:.0f}°F)")
        else:
            score -= 35
            reasons.append(f"Oppressive (DP {dew_point:.0f}°F)")

    if not is_daylight(hour_time.replace(tzinfo=None), lat, lon):
        score = 5
        reasons.insert(0, "Dark")

    return max(0, score), reasons
