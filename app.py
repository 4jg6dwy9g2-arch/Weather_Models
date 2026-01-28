"""
Weather Forecast Comparison Flask App
Compare GFS and ECMWF AIFS model forecasts.
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import logging
import json
from functools import lru_cache
import time

from gfs import GFSModel
from ecmwf_aifs import AIFSModel, AIFS_VARIABLES
from ecmwf_ifs import IFSModel, IFS_VARIABLES
import weatherlink
import asos

# Single JSON file for storing all forecast data
FORECASTS_FILE = Path(__file__).parent / "forecasts.json"


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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


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


def fetch_gfs_data(region, forecast_hours):
    """Fetch temperature, precipitation, and MSLP data from GFS.

    Note: GFS precipitation is already per-interval (e.g., 0-6h, 6-12h).
    """
    gfs_model = GFSModel()
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

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat()
    }


def fetch_aifs_data(region, forecast_hours):
    """Fetch temperature, precipitation, and MSLP data from ECMWF AIFS.

    Note: AIFS precipitation is cumulative from init time, so we convert
    to 6-hour interval totals to match GFS behavior.
    """
    aifs_model = AIFSModel()
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

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat()
    }


def fetch_ifs_data(region, forecast_hours):
    """Fetch temperature, precipitation, and MSLP data from ECMWF IFS.

    Note: IFS precipitation is cumulative from init time, so we convert
    to 6-hour interval totals to match GFS behavior.
    """
    ifs_model = IFSModel()
    init_time = ifs_model.get_latest_init_time()

    temp_var = IFS_VARIABLES["t2m"]
    precip_var = IFS_VARIABLES["tp"]
    mslp_var = IFS_VARIABLES["mslp"]

    temps = []
    cumulative_precips = []
    mslps = []
    times = []

    # IFS only goes to 240 hours via open data
    max_ifs_hour = 240
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

    return {
        "temps": temps,
        "precips": precips,
        "mslps": mslps,
        "times": times,
        "init_time": init_time.isoformat()
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
    from datetime import timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    errors = []

    for i, time_str in enumerate(forecast_times):
        # Only verify past times
        try:
            valid_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            if valid_time.tzinfo is not None:
                valid_time = valid_time.replace(tzinfo=None)
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
    from datetime import timezone

    db = load_forecasts_db()

    if location_name not in db:
        return {"error": "Location not found"}

    runs = db[location_name].get("runs", {})
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Collect errors by lead time
    # Key: lead_time_hours, Value: {"gfs_temp": [...], "aifs_temp": [...], "gfs_mslp": [...], "aifs_mslp": [...]}
    errors_by_lead_time = {}

    runs_with_obs = 0

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
            except (ValueError, TypeError):
                continue

            # Only include past times (verified)
            if valid_time >= now:
                continue

            # Calculate lead time in hours
            lead_time_hours = int((valid_time - init_time).total_seconds() / 3600)

            if lead_time_hours not in errors_by_lead_time:
                errors_by_lead_time[lead_time_hours] = {
                    "gfs_temp": [], "aifs_temp": [], "ifs_temp": [],
                    "gfs_mslp": [], "aifs_mslp": [], "ifs_mslp": []
                }

            # Collect temperature errors
            if i < len(gfs_temps) and i < len(obs_temps):
                gfs_temp = gfs_temps[i]
                obs_temp = obs_temps[i]
                if gfs_temp is not None and obs_temp is not None:
                    errors_by_lead_time[lead_time_hours]["gfs_temp"].append(gfs_temp - obs_temp)

            if i < len(aifs_temps) and i < len(obs_temps):
                aifs_temp = aifs_temps[i]
                obs_temp = obs_temps[i]
                if aifs_temp is not None and obs_temp is not None:
                    errors_by_lead_time[lead_time_hours]["aifs_temp"].append(aifs_temp - obs_temp)

            if i < len(ifs_temps) and i < len(obs_temps):
                ifs_temp = ifs_temps[i]
                obs_temp = obs_temps[i]
                if ifs_temp is not None and obs_temp is not None:
                    errors_by_lead_time[lead_time_hours]["ifs_temp"].append(ifs_temp - obs_temp)

            # Collect MSLP errors
            if i < len(gfs_mslps) and i < len(obs_mslps):
                gfs_mslp = gfs_mslps[i]
                obs_mslp = obs_mslps[i]
                if gfs_mslp is not None and obs_mslp is not None:
                    errors_by_lead_time[lead_time_hours]["gfs_mslp"].append(gfs_mslp - obs_mslp)

            if i < len(aifs_mslps) and i < len(obs_mslps):
                aifs_mslp = aifs_mslps[i]
                obs_mslp = obs_mslps[i]
                if aifs_mslp is not None and obs_mslp is not None:
                    errors_by_lead_time[lead_time_hours]["aifs_mslp"].append(aifs_mslp - obs_mslp)

            if i < len(ifs_mslps) and i < len(obs_mslps):
                ifs_mslp = ifs_mslps[i]
                obs_mslp = obs_mslps[i]
                if ifs_mslp is not None and obs_mslp is not None:
                    errors_by_lead_time[lead_time_hours]["ifs_mslp"].append(ifs_mslp - obs_mslp)

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

        if gfs_temp_errors:
            result["gfs_temp_mae"].append(round(sum(abs(e) for e in gfs_temp_errors) / len(gfs_temp_errors), 2))
            result["gfs_temp_bias"].append(round(sum(gfs_temp_errors) / len(gfs_temp_errors), 2))
        else:
            result["gfs_temp_mae"].append(None)
            result["gfs_temp_bias"].append(None)

        if aifs_temp_errors:
            result["aifs_temp_mae"].append(round(sum(abs(e) for e in aifs_temp_errors) / len(aifs_temp_errors), 2))
            result["aifs_temp_bias"].append(round(sum(aifs_temp_errors) / len(aifs_temp_errors), 2))
        else:
            result["aifs_temp_mae"].append(None)
            result["aifs_temp_bias"].append(None)

        ifs_temp_errors = errors["ifs_temp"]
        if ifs_temp_errors:
            result["ifs_temp_mae"].append(round(sum(abs(e) for e in ifs_temp_errors) / len(ifs_temp_errors), 2))
            result["ifs_temp_bias"].append(round(sum(ifs_temp_errors) / len(ifs_temp_errors), 2))
        else:
            result["ifs_temp_mae"].append(None)
            result["ifs_temp_bias"].append(None)

        result["temp_sample_counts"].append(max(len(gfs_temp_errors), len(aifs_temp_errors), len(ifs_temp_errors)))

        # MSLP
        gfs_mslp_errors = errors["gfs_mslp"]
        aifs_mslp_errors = errors["aifs_mslp"]

        if gfs_mslp_errors:
            result["gfs_mslp_mae"].append(round(sum(abs(e) for e in gfs_mslp_errors) / len(gfs_mslp_errors), 2))
            result["gfs_mslp_bias"].append(round(sum(gfs_mslp_errors) / len(gfs_mslp_errors), 2))
        else:
            result["gfs_mslp_mae"].append(None)
            result["gfs_mslp_bias"].append(None)

        if aifs_mslp_errors:
            result["aifs_mslp_mae"].append(round(sum(abs(e) for e in aifs_mslp_errors) / len(aifs_mslp_errors), 2))
            result["aifs_mslp_bias"].append(round(sum(aifs_mslp_errors) / len(aifs_mslp_errors), 2))
        else:
            result["aifs_mslp_mae"].append(None)
            result["aifs_mslp_bias"].append(None)

        ifs_mslp_errors = errors["ifs_mslp"]
        if ifs_mslp_errors:
            result["ifs_mslp_mae"].append(round(sum(abs(e) for e in ifs_mslp_errors) / len(ifs_mslp_errors), 2))
            result["ifs_mslp_bias"].append(round(sum(ifs_mslp_errors) / len(ifs_mslp_errors), 2))
        else:
            result["ifs_mslp_mae"].append(None)
            result["ifs_mslp_bias"].append(None)

        result["mslp_sample_counts"].append(max(len(gfs_mslp_errors), len(aifs_mslp_errors), len(ifs_mslp_errors)))

    return result


def save_forecast_data(location_name, gfs_data, aifs_data, observed=None, verification=None, ifs_data=None):
    """
    Save forecast data to the central JSON file.
    Stores each model run as a separate entry keyed by init_time.
    """
    from datetime import timezone
    db = load_forecasts_db()

    # Initialize location if needed
    if location_name not in db:
        db[location_name] = {"runs": {}, "latest_run": None}

    # Use GFS init_time as the run key (all models should have same init time)
    run_key = gfs_data.get("init_time")
    if not run_key:
        logger.warning("No init_time in GFS data, cannot save run")
        return str(FORECASTS_FILE)

    # Create run entry
    run_data = {
        "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
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

    save_forecasts_db(db)
    logger.info(f"Saved run {run_key} for {location_name} (total runs: {len(db[location_name]['runs'])})")
    return str(FORECASTS_FILE)


def get_latest_init_times():
    """Get the latest available init times from both models without fetching full data."""
    gfs_model = GFSModel()
    aifs_model = AIFSModel()

    gfs_init = gfs_model.get_latest_init_time()
    aifs_init = aifs_model.get_latest_init_time()

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
    """Main dashboard page - displays latest saved data."""
    return render_template(
        'dashboard.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA",
        forecast_days=7
    )


@app.route('/sync')
def sync():
    """Sync page - fetch new data from models."""
    return render_template(
        'sync.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA"
    )


@app.route('/api/forecast')
def api_forecast():
    """API endpoint to fetch forecast data."""
    location_name = request.args.get('location', 'Fairfax, VA')
    days = int(request.args.get('days', 7))
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
    """Verification dashboard page - shows forecast skill by lead time."""
    return render_template(
        'verification.html',
        locations=list(LOCATIONS.keys()),
        selected_location="Fairfax, VA"
    )


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


def fetch_asos_forecasts_for_model(model_name, init_time, forecast_hours, stations):
    """
    Fetch forecasts at all ASOS station locations for a model.

    Args:
        model_name: 'gfs', 'aifs', or 'ifs'
        init_time: Model initialization time
        forecast_hours: List of forecast hours
        stations: List of station dicts

    Returns:
        Dict mapping station_id to forecast data
    """
    import tempfile
    from pathlib import Path

    region = Region("CONUS", CONUS_BOUNDS)

    # Initialize model
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

    # For IFS, limit to available hours
    if model_name.lower() == 'ifs':
        max_ifs_hour = 240
        model_hours = [h for h in forecast_hours if h <= max_ifs_hour]
    else:
        model_hours = forecast_hours

    # Initialize result dict
    station_forecasts = {
        s['station_id']: {'temps': [], 'mslps': [], 'precips': []}
        for s in stations
    }

    # Fetch each forecast hour
    for hour in model_hours:
        logger.info(f"Extracting {model_name.upper()} F{hour:03d} at ASOS stations...")

        try:
            # Fetch temperature
            temp_data = model.fetch_data(temp_var, init_time, hour, region)
            temp_vals = extract_model_at_stations(temp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} temp F{hour} failed: {e}")
            temp_vals = [None] * len(stations)

        try:
            # Fetch MSLP
            mslp_data = model.fetch_data(mslp_var, init_time, hour, region)
            mslp_vals = extract_model_at_stations(mslp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} mslp F{hour} failed: {e}")
            mslp_vals = [None] * len(stations)

        try:
            # Fetch precip
            precip_data = model.fetch_data(precip_var, init_time, hour, region)
            precip_vals = extract_model_at_stations(precip_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} precip F{hour} failed: {e}")
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

    return station_forecasts


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

    # Check cache
    cache_key = f"{variable}_{metric}_{model}_{lead_time}"
    cache_ttl = 3600  # 1 hour

    if cache_key in _verification_cache:
        cache_time = _verification_cache_time.get(cache_key, 0)
        if time.time() - cache_time < cache_ttl:
            return jsonify(_verification_cache[cache_key])

    try:
        # Get verification data
        verification = asos.get_verification_data(model, variable, lead_time)

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


@app.route('/api/asos/station-verification')
def api_asos_station_verification():
    """Get detailed verification for a single station across all lead times."""
    station_id = request.args.get('station_id')
    model = request.args.get('model') # Required model filter

    if not station_id:
        return jsonify({"success": False, "error": "Missing station_id parameter"}), 400
    if not model:
        return jsonify({"success": False, "error": "Missing model parameter"}), 400

    try:
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
            "verification": verification_data
        })

    except Exception as e:
        logger.error(f"Error getting station verification for {station_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/asos/mean-verification')
def api_asos_mean_verification():
    """Get mean verification (MAE and Bias) across all stations by lead time for all models."""
    # The 'model' parameter is no longer taken here, as we fetch for all models.
    # Location is not directly used for mean across all stations.

    try:
        gfs_results = asos.get_mean_verification_by_lead_time('gfs')
        aifs_results = asos.get_mean_verification_by_lead_time('aifs')
        ifs_results = asos.get_mean_verification_by_lead_time('ifs')

        # Check for errors from asos functions
        if "error" in gfs_results:
            return jsonify({"success": False, "error": gfs_results["error"]}), 404
        if "error" in aifs_results:
            return jsonify({"success": False, "error": aifs_results["error"]}), 404
        if "error" in ifs_results:
            return jsonify({"success": False, "error": ifs_results["error"]}), 404

        # Combine results
        # Assuming lead_times are the same for all models
        lead_times = gfs_results["lead_times"]

        combined_verification = {
            "lead_times": lead_times,
            "gfs_temp_mae": gfs_results["temp_mae"],
            "gfs_temp_bias": gfs_results["temp_bias"],
            "aifs_temp_mae": aifs_results["temp_mae"],
            "aifs_temp_bias": aifs_results["temp_bias"],
            "ifs_temp_mae": ifs_results["temp_mae"],
            "ifs_temp_bias": ifs_results["temp_bias"],
            "gfs_mslp_mae": gfs_results["mslp_mae"],
            "gfs_mslp_bias": gfs_results["mslp_bias"],
            "aifs_mslp_mae": aifs_results["mslp_mae"],
            "aifs_mslp_bias": aifs_results["mslp_bias"],
            "ifs_mslp_mae": ifs_results["mslp_mae"],
            "ifs_mslp_bias": ifs_results["mslp_bias"],
        }

        return jsonify({
            "success": True,
            "verification": combined_verification
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
    """
    force = request.args.get('force', 'false').lower() == 'true'

    try:
        # Get stations
        stations_list = asos.fetch_all_stations()
        stations = [
            {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
            for s in stations_list
        ]

        logger.info(f"Syncing ASOS forecasts for {len(stations)} stations...")

        # Define forecast hours (6-hourly out to 15 days)
        forecast_hours = list(range(0, 361, 6))

        results = {}

        # Fetch GFS
        gfs_model = GFSModel()
        gfs_init = gfs_model.get_latest_init_time()

        if not force and check_asos_model_exists(gfs_init, 'gfs'):
            logger.info(f"GFS already synced for {gfs_init}, skipping")
            results['gfs'] = {'status': 'skipped', 'reason': 'already synced'}
        else:
            try:
                logger.info("Fetching GFS at ASOS stations...")
                gfs_forecasts = fetch_asos_forecasts_for_model('gfs', gfs_init, forecast_hours, stations)
                asos.store_asos_forecasts(gfs_init, forecast_hours, 'gfs', gfs_forecasts)
                results['gfs'] = {'status': 'success', 'stations': len(gfs_forecasts)}
            except Exception as e:
                logger.error(f"GFS ASOS sync failed: {e}")
                results['gfs'] = {'status': 'error', 'error': str(e)}

        # Fetch AIFS
        aifs_model = AIFSModel()
        aifs_init = aifs_model.get_latest_init_time()

        if not force and check_asos_model_exists(aifs_init, 'aifs'):
            logger.info(f"AIFS already synced for {aifs_init}, skipping")
            results['aifs'] = {'status': 'skipped', 'reason': 'already synced'}
        else:
            try:
                logger.info("Fetching AIFS at ASOS stations...")
                aifs_forecasts = fetch_asos_forecasts_for_model('aifs', aifs_init, forecast_hours, stations)
                asos.store_asos_forecasts(aifs_init, forecast_hours, 'aifs', aifs_forecasts)
                results['aifs'] = {'status': 'success', 'stations': len(aifs_forecasts)}
            except Exception as e:
                logger.error(f"AIFS ASOS sync failed: {e}")
                results['aifs'] = {'status': 'error', 'error': str(e)}

        # Fetch IFS
        ifs_model = IFSModel()
        ifs_init = ifs_model.get_latest_init_time()

        if not force and check_asos_model_exists(ifs_init, 'ifs'):
            logger.info(f"IFS already synced for {ifs_init}, skipping")
            results['ifs'] = {'status': 'skipped', 'reason': 'already synced'}
        else:
            try:
                logger.info("Fetching IFS at ASOS stations...")
                ifs_forecasts = fetch_asos_forecasts_for_model('ifs', ifs_init, forecast_hours, stations)
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


@app.route('/api/sync-all')
def api_sync_all():
    """
    Master sync endpoint that syncs everything:
    - Fairfax WeatherLink forecast data (GFS, AIFS, IFS)
    - ASOS station forecasts
    - ASOS observations
    """
    force = request.args.get('force', 'false').lower() == 'true'

    results = {
        'fairfax': {},
        'asos': {},
        'success': True,
        'errors': []
    }

    try:
        # 1. Sync Fairfax forecast data
        logger.info("Master sync: Fetching Fairfax forecast data...")
        location = "Fairfax, VA"
        days = 7

        try:
            bounds = LOCATIONS[location]
            region = Region(location, bounds)

            # Check if we already have the latest data
            gfs_init, aifs_init = get_latest_init_times()
            already_fetched, message = check_if_already_fetched(location, gfs_init, aifs_init)

            if already_fetched and not force:
                logger.info(f"Fairfax data already cached: {message}")
                db = load_forecasts_db()
                loc_data = db[location]
                run_data = loc_data.get("runs", {}).get(gfs_init, {})
                results['fairfax'] = {
                    'status': 'cached',
                    'message': message,
                    'init_time': gfs_init
                }
            else:
                # Fetch new data
                forecast_hours = list(range(0, min(days * 24, 360) + 1, 6))
                gfs_data = fetch_gfs_data(region, forecast_hours)
                aifs_data = fetch_aifs_data(region, forecast_hours)
                ifs_data = fetch_ifs_data(region, forecast_hours)
                observed = fetch_observations(gfs_data.get("times", []), location)
                verification = calculate_all_verification(gfs_data, aifs_data, observed, ifs_data)
                save_forecast_data(location, gfs_data, aifs_data, observed, verification, ifs_data)

                results['fairfax'] = {
                    'status': 'synced',
                    'message': 'Fetched new forecast data',
                    'init_time': gfs_init
                }
                logger.info(f"Fairfax data synced: {gfs_init}")
        except Exception as e:
            logger.error(f"Fairfax sync failed: {e}")
            results['fairfax'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"Fairfax: {str(e)}")

        # 2. Sync ASOS forecasts
        logger.info("Master sync: Fetching ASOS station forecasts...")
        try:
            stations_list = asos.fetch_all_stations()
            stations = [
                {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
                for s in stations_list
            ]

            forecast_hours = list(range(0, 361, 6))
            asos_results = {}

            # Fetch GFS
            gfs_model = GFSModel()
            gfs_init = gfs_model.get_latest_init_time()

            if not force and check_asos_model_exists(gfs_init, 'gfs'):
                logger.info(f"ASOS GFS already synced for {gfs_init}")
                asos_results['gfs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                gfs_forecasts = fetch_asos_forecasts_for_model('gfs', gfs_init, forecast_hours, stations)
                asos.store_asos_forecasts(gfs_init, forecast_hours, 'gfs', gfs_forecasts)
                asos_results['gfs'] = {'status': 'synced', 'stations': len(gfs_forecasts)}

            # Fetch AIFS
            aifs_model = AIFSModel()
            aifs_init = aifs_model.get_latest_init_time()

            if not force and check_asos_model_exists(aifs_init, 'aifs'):
                logger.info(f"ASOS AIFS already synced for {aifs_init}")
                asos_results['aifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                aifs_forecasts = fetch_asos_forecasts_for_model('aifs', aifs_init, forecast_hours, stations)
                asos.store_asos_forecasts(aifs_init, forecast_hours, 'aifs', aifs_forecasts)
                asos_results['aifs'] = {'status': 'synced', 'stations': len(aifs_forecasts)}

            # Fetch IFS
            ifs_model = IFSModel()
            ifs_init = ifs_model.get_latest_init_time()

            if not force and check_asos_model_exists(ifs_init, 'ifs'):
                logger.info(f"ASOS IFS already synced for {ifs_init}")
                asos_results['ifs'] = {'status': 'skipped', 'reason': 'already synced'}
            else:
                ifs_forecasts = fetch_asos_forecasts_for_model('ifs', ifs_init, forecast_hours, stations)
                asos.store_asos_forecasts(ifs_init, forecast_hours, 'ifs', ifs_forecasts)
                asos_results['ifs'] = {'status': 'synced', 'stations': len(ifs_forecasts)}

            # Fetch observations
            obs_count = asos.fetch_and_store_observations()
            asos_results['observations'] = {'status': 'synced', 'count': obs_count}

            results['asos'] = {
                'status': 'success',
                'models': asos_results,
                'station_count': len(stations)
            }
            logger.info(f"ASOS sync complete: {len(stations)} stations")

        except Exception as e:
            logger.error(f"ASOS sync failed: {e}")
            results['asos'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"ASOS: {str(e)}")

        # Check if any critical errors occurred
        if results['errors']:
            results['success'] = False

        return jsonify(results)

    except Exception as e:
        logger.error(f"Master sync error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'fairfax': results.get('fairfax', {}),
            'asos': results.get('asos', {})
        })


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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
