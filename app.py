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

from gfs import GFSModel
from ecmwf_aifs import AIFSModel, AIFS_VARIABLES
from ecmwf_ifs import IFSModel, IFS_VARIABLES
import weatherlink

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
    "New York, NY": (-74.25, -73.75, 40.5, 41.0),
    "Los Angeles, CA": (-118.5, -118.0, 33.75, 34.25),
    "Chicago, IL": (-88.0, -87.5, 41.75, 42.25),
    "Houston, TX": (-95.5, -95.0, 29.5, 30.0),
    "Phoenix, AZ": (-112.25, -111.75, 33.25, 33.75),
    "Miami, FL": (-80.5, -80.0, 25.5, 26.0),
    "Seattle, WA": (-122.5, -122.0, 47.25, 47.75),
    "Denver, CO": (-105.0, -104.5, 39.5, 40.0),
    "Boston, MA": (-71.25, -70.75, 42.25, 42.75),
}


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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
