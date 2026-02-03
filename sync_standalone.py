#!/usr/bin/env python3
"""
Standalone sync script for weather model data.
Can be run directly without Flask server.

Usage:
    python sync_standalone.py [--force] [--init-hour HOUR]

Options:
    --force         Force refresh even if data already exists
    --init-hour     Specific init hour (0, 6, 12, or 18). If not specified, uses latest available.
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import json

import numpy as np

from gfs import GFSModel
from ecmwf_aifs import AIFSModel, AIFS_VARIABLES
from ecmwf_ifs import IFSModel, IFS_VARIABLES
import weatherlink
import asos
import rossby_waves

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Single JSON file for storing all forecast data
FORECASTS_FILE = Path(__file__).parent / "forecasts.json"

# Variable definitions
class Variable:
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

Z500_GFS = Variable(
    name="z500",
    display_name="500 hPa Geopotential Height",
    units="dm",
    herbie_search=":HGT:500 mb:",
    category="upper_air",
    colormap="viridis",
    contour_levels=list(range(480, 600, 6))
)

# Preset locations
LOCATIONS = {
    "Fairfax, VA": (-77.5, -77.0, 38.5, 39.0),
}

# CONUS region for ASOS
CONUS_BOUNDS = (-130.0, -60.0, 20.0, 55.0)


def load_forecasts_db():
    """Load the forecasts database from JSON file."""
    if FORECASTS_FILE.exists():
        try:
            with open(FORECASTS_FILE) as f:
                data = json.load(f)
                return migrate_db_format(data)
        except Exception as e:
            logger.warning(f"Error loading forecasts.json: {e}")
    return {}


def migrate_db_format(data: dict) -> dict:
    """Migrate from old flat format to new historical runs format."""
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
            migrated[location] = {"runs": {}, "latest_run": None}

    return migrated


def save_forecasts_db(data):
    """Save the forecasts database to JSON file."""
    with open(FORECASTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved forecasts to {FORECASTS_FILE}")


def fetch_gfs_data(region, forecast_hours, init_hour: Optional[int] = None):
    """Fetch temperature, precipitation, and MSLP data from GFS."""
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

        # Precipitation
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
        "ridges": [],
        "troughs": []
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
            z500_waves_forecast["ridges"].append(wave_metrics.get('ridges'))
            z500_waves_forecast["troughs"].append(wave_metrics.get('troughs'))

            # Store F000 as current wave number
            if fhr == 0:
                z500_waves = wave_metrics
                logger.info(f"GFS wave number: {wave_metrics.get('wave_number')} ({wave_metrics.get('ridges')} ridges, {wave_metrics.get('troughs')} troughs)")

        except Exception as e:
            logger.warning(f"Wave analysis failed for GFS F{fhr:03d}: {e}")
            # Add None values to maintain array alignment
            valid_time = (init_time + timedelta(hours=fhr)).isoformat()
            z500_waves_forecast["times"].append(valid_time)
            z500_waves_forecast["wave_numbers"].append(None)
            z500_waves_forecast["ridges"].append(None)
            z500_waves_forecast["troughs"].append(None)

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
    """Fetch temperature, precipitation, and MSLP data from ECMWF AIFS."""
    aifs_model = AIFSModel()

    if init_hour is not None:
        init_time = aifs_model.get_init_time_for_hour(init_hour)
    else:
        init_time = aifs_model.get_latest_init_time()

    temp_var = AIFS_VARIABLES["t2m"]
    precip_var = AIFS_VARIABLES["tp"]
    mslp_var = AIFS_VARIABLES["mslp"]

    temps = []
    cumulative_precips = []
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

        # Precipitation (cumulative)
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
            precips.append(cum)
        else:
            prev_cum = cumulative_precips[i - 1]
            if prev_cum is not None:
                interval = cum - prev_cum
                precips.append(max(0.0, interval))
            else:
                precips.append(None)

    # Calculate wave number forecast at 24-hour intervals for 15 days
    z500_var = AIFS_VARIABLES.get("z500")
    z500_waves = None
    z500_waves_forecast = {
        "times": [],
        "wave_numbers": [],
        "ridges": [],
        "troughs": []
    }

    # Calculate at 24-hour intervals up to 360 hours (15 days)
    wave_forecast_hours = list(range(0, 361, 24))
    global_region = Region("Global", (-180, 180, 20, 70))

    if z500_var:
        for fhr in wave_forecast_hours:
            try:
                logger.info(f"Calculating Rossby wave number for AIFS F{fhr:03d}")
                z500_data = aifs_model.fetch_data(z500_var, init_time, fhr, global_region)
                wave_metrics = rossby_waves.calculate_wave_number(z500_data, latitude=55.0)

                # Store forecast time series
                valid_time = (init_time + timedelta(hours=fhr)).isoformat()
                z500_waves_forecast["times"].append(valid_time)
                z500_waves_forecast["wave_numbers"].append(wave_metrics.get('wave_number'))
                z500_waves_forecast["ridges"].append(wave_metrics.get('ridges'))
                z500_waves_forecast["troughs"].append(wave_metrics.get('troughs'))

                # Store F000 as current wave number
                if fhr == 0:
                    z500_waves = wave_metrics
                    logger.info(f"AIFS wave number: {wave_metrics.get('wave_number')} ({wave_metrics.get('ridges')} ridges, {wave_metrics.get('troughs')} troughs)")

            except Exception as e:
                logger.warning(f"Wave analysis failed for AIFS F{fhr:03d}: {e}")
                # Add None values to maintain array alignment
                valid_time = (init_time + timedelta(hours=fhr)).isoformat()
                z500_waves_forecast["times"].append(valid_time)
                z500_waves_forecast["wave_numbers"].append(None)
                z500_waves_forecast["ridges"].append(None)
                z500_waves_forecast["troughs"].append(None)

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
    """Fetch temperature, precipitation, and MSLP data from ECMWF IFS."""
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

        # Precipitation
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

    # Calculate wave number forecast at 24-hour intervals (IFS: up to 240 hours / 10 days)
    z500_var = IFS_VARIABLES.get("z500")
    z500_waves = None
    z500_waves_forecast = {
        "times": [],
        "wave_numbers": [],
        "ridges": [],
        "troughs": []
    }

    # Calculate at 24-hour intervals up to 240 hours (10 days - IFS limit)
    wave_forecast_hours = list(range(0, min(241, 361), 24))
    global_region = Region("Global", (-180, 180, 20, 70))

    if z500_var:
        for fhr in wave_forecast_hours:
            try:
                logger.info(f"Calculating Rossby wave number for IFS F{fhr:03d}")
                z500_data = ifs_model.fetch_data(z500_var, init_time, fhr, global_region)
                wave_metrics = rossby_waves.calculate_wave_number(z500_data, latitude=55.0)

                # Store forecast time series
                valid_time = (init_time + timedelta(hours=fhr)).isoformat()
                z500_waves_forecast["times"].append(valid_time)
                z500_waves_forecast["wave_numbers"].append(wave_metrics.get('wave_number'))
                z500_waves_forecast["ridges"].append(wave_metrics.get('ridges'))
                z500_waves_forecast["troughs"].append(wave_metrics.get('troughs'))

                # Store F000 as current wave number
                if fhr == 0:
                    z500_waves = wave_metrics
                    logger.info(f"IFS wave number: {wave_metrics.get('wave_number')} ({wave_metrics.get('ridges')} ridges, {wave_metrics.get('troughs')} troughs)")

            except Exception as e:
                logger.warning(f"Wave analysis failed for IFS F{fhr:03d}: {e}")
                # Add None values to maintain array alignment
                valid_time = (init_time + timedelta(hours=fhr)).isoformat()
                z500_waves_forecast["times"].append(valid_time)
                z500_waves_forecast["wave_numbers"].append(None)
                z500_waves_forecast["ridges"].append(None)
                z500_waves_forecast["troughs"].append(None)

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
    """Fetch observed data matching forecast times."""
    if location_name != "Fairfax, VA":
        return {}

    try:
        weatherlink.fetch_missing_data(silent=True)
        return weatherlink.get_observations_for_forecast_times(forecast_times)
    except Exception as e:
        logger.warning(f"Error fetching observations: {e}")
        return {}


def calculate_verification_metrics(forecast_values: list, observed_values: list, forecast_times: list[str]) -> dict:
    """Calculate verification metrics (MAE, bias) for forecasts vs observations."""
    now = datetime.now(timezone.utc)
    errors = []

    for i, time_str in enumerate(forecast_times):
        try:
            valid_time = datetime.fromisoformat(time_str)
        except (ValueError, TypeError):
            continue

        if valid_time >= now:
            continue

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
    """Calculate verification metrics for all variables."""
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


def save_forecast_data(location_name, gfs_data, aifs_data, observed=None, verification=None, ifs_data=None):
    """Save forecast data to the central JSON file."""
    db = load_forecasts_db()

    # Initialize location if needed
    if location_name not in db:
        db[location_name] = {"runs": {}, "latest_run": None}

    # Use GFS init_time as the run key
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

    if ifs_data:
        run_data["ifs"] = ifs_data

    if observed:
        run_data["observed"] = observed

    if verification:
        run_data["verification"] = verification

    # Store the run
    db[location_name]["runs"][run_key] = run_data
    db[location_name]["latest_run"] = run_key

    save_forecasts_db(db)
    logger.info(f"Saved run {run_key} for {location_name} (total runs: {len(db[location_name]['runs'])})")
    return str(FORECASTS_FILE)


def check_if_already_fetched(location_name, gfs_init, aifs_init):
    """Check if we already have data for these init times."""
    db = load_forecasts_db()

    if location_name not in db:
        return False, "No cached data for this location"

    runs = db[location_name].get("runs", {})

    if gfs_init in runs:
        return True, f"Already have this run (GFS: {gfs_init}, AIFS: {aifs_init})"

    return False, f"New model run available (GFS: {gfs_init}, AIFS: {aifs_init})"


def extract_model_at_stations(model_data_array, stations):
    """Extract model values at station locations via bilinear interpolation."""
    values = []
    lat_name = 'latitude' if 'latitude' in model_data_array.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in model_data_array.coords else 'lon'

    lats = model_data_array[lat_name].values
    lons = model_data_array[lon_name].values

    lon_is_0_360 = lons.min() >= 0 and lons.max() > 180

    for station in stations:
        try:
            slat = station['lat']
            slon = station['lon']

            if lon_is_0_360 and slon < 0:
                slon = slon + 360

            if slat < lats.min() or slat > lats.max():
                values.append(None)
                continue
            if slon < lons.min() or slon > lons.max():
                values.append(None)
                continue

            val = model_data_array.interp(
                {lat_name: slat, lon_name: slon},
                method='linear'
            ).values

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
    """Fetch forecasts at all ASOS station locations for a model."""
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
    for i, hour in enumerate(model_hours, 1):
        progress_pct = int((i / len(model_hours)) * 100)
        logger.info(f"  [{model_name.upper()}] Extracting F{hour:03d} ({i}/{len(model_hours)}, {progress_pct}%)")

        try:
            temp_data = model.fetch_data(temp_var, init_time, hour, region)
            temp_vals = extract_model_at_stations(temp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} temp F{hour} failed: {e}")
            temp_vals = [None] * len(stations)

        try:
            mslp_data = model.fetch_data(mslp_var, init_time, hour, region)
            mslp_vals = extract_model_at_stations(mslp_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} mslp F{hour} failed: {e}")
            mslp_vals = [None] * len(stations)

        try:
            precip_data = model.fetch_data(precip_var, init_time, hour, region)
            precip_vals = extract_model_at_stations(precip_data, stations)
        except Exception as e:
            logger.warning(f"{model_name.upper()} precip F{hour} failed: {e}")
            precip_vals = [None] * len(stations)

        # Store values
        for j, station in enumerate(stations):
            sid = station['station_id']
            station_forecasts[sid]['temps'].append(temp_vals[j])
            station_forecasts[sid]['mslps'].append(mslp_vals[j])
            station_forecasts[sid]['precips'].append(precip_vals[j])

    # Pad IFS with None for hours beyond its range
    if model_name.lower() == 'ifs':
        pad_count = len(forecast_hours) - len(model_hours)
        for sid in station_forecasts:
            station_forecasts[sid]['temps'].extend([None] * pad_count)
            station_forecasts[sid]['mslps'].extend([None] * pad_count)
            station_forecasts[sid]['precips'].extend([None] * pad_count)

    return init_time, station_forecasts


def check_asos_model_exists(init_time, model_name):
    """Check if we already have ASOS forecasts for this model and init time."""
    db = asos.load_asos_forecasts_db()
    run_key = init_time.isoformat()
    runs = db.get("runs", {})

    if run_key not in runs:
        return False

    run_data = runs[run_key]
    model_data = run_data.get(model_name.lower())

    if model_data and len(model_data) > 100:
        return True

    return False


def sync_fairfax(force: bool = False, init_hour: Optional[int] = None):
    """Sync Fairfax verification location."""
    logger.info("=" * 60)
    logger.info("SYNCING FAIRFAX VERIFICATION LOCATION")
    if init_hour is not None:
        logger.info(f"Using model init hour: {init_hour:02d}Z")
    logger.info("=" * 60)

    location = "Fairfax, VA"
    days = 15

    bounds = LOCATIONS[location]
    region = Region(location, bounds)

    # Get init times
    gfs_model = GFSModel()
    aifs_model = AIFSModel()

    if init_hour is not None:
        gfs_init = gfs_model.get_init_time_for_hour(init_hour)
        aifs_init = aifs_model.get_init_time_for_hour(init_hour)
    else:
        gfs_init = gfs_model.get_latest_init_time()
        aifs_init = aifs_model.get_latest_init_time()

    gfs_init_str = gfs_init.isoformat()
    aifs_init_str = aifs_init.isoformat()

    # Check if already fetched
    already_fetched, message = check_if_already_fetched(location, gfs_init_str, aifs_init_str)

    if already_fetched and not force:
        logger.info(f"Fairfax data already cached: {message}")
        return

    # Fetch new data
    logger.info(f"Fetching new forecast data for {location}...")
    forecast_hours = list(range(0, min(days * 24, 360) + 1, 6))

    logger.info(f"Fetching GFS model data (init: {gfs_init_str})...")
    gfs_data = fetch_gfs_data(region, forecast_hours, init_hour)
    logger.info("GFS data fetched successfully")

    logger.info(f"Fetching ECMWF AIFS model data (init: {aifs_init_str})...")
    aifs_data = fetch_aifs_data(region, forecast_hours, init_hour)
    logger.info("AIFS data fetched successfully")

    logger.info("Fetching ECMWF IFS model data...")
    ifs_data = fetch_ifs_data(region, forecast_hours, init_hour)
    logger.info("IFS data fetched successfully")

    logger.info("Fetching WeatherLink observations...")
    observed = fetch_observations(gfs_data.get("times", []), location)
    if observed:
        logger.info("Observations fetched successfully")

    logger.info("Calculating verification metrics...")
    verification = calculate_all_verification(gfs_data, aifs_data, observed, ifs_data)

    logger.info("Saving forecast data to forecasts.json...")
    save_forecast_data(location, gfs_data, aifs_data, observed, verification, ifs_data)
    logger.info("Fairfax forecast data saved successfully")


def sync_asos(force: bool = False, init_hour: Optional[int] = None):
    """Sync ASOS station network."""
    logger.info("-" * 60)
    logger.info("SYNCING ASOS STATION NETWORK")
    logger.info("-" * 60)

    # Get stations
    logger.info("Loading ASOS station list...")
    stations_list = asos.fetch_all_stations()
    stations = [
        {'station_id': s.station_id, 'lat': s.lat, 'lon': s.lon, 'name': s.name}
        for s in stations_list
    ]
    logger.info(f"Found {len(stations)} ASOS stations across CONUS")

    forecast_hours = list(range(0, 361, 6))

    # Fetch GFS
    logger.info(f"Extracting GFS forecasts at {len(stations)} ASOS stations...")
    gfs_init, gfs_forecasts = fetch_asos_forecasts_for_model('gfs', forecast_hours, stations, init_hour)

    if not force and check_asos_model_exists(gfs_init, 'gfs'):
        logger.info(f"GFS ASOS data already synced for {gfs_init} - skipping")
    else:
        asos.store_asos_forecasts(gfs_init, forecast_hours, 'gfs', gfs_forecasts)
        logger.info(f"GFS ASOS data synced for {len(gfs_forecasts)} stations")

    # Fetch AIFS
    logger.info(f"Extracting AIFS forecasts at {len(stations)} ASOS stations...")
    aifs_init, aifs_forecasts = fetch_asos_forecasts_for_model('aifs', forecast_hours, stations, init_hour)

    if not force and check_asos_model_exists(aifs_init, 'aifs'):
        logger.info(f"AIFS ASOS data already synced for {aifs_init} - skipping")
    else:
        asos.store_asos_forecasts(aifs_init, forecast_hours, 'aifs', aifs_forecasts)
        logger.info(f"AIFS ASOS data synced for {len(aifs_forecasts)} stations")

    # Fetch IFS
    logger.info(f"Extracting IFS forecasts at {len(stations)} ASOS stations...")
    ifs_init, ifs_forecasts = fetch_asos_forecasts_for_model('ifs', forecast_hours, stations, init_hour)

    if not force and check_asos_model_exists(ifs_init, 'ifs'):
        logger.info(f"IFS ASOS data already synced for {ifs_init} - skipping")
    else:
        asos.store_asos_forecasts(ifs_init, forecast_hours, 'ifs', ifs_forecasts)
        logger.info(f"IFS ASOS data synced for {len(ifs_forecasts)} stations")

    # Fetch observations
    logger.info("Fetching ASOS observations from Iowa Environmental Mesonet...")
    obs_count = asos.fetch_and_store_observations()
    logger.info(f"Fetched {obs_count} ASOS observations")

    logger.info(f"ASOS sync complete: {len(stations)} stations processed")


def main():
    """Main sync function."""
    parser = argparse.ArgumentParser(
        description='Standalone sync script for weather model data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if data already exists'
    )
    parser.add_argument(
        '--init-hour',
        type=int,
        choices=[0, 6, 12, 18],
        help='Specific init hour (0, 6, 12, or 18). If not specified, uses latest available.'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STARTING MASTER SYNC")
    logger.info("=" * 60)

    try:
        # Sync Fairfax
        sync_fairfax(force=args.force, init_hour=args.init_hour)

        # Sync ASOS
        sync_asos(force=args.force, init_hour=args.init_hour)

        logger.info("=" * 60)
        logger.info("SYNC COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        logger.error("=" * 60)
        logger.error("SYNC FAILED")
        logger.error("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
