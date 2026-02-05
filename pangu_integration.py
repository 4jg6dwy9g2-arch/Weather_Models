"""
PanguWeather Flask App
Run and visualize PanguWeather AI forecasts
"""

from flask import Blueprint, render_template, jsonify, request, Response
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import logging
import json
import xarray as xr
import numpy as np
import queue
import threading
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tempfile
import shutil
import imageio
from PIL import Image
from eccodes import (
    codes_grib_new_from_file,
    codes_get_array,
    codes_set_array,
    codes_set,
    codes_write,
    codes_release,
    codes_get,
    CODES_MISSING_DOUBLE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pangu_bp = Blueprint("pangu", __name__)

# Paths
PANGU_BASE_DIR = Path("/Users/kennypratt/Documents/Weather_Models/pangu")
BASE_DIR = PANGU_BASE_DIR
FORECASTS_DIR = BASE_DIR / "pangu_forecasts"
FORECASTS_DB = BASE_DIR / "pangu_runs.json"
FORECASTS_DIR.mkdir(exist_ok=True)

# Regional bounds (west, east, south, north)
REGIONS = {
    "Global": (-180, 180, -90, 90),
    "North America": (-170, -50, 15, 75),
    "South America": (-85, -30, -60, 15),
    "Europe": (-15, 45, 35, 72),
    "Asia": (40, 150, 5, 75),
    "Africa": (-20, 55, -40, 40),
    "Australia": (110, 160, -45, -10),
    "Pacific": (120, -70, -60, 60),
    "Atlantic": (-100, 20, -60, 70)
}

# Queue for streaming logs to clients
log_queues = []
log_lock = threading.Lock()


class StreamingLogHandler(logging.Handler):
    """Custom log handler for SSE streaming."""

    def emit(self, record):
        try:
            msg = self.format(record)
            log_type = 'info'

            if record.levelno >= logging.ERROR:
                log_type = 'error'
            elif record.levelno >= logging.WARNING:
                log_type = 'warning'
            elif 'complete' in msg.lower() or 'success' in msg.lower():
                log_type = 'success'

            with log_lock:
                for q in log_queues[:]:
                    try:
                        q.put_nowait({'message': msg, 'type': log_type})
                    except queue.Full:
                        log_queues.remove(q)
        except Exception:
            pass


def broadcast_log(message, log_type='info'):
    """Broadcast a message to all log listeners."""
    with log_lock:
        for q in log_queues[:]:
            try:
                q.put_nowait({'message': message, 'type': log_type})
            except queue.Full:
                log_queues.remove(q)


def apply_climate_perturbation(input_grib, output_grib, temp_offset,
                               adjust_humidity=True, adjust_pressure=True):
    """
    Apply systematic climate change perturbation to initial conditions.

    Implements pseudo-global warming (PGW) by adding temperature offset
    and adjusting humidity using Clausius-Clapeyron relation.

    Args:
        input_grib: Path to input GRIB file
        output_grib: Path to save perturbed GRIB file
        temp_offset: Temperature offset in Celsius (e.g., 2.0 for +2°C)
        adjust_humidity: Whether to scale humidity with temperature
        adjust_pressure: Whether to adjust surface pressure

    Returns:
        True if successful, False otherwise
    """
    try:
        broadcast_log(f"Applying +{temp_offset}°C climate perturbation...", 'info')

        # Clausius-Clapeyron: ~7% more moisture per 1°C warming
        # q_new = q_old * exp(0.067 * ΔT)
        humidity_scale = np.exp(0.067 * temp_offset) if adjust_humidity else 1.0

        # Surface pressure adjustment (small): ~0.3% per °C from thermal expansion
        pressure_offset = -temp_offset * 0.003 if adjust_pressure else 0.0

        perturbed_count = 0
        total_count = 0

        # Surface variables come from step 6, upper air from step 0
        surface_vars = {'2t', 't2m', 'msl', 'prmsl', '10u', '10v'}
        written_vars = set()

        with open(input_grib, 'rb') as fin:
            with open(output_grib, 'wb') as fout:

                while True:
                    gid = codes_grib_new_from_file(fin)
                    if gid is None:
                        break

                    try:
                        total_count += 1
                        step = codes_get(gid, 'step')
                        var_name = codes_get(gid, 'shortName')

                        # Create unique key for deduplication
                        try:
                            level = codes_get(gid, 'level')
                            var_key = f"{var_name}_{level}"
                        except:
                            var_key = var_name

                        values = codes_get_array(gid, 'values')

                        # Apply perturbations
                        if var_name in ['t', 't2m', '2t']:
                            original_mean = values.mean()
                            perturbed_values = values + temp_offset
                            new_mean = perturbed_values.mean()
                            codes_set_array(gid, 'values', perturbed_values)
                            perturbed_count += 1
                            if perturbed_count <= 3:
                                broadcast_log(f"  {var_name} (step {step}): {original_mean:.1f}K -> {new_mean:.1f}K (+{temp_offset}K)", 'info')

                        elif var_name == 'q' and adjust_humidity:
                            perturbed_values = values * humidity_scale
                            codes_set_array(gid, 'values', perturbed_values)
                            perturbed_count += 1

                        elif var_name in ['msl', 'prmsl'] and adjust_pressure:
                            perturbed_values = values * (1 + pressure_offset)
                            codes_set_array(gid, 'values', perturbed_values)
                            perturbed_count += 1

                        # Only write each field once: surface from step 6, upper air from step 0
                        should_write = False
                        if var_name in surface_vars:
                            should_write = (step == 6 and var_key not in written_vars)
                        else:
                            should_write = (step == 0 and var_key not in written_vars)

                        if should_write:
                            if step != 0:
                                codes_set(gid, 'step', 0)
                                codes_set(gid, 'stepRange', 0)
                            codes_write(gid, fout)
                            written_vars.add(var_key)

                    finally:
                        codes_release(gid)

        if adjust_humidity:
            moisture_increase = (humidity_scale - 1) * 100
            broadcast_log(f"  Temperature: +{temp_offset}°C", 'info')
            broadcast_log(f"  Moisture: +{moisture_increase:.1f}%", 'info')
        else:
            broadcast_log(f"  Temperature: +{temp_offset}°C (humidity unchanged)", 'info')

        broadcast_log(f"Climate perturbation applied to {perturbed_count}/{total_count} fields", 'success')

        return True

    except Exception as e:
        logger.error(f"Error applying climate perturbation: {e}")
        import traceback
        traceback.print_exc()
        broadcast_log(f"Error: {str(e)}", 'error')
        return False


def perturb_grib_file(input_grib, output_grib, scale='medium', seed=None):
    """
    Apply Gaussian noise perturbations to initial conditions in a GRIB file.

    Uses eccodes to read/write GRIB files and adds random Gaussian noise
    to create ensemble members.

    Args:
        input_grib: Path to input GRIB file
        output_grib: Path to save perturbed GRIB file
        scale: Perturbation magnitude ('small', 'medium', 'large')
        seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    try:
        if seed is not None:
            np.random.seed(seed)

        # Define perturbation standard deviations for different variables
        # Based on typical analysis error std devs
        perturbation_scales = {
            'small': {
                't': 0.5,      # 0.5 K temperature
                'u': 0.5,      # 0.5 m/s wind
                'v': 0.5,
                'q': 0.00005,  # 0.05 g/kg humidity
                'z': 5.0,      # 5 m geopotential
                'gh': 5.0,
                'msl': 50.0    # 50 Pa pressure
            },
            'medium': {
                't': 1.0,      # 1.0 K temperature
                'u': 1.0,      # 1.0 m/s wind
                'v': 1.0,
                'q': 0.0001,   # 0.1 g/kg humidity
                'z': 10.0,     # 10 m geopotential
                'gh': 10.0,
                'msl': 100.0   # 100 Pa pressure
            },
            'large': {
                't': 2.0,      # 2.0 K temperature
                'u': 2.0,      # 2.0 m/s wind
                'v': 2.0,
                'q': 0.0002,   # 0.2 g/kg humidity
                'z': 20.0,     # 20 m geopotential
                'gh': 20.0,
                'msl': 200.0   # 200 Pa pressure
            }
        }

        std_devs = perturbation_scales.get(scale, perturbation_scales['medium'])

        broadcast_log(f"Applying {scale} Gaussian perturbations...", 'info')

        perturbed_count = 0
        total_count = 0

        # Surface variables come from step 6, upper air from step 0
        surface_vars = {'2t', 't2m', 'msl', 'prmsl', '10u', '10v'}
        written_vars = set()

        # Open input and output GRIB files
        with open(input_grib, 'rb') as fin:
            with open(output_grib, 'wb') as fout:

                # Process each GRIB message
                while True:
                    gid = codes_grib_new_from_file(fin)
                    if gid is None:
                        break

                    try:
                        total_count += 1
                        step = codes_get(gid, 'step')
                        var_name = codes_get(gid, 'shortName')

                        # Create unique key for deduplication
                        try:
                            level = codes_get(gid, 'level')
                            var_key = f"{var_name}_{level}"
                        except:
                            var_key = var_name

                        values = codes_get_array(gid, 'values')

                        # Check if we should perturb this variable
                        if var_name in std_devs:
                            # Generate Gaussian noise
                            noise = np.random.normal(0, std_devs[var_name], values.shape)

                            # Add noise to values
                            perturbed_values = values + noise

                            # Set the perturbed values back
                            codes_set_array(gid, 'values', perturbed_values)

                            perturbed_count += 1

                        # Only write each field once: surface from step 6, upper air from step 0
                        should_write = False
                        if var_name in surface_vars:
                            should_write = (step == 6 and var_key not in written_vars)
                        else:
                            should_write = (step == 0 and var_key not in written_vars)

                        if should_write:
                            if step != 0:
                                codes_set(gid, 'step', 0)
                                codes_set(gid, 'stepRange', 0)
                            codes_write(gid, fout)
                            written_vars.add(var_key)

                    finally:
                        codes_release(gid)

        broadcast_log(f"Perturbed {perturbed_count}/{total_count} fields", 'info')
        logger.info(f"Successfully perturbed {perturbed_count} out of {total_count} fields")

        return True

    except Exception as e:
        logger.error(f"Error perturbing GRIB file: {e}")
        import traceback
        traceback.print_exc()
        broadcast_log(f"Error perturbing file: {str(e)}", 'error')
        return False


# Add streaming handler
streaming_handler = StreamingLogHandler()
streaming_handler.setLevel(logging.INFO)
streaming_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(streaming_handler)


def load_runs_db():
    """Load the runs database."""
    if FORECASTS_DB.exists():
        try:
            with open(FORECASTS_DB) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading runs database: {e}")
    return {"runs": []}


def save_runs_db(data):
    """Save the runs database."""
    with open(FORECASTS_DB, 'w') as f:
        json.dump(data, f, indent=2)


def run_forecast(date_str, time_str, lead_time, model="panguweather",
                 perturbation_type='none', ensemble_members=1, perturbation_scale='medium',
                 temp_offset=0, adjust_humidity=True, adjust_pressure=True):
    """
    Run weather forecast with selected AI model, optionally with ensemble perturbations.

    Args:
        date_str: Date in YYYY-MM-DD format
        time_str: Time (00, 06, 12, or 18)
        lead_time: Forecast length in hours
        model: Model name ('panguweather' or 'graphcast')
        enable_perturbations: Whether to create ensemble with perturbed ICs
        ensemble_members: Number of ensemble members (if perturbations enabled)
        perturbation_scale: Scale of perturbations ('small', 'medium', 'large')

    Returns:
        dict with run info and status
    """
    try:
        model_display = {
            "panguweather": "PanguWeather",
            "graphcast": "GraphCast",
            "fourcastnet": "FourCastNet"
        }.get(model, model.capitalize())

        if perturbation_type == 'ensemble' and ensemble_members > 1:
            # Run ensemble forecast with random perturbations
            return run_ensemble_forecast(date_str, time_str, lead_time, model,
                                        ensemble_members, perturbation_scale)
        elif perturbation_type == 'climate':
            # Run climate scenario forecast
            return run_climate_scenario(date_str, time_str, lead_time, model,
                                       temp_offset, adjust_humidity, adjust_pressure)

        # Single deterministic forecast
        run_id = f"{date_str}T{time_str}:00:00_F{lead_time:03d}_{model}"
        output_file = FORECASTS_DIR / f"{run_id}.grib"

        broadcast_log(f"Starting {model_display} forecast", 'info')
        broadcast_log(f"  Date: {date_str} {time_str}:00 UTC", 'info')
        broadcast_log(f"  Lead time: {lead_time} hours", 'info')
        broadcast_log(f"  Model: {model_display}", 'info')
        broadcast_log(f"  Output: {output_file.name}", 'info')

        # Build command
        cmd = [
            "ai-models",
            model,
            "--input", "ecmwf-open-data",
            "--date", date_str.replace("-", ""),
            "--time", time_str,
            "--lead-time", str(lead_time),
            "--path", str(output_file),  # File path for ecmwf-open-data
        ]

        broadcast_log("Running model...", 'info')
        logger.info(f"Command: {' '.join(cmd)}")

        # Run the model
        start_time = datetime.now(timezone.utc)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PANGU_BASE_DIR)
        )
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()

        broadcast_log(f"Model run complete in {elapsed:.1f} seconds", 'success')

        # Extract data from GRIB
        broadcast_log("Processing forecast output...", 'info')
        forecast_data = extract_grib_data(output_file, date_str, time_str, lead_time)

        # Save run info
        run_info = {
            "run_id": run_id,
            "model": model,
            "model_name": model_display,
            "init_date": date_str,
            "init_time": time_str,
            "lead_time": lead_time,
            "output_file": str(output_file),
            "run_time": start_time.isoformat(),
            "elapsed_seconds": elapsed,
            "data": forecast_data
        }

        db = load_runs_db()
        db["runs"].insert(0, run_info)  # Add to front
        db["runs"] = db["runs"][:50]  # Keep last 50 runs
        save_runs_db(db)

        broadcast_log(f"Forecast saved: {run_id}", 'success')

        return {
            "success": True,
            "run_id": run_id,
            "run_info": run_info
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"Model run failed: {e.stderr if e.stderr else str(e)}"
        logger.error(error_msg)
        broadcast_log(error_msg, 'error')
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        broadcast_log(error_msg, 'error')
        return {"success": False, "error": error_msg}


def run_climate_scenario(date_str, time_str, lead_time, model, temp_offset,
                         adjust_humidity, adjust_pressure):
    """
    Run climate change scenario forecast with systematic perturbation.

    Downloads initial conditions, applies temperature offset and humidity scaling,
    then runs the model to see how the weather pattern evolves in a warmer climate.
    """
    try:
        model_display = "PanguWeather" if model == "panguweather" else "GraphCast"
        scenario_id = f"{date_str}T{time_str}:00:00_F{lead_time:03d}_{model}_CLIMATE{temp_offset:+.1f}C"

        broadcast_log(f"Starting {model_display} climate scenario", 'info')
        broadcast_log(f"  Temperature offset: +{temp_offset}°C", 'info')
        broadcast_log(f"  Date: {date_str} {time_str}:00 UTC", 'info')

        # Step 1: Download initial conditions
        broadcast_log("Downloading initial conditions...", 'info')
        temp_dir = Path(tempfile.mkdtemp(prefix="pangu_climate_"))
        base_ic_file = temp_dir / "initial_conditions.grib"

        cmd_download = [
            "ai-models",
            model,
            "--input", "ecmwf-open-data",
            "--date", date_str.replace("-", ""),
            "--time", time_str,
            "--lead-time", "6",
            "--path", str(base_ic_file),  # File path for download
        ]

        result = subprocess.run(cmd_download, capture_output=True, text=True, cwd=str(PANGU_BASE_DIR))
        if result.returncode != 0:
            broadcast_log(f"Download failed with exit code {result.returncode}", 'error')
            if result.stdout:
                broadcast_log("STDOUT:", 'error')
                broadcast_log(result.stdout, 'error')
            if result.stderr:
                broadcast_log("STDERR:", 'error')
                broadcast_log(result.stderr, 'error')
            # Cleanup temp dir on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to download initial conditions: {result.stderr or result.stdout or 'Unknown error'}")

        broadcast_log("Initial conditions downloaded", 'success')

        # Step 2: Apply climate perturbation
        perturbed_ic = temp_dir / "ic_climate.grib"
        success = apply_climate_perturbation(base_ic_file, perturbed_ic, temp_offset,
                                            adjust_humidity, adjust_pressure)
        if not success:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Failed to apply climate perturbation")

        # Step 3: Run model with perturbed ICs
        broadcast_log("Running climate scenario forecast...", 'info')
        output_file = FORECASTS_DIR / f"{scenario_id}.grib"

        cmd_run = [
            "ai-models",
            model,
            "--input", "file",
            "--file", str(perturbed_ic),
            "--lead-time", str(lead_time),
            "--path", str(output_file),  # File path for output
        ]

        start_time = datetime.now(timezone.utc)
        result = subprocess.run(cmd_run, capture_output=True, text=True, cwd=str(PANGU_BASE_DIR))
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()

        if result.returncode != 0:
            broadcast_log(f"Forecast failed with exit code {result.returncode}", 'error')
            if result.stdout:
                broadcast_log("STDOUT:", 'error')
                broadcast_log(result.stdout, 'error')
            if result.stderr:
                broadcast_log("STDERR:", 'error')
                broadcast_log(result.stderr, 'error')
            # Cleanup temp dir on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to run forecast: {result.stderr or result.stdout or 'Unknown error'}")

        broadcast_log(f"Climate scenario complete in {elapsed:.1f}s", 'success')

        # Step 4: Extract data
        broadcast_log("Processing forecast output...", 'info')
        forecast_data = extract_grib_data(output_file, date_str, time_str, lead_time)

        # Cleanup temp files
        shutil.rmtree(temp_dir)

        # Save scenario info
        scenario_info = {
            "run_id": scenario_id,
            "model": model,
            "model_name": model_display,
            "climate_scenario": True,
            "temp_offset": temp_offset,
            "adjust_humidity": adjust_humidity,
            "adjust_pressure": adjust_pressure,
            "init_date": date_str,
            "init_time": time_str,
            "lead_time": lead_time,
            "output_file": str(output_file),
            "run_time": start_time.isoformat(),
            "elapsed_seconds": elapsed,
            "data": forecast_data
        }

        db = load_runs_db()
        db["runs"].insert(0, scenario_info)
        db["runs"] = db["runs"][:50]
        save_runs_db(db)

        broadcast_log(f"Climate scenario saved: {scenario_id}", 'success')

        return {
            "success": True,
            "run_id": scenario_id,
            "run_info": scenario_info
        }

    except Exception as e:
        error_msg = f"Climate scenario error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        broadcast_log(error_msg, 'error')
        return {"success": False, "error": error_msg}


def run_ensemble_forecast(date_str, time_str, lead_time, model, num_members, scale):
    """
    Run ensemble forecast with perturbed initial conditions.

    Downloads initial conditions once, then creates perturbed copies
    and runs the model for each ensemble member.
    """
    try:
        model_display = "PanguWeather" if model == "panguweather" else "GraphCast"
        ensemble_id = f"{date_str}T{time_str}:00:00_F{lead_time:03d}_{model}_ENS{num_members}"

        broadcast_log(f"Starting {model_display} ensemble forecast", 'info')
        broadcast_log(f"  Members: {num_members}", 'info')
        broadcast_log(f"  Perturbation scale: {scale}", 'info')
        broadcast_log(f"  Date: {date_str} {time_str}:00 UTC", 'info')

        # Step 1: Download unperturbed initial conditions
        broadcast_log("Downloading initial conditions...", 'info')
        temp_dir = Path(tempfile.mkdtemp(prefix="pangu_ensemble_"))
        base_ic_file = temp_dir / "initial_conditions.grib"

        # Download ICs using ai-models (to a temporary location)
        cmd_download = [
            "ai-models",
            model,
            "--input", "ecmwf-open-data",
            "--date", date_str.replace("-", ""),
            "--time", time_str,
            "--lead-time", "6",  # Just download ICs (6h min)
            "--path", str(base_ic_file),  # File path for download
        ]

        result = subprocess.run(cmd_download, capture_output=True, text=True, check=True, cwd=str(PANGU_BASE_DIR))
        broadcast_log("Initial conditions downloaded", 'success')

        # Step 2: Run ensemble members
        ensemble_runs = []
        start_time = datetime.now(timezone.utc)

        for member in range(num_members):
            broadcast_log(f"Running ensemble member {member+1}/{num_members}...", 'info')

            # Create perturbed IC file
            perturbed_ic = temp_dir / f"ic_member_{member:03d}.grib"
            perturb_grib_file(base_ic_file, perturbed_ic, scale=scale, seed=member)

            # Run model with perturbed ICs
            member_output = FORECASTS_DIR / f"{ensemble_id}_M{member:03d}.grib"

            cmd_run = [
                "ai-models",
                model,
                "--input", "file",
                "--file", str(perturbed_ic),
                "--lead-time", str(lead_time),
                "--path", str(member_output),  # File path for output
            ]

            result = subprocess.run(cmd_run, capture_output=True, text=True, check=True, cwd=str(PANGU_BASE_DIR))

            # Extract data
            forecast_data = extract_grib_data(member_output, date_str, time_str, lead_time)

            member_info = {
                "member_id": member,
                "output_file": str(member_output),
                "data": forecast_data
            }
            ensemble_runs.append(member_info)

            broadcast_log(f"Member {member+1} complete", 'success')

        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()

        # Cleanup temp files
        shutil.rmtree(temp_dir)

        # Save ensemble info
        ensemble_info = {
            "run_id": ensemble_id,
            "model": model,
            "model_name": model_display,
            "ensemble": True,
            "num_members": num_members,
            "perturbation_scale": scale,
            "init_date": date_str,
            "init_time": time_str,
            "lead_time": lead_time,
            "run_time": start_time.isoformat(),
            "elapsed_seconds": elapsed,
            "members": ensemble_runs
        }

        db = load_runs_db()
        db["runs"].insert(0, ensemble_info)
        db["runs"] = db["runs"][:50]
        save_runs_db(db)

        broadcast_log(f"Ensemble forecast complete! ({elapsed:.1f}s total)", 'success')

    except subprocess.CalledProcessError as e:
        error_msg = (f"Ensemble forecast error: Command '{' '.join(e.cmd)}' "
                     f"returned non-zero exit status {e.returncode}.\n"
                     f"STDOUT: {e.stdout}\n"
                     f"STDERR: {e.stderr}")
        logger.error(error_msg)
        broadcast_log(error_msg, 'error')
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Ensemble forecast error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        broadcast_log(error_msg, 'error')
        return {"success": False, "error": error_msg}


def extract_grib_data(grib_file, init_date, init_time, lead_time):
    """
    Extract forecast data from GRIB file for Fairfax, VA.

    Returns dict with time series and spatial data.
    """
    # Fairfax, VA coordinates
    FAIRFAX_LAT = 38.85
    FAIRFAX_LON = -77.31  # West is negative

    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})

        # Get valid times
        times = []
        if 'valid_time' in ds.coords:
            for t in ds.coords['valid_time'].values:
                dt = np.datetime64(t).astype('datetime64[s]').astype(datetime)
                times.append(dt.isoformat())
        elif 'time' in ds.coords and 'step' in ds.coords:
            init_time = ds.coords['time'].values
            for step in ds.coords['step'].values:
                valid = init_time + step
                dt = np.datetime64(valid).astype('datetime64[s]').astype(datetime)
                times.append(dt.isoformat())

        # Initialize time series dictionary for all variables
        time_series = {
            "times": times,
            "temp_surface": [],
            "temp_850": [],
            "temp_700": [],
            "temp_500": [],
            "temp_250": [],
            "pressure_msl": [],
            "u_wind_850": [],
            "v_wind_850": [],
            "wind_speed_850": [],
            "u_wind_500": [],
            "v_wind_500": [],
            "wind_speed_500": [],
            "geopotential_850": [],
            "geopotential_500": [],
            "humidity_850": [],
            "humidity_700": []
        }

        # Get coordinate names
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

        # Interpolate to Fairfax location
        def interp_location(data_array):
            """Interpolate data to Fairfax coordinates."""
            try:
                # Check if longitude is 0-360 or -180-180
                lon_vals = ds.coords[lon_name].values
                if lon_vals.min() >= 0 and lon_vals.max() > 180:
                    # Convert to 0-360
                    fairfax_lon = FAIRFAX_LON + 360
                else:
                    fairfax_lon = FAIRFAX_LON

                result = data_array.interp(
                    {lat_name: FAIRFAX_LAT, lon_name: fairfax_lon},
                    method='linear'
                )
                return result
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}")
                return None

        # Extract temperature at multiple levels
        if 't' in ds:
            for level, key in [(850, "temp_850"), (700, "temp_700"), (500, "temp_500"), (250, "temp_250")]:
                if level in ds.coords.get('isobaricInhPa', []):
                    temp_data = ds['t'].sel(isobaricInhPa=level)
                    temp_loc = interp_location(temp_data)
                    if temp_loc is not None:
                        for i in range(len(times)):
                            val = temp_loc.isel(step=i).values
                            # Convert Kelvin to Fahrenheit
                            temp_f = (float(val) - 273.15) * 9/5 + 32
                            time_series[key].append(temp_f)

        # Extract wind components and compute speed
        if 'u' in ds and 'v' in ds:
            for level in [850, 500]:
                if level in ds.coords.get('isobaricInhPa', []):
                    u_data = ds['u'].sel(isobaricInhPa=level)
                    v_data = ds['v'].sel(isobaricInhPa=level)

                    u_loc = interp_location(u_data)
                    v_loc = interp_location(v_data)

                    if u_loc is not None and v_loc is not None:
                        for i in range(len(times)):
                            u_val = float(u_loc.isel(step=i).values)
                            v_val = float(v_loc.isel(step=i).values)

                            # Convert m/s to mph
                            u_mph = u_val * 2.23694
                            v_mph = v_val * 2.23694
                            wind_speed_mph = np.sqrt(u_mph**2 + v_mph**2)

                            time_series[f"u_wind_{level}"].append(u_mph)
                            time_series[f"v_wind_{level}"].append(v_mph)
                            time_series[f"wind_speed_{level}"].append(wind_speed_mph)

        # Extract geopotential height
        if 'z' in ds or 'gh' in ds:
            var_name = 'gh' if 'gh' in ds else 'z'
            for level, key in [(850, "geopotential_850"), (500, "geopotential_500")]:
                if level in ds.coords.get('isobaricInhPa', []):
                    geo_data = ds[var_name].sel(isobaricInhPa=level)
                    geo_loc = interp_location(geo_data)
                    if geo_loc is not None:
                        for i in range(len(times)):
                            val = float(geo_loc.isel(step=i).values)
                            time_series[key].append(val)

        # Extract specific humidity
        if 'q' in ds:
            for level, key in [(850, "humidity_850"), (700, "humidity_700")]:
                if level in ds.coords.get('isobaricInhPa', []):
                    q_data = ds['q'].sel(isobaricInhPa=level)
                    q_loc = interp_location(q_data)
                    if q_loc is not None:
                        for i in range(len(times)):
                            val = float(q_loc.isel(step=i).values)
                            # Convert to g/kg for easier interpretation
                            val_gkg = val * 1000
                            time_series[key].append(val_gkg)

        ds.close()

        # Extract surface temperature (2m temperature)
        try:
            ds_sfc = xr.open_dataset(grib_file, engine='cfgrib',
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})

            # Get coordinate names for surface data
            lat_name_sfc = 'latitude' if 'latitude' in ds_sfc.coords else 'lat'
            lon_name_sfc = 'longitude' if 'longitude' in ds_sfc.coords else 'lon'

            if 't' in ds_sfc or 't2m' in ds_sfc:
                var_name = 't2m' if 't2m' in ds_sfc else 't'

                # Check longitude convention
                lon_vals = ds_sfc.coords[lon_name_sfc].values
                if lon_vals.min() >= 0 and lon_vals.max() > 180:
                    fairfax_lon = FAIRFAX_LON + 360
                else:
                    fairfax_lon = FAIRFAX_LON

                # Interpolate to Fairfax location
                t2m_loc = ds_sfc[var_name].interp(
                    {lat_name_sfc: FAIRFAX_LAT, lon_name_sfc: fairfax_lon},
                    method='linear'
                )

                for i in range(len(times)):
                    val = float(t2m_loc.isel(step=i).values)
                    # Convert Kelvin to Fahrenheit
                    temp_f = (val - 273.15) * 9/5 + 32
                    time_series["temp_surface"].append(temp_f)

            ds_sfc.close()
        except Exception as e:
            logger.warning(f"Could not extract surface temperature: {e}")
            # Fill with empty values if surface temp not available
            pass

        # Extract mean sea level pressure
        try:
            ds_msl = xr.open_dataset(grib_file, engine='cfgrib',
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})

            # Get coordinate names for MSL data
            lat_name_msl = 'latitude' if 'latitude' in ds_msl.coords else 'lat'
            lon_name_msl = 'longitude' if 'longitude' in ds_msl.coords else 'lon'

            if 'msl' in ds_msl or 'prmsl' in ds_msl:
                var_name = 'msl' if 'msl' in ds_msl else 'prmsl'

                # Check longitude convention
                lon_vals = ds_msl.coords[lon_name_msl].values
                if lon_vals.min() >= 0 and lon_vals.max() > 180:
                    fairfax_lon = FAIRFAX_LON + 360
                else:
                    fairfax_lon = FAIRFAX_LON

                # Interpolate to Fairfax location
                msl_loc = ds_msl[var_name].interp(
                    {lat_name_msl: FAIRFAX_LAT, lon_name_msl: fairfax_lon},
                    method='linear'
                )

                for i in range(len(times)):
                    val = float(msl_loc.isel(step=i).values)
                    # Convert Pascals to millibars (hPa)
                    pressure_mb = val / 100.0
                    time_series["pressure_msl"].append(pressure_mb)

            ds_msl.close()
        except Exception as e:
            logger.warning(f"Could not extract mean sea level pressure: {e}")
            # Fill with empty values if MSLP not available
            pass

        # Convert numpy types to native Python types
        var_list = []
        if 'ds' in locals():
            var_list = [str(v) for v in ds.data_vars.keys()]

        levels_list = []
        if 'ds' in locals() and 'isobaricInhPa' in ds.coords:
            levels_list = [int(l) for l in ds.coords['isobaricInhPa'].values]

        return {
            "time_series": time_series,
            "location": "Fairfax, VA",
            "latitude": float(FAIRFAX_LAT),
            "longitude": float(FAIRFAX_LON),
            "variables_available": var_list,
            "levels_available": levels_list
        }

    except Exception as e:
        logger.error(f"Error extracting GRIB data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "time_series": {"times": [], "temp_850": []},
            "location": "Fairfax, VA",
            "error": str(e)
        }


@pangu_bp.route('/')
def dashboard():
    """Main dashboard - view saved forecasts."""
    return render_template('pangu_dashboard.html')


@pangu_bp.route('/run')
def run_page():
    """Page to run new forecasts."""
    return render_template('pangu_run.html')


@pangu_bp.route('/api/runs')
def api_runs():
    """Get list of saved forecast runs."""
    db = load_runs_db()
    return jsonify({
        "success": True,
        "runs": db.get("runs", [])
    })


@pangu_bp.route('/api/run/<run_id>')
def api_run_detail(run_id):
    """Get details for a specific run."""
    db = load_runs_db()
    runs = db.get("runs", [])

    for run in runs:
        if run["run_id"] == run_id:
            return jsonify({"success": True, "run": run})

    return jsonify({"success": False, "error": "Run not found"}), 404


@pangu_bp.route('/api/forecast/run', methods=['POST'])
def api_run_forecast():
    """Run a new forecast."""
    data = request.json
    date_str = data.get('date')
    time_str = data.get('time', '00')
    lead_time = int(data.get('lead_time', 24))
    model = data.get('model', 'panguweather')

    # Perturbation parameters
    perturbation_type = data.get('perturbation_type', 'none')
    ensemble_members = int(data.get('ensemble_members', 1))
    perturbation_scale = data.get('perturbation_scale', 'medium')

    # Climate scenario parameters
    temp_offset = float(data.get('temp_offset', 0))
    adjust_humidity = data.get('adjust_humidity', True)
    adjust_pressure = data.get('adjust_pressure', True)

    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"})

    if time_str not in ['00', '06', '12', '18']:
        return jsonify({"success": False, "error": "Time must be 00, 06, 12, or 18"})

    if model not in ['panguweather', 'graphcast', 'fourcastnet']:
        return jsonify({"success": False, "error": "Model must be 'panguweather', 'graphcast', or 'fourcastnet'"})

    # Run forecast in background thread
    def run_in_background():
        try:
            run_forecast(date_str, time_str, lead_time, model,
                        perturbation_type, ensemble_members, perturbation_scale,
                        temp_offset, adjust_humidity, adjust_pressure)
        except Exception as e:
            logger.error(f"Background forecast error: {e}")
            broadcast_log(f"Forecast failed: {str(e)}", 'error')
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_in_background)
    thread.daemon = True
    thread.start()

    model_display = "PanguWeather" if model == "panguweather" else "GraphCast"
    if perturbation_type == 'ensemble' and ensemble_members > 1:
        message = f"{model_display} ensemble forecast started ({ensemble_members} members)"
    elif perturbation_type == 'climate':
        message = f"{model_display} climate scenario started (+{temp_offset}°C)"
    else:
        message = f"{model_display} forecast started"

    return jsonify({
        "success": True,
        "message": message,
        "status": "running"
    })


@pangu_bp.route('/api/logs')
def api_logs():
    """Server-Sent Events endpoint for streaming logs."""
    def event_stream():
        q = queue.Queue(maxsize=100)

        with log_lock:
            log_queues.append(q)

        try:
            yield f"data: {json.dumps({'message': 'Connected to log stream', 'type': 'info'})}\n\n"

            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    yield f": keepalive\n\n"
        except GeneratorExit:
            with log_lock:
                if q in log_queues:
                    log_queues.remove(q)

    return Response(event_stream(), mimetype='text/event-stream')


@pangu_bp.route('/map')
def map_viewer():
    """Map viewer page for spatial visualization."""
    return render_template('pangu_map.html', regions=REGIONS)


def _generate_custom_map_frame(grib_file, custom_type, time_index, region_name):
    """Generate a single frame for a custom map animation and its stats."""
    if custom_type == 'custom_q850_mslp':
        # Use _get_map_data_subset for q_data
        lons_q, lats_q, q_values, region_bounds, valid_time_q = _get_map_data_subset(
            grib_file, 'q', '850', time_index, region_name
        )
        # Use _get_map_data_subset for msl_data
        lons_msl, lats_msl, msl_values, _, valid_time_msl = _get_map_data_subset(
            grib_file, 'msl', 'surface', time_index, region_name
        )

        if q_values is None or msl_values is None:
            raise ValueError("Failed to retrieve data for custom map frame.")

        # Convert q to g/kg
        q_gkg = q_values * 1000
        msl_mb = msl_values / 100  # Convert Pa to mb

        west, east, south, north = region_bounds

        # Create figure
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')

        # Plot specific humidity as filled contours
        levels_q = np.arange(0, 20, 1)
        cf = ax.contourf(lons_q, lats_q, q_gkg, levels=levels_q, cmap='YlGnBu',
                       transform=ccrs.PlateCarree(), alpha=0.8)

        # Overlay MSLP as black contours
        levels_msl = np.arange(960, 1050, 4)
        cs = ax.contour(lons_msl, lats_msl, msl_mb, levels=levels_msl, colors='black',
                      linewidths=1.2, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d mb')

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Specific Humidity @ 850 hPa (g/kg)', fontsize=10)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Add title
        valid_time = valid_time_q # Use humidity valid time as primary
        plt.title(f'850 hPa Specific Humidity + MSLP Contours\n{region_name} - Valid: {np.datetime64(valid_time).astype("datetime64[s]")}',
                 fontsize=12, weight='bold')

        # Calculate stats for the humidity field
        stats = {
            "min": float(q_gkg.min()),
            "max": float(q_gkg.max()),
            "mean": float(q_gkg.mean()),
            "std": float(q_gkg.std())
        }

        # Save to buffer and return PIL Image and stats
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer), stats, valid_time

    raise ValueError(f"Unknown custom map type: {custom_type}")


def generate_custom_map(grib_file, custom_type, time_index, region_name, run):
    """Generate custom overlay maps."""
    try:
        # Generate the map image and stats using the new helper function
        img, stats, valid_time = _generate_custom_map_frame(grib_file, custom_type, time_index, region_name)

        # Convert image to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "success": True,
            "map_image": img_base64,
            "variable": custom_type,
            "level": "850",
            "region": region_name,
            "valid_time": str(np.datetime64(valid_time).astype('datetime64[s]')),
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Error generating custom map: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _get_map_data_subset(grib_file, variable, level, time_index, region_name):
    """
    Helper function to get subsetted xarray DataArray for plotting.
    Handles GRIB file opening, variable/level selection, time indexing, and regional subsetting.
    """
    region_bounds = REGIONS.get(region_name, REGIONS["Global"])
    west, east, south, north = region_bounds

    # Open GRIB file - determine typeOfLevel based on variable
    if variable == 'msl' or variable == 'prmsl':
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
    elif variable == 't2m' or (variable == 't' and level == 'surface'):
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
    elif variable in ['u10', 'v10'] or ((variable == 'u' or variable == 'v') and level == 'surface'):
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
    elif level == 'surface':
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    else:
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})

    # Check if variable exists
    if variable not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable '{variable}' not found. Available: {list(ds.data_vars.keys())}")

    # Select data
    data = ds[variable]

    # Select level if not surface
    if level != 'surface' and 'isobaricInhPa' in ds.coords:
        level_val = int(level)
        if level_val not in ds.coords['isobaricInhPa'].values:
            ds.close()
            raise ValueError(f"Level {level} not available. Available: {list(ds.coords['isobaricInhPa'].values)}")
        data = data.sel(isobaricInhPa=level_val)

    # Select time step
    if 'step' in data.dims and len(data.coords['step']) > time_index:
        data = data.isel(step=time_index)
    elif 'step' in data.dims and len(data.coords['step']) == 1 and time_index > 0:
        # If only one step exists, we can't select other time indices
        ds.close()
        raise ValueError(f"Time index {time_index} out of range for variable '{variable}' which only has 1 step.")


    # Check longitude convention and adjust region bounds if needed
    lon_vals = data.coords['longitude'].values
    lon_min, lon_max = lon_vals.min(), lon_vals.max()

    # Determine if data uses 0-360 or -180-180 convention
    uses_0_360 = lon_min >= 0 and lon_max > 180

    # Check if this is a global view (covers ~360 degrees)
    lon_span = east - west
    is_global = lon_span >= 350 or (west == -180 and east == 180)

    # Subset to region using proper coordinate selection
    try:
        if is_global:
            # For global view, only subset latitude
            data_subset = data.sel(latitude=slice(north, south))
        else:
            # Adjust region bounds to match data convention
            if uses_0_360 and west < 0:
                # Convert -180/180 bounds to 0/360
                west_adj = west + 360 if west < 0 else west
                east_adj = east + 360 if east < 0 else east
            else:
                west_adj = west
                east_adj = east

            # Handle case where region crosses prime meridian
            if west_adj > east_adj:
                # Region crosses 0° (or 180° in 0-360 convention)
                data_subset1 = data.sel(
                    latitude=slice(north, south),
                    longitude=slice(west_adj, lon_max)
                )
                data_subset2 = data.sel(
                    latitude=slice(north, south),
                    longitude=slice(lon_min, east_adj)
                )
                # Concatenate along longitude dimension
                data_subset = xr.concat([data_subset1, data_subset2], dim='longitude')
            else:
                data_subset = data.sel(
                    latitude=slice(north, south),
                    longitude=slice(west_adj, east_adj)
                )
    except Exception as e:
        logger.warning(f"Region selection failed: {e}, using full data")
        data_subset = data

    # Convert to numpy and handle any NaNs
    values = data_subset.values
    lats_subset = data_subset.coords['latitude'].values
    lons_subset = data_subset.coords['longitude'].values

    # For regional views in 0-360 convention, we need to handle longitude wrapping
    if uses_0_360 and not is_global and west < 0:
        # Convert longitudes to -180/180 AND reorder the data
        split_idx = np.where(lons_subset > 180)[0]

        if len(split_idx) > 0:
            split_idx = split_idx[0]

            # Split and reorder longitudes
            lons_left = lons_subset[:split_idx]  # 0 to 180
            lons_right = lons_subset[split_idx:] - 360  # 180-360 -> -180 to 0
            lons_subset = np.concatenate([lons_right, lons_left])

            # Reorder the data values to match
            if values.ndim == 2:
                values = np.concatenate([values[:, split_idx:], values[:, :split_idx]], axis=1)
            elif values.ndim == 1:
                values = np.concatenate([values[split_idx:], values[:split_idx]])

            # Sort by longitude to ensure monotonic increasing
            sort_idx = np.argsort(lons_subset)
            lons_subset = lons_subset[sort_idx]
            if values.ndim == 2:
                values = values[:, sort_idx]
            else:
                values = values[sort_idx]
    
    # Determine valid_time_str after potentially modifying 'data' with isel
    valid_time = None
    if 'valid_time' in data.coords:
        valid_time = data.coords['valid_time'].values
    elif 'time' in ds.coords and 'step' in ds.coords:
        # Fallback for cases where 'valid_time' might not be directly in data.coords after subsetting
        try:
            init_time_val = ds.coords['time'].values
            # Ensure step_val is within bounds and convert to timedelta in hours
            step_val = ds.coords['step'].values[time_index] if time_index < len(ds.coords['step']) else ds.coords['step'].values[0]
            valid_time = np.datetime64(init_time_val) + np.timedelta64(int(step_val), 'h')
        except Exception as e:
            logger.warning(f"Could not determine valid_time using fallback for {variable}: {e}")

    ds.close()
    return lons_subset, lats_subset, values, region_bounds, valid_time

def _fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def generate_map_image(lons, lats, values, variable, level, region_bounds, valid_time_str=""):
    """Generate a map image from spatial data and return the matplotlib Figure and Axes."""
    west, east, south, north = region_bounds

    # Validate dimensions
    if lons.shape != lats.shape:
        logger.error(f"Dimension mismatch: lons {lons.shape} != lats {lats.shape}")
        return None, None

    if values.ndim == 1:
        if len(values) == lons.size:
            values = values.reshape(lons.shape)
        else:
            logger.error(f"Cannot reshape values of length {len(values)} to match lons/lats shape {lons.shape}")
            return None, None

    if values.shape != lons.shape:
        logger.error(f"Values shape {values.shape} doesn't match lons/lats shape {lons.shape}")
        return None, None

    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    # Choose colormap based on variable
    if variable in ['t', 't2m']:
        cmap = 'RdYlBu_r'
        label = 'Temperature (K)'
    elif variable in ['u', 'v', 'u10', 'v10']:
        cmap = 'coolwarm'
        label = f'{variable} (m/s)'
    elif variable in ['z', 'gh']:
        cmap = 'viridis'
        label = 'Geopotential (m²/s²)'
    elif variable == 'q':
        cmap = 'YlGnBu'
        label = 'Specific Humidity (kg/kg)'
    elif variable == 'msl':
        cmap = 'RdBu_r'
        label = 'Pressure (Pa)'
    else:
        cmap = 'viridis'
        label = variable

    # Plot data
    im = ax.contourf(lons, lats, values, levels=15, cmap=cmap, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label(label)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Title
    title = f'{variable.upper()}'
    if level != 'surface':
        title += f' at {level} hPa'
    if valid_time_str:
        title += f'\nValid: {valid_time_str}'
    plt.title(title, fontsize=14, fontweight='bold')

    return fig, ax


@pangu_bp.route('/api/spatial-data')
def api_spatial_data():
    """
    Get spatial data for a specific variable, level, and time.

    Query params:
        run_id: Run identifier
        variable: Variable name (t, u, v, z, q, etc.)
        level: Pressure level (850, 500, etc.) or 'surface'
        time_index: Time step index (0, 1, 2, ...)
        region: Region name (optional, defaults to 'Global')
    """
    run_id = request.args.get('run_id')
    variable = request.args.get('variable', 't')
    level = request.args.get('level', '850')
    time_index = int(request.args.get('time_index', 0))
    region_name = request.args.get('region', 'Global')

    if not run_id:
        return jsonify({"success": False, "error": "Missing run_id parameter"}), 400

    # Get run info
    db = load_runs_db()
    run = None
    for r in db.get("runs", []):
        if r["run_id"] == run_id:
            run = r
            break

    if not run:
        return jsonify({"success": False, "error": "Run not found"}), 404

    member_id = request.args.get('member_id')
    if run.get("ensemble") and run.get("members"):
        if member_id is None:
            return jsonify({"success": False, "error": "Missing member_id for ensemble run"}), 400
        try:
            member_id_int = int(member_id)
        except ValueError:
            return jsonify({"success": False, "error": "Invalid member_id"}), 400

        member = None
        for m in run["members"]:
            if int(m.get("member_id", -1)) == member_id_int:
                member = m
                break
        if member is None:
            return jsonify({"success": False, "error": "Ensemble member not found"}), 404

        grib_file = Path(member["output_file"])
    else:
        grib_file = Path(run["output_file"])
    if not grib_file.exists():
        return jsonify({"success": False, "error": "GRIB file not found"}), 404

    # Handle custom maps
    if variable.startswith('custom_'):
        return generate_custom_map(grib_file, variable, time_index, region_name, run)

    try:
        lons_subset, lats_subset, values, region_bounds, valid_time = _get_map_data_subset(
            grib_file, variable, level, time_index, region_name
        )

        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(lons_subset, lats_subset)

        # Generate map image
        fig, ax = generate_map_image(lon_grid, lat_grid, values, variable, level, region_bounds, str(np.datetime64(valid_time).astype('datetime64[s]')) if valid_time is not None else "")
        map_image_base64 = _fig_to_base64(fig)

        # Get statistics
        stats = {
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values))
        }

        return jsonify({
            "success": True,
            "map_image": map_image_base64,
            "stats": stats,
            "valid_time": str(np.datetime64(valid_time).astype('datetime64[s]')) if valid_time is not None else "Unknown",
            "variable": variable,
            "level": level,
            "region": region_name,
            "shape": values.shape
        })

    except Exception as e:
        logger.error(f"Error extracting spatial data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@pangu_bp.route('/api/animate-spatial-data')
def api_animate_spatial_data():
    """
    Generate an animated GIF of spatial data over all time steps.

    Query params:
        run_id: Run identifier
        variable: Variable name (t, u, v, z, q, etc.)
        level: Pressure level (850, 500, etc.) or 'surface'
        region: Region name (optional, defaults to 'Global')
        fps: Frames per second for the GIF (optional, defaults to 5)
    """
    run_id = request.args.get('run_id')
    variable = request.args.get('variable', 't')
    level = request.args.get('level', '850')
    region_name = request.args.get('region', 'Global')
    fps = int(request.args.get('fps', 5))

    if not run_id:
        return jsonify({"success": False, "error": "Missing run_id parameter"}), 400

    db = load_runs_db()
    run = None
    for r in db.get("runs", []):
        if r["run_id"] == run_id:
            run = r
            break

    if not run:
        return jsonify({"success": False, "error": "Run not found"}), 404

    grib_file = Path(run["output_file"])
    if not grib_file.exists():
        return jsonify({"success": False, "error": "GRIB file not found"}), 404

    try:
        # Determine max time index
        max_time_index = int(run["lead_time"] / 6)
        frames = []

        if variable.startswith('custom_'):
            target_size = None
            for time_index in range(max_time_index + 1):
                try:
                    img, _, _ = _generate_custom_map_frame(grib_file, variable, time_index, region_name)
                    if target_size is None:
                        target_size = img.size
                    if img.size != target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                    frames.append(img)
                except Exception as e:
                    logger.warning(f"Skipping timestep {time_index} for custom map due to error: {e}")
                    continue
        else:
            all_values = []
            # First pass: collect all data values to determine global min/max
            for time_index in range(max_time_index + 1):
                try:
                    _, _, values, _, _ = _get_map_data_subset(
                        grib_file, variable, level, time_index, region_name
                    )
                    if values is not None:
                        all_values.append(values.flatten())
                except Exception as e:
                    logger.warning(f"Skipping timestep {time_index} due to error: {e}")
                    continue
            
            if not all_values:
                return jsonify({"success": False, "error": "No data available to generate animation across all timesteps"}), 500

            # Calculate global min/max for consistent colorbar
            global_min = float(np.nanmin(np.concatenate(all_values)))
            global_max = float(np.nanmax(np.concatenate(all_values)))

            # Second pass: generate frames with fixed colorbar range
            target_size = None
            for time_index in range(max_time_index + 1):
                try:
                    lons_subset, lats_subset, values, region_bounds, valid_time = _get_map_data_subset(
                        grib_file, variable, level, time_index, region_name
                    )

                    lon_grid, lat_grid = np.meshgrid(lons_subset, lats_subset)

                    fig, ax = generate_map_image(lon_grid, lat_grid, values, variable, level, region_bounds,
                                                 str(np.datetime64(valid_time).astype('datetime64[s]')) if valid_time is not None else "",
                                                 vmin=global_min, vmax=global_max)

                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    buffer.seek(0)
                    img = Image.open(buffer)

                    if target_size is None:
                        target_size = img.size
                    
                    if img.size != target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)

                    frames.append(img)
                except Exception as e:
                    logger.warning(f"Failed to generate frame for timestep {time_index}: {e}")
                    continue

        if not frames:
            return jsonify({"success": False, "error": "No frames were generated for the animation."}), 500

        # Save GIF to a temporary file
        temp_gif_path = Path(tempfile.mkstemp(suffix=".gif")[1])
        imageio.mimsave(str(temp_gif_path), frames, fps=fps, loop=0)

        # Return GIF as a file response
        def generate_file_content():
            with open(temp_gif_path, 'rb') as f:
                yield from f
            temp_gif_path.unlink() # Clean up the temporary file

        return Response(generate_file_content(), mimetype='image/gif',
                        headers={'Content-Disposition': 'attachment;filename=animation.gif'})

    except Exception as e:
        logger.error(f"Error generating animation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500





def generate_map_image(lons, lats, values, variable, level, region_bounds, valid_time_str="", vmin=None, vmax=None):
    """Generate a map image from spatial data and return the matplotlib Figure and Axes."""
    try:
        west, east, south, north = region_bounds

        # Validate dimensions
        if lons.shape != lats.shape:
            logger.error(f"Dimension mismatch: lons {lons.shape} != lats {lats.shape}")
            return None, None

        if values.ndim == 1:
            if len(values) == lons.size:
                values = values.reshape(lons.shape)
            else:
                logger.error(f"Cannot reshape values of length {len(values)} to match lons/lats shape {lons.shape}")
                return None, None

        if values.shape != lons.shape:
            logger.error(f"Values shape {values.shape} doesn't match lons/lats shape {lons.shape}")
            return None, None

        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

        # Choose colormap based on variable
        if variable in ['t', 't2m']:
            cmap = 'RdYlBu_r'
            label = 'Temperature (K)'
        elif variable in ['u', 'v', 'u10', 'v10']:
            cmap = 'coolwarm'
            label = f'{variable} (m/s)'
        elif variable in ['z', 'gh']:
            cmap = 'viridis'
            label = 'Geopotential (m²/s²)'
        elif variable == 'q':
            cmap = 'YlGnBu'
            label = 'Specific Humidity (kg/kg)'
        elif variable == 'msl':
            cmap = 'RdBu_r'
            label = 'Pressure (Pa)'
        else:
            cmap = 'viridis'
            label = variable

        # Plot data using fixed vmin/vmax if provided
        if vmin is not None and vmax is not None:
            im = ax.contourf(lons, lats, values, levels=15, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        else:
            im = ax.contourf(lons, lats, values, levels=15, cmap=cmap, transform=ccrs.PlateCarree())

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label(label)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Title
        title = f'{variable.upper()}'
        if level != 'surface':
            title += f' at {level} hPa'
        if valid_time_str:
            title += f'\nValid: {valid_time_str}'
        plt.title(title, fontsize=14, fontweight='bold')

        return fig, ax

    except Exception as e:
        logger.error(f"Error generating map image: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return None, None


def cleanup_database():
    """Remove orphaned entries from database where GRIB files no longer exist."""
    if FORECASTS_DB.exists():
        try:
            db = load_runs_db()
            original_count = len(db['runs'])

            # Keep only runs with existing GRIB files
            valid_runs = [run for run in db['runs'] if Path(run['output_file']).exists()]

            if len(valid_runs) < original_count:
                db['runs'] = valid_runs
                save_runs_db(db)
                removed = original_count - len(valid_runs)
                logger.info(f"Database cleanup: removed {removed} orphaned entries, {len(valid_runs)} valid runs remain")
        except Exception as e:
            logger.warning(f"Error during database cleanup: {e}")
