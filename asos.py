"""
ASOS (Automated Surface Observing System) Station Data Handler

Fetches ASOS station metadata and observations from the Iowa Environmental Mesonet (IEM).
Manages storage of model forecasts at ASOS station locations for verification.

IEM Data Sources (free, no authentication required):
- Station metadata: https://mesonet.agron.iastate.edu/geojson/network/{STATE}_ASOS.geojson
- Observations: http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request
import urllib.parse
import time

logger = logging.getLogger(__name__)

# Cache directory for station metadata
CACHE_DIR = Path.home() / ".cache" / "weather_models" / "asos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Station list cache file
STATIONS_CACHE_FILE = CACHE_DIR / "stations.json"
STATIONS_CACHE_TTL_DAYS = 7

# ASOS forecasts storage file (in project directory)
ASOS_FORECASTS_FILE = Path(__file__).parent / "asos_forecasts.json"

# Retention period for stored forecasts
FORECASTS_RETENTION_DAYS = 21

# US states with ASOS networks
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'PR', 'VI'  # Include DC, Puerto Rico, Virgin Islands
]

# IEM base URLs
IEM_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/{state}_ASOS.geojson"
IEM_ASOS_URL = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


@dataclass
class ASOSStation:
    """ASOS station metadata."""
    station_id: str
    name: str
    lat: float
    lon: float
    state: str
    elevation: Optional[float] = None


def fetch_state_stations(state: str) -> List[ASOSStation]:
    """
    Fetch ASOS stations for a single state from IEM.

    Args:
        state: Two-letter state code

    Returns:
        List of ASOSStation objects
    """
    url = IEM_GEOJSON_URL.format(state=state)
    stations = []

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        for feature in data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [])

            if len(coords) >= 2:
                station = ASOSStation(
                    station_id=props.get('sid', props.get('id', '')),
                    name=props.get('sname', props.get('name', '')),
                    lat=coords[1],
                    lon=coords[0],
                    state=state,
                    elevation=props.get('elevation')
                )
                if station.station_id:
                    stations.append(station)

    except Exception as e:
        logger.warning(f"Failed to fetch stations for {state}: {e}")

    return stations


def fetch_all_stations(force_refresh: bool = False) -> List[ASOSStation]:
    """
    Fetch all US ASOS stations from IEM, with caching.

    Args:
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        List of all ASOSStation objects
    """
    # Check cache first
    if not force_refresh and STATIONS_CACHE_FILE.exists():
        try:
            with open(STATIONS_CACHE_FILE) as f:
                cache_data = json.load(f)

            cache_time = datetime.fromisoformat(cache_data.get('fetched_at', '2000-01-01'))
            if datetime.utcnow() - cache_time < timedelta(days=STATIONS_CACHE_TTL_DAYS):
                stations = [ASOSStation(**s) for s in cache_data.get('stations', [])]
                logger.info(f"Loaded {len(stations)} stations from cache")
                return stations
        except Exception as e:
            logger.warning(f"Error reading stations cache: {e}")

    # Fetch fresh data from IEM
    logger.info("Fetching ASOS stations from IEM...")
    all_stations = []

    for state in US_STATES:
        stations = fetch_state_stations(state)
        all_stations.extend(stations)
        # Be nice to IEM servers
        time.sleep(0.1)

    logger.info(f"Fetched {len(all_stations)} ASOS stations from IEM")

    # Save to cache
    try:
        cache_data = {
            'fetched_at': datetime.utcnow().isoformat(),
            'stations': [asdict(s) for s in all_stations]
        }
        with open(STATIONS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        logger.info(f"Saved stations cache to {STATIONS_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Error saving stations cache: {e}")

    return all_stations


def get_stations_dict() -> Dict[str, dict]:
    """
    Get all stations as a dict keyed by station_id.

    Returns:
        Dict mapping station_id to station metadata dict
    """
    stations = fetch_all_stations()
    return {s.station_id: asdict(s) for s in stations}


def fetch_observations(
    station_ids: List[str],
    start_time: datetime,
    end_time: datetime,
    variables: List[str] = None
) -> Dict[str, List[dict]]:
    """
    Fetch observations from IEM for given stations and time range.

    Args:
        station_ids: List of station IDs to fetch
        start_time: Start of time range (UTC)
        end_time: End of time range (UTC)
        variables: List of variables to fetch (default: tmpf, mslp, p01i)

    Returns:
        Dict mapping station_id to list of observation dicts
    """
    if variables is None:
        variables = ['tmpf', 'mslp', 'p01i']

    # IEM request parameters
    params = {
        'data': variables,
        'tz': 'Etc/UTC',
        'format': 'comma',
        'latlon': 'no',
        'elev': 'no',
        'missing': 'M',
        'trace': 'T',
        'direct': 'no',
        'report_type': [1, 2],  # Routine and special METAR
    }

    # Add station IDs
    for sid in station_ids:
        params.setdefault('station', []).append(sid)

    # Add time range
    params['year1'] = start_time.year
    params['month1'] = start_time.month
    params['day1'] = start_time.day
    params['hour1'] = start_time.hour
    params['year2'] = end_time.year
    params['month2'] = end_time.month
    params['day2'] = end_time.day
    params['hour2'] = end_time.hour

    # Build URL with repeated params
    query_parts = []
    for key, value in params.items():
        if isinstance(value, list):
            for v in value:
                query_parts.append(f"{key}={urllib.parse.quote(str(v))}")
        else:
            query_parts.append(f"{key}={urllib.parse.quote(str(value))}")

    url = f"{IEM_ASOS_URL}?{'&'.join(query_parts)}"

    observations = {sid: [] for sid in station_ids}

    # Build a mapping from short station ID (without K prefix) to full ID
    # IEM returns "DCA" but we pass "KDCA"
    short_to_full = {}
    for sid in station_ids:
        short_to_full[sid] = sid  # Full ID maps to itself
        if sid.startswith('K') and len(sid) == 4:
            short_to_full[sid[1:]] = sid  # "DCA" -> "KDCA"

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read().decode('utf-8')

        # Parse CSV response
        lines = content.strip().split('\n')

        # Skip debug lines (start with #) and find header
        header = None
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            # First non-comment line is the header
            header = line.split(',')
            data_start_idx = i + 1
            break

        if header is None or data_start_idx >= len(lines):
            return observations

        # Find column indices
        col_station = header.index('station') if 'station' in header else 0
        col_valid = header.index('valid') if 'valid' in header else 1
        col_tmpf = header.index('tmpf') if 'tmpf' in header else -1
        col_mslp = header.index('mslp') if 'mslp' in header else -1
        col_p01i = header.index('p01i') if 'p01i' in header else -1

        for line in lines[data_start_idx:]:
            if not line.strip() or line.startswith('#'):
                continue

            fields = line.split(',')
            if len(fields) < 2:
                continue

            station_short = fields[col_station].strip()
            # Map short ID back to full ID
            station = short_to_full.get(station_short)
            if station is None:
                continue

            try:
                # Parse timestamp
                valid_str = fields[col_valid].strip()
                valid_time = datetime.strptime(valid_str, '%Y-%m-%d %H:%M')

                obs = {
                    'valid_time': valid_time.isoformat(),
                }

                # Parse temperature (°F)
                if col_tmpf >= 0 and col_tmpf < len(fields):
                    val = fields[col_tmpf].strip()
                    obs['temp'] = float(val) if val not in ['M', 'T', ''] else None

                # Parse pressure (mb/hPa)
                if col_mslp >= 0 and col_mslp < len(fields):
                    val = fields[col_mslp].strip()
                    obs['mslp'] = float(val) if val not in ['M', 'T', ''] else None

                # Parse 1-hour precipitation (inches)
                if col_p01i >= 0 and col_p01i < len(fields):
                    val = fields[col_p01i].strip()
                    if val == 'T':
                        obs['precip'] = 0.001  # Trace
                    elif val not in ['M', '']:
                        obs['precip'] = float(val)
                    else:
                        obs['precip'] = None

                observations[station].append(obs)

            except (ValueError, IndexError) as e:
                continue

    except Exception as e:
        logger.error(f"Error fetching ASOS observations: {e}")

    return observations


def get_observation_at_time(
    station_obs: List[dict],
    target_time: datetime,
    max_delta_minutes: int = 30
) -> Optional[dict]:
    """
    Find the observation nearest to target_time within tolerance.

    Args:
        station_obs: List of observation dicts for a station
        target_time: Target valid time (UTC)
        max_delta_minutes: Maximum time difference allowed

    Returns:
        Observation dict or None
    """
    if not station_obs:
        return None

    best_match = None
    best_delta = timedelta(minutes=max_delta_minutes + 1)

    for obs in station_obs:
        try:
            obs_time = datetime.fromisoformat(obs['valid_time'])
            delta = abs(obs_time - target_time)

            if delta < best_delta:
                best_delta = delta
                best_match = obs
        except (KeyError, ValueError):
            continue

    if best_delta <= timedelta(minutes=max_delta_minutes):
        return best_match
    return None


def load_asos_forecasts_db() -> dict:
    """Load the ASOS forecasts database from JSON file."""
    if ASOS_FORECASTS_FILE.exists():
        try:
            with open(ASOS_FORECASTS_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading asos_forecasts.json: {e}")
    return {"stations": {}, "runs": {}}


def save_asos_forecasts_db(data: dict):
    """Save the ASOS forecasts database to JSON file."""
    with open(ASOS_FORECASTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved ASOS forecasts to {ASOS_FORECASTS_FILE}")


def cleanup_old_runs(data: dict) -> dict:
    """Remove forecast runs older than retention period."""
    cutoff = datetime.utcnow() - timedelta(days=FORECASTS_RETENTION_DAYS)
    cutoff_str = cutoff.isoformat()

    old_runs = [run_id for run_id in data.get("runs", {}) if run_id < cutoff_str]

    for run_id in old_runs:
        del data["runs"][run_id]

    if old_runs:
        logger.info(f"Cleaned up {len(old_runs)} old forecast runs")

    # Also cleanup old observations
    obs_data = data.get("observations", {})
    for station_id in list(obs_data.keys()):
        station_obs = obs_data[station_id]
        old_times = [t for t in station_obs if t < cutoff_str]
        for t in old_times:
            del station_obs[t]
        # Remove station if no observations left
        if not station_obs:
            del obs_data[station_id]

    return data


def fetch_and_store_observations():
    """
    Fetch observations from IEM for all stored forecast valid times and store them.

    This should be called during sync to collect observations for verification.
    Only fetches observations for valid times that have already passed.
    """
    from datetime import timezone

    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    if not runs or not stations:
        logger.info("No forecast runs to fetch observations for")
        return 0

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Initialize observations dict if needed
    if "observations" not in db:
        db["observations"] = {}

    # Collect all valid times we need observations for
    valid_times_needed = set()

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        for hour in forecast_hours:
            valid_time = init_time + timedelta(hours=hour)
            # Only fetch for past times
            if valid_time < now:
                valid_times_needed.add(valid_time)

    if not valid_times_needed:
        logger.info("No past valid times to fetch observations for")
        return 0

    # Filter out times we already have observations for (for most stations)
    existing_obs = db.get("observations", {})
    times_to_fetch = []

    for vt in sorted(valid_times_needed):
        vt_iso = vt.isoformat()
        # Check if we have observations for this time for at least half the stations
        stations_with_obs = sum(1 for sid in existing_obs if vt_iso in existing_obs.get(sid, {}))
        if stations_with_obs < len(stations) * 0.5:
            times_to_fetch.append(vt)

    if not times_to_fetch:
        logger.info("Already have observations for all valid times")
        return 0

    # Determine time range to fetch
    min_time = min(times_to_fetch) - timedelta(hours=1)
    max_time = max(times_to_fetch) + timedelta(hours=1)

    logger.info(f"Fetching ASOS observations from {min_time} to {max_time} for {len(times_to_fetch)} valid times")

    # Fetch observations in chunks of stations
    station_ids = list(stations.keys())
    chunk_size = 100
    all_observations = {}

    for i in range(0, len(station_ids), chunk_size):
        chunk_ids = station_ids[i:i+chunk_size]
        logger.info(f"Fetching observations for stations {i+1}-{min(i+chunk_size, len(station_ids))} of {len(station_ids)}...")

        obs_chunk = fetch_observations(
            chunk_ids,
            min_time,
            max_time,
            variables=['tmpf', 'mslp', 'p01i']
        )
        all_observations.update(obs_chunk)
        time.sleep(0.5)  # Be nice to IEM

    # Store observations keyed by station and valid time
    obs_count = 0

    for station_id, obs_list in all_observations.items():
        if station_id not in db["observations"]:
            db["observations"][station_id] = {}

        for obs in obs_list:
            obs_time_str = obs.get('valid_time')
            if not obs_time_str:
                continue

            # Store observation data
            db["observations"][station_id][obs_time_str] = {
                'temp': obs.get('temp'),
                'mslp': obs.get('mslp'),
                'precip': obs.get('precip')
            }
            obs_count += 1

    # Cleanup old data and save
    db = cleanup_old_runs(db)
    save_asos_forecasts_db(db)

    logger.info(f"Stored {obs_count} observations for {len(all_observations)} stations")
    return obs_count


def get_stored_observation(db: dict, station_id: str, target_time: datetime, max_delta_minutes: int = 30) -> Optional[dict]:
    """
    Get a stored observation for a station near the target time.

    Args:
        db: The loaded database
        station_id: Station ID
        target_time: Target valid time
        max_delta_minutes: Maximum time difference allowed

    Returns:
        Observation dict or None
    """
    station_obs = db.get("observations", {}).get(station_id, {})

    if not station_obs:
        return None

    best_match = None
    best_delta = timedelta(minutes=max_delta_minutes + 1)

    for obs_time_str, obs_data in station_obs.items():
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            delta = abs(obs_time - target_time)

            if delta < best_delta:
                best_delta = delta
                best_match = obs_data
        except ValueError:
            continue

    if best_delta <= timedelta(minutes=max_delta_minutes):
        return best_match
    return None


def store_asos_forecasts(
    init_time: datetime,
    forecast_hours: List[int],
    model_name: str,
    station_forecasts: Dict[str, dict]
):
    """
    Store model forecasts at ASOS station locations.

    Args:
        init_time: Model initialization time
        forecast_hours: List of forecast hours
        model_name: Model name ('gfs', 'aifs', 'ifs')
        station_forecasts: Dict mapping station_id to forecast data
            Each station dict has: temps, mslps, precips (lists aligned with forecast_hours)
    """
    from datetime import timezone

    db = load_asos_forecasts_db()

    # Ensure stations are up to date
    stations = get_stations_dict()
    db["stations"] = stations

    # Create run entry if needed
    run_key = init_time.isoformat()
    if run_key not in db.get("runs", {}):
        db.setdefault("runs", {})[run_key] = {
            "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "forecast_hours": forecast_hours,
        }

    # Store forecasts for this model
    db["runs"][run_key][model_name.lower()] = station_forecasts

    # Cleanup old runs
    db = cleanup_old_runs(db)

    # Save
    save_asos_forecasts_db(db)
    logger.info(f"Stored {model_name} forecasts for {len(station_forecasts)} stations at {run_key}")


def get_verification_data(
    model: str,
    variable: str,
    lead_time_hours: int
) -> Dict[str, dict]:
    """
    Get verification data for all stations at a specific lead time.

    Matches stored forecasts with stored ASOS observations to calculate
    MAE and bias per station.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours

    Returns:
        Dict mapping station_id to verification data:
        {
            'station_id': {
                'mae': float,
                'bias': float,
                'count': int,
                'lat': float,
                'lon': float,
                'name': str
            }
        }
    """
    from datetime import timezone

    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    if not runs or not stations:
        return {}

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Map variable name to forecast/obs keys
    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip')
    }

    if variable not in var_map:
        return {}

    fcst_key, obs_key = var_map[variable]

    # Collect errors by station
    station_errors = {}  # station_id -> list of (forecast - observed)

    # Process each run
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])
        model_data = run_data.get(model.lower())

        if not model_data:
            continue

        # Find index for this lead time
        if lead_time_hours not in forecast_hours:
            continue
        fcst_idx = forecast_hours.index(lead_time_hours)

        # Calculate valid time
        valid_time = init_time + timedelta(hours=lead_time_hours)

        # Only include past valid times
        if valid_time >= now:
            continue

        # Match forecasts with stored observations
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            # Get forecast value
            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue
            fcst_val = fcst_values[fcst_idx]

            # Get stored observation
            obs = get_stored_observation(db, station_id, valid_time)

            if obs is None or obs.get(obs_key) is None:
                continue
            obs_val = obs[obs_key]

            # Calculate error
            error = fcst_val - obs_val
            station_errors.setdefault(station_id, []).append(error)

    # Calculate metrics per station
    results = {}

    for station_id, errors in station_errors.items():
        if not errors:
            continue

        station = stations.get(station_id, {})

        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)

        results[station_id] = {
            'mae': round(mae, 2),
            'bias': round(bias, 2),
            'count': len(errors),
            'lat': station.get('lat'),
            'lon': station.get('lon'),
            'name': station.get('name', station_id),
            'state': station.get('state', '')
        }

    return results


def get_station_detail(station_id: str, model: str = None) -> dict:
    """
    Get detailed verification for a single station across all lead times.

    Uses stored observations from the database.

    Args:
        station_id: ASOS station ID
        model: Optional model filter (default: all models)

    Returns:
        Dict with lead time breakdown for each variable and model
    """
    from datetime import timezone

    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    station = stations.get(station_id)
    if not station:
        return {"error": "Station not found"}

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    models = [model.lower()] if model else ['gfs', 'aifs', 'ifs']

    # Collect all forecast hours we have
    all_forecast_hours = set()
    for run_data in runs.values():
        all_forecast_hours.update(run_data.get("forecast_hours", []))

    lead_times = sorted(all_forecast_hours)

    if not lead_times:
        return {
            "station": station,
            "lead_times": [],
            "data": {}
        }

    # Collect errors by lead time and model
    errors_by_lt = {lt: {m: {'temp': [], 'mslp': [], 'precip': []} for m in models} for lt in lead_times}

    # Match forecasts with stored observations
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        for m in models:
            model_data = run_data.get(m, {})
            fcst_data = model_data.get(station_id)

            if not fcst_data:
                continue

            for i, lt in enumerate(forecast_hours):
                valid_time = init_time + timedelta(hours=lt)
                if valid_time >= now:
                    continue

                # Get stored observation
                obs = get_stored_observation(db, station_id, valid_time)
                if not obs:
                    continue

                # Temperature
                fcst_temps = fcst_data.get('temps', [])
                if i < len(fcst_temps) and fcst_temps[i] is not None and obs.get('temp') is not None:
                    errors_by_lt[lt][m]['temp'].append(fcst_temps[i] - obs['temp'])

                # MSLP
                fcst_mslps = fcst_data.get('mslps', [])
                if i < len(fcst_mslps) and fcst_mslps[i] is not None and obs.get('mslp') is not None:
                    errors_by_lt[lt][m]['mslp'].append(fcst_mslps[i] - obs['mslp'])

                # Precip
                fcst_precips = fcst_data.get('precips', [])
                if i < len(fcst_precips) and fcst_precips[i] is not None and obs.get('precip') is not None:
                    errors_by_lt[lt][m]['precip'].append(fcst_precips[i] - obs['precip'])

    # Calculate metrics
    result_data = {}

    for lt in lead_times:
        result_data[lt] = {}
        for m in models:
            result_data[lt][m] = {}
            for var in ['temp', 'mslp', 'precip']:
                errors = errors_by_lt[lt][m][var]
                if errors:
                    result_data[lt][m][var] = {
                        'mae': round(sum(abs(e) for e in errors) / len(errors), 2),
                        'bias': round(sum(errors) / len(errors), 2),
                        'count': len(errors)
                    }
                else:
                    result_data[lt][m][var] = None

    return {
        "station": station,
        "lead_times": lead_times,
        "data": result_data
    }
