"""
ASOS (Automated Surface Observing System) Station Data Handler

Fetches ASOS station metadata and observations from the Iowa Environmental Mesonet (IEM).
Manages storage of model forecasts at ASOS station locations for verification.

IEM Data Sources (free, no authentication required):
- Station metadata: https://mesonet.agron.iastate.edu/geojson/network/{STATE}_ASOS.geojson
- Observations: http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Precipitation Handling:
- IEM provides p01i (1-hour precipitation increments)
- Model forecasts (GFS/AIFS/IFS) provide 6-hour accumulated precipitation
- For verification, 1-hour observations are accumulated into 6-hour totals at synoptic times
  (00Z, 06Z, 12Z, 18Z) to match model forecast periods
"""

import time
from functools import wraps
import urllib.request
import urllib.parse
import urllib.error # Added for specific error handling
import socket # Added for specific error handling
from rate_limiter import RateLimiter
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import json
import concurrent.futures # Import for ThreadPoolExecutor

logger = logging.getLogger(__name__)

# IEM Rate Limiter - 1 call per second
iem_rate_limiter = RateLimiter(calls_per_second=1)

@iem_rate_limiter
def _rate_limited_urlopen(*args, **kwargs):
    return urllib.request.urlopen(*args, **kwargs)

# Cache directory for station metadata
CACHE_DIR = Path.home() / ".cache" / "weather_models" / "asos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Station list cache file
STATIONS_CACHE_FILE = CACHE_DIR / "stations.json"
STATIONS_CACHE_TTL_DAYS = 7

# ASOS forecasts storage file (in project directory)
ASOS_FORECASTS_FILE = Path(__file__).parent / "asos_forecasts.json"

# ASOS verification cache file (precomputed stats)
ASOS_VERIFICATION_CACHE_FILE = Path(__file__).parent / "asos_verification_cache.json"

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
        with _rate_limited_urlopen(url, timeout=30) as response:
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

    except (urllib.error.URLError, socket.timeout) as e:
        logger.warning(f"Network error fetching stations for {state} from {url}: {e}")
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

            fetched_at_str = cache_data.get('fetched_at', '2000-01-01T00:00:00+00:00')
            cache_time = datetime.fromisoformat(fetched_at_str)
            if cache_time.tzinfo is None:
                cache_time = cache_time.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) - cache_time < timedelta(days=STATIONS_CACHE_TTL_DAYS):
                stations = [ASOSStation(**s) for s in cache_data.get('stations', [])]
                logger.info(f"Loaded {len(stations)} stations from cache")
                return stations
        except Exception as e:
            logger.warning(f"Error reading stations cache: {e}")

    # Fetch fresh data from IEM
    logger.info("Fetching ASOS stations from IEM...")
    all_stations = []

    # Use ThreadPoolExecutor to fetch states concurrently
    # The rate limiter on _rate_limited_urlopen will handle rate limiting
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each state
        future_to_state = {executor.submit(fetch_state_stations, state): state for state in US_STATES}
        for future in concurrent.futures.as_completed(future_to_state):
            state = future_to_state[future]
            try:
                stations = future.result()
                all_stations.extend(stations)
            except Exception as exc:
                logger.warning(f"Failed to fetch stations for {state}: {exc}")

    logger.info(f"Fetched {len(all_stations)} ASOS stations from IEM")

    # Save to cache
    try:
        cache_data = {
            'fetched_at': datetime.now(timezone.utc).isoformat(),
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
        with _rate_limited_urlopen(url, timeout=120) as response:
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
                valid_time = datetime.strptime(valid_str, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)

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

    except (urllib.error.URLError, socket.timeout) as e:
        logger.error(f"Network error fetching ASOS observations from {url}: {e}")
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


def _get_6hr_window_end(dt: datetime) -> datetime:
    """
    Calculates the end time of the 6-hour precipitation window for a given datetime.
    Windows end at 00Z, 06Z, 12Z, 18Z.
    An observation with valid_time = T (meaning precip from T-1h to T) contributes to the
    6-hour total ending at the nearest 6-hour mark (00, 06, 12, 18) that is >= T.
    """
    hour = dt.hour
    
    if hour < 6:
        target_hour = 6
        target_date = dt
    elif hour < 12:
        target_hour = 12
        target_date = dt
    elif hour < 18:
        target_hour = 18
        target_date = dt
    else: # hour >= 18
        target_hour = 0
        target_date = dt + timedelta(days=1)

    return datetime(target_date.year, target_date.month, target_date.day, target_hour, 0, 0, tzinfo=timezone.utc)


def load_asos_forecasts_db() -> dict:
    """Load the ASOS forecasts database from JSON file."""
    if ASOS_FORECASTS_FILE.exists():
        try:
            with open(ASOS_FORECASTS_FILE) as f:
                data = json.load(f)
                # Ensure cumulative_stats structure exists
                if "cumulative_stats" not in data:
                    data["cumulative_stats"] = {
                        "by_station": {},
                        "by_lead_time": {},
                        "time_series": {}
                    }
                # Ensure time_series exists in cumulative_stats
                if "time_series" not in data["cumulative_stats"]:
                    data["cumulative_stats"]["time_series"] = {}
                return data
        except Exception as e:
            logger.warning(f"Error loading asos_forecasts.json: {e}")
    return {
        "stations": {},
        "runs": {},
        "cumulative_stats": {
            "by_station": {},
            "by_lead_time": {},
            "time_series": {}
        }
    }


def save_asos_forecasts_db(data: dict):
    """Save the ASOS forecasts database to JSON file."""
    with open(ASOS_FORECASTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved ASOS forecasts to {ASOS_FORECASTS_FILE}")


def accumulate_stats_from_run(
    data: dict,
    run_key: str,
    run_data: dict
) -> None:
    """
    Accumulate statistics from a run into cumulative_stats before deletion.

    Updates both by_station and by_lead_time cumulative statistics, plus daily time series.

    Args:
        data: Full database dict
        run_key: Run ID (init_time ISO string)
        run_data: Run data dict with model forecasts
    """
    try:
        init_time = datetime.fromisoformat(run_key)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return

    now = datetime.now(timezone.utc)
    stations = data.get("stations", {})
    forecast_hours = run_data.get("forecast_hours", [])

    # Initialize cumulative stats if needed
    if "cumulative_stats" not in data:
        data["cumulative_stats"] = {"by_station": {}, "by_lead_time": {}, "time_series": {}}

    cumulative_by_station = data["cumulative_stats"]["by_station"]
    cumulative_by_lead_time = data["cumulative_stats"]["by_lead_time"]

    # Initialize time series structure if needed
    if "time_series" not in data["cumulative_stats"]:
        data["cumulative_stats"]["time_series"] = {}
    cumulative_time_series = data["cumulative_stats"]["time_series"]

    # Initialize time series for each model if needed
    for model in ['gfs', 'aifs', 'ifs']:
        if model not in cumulative_time_series:
            cumulative_time_series[model] = {}
        for var in ['temp', 'mslp', 'precip']:
            if var not in cumulative_time_series[model]:
                cumulative_time_series[model][var] = {}

    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr')
    }

    cumulative_by_station = data["cumulative_stats"]["by_station"]
    cumulative_by_lead_time = data["cumulative_stats"]["by_lead_time"]

    # Process each model
    for model in ['gfs', 'aifs', 'ifs']:
        model_data = run_data.get(model)
        if not model_data:
            continue

        # Initialize model in by_lead_time if needed
        if model not in cumulative_by_lead_time:
            cumulative_by_lead_time[model] = {}

        # Process each station
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            # Initialize station in by_station if needed
            if station_id not in cumulative_by_station:
                cumulative_by_station[station_id] = {}
            if model not in cumulative_by_station[station_id]:
                cumulative_by_station[station_id][model] = {}

            # Process each lead time
            for i, lt in enumerate(forecast_hours):
                valid_time = init_time + timedelta(hours=lt)

                # Only process past times
                if valid_time >= now:
                    continue

                # Get stored observation
                obs = get_stored_observation(data, station_id, valid_time)
                if not obs:
                    continue

                lt_str = str(lt)

                # Process each variable
                for var, (fcst_key, obs_key) in [
                    ('temp', ('temps', 'temp')),
                    ('mslp', ('mslps', 'mslp')),
                    ('precip', ('precips', 'precip_6hr'))  # Use 6-hour accumulated precip
                ]:
                    fcst_values = fcst_data.get(fcst_key, [])
                    if i >= len(fcst_values) or fcst_values[i] is None:
                        continue
                    if obs.get(obs_key) is None:
                        continue

                    fcst_val = fcst_values[i]
                    obs_val = obs[obs_key]
                    error = fcst_val - obs_val
                    abs_error = abs(error)

                    # Update by_station stats
                    if var not in cumulative_by_station[station_id][model]:
                        cumulative_by_station[station_id][model][var] = {}
                    if lt_str not in cumulative_by_station[station_id][model][var]:
                        cumulative_by_station[station_id][model][var][lt_str] = {
                            "sum_abs_errors": 0.0,
                            "sum_errors": 0.0,
                            "count": 0
                        }

                    cumulative_by_station[station_id][model][var][lt_str]["sum_abs_errors"] += abs_error
                    cumulative_by_station[station_id][model][var][lt_str]["sum_errors"] += error
                    cumulative_by_station[station_id][model][var][lt_str]["count"] += 1

                    # Update by_lead_time stats
                    if var not in cumulative_by_lead_time[model]:
                        cumulative_by_lead_time[model][var] = {}
                    if lt_str not in cumulative_by_lead_time[model][var]:
                        cumulative_by_lead_time[model][var][lt_str] = {
                            "sum_abs_errors": 0.0,
                            "sum_errors": 0.0,
                            "count": 0
                        }

                    cumulative_by_lead_time[model][var][lt_str]["sum_abs_errors"] += abs_error
                    cumulative_by_lead_time[model][var][lt_str]["sum_errors"] += error
                    cumulative_by_lead_time[model][var][lt_str]["count"] += 1

    # Accumulate time series data (daily errors by lead time)
    for model in ['gfs', 'aifs', 'ifs']:
        model_data = run_data.get(model)
        if not model_data:
            continue

        # Process each lead time
        for i, lt in enumerate(forecast_hours):
            valid_time = init_time + timedelta(hours=lt)

            # Only process past times
            if valid_time >= now:
                continue

            date_key = valid_time.date().isoformat()
            lt_str = str(lt)

            # Process each station
            for station_id, fcst_data in model_data.items():
                if station_id not in stations:
                    continue

                # Get stored observation
                obs = get_stored_observation(data, station_id, valid_time)
                if not obs:
                    continue

                # Process each variable
                for var, (fcst_key, obs_key) in var_map.items():
                    fcst_values = fcst_data.get(fcst_key, [])
                    if i >= len(fcst_values) or fcst_values[i] is None:
                        continue
                    if obs.get(obs_key) is None:
                        continue

                    fcst_val = fcst_values[i]
                    obs_val = obs[obs_key]
                    error = fcst_val - obs_val

                    # Initialize structures if needed
                    if lt_str not in cumulative_time_series[model][var]:
                        cumulative_time_series[model][var][lt_str] = {}
                    if date_key not in cumulative_time_series[model][var][lt_str]:
                        cumulative_time_series[model][var][lt_str][date_key] = []

                    # Append error to this date's list
                    cumulative_time_series[model][var][lt_str][date_key].append(error)


def cleanup_old_runs(data: dict) -> dict:
    """
    Remove forecast runs older than retention period.

    Before deleting runs, accumulates their statistics into cumulative_stats
    to preserve historical verification data.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=FORECASTS_RETENTION_DAYS)

    old_runs = []
    for run_id in data.get("runs", {}):
        try:
            run_time = datetime.fromisoformat(run_id)
            if run_time.tzinfo is None:
                run_time = run_time.replace(tzinfo=timezone.utc)  # Assume UTC for old naive datetimes
            if run_time < cutoff:
                old_runs.append(run_id)
        except ValueError:
            continue # Ignore invalid run_ids

    # Accumulate statistics from old runs before deleting them
    for run_id in old_runs:
        run_data = data["runs"][run_id]
        accumulate_stats_from_run(data, run_id, run_data)

    # Now delete the old runs
    for run_id in old_runs:
        del data["runs"][run_id]

    if old_runs:
        logger.info(f"Accumulated stats from {len(old_runs)} old runs before cleanup")

    # Also cleanup old observations
    obs_data = data.get("observations", {})
    for station_id in list(obs_data.keys()):
        station_obs = obs_data[station_id]
        old_times_to_delete = []
        for obs_time_str in station_obs:
            try:
                obs_time = datetime.fromisoformat(obs_time_str)
                if obs_time.tzinfo is None:
                    obs_time = obs_time.replace(tzinfo=timezone.utc)
                if obs_time < cutoff:
                    old_times_to_delete.append(obs_time_str)
            except ValueError:
                continue

        for t in old_times_to_delete:
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
    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    if not runs or not stations:
        logger.info("No forecast runs to fetch observations for")
        return 0

    now = datetime.now(timezone.utc)

    # Initialize observations dict if needed
    if "observations" not in db:
        db["observations"] = {}

    # Collect all valid times we need observations for
    valid_times_needed = set()

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
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

    # Fetch observations in chunks of stations concurrently
    station_ids = list(stations.keys())
    chunk_size = 50 # Reduce chunk size to be more granular with rate limiting and concurrency
    all_observations = {}
    
    # Using a smaller max_workers to avoid overwhelming the local system with too many open connections,
    # while still allowing some concurrency. The iem_rate_limiter further paces requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(0, len(station_ids), chunk_size):
            chunk_ids = station_ids[i:i+chunk_size]
            logger.info(f"Submitting observations fetch for stations {i+1}-{min(i+chunk_size, len(station_ids))} of {len(station_ids)}...")
            future = executor.submit(
                fetch_observations,
                chunk_ids,
                min_time,
                max_time,
                variables=['tmpf', 'mslp', 'p01i']
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                obs_chunk = future.result()
                all_observations.update(obs_chunk)
            except Exception as exc:
                logger.error(f"Observation fetch task generated an exception: {exc}")

    # Store observations keyed by station and valid time
    obs_count = 0

    for station_id, obs_list in all_observations.items():
        if station_id not in db["observations"]:
            db["observations"][station_id] = {}

        for obs in obs_list:
            obs_time_str = obs.get('valid_time')
            if not obs_time_str:
                continue

            # Store observation data (skip if all values are None)
            obs_data = {
                'temp': obs.get('temp'),
                'mslp': obs.get('mslp'),
                'precip': obs.get('precip')
            }

            # Only store if at least one value is not None
            if any(v is not None for v in obs_data.values()):
                db["observations"][station_id][obs_time_str] = obs_data
                obs_count += 1

    # Cleanup old data and save
    db = cleanup_old_runs(db)
    save_asos_forecasts_db(db)

    logger.info(f"Stored {obs_count} observations for {len(all_observations)} stations")

    # Precompute verification cache with updated observations
    logger.info("Precomputing verification cache with updated observations...")
    try:
        precompute_verification_cache()
    except Exception as e:
        logger.error(f"Error precomputing verification cache: {e}")

    return obs_count


def calculate_6hr_precip_total(db: dict, station_id: str, end_time: datetime) -> Optional[float]:
    """
    Calculate 6-hour accumulated precipitation ending at the specified time.

    Accumulates 1-hour precipitation observations over the 6-hour period ending
    at end_time. This is used to match model 6-hour precipitation forecasts.

    Args:
        db: The loaded database
        station_id: Station ID
        end_time: End of the 6-hour accumulation period

    Returns:
        6-hour precipitation total in inches, or None if insufficient data
    """
    station_obs = db.get("observations", {}).get(station_id, {})

    if not station_obs:
        return None

    # Collect all 1-hour precipitation values in the 6-hour window
    start_time = end_time - timedelta(hours=6)
    precip_values = []

    # Look for observations within each hour of the 6-hour period
    for hour_offset in range(6):
        # Target time for this hour (e.g., if end_time is 06Z, we want 01Z, 02Z, 03Z, 04Z, 05Z, 06Z)
        target = start_time + timedelta(hours=hour_offset + 1)

        # Find observation closest to this hour (within 30 minutes)
        best_obs = None
        best_delta = timedelta(minutes=31)

        for obs_time_str, obs_data in station_obs.items():
            try:
                obs_time = datetime.fromisoformat(obs_time_str)
                delta = abs(obs_time - target)

                if delta < best_delta:
                    best_delta = delta
                    best_obs = obs_data
            except ValueError:
                continue

        # If we found an observation within 30 minutes and it has precip data
        if best_obs and best_obs.get('precip') is not None:
            precip_values.append(best_obs['precip'])
        # else: missing data for this hour

    # Require at least 4 out of 6 hours to have data (allows some missing obs)
    if len(precip_values) < 4:
        return None

    # Sum up the 1-hour values to get 6-hour total
    return sum(precip_values)


def get_stored_observation(db: dict, station_id: str, target_time: datetime, max_delta_minutes: int = 30) -> Optional[dict]:
    """
    Get a stored observation for a station near the target time.

    For precipitation, this function calculates 6-hour accumulated totals to match
    model forecast accumulation periods. For temp and mslp, it returns point observations.

    Args:
        db: The loaded database
        station_id: Station ID
        target_time: Target valid time
        max_delta_minutes: Maximum time difference allowed

    Returns:
        Observation dict with keys: temp, mslp, precip (1-hr), precip_6hr (6-hr total)
        Returns None if no observation found
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
                best_match = obs_data.copy()  # Copy so we can add precip_6hr
        except ValueError:
            continue

    if best_delta <= timedelta(minutes=max_delta_minutes):
        # Add 6-hour accumulated precipitation
        # Check if target_time aligns with a 6-hour synoptic time (00Z, 06Z, 12Z, 18Z)
        if target_time.hour % 6 == 0:
            precip_6hr = calculate_6hr_precip_total(db, station_id, target_time)
            best_match['precip_6hr'] = precip_6hr
        else:
            # For non-synoptic times, don't calculate 6-hour total
            best_match['precip_6hr'] = None

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
    db = load_asos_forecasts_db()

    # Ensure stations are up to date
    stations = get_stations_dict()
    db["stations"] = stations

    # Create run entry if needed
    run_key = init_time.isoformat()
    if run_key not in db.get("runs", {}):
        db.setdefault("runs", {})[run_key] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
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

    Combines cumulative historical statistics with fresh calculations from
    current runs to provide lifetime MAE and bias per station.

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
    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})
    cumulative_by_station = db.get("cumulative_stats", {}).get("by_station", {})

    if not stations:
        return {}

    now = datetime.now(timezone.utc)

    # Map variable name to forecast/obs keys
    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr')  # Use 6-hour accumulated precip
    }

    if variable not in var_map:
        return {}

    fcst_key, obs_key = var_map[variable]
    lt_str = str(lead_time_hours)

    # Start with cumulative stats for this lead time
    station_stats = {}  # station_id -> {sum_abs_errors, sum_errors, count}

    # Load cumulative stats
    for station_id in stations:
        if station_id in cumulative_by_station:
            model_stats = cumulative_by_station[station_id].get(model.lower(), {})
            var_stats = model_stats.get(variable, {})
            lt_stats = var_stats.get(lt_str)
            if lt_stats:
                station_stats[station_id] = {
                    'sum_abs_errors': lt_stats.get('sum_abs_errors', 0.0),
                    'sum_errors': lt_stats.get('sum_errors', 0.0),
                    'count': lt_stats.get('count', 0)
                }

    # Add fresh calculations from current runs
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
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

            # Calculate error and add to running totals
            error = fcst_val - obs_val
            if station_id not in station_stats:
                station_stats[station_id] = {
                    'sum_abs_errors': 0.0,
                    'sum_errors': 0.0,
                    'count': 0
                }
            station_stats[station_id]['sum_abs_errors'] += abs(error)
            station_stats[station_id]['sum_errors'] += error
            station_stats[station_id]['count'] += 1

    # Calculate final metrics per station
    results = {}

    for station_id, stats in station_stats.items():
        if stats['count'] == 0:
            continue

        station = stations.get(station_id, {})

        mae = stats['sum_abs_errors'] / stats['count']
        bias = stats['sum_errors'] / stats['count']

        results[station_id] = {
            'mae': round(mae, 2),
            'bias': round(bias, 2),
            'count': stats['count'],
            'lat': station.get('lat'),
            'lon': station.get('lon'),
            'name': station.get('name', station_id),
            'state': station.get('state', '')
        }

    return results


def get_station_detail(station_id: str, model: str = None) -> dict:
    """
    Get detailed verification for a single station across all lead times.

    Combines cumulative historical statistics with fresh calculations from
    current runs to provide lifetime verification metrics.

    Args:
        station_id: ASOS station ID
        model: Optional model filter (default: all models)

    Returns:
        Dict with lead time breakdown for each variable and model
    """
    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})
    cumulative_by_station = db.get("cumulative_stats", {}).get("by_station", {})

    station = stations.get(station_id)
    if not station:
        raise ValueError("Station not found")

    now = datetime.now(timezone.utc)
    models = [model.lower()] if model else ['gfs', 'aifs', 'ifs']

    # Collect all forecast hours (from current runs and cumulative stats)
    all_forecast_hours = set()
    for run_data in runs.values():
        all_forecast_hours.update(run_data.get("forecast_hours", []))

    # Also include lead times from cumulative stats for this station
    station_cumulative = cumulative_by_station.get(station_id, {})
    for m in models:
        model_stats = station_cumulative.get(m, {})
        for var in ['temp', 'mslp', 'precip']:
            var_stats = model_stats.get(var, {})
            all_forecast_hours.update(int(lt) for lt in var_stats.keys())

    logger.debug(f"Raw all_forecast_hours for station detail: {sorted(list(all_forecast_hours))}")

    # Filter lead times: use all unique forecast hours in sorted order
    lead_times = sorted(list(all_forecast_hours)) # Ensure uniqueness and sort
    logger.debug(f"Lead times for station detail after removing filtering: {lead_times}")

    if not lead_times:
        return {
            "station": station,
            "lead_times": [],
            "data": {}
        }

    # Collect stats by lead time and model
    # Structure: {lt: {model: {var: {sum_abs_errors, sum_errors, count}}}}
    stats_by_lt = {
        lt: {
            m: {
                'temp': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
                'mslp': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
                'precip': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0}
            }
            for m in models
        }
        for lt in lead_times
    }

    # Start with cumulative stats
    for m in models:
        model_cumulative = station_cumulative.get(m, {})
        for var in ['temp', 'mslp', 'precip']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, stats in var_cumulative.items():
                lt = int(lt_str)
                if lt in stats_by_lt:
                    stats_by_lt[lt][m][var]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    stats_by_lt[lt][m][var]['sum_errors'] += stats.get('sum_errors', 0.0)
                    stats_by_lt[lt][m][var]['count'] += stats.get('count', 0)

    # Add fresh calculations from current runs
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
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
                    error = fcst_temps[i] - obs['temp']
                    stats_by_lt[lt][m]['temp']['sum_abs_errors'] += abs(error)
                    stats_by_lt[lt][m]['temp']['sum_errors'] += error
                    stats_by_lt[lt][m]['temp']['count'] += 1

                # MSLP
                fcst_mslps = fcst_data.get('mslps', [])
                if i < len(fcst_mslps) and fcst_mslps[i] is not None and obs.get('mslp') is not None:
                    error = fcst_mslps[i] - obs['mslp']
                    stats_by_lt[lt][m]['mslp']['sum_abs_errors'] += abs(error)
                    stats_by_lt[lt][m]['mslp']['sum_errors'] += error
                    stats_by_lt[lt][m]['mslp']['count'] += 1

                # Precip (6-hour accumulated)
                fcst_precips = fcst_data.get('precips', [])
                if i < len(fcst_precips) and fcst_precips[i] is not None and obs.get('precip_6hr') is not None:
                    error = fcst_precips[i] - obs['precip_6hr']
                    stats_by_lt[lt][m]['precip']['sum_abs_errors'] += abs(error)
                    stats_by_lt[lt][m]['precip']['sum_errors'] += error
                    stats_by_lt[lt][m]['precip']['count'] += 1

    # Calculate final metrics
    result_data = {}

    for lt in lead_times:
        result_data[lt] = {}
        for m in models:
            result_data[lt][m] = {}
            for var in ['temp', 'mslp', 'precip']:
                stats = stats_by_lt[lt][m][var]
                if stats['count'] > 0:
                    result_data[lt][m][var] = {
                        'mae': round(stats['sum_abs_errors'] / stats['count'], 2),
                        'bias': round(stats['sum_errors'] / stats['count'], 2),
                        'count': stats['count']
                    }
                else:
                    result_data[lt][m][var] = None

    return {
        "station": station,
        "lead_times": lead_times,
        "data": result_data
    }


def get_cumulative_stats_summary() -> dict:
    """
    Get a summary of cumulative statistics stored in the database.

    Returns:
        Dict with information about cumulative statistics coverage
    """
    db = load_asos_forecasts_db()
    cumulative_by_station = db.get("cumulative_stats", {}).get("by_station", {})
    cumulative_by_lead_time = db.get("cumulative_stats", {}).get("by_lead_time", {})

    # Count stations with cumulative data
    stations_with_data = len(cumulative_by_station)

    # Get total sample counts by model
    model_totals = {}
    for model in ['gfs', 'aifs', 'ifs']:
        model_data = cumulative_by_lead_time.get(model, {})
        total_count = 0
        for var in ['temp', 'mslp', 'precip']:
            var_data = model_data.get(var, {})
            for lt_stats in var_data.values():
                total_count += lt_stats.get('count', 0)
        model_totals[model] = total_count

    # Get lead time coverage
    lead_times_per_model = {}
    for model in ['gfs', 'aifs', 'ifs']:
        model_data = cumulative_by_lead_time.get(model, {})
        lead_times = set()
        for var in ['temp', 'mslp', 'precip']:
            var_data = model_data.get(var, {})
            lead_times.update(var_data.keys())
        lead_times_per_model[model] = sorted([int(lt) for lt in lead_times])

    return {
        "stations_with_cumulative_data": stations_with_data,
        "total_samples_by_model": model_totals,
        "lead_times_by_model": lead_times_per_model,
        "has_cumulative_data": stations_with_data > 0 or any(v > 0 for v in model_totals.values())
    }


def get_verification_time_series(
    model: str,
    variable: str,
    lead_time_hours: int,
    days_back: int = 30
) -> dict:
    """
    Get time series of verification metrics (MAE and Bias) over time for a specific
    model, variable, and lead time.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours
        days_back: Number of days to look back

    Returns:
        Dict with structure:
        {
            "dates": ["2026-01-20", "2026-01-21", ...],
            "mae": [2.1, 2.3, ...],
            "bias": [0.5, 0.3, ...],
            "counts": [150, 145, ...]
        }
    """
    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    if not stations or not runs:
        return {"error": "No data available"}

    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back)

    # Map variable name to forecast/obs keys
    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr')
    }

    if variable not in var_map:
        return {"error": "Invalid variable"}

    fcst_key, obs_key = var_map[variable]

    # Collect errors grouped by date
    # Key: date string (YYYY-MM-DD), Value: list of errors
    errors_by_date = {}

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        # Skip runs outside the time window
        if init_time < cutoff_date:
            continue

        forecast_hours = run_data.get("forecast_hours", [])
        if lead_time_hours not in forecast_hours:
            continue

        fcst_idx = forecast_hours.index(lead_time_hours)
        valid_time = init_time + timedelta(hours=lead_time_hours)

        # Only include past valid times
        if valid_time >= now:
            continue

        # Use the date of the valid time for grouping
        date_key = valid_time.date().isoformat()

        if date_key not in errors_by_date:
            errors_by_date[date_key] = []

        # Get model data
        model_data = run_data.get(model.lower())
        if not model_data:
            continue

        # Process each station
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue

            fcst_val = fcst_values[fcst_idx]

            # Get observation
            obs = get_stored_observation(db, station_id, valid_time)
            if obs is None or obs.get(obs_key) is None:
                continue

            obs_val = obs[obs_key]
            error = fcst_val - obs_val
            errors_by_date[date_key].append(error)

    # Calculate daily MAE and Bias
    dates = sorted(errors_by_date.keys())
    daily_mae = []
    daily_bias = []
    daily_counts = []

    for date in dates:
        errors = errors_by_date[date]
        if errors:
            mae = sum(abs(e) for e in errors) / len(errors)
            bias = sum(errors) / len(errors)
            daily_mae.append(mae)
            daily_bias.append(bias)
            daily_counts.append(len(errors))
        else:
            daily_mae.append(None)
            daily_bias.append(None)
            daily_counts.append(0)

    return {
        "dates": dates,
        "mae": [round(m, 2) if m is not None else None for m in daily_mae],
        "bias": [round(b, 2) if b is not None else None for b in daily_bias],
        "counts": daily_counts
    }


def get_mean_verification_by_lead_time(model: str) -> dict:
    """
    Get mean verification (MAE and Bias) across all stations by lead time for a given model.

    Combines cumulative historical statistics with fresh calculations from
    current runs to provide lifetime mean MAE and bias.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')

    Returns:
        Dict with structure:
        {
            "lead_times": [6, 12, 18, ...],
            "temp_mae": [avg_mae_lt6, avg_mae_lt12, ...],
            "temp_bias": [avg_bias_lt6, avg_bias_lt12, ...],
            "mslp_mae": [avg_mae_lt6, avg_mae_lt12, ...],
            "mslp_bias": [avg_bias_lt6, avg_bias_lt12, ...],
        }
    """
    try:
        db = load_asos_forecasts_db()
        stations = db.get("stations", {})
        runs = db.get("runs", {})
        cumulative_by_lead_time = db.get("cumulative_stats", {}).get("by_lead_time", {})

        if not stations:
            raise ValueError("No ASOS stations available.")

        now = datetime.now(timezone.utc)

        # Collect all forecast hours (from current runs and cumulative stats)
        all_forecast_hours = set()
        for run_data in runs.values():
            all_forecast_hours.update(run_data.get("forecast_hours", []))

        # Also include lead times from cumulative stats
        model_cumulative = cumulative_by_lead_time.get(model.lower(), {})
        for var in ['temp', 'mslp', 'precip']:
            var_stats = model_cumulative.get(var, {})
            all_forecast_hours.update(int(lt) for lt in var_stats.keys())

        logger.debug(f"Raw all_forecast_hours: {sorted(list(all_forecast_hours))}")

        # Filter lead times: use all unique forecast hours in sorted order
        lead_times = sorted(list(all_forecast_hours))
        logger.debug(f"Lead times after removing filtering: {lead_times}")

        if not lead_times:
            raise ValueError("No lead times found in ASOS forecast runs.")

        # Aggregate errors across all stations for each lead time and variable
        # Structure: {lead_time: {var: {sum_abs_errors, sum_errors, count}}}
        aggregated_stats = {
            lt: {
                'temp': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
                'mslp': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
                'precip': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0}
            }
            for lt in lead_times
        }

        # Start with cumulative stats
        for var in ['temp', 'mslp', 'precip']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, stats in var_cumulative.items():
                lt = int(lt_str)
                if lt in aggregated_stats:
                    aggregated_stats[lt][var]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    aggregated_stats[lt][var]['sum_errors'] += stats.get('sum_errors', 0.0)
                    aggregated_stats[lt][var]['count'] += stats.get('count', 0)

        # Add fresh calculations from current runs
        for run_key, run_data in runs.items():
            try:
                init_time = datetime.fromisoformat(run_key)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            forecast_hours = run_data.get("forecast_hours", [])
            model_data = run_data.get(model.lower())

            if not model_data:
                continue

            for station_id, fcst_data in model_data.items():
                if station_id not in stations:
                    continue

                for i, lt in enumerate(forecast_hours):
                    valid_time = init_time + timedelta(hours=lt)
                    if valid_time >= now:
                        continue

                    obs = get_cached_observation(station_id, valid_time)
                    if not obs:
                        continue

                    # Temperature
                    fcst_temps = fcst_data.get('temps', [])
                    if i < len(fcst_temps) and fcst_temps[i] is not None and obs.get('temp') is not None:
                        error = fcst_temps[i] - obs['temp']
                        aggregated_stats[lt]['temp']['sum_abs_errors'] += abs(error)
                        aggregated_stats[lt]['temp']['sum_errors'] += error
                        aggregated_stats[lt]['temp']['count'] += 1

                    # MSLP
                    fcst_mslps = fcst_data.get('mslps', [])
                    if i < len(fcst_mslps) and fcst_mslps[i] is not None and obs.get('mslp') is not None:
                        error = fcst_mslps[i] - obs['mslp']
                        aggregated_stats[lt]['mslp']['sum_abs_errors'] += abs(error)
                        aggregated_stats[lt]['mslp']['sum_errors'] += error
                        aggregated_stats[lt]['mslp']['count'] += 1

                    # Precipitation (6-hour accumulated)
                    fcst_precips = fcst_data.get('precips', [])
                    if i < len(fcst_precips) and fcst_precips[i] is not None and obs.get('precip_6hr') is not None:
                        error = fcst_precips[i] - obs['precip_6hr']
                        aggregated_stats[lt]['precip']['sum_abs_errors'] += abs(error)
                        aggregated_stats[lt]['precip']['sum_errors'] += error
                        aggregated_stats[lt]['precip']['count'] += 1

        # Calculate mean MAE and Bias for each lead time
        result = {
            "lead_times": lead_times,
            "temp_mae": [],
            "temp_bias": [],
            "mslp_mae": [],
            "mslp_bias": [],
            "precip_mae": [],
            "precip_bias": []
        }

        for lt in lead_times:
            # Temperature
            temp_stats = aggregated_stats[lt]['temp']
            if temp_stats['count'] > 0:
                result["temp_mae"].append(round(temp_stats['sum_abs_errors'] / temp_stats['count'], 2))
                result["temp_bias"].append(round(temp_stats['sum_errors'] / temp_stats['count'], 2))
            else:
                result["temp_mae"].append(None)
                result["temp_bias"].append(None)

            # MSLP
            mslp_stats = aggregated_stats[lt]['mslp']
            if mslp_stats['count'] > 0:
                result["mslp_mae"].append(round(mslp_stats['sum_abs_errors'] / mslp_stats['count'], 2))
                result["mslp_bias"].append(round(mslp_stats['sum_errors'] / mslp_stats['count'], 2))
            else:
                result["mslp_mae"].append(None)
                result["mslp_bias"].append(None)

            # Precipitation
            precip_stats = aggregated_stats[lt]['precip']
            if precip_stats['count'] > 0:
                result["precip_mae"].append(round(precip_stats['sum_abs_errors'] / precip_stats['count'], 2))
                result["precip_bias"].append(round(precip_stats['sum_errors'] / precip_stats['count'], 2))
            else:
                result["precip_mae"].append(None)
                result["precip_bias"].append(None)

        return result
    except Exception as e:
        logger.error(f"DEBUG: Exception in get_mean_verification_by_lead_time: Type: {type(e).__name__}, Value: {e}")
        raise


def precompute_verification_cache() -> dict:
    """
    Precompute all verification statistics and save to cache file.

    This should be called after fetching observations during sync to update
    the verification cache with the latest data. This eliminates the need to
    compute stats on-the-fly when the verification page loads.

    Returns:
        Dict with precomputed verification data
    """
    logger.info("Precomputing verification cache...")
    start_time = time.time()

    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})
    cumulative_by_station = db.get("cumulative_stats", {}).get("by_station", {})
    cumulative_by_lead_time = db.get("cumulative_stats", {}).get("by_lead_time", {})

    if not stations:
        logger.warning("No stations available for verification cache")
        return {}

    now = datetime.now(timezone.utc)

    # Collect all forecast hours
    all_forecast_hours = set()
    for run_data in runs.values():
        all_forecast_hours.update(run_data.get("forecast_hours", []))

    # Also include lead times from cumulative stats
    for model in ['gfs', 'aifs', 'ifs']:
        model_cumulative = cumulative_by_lead_time.get(model, {})
        for var in ['temp', 'mslp', 'precip']:
            var_stats = model_cumulative.get(var, {})
            all_forecast_hours.update(int(lt) for lt in var_stats.keys())

    lead_times = sorted(list(all_forecast_hours))

    if not lead_times:
        logger.warning("No lead times available for verification cache")
        return {}

    # Build fast observation lookup cache (without 6hr precip pre-calculation)
    logger.info("Building observation lookup cache...")
    obs_cache = {}  # {station_id: {valid_time_str: obs_dict}}
    observations_data = db.get("observations", {})

    for station_id, station_obs in observations_data.items():
        obs_cache[station_id] = station_obs

    logger.info(f"Built observation cache for {len(obs_cache)} stations")

    # Cache for 6hr precip calculations to avoid recomputing
    precip_6hr_cache = {}  # {(station_id, time_str): value}

    # Helper function for fast observation lookup
    def get_cached_observation(station_id: str, target_time: datetime, max_delta_minutes: int = 30):
        """Fast observation lookup using pre-built cache."""
        station_cache = obs_cache.get(station_id, {})
        if not station_cache:
            return None

        best_match = None
        best_delta = timedelta(minutes=max_delta_minutes + 1)
        target_time_str = target_time.isoformat()

        # Quick check for exact match first
        if target_time_str in station_cache:
            best_match = station_cache[target_time_str].copy()
        else:
            # Find nearest within tolerance
            for obs_time_str, obs_data in station_cache.items():
                try:
                    obs_time = datetime.fromisoformat(obs_time_str)
                    delta = abs(obs_time - target_time)
                    if delta < best_delta:
                        best_delta = delta
                        best_match = obs_data.copy()
                except ValueError:
                    continue

        if best_match and best_delta <= timedelta(minutes=max_delta_minutes):
            # Calculate 6hr precip on demand for synoptic times only
            if target_time.hour % 6 == 0:
                cache_key = (station_id, target_time_str)
                if cache_key in precip_6hr_cache:
                    best_match['precip_6hr'] = precip_6hr_cache[cache_key]
                else:
                    precip_6hr = calculate_6hr_precip_total(db, station_id, target_time)
                    precip_6hr_cache[cache_key] = precip_6hr
                    best_match['precip_6hr'] = precip_6hr
            else:
                best_match['precip_6hr'] = None
            return best_match

        return None

    # Initialize cache structure
    cache_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "lead_times": lead_times,
        "by_station": {},
        "by_lead_time": {},
        "time_series": {},
        "stations": stations
    }

    # Map variable name to forecast/obs keys
    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr')
    }

    # Initialize aggregated stats for by_station and by_lead_time
    # Structure: station_stats[station_id][model][var][lt_str] = {sum_abs_errors, sum_errors, count}
    station_stats = {}
    for station_id in stations:
        station_stats[station_id] = {
            'gfs': {'temp': {}, 'mslp': {}, 'precip': {}},
            'aifs': {'temp': {}, 'mslp': {}, 'precip': {}},
            'ifs': {'temp': {}, 'mslp': {}, 'precip': {}}
        }

    # Structure: aggregated_stats[model][var][lt_str] = {sum_abs_errors, sum_errors, count}
    aggregated_stats = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}}
    }

    # Initialize with zeros for all lead times
    for model in ['gfs', 'aifs', 'ifs']:
        for var in ['temp', 'mslp', 'precip']:
            for lt in lead_times:
                lt_str = str(lt)
                aggregated_stats[model][var][lt_str] = {
                    'sum_abs_errors': 0.0,
                    'sum_errors': 0.0,
                    'count': 0
                }
                for station_id in stations:
                    station_stats[station_id][model][var][lt_str] = {
                        'sum_abs_errors': 0.0,
                        'sum_errors': 0.0,
                        'count': 0
                    }

    logger.info(f"Loading cumulative stats for {len(stations)} stations...")

    # Load cumulative stats
    for station_id in stations:
        if station_id in cumulative_by_station:
            for model in ['gfs', 'aifs', 'ifs']:
                model_cumulative = cumulative_by_station[station_id].get(model, {})
                for var in ['temp', 'mslp', 'precip']:
                    var_cumulative = model_cumulative.get(var, {})
                    for lt_str, stats in var_cumulative.items():
                        if int(lt_str) in lead_times:
                            station_stats[station_id][model][var][lt_str]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                            station_stats[station_id][model][var][lt_str]['sum_errors'] += stats.get('sum_errors', 0.0)
                            station_stats[station_id][model][var][lt_str]['count'] += stats.get('count', 0)

    for model in ['gfs', 'aifs', 'ifs']:
        model_cumulative = cumulative_by_lead_time.get(model, {})
        for var in ['temp', 'mslp', 'precip']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, stats in var_cumulative.items():
                if int(lt_str) in lead_times:
                    aggregated_stats[model][var][lt_str]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    aggregated_stats[model][var][lt_str]['sum_errors'] += stats.get('sum_errors', 0.0)
                    aggregated_stats[model][var][lt_str]['count'] += stats.get('count', 0)

    logger.info(f"Computing fresh stats from {len(runs)} current runs...")

    # Add fresh calculations from current runs
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        for model in ['gfs', 'aifs', 'ifs']:
            model_data = run_data.get(model)
            if not model_data:
                continue

            for station_id, fcst_data in model_data.items():
                if station_id not in stations:
                    continue

                for i, lt in enumerate(forecast_hours):
                    if lt not in lead_times:
                        continue

                    valid_time = init_time + timedelta(hours=lt)
                    if valid_time >= now:
                        continue

                    obs = get_cached_observation(station_id, valid_time)
                    if not obs:
                        continue

                    lt_str = str(lt)

                    # Process each variable
                    for var, (fcst_key, obs_key) in var_map.items():
                        fcst_values = fcst_data.get(fcst_key, [])
                        if i >= len(fcst_values) or fcst_values[i] is None:
                            continue
                        if obs.get(obs_key) is None:
                            continue

                        fcst_val = fcst_values[i]
                        obs_val = obs[obs_key]
                        error = fcst_val - obs_val
                        abs_error = abs(error)

                        # Update station stats
                        station_stats[station_id][model][var][lt_str]['sum_abs_errors'] += abs_error
                        station_stats[station_id][model][var][lt_str]['sum_errors'] += error
                        station_stats[station_id][model][var][lt_str]['count'] += 1

                        # Update aggregated stats
                        aggregated_stats[model][var][lt_str]['sum_abs_errors'] += abs_error
                        aggregated_stats[model][var][lt_str]['sum_errors'] += error
                        aggregated_stats[model][var][lt_str]['count'] += 1

    logger.info("Finalizing cache data...")

    # Precompute time series data (all historical data) using cached observations
    logger.info("Computing verification time series with cached observations...")
    time_series_data = precompute_verification_time_series(db, lead_times, obs_cache=obs_cache, get_cached_obs_fn=get_cached_observation)

    # Convert to final cache format - by_station
    for station_id in stations:
        cache_data["by_station"][station_id] = {}
        for model in ['gfs', 'aifs', 'ifs']:
            cache_data["by_station"][station_id][model] = {}
            for lt in lead_times:
                lt_str = str(lt)
                cache_data["by_station"][station_id][model][lt_str] = {}
                for var in ['temp', 'mslp', 'precip']:
                    stats = station_stats[station_id][model][var][lt_str]
                    if stats['count'] > 0:
                        cache_data["by_station"][station_id][model][lt_str][var] = {
                            'mae': round(stats['sum_abs_errors'] / stats['count'], 2),
                            'bias': round(stats['sum_errors'] / stats['count'], 2),
                            'count': stats['count']
                        }
                    else:
                        cache_data["by_station"][station_id][model][lt_str][var] = None

    # Convert to final cache format - by_lead_time
    for model in ['gfs', 'aifs', 'ifs']:
        cache_data["by_lead_time"][model] = {}
        for lt in lead_times:
            lt_str = str(lt)
            cache_data["by_lead_time"][model][lt_str] = {}
            for var in ['temp', 'mslp', 'precip']:
                stats = aggregated_stats[model][var][lt_str]
                if stats['count'] > 0:
                    cache_data["by_lead_time"][model][lt_str][var] = {
                        'mae': round(stats['sum_abs_errors'] / stats['count'], 2),
                        'bias': round(stats['sum_errors'] / stats['count'], 2)
                    }
                else:
                    cache_data["by_lead_time"][model][lt_str][var] = {
                        'mae': None,
                        'bias': None
                    }

    # Add time series data to cache
    cache_data["time_series"] = time_series_data

    # Save to file
    try:
        with open(ASOS_VERIFICATION_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Verification cache precomputed and saved in {elapsed:.1f}s to {ASOS_VERIFICATION_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving verification cache: {e}")

    return cache_data


def load_verification_cache() -> Optional[dict]:
    """
    Load precomputed verification cache from file.

    Returns:
        Dict with cached verification data, or None if cache doesn't exist
    """
    if not ASOS_VERIFICATION_CACHE_FILE.exists():
        logger.warning(f"Verification cache file not found: {ASOS_VERIFICATION_CACHE_FILE}")
        return None

    try:
        with open(ASOS_VERIFICATION_CACHE_FILE) as f:
            cache_data = json.load(f)

        last_updated = cache_data.get("last_updated", "unknown")
        logger.info(f"Loaded verification cache (last updated: {last_updated})")
        return cache_data
    except Exception as e:
        logger.error(f"Error loading verification cache: {e}")
        return None


def precompute_verification_time_series(db: dict, lead_times: list, days_back: int = None, obs_cache: dict = None, get_cached_obs_fn=None) -> dict:
    """
    Precompute verification time series data for all models, variables, and lead times.

    Combines cumulative historical data (all time) with fresh calculations from current runs
    to provide complete historical daily time series.

    Args:
        db: Loaded ASOS database
        lead_times: List of lead times to compute for
        days_back: How many days back to compute from current runs (None = all available)
        obs_cache: Prebuilt observation cache (optional, for performance)
        get_cached_obs_fn: Function to get cached observations (optional, for performance)

    Returns:
        Dict with time series data: {model: {variable: {lead_time: {dates, mae, bias, counts}}}}
    """
    logger.info("Precomputing verification time series (all historical data)...")

    stations = db.get("stations", {})
    runs = db.get("runs", {})
    cumulative_time_series = db.get("cumulative_stats", {}).get("time_series", {})

    if not stations:
        return {}

    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days_back) if days_back else datetime.min.replace(tzinfo=timezone.utc)

    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr')
    }

    # Structure: time_series[model][var][lt][date] = [errors]
    time_series = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}}
    }

    # Initialize for all lead times
    for model in ['gfs', 'aifs', 'ifs']:
        for var in ['temp', 'mslp', 'precip']:
            for lt in lead_times:
                time_series[model][var][lt] = {}

    # Load cumulative time series data (historical data from deleted runs)
    logger.info("Loading cumulative time series data...")
    for model in ['gfs', 'aifs', 'ifs']:
        model_cumulative = cumulative_time_series.get(model, {})
        for var in ['temp', 'mslp', 'precip']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, date_errors in var_cumulative.items():
                lt = int(lt_str)
                if lt in lead_times:
                    # Copy the historical date->errors mapping
                    time_series[model][var][lt] = dict(date_errors)

    # Collect errors by date from current runs
    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if init_time < cutoff_date:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        for model in ['gfs', 'aifs', 'ifs']:
            model_data = run_data.get(model)
            if not model_data:
                continue

            for lt in lead_times:
                if lt not in forecast_hours:
                    continue

                fcst_idx = forecast_hours.index(lt)
                valid_time = init_time + timedelta(hours=lt)

                if valid_time >= now:
                    continue

                date_key = valid_time.date().isoformat()

                # Process each station
                for station_id, fcst_data in model_data.items():
                    if station_id not in stations:
                        continue

                    # Use cached observation lookup if available, otherwise fall back to regular
                    if get_cached_obs_fn:
                        obs = get_cached_obs_fn(station_id, valid_time)
                    else:
                        obs = get_stored_observation(db, station_id, valid_time)

                    if not obs:
                        continue

                    # Process each variable
                    for var, (fcst_key, obs_key) in var_map.items():
                        fcst_values = fcst_data.get(fcst_key, [])
                        if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                            continue
                        if obs.get(obs_key) is None:
                            continue

                        fcst_val = fcst_values[fcst_idx]
                        obs_val = obs[obs_key]
                        error = fcst_val - obs_val

                        # Store error by date (append to any existing cumulative data)
                        if date_key not in time_series[model][var][lt]:
                            time_series[model][var][lt][date_key] = []
                        time_series[model][var][lt][date_key].append(error)

    # Convert to final format with MAE/bias per day
    result = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}}
    }

    for model in ['gfs', 'aifs', 'ifs']:
        for var in ['temp', 'mslp', 'precip']:
            for lt in lead_times:
                errors_by_date = time_series[model][var][lt]

                if not errors_by_date:
                    continue

                dates = sorted(errors_by_date.keys())
                daily_mae = []
                daily_bias = []
                daily_counts = []

                for date in dates:
                    errors = errors_by_date[date]
                    if errors:
                        mae = sum(abs(e) for e in errors) / len(errors)
                        bias = sum(errors) / len(errors)
                        daily_mae.append(round(mae, 2))
                        daily_bias.append(round(bias, 2))
                        daily_counts.append(len(errors))
                    else:
                        daily_mae.append(None)
                        daily_bias.append(None)
                        daily_counts.append(0)

                result[model][var][lt] = {
                    'dates': dates,
                    'mae': daily_mae,
                    'bias': daily_bias,
                    'counts': daily_counts
                }

    return result


def get_verification_data_from_cache(
    model: str,
    variable: str,
    lead_time_hours: int
) -> Dict[str, dict]:
    """
    Get verification data from cache for a specific model/variable/lead_time.

    Falls back to computing on-the-fly if cache doesn't exist.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours

    Returns:
        Dict mapping station_id to verification data (same format as get_verification_data)
    """
    cache = load_verification_cache()

    if cache is None:
        logger.warning("Cache not available, computing verification on-the-fly")
        return get_verification_data(model, variable, lead_time_hours)

    # Extract from cache
    lt_str = str(lead_time_hours)
    results = {}

    for station_id, station_data in cache.get("by_station", {}).items():
        model_data = station_data.get(model.lower(), {})
        lt_data = model_data.get(lt_str, {})
        var_data = lt_data.get(variable)

        if var_data:
            station_info = cache.get("stations", {}).get(station_id, {})
            results[station_id] = {
                'mae': var_data['mae'],
                'bias': var_data['bias'],
                'count': var_data['count'],
                'lat': station_info.get('lat'),
                'lon': station_info.get('lon'),
                'name': station_info.get('name', station_id),
                'state': station_info.get('state', '')
            }

    return results


def get_mean_verification_from_cache(model: str) -> dict:
    """
    Get mean verification from cache for a specific model.

    Falls back to computing on-the-fly if cache doesn't exist.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')

    Returns:
        Dict with mean verification data (same format as get_mean_verification_by_lead_time)
    """
    cache = load_verification_cache()

    if cache is None:
        logger.warning("Cache not available, computing verification on-the-fly")
        return get_mean_verification_by_lead_time(model)

    # Extract from cache
    lead_times = cache.get("lead_times", [])
    model_data = cache.get("by_lead_time", {}).get(model.lower(), {})

    result = {
        "lead_times": lead_times,
        "temp_mae": [],
        "temp_bias": [],
        "mslp_mae": [],
        "mslp_bias": [],
        "precip_mae": [],
        "precip_bias": []
    }

    for lt in lead_times:
        lt_str = str(lt)
        lt_data = model_data.get(lt_str, {})

        result["temp_mae"].append(lt_data.get("temp", {}).get("mae"))
        result["temp_bias"].append(lt_data.get("temp", {}).get("bias"))
        result["mslp_mae"].append(lt_data.get("mslp", {}).get("mae"))
        result["mslp_bias"].append(lt_data.get("mslp", {}).get("bias"))
        result["precip_mae"].append(lt_data.get("precip", {}).get("mae"))
        result["precip_bias"].append(lt_data.get("precip", {}).get("bias"))

    return result


def get_verification_time_series_from_cache(
    model: str,
    variable: str,
    lead_time_hours: int,
    days_back: int = 30
) -> dict:
    """
    Get verification time series from cache for a specific model/variable/lead_time.

    Falls back to computing on-the-fly if cache doesn't exist.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours
        days_back: Number of days to include (default 30)

    Returns:
        Dict with time series data (same format as get_verification_time_series)
    """
    cache = load_verification_cache()

    if cache is None:
        logger.warning("Cache not available, computing time series on-the-fly")
        return get_verification_time_series(model, variable, lead_time_hours, days_back)

    # Extract from cache
    time_series = cache.get("time_series", {})
    model_data = time_series.get(model.lower(), {})
    var_data = model_data.get(variable, {})
    lt_data = var_data.get(str(lead_time_hours))

    if not lt_data:
        # No data for this combination, return empty result
        return {
            "dates": [],
            "mae": [],
            "bias": [],
            "counts": []
        }

    # Get all dates and slice to requested days_back
    all_dates = lt_data.get('dates', [])
    all_mae = lt_data.get('mae', [])
    all_bias = lt_data.get('bias', [])
    all_counts = lt_data.get('counts', [])

    if not all_dates:
        return {
            "dates": [],
            "mae": [],
            "bias": [],
            "counts": []
        }

    # Filter to last N days
    try:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()

        # Find the index where dates >= cutoff_date
        start_idx = 0
        for i, date in enumerate(all_dates):
            if date >= cutoff_date:
                start_idx = i
                break

        return {
            "dates": all_dates[start_idx:],
            "mae": all_mae[start_idx:],
            "bias": all_bias[start_idx:],
            "counts": all_counts[start_idx:]
        }
    except Exception as e:
        logger.error(f"Error filtering time series data: {e}")
        return {
            "dates": all_dates,
            "mae": all_mae,
            "bias": all_bias,
            "counts": all_counts
        }


def _calculate_asos_mean_mae(
    db: dict,
    run_time_str: str,
    model: str,
    variable: str,
    lead_time_hours: int
) -> Optional[float]:
    """
    Calculate mean MAE across all ASOS stations for a specific run, model, variable, and lead time.

    Args:
        db: ASOS forecasts database
        run_time_str: Run time as ISO string (e.g., "2026-02-04T00:00:00")
        model: Model name ('gfs', 'aifs', 'ifs')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours

    Returns:
        Mean MAE across all stations, or None if insufficient data
    """
    try:
        run_data = db.get("runs", {}).get(run_time_str)
        if not run_data:
            return None

        forecast_hours = run_data.get("forecast_hours", [])
        if lead_time_hours not in forecast_hours:
            return None

        fcst_idx = forecast_hours.index(lead_time_hours)

        # Map variable to keys
        var_map = {
            'temp': ('temps', 'temp'),
            'mslp': ('mslps', 'mslp'),
            'precip': ('precips', 'precip_6hr')
        }

        if variable not in var_map:
            return None

        fcst_key, obs_key = var_map[variable]

        # Calculate valid time for this forecast
        init_time = datetime.fromisoformat(run_time_str)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)

        valid_time = init_time + timedelta(hours=lead_time_hours)

        # Skip if valid time is in the future
        now = datetime.now(timezone.utc)
        if valid_time >= now:
            return None

        # Collect errors from all stations
        errors = []
        stations = db.get("stations", {})
        model_data = run_data.get(model.lower())

        if not model_data:
            return None

        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue

            fcst_val = fcst_values[fcst_idx]

            # Get observation
            obs = get_stored_observation(db, station_id, valid_time)
            if obs is None or obs.get(obs_key) is None:
                continue

            obs_val = obs[obs_key]

            # Calculate error
            error = abs(fcst_val - obs_val)
            errors.append(error)

        # Return mean MAE if we have at least 10 stations
        if len(errors) >= 10:
            return sum(errors) / len(errors)
        else:
            return None

    except Exception as e:
        logger.error(f"Error calculating ASOS mean MAE: {e}")
        return None
