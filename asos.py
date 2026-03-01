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
import math
from rate_limiter import RateLimiter
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import json
import bisect
import os

logger = logging.getLogger(__name__)

# Stations known to report cumulative or unreliable precip that breaks 6-hr totals
PRECIP_EXCLUDE_STATIONS = {"SMD", "ADF", "PVW"}  # ADF gauge stuck at 2.56/5.12" (tipping bucket overflow); PVW unreliable precip
PRESSURE_EXCLUDE_STATIONS = {"TAH", "PVW", "TCM"}  # Persistent unrealistic altimeter/MSLP readings

# Stations that report p01i in millimeters instead of the standard inches.
# Verified by observing hourly maxima of 3–19 in the database — impossible as inches
# but physically reasonable as mm (0.12–0.77").  All other PA* Alaska stations top
# out at ≤0.48", confirming PAAK is the sole outlier.
# Their raw sums are divided by 25.4 before being returned.
PRECIP_MM_STATIONS = {"PAAK"}

# ASOS tipping-bucket gauges overflow at 256 tips × 0.01"/tip = 2.56 inches.
# When the internal counter wraps the sensor reports 2.56, 5.12, 7.68, … inches
# rather than the true accumulation.  These are not real precipitation and must
# be discarded before computing hourly maxima.
_TIPPING_BUCKET_OVERFLOW_IN = 2.56
_MAX_REASONABLE_P01I_IN = 3.0
_MAX_REASONABLE_P01I_MM = _MAX_REASONABLE_P01I_IN * 25.4


def _is_overflow_value(v: float) -> bool:
    """Return True if v is an exact multiple of the tipping-bucket overflow increment."""
    if v <= 0:
        return False
    ratio = v / _TIPPING_BUCKET_OVERFLOW_IN
    return abs(ratio - round(ratio)) < 1e-4


def _is_implausible_precip_value(station_id: str, v: float) -> bool:
    """
    Return True for clearly bad hourly p01i values that should be excluded.

    Most stations report p01i in inches; a small set report mm (PRECIP_MM_STATIONS).
    """
    if not math.isfinite(v) or v < 0:
        return True
    max_allowed = _MAX_REASONABLE_P01I_MM if station_id in PRECIP_MM_STATIONS else _MAX_REASONABLE_P01I_IN
    return v > max_allowed


# Physically plausible range for MSLP (sea-level pressure in millibars).
# World record low: ~870 mb (Typhoon Tip, 1979); world record high: ~1084 mb (Siberia, 2001).
# Values outside this range are sensor errors, unit mismatches, or default/uninitialized readings.
_MSLP_MIN_MB = 870.0
_MSLP_MAX_MB = 1090.0

# Physically plausible range for temperature and dewpoint (°F).
# Temp: world record low ~-129°F (Vostok), high ~130°F (Death Valley).
# Dewpoint: bounded above by ~95°F (highest ever recorded globally).
_TEMP_MIN_F = -130.0
_TEMP_MAX_F = 135.0
_DEWPOINT_MIN_F = -130.0
_DEWPOINT_MAX_F = 100.0


def _is_implausible_pressure_value(val_mb: float) -> bool:
    """Return True if a pressure reading (in mb/hPa) is physically implausible."""
    return not math.isfinite(val_mb) or val_mb < _MSLP_MIN_MB or val_mb > _MSLP_MAX_MB


def _is_implausible_temp_value(val_f: float) -> bool:
    """Return True if a temperature reading (in °F) is physically implausible."""
    return not math.isfinite(val_f) or val_f < _TEMP_MIN_F or val_f > _TEMP_MAX_F


def _is_implausible_dewpoint_value(val_f: float) -> bool:
    """Return True if a dewpoint reading (in °F) is physically implausible."""
    return not math.isfinite(val_f) or val_f < _DEWPOINT_MIN_F or val_f > _DEWPOINT_MAX_F


def should_include_precip(fcst_val, obs_val) -> bool:
    """
    Include precip verification only for meaningful precipitation events.

    Excludes zero-zero pairs where both forecast and observation are zero, because:
    - They inflate sample size without adding information about precipitation skill
    - MAE should represent error in actual precipitation amounts, not dry periods
    - Correctly predicting "no rain" when there's no rain is trivial

    Includes:
    - Forecast > 0, Observed = 0 (false alarm)
    - Forecast = 0, Observed > 0 (missed precipitation)
    - Forecast > 0, Observed > 0 (verify amount)

    Excludes:
    - Forecast = 0, Observed = 0 (both dry, no skill test)

    NOTE: Batch verification functions (precompute_verification_cache,
    accumulate_stats_from_run, etc.) use _qualifying_precip_sets() instead,
    which triggers on ANY model forecasting >= 0.01" so all models are scored
    on the same set of events. This function is kept for on-the-fly single-model queries.
    """
    if fcst_val is None or obs_val is None:
        return False
    # Exclude trace-and-below pairs: both values under 0.01" are either
    # genuine dry periods or sub-hundredth model noise — not meaningful skill tests.
    if max(fcst_val, obs_val) < 0.01:
        return False
    return True


def _is_precip_var(variable: str) -> bool:
    return variable in ("precip", "precip_24hr")


def _precip_weight(obs_val: float) -> float:
    # Keep light events in-scope while increasing influence as observed precip grows.
    return max(float(obs_val), 0.01)


def _stats_wmae(stats: dict, fallback_mae: Optional[float] = None) -> Optional[float]:
    weight_sum = stats.get("sum_weights", 0.0) or 0.0
    if weight_sum > 0:
        return (stats.get("sum_weighted_abs_errors", 0.0) or 0.0) / weight_sum
    return fallback_mae


ASOS_BASE_MODELS = ("gfs", "aifs", "ifs", "nws")
ASOS_VERIFICATION_MODELS = ("gfs", "aifs", "kenny", "ifs", "nws")


def _source_model_for_verification(model: str) -> str:
    m = (model or "").lower()
    return "aifs" if m == "kenny" else m


def _is_bias_corrected_model(model: str) -> bool:
    return (model or "").lower() == "kenny"


def _bias_corrected_metric_value(model: str, metric_name: str, mae_value, raw_value):
    return raw_value


def _is_kenny_bias_corrected_var(model: str, variable: str) -> bool:
    return (model or "").lower() == "kenny" and variable in ("temp", "dewpoint")


def _compute_kenny_station_hour_biases(db: dict, variable: str = "temp") -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """
    Compute AIFS 6-hour variable bias by station and valid time-of-day hour.

    Returns:
      - station_hour_biases: {station_id: {valid_hour: mean_bias}}
      - global_hour_biases: {valid_hour: mean_bias}  # fallback when station is sparse
    """
    if variable not in ("temp", "dewpoint"):
        return {}, {0: 0.0, 6: 0.0, 12: 0.0, 18: 0.0}

    fcst_key = "temps" if variable == "temp" else "dewpoints"
    obs_key = "temp" if variable == "temp" else "dewpoint"

    cycle_hours = (0, 6, 12, 18)
    global_stats = {h: {"sum": 0.0, "count": 0} for h in cycle_hours}
    station_stats: dict[str, dict[int, dict[str, float]]] = {}

    # Historical contribution from accumulated by-valid-hour stats.
    # lead_time=6 bias already grouped by valid hour.
    by_vh = (
        db.get("cumulative_stats", {})
        .get("by_lead_time_by_valid_hour", {})
        .get("aifs", {})
        .get(variable, {})
        .get("6", {})
    )
    for vh_str, vh_stats in by_vh.items():
        try:
            valid_hour = int(vh_str)
        except Exception:
            continue
        if valid_hour not in global_stats:
            continue
        count = int(vh_stats.get("count", 0) or 0)
        if count <= 0:
            continue
        global_stats[valid_hour]["sum"] += float(vh_stats.get("sum_errors", 0.0) or 0.0)
        global_stats[valid_hour]["count"] += count

    # Fresh runs not yet folded into cumulative stats.
    runs = db.get("runs", {})
    stations = db.get("stations", {})
    accumulated_run_keys = db.get("cumulative_stats", {}).get("accumulated_run_keys", {})
    now = datetime.now(timezone.utc)

    for run_key, run_data in runs.items():
        if run_key in accumulated_run_keys:
            continue
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        forecast_hours = run_data.get("forecast_hours", [])
        if 6 not in forecast_hours:
            continue
        idx = forecast_hours.index(6)
        valid_time = init_time + timedelta(hours=6)
        if valid_time >= now:
            continue
        valid_hour = valid_time.hour
        if valid_hour not in global_stats:
            continue

        model_data = run_data.get("aifs", {})
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue
            fcst_vals = fcst_data.get(fcst_key, [])
            if idx >= len(fcst_vals) or fcst_vals[idx] is None:
                continue
            obs = get_composite_observation(db, station_id, valid_time)
            if not obs or obs.get(obs_key) is None:
                continue
            error = fcst_vals[idx] - obs[obs_key]
            global_stats[valid_hour]["sum"] += error
            global_stats[valid_hour]["count"] += 1

            st = station_stats.setdefault(
                station_id,
                {h: {"sum": 0.0, "count": 0} for h in cycle_hours}
            )
            st[valid_hour]["sum"] += error
            st[valid_hour]["count"] += 1

    global_bias = {}
    for h in cycle_hours:
        c = global_stats[h]["count"]
        global_bias[h] = (global_stats[h]["sum"] / c) if c > 0 else 0.0

    station_bias: dict[str, dict[int, float]] = {}
    for sid, per_hour in station_stats.items():
        station_bias[sid] = {}
        for h in cycle_hours:
            c = per_hour[h]["count"]
            if c > 0:
                station_bias[sid][h] = per_hour[h]["sum"] / c

    return station_bias, global_bias


def _apply_model_value_adjustment(
    model: str,
    variable: str,
    station_id: str,
    valid_time: datetime,
    fcst_val,
    kenny_station_hour_biases: Optional[dict[str, dict[int, float]]] = None,
    kenny_global_hour_biases: Optional[dict[int, float]] = None
):
    if fcst_val is None:
        return None
    if _is_kenny_bias_corrected_var(model, variable):
        hour = valid_time.hour
        hour_bias = None
        if kenny_station_hour_biases is not None:
            hour_bias = (kenny_station_hour_biases.get(station_id) or {}).get(hour)
        if hour_bias is None and kenny_global_hour_biases is not None:
            hour_bias = kenny_global_hour_biases.get(hour, 0.0)
        if hour_bias is None:
            hour_bias = 0.0
        # Bias is defined as (forecast - observed); correction is forecast - bias.
        return fcst_val - hour_bias
    return fcst_val


def _qualifying_precip_sets(run_data: dict, stations) -> tuple:
    """
    Precompute qualifying precipitation events across all models for a run.

    Returns (qualifying_precip, qualifying_precip_24hr) — sets of (station_id, lt_index)
    where at least one model forecasts >= 0.01" of precipitation.

    Using the union of all model forecasts ensures every model is scored on the same
    set of events. A model that correctly forecasts zero when others over-forecast is
    rewarded with a low MAE rather than being silently excluded from the sample.
    """
    qualifying_precip: set = set()
    qualifying_precip_24hr: set = set()
    for model in ('gfs', 'aifs', 'ifs', 'nws'):
        md = run_data.get(model) or {}
        for sid, fd in md.items():
            if sid not in stations:
                continue
            for i, p in enumerate(fd.get('precips') or []):
                if p is not None and p >= 0.01:
                    qualifying_precip.add((sid, i))
            for i, p in enumerate(fd.get('precips_24hr') or []):
                if p is not None and p >= 0.01:
                    qualifying_precip_24hr.add((sid, i))
    return qualifying_precip, qualifying_precip_24hr

# IEM Rate Limiter - 1 call per second
iem_rate_limiter = RateLimiter(calls_per_second=3)  # IEM university server - moderate rate

@iem_rate_limiter
def _rate_limited_urlopen(*args, **kwargs):
    return urllib.request.urlopen(*args, **kwargs)

# Cache directory for station metadata
CACHE_DIR = Path.home() / ".cache" / "weather_models" / "asos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Station list cache file
STATIONS_CACHE_FILE = CACHE_DIR / "stations.json"
STATIONS_CACHE_TTL_DAYS = 7

# ASOS forecasts storage file (on T7 external drive)
DATA_DIR = Path("/Volumes/T7/Weather_Models/data")
ASOS_FORECASTS_FILE = DATA_DIR / "asos_forecasts.json"

# ASOS verification cache file (precomputed stats)
ASOS_VERIFICATION_CACHE_FILE = DATA_DIR / "asos_verification_cache.json"

# 5-minute METAR pressure archive (dedicated, independent of verification data)
ASOS_METAR_PRESSURE_FILE = DATA_DIR / "asos_metar_pressure.json"
ASOS_MONTHLY_STATS_FILE = DATA_DIR / "asos_monthly_stats.json"
MONTHLY_WINDOW_DAYS = 20

# Retention period for stored forecasts
FORECASTS_RETENTION_DAYS = 21

# ---------------------------------------------------------------------------
# In-memory caches for large JSON files (invalidated when file mtime changes)
# ---------------------------------------------------------------------------
_asos_forecasts_db_cache: dict | None = None
_asos_forecasts_db_mtime: float | None = None

_verification_cache_data: dict | None = None
_verification_cache_mtime: float | None = None

_asos_metar_pressure_cache: dict | None = None
_asos_metar_pressure_mtime: float | None = None

_monthly_stats_cache: dict | None = None
_monthly_stats_mtime: float | None = None

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

    # Fetch states sequentially (rate limiter makes parallelization ineffective)
    for state in US_STATES:
        try:
            stations = fetch_state_stations(state)
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
        variables: List of variables to fetch (default: tmpf, alti, p01i)

    Returns:
        Dict mapping station_id to list of observation dicts
    """
    if variables is None:
        variables = ['tmpf', 'dwpf', 'alti', 'p01i']  # Use 'alti' (altimeter) instead of 'mslp'

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
        col_dwpf = header.index('dwpf') if 'dwpf' in header else -1
        col_alti = header.index('alti') if 'alti' in header else -1
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
                    if val not in ['M', 'T', '']:
                        temp_val = float(val)
                        obs['temp'] = None if _is_implausible_temp_value(temp_val) else temp_val
                    else:
                        obs['temp'] = None

                # Parse dewpoint (°F)
                if col_dwpf >= 0 and col_dwpf < len(fields):
                    val = fields[col_dwpf].strip()
                    if val not in ['M', 'T', '']:
                        dew_val = float(val)
                        obs['dewpoint'] = None if _is_implausible_dewpoint_value(dew_val) else dew_val
                    else:
                        obs['dewpoint'] = None

                # Parse altimeter setting (inches Hg) and convert to mb/hPa
                # Note: Altimeter setting differs slightly from true MSLP (models provide MSLP),
                # but altimeter is much more widely reported by ASOS stations (~95% vs ~20%).
                # The difference is typically 1-3 mb due to temperature corrections, but this
                # systematic bias can be measured and is acceptable for verification purposes.
                if col_alti >= 0 and col_alti < len(fields):
                    val = fields[col_alti].strip()
                    if station in PRESSURE_EXCLUDE_STATIONS:
                        obs['mslp'] = None
                    elif val not in ['M', 'T', '']:
                        # Convert inches of mercury to millibars (1 inHg = 33.8639 mb)
                        mslp_val = round(float(val) * 33.8639, 1)
                        obs['mslp'] = None if _is_implausible_pressure_value(mslp_val) else mslp_val
                    else:
                        obs['mslp'] = None

                # Parse 1-hour precipitation (inches)
                if col_p01i >= 0 and col_p01i < len(fields):
                    val = fields[col_p01i].strip()
                    if val == 'T':
                        obs['precip'] = 0.001  # Trace
                    elif val not in ['M', '']:
                        precip_val = float(val)
                        obs['precip'] = None if _is_implausible_precip_value(station, precip_val) else precip_val
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
    """Load the ASOS forecasts database from JSON file, with in-memory mtime cache."""
    global _asos_forecasts_db_cache, _asos_forecasts_db_mtime
    if not ASOS_FORECASTS_FILE.exists():
        return {
            "stations": {},
            "runs": {},
            "cumulative_stats": {
                "by_station": {},
                "by_lead_time": {},
                "time_series": {},
                "by_station_monthly": {},
                "monthly_generated_at": None,
                "accumulated_run_keys": {}
            }
        }
    try:
        current_mtime = ASOS_FORECASTS_FILE.stat().st_mtime
        if _asos_forecasts_db_cache is not None and _asos_forecasts_db_mtime == current_mtime:
            return _asos_forecasts_db_cache
        with open(ASOS_FORECASTS_FILE) as f:
            data = json.load(f)
        # Ensure cumulative_stats structure exists
        if "cumulative_stats" not in data:
            data["cumulative_stats"] = {
                "by_station": {},
                "by_lead_time": {},
                "time_series": {}
            }
        if "time_series" not in data["cumulative_stats"]:
            data["cumulative_stats"]["time_series"] = {}
        if "by_station_monthly" not in data["cumulative_stats"]:
            data["cumulative_stats"]["by_station_monthly"] = {}
        if "monthly_generated_at" not in data["cumulative_stats"]:
            data["cumulative_stats"]["monthly_generated_at"] = None
        if "accumulated_run_keys" not in data["cumulative_stats"]:
            data["cumulative_stats"]["accumulated_run_keys"] = {}
        _asos_forecasts_db_cache = data
        _asos_forecasts_db_mtime = current_mtime
        return _asos_forecasts_db_cache
    except Exception as e:
        logger.warning(f"Error loading asos_forecasts.json: {e}")
    return {
        "stations": {},
        "runs": {},
        "cumulative_stats": {
            "by_station": {},
            "by_lead_time": {},
            "time_series": {},
            "by_station_monthly": {},
            "monthly_generated_at": None,
            "accumulated_run_keys": {}
        }
    }


def save_asos_forecasts_db(data: dict):
    """Save the ASOS forecasts database to JSON file (atomic write)."""
    tmp = ASOS_FORECASTS_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, ASOS_FORECASTS_FILE)
    logger.info(f"Saved ASOS forecasts to {ASOS_FORECASTS_FILE}")


def load_monthly_stats_cache() -> dict:
    """Load the monthly per-station stats from the small sidecar file."""
    global _monthly_stats_cache, _monthly_stats_mtime
    if not ASOS_MONTHLY_STATS_FILE.exists():
        return {}
    current_mtime = ASOS_MONTHLY_STATS_FILE.stat().st_mtime
    if _monthly_stats_cache is not None and _monthly_stats_mtime == current_mtime:
        return _monthly_stats_cache
    with open(ASOS_MONTHLY_STATS_FILE) as f:
        _monthly_stats_cache = json.load(f)
    _monthly_stats_mtime = current_mtime
    return _monthly_stats_cache


def save_monthly_stats_cache(data: dict) -> None:
    """Save the monthly per-station stats to the small sidecar file (atomic write)."""
    global _monthly_stats_cache, _monthly_stats_mtime
    tmp = ASOS_MONTHLY_STATS_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, ASOS_MONTHLY_STATS_FILE)
    _monthly_stats_mtime = ASOS_MONTHLY_STATS_FILE.stat().st_mtime
    _monthly_stats_cache = data
    logger.info(f"Saved monthly stats to {ASOS_MONTHLY_STATS_FILE}")


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
        data["cumulative_stats"] = {"by_station": {}, "by_lead_time": {}, "by_lead_time_by_valid_hour": {}, "time_series": {}}
    if "by_lead_time_by_valid_hour" not in data["cumulative_stats"]:
        data["cumulative_stats"]["by_lead_time_by_valid_hour"] = {}

    cumulative_by_station = data["cumulative_stats"]["by_station"]
    cumulative_by_lead_time = data["cumulative_stats"]["by_lead_time"]
    cumulative_by_lt_by_vh = data["cumulative_stats"]["by_lead_time_by_valid_hour"]

    # Initialize time series structure if needed
    if "time_series" not in data["cumulative_stats"]:
        data["cumulative_stats"]["time_series"] = {}
    cumulative_time_series = data["cumulative_stats"]["time_series"]

    # Initialize time series for each model if needed
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        if model not in cumulative_time_series:
            cumulative_time_series[model] = {}
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            if var not in cumulative_time_series[model]:
                cumulative_time_series[model][var] = {}

    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }

    cumulative_by_station = data["cumulative_stats"]["by_station"]
    cumulative_by_lead_time = data["cumulative_stats"]["by_lead_time"]

    # Precompute qualifying precip events across all models for fair cross-model comparison
    qualifying_precip, qualifying_precip_24hr = _qualifying_precip_sets(run_data, stations)

    # Process each model
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_data = run_data.get(model)
        if not model_data:
            continue

        # Initialize model in by_lead_time and by_lead_time_by_valid_hour if needed
        if model not in cumulative_by_lead_time:
            cumulative_by_lead_time[model] = {}
        if model not in cumulative_by_lt_by_vh:
            cumulative_by_lt_by_vh[model] = {}

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
                    ('precip', ('precips', 'precip_6hr')),
                    ('precip_24hr', ('precips_24hr', 'precip_24hr')),
                    ('dewpoint', ('dewpoints', 'dewpoint')),
                ]:
                    fcst_values = fcst_data.get(fcst_key, [])
                    if i >= len(fcst_values) or fcst_values[i] is None:
                        continue
                    if obs.get(obs_key) is None:
                        continue

                    fcst_val = fcst_values[i]
                    obs_val = obs[obs_key]
                    if var == 'precip' and (station_id, i) not in qualifying_precip and obs_val < 0.01:
                        continue
                    if var == 'precip_24hr' and (station_id, i) not in qualifying_precip_24hr and obs_val < 0.01:
                        continue
                    error = fcst_val - obs_val
                    abs_error = abs(error)

                    # Update by_station stats
                    if var not in cumulative_by_station[station_id][model]:
                        cumulative_by_station[station_id][model][var] = {}
                    if lt_str not in cumulative_by_station[station_id][model][var]:
                        cumulative_by_station[station_id][model][var][lt_str] = {
                            "sum_abs_errors": 0.0,
                            "sum_errors": 0.0,
                            "count": 0,
                            "sum_weighted_abs_errors": 0.0,
                            "sum_weights": 0.0,
                        }

                    cumulative_by_station[station_id][model][var][lt_str]["sum_abs_errors"] += abs_error
                    cumulative_by_station[station_id][model][var][lt_str]["sum_errors"] += error
                    cumulative_by_station[station_id][model][var][lt_str]["count"] += 1
                    if _is_precip_var(var):
                        weight = _precip_weight(obs_val)
                        cumulative_by_station[station_id][model][var][lt_str]["sum_weighted_abs_errors"] += abs_error * weight
                        cumulative_by_station[station_id][model][var][lt_str]["sum_weights"] += weight

                    # Update by_lead_time stats
                    if var not in cumulative_by_lead_time[model]:
                        cumulative_by_lead_time[model][var] = {}
                    if lt_str not in cumulative_by_lead_time[model][var]:
                        cumulative_by_lead_time[model][var][lt_str] = {
                            "sum_abs_errors": 0.0,
                            "sum_errors": 0.0,
                            "count": 0,
                            "sum_weighted_abs_errors": 0.0,
                            "sum_weights": 0.0,
                        }

                    cumulative_by_lead_time[model][var][lt_str]["sum_abs_errors"] += abs_error
                    cumulative_by_lead_time[model][var][lt_str]["sum_errors"] += error
                    cumulative_by_lead_time[model][var][lt_str]["count"] += 1
                    if _is_precip_var(var):
                        weight = _precip_weight(obs_val)
                        cumulative_by_lead_time[model][var][lt_str]["sum_weighted_abs_errors"] += abs_error * weight
                        cumulative_by_lead_time[model][var][lt_str]["sum_weights"] += weight

                    # Update by_lead_time_by_valid_hour stats (store at exact lt; snapping happens at read time)
                    vh_str = str(valid_time.hour)
                    if var not in cumulative_by_lt_by_vh[model]:
                        cumulative_by_lt_by_vh[model][var] = {}
                    if lt_str not in cumulative_by_lt_by_vh[model][var]:
                        cumulative_by_lt_by_vh[model][var][lt_str] = {}
                    if vh_str not in cumulative_by_lt_by_vh[model][var][lt_str]:
                        cumulative_by_lt_by_vh[model][var][lt_str][vh_str] = {
                            "sum_abs_errors": 0.0, "sum_errors": 0.0, "count": 0,
                            "sum_weighted_abs_errors": 0.0, "sum_weights": 0.0
                        }
                    cumulative_by_lt_by_vh[model][var][lt_str][vh_str]["sum_abs_errors"] += abs_error
                    cumulative_by_lt_by_vh[model][var][lt_str][vh_str]["sum_errors"] += error
                    cumulative_by_lt_by_vh[model][var][lt_str][vh_str]["count"] += 1
                    if _is_precip_var(var):
                        weight = _precip_weight(obs_val)
                        cumulative_by_lt_by_vh[model][var][lt_str][vh_str]["sum_weighted_abs_errors"] += abs_error * weight
                        cumulative_by_lt_by_vh[model][var][lt_str][vh_str]["sum_weights"] += weight
                    # Note: lt_str stored here may be non-canonical; _snap_to_canonical_lt() in
                    # precompute_verification_cache() remaps it to the correct canonical bucket.

    # Accumulate time series data (daily errors by lead time)
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
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
                    if var == 'precip' and (station_id, i) not in qualifying_precip and obs_val < 0.01:
                        continue
                    if var == 'precip_24hr' and (station_id, i) not in qualifying_precip_24hr and obs_val < 0.01:
                        continue
                    error = fcst_val - obs_val

                    # Initialize structures if needed
                    if lt_str not in cumulative_time_series[model][var]:
                        cumulative_time_series[model][var][lt_str] = {}
                    if date_key not in cumulative_time_series[model][var][lt_str]:
                        cumulative_time_series[model][var][lt_str][date_key] = {
                            "sum_abs_errors": 0.0,
                            "sum_errors": 0.0,
                            "count": 0,
                            "sum_weighted_abs_errors": 0.0,
                            "sum_weights": 0.0,
                        }

                    stats = cumulative_time_series[model][var][lt_str][date_key]
                    stats["sum_abs_errors"] += abs(error)
                    stats["sum_errors"] += error
                    stats["count"] += 1
                    if _is_precip_var(var):
                        weight = _precip_weight(obs_val)
                        stats["sum_weighted_abs_errors"] += abs(error) * weight
                        stats["sum_weights"] += weight


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
    # Skip runs already eagerly accumulated by precompute_verification_cache()
    accumulated_run_keys = data.get("cumulative_stats", {}).get("accumulated_run_keys", {})
    accumulated_count = 0
    for run_id in old_runs:
        if run_id in accumulated_run_keys:
            # Already accumulated — just clean up the tracking key
            del accumulated_run_keys[run_id]
        else:
            run_data = data["runs"][run_id]
            accumulate_stats_from_run(data, run_id, run_data)
            accumulated_count += 1

    # Now delete the old runs
    for run_id in old_runs:
        del data["runs"][run_id]

    if old_runs:
        logger.info(f"Accumulated stats from {accumulated_count} old runs before cleanup ({len(old_runs) - accumulated_count} already eagerly accumulated)")

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

    # Use a high-water mark to avoid re-fetching the entire 20-day window every sync.
    #
    # The old per-timestamp skip check compared forecast valid times (exact hour boundaries
    # like 12:00:00Z) against stored observation keys (actual ASOS report times like 11:53:00Z).
    # These almost never matched, so every sync refetched the full ~20-day window.
    last_fetch_end_str = db.get("last_obs_fetch_end")
    if last_fetch_end_str:
        last_fetch_end = datetime.fromisoformat(last_fetch_end_str)
        if last_fetch_end.tzinfo is None:
            last_fetch_end = last_fetch_end.replace(tzinfo=timezone.utc)
        # Only fetch times after our last fetch (with 1h overlap for safety)
        min_time = last_fetch_end - timedelta(hours=1)
    else:
        # Initial backfill: start from oldest needed valid time
        min_time = min(valid_times_needed) - timedelta(hours=1)

    max_time = now + timedelta(hours=1)

    new_times = [vt for vt in valid_times_needed if vt >= min_time]
    if not new_times:
        logger.info("Already have observations for all valid times")
        return 0

    logger.info(f"Fetching ASOS observations from {min_time} to {max_time} for {len(new_times)} valid times")

    # Fetch observations in chunks of stations sequentially (rate limiter makes parallelization ineffective)
    station_ids = list(stations.keys())
    chunk_size = 50
    all_observations = {}

    for i in range(0, len(station_ids), chunk_size):
        chunk_ids = station_ids[i:i+chunk_size]
        logger.info(f"Fetching observations for stations {i+1}-{min(i+chunk_size, len(station_ids))} of {len(station_ids)}...")
        try:
            obs_chunk = fetch_observations(
                chunk_ids,
                min_time,
                max_time,
                variables=['tmpf', 'dwpf', 'alti', 'p01i']  # Use 'alti' (altimeter) instead of 'mslp'
            )
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
                'precip': obs.get('precip'),
                'dewpoint': obs.get('dewpoint'),
            }

            # Only store if at least one value is not None
            if any(v is not None for v in obs_data.values()):
                db["observations"][station_id][obs_time_str] = obs_data
                obs_count += 1

    # Update high-water mark so subsequent syncs only fetch new data
    db["last_obs_fetch_end"] = max_time.isoformat()

    # Cleanup old data and save
    db = cleanup_old_runs(db)
    save_asos_forecasts_db(db)

    logger.info(f"Stored {obs_count} observations for {len(all_observations)} stations")
    return obs_count


def calculate_6hr_precip_total(db: dict, station_id: str, end_time: datetime) -> Optional[float]:
    """
    Calculate 6-hour accumulated precipitation ending at the specified time.

    ASOS p01i is a running accumulation that resets after each hourly METAR (:53).
    Within a clock hour the values grow from ~0 up to the full hour total, so
    the maximum p01i in each clock hour equals that hour's true precipitation total.

    Correct approach: take the maximum p01i per clock hour, then sum over 6 hours.
    This avoids the double-counting error that occurs when summing all sub-hourly
    observations (each of which is a cumulative sub-total, not an independent increment).

    Args:
        db: The loaded database
        station_id: Station ID
        end_time: End of the 6-hour accumulation period

    Returns:
        6-hour precipitation total in inches, or None if insufficient data
    """
    if station_id in PRECIP_EXCLUDE_STATIONS:
        return None
    station_obs = db.get("observations", {}).get(station_id, {})

    if not station_obs:
        return None

    # Define the 6-hour window
    start_time = end_time - timedelta(hours=6)

    # Collect all observations in the window, sorted by time.
    window_obs = []
    for obs_time_str, obs_data in station_obs.items():
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            if start_time < obs_time <= end_time:
                precip = obs_data.get('precip')
                if precip is not None and not _is_implausible_precip_value(station_id, precip):
                    window_obs.append((obs_time, precip))
        except (ValueError, AttributeError):
            continue

    if not window_obs:
        return None

    window_obs.sort(key=lambda x: x[0])

    # Pre-filter tipping-bucket overflow artifacts (2.56", 5.12", 7.68", …).
    # These appear as isolated spikes before or during a stuck-gauge run and
    # escape the consecutive-duplicate detector below.
    window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]

    if not window_obs:
        return None

    # Detect stuck-gauge readings: a non-zero value repeated 2+ consecutive times
    # is a sensor malfunction.  Real ASOS running totals always increase between
    # reports; a flat non-zero reading means the gauge is stuck.
    # Build a set of timestamps to exclude.
    stuck_times: set = set()
    if len(window_obs) >= 2:
        for i in range(len(window_obs) - 1):
            v0, v1 = window_obs[i][1], window_obs[i+1][1]
            if v0 > 0 and v0 == v1:
                # Mark this run (and any continuations) as stuck
                j = i
                while j < len(window_obs) and window_obs[j][1] == v0:
                    stuck_times.add(window_obs[j][0])
                    j += 1

    # Group non-stuck observations by clock hour and take the maximum p01i per hour.
    # The maximum equals the full-hour METAR total because sub-hourly automated
    # reports are running sub-totals that peak at the :53 METAR observation.
    hourly_max: dict = {}  # hour_key (datetime truncated to hour) -> max p01i

    for obs_time, precip in window_obs:
        if obs_time in stuck_times:
            continue
        hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
        if hour_key not in hourly_max or precip > hourly_max[hour_key]:
            hourly_max[hour_key] = precip

    if not hourly_max:
        return None

    total = sum(hourly_max.values())
    if station_id in PRECIP_MM_STATIONS:
        total /= 25.4  # stored in mm, return inches
    return total


def calculate_24hr_precip_total(db: dict, station_id: str, end_time: datetime) -> Optional[float]:
    """
    Calculate 24-hour accumulated precipitation ending at the specified time (12Z only).

    Uses the same max-per-hour logic as calculate_6hr_precip_total to correctly handle
    ASOS running accumulations. Returns None if end_time.hour != 12 or data is insufficient.

    Args:
        db: The loaded database
        station_id: Station ID
        end_time: End of the 24-hour accumulation period (must be 12Z)

    Returns:
        24-hour precipitation total in inches, or None if insufficient data
    """
    if end_time.hour != 12:
        return None
    if station_id in PRECIP_EXCLUDE_STATIONS:
        return None
    station_obs = db.get("observations", {}).get(station_id, {})

    if not station_obs:
        return None

    start_time = end_time - timedelta(hours=24)

    window_obs = []
    for obs_time_str, obs_data in station_obs.items():
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            if start_time < obs_time <= end_time:
                precip = obs_data.get('precip')
                if precip is not None and not _is_implausible_precip_value(station_id, precip):
                    window_obs.append((obs_time, precip))
        except (ValueError, AttributeError):
            continue

    if not window_obs:
        return None

    window_obs.sort(key=lambda x: x[0])
    window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]
    if not window_obs:
        return None

    stuck_times: set = set()
    if len(window_obs) >= 2:
        for i in range(len(window_obs) - 1):
            v0, v1 = window_obs[i][1], window_obs[i+1][1]
            if v0 > 0 and v0 == v1:
                j = i
                while j < len(window_obs) and window_obs[j][1] == v0:
                    stuck_times.add(window_obs[j][0])
                    j += 1

    hourly_max: dict = {}
    for obs_time, precip in window_obs:
        if obs_time in stuck_times:
            continue
        hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
        if hour_key not in hourly_max or precip > hourly_max[hour_key]:
            hourly_max[hour_key] = precip

    if len(hourly_max) < 20:
        return None

    total = sum(hourly_max.values())
    if station_id in PRECIP_MM_STATIONS:
        total /= 25.4
    return total


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
            best_match['precip_6hr'] = None

        # Add 24-hour accumulated precipitation (12Z only)
        if target_time.hour == 12:
            best_match['precip_24hr'] = calculate_24hr_precip_total(db, station_id, target_time)
        else:
            best_match['precip_24hr'] = None

        return best_match
    return None


def get_composite_observation(
    db: dict,
    station_id: str,
    target_time: datetime,
    max_delta_minutes: int = 30
) -> Optional[dict]:
    """
    Composite observation lookup: nearest value per variable within tolerance.

    Unlike get_stored_observation(), this does not require one observation timestamp
    to contain all variables; temp/dewpoint/mslp/precip are selected independently.
    """
    observations_data = db.get("observations", {})
    station_obs = observations_data.get(station_id, {})
    if not station_obs:
        return None

    tol = timedelta(minutes=max_delta_minutes)
    best_temp = (None, tol + timedelta(seconds=1))
    best_mslp = (None, tol + timedelta(seconds=1))
    best_precip = (None, tol + timedelta(seconds=1))
    best_dew = (None, tol + timedelta(seconds=1))

    for obs_time_str, obs_data in station_obs.items():
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        delta = abs(obs_time - target_time)
        if delta > tol:
            continue

        t = obs_data.get("temp")
        if t is not None and delta < best_temp[1]:
            best_temp = (t, delta)

        p = obs_data.get("mslp")
        if p is not None and delta < best_mslp[1]:
            best_mslp = (p, delta)

        pr = obs_data.get("precip")
        if pr is not None and delta < best_precip[1]:
            best_precip = (pr, delta)

        d = obs_data.get("dewpoint")
        if d is not None and delta < best_dew[1]:
            best_dew = (d, delta)

    if all(v[0] is None for v in (best_temp, best_mslp, best_precip, best_dew)):
        return None

    composite = {
        "temp": best_temp[0],
        "mslp": best_mslp[0],
        "precip": best_precip[0],
        "dewpoint": best_dew[0],
    }

    if target_time.hour % 6 == 0:
        composite["precip_6hr"] = calculate_6hr_precip_total(db, station_id, target_time)
    else:
        composite["precip_6hr"] = None

    if target_time.hour == 12:
        composite["precip_24hr"] = calculate_24hr_precip_total(db, station_id, target_time)
    else:
        composite["precip_24hr"] = None

    return composite


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
        model_name: Model name ('gfs', 'aifs', 'ifs', 'nws')
        station_forecasts: Dict mapping station_id to forecast data
            Each station dict has: temps, mslps, precips (lists aligned with forecast_hours)
    """
    store_asos_forecasts_batch([(init_time, forecast_hours, model_name, station_forecasts)])


def store_asos_forecasts_batch(
    entries: list,
):
    """
    Store multiple models' forecasts in a single DB load / cleanup / save cycle.

    Each entry is a tuple: (init_time, forecast_hours, model_name, station_forecasts).
    Compared to calling store_asos_forecasts() once per model this eliminates the
    redundant load→cleanup→save cycles that otherwise happen for every model.
    """
    if not entries:
        return

    db = load_asos_forecasts_db()

    # Ensure stations are up to date
    stations = get_stations_dict()
    db["stations"] = stations

    for init_time, forecast_hours, model_name, station_forecasts in entries:
        run_key = init_time.isoformat()
        if run_key not in db.get("runs", {}):
            db.setdefault("runs", {})[run_key] = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "forecast_hours": forecast_hours,
            }
        db["runs"][run_key][model_name.lower()] = station_forecasts
        logger.info(f"Staged {model_name} forecasts for {len(station_forecasts)} stations at {run_key}")

    db = cleanup_old_runs(db)
    save_asos_forecasts_db(db)
    logger.info(f"Saved {len(entries)} model forecast(s) in single DB write")


def get_verification_data(
    model: str,
    variable: str,
    lead_time_hours: int,
    valid_hour: Optional[int] = None
) -> Dict[str, dict]:
    """
    Get verification data for all stations at a specific lead time.

    Combines cumulative historical statistics with fresh calculations from
    current runs to provide lifetime MAE and bias per station.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
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
    source_model = _source_model_for_verification(model)
    db = load_asos_forecasts_db()
    if _is_kenny_bias_corrected_var(model, variable):
        kenny_station_hour_biases, kenny_global_hour_biases = _compute_kenny_station_hour_biases(db, variable)
    else:
        kenny_station_hour_biases, kenny_global_hour_biases = None, None
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
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }

    if variable not in var_map:
        return {}

    fcst_key, obs_key = var_map[variable]
    lt_str = str(lead_time_hours)

    # Start with cumulative stats for this lead time
    station_stats = {}  # station_id -> {sum_abs_errors, sum_errors, count}

    # Load cumulative stats
    if not _is_kenny_bias_corrected_var(model, variable):
        for station_id in stations:
            if station_id in cumulative_by_station:
                model_stats = cumulative_by_station[station_id].get(source_model, {})
                var_stats = model_stats.get(variable, {})
                lt_stats = var_stats.get(lt_str)
                if lt_stats:
                    station_stats[station_id] = {
                        'sum_abs_errors': lt_stats.get('sum_abs_errors', 0.0),
                        'sum_errors': lt_stats.get('sum_errors', 0.0),
                        'count': lt_stats.get('count', 0),
                        'sum_weighted_abs_errors': lt_stats.get('sum_weighted_abs_errors', 0.0),
                        'sum_weights': lt_stats.get('sum_weights', 0.0),
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
        model_data = run_data.get(source_model)

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
        if valid_hour is not None and valid_time.hour != valid_hour:
            continue

        # Match forecasts with stored observations
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            # Get forecast value
            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue
            fcst_val = _apply_model_value_adjustment(
                model, variable, station_id, valid_time, fcst_values[fcst_idx],
                kenny_station_hour_biases, kenny_global_hour_biases
            )

            # Get stored observation
            obs = (
                get_composite_observation(db, station_id, valid_time)
                if _is_kenny_bias_corrected_var(model, variable)
                else get_stored_observation(db, station_id, valid_time)
            )

            if obs is None or obs.get(obs_key) is None:
                continue
            obs_val = obs[obs_key]

            # Calculate error and add to running totals
            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue
            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue
            error = fcst_val - obs_val
            if station_id not in station_stats:
                station_stats[station_id] = {
                    'sum_abs_errors': 0.0,
                    'sum_errors': 0.0,
                    'count': 0,
                    'sum_weighted_abs_errors': 0.0,
                    'sum_weights': 0.0,
                }
            station_stats[station_id]['sum_abs_errors'] += abs(error)
            station_stats[station_id]['sum_errors'] += error
            station_stats[station_id]['count'] += 1
            if _is_precip_var(variable):
                weight = _precip_weight(obs_val)
                station_stats[station_id]['sum_weighted_abs_errors'] += abs(error) * weight
                station_stats[station_id]['sum_weights'] += weight

    # Calculate final metrics per station
    results = {}

    for station_id, stats in station_stats.items():
        if stats['count'] == 0:
            continue

        station = stations.get(station_id, {})

        mae = stats['sum_abs_errors'] / stats['count']
        bias = stats['sum_errors'] / stats['count']
        wmae = _stats_wmae(stats, fallback_mae=mae) if _is_precip_var(variable) else None

        results[station_id] = {
            'mae': round(mae, 2),
            'bias': _bias_corrected_metric_value(model, f"{variable}_bias", round(mae, 2), round(bias, 2)),
            'wmae': round(wmae, 2) if wmae is not None else None,
            'count': stats['count'],
            'lat': station.get('lat'),
            'lon': station.get('lon'),
            'name': station.get('name', station_id),
            'state': station.get('state', '')
        }

    return results


def get_verification_data_recent(
    model: str,
    variable: str,
    lead_time_hours: int,
    days_back: int = 30,
    valid_hour: Optional[int] = None
) -> Dict[str, dict]:
    """
    Get verification data for all stations within a recent window.
    Uses run data + stored observations only (no cumulative stats).
    """
    source_model = _source_model_for_verification(model)
    db = load_asos_forecasts_db()
    if _is_kenny_bias_corrected_var(model, variable):
        kenny_station_hour_biases, kenny_global_hour_biases = _compute_kenny_station_hour_biases(db, variable)
    else:
        kenny_station_hour_biases, kenny_global_hour_biases = None, None
    stations = db.get("stations", {})
    runs = db.get("runs", {})

    if not stations:
        return {}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_back)

    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }
    if variable not in var_map:
        return {}

    fcst_key, obs_key = var_map[variable]

    station_stats = {}

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])
        model_data = run_data.get(source_model)
        if not model_data:
            continue

        if lead_time_hours not in forecast_hours:
            continue
        fcst_idx = forecast_hours.index(lead_time_hours)

        valid_time = init_time + timedelta(hours=lead_time_hours)
        if valid_time >= now or valid_time < cutoff:
            continue
        if valid_hour is not None and valid_time.hour != valid_hour:
            continue

        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue
            fcst_val = _apply_model_value_adjustment(
                model, variable, station_id, valid_time, fcst_values[fcst_idx],
                kenny_station_hour_biases, kenny_global_hour_biases
            )

            obs = (
                get_composite_observation(db, station_id, valid_time)
                if _is_kenny_bias_corrected_var(model, variable)
                else get_stored_observation(db, station_id, valid_time)
            )
            if obs is None or obs.get(obs_key) is None:
                continue
            obs_val = obs[obs_key]

            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue
            error = fcst_val - obs_val
            if station_id not in station_stats:
                station_stats[station_id] = {
                    'sum_abs_errors': 0.0,
                    'sum_errors': 0.0,
                    'count': 0,
                    'sum_weighted_abs_errors': 0.0,
                    'sum_weights': 0.0,
                }
            station_stats[station_id]['sum_abs_errors'] += abs(error)
            station_stats[station_id]['sum_errors'] += error
            station_stats[station_id]['count'] += 1
            if _is_precip_var(variable):
                weight = _precip_weight(obs_val)
                station_stats[station_id]['sum_weighted_abs_errors'] += abs(error) * weight
                station_stats[station_id]['sum_weights'] += weight

    results = {}
    for station_id, stats in station_stats.items():
        if stats['count'] == 0:
            continue
        station = stations.get(station_id, {})
        mae = stats['sum_abs_errors'] / stats['count']
        bias = stats['sum_errors'] / stats['count']
        wmae = _stats_wmae(stats, fallback_mae=mae) if _is_precip_var(variable) else None
        results[station_id] = {
            'mae': round(mae, 2),
            'bias': _bias_corrected_metric_value(model, f"{variable}_bias", round(mae, 2), round(bias, 2)),
            'wmae': round(wmae, 2) if wmae is not None else None,
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
    models = [model.lower()] if model else ['gfs', 'aifs', 'ifs', 'nws']
    if model and model.lower() == "kenny":
        kenny_temp_station_hour_biases, kenny_temp_global_hour_biases = _compute_kenny_station_hour_biases(db, "temp")
        kenny_dew_station_hour_biases, kenny_dew_global_hour_biases = _compute_kenny_station_hour_biases(db, "dewpoint")
    else:
        kenny_temp_station_hour_biases = kenny_temp_global_hour_biases = None
        kenny_dew_station_hour_biases = kenny_dew_global_hour_biases = None

    # Collect all forecast hours (from current runs and cumulative stats)
    all_forecast_hours = set()
    for run_data in runs.values():
        all_forecast_hours.update(run_data.get("forecast_hours", []))

    # Also include lead times from cumulative stats for this station
    station_cumulative = cumulative_by_station.get(station_id, {})
    for m in models:
        model_stats = station_cumulative.get(m, {})
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
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
                'precip': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
                'dewpoint': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
            }
            for m in models
        }
        for lt in lead_times
    }

    # Start with cumulative stats
    for m in models:
        source_model = _source_model_for_verification(m)
        model_cumulative = station_cumulative.get(source_model, {})
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
            if _is_kenny_bias_corrected_var(m, var):
                continue
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
        qualifying_precip, _ = _qualifying_precip_sets(run_data, stations)

        for m in models:
            source_model = _source_model_for_verification(m)
            model_data = run_data.get(source_model, {})
            fcst_data = model_data.get(station_id)

            if not fcst_data:
                continue

            for i, lt in enumerate(forecast_hours):
                valid_time = init_time + timedelta(hours=lt)
                if valid_time >= now:
                    continue

                # Get stored observation
                obs = get_composite_observation(db, station_id, valid_time)
                if not obs:
                    continue

                # Temperature
                fcst_temps = fcst_data.get('temps', [])
                if i < len(fcst_temps) and fcst_temps[i] is not None and obs.get('temp') is not None:
                    fcst_temp = _apply_model_value_adjustment(
                        m, "temp", station_id, valid_time, fcst_temps[i],
                        kenny_temp_station_hour_biases, kenny_temp_global_hour_biases
                    )
                    error = fcst_temp - obs['temp']
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

                # Precip (6-hour accumulated) — union-based qualifying (same as cache)
                fcst_precips = fcst_data.get('precips', [])
                if i < len(fcst_precips):
                    fcst_val = fcst_precips[i]
                    obs_val = obs.get('precip_6hr')
                    if fcst_val is not None and obs_val is not None and (
                        (station_id, i) in qualifying_precip or obs_val >= 0.01
                    ):
                        error = fcst_val - obs_val
                        stats_by_lt[lt][m]['precip']['sum_abs_errors'] += abs(error)
                        stats_by_lt[lt][m]['precip']['sum_errors'] += error
                        stats_by_lt[lt][m]['precip']['count'] += 1

                # Dewpoint
                fcst_dewpoints = fcst_data.get('dewpoints', [])
                if i < len(fcst_dewpoints) and fcst_dewpoints[i] is not None and obs.get('dewpoint') is not None:
                    fcst_dew = _apply_model_value_adjustment(
                        m, "dewpoint", station_id, valid_time, fcst_dewpoints[i],
                        kenny_dew_station_hour_biases, kenny_dew_global_hour_biases
                    )
                    error = fcst_dew - obs['dewpoint']
                    stats_by_lt[lt][m]['dewpoint']['sum_abs_errors'] += abs(error)
                    stats_by_lt[lt][m]['dewpoint']['sum_errors'] += error
                    stats_by_lt[lt][m]['dewpoint']['count'] += 1

    # Calculate final metrics
    result_data = {}

    for lt in lead_times:
        result_data[lt] = {}
        for m in models:
            result_data[lt][m] = {}
            for var in ['temp', 'mslp', 'precip', 'dewpoint']:
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


def get_station_detail_from_cache(station_id: str, model: str) -> dict:
    """
    Get detailed verification for a single station from the precomputed verification cache.

    The verification cache is built using composite observation matching (get_cached_observation),
    which correctly finds temperature data at :56 even when MSLP-only reports exist at :00.
    This avoids the issue where get_station_detail() uses non-composite get_stored_observation()
    and misses temperature data for some stations.

    Returns a flat structure with lead_times, temp_mae, temp_bias, mslp_mae, mslp_bias lists.
    """
    if (model or "").lower() == "kenny":
        # Kenny temperature applies cycle-dependent forecast adjustment; compute directly.
        detailed = get_station_detail(station_id, model)
        if "data" not in detailed:
            return {"error": "No verification data for this station"}
        lead_times = detailed.get("lead_times", [])
        temp_mae, temp_bias, temp_count = [], [], []
        precip_mae, precip_bias, precip_count = [], [], []
        dew_mae, dew_bias, dew_count = [], [], []
        for lt in lead_times:
            mdata = detailed.get("data", {}).get(lt, {}).get("kenny", {})
            t = mdata.get("temp")
            p = mdata.get("precip")
            d = mdata.get("dewpoint")
            temp_mae.append(t.get("mae") if t else None)
            temp_bias.append(t.get("bias") if t else None)
            temp_count.append(t.get("count", 0) if t else 0)
            precip_mae.append(p.get("mae") if p else None)
            precip_bias.append(p.get("bias") if p else None)
            precip_count.append(p.get("count", 0) if p else 0)
            dew_mae.append(d.get("mae") if d else None)
            dew_bias.append(d.get("bias") if d else None)
            dew_count.append(d.get("count", 0) if d else 0)
        return {
            "station": detailed.get("station", {}),
            "lead_times": lead_times,
            "temp_mae": temp_mae,
            "temp_bias": temp_bias,
            "temp_count": temp_count,
            "precip_mae": precip_mae,
            "precip_bias": precip_bias,
            "precip_count": precip_count,
            "dewpoint_mae": dew_mae,
            "dewpoint_bias": dew_bias,
            "dewpoint_count": dew_count,
        }

    cache = load_verification_cache()
    if cache is None:
        return {"error": "Verification cache not available. Run a sync to rebuild it."}

    stations = cache.get("stations", {})
    station = stations.get(station_id)
    if not station:
        # Try the main DB as fallback for station metadata
        db = load_asos_forecasts_db()
        station = db.get("stations", {}).get(station_id)
    if not station:
        return {"error": "Station not found"}

    source_model = _source_model_for_verification(model)
    station_data = cache.get("by_station", {}).get(station_id, {})
    model_data = station_data.get(source_model, {})

    if not model_data:
        return {"error": "No verification data for this station"}

    lead_times = cache.get("lead_times", [])  # List of integers

    temp_mae = []
    temp_bias = []
    precip_mae = []
    precip_bias = []
    dewpoint_mae = []
    dewpoint_bias = []

    temp_count = []
    precip_count = []
    dewpoint_count = []

    for lt in lead_times:
        lt_str = str(lt)
        lt_data = model_data.get(lt_str, {})

        temp = lt_data.get('temp')
        temp_mae.append(temp['mae'] if temp else None)
        temp_bias.append(_bias_corrected_metric_value(model, "temp_bias", temp['mae'] if temp else None, temp['bias'] if temp else None) if temp else None)
        temp_count.append(temp['count'] if temp else 0)

        precip = lt_data.get('precip')
        precip_mae.append(precip['mae'] if precip else None)
        precip_bias.append(_bias_corrected_metric_value(model, "precip_bias", precip['mae'] if precip else None, precip['bias'] if precip else None) if precip else None)
        precip_count.append(precip['count'] if precip else 0)

        dewpoint = lt_data.get('dewpoint')
        dewpoint_mae.append(dewpoint['mae'] if dewpoint else None)
        dewpoint_bias.append(_bias_corrected_metric_value(model, "dewpoint_bias", dewpoint['mae'] if dewpoint else None, dewpoint['bias'] if dewpoint else None) if dewpoint else None)
        dewpoint_count.append(dewpoint['count'] if dewpoint else 0)

    return {
        "station": station,
        "lead_times": lead_times,
        "temp_mae": temp_mae,
        "temp_bias": temp_bias,
        "temp_count": temp_count,
        "precip_mae": precip_mae,
        "precip_bias": precip_bias,
        "precip_count": precip_count,
        "dewpoint_mae": dewpoint_mae,
        "dewpoint_bias": dewpoint_bias,
        "dewpoint_count": dewpoint_count,
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
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_data = cumulative_by_lead_time.get(model, {})
        total_count = 0
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
            var_data = model_data.get(var, {})
            for lt_stats in var_data.values():
                total_count += lt_stats.get('count', 0)
        model_totals[model] = total_count

    # Get lead time coverage
    lead_times_per_model = {}
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_data = cumulative_by_lead_time.get(model, {})
        lead_times = set()
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
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
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
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
    source_model = _source_model_for_verification(model)
    db = load_asos_forecasts_db()
    if _is_kenny_bias_corrected_var(model, variable):
        kenny_station_hour_biases, kenny_global_hour_biases = _compute_kenny_station_hour_biases(db, variable)
    else:
        kenny_station_hour_biases, kenny_global_hour_biases = None, None
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
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
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
            errors_by_date[date_key] = {
                "sum_abs_errors": 0.0,
                "sum_errors": 0.0,
                "count": 0,
                "sum_weighted_abs_errors": 0.0,
                "sum_weights": 0.0,
            }

        # Get model data
        model_data = run_data.get(source_model)
        if not model_data:
            continue

        # Process each station
        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue

            fcst_val = _apply_model_value_adjustment(
                model, variable, station_id, valid_time, fcst_values[fcst_idx],
                kenny_station_hour_biases, kenny_global_hour_biases
            )

            # Get observation
            obs = get_stored_observation(db, station_id, valid_time)
            if obs is None or obs.get(obs_key) is None:
                continue

            obs_val = obs[obs_key]
            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue
            if variable == 'precip_24hr' and not should_include_precip(fcst_val, obs_val):
                continue
            error = fcst_val - obs_val
            stats = errors_by_date[date_key]
            stats["sum_abs_errors"] += abs(error)
            stats["sum_errors"] += error
            stats["count"] += 1
            if _is_precip_var(variable):
                weight = _precip_weight(obs_val)
                stats["sum_weighted_abs_errors"] += abs(error) * weight
                stats["sum_weights"] += weight

    # Calculate daily MAE and Bias
    dates = sorted(errors_by_date.keys())
    daily_mae = []
    daily_wmae = []
    daily_bias = []
    daily_counts = []

    for date in dates:
        stats = errors_by_date[date]
        count = stats.get("count", 0)
        if count > 0:
            mae = stats["sum_abs_errors"] / count
            bias = stats["sum_errors"] / count
            wmae = _stats_wmae(stats, fallback_mae=mae) if _is_precip_var(variable) else None
            daily_mae.append(mae)
            daily_wmae.append(wmae)
            daily_bias.append(bias)
            daily_counts.append(count)
        else:
            daily_mae.append(None)
            daily_wmae.append(None)
            daily_bias.append(None)
            daily_counts.append(0)

    result = {
        "dates": dates,
        "mae": [round(m, 2) if m is not None else None for m in daily_mae],
        "wmae": [round(m, 2) if m is not None else None for m in daily_wmae],
        "bias": [round(b, 2) if b is not None else None for b in daily_bias],
        "counts": daily_counts
    }
    return result


def get_mean_verification_by_lead_time(
    model: str,
    valid_hour: Optional[int] = None,
    days_back: Optional[int] = None
) -> dict:
    """
    Get mean verification (MAE and Bias) across all stations by lead time for a given model.

    Combines cumulative historical statistics with fresh calculations from
    current runs to provide lifetime mean MAE and bias.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')

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
        source_model = _source_model_for_verification(model)
        db = load_asos_forecasts_db()
        if _is_kenny_bias_corrected_var(model, "temp"):
            kenny_temp_station_hour_biases, kenny_temp_global_hour_biases = _compute_kenny_station_hour_biases(db, "temp")
            kenny_dew_station_hour_biases, kenny_dew_global_hour_biases = _compute_kenny_station_hour_biases(db, "dewpoint")
        else:
            kenny_temp_station_hour_biases = kenny_temp_global_hour_biases = None
            kenny_dew_station_hour_biases = kenny_dew_global_hour_biases = None
        stations = db.get("stations", {})
        runs = db.get("runs", {})
        cumulative_by_lead_time = db.get("cumulative_stats", {}).get("by_lead_time", {})

        if not stations:
            raise ValueError("No ASOS stations available.")

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days_back)) if days_back is not None else None

        # Collect all forecast hours (from current runs and cumulative stats)
        all_forecast_hours = set()
        for run_data in runs.values():
            all_forecast_hours.update(run_data.get("forecast_hours", []))

        # Also include lead times from cumulative stats
        model_cumulative = cumulative_by_lead_time.get(source_model, {})
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
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
                'precip': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0, 'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0},
                'precip_24hr': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0, 'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0},
                'dewpoint': {'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0},
            }
            for lt in lead_times
        }

        # Start with cumulative stats
        include_cumulative = (valid_hour is None and cutoff is None)
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            if _is_kenny_bias_corrected_var(model, var) or not include_cumulative:
                continue
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, stats in var_cumulative.items():
                lt = int(lt_str)
                if lt in aggregated_stats:
                    aggregated_stats[lt][var]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    aggregated_stats[lt][var]['sum_errors'] += stats.get('sum_errors', 0.0)
                    aggregated_stats[lt][var]['count'] += stats.get('count', 0)
                    if _is_precip_var(var):
                        aggregated_stats[lt][var]['sum_weighted_abs_errors'] += stats.get('sum_weighted_abs_errors', 0.0)
                        aggregated_stats[lt][var]['sum_weights'] += stats.get('sum_weights', 0.0)

        # Add fresh calculations from current runs
        for run_key, run_data in runs.items():
            try:
                init_time = datetime.fromisoformat(run_key)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            forecast_hours = run_data.get("forecast_hours", [])
            model_data = run_data.get(source_model)

            if not model_data:
                continue

            qualifying_precip, qualifying_precip_24hr = _qualifying_precip_sets(run_data, stations)

            for station_id, fcst_data in model_data.items():
                if station_id not in stations:
                    continue

                for i, lt in enumerate(forecast_hours):
                    valid_time = init_time + timedelta(hours=lt)
                    if valid_time >= now:
                        continue
                    if cutoff is not None and valid_time < cutoff:
                        continue
                    if valid_hour is not None and valid_time.hour != valid_hour:
                        continue

                    obs = get_composite_observation(db, station_id, valid_time)
                    if not obs:
                        continue

                    # Temperature
                    fcst_temps = fcst_data.get('temps', [])
                    if i < len(fcst_temps) and fcst_temps[i] is not None and obs.get('temp') is not None:
                        fcst_temp = _apply_model_value_adjustment(
                            model, "temp", station_id, valid_time, fcst_temps[i],
                            kenny_temp_station_hour_biases, kenny_temp_global_hour_biases
                        )
                        error = fcst_temp - obs['temp']
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

                    # Precipitation (6-hour accumulated) — union-based qualifying (same as cache)
                    fcst_precips = fcst_data.get('precips', [])
                    if i < len(fcst_precips):
                        fcst_val = fcst_precips[i]
                        obs_val = obs.get('precip_6hr')
                        if fcst_val is not None and obs_val is not None and (
                            (station_id, i) in qualifying_precip or obs_val >= 0.01
                        ):
                            error = fcst_val - obs_val
                            aggregated_stats[lt]['precip']['sum_abs_errors'] += abs(error)
                            aggregated_stats[lt]['precip']['sum_errors'] += error
                            aggregated_stats[lt]['precip']['count'] += 1
                            weight = _precip_weight(obs_val)
                            aggregated_stats[lt]['precip']['sum_weighted_abs_errors'] += abs(error) * weight
                            aggregated_stats[lt]['precip']['sum_weights'] += weight

                    # Precipitation (24-hour accumulated) — union-based qualifying (same as cache)
                    fcst_precips_24hr = fcst_data.get('precips_24hr', [])
                    if i < len(fcst_precips_24hr):
                        fcst_val = fcst_precips_24hr[i]
                        obs_val = obs.get('precip_24hr')
                        if fcst_val is not None and obs_val is not None and (
                            (station_id, i) in qualifying_precip_24hr or obs_val >= 0.01
                        ):
                            error = fcst_val - obs_val
                            aggregated_stats[lt]['precip_24hr']['sum_abs_errors'] += abs(error)
                            aggregated_stats[lt]['precip_24hr']['sum_errors'] += error
                            aggregated_stats[lt]['precip_24hr']['count'] += 1
                            weight = _precip_weight(obs_val)
                            aggregated_stats[lt]['precip_24hr']['sum_weighted_abs_errors'] += abs(error) * weight
                            aggregated_stats[lt]['precip_24hr']['sum_weights'] += weight

                    # Dewpoint
                    fcst_dewpoints = fcst_data.get('dewpoints', [])
                    if i < len(fcst_dewpoints) and fcst_dewpoints[i] is not None and obs.get('dewpoint') is not None:
                        fcst_dew = _apply_model_value_adjustment(
                            model, "dewpoint", station_id, valid_time, fcst_dewpoints[i],
                            kenny_dew_station_hour_biases, kenny_dew_global_hour_biases
                        )
                        error = fcst_dew - obs['dewpoint']
                        aggregated_stats[lt]['dewpoint']['sum_abs_errors'] += abs(error)
                        aggregated_stats[lt]['dewpoint']['sum_errors'] += error
                        aggregated_stats[lt]['dewpoint']['count'] += 1

        # Calculate mean MAE and Bias for each lead time
        result = {
            "lead_times": lead_times,
            "temp_mae": [],
            "temp_bias": [],
            "mslp_mae": [],
            "mslp_bias": [],
            "precip_mae": [],
            "precip_bias": [],
            "precip_wmae": [],
            "precip_24hr_mae": [],
            "precip_24hr_bias": [],
            "precip_24hr_wmae": [],
            "dewpoint_mae": [],
            "dewpoint_bias": [],
        }

        for lt in lead_times:
            # Temperature
            temp_stats = aggregated_stats[lt]['temp']
            if temp_stats['count'] > 0:
                result["temp_mae"].append(round(temp_stats['sum_abs_errors'] / temp_stats['count'], 2))
                result["temp_bias"].append(_bias_corrected_metric_value(model, "temp_bias", round(temp_stats['sum_abs_errors'] / temp_stats['count'], 2), round(temp_stats['sum_errors'] / temp_stats['count'], 2)))
            else:
                result["temp_mae"].append(None)
                result["temp_bias"].append(None)

            # MSLP
            mslp_stats = aggregated_stats[lt]['mslp']
            if mslp_stats['count'] > 0:
                result["mslp_mae"].append(round(mslp_stats['sum_abs_errors'] / mslp_stats['count'], 2))
                result["mslp_bias"].append(_bias_corrected_metric_value(model, "mslp_bias", round(mslp_stats['sum_abs_errors'] / mslp_stats['count'], 2), round(mslp_stats['sum_errors'] / mslp_stats['count'], 2)))
            else:
                result["mslp_mae"].append(None)
                result["mslp_bias"].append(None)

            # Precipitation (6-hour)
            precip_stats = aggregated_stats[lt]['precip']
            if precip_stats['count'] > 0:
                precip_mae = precip_stats['sum_abs_errors'] / precip_stats['count']
                result["precip_mae"].append(round(precip_mae, 2))
                result["precip_bias"].append(_bias_corrected_metric_value(model, "precip_bias", round(precip_mae, 2), round(precip_stats['sum_errors'] / precip_stats['count'], 2)))
                result["precip_wmae"].append(round(_stats_wmae(precip_stats, fallback_mae=precip_mae), 2))
            else:
                result["precip_mae"].append(None)
                result["precip_bias"].append(None)
                result["precip_wmae"].append(None)

            # Precipitation (24-hour)
            p24_stats = aggregated_stats[lt]['precip_24hr']
            if p24_stats['count'] > 0:
                p24_mae = p24_stats['sum_abs_errors'] / p24_stats['count']
                result["precip_24hr_mae"].append(round(p24_mae, 2))
                result["precip_24hr_bias"].append(_bias_corrected_metric_value(model, "precip_24hr_bias", round(p24_mae, 2), round(p24_stats['sum_errors'] / p24_stats['count'], 2)))
                result["precip_24hr_wmae"].append(round(_stats_wmae(p24_stats, fallback_mae=p24_mae), 2))
            else:
                result["precip_24hr_mae"].append(None)
                result["precip_24hr_bias"].append(None)
                result["precip_24hr_wmae"].append(None)

            # Dewpoint
            dewpoint_stats = aggregated_stats[lt]['dewpoint']
            if dewpoint_stats['count'] > 0:
                result["dewpoint_mae"].append(round(dewpoint_stats['sum_abs_errors'] / dewpoint_stats['count'], 2))
                result["dewpoint_bias"].append(_bias_corrected_metric_value(model, "dewpoint_bias", round(dewpoint_stats['sum_abs_errors'] / dewpoint_stats['count'], 2), round(dewpoint_stats['sum_errors'] / dewpoint_stats['count'], 2)))
            else:
                result["dewpoint_mae"].append(None)
                result["dewpoint_bias"].append(None)

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

    # Collect all forecast hours available in the data
    all_forecast_hours = set()
    for run_data in runs.values():
        all_forecast_hours.update(run_data.get("forecast_hours", []))

    # Also include lead times from cumulative stats
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_cumulative = cumulative_by_lead_time.get(model, {})
        for var in ['temp', 'mslp', 'precip', 'dewpoint']:
            var_stats = model_cumulative.get(var, {})
            all_forecast_hours.update(int(lt) for lt in var_stats.keys())

    # Define verification lead times (key hours for computing statistics)
    # Only compute verification stats for these specific lead times to save computation
    verification_lead_times = set(list(range(6, 25, 6)) + list(range(48, 361, 24)))

    # Filter to only compute stats for verification lead times that exist in the data
    lead_times = sorted([lt for lt in all_forecast_hours if lt in verification_lead_times])

    def _snap_to_canonical_lt(lt: int) -> Optional[int]:
        """Return nearest canonical lead time for lt, or None if gap > 12h."""
        lo = bisect.bisect_left(lead_times, lt)
        best = None
        best_dist = 13  # must be <= 12 to qualify
        if lo < len(lead_times):
            d = lead_times[lo] - lt
            if d <= 12:
                best, best_dist = lead_times[lo], d
        if lo > 0:
            d = lt - lead_times[lo - 1]
            if d < best_dist:
                best, best_dist = lead_times[lo - 1], d
        return best

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

    # Build sorted timestamp index for O(log n) binary-search lookups
    logger.info("Building sorted observation index for binary search...")
    obs_sorted_ts = {}   # {station_id: [float unix timestamp]}
    obs_sorted_keys = {} # {station_id: [time_str]}
    for station_id, station_obs in obs_cache.items():
        pairs = []
        for t in station_obs:
            try:
                pairs.append((datetime.fromisoformat(t).timestamp(), t))
            except ValueError:
                pass
        pairs.sort()
        obs_sorted_ts[station_id] = [p[0] for p in pairs]
        obs_sorted_keys[station_id] = [p[1] for p in pairs]
    logger.info(f"Built sorted index for {len(obs_sorted_ts)} stations")

    # Phase 2: Eagerly accumulate mature runs (>16 days old) into cumulative_stats so
    # they can be skipped in the fresh-computation loop on this and all future syncs.
    MATURE_RUN_HOURS = 16 * 24
    accumulated_run_keys = db.get("cumulative_stats", {}).get("accumulated_run_keys", {})
    to_accumulate = []
    for k in runs:
        if k in accumulated_run_keys:
            continue
        try:
            run_init = datetime.fromisoformat(k)
            if run_init.tzinfo is None:
                run_init = run_init.replace(tzinfo=timezone.utc)
            if (now - run_init).total_seconds() / 3600 >= MATURE_RUN_HOURS:
                to_accumulate.append(k)
        except ValueError:
            pass
    if to_accumulate:
        logger.info(f"Eagerly accumulating {len(to_accumulate)} mature runs into cumulative stats...")
        for run_key in to_accumulate:
            accumulate_stats_from_run(db, run_key, runs[run_key])
            db["cumulative_stats"]["accumulated_run_keys"][run_key] = now.isoformat()
        save_asos_forecasts_db(db)
        accumulated_run_keys = db["cumulative_stats"]["accumulated_run_keys"]
        # Refresh cumulative stat references so the loading step below sees new data
        cumulative_by_station = db["cumulative_stats"]["by_station"]
        cumulative_by_lead_time = db["cumulative_stats"]["by_lead_time"]
        logger.info(f"Eager accumulation complete ({len(accumulated_run_keys)} total accumulated runs)")

    # Cache for 6hr precip calculations to avoid recomputing
    precip_6hr_cache = {}  # {(station_id, time_str): value}

    def fast_calculate_6hr_precip(station_id: str, end_time: datetime) -> Optional[float]:
        """
        Binary-search version of calculate_6hr_precip_total() for use inside
        precompute_verification_cache().  Uses obs_sorted_ts/keys instead of
        iterating all observations.  Logic is otherwise identical to the module-
        level function: overflow filter → stuck-gauge detection → hourly max.
        """
        if station_id in PRECIP_EXCLUDE_STATIONS:
            return None
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None

        end_ts = end_time.timestamp()
        start_ts = end_ts - 6 * 3600.0
        # Exclusive-start, inclusive-end window: start_ts < ts <= end_ts
        lo = bisect.bisect_right(sorted_ts, start_ts)
        hi = bisect.bisect_right(sorted_ts, end_ts)

        if lo >= hi:
            return None

        window_obs = []
        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys[idx])
            if obs_data is None:
                continue
            precip = obs_data.get('precip')
            if precip is None or _is_implausible_precip_value(station_id, precip):
                continue
            obs_time = datetime.fromtimestamp(sorted_ts[idx], tz=timezone.utc)
            window_obs.append((obs_time, precip))

        if not window_obs:
            return None

        # Pre-filter tipping-bucket overflow artifacts (2.56", 5.12", …)
        window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]
        if not window_obs:
            return None

        # Detect stuck-gauge: non-zero value repeated consecutively
        stuck_times: set = set()
        if len(window_obs) >= 2:
            for i in range(len(window_obs) - 1):
                v0, v1 = window_obs[i][1], window_obs[i + 1][1]
                if v0 > 0 and v0 == v1:
                    j = i
                    while j < len(window_obs) and window_obs[j][1] == v0:
                        stuck_times.add(window_obs[j][0])
                        j += 1

        # Group by clock hour, take max p01i (= full METAR total at :53)
        hourly_max: dict = {}
        for obs_time, precip in window_obs:
            if obs_time in stuck_times:
                continue
            hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_max or precip > hourly_max[hour_key]:
                hourly_max[hour_key] = precip

        if not hourly_max:
            return None

        total = sum(hourly_max.values())
        if station_id in PRECIP_MM_STATIONS:
            total /= 25.4
        return total

    precip_24hr_cache = {}  # {(station_id, time_str): value}

    def fast_calculate_24hr_precip(station_id: str, end_time: datetime) -> Optional[float]:
        """
        Binary-search version of calculate_24hr_precip_total() for use inside
        precompute_verification_cache(). Returns None if end_time.hour != 12 or
        fewer than 20 hours have data. Logic otherwise identical to the module-level
        function: overflow filter → stuck-gauge detection → hourly max.
        """
        if end_time.hour != 12:
            return None
        if station_id in PRECIP_EXCLUDE_STATIONS:
            return None
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None

        end_ts = end_time.timestamp()
        start_ts = end_ts - 24 * 3600.0
        lo = bisect.bisect_right(sorted_ts, start_ts)
        hi = bisect.bisect_right(sorted_ts, end_ts)

        if lo >= hi:
            return None

        window_obs = []
        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys[idx])
            if obs_data is None:
                continue
            precip = obs_data.get('precip')
            if precip is None or _is_implausible_precip_value(station_id, precip):
                continue
            obs_time = datetime.fromtimestamp(sorted_ts[idx], tz=timezone.utc)
            window_obs.append((obs_time, precip))

        if not window_obs:
            return None

        window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]
        if not window_obs:
            return None

        stuck_times: set = set()
        if len(window_obs) >= 2:
            for i in range(len(window_obs) - 1):
                v0, v1 = window_obs[i][1], window_obs[i + 1][1]
                if v0 > 0 and v0 == v1:
                    j = i
                    while j < len(window_obs) and window_obs[j][1] == v0:
                        stuck_times.add(window_obs[j][0])
                        j += 1

        hourly_max: dict = {}
        for obs_time, precip in window_obs:
            if obs_time in stuck_times:
                continue
            hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_max or precip > hourly_max[hour_key]:
                hourly_max[hour_key] = precip

        if len(hourly_max) < 20:
            return None

        total = sum(hourly_max.values())
        if station_id in PRECIP_MM_STATIONS:
            total /= 25.4
        return total

    def get_cached_observation(station_id: str, target_time: datetime, max_delta_minutes: int = 30):
        """
        Fast observation lookup using sorted index + binary search (O(log n)).

        Returns composite observation with each variable from its nearest available
        time.  Handles ASOS stations that report MSLP every 5 min but temp/precip
        only at :56 (hourly METARs).
        """
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None

        target_ts = target_time.timestamp()
        window_s = max_delta_minutes * 60.0

        # Narrow to the ±window candidates with two binary searches
        lo = bisect.bisect_left(sorted_ts, target_ts - window_s)
        hi = bisect.bisect_right(sorted_ts, target_ts + window_s)

        if lo >= hi:
            return None

        best_temp_val = None
        best_temp_delta = window_s + 1.0
        best_mslp_val = None
        best_mslp_delta = window_s + 1.0
        best_precip_val = None
        best_precip_delta = window_s + 1.0
        best_dewpoint_val = None
        best_dewpoint_delta = window_s + 1.0

        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys[idx])
            if obs_data is None:
                continue
            delta = abs(sorted_ts[idx] - target_ts)

            v = obs_data.get('temp')
            if v is not None and _is_implausible_temp_value(v):
                v = None
            if v is not None and delta < best_temp_delta:
                best_temp_val = v
                best_temp_delta = delta

            v = None if station_id in PRESSURE_EXCLUDE_STATIONS else obs_data.get('mslp')
            if v is not None and _is_implausible_pressure_value(v):
                v = None
            if v is not None and delta < best_mslp_delta:
                best_mslp_val = v
                best_mslp_delta = delta

            v = obs_data.get('precip')
            if v is not None and delta < best_precip_delta:
                best_precip_val = v
                best_precip_delta = delta

            v = obs_data.get('dewpoint')
            if v is not None and _is_implausible_dewpoint_value(v):
                v = None
            if v is not None and delta < best_dewpoint_delta:
                best_dewpoint_val = v
                best_dewpoint_delta = delta

        if best_temp_val is None and best_mslp_val is None and best_precip_val is None and best_dewpoint_val is None:
            return None

        composite_obs = {
            'temp': best_temp_val,
            'mslp': best_mslp_val,
            'precip': best_precip_val,
            'dewpoint': best_dewpoint_val,
        }

        # Calculate 6hr precip on demand for synoptic times only
        if target_time.hour % 6 == 0:
            cache_key = (station_id, target_time.isoformat())
            if cache_key in precip_6hr_cache:
                composite_obs['precip_6hr'] = precip_6hr_cache[cache_key]
            else:
                precip_6hr = fast_calculate_6hr_precip(station_id, target_time)
                precip_6hr_cache[cache_key] = precip_6hr
                composite_obs['precip_6hr'] = precip_6hr
        else:
            composite_obs['precip_6hr'] = None

        # Calculate 24hr precip for 12Z times only
        if target_time.hour == 12:
            p24_key = (station_id, target_time.isoformat())
            if p24_key not in precip_24hr_cache:
                precip_24hr_cache[p24_key] = fast_calculate_24hr_precip(station_id, target_time)
            composite_obs['precip_24hr'] = precip_24hr_cache[p24_key]
        else:
            composite_obs['precip_24hr'] = None

        return composite_obs

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
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }

    # Initialize aggregated stats for by_station and by_lead_time
    # Structure: station_stats[station_id][model][var][lt_str] = {sum_abs_errors, sum_errors, count}
    station_stats = {}
    for station_id in stations:
        station_stats[station_id] = {
            'gfs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
            'aifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
            'ifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
            'nws': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        }

    # Structure: aggregated_stats[model][var][lt_str] = {sum_abs_errors, sum_errors, count}
    aggregated_stats = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'nws': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
    }
    # Structure: hourly_aggregated_stats[model][var][lt_str][vh_str] = {sum_abs, sum, count}
    hourly_aggregated_stats = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'nws': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
    }
    # Structure: station_hourly_stats[station_id][model][var][snapped_lt_str][vh_str] = {sum_abs, sum, count}
    # Built from fresh (non-accumulated) runs only; used for by_station_by_valid_hour in cache.
    station_hourly_stats: dict = {}

    # Initialize with zeros for all lead times
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            for lt in lead_times:
                lt_str = str(lt)
                aggregated_stats[model][var][lt_str] = {
                    'sum_abs_errors': 0.0,
                    'sum_errors': 0.0,
                    'count': 0,
                    'sum_weighted_abs_errors': 0.0,
                    'sum_weights': 0.0,
                }
                hourly_aggregated_stats[model][var][lt_str] = {}
                for station_id in stations:
                    station_stats[station_id][model][var][lt_str] = {
                        'sum_abs_errors': 0.0,
                        'sum_errors': 0.0,
                        'count': 0,
                        'sum_weighted_abs_errors': 0.0,
                        'sum_weights': 0.0,
                    }

    logger.info(f"Loading cumulative stats for {len(stations)} stations...")

    # Load cumulative stats
    for station_id in stations:
        if station_id in cumulative_by_station:
            for model in ['gfs', 'aifs', 'ifs', 'nws']:
                model_cumulative = cumulative_by_station[station_id].get(model, {})
                for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
                    var_cumulative = model_cumulative.get(var, {})
                    for lt_str, stats in var_cumulative.items():
                        if int(lt_str) in lead_times:
                            station_stats[station_id][model][var][lt_str]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                            station_stats[station_id][model][var][lt_str]['sum_errors'] += stats.get('sum_errors', 0.0)
                            station_stats[station_id][model][var][lt_str]['count'] += stats.get('count', 0)
                            station_stats[station_id][model][var][lt_str]['sum_weighted_abs_errors'] += stats.get('sum_weighted_abs_errors', 0.0)
                            station_stats[station_id][model][var][lt_str]['sum_weights'] += stats.get('sum_weights', 0.0)

    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_cumulative = cumulative_by_lead_time.get(model, {})
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, stats in var_cumulative.items():
                if int(lt_str) in lead_times:
                    aggregated_stats[model][var][lt_str]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    aggregated_stats[model][var][lt_str]['sum_errors'] += stats.get('sum_errors', 0.0)
                    aggregated_stats[model][var][lt_str]['count'] += stats.get('count', 0)
                    aggregated_stats[model][var][lt_str]['sum_weighted_abs_errors'] += stats.get('sum_weighted_abs_errors', 0.0)
                    aggregated_stats[model][var][lt_str]['sum_weights'] += stats.get('sum_weights', 0.0)

    # Load cumulative hourly breakdown (snap any non-canonical LT to nearest canonical)
    cumulative_by_lt_by_vh = db.get("cumulative_stats", {}).get("by_lead_time_by_valid_hour", {})
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_cumulative = cumulative_by_lt_by_vh.get(model, {})
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, vh_data in var_cumulative.items():
                snapped = _snap_to_canonical_lt(int(lt_str))
                if snapped is None:
                    continue
                snapped_lt_str = str(snapped)
                for vh_str, stats in vh_data.items():
                    if vh_str not in hourly_aggregated_stats[model][var][snapped_lt_str]:
                        hourly_aggregated_stats[model][var][snapped_lt_str][vh_str] = {
                            'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0,
                            'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0
                        }
                    hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_abs_errors'] += stats.get('sum_abs_errors', 0.0)
                    hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_errors'] += stats.get('sum_errors', 0.0)
                    hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['count'] += stats.get('count', 0)
                    hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_weighted_abs_errors'] += stats.get('sum_weighted_abs_errors', 0.0)
                    hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_weights'] += stats.get('sum_weights', 0.0)

    fresh_run_count = sum(1 for k in runs if k not in accumulated_run_keys)
    logger.info(f"Computing fresh stats from {fresh_run_count} current runs (skipping {len(accumulated_run_keys)} accumulated)...")

    # Add fresh calculations from current runs (skip mature runs already in cumulative_stats)
    for run_key, run_data in runs.items():
        if run_key in accumulated_run_keys:
            continue  # already folded into cumulative_stats

        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        # Precompute qualifying precip events across all models for fair cross-model comparison
        qualifying_precip, qualifying_precip_24hr = _qualifying_precip_sets(run_data, stations)

        for model in ['gfs', 'aifs', 'ifs', 'nws']:
            model_data = run_data.get(model)
            if not model_data:
                continue

            for station_id, fcst_data in model_data.items():
                if station_id not in stations:
                    continue

                lead_times_set = set(lead_times)
                for i, lt in enumerate(forecast_hours):
                    is_canonical = lt in lead_times_set
                    snapped_lt = lt if is_canonical else _snap_to_canonical_lt(lt)

                    # Skip if not canonical and can't be snapped to any canonical bucket
                    if not is_canonical and snapped_lt is None:
                        continue

                    valid_time = init_time + timedelta(hours=lt)
                    if valid_time >= now:
                        continue

                    obs = get_cached_observation(station_id, valid_time)
                    if not obs:
                        continue

                    lt_str = str(lt)
                    snapped_lt_str = str(snapped_lt)

                    # Process each variable
                    for var, (fcst_key, obs_key) in var_map.items():
                        fcst_values = fcst_data.get(fcst_key, [])
                        if i >= len(fcst_values) or fcst_values[i] is None:
                            continue
                        if obs.get(obs_key) is None:
                            continue

                        fcst_val = fcst_values[i]
                        obs_val = obs[obs_key]
                        if var == 'precip' and (station_id, i) not in qualifying_precip and obs_val < 0.01:
                            continue
                        if var == 'precip_24hr' and (station_id, i) not in qualifying_precip_24hr and obs_val < 0.01:
                            continue
                        error = fcst_val - obs_val
                        abs_error = abs(error)

                        if is_canonical:
                            # Update station stats and aggregated stats (canonical LTs only)
                            station_stats[station_id][model][var][lt_str]['sum_abs_errors'] += abs_error
                            station_stats[station_id][model][var][lt_str]['sum_errors'] += error
                            station_stats[station_id][model][var][lt_str]['count'] += 1
                            if _is_precip_var(var):
                                weight = _precip_weight(obs_val)
                                station_stats[station_id][model][var][lt_str]['sum_weighted_abs_errors'] += abs_error * weight
                                station_stats[station_id][model][var][lt_str]['sum_weights'] += weight

                            aggregated_stats[model][var][lt_str]['sum_abs_errors'] += abs_error
                            aggregated_stats[model][var][lt_str]['sum_errors'] += error
                            aggregated_stats[model][var][lt_str]['count'] += 1
                            if _is_precip_var(var):
                                weight = _precip_weight(obs_val)
                                aggregated_stats[model][var][lt_str]['sum_weighted_abs_errors'] += abs_error * weight
                                aggregated_stats[model][var][lt_str]['sum_weights'] += weight

                        # Update hourly aggregated stats for all lead times, snapped to nearest canonical
                        vh_str = str(valid_time.hour)
                        if vh_str not in hourly_aggregated_stats[model][var][snapped_lt_str]:
                            hourly_aggregated_stats[model][var][snapped_lt_str][vh_str] = {
                                'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0,
                                'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0
                            }
                        hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_abs_errors'] += abs_error
                        hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_errors'] += error
                        hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['count'] += 1
                        if _is_precip_var(var):
                            weight = _precip_weight(obs_val)
                            hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_weighted_abs_errors'] += abs_error * weight
                            hourly_aggregated_stats[model][var][snapped_lt_str][vh_str]['sum_weights'] += weight

                        # Update per-station hourly stats (for by_station_by_valid_hour in cache)
                        s_h_station = (station_hourly_stats
                                       .setdefault(station_id, {})
                                       .setdefault(model, {})
                                       .setdefault(var, {})
                                       .setdefault(snapped_lt_str, {}))
                        if vh_str not in s_h_station:
                            s_h_station[vh_str] = {
                                'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0,
                                'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0
                            }
                        s_h_station[vh_str]['sum_abs_errors'] += abs_error
                        s_h_station[vh_str]['sum_errors'] += error
                        s_h_station[vh_str]['count'] += 1
                        if _is_precip_var(var):
                            weight = _precip_weight(obs_val)
                            s_h_station[vh_str]['sum_weighted_abs_errors'] += abs_error * weight
                            s_h_station[vh_str]['sum_weights'] += weight

    logger.info("Finalizing cache data...")

    # Precompute time series data (all historical data) using cached observations
    logger.info("Computing verification time series with cached observations...")
    time_series_data = precompute_verification_time_series(db, lead_times, obs_cache=obs_cache, get_cached_obs_fn=get_cached_observation, skip_run_keys=accumulated_run_keys)

    # Convert to final cache format - by_station
    for station_id in stations:
        cache_data["by_station"][station_id] = {}
        for model in ['gfs', 'aifs', 'ifs', 'nws']:
            cache_data["by_station"][station_id][model] = {}
            for lt in lead_times:
                lt_str = str(lt)
                cache_data["by_station"][station_id][model][lt_str] = {}
                for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
                    stats = station_stats[station_id][model][var][lt_str]
                    if stats['count'] > 0:
                        mae = stats['sum_abs_errors'] / stats['count']
                        entry = {
                            'mae': round(mae, 2),
                            'bias': round(stats['sum_errors'] / stats['count'], 2),
                            'count': stats['count']
                        }
                        if _is_precip_var(var):
                            entry['wmae'] = round(_stats_wmae(stats, fallback_mae=mae), 2)
                        cache_data["by_station"][station_id][model][lt_str][var] = {
                            **entry
                        }
                    else:
                        cache_data["by_station"][station_id][model][lt_str][var] = None

    # Convert to final cache format - by_lead_time
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        cache_data["by_lead_time"][model] = {}
        for lt in lead_times:
            lt_str = str(lt)
            cache_data["by_lead_time"][model][lt_str] = {}
            for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
                stats = aggregated_stats[model][var][lt_str]
                if stats['count'] > 0:
                    mae = stats['sum_abs_errors'] / stats['count']
                    entry = {
                        'mae': round(mae, 2),
                        'bias': round(stats['sum_errors'] / stats['count'], 2)
                    }
                    if _is_precip_var(var):
                        entry['wmae'] = round(_stats_wmae(stats, fallback_mae=mae), 2)
                    cache_data["by_lead_time"][model][lt_str][var] = {
                        **entry
                    }
                else:
                    cache_data["by_lead_time"][model][lt_str][var] = {
                        'mae': None,
                        'bias': None,
                        **({'wmae': None} if _is_precip_var(var) else {}),
                    }

    # Convert hourly_aggregated_stats to final format
    cache_data["by_lead_time_by_valid_hour"] = {}
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        cache_data["by_lead_time_by_valid_hour"][model] = {}
        for lt in lead_times:
            lt_str = str(lt)
            cache_data["by_lead_time_by_valid_hour"][model][lt_str] = {}
            for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
                for vh_str, stats in hourly_aggregated_stats[model][var][lt_str].items():
                    if vh_str not in cache_data["by_lead_time_by_valid_hour"][model][lt_str]:
                        cache_data["by_lead_time_by_valid_hour"][model][lt_str][vh_str] = {}
                    entry = cache_data["by_lead_time_by_valid_hour"][model][lt_str][vh_str]
                    if stats['count'] > 0:
                        mae = stats['sum_abs_errors'] / stats['count']
                        var_entry = {
                            'mae': round(mae, 2),
                            'bias': round(stats['sum_errors'] / stats['count'], 2)
                        }
                        if _is_precip_var(var):
                            var_entry['wmae'] = round(_stats_wmae(stats, fallback_mae=mae), 2)
                        entry[var] = {
                            **var_entry
                        }
                    else:
                        entry[var] = {'mae': None, 'bias': None, **({'wmae': None} if _is_precip_var(var) else {})}

    # Convert station_hourly_stats to final format - by_station_by_valid_hour
    # (covers fresh/non-accumulated runs; grows over time as runs remain unaccumulated)
    cache_data["by_station_by_valid_hour"] = {}
    for station_id, model_data in station_hourly_stats.items():
        for model, var_data in model_data.items():
            for var, lt_data in var_data.items():
                for lt_str, vh_data in lt_data.items():
                    for vh_str, stats in vh_data.items():
                        if stats['count'] > 0:
                            entry = (cache_data["by_station_by_valid_hour"]
                                     .setdefault(station_id, {})
                                     .setdefault(model, {})
                                     .setdefault(lt_str, {})
                                     .setdefault(vh_str, {}))
                            mae = stats['sum_abs_errors'] / stats['count']
                            var_entry = {
                                'mae': round(mae, 2),
                                'bias': round(stats['sum_errors'] / stats['count'], 2),
                                'count': stats['count']
                            }
                            if _is_precip_var(var):
                                var_entry['wmae'] = round(_stats_wmae(stats, fallback_mae=mae), 2)
                            entry[var] = {
                                **var_entry
                            }

    # Add time series data to cache
    cache_data["time_series"] = time_series_data

    # Save to file (atomic write — never leave the cache in a half-written state)
    try:
        tmp = ASOS_VERIFICATION_CACHE_FILE.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(cache_data, f, indent=2)
        os.replace(tmp, ASOS_VERIFICATION_CACHE_FILE)

        elapsed = time.time() - start_time
        logger.info(f"Verification cache precomputed and saved in {elapsed:.1f}s to {ASOS_VERIFICATION_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving verification cache: {e}")

    return cache_data


def load_verification_cache() -> Optional[dict]:
    """
    Load precomputed verification cache from file, with in-memory mtime cache.

    Returns:
        Dict with cached verification data, or None if cache doesn't exist
    """
    global _verification_cache_data, _verification_cache_mtime
    if not ASOS_VERIFICATION_CACHE_FILE.exists():
        logger.warning(f"Verification cache file not found: {ASOS_VERIFICATION_CACHE_FILE}")
        return None

    try:
        current_mtime = ASOS_VERIFICATION_CACHE_FILE.stat().st_mtime
        if _verification_cache_data is not None and _verification_cache_mtime == current_mtime:
            return _verification_cache_data
        with open(ASOS_VERIFICATION_CACHE_FILE) as f:
            cache_data = json.load(f)
        last_updated = cache_data.get("last_updated", "unknown")
        logger.info(f"Loaded verification cache (last updated: {last_updated})")
        _verification_cache_data = cache_data
        _verification_cache_mtime = current_mtime
        return _verification_cache_data
    except Exception as e:
        logger.error(f"Error loading verification cache: {e}")
        return None


def precompute_verification_time_series(db: dict, lead_times: list, days_back: int = None, obs_cache: dict = None, get_cached_obs_fn=None, skip_run_keys: dict = None) -> dict:
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
        skip_run_keys: Set/dict of run keys already folded into cumulative_stats (Phase 2)

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
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }

    # Structure: time_series[model][var][lt][date] = daily stats dict
    time_series = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'nws': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
    }

    # Initialize for all lead times
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            for lt in lead_times:
                time_series[model][var][lt] = {}

    # Load cumulative time series data (historical data from deleted runs)
    logger.info("Loading cumulative time series data...")
    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        model_cumulative = cumulative_time_series.get(model, {})
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            var_cumulative = model_cumulative.get(var, {})
            for lt_str, date_errors in var_cumulative.items():
                lt = int(lt_str)
                if lt in lead_times:
                    merged = {}
                    for date_key, payload in date_errors.items():
                        if isinstance(payload, dict):
                            merged[date_key] = {
                                "sum_abs_errors": float(payload.get("sum_abs_errors", 0.0)),
                                "sum_errors": float(payload.get("sum_errors", 0.0)),
                                "count": int(payload.get("count", 0)),
                                "sum_weighted_abs_errors": float(payload.get("sum_weighted_abs_errors", 0.0)),
                                "sum_weights": float(payload.get("sum_weights", 0.0)),
                            }
                        else:
                            # Backward-compatible path for older caches that stored per-day error lists.
                            errs = payload if isinstance(payload, list) else []
                            merged[date_key] = {
                                "sum_abs_errors": float(sum(abs(e) for e in errs)),
                                "sum_errors": float(sum(errs)),
                                "count": len(errs),
                                "sum_weighted_abs_errors": 0.0,
                                "sum_weights": 0.0,
                            }
                    time_series[model][var][lt] = merged

    # Collect errors by date from current runs (skip mature runs already in cumulative)
    for run_key, run_data in runs.items():
        if skip_run_keys and run_key in skip_run_keys:
            continue  # already represented in cumulative_time_series

        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if init_time < cutoff_date:
            continue

        forecast_hours = run_data.get("forecast_hours", [])

        # Precompute qualifying precip events across all models for fair cross-model comparison
        qualifying_precip, qualifying_precip_24hr = _qualifying_precip_sets(run_data, stations)

        for model in ['gfs', 'aifs', 'ifs', 'nws']:
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
                        if var == 'precip' and (station_id, fcst_idx) not in qualifying_precip and obs_val < 0.01:
                            continue
                        if var == 'precip_24hr' and (station_id, fcst_idx) not in qualifying_precip_24hr and obs_val < 0.01:
                            continue
                        error = fcst_val - obs_val

                        # Store daily aggregated stats (merging with cumulative historical data)
                        if date_key not in time_series[model][var][lt]:
                            time_series[model][var][lt][date_key] = {
                                "sum_abs_errors": 0.0,
                                "sum_errors": 0.0,
                                "count": 0,
                                "sum_weighted_abs_errors": 0.0,
                                "sum_weights": 0.0,
                            }
                        daily = time_series[model][var][lt][date_key]
                        daily["sum_abs_errors"] += abs(error)
                        daily["sum_errors"] += error
                        daily["count"] += 1
                        if _is_precip_var(var):
                            weight = _precip_weight(obs_val)
                            daily["sum_weighted_abs_errors"] += abs(error) * weight
                            daily["sum_weights"] += weight

    # Convert to final format with MAE/bias per day
    result = {
        'gfs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'aifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'ifs': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
        'nws': {'temp': {}, 'mslp': {}, 'precip': {}, 'precip_24hr': {}, 'dewpoint': {}},
    }

    for model in ['gfs', 'aifs', 'ifs', 'nws']:
        for var in ['temp', 'mslp', 'precip', 'precip_24hr', 'dewpoint']:
            for lt in lead_times:
                errors_by_date = time_series[model][var][lt]

                if not errors_by_date:
                    continue

                dates = sorted(errors_by_date.keys())
                daily_mae = []
                daily_wmae = []
                daily_bias = []
                daily_counts = []

                for date in dates:
                    stats = errors_by_date[date]
                    count = int(stats.get("count", 0)) if isinstance(stats, dict) else 0
                    if count > 0:
                        mae = stats["sum_abs_errors"] / count
                        bias = stats["sum_errors"] / count
                        wmae = None
                        if _is_precip_var(var):
                            wmae = _stats_wmae(stats, fallback_mae=mae)
                        daily_mae.append(round(mae, 2))
                        daily_wmae.append(round(wmae, 2) if wmae is not None else None)
                        daily_bias.append(round(bias, 2))
                        daily_counts.append(count)
                    else:
                        daily_mae.append(None)
                        daily_wmae.append(None)
                        daily_bias.append(None)
                        daily_counts.append(0)

                result[model][var][lt] = {
                    'dates': dates,
                    'mae': daily_mae,
                    'wmae': daily_wmae,
                    'bias': daily_bias,
                    'counts': daily_counts
                }

    return result


def get_station_detail_monthly(station_id: str, model: str, days_back: int = MONTHLY_WINDOW_DAYS) -> dict:
    """
    Get detailed verification for a single station using the monthly cache.
    """
    if (model or "").lower() == "kenny":
        db = load_asos_forecasts_db()
        station = db.get("stations", {}).get(station_id)
        if not station:
            return {"error": "Station not found"}

        runs = db.get("runs", {})
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)
        kenny_temp_station_hour_biases, kenny_temp_global_hour_biases = _compute_kenny_station_hour_biases(db, "temp")
        kenny_dew_station_hour_biases, kenny_dew_global_hour_biases = _compute_kenny_station_hour_biases(db, "dewpoint")

        lead_times_set = set()
        for run_data in runs.values():
            lead_times_set.update(run_data.get("forecast_hours", []))
        lead_times = sorted(int(lt) for lt in lead_times_set)
        if not lead_times:
            return {"error": "No monthly data for this station"}

        stats = {
            lt: {
                "temp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "mslp": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "precip": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
                "dewpoint": {"sum_abs": 0.0, "sum": 0.0, "count": 0},
            }
            for lt in lead_times
        }

        for run_key, run_data in runs.items():
            try:
                init_time = datetime.fromisoformat(run_key)
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            forecast_hours = run_data.get("forecast_hours", [])
            fcst_data = (run_data.get("aifs") or {}).get(station_id)
            if not fcst_data:
                continue

            for i, lt in enumerate(forecast_hours):
                lt = int(lt)
                valid_time = init_time + timedelta(hours=lt)
                if valid_time >= now or valid_time < cutoff or lt not in stats:
                    continue

                obs = get_stored_observation(db, station_id, valid_time)
                if not obs:
                    continue

                temps = fcst_data.get("temps", [])
                if i < len(temps) and temps[i] is not None and obs.get("temp") is not None:
                    fcst_temp = _apply_model_value_adjustment(
                        "kenny", "temp", station_id, valid_time, temps[i],
                        kenny_temp_station_hour_biases, kenny_temp_global_hour_biases
                    )
                    err = fcst_temp - obs["temp"]
                    stats[lt]["temp"]["sum_abs"] += abs(err)
                    stats[lt]["temp"]["sum"] += err
                    stats[lt]["temp"]["count"] += 1

                mslps = fcst_data.get("mslps", [])
                if i < len(mslps) and mslps[i] is not None and obs.get("mslp") is not None:
                    err = mslps[i] - obs["mslp"]
                    stats[lt]["mslp"]["sum_abs"] += abs(err)
                    stats[lt]["mslp"]["sum"] += err
                    stats[lt]["mslp"]["count"] += 1

                precips = fcst_data.get("precips", [])
                if i < len(precips):
                    fv = precips[i]
                    ov = obs.get("precip_6hr")
                    if should_include_precip(fv, ov):
                        err = fv - ov
                        stats[lt]["precip"]["sum_abs"] += abs(err)
                        stats[lt]["precip"]["sum"] += err
                        stats[lt]["precip"]["count"] += 1

                dewpoints = fcst_data.get("dewpoints", [])
                if i < len(dewpoints) and dewpoints[i] is not None and obs.get("dewpoint") is not None:
                    fcst_dew = _apply_model_value_adjustment(
                        "kenny", "dewpoint", station_id, valid_time, dewpoints[i],
                        kenny_dew_station_hour_biases, kenny_dew_global_hour_biases
                    )
                    err = fcst_dew - obs["dewpoint"]
                    stats[lt]["dewpoint"]["sum_abs"] += abs(err)
                    stats[lt]["dewpoint"]["sum"] += err
                    stats[lt]["dewpoint"]["count"] += 1

        temp_mae, temp_bias = [], []
        mslp_mae, mslp_bias = [], []
        precip_mae, precip_bias = [], []
        dewpoint_mae, dewpoint_bias = [], []

        for lt in lead_times:
            for var, mae_out, bias_out in [
                ("temp", temp_mae, temp_bias),
                ("mslp", mslp_mae, mslp_bias),
                ("precip", precip_mae, precip_bias),
                ("dewpoint", dewpoint_mae, dewpoint_bias),
            ]:
                c = stats[lt][var]["count"]
                if c > 0:
                    mae_out.append(round(stats[lt][var]["sum_abs"] / c, 2))
                    bias_out.append(round(stats[lt][var]["sum"] / c, 2))
                else:
                    mae_out.append(None)
                    bias_out.append(None)

        return {
            "station": station,
            "lead_times": lead_times,
            "temp_mae": temp_mae,
            "temp_bias": temp_bias,
            "mslp_mae": mslp_mae,
            "mslp_bias": mslp_bias,
            "precip_mae": precip_mae,
            "precip_bias": precip_bias,
            "dewpoint_mae": dewpoint_mae,
            "dewpoint_bias": dewpoint_bias,
            "period_days": days_back
        }

    monthly = load_monthly_stats_cache().get("by_station_monthly", {})
    db = load_asos_forecasts_db()

    station = db.get("stations", {}).get(station_id)
    if not station:
        return {"error": "Station not found"}

    source_model = _source_model_for_verification(model)
    model_data = monthly.get(station_id, {}).get(source_model, {})
    if not model_data:
        return {"error": "No monthly data for this station"}

    # Build lead time list from available temp, mslp, precip, or dewpoint data
    lead_times = sorted(
        {int(k) for k in model_data.get("temp", {}).keys()} |
        {int(k) for k in model_data.get("mslp", {}).keys()} |
        {int(k) for k in model_data.get("precip", {}).keys()} |
        {int(k) for k in model_data.get("dewpoint", {}).keys()}
    )

    temp_mae = []
    temp_bias = []
    mslp_mae = []
    mslp_bias = []
    precip_mae = []
    precip_bias = []
    dewpoint_mae = []
    dewpoint_bias = []

    for lt in lead_times:
        lt_str = str(lt)

        temp_stats = model_data.get("temp", {}).get(lt_str)
        if temp_stats and temp_stats.get("count", 0) > 0:
            count = temp_stats["count"]
            temp_mae.append(round(temp_stats["sum_abs_errors"] / count, 2))
            temp_bias.append(_bias_corrected_metric_value(model, "temp_bias", temp_mae[-1], round(temp_stats["sum_errors"] / count, 2)))
        else:
            temp_mae.append(None)
            temp_bias.append(None)

        mslp_stats = model_data.get("mslp", {}).get(lt_str)
        if mslp_stats and mslp_stats.get("count", 0) > 0:
            count = mslp_stats["count"]
            mslp_mae.append(round(mslp_stats["sum_abs_errors"] / count, 2))
            mslp_bias.append(_bias_corrected_metric_value(model, "mslp_bias", mslp_mae[-1], round(mslp_stats["sum_errors"] / count, 2)))
        else:
            mslp_mae.append(None)
            mslp_bias.append(None)

        precip_stats = model_data.get("precip", {}).get(lt_str)
        if precip_stats and precip_stats.get("count", 0) > 0:
            count = precip_stats["count"]
            precip_mae.append(round(precip_stats["sum_abs_errors"] / count, 2))
            precip_bias.append(_bias_corrected_metric_value(model, "precip_bias", precip_mae[-1], round(precip_stats["sum_errors"] / count, 2)))
        else:
            precip_mae.append(None)
            precip_bias.append(None)

        dewpoint_stats = model_data.get("dewpoint", {}).get(lt_str)
        if dewpoint_stats and dewpoint_stats.get("count", 0) > 0:
            count = dewpoint_stats["count"]
            dewpoint_mae.append(round(dewpoint_stats["sum_abs_errors"] / count, 2))
            dewpoint_bias.append(_bias_corrected_metric_value(model, "dewpoint_bias", dewpoint_mae[-1], round(dewpoint_stats["sum_errors"] / count, 2)))
        else:
            dewpoint_mae.append(None)
            dewpoint_bias.append(None)

    return {
        "station": station,
        "lead_times": lead_times,
        "temp_mae": temp_mae,
        "temp_bias": temp_bias,
        "mslp_mae": mslp_mae,
        "mslp_bias": mslp_bias,
        "precip_mae": precip_mae,
        "precip_bias": precip_bias,
        "dewpoint_mae": dewpoint_mae,
        "dewpoint_bias": dewpoint_bias,
        "period_days": days_back
    }


def get_run_counts_by_lead_time(model: str, period: str = "all", valid_hour: Optional[int] = None) -> dict:
    """
    Return forecast run counts by lead time for a model.

    A run is counted for lead time LT when:
    - that model exists for the run,
    - LT exists in the run's forecast_hours, and
    - run init + LT is in the past (verifiable).

    When valid_hour is set, only counts forecasts whose valid time matches that
    UTC hour, snapping non-canonical lead times to the nearest canonical bucket
    (same logic as the hourly verification stats).
    """
    db = load_asos_forecasts_db()
    runs = db.get("runs", {})
    now = datetime.now(timezone.utc)
    period = (period or "all").lower()
    cutoff = now - timedelta(days=MONTHLY_WINDOW_DAYS) if period == "monthly" else None

    verification_lead_times = sorted(list(range(6, 25, 6)) + list(range(48, 361, 24)))

    def _snap(lt: int) -> Optional[int]:
        lo = bisect.bisect_left(verification_lead_times, lt)
        best, best_dist = None, 13
        if lo < len(verification_lead_times):
            d = verification_lead_times[lo] - lt
            if d <= 12:
                best, best_dist = verification_lead_times[lo], d
        if lo > 0:
            d = lt - verification_lead_times[lo - 1]
            if d < best_dist:
                best = verification_lead_times[lo - 1]
        return best

    source_model = _source_model_for_verification(model)
    counts_by_lt: dict[int, int] = {}
    nws_max_lead = 168 if source_model == 'nws' else None

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if cutoff is not None and init_time < cutoff:
            continue

        model_data = run_data.get(source_model)
        if not model_data:
            continue

        lead_times = run_data.get("forecast_hours", [])

        # Track which canonical buckets this run has already contributed to (avoid double-counting)
        counted_buckets: set[int] = set()

        for lt in lead_times:
            lt = int(lt)
            if nws_max_lead is not None and lt > nws_max_lead:
                continue
            valid_time = init_time + timedelta(hours=lt)
            if valid_time >= now:
                continue

            if valid_hour is not None:
                if valid_time.hour != valid_hour:
                    continue
                # Snap to nearest canonical bucket
                bucket = _snap(lt)
                if bucket is None or bucket in counted_buckets:
                    continue
                counted_buckets.add(bucket)
                counts_by_lt[bucket] = counts_by_lt.get(bucket, 0) + 1
            else:
                counts_by_lt[lt] = counts_by_lt.get(lt, 0) + 1

    return counts_by_lt


def get_verification_data_from_cache(
    model: str,
    variable: str,
    lead_time_hours: int,
    valid_hour: Optional[int] = None
) -> Dict[str, dict]:
    """
    Get verification data from cache for a specific model/variable/lead_time.

    Falls back to computing on-the-fly if cache doesn't exist.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
        variable: Variable name ('temp', 'mslp', 'precip')
        lead_time_hours: Lead time in hours
        valid_hour: If set (0, 6, 12, or 18), filter to that UTC valid hour only.
                    Uses by_station_by_valid_hour (fresh runs only); falls back to
                    by_station when no valid_hour is requested.

    Returns:
        Dict mapping station_id to verification data (same format as get_verification_data)
    """
    source_model = _source_model_for_verification(model)
    if _is_kenny_bias_corrected_var(model, variable):
        return get_verification_data(model, variable, lead_time_hours, valid_hour=valid_hour)
    if source_model == 'nws' and lead_time_hours > 168:
        return {}

    cache = load_verification_cache()

    if cache is None:
        logger.warning("Cache not available, computing verification on-the-fly")
        return get_verification_data(model, variable, lead_time_hours)

    lt_str = str(lead_time_hours)
    results = {}
    stations_info = cache.get("stations", {})

    if valid_hour is not None:
        vh_str = str(valid_hour)
        for station_id, model_data in cache.get("by_station_by_valid_hour", {}).items():
            var_data = model_data.get(source_model, {}).get(lt_str, {}).get(vh_str, {}).get(variable)
            if var_data:
                station_info = stations_info.get(station_id, {})
                wmae = var_data.get('wmae')
                if wmae is None and _is_precip_var(variable):
                    wmae = var_data.get('mae')
                mae = var_data.get('mae')
                raw_bias = var_data.get('bias')
                results[station_id] = {
                    'mae': mae,
                    'bias': _bias_corrected_metric_value(model, f"{variable}_bias", mae, raw_bias),
                    'wmae': wmae if _is_precip_var(variable) else None,
                    'count': var_data['count'],
                    'lat': station_info.get('lat'),
                    'lon': station_info.get('lon'),
                    'name': station_info.get('name', station_id),
                    'state': station_info.get('state', '')
                }
    else:
        for station_id, station_data in cache.get("by_station", {}).items():
            model_data = station_data.get(source_model, {})
            lt_data = model_data.get(lt_str, {})
            var_data = lt_data.get(variable)

            if var_data:
                station_info = stations_info.get(station_id, {})
                wmae = var_data.get('wmae')
                if wmae is None and _is_precip_var(variable):
                    wmae = var_data.get('mae')
                mae = var_data.get('mae')
                raw_bias = var_data.get('bias')
                results[station_id] = {
                    'mae': mae,
                    'bias': _bias_corrected_metric_value(model, f"{variable}_bias", mae, raw_bias),
                    'wmae': wmae if _is_precip_var(variable) else None,
                    'count': var_data['count'],
                    'lat': station_info.get('lat'),
                    'lon': station_info.get('lon'),
                    'name': station_info.get('name', station_id),
                    'state': station_info.get('state', '')
                }

    return results


def get_verification_data_from_monthly_cache(
    model: str,
    variable: str,
    lead_time_hours: int,
    valid_hour: Optional[int] = None
) -> Dict[str, dict]:
    """
    Get verification data from the rolling monthly cache.

    Args:
        model: Model name
        variable: Variable name
        lead_time_hours: Lead time in hours
        valid_hour: If set (0, 6, 12, or 18), filter to that UTC valid hour only.
    """
    source_model = _source_model_for_verification(model)
    if _is_kenny_bias_corrected_var(model, variable):
        return get_verification_data_recent(model, variable, lead_time_hours, days_back=MONTHLY_WINDOW_DAYS, valid_hour=valid_hour)
    if source_model == 'nws' and lead_time_hours > 168:
        return {}

    monthly_cache = load_monthly_stats_cache()
    db = load_asos_forecasts_db()
    stations_info = db.get("stations", {})
    results = {}

    if valid_hour is not None:
        vh_str = str(valid_hour)
        monthly_vh = monthly_cache.get("by_station_monthly_by_valid_hour", {})

        # Monthly valid_hour data uses canonical lead times (snapped)
        _canonical_lts = sorted(set(list(range(6, 25, 6)) + list(range(48, 361, 24))))
        lo = bisect.bisect_left(_canonical_lts, lead_time_hours)
        snapped_lt = None
        best_dist = 13
        if lo < len(_canonical_lts):
            d = _canonical_lts[lo] - lead_time_hours
            if d <= 12:
                snapped_lt, best_dist = _canonical_lts[lo], d
        if lo > 0:
            d = lead_time_hours - _canonical_lts[lo - 1]
            if d < best_dist:
                snapped_lt = _canonical_lts[lo - 1]
        if snapped_lt is None:
            return {}

        lt_str = str(snapped_lt)
        for station_id, station_data in monthly_vh.items():
            vh_stats = station_data.get(source_model, {}).get(variable, {}).get(lt_str, {}).get(vh_str)
            if not vh_stats:
                continue
            count = vh_stats.get("count", 0)
            if count <= 0:
                continue
            station_info = stations_info.get(station_id, {})
            mae = round(vh_stats["sum_abs_errors"] / count, 2)
            raw_bias = round(vh_stats["sum_errors"] / count, 2)
            results[station_id] = {
                'mae': mae,
                'bias': _bias_corrected_metric_value(model, f"{variable}_bias", mae, raw_bias),
                'wmae': (
                    round(_stats_wmae(vh_stats, fallback_mae=(vh_stats["sum_abs_errors"] / count)), 2)
                    if _is_precip_var(variable) else None
                ),
                'count': count,
                'lat': station_info.get('lat'),
                'lon': station_info.get('lon'),
                'name': station_info.get('name', station_id),
                'state': station_info.get('state', '')
            }
    else:
        monthly = monthly_cache.get("by_station_monthly", {})
        lt_str = str(lead_time_hours)

        for station_id, station_data in monthly.items():
            model_data = station_data.get(source_model, {})
            var_data = model_data.get(variable, {})
            lt_stats = var_data.get(lt_str)
            if not lt_stats:
                continue

            station_info = stations_info.get(station_id, {})
            count = lt_stats.get("count", 0)
            if count <= 0:
                continue

            mae = lt_stats.get("sum_abs_errors", 0.0) / count
            bias = lt_stats.get("sum_errors", 0.0) / count

            results[station_id] = {
                'mae': round(mae, 2),
                'bias': _bias_corrected_metric_value(model, f"{variable}_bias", round(mae, 2), round(bias, 2)),
                'wmae': round(_stats_wmae(lt_stats, fallback_mae=mae), 2) if _is_precip_var(variable) else None,
                'count': count,
                'lat': station_info.get('lat'),
                'lon': station_info.get('lon'),
                'name': station_info.get('name', station_id),
                'state': station_info.get('state', '')
            }

    return results


def rebuild_monthly_station_cache(days_back: int = 20) -> None:
    """
    Build rolling monthly per-station stats for the last N days.
    Stored in cumulative_stats['by_station_monthly'].
    """
    db = load_asos_forecasts_db()
    stations = db.get("stations", {})
    runs = db.get("runs", {})
    observations_data = db.get("observations", {})

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_back)

    # Build sorted observation index for O(log n) lookups (same approach as
    # precompute_verification_cache — avoids O(n) scan per station per time).
    obs_cache = {}
    obs_sorted_ts = {}
    obs_sorted_keys = {}
    for station_id, station_obs in observations_data.items():
        obs_cache[station_id] = station_obs
        pairs = []
        for t in station_obs:
            try:
                pairs.append((datetime.fromisoformat(t).timestamp(), t))
            except ValueError:
                pass
        pairs.sort()
        obs_sorted_ts[station_id] = [p[0] for p in pairs]
        obs_sorted_keys[station_id] = [p[1] for p in pairs]

    precip_6hr_cache = {}  # {(station_id, time_str): value}

    def _fast_6hr_precip(station_id: str, end_time: datetime) -> Optional[float]:
        if station_id in PRECIP_EXCLUDE_STATIONS:
            return None
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys_list = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None
        end_ts = end_time.timestamp()
        start_ts = end_ts - 6 * 3600.0
        lo = bisect.bisect_right(sorted_ts, start_ts)
        hi = bisect.bisect_right(sorted_ts, end_ts)
        if lo >= hi:
            return None
        window_obs = []
        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys_list[idx])
            if obs_data is None:
                continue
            precip = obs_data.get('precip')
            if precip is None or _is_implausible_precip_value(station_id, precip):
                continue
            window_obs.append((datetime.fromtimestamp(sorted_ts[idx], tz=timezone.utc), precip))
        if not window_obs:
            return None
        window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]
        if not window_obs:
            return None
        stuck_times: set = set()
        if len(window_obs) >= 2:
            for i in range(len(window_obs) - 1):
                v0, v1 = window_obs[i][1], window_obs[i + 1][1]
                if v0 > 0 and v0 == v1:
                    j = i
                    while j < len(window_obs) and window_obs[j][1] == v0:
                        stuck_times.add(window_obs[j][0])
                        j += 1
        hourly_max: dict = {}
        for obs_time, precip in window_obs:
            if obs_time in stuck_times:
                continue
            hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_max or precip > hourly_max[hour_key]:
                hourly_max[hour_key] = precip
        if not hourly_max:
            return None
        total = sum(hourly_max.values())
        if station_id in PRECIP_MM_STATIONS:
            total /= 25.4
        return total

    precip_24hr_cache = {}  # {(station_id, time_str): value}

    def _fast_24hr_precip(station_id: str, end_time: datetime) -> Optional[float]:
        if end_time.hour != 12:
            return None
        if station_id in PRECIP_EXCLUDE_STATIONS:
            return None
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys_list = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None
        end_ts = end_time.timestamp()
        start_ts = end_ts - 24 * 3600.0
        lo = bisect.bisect_right(sorted_ts, start_ts)
        hi = bisect.bisect_right(sorted_ts, end_ts)
        if lo >= hi:
            return None
        window_obs = []
        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys_list[idx])
            if obs_data is None:
                continue
            precip = obs_data.get('precip')
            if precip is None or _is_implausible_precip_value(station_id, precip):
                continue
            window_obs.append((datetime.fromtimestamp(sorted_ts[idx], tz=timezone.utc), precip))
        if not window_obs:
            return None
        window_obs = [(t, p) for t, p in window_obs if not _is_overflow_value(p)]
        if not window_obs:
            return None
        stuck_times: set = set()
        if len(window_obs) >= 2:
            for i in range(len(window_obs) - 1):
                v0, v1 = window_obs[i][1], window_obs[i + 1][1]
                if v0 > 0 and v0 == v1:
                    j = i
                    while j < len(window_obs) and window_obs[j][1] == v0:
                        stuck_times.add(window_obs[j][0])
                        j += 1
        hourly_max: dict = {}
        for obs_time, precip in window_obs:
            if obs_time in stuck_times:
                continue
            hour_key = obs_time.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_max or precip > hourly_max[hour_key]:
                hourly_max[hour_key] = precip
        if len(hourly_max) < 20:
            return None
        total = sum(hourly_max.values())
        if station_id in PRECIP_MM_STATIONS:
            total /= 25.4
        return total

    def fast_get_obs(station_id: str, target_time: datetime) -> Optional[dict]:
        """Binary-search observation lookup returning composite obs dict."""
        sorted_ts = obs_sorted_ts.get(station_id)
        sorted_keys_list = obs_sorted_keys.get(station_id)
        station_cache = obs_cache.get(station_id)
        if not sorted_ts or station_cache is None:
            return None
        target_ts = target_time.timestamp()
        window_s = 30 * 60.0
        lo = bisect.bisect_left(sorted_ts, target_ts - window_s)
        hi = bisect.bisect_right(sorted_ts, target_ts + window_s)
        if lo >= hi:
            return None
        best_temp_val = None; best_temp_delta = window_s + 1.0
        best_mslp_val = None; best_mslp_delta = window_s + 1.0
        best_precip_val = None; best_precip_delta = window_s + 1.0
        best_dewpoint_val = None; best_dewpoint_delta = window_s + 1.0
        for idx in range(lo, hi):
            obs_data = station_cache.get(sorted_keys_list[idx])
            if obs_data is None:
                continue
            delta = abs(sorted_ts[idx] - target_ts)
            v = obs_data.get('temp')
            if v is not None and delta < best_temp_delta:
                best_temp_val = v; best_temp_delta = delta
            v = None if station_id in PRESSURE_EXCLUDE_STATIONS else obs_data.get('mslp')
            if v is not None and delta < best_mslp_delta:
                best_mslp_val = v; best_mslp_delta = delta
            v = obs_data.get('precip')
            if v is not None and delta < best_precip_delta:
                best_precip_val = v; best_precip_delta = delta
            v = obs_data.get('dewpoint')
            if v is not None and delta < best_dewpoint_delta:
                best_dewpoint_val = v; best_dewpoint_delta = delta
        if best_temp_val is None and best_mslp_val is None and best_precip_val is None and best_dewpoint_val is None:
            return None
        result = {'temp': best_temp_val, 'mslp': best_mslp_val, 'precip': best_precip_val, 'dewpoint': best_dewpoint_val}
        if target_time.hour % 6 == 0:
            cache_key = (station_id, target_time.isoformat())
            if cache_key not in precip_6hr_cache:
                precip_6hr_cache[cache_key] = _fast_6hr_precip(station_id, target_time)
            result['precip_6hr'] = precip_6hr_cache[cache_key]
        else:
            result['precip_6hr'] = None
        if target_time.hour == 12:
            p24_key = (station_id, target_time.isoformat())
            if p24_key not in precip_24hr_cache:
                precip_24hr_cache[p24_key] = _fast_24hr_precip(station_id, target_time)
            result['precip_24hr'] = precip_24hr_cache[p24_key]
        else:
            result['precip_24hr'] = None
        return result

    # Canonical lead times for snapping (must match precompute_verification_cache)
    _canonical_lts = sorted(set(list(range(6, 25, 6)) + list(range(48, 361, 24))))

    def _snap_lt(lt: int) -> Optional[int]:
        """Snap lt to nearest canonical lead time within 12h, or None."""
        lo = bisect.bisect_left(_canonical_lts, lt)
        best, best_dist = None, 13
        if lo < len(_canonical_lts):
            d = _canonical_lts[lo] - lt
            if d <= 12:
                best, best_dist = _canonical_lts[lo], d
        if lo > 0:
            d = lt - _canonical_lts[lo - 1]
            if d < best_dist:
                best, best_dist = _canonical_lts[lo - 1], d
        return best

    monthly = {}
    monthly_vh = {}  # {station_id: {model: {var: {canonical_lt_str: {vh_str: {sum_abs, sum, count}}}}}}

    var_map = {
        'temp': ('temps', 'temp'),
        'mslp': ('mslps', 'mslp'),
        'precip': ('precips', 'precip_6hr'),
        'precip_24hr': ('precips_24hr', 'precip_24hr'),
        'dewpoint': ('dewpoints', 'dewpoint'),
    }

    for run_key, run_data in runs.items():
        try:
            init_time = datetime.fromisoformat(run_key)
            if init_time.tzinfo is None:
                init_time = init_time.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        forecast_hours = run_data.get("forecast_hours", [])
        if not forecast_hours:
            continue

        # Pre-build index map once per run to avoid list.index() in inner loop
        fh_to_idx = {lt: i for i, lt in enumerate(forecast_hours)}

        # Precompute qualifying precip events across all models for fair cross-model comparison
        qualifying_precip, qualifying_precip_24hr = _qualifying_precip_sets(run_data, stations)

        for lead_time_hours in forecast_hours:
            valid_time = init_time + timedelta(hours=lead_time_hours)
            if valid_time >= now or valid_time < cutoff:
                continue

            lt_str = str(lead_time_hours)
            fcst_idx = fh_to_idx[lead_time_hours]

            # Cache obs per station for this valid_time so each station is looked up
            # once regardless of how many models have data for it (was 4x redundant).
            obs_for_time: dict = {}

            for model in ['gfs', 'aifs', 'ifs', 'nws']:
                model_data = run_data.get(model)
                if not model_data:
                    continue

                for station_id, fcst_data in model_data.items():
                    if station_id not in stations:
                        continue

                    # Look up obs once per (station, valid_time) — shared across all models and vars
                    if station_id not in obs_for_time:
                        obs_for_time[station_id] = fast_get_obs(station_id, valid_time)
                    obs = obs_for_time[station_id]
                    if obs is None:
                        continue

                    for var, (fcst_key, obs_key) in var_map.items():
                        fcst_values = fcst_data.get(fcst_key, [])
                        if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                            continue
                        obs_val = obs.get(obs_key)
                        if obs_val is None:
                            continue

                        fcst_val = fcst_values[fcst_idx]
                        if var == 'precip' and (station_id, fcst_idx) not in qualifying_precip and obs_val < 0.01:
                            continue
                        if var == 'precip_24hr' and (station_id, fcst_idx) not in qualifying_precip_24hr and obs_val < 0.01:
                            continue
                        error = fcst_val - obs_val

                        monthly.setdefault(station_id, {}).setdefault(model, {}).setdefault(var, {}).setdefault(
                            lt_str, {
                                'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0,
                                'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0
                            }
                        )
                        stats = monthly[station_id][model][var][lt_str]
                        stats['sum_abs_errors'] += abs(error)
                        stats['sum_errors'] += error
                        stats['count'] += 1
                        if _is_precip_var(var):
                            weight = _precip_weight(obs_val)
                            stats['sum_weighted_abs_errors'] += abs(error) * weight
                            stats['sum_weights'] += weight

                        # Also accumulate by valid hour, snapping to nearest canonical lt
                        snapped_lt = _snap_lt(lead_time_hours)
                        if snapped_lt is not None:
                            vh_str = str(valid_time.hour)
                            snapped_lt_str = str(snapped_lt)
                            monthly_vh.setdefault(station_id, {}).setdefault(model, {}).setdefault(var, {}).setdefault(
                                snapped_lt_str, {}).setdefault(vh_str, {
                                    'sum_abs_errors': 0.0, 'sum_errors': 0.0, 'count': 0,
                                    'sum_weighted_abs_errors': 0.0, 'sum_weights': 0.0
                                })
                            vh_stats = monthly_vh[station_id][model][var][snapped_lt_str][vh_str]
                            vh_stats['sum_abs_errors'] += abs(error)
                            vh_stats['sum_errors'] += error
                            vh_stats['count'] += 1
                            if _is_precip_var(var):
                                weight = _precip_weight(obs_val)
                                vh_stats['sum_weighted_abs_errors'] += abs(error) * weight
                                vh_stats['sum_weights'] += weight

    save_monthly_stats_cache({
        "by_station_monthly": monthly,
        "by_station_monthly_by_valid_hour": monthly_vh,
        "monthly_generated_at": now.isoformat(),
    })


def get_mean_verification_from_cache(model: str, valid_hour: Optional[int] = None) -> dict:
    """
    Get mean verification from cache for a specific model.

    Falls back to computing on-the-fly if cache doesn't exist.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
        valid_hour: Optional UTC hour filter (0, 6, 12, 18). None = all hours.

    Returns:
        Dict with mean verification data (same format as get_mean_verification_by_lead_time)
    """
    cache = load_verification_cache()

    if cache is None:
        logger.warning("Cache not available, computing verification on-the-fly")
        return get_mean_verification_by_lead_time(model)

    source_model = _source_model_for_verification(model)
    if (model or "").lower() == "kenny":
        return get_mean_verification_by_lead_time(model, valid_hour=valid_hour)
    lead_times = cache.get("lead_times", [])
    nws_max_lead = 168 if source_model == 'nws' else None

    result = {
        "lead_times": lead_times,
        "temp_mae": [],
        "temp_bias": [],
        "mslp_mae": [],
        "mslp_bias": [],
        "precip_mae": [],
        "precip_bias": [],
        "precip_wmae": [],
        "precip_24hr_mae": [],
        "precip_24hr_bias": [],
        "precip_24hr_wmae": [],
        "dewpoint_mae": [],
        "dewpoint_bias": [],
    }

    if valid_hour is not None:
        vh_str = str(valid_hour)
        model_vh_data = cache.get("by_lead_time_by_valid_hour", {}).get(source_model, {})
        for lt in lead_times:
            lt_str = str(lt)
            if nws_max_lead is not None and lt > nws_max_lead:
                result["temp_mae"].append(None); result["temp_bias"].append(None)
                result["mslp_mae"].append(None); result["mslp_bias"].append(None)
                result["precip_mae"].append(None); result["precip_bias"].append(None); result["precip_wmae"].append(None)
                result["precip_24hr_mae"].append(None); result["precip_24hr_bias"].append(None); result["precip_24hr_wmae"].append(None)
                result["dewpoint_mae"].append(None); result["dewpoint_bias"].append(None)
                continue
            lt_data = model_vh_data.get(lt_str, {}).get(vh_str, {})
            temp_mae = lt_data.get("temp", {}).get("mae")
            mslp_mae = lt_data.get("mslp", {}).get("mae")
            precip_mae = lt_data.get("precip", {}).get("mae")
            p24_mae = lt_data.get("precip_24hr", {}).get("mae")
            dew_mae = lt_data.get("dewpoint", {}).get("mae")

            result["temp_mae"].append(temp_mae)
            result["temp_bias"].append(_bias_corrected_metric_value(model, "temp_bias", temp_mae, lt_data.get("temp", {}).get("bias")))
            result["mslp_mae"].append(mslp_mae)
            result["mslp_bias"].append(_bias_corrected_metric_value(model, "mslp_bias", mslp_mae, lt_data.get("mslp", {}).get("bias")))
            result["precip_mae"].append(precip_mae)
            result["precip_bias"].append(_bias_corrected_metric_value(model, "precip_bias", precip_mae, lt_data.get("precip", {}).get("bias")))
            result["precip_wmae"].append(lt_data.get("precip", {}).get("wmae", precip_mae))
            result["precip_24hr_mae"].append(p24_mae)
            result["precip_24hr_bias"].append(_bias_corrected_metric_value(model, "precip_24hr_bias", p24_mae, lt_data.get("precip_24hr", {}).get("bias")))
            result["precip_24hr_wmae"].append(lt_data.get("precip_24hr", {}).get("wmae", p24_mae))
            result["dewpoint_mae"].append(dew_mae)
            result["dewpoint_bias"].append(_bias_corrected_metric_value(model, "dewpoint_bias", dew_mae, lt_data.get("dewpoint", {}).get("bias")))
        return result

    # All hours — use existing by_lead_time data
    model_data = cache.get("by_lead_time", {}).get(source_model, {})
    for lt in lead_times:
        lt_str = str(lt)
        lt_data = model_data.get(lt_str, {})

        if nws_max_lead is not None and lt > nws_max_lead:
            result["temp_mae"].append(None); result["temp_bias"].append(None)
            result["mslp_mae"].append(None); result["mslp_bias"].append(None)
            result["precip_mae"].append(None); result["precip_bias"].append(None); result["precip_wmae"].append(None)
            result["precip_24hr_mae"].append(None); result["precip_24hr_bias"].append(None); result["precip_24hr_wmae"].append(None)
            result["dewpoint_mae"].append(None); result["dewpoint_bias"].append(None)
            continue

        temp_mae = lt_data.get("temp", {}).get("mae")
        mslp_mae = lt_data.get("mslp", {}).get("mae")
        precip_mae = lt_data.get("precip", {}).get("mae")
        p24_mae = lt_data.get("precip_24hr", {}).get("mae")
        dew_mae = lt_data.get("dewpoint", {}).get("mae")

        result["temp_mae"].append(temp_mae)
        result["temp_bias"].append(_bias_corrected_metric_value(model, "temp_bias", temp_mae, lt_data.get("temp", {}).get("bias")))
        result["mslp_mae"].append(mslp_mae)
        result["mslp_bias"].append(_bias_corrected_metric_value(model, "mslp_bias", mslp_mae, lt_data.get("mslp", {}).get("bias")))
        result["precip_mae"].append(precip_mae)
        result["precip_bias"].append(_bias_corrected_metric_value(model, "precip_bias", precip_mae, lt_data.get("precip", {}).get("bias")))
        result["precip_wmae"].append(lt_data.get("precip", {}).get("wmae", precip_mae))
        result["precip_24hr_mae"].append(p24_mae)
        result["precip_24hr_bias"].append(_bias_corrected_metric_value(model, "precip_24hr_bias", p24_mae, lt_data.get("precip_24hr", {}).get("bias")))
        result["precip_24hr_wmae"].append(lt_data.get("precip_24hr", {}).get("wmae", p24_mae))
        result["dewpoint_mae"].append(dew_mae)
        result["dewpoint_bias"].append(_bias_corrected_metric_value(model, "dewpoint_bias", dew_mae, lt_data.get("dewpoint", {}).get("bias")))

    return result


def get_mean_verification_from_monthly_cache(model: str, valid_hour: Optional[int] = None) -> dict:
    """
    Get mean verification for the rolling monthly window from the monthly cache.

    Args:
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
        valid_hour: Optional UTC hour filter (0, 6, 12, 18). None = all hours.
    """
    source_model = _source_model_for_verification(model)
    if (model or "").lower() == "kenny":
        return get_mean_verification_by_lead_time(model, valid_hour=valid_hour, days_back=MONTHLY_WINDOW_DAYS)
    monthly_cache = load_monthly_stats_cache()
    nws_max_lead = 168 if source_model == 'nws' else None

    canonical_lead_times = set(list(range(6, 25, 6)) + list(range(48, 361, 24)))

    if valid_hour is not None:
        monthly_vh_all = monthly_cache.get("by_station_monthly_by_valid_hour", {})
        vh_str = str(valid_hour)

        # Collect lead times that have data for this model and valid hour
        lead_times = set()
        for station_data in monthly_vh_all.values():
            model_data = station_data.get(source_model, {})
            for var in ["temp", "mslp", "precip", "precip_24hr", "dewpoint"]:
                for lt_str, vh_data in model_data.get(var, {}).items():
                    if vh_str in vh_data:
                        lead_times.add(int(lt_str))

        lead_times &= canonical_lead_times
        if nws_max_lead is not None:
            lead_times = {lt for lt in lead_times if lt <= nws_max_lead}
        lead_times = sorted(lead_times)

        def _init_acc():
            return {"sum_abs": 0.0, "sum": 0.0, "count": 0, "sum_weighted_abs": 0.0, "sum_weights": 0.0}

        result = {
            "lead_times": lead_times,
            "temp_mae": [], "temp_bias": [],
            "mslp_mae": [], "mslp_bias": [],
            "precip_mae": [], "precip_bias": [],
            "precip_wmae": [],
            "precip_24hr_mae": [], "precip_24hr_bias": [],
            "precip_24hr_wmae": [],
            "dewpoint_mae": [], "dewpoint_bias": [],
        }

        for lt in lead_times:
            lt_str = str(lt)
            acc = {"temp": _init_acc(), "mslp": _init_acc(), "precip": _init_acc(), "precip_24hr": _init_acc(), "dewpoint": _init_acc()}

            for station_data in monthly_vh_all.values():
                model_data = station_data.get(source_model, {})
                for var in ["temp", "mslp", "precip", "precip_24hr", "dewpoint"]:
                    stats = model_data.get(var, {}).get(lt_str, {}).get(vh_str)
                    if not stats or stats.get("count", 0) <= 0:
                        continue
                    acc[var]["sum_abs"] += stats.get("sum_abs_errors", 0.0)
                    acc[var]["sum"] += stats.get("sum_errors", 0.0)
                    acc[var]["count"] += stats.get("count", 0)
                    if _is_precip_var(var):
                        acc[var]["sum_weighted_abs"] += stats.get("sum_weighted_abs_errors", 0.0)
                        acc[var]["sum_weights"] += stats.get("sum_weights", 0.0)

            for var, mae_key, bias_key, wmae_key in [
                ("temp", "temp_mae", "temp_bias", None),
                ("mslp", "mslp_mae", "mslp_bias", None),
                ("precip", "precip_mae", "precip_bias", "precip_wmae"),
                ("precip_24hr", "precip_24hr_mae", "precip_24hr_bias", "precip_24hr_wmae"),
                ("dewpoint", "dewpoint_mae", "dewpoint_bias", None),
            ]:
                if acc[var]["count"] > 0:
                    mae = acc[var]["sum_abs"] / acc[var]["count"]
                    rounded_mae = round(mae, 2)
                    raw_bias = round(acc[var]["sum"] / acc[var]["count"], 2)
                    result[mae_key].append(rounded_mae)
                    result[bias_key].append(_bias_corrected_metric_value(model, bias_key, rounded_mae, raw_bias))
                    if wmae_key is not None:
                        wmae = (acc[var]["sum_weighted_abs"] / acc[var]["sum_weights"]) if acc[var]["sum_weights"] > 0 else mae
                        result[wmae_key].append(round(wmae, 2))
                else:
                    result[mae_key].append(None)
                    result[bias_key].append(None)
                    if wmae_key is not None:
                        result[wmae_key].append(None)

        return result

    # All hours — use existing by_station_monthly data
    monthly = monthly_cache.get("by_station_monthly", {})

    # Collect lead times available for this model, restricted to canonical set
    lead_times = set()
    for station_data in monthly.values():
        model_data = station_data.get(source_model, {})
        for var in ["temp", "mslp", "precip", "precip_24hr", "dewpoint"]:
            lead_times.update({int(k) for k in model_data.get(var, {}).keys()})

    lead_times &= canonical_lead_times
    if nws_max_lead is not None:
        lead_times = {lt for lt in lead_times if lt <= nws_max_lead}
    lead_times = sorted(lead_times)

    def _init_acc():
        return {"sum_abs": 0.0, "sum": 0.0, "count": 0, "sum_weighted_abs": 0.0, "sum_weights": 0.0}

    result = {
        "lead_times": lead_times,
        "temp_mae": [],
        "temp_bias": [],
        "mslp_mae": [],
        "mslp_bias": [],
        "precip_mae": [],
        "precip_bias": [],
        "precip_wmae": [],
        "precip_24hr_mae": [],
        "precip_24hr_bias": [],
        "precip_24hr_wmae": [],
        "dewpoint_mae": [],
        "dewpoint_bias": [],
    }

    for lt in lead_times:
        lt_str = str(lt)
        acc = {
            "temp": _init_acc(),
            "mslp": _init_acc(),
            "precip": _init_acc(),
            "precip_24hr": _init_acc(),
            "dewpoint": _init_acc(),
        }

        for station_data in monthly.values():
            model_data = station_data.get(source_model, {})
            for var in ["temp", "mslp", "precip", "precip_24hr", "dewpoint"]:
                stats = model_data.get(var, {}).get(lt_str)
                if not stats or stats.get("count", 0) <= 0:
                    continue
                acc[var]["sum_abs"] += stats.get("sum_abs_errors", 0.0)
                acc[var]["sum"] += stats.get("sum_errors", 0.0)
                acc[var]["count"] += stats.get("count", 0)
                if _is_precip_var(var):
                    acc[var]["sum_weighted_abs"] += stats.get("sum_weighted_abs_errors", 0.0)
                    acc[var]["sum_weights"] += stats.get("sum_weights", 0.0)

        for var, mae_key, bias_key, wmae_key in [
            ("temp", "temp_mae", "temp_bias", None),
            ("mslp", "mslp_mae", "mslp_bias", None),
            ("precip", "precip_mae", "precip_bias", "precip_wmae"),
            ("precip_24hr", "precip_24hr_mae", "precip_24hr_bias", "precip_24hr_wmae"),
            ("dewpoint", "dewpoint_mae", "dewpoint_bias", None),
        ]:
            if acc[var]["count"] > 0:
                mae = acc[var]["sum_abs"] / acc[var]["count"]
                rounded_mae = round(mae, 2)
                raw_bias = round(acc[var]["sum"] / acc[var]["count"], 2)
                result[mae_key].append(rounded_mae)
                result[bias_key].append(_bias_corrected_metric_value(model, bias_key, rounded_mae, raw_bias))
                if wmae_key is not None:
                    wmae = (acc[var]["sum_weighted_abs"] / acc[var]["sum_weights"]) if acc[var]["sum_weights"] > 0 else mae
                    result[wmae_key].append(round(wmae, 2))
            else:
                result[mae_key].append(None)
                result[bias_key].append(None)
                if wmae_key is not None:
                    result[wmae_key].append(None)

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
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
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

    source_model = _source_model_for_verification(model)
    if _is_kenny_bias_corrected_var(model, variable):
        return get_verification_time_series(model, variable, lead_time_hours, days_back)

    # Extract from cache
    time_series = cache.get("time_series", {})
    model_data = time_series.get(source_model, {})
    var_data = model_data.get(variable, {})
    lt_data = var_data.get(str(lead_time_hours))

    if not lt_data:
        # No data for this combination, return empty result
        return {
            "dates": [],
            "mae": [],
            "wmae": [],
            "bias": [],
            "counts": []
        }

    # Get all dates and slice to requested days_back
    all_dates = lt_data.get('dates', [])
    all_mae = lt_data.get('mae', [])
    all_wmae = lt_data.get('wmae', [])
    all_bias = lt_data.get('bias', [])
    all_counts = lt_data.get('counts', [])

    if not all_dates:
        return {
            "dates": [],
            "mae": [],
            "wmae": [],
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

        if _is_precip_var(variable):
            if not all_wmae:
                all_wmae = list(all_mae)
        else:
            all_wmae = [None] * len(all_dates)

        result = {
            "dates": all_dates[start_idx:],
            "mae": all_mae[start_idx:],
            "wmae": all_wmae[start_idx:],
            "bias": all_bias[start_idx:],
            "counts": all_counts[start_idx:]
        }
        return result
    except Exception as e:
        logger.error(f"Error filtering time series data: {e}")
        result = {
            "dates": all_dates,
            "mae": all_mae,
            "wmae": all_wmae if all_wmae else (list(all_mae) if _is_precip_var(variable) else [None] * len(all_dates)),
            "bias": all_bias,
            "counts": all_counts
        }
        return result


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
        model: Model name ('gfs', 'aifs', 'ifs', 'nws')
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
            'precip': ('precips', 'precip_6hr'),
            'precip_24hr': ('precips_24hr', 'precip_24hr'),
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


def _calculate_asos_mean_error_stats(
    db: dict,
    run_time_str: str,
    model: str,
    variable: str,
    lead_time_hours: int
) -> Optional[dict]:
    """
    Calculate mean MAE and bias across all ASOS stations for a specific run.

    Returns dict with keys: mae, bias, count (stations), or None if insufficient data.
    """
    try:
        run_data = db.get("runs", {}).get(run_time_str)
        if not run_data:
            return None

        forecast_hours = run_data.get("forecast_hours", [])
        if lead_time_hours not in forecast_hours:
            return None

        fcst_idx = forecast_hours.index(lead_time_hours)

        var_map = {
            'temp': ('temps', 'temp'),
            'mslp': ('mslps', 'mslp'),
            'precip': ('precips', 'precip_6hr'),
            'precip_24hr': ('precips_24hr', 'precip_24hr'),
        }

        if variable not in var_map:
            return None

        fcst_key, obs_key = var_map[variable]

        init_time = datetime.fromisoformat(run_time_str)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)

        valid_time = init_time + timedelta(hours=lead_time_hours)
        now = datetime.now(timezone.utc)
        if valid_time >= now:
            return None

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

            obs = get_stored_observation(db, station_id, valid_time)
            if obs is None or obs.get(obs_key) is None:
                continue

            obs_val = obs[obs_key]

            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue

            errors.append(fcst_val - obs_val)

        if len(errors) >= 10:
            mae = sum(abs(e) for e in errors) / len(errors)
            bias = sum(errors) / len(errors)
            return {"mae": mae, "bias": bias, "count": len(errors)}
        return None

    except Exception as e:
        logger.error(f"Error calculating ASOS mean error stats: {e}")
        return None


def _calculate_asos_station_errors(
    db: dict,
    run_time_str: str,
    model: str,
    variable: str,
    lead_time_hours: int
) -> dict:
    """
    Return per-station forecast errors (fcst - obs) for a run/model/variable/lead time.
    """
    errors = {}
    try:
        run_data = db.get("runs", {}).get(run_time_str)
        if not run_data:
            return errors

        forecast_hours = run_data.get("forecast_hours", [])
        if lead_time_hours not in forecast_hours:
            return errors

        fcst_idx = forecast_hours.index(lead_time_hours)

        var_map = {
            'temp': ('temps', 'temp'),
            'mslp': ('mslps', 'mslp'),
            'precip': ('precips', 'precip_6hr'),
            'precip_24hr': ('precips_24hr', 'precip_24hr'),
        }
        if variable not in var_map:
            return errors

        fcst_key, obs_key = var_map[variable]

        init_time = datetime.fromisoformat(run_time_str)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)

        valid_time = init_time + timedelta(hours=lead_time_hours)
        now = datetime.now(timezone.utc)
        if valid_time >= now:
            return errors

        stations = db.get("stations", {})
        model_data = run_data.get(model.lower())
        if not model_data:
            return errors

        for station_id, fcst_data in model_data.items():
            if station_id not in stations:
                continue

            fcst_values = fcst_data.get(fcst_key, [])
            if fcst_idx >= len(fcst_values) or fcst_values[fcst_idx] is None:
                continue

            fcst_val = fcst_values[fcst_idx]
            obs = get_stored_observation(db, station_id, valid_time)
            if obs is None or obs.get(obs_key) is None:
                continue

            obs_val = obs[obs_key]
            if variable == 'precip' and not should_include_precip(fcst_val, obs_val):
                continue

            errors[station_id] = fcst_val - obs_val

        return errors

    except Exception as e:
        logger.error(f"Error calculating ASOS station errors: {e}")
        return errors


# ---------------------------------------------------------------------------
# 5-minute METAR pressure archive (dedicated, separate from verification data)
# ---------------------------------------------------------------------------

def load_asos_metar_pressure_cache() -> dict | None:
    """Load the 5-min METAR pressure cache with in-memory mtime caching."""
    global _asos_metar_pressure_cache, _asos_metar_pressure_mtime
    if not ASOS_METAR_PRESSURE_FILE.exists():
        return None
    try:
        mtime = ASOS_METAR_PRESSURE_FILE.stat().st_mtime
        if _asos_metar_pressure_cache is not None and _asos_metar_pressure_mtime == mtime:
            return _asos_metar_pressure_cache
        with open(ASOS_METAR_PRESSURE_FILE) as f:
            data = json.load(f)
        _asos_metar_pressure_cache = data
        _asos_metar_pressure_mtime = mtime
        return data
    except Exception as e:
        logger.warning(f"Error loading asos_metar_pressure.json: {e}")
        return None


def sync_asos_metar_pressure(lookback_hours: int = 28) -> dict:
    """
    Fetch recent 5-min METAR altimeter data from IEM for all ASOS stations
    and store in the dedicated asos_metar_pressure.json file.

    Completely independent of asos_forecasts.json — station metadata is
    embedded in the pressure file so the perturbation rebuild needs nothing
    from the verification database.
    """
    global _asos_metar_pressure_cache, _asos_metar_pressure_mtime

    now = datetime.now(timezone.utc)
    fetch_start = now - timedelta(hours=lookback_hours)
    fetch_end   = now

    # Station list: pull from verification db if available (already has lat/lon),
    # otherwise fall back to a fresh IEM fetch.
    db = load_asos_forecasts_db()
    stations_meta = db.get("stations", {})
    if not stations_meta:
        try:
            stations_meta = get_stations_dict()
        except Exception as e:
            logger.error(f"sync_asos_metar_pressure: could not get station list: {e}")
            return {"status": "no_stations", "stations_updated": 0}

    station_ids = list(stations_meta.keys())
    logger.info(
        f"Fetching METAR pressure (alti only) for {len(station_ids)} stations, "
        f"{fetch_start.strftime('%Y-%m-%dT%H:%MZ')} → {fetch_end.strftime('%Y-%m-%dT%H:%MZ')}"
    )

    chunk_size = 50
    all_obs: dict[str, list] = {}
    for i in range(0, len(station_ids), chunk_size):
        chunk = station_ids[i:i + chunk_size]
        try:
            chunk_obs = fetch_observations(chunk, fetch_start, fetch_end, variables=['alti'])
            all_obs.update(chunk_obs)
        except Exception as e:
            logger.error(f"METAR pressure chunk {i // chunk_size} failed: {e}")

    # Load existing cache so we can merge & trim
    existing = load_asos_metar_pressure_cache() or {}
    stations_data: dict = existing.get("stations", {})
    cutoff = now - timedelta(hours=lookback_hours)
    updated_count = 0

    for sid, obs_list in all_obs.items():
        meta = stations_meta.get(sid, {})
        # Existing obs keyed by "YYYY-MM-DD HH:MM" timestamp
        existing_obs: dict[str, float] = {
            r[0]: r[1] for r in stations_data.get(sid, {}).get("obs", [])
        }

        for obs in obs_list:
            mslp = obs.get("mslp")
            vt   = obs.get("valid_time")
            if mslp is None or vt is None:
                continue
            # valid_time is ISO string from fetch_observations, e.g. "2026-02-17T10:00:00+00:00"
            ts_str = vt[:16].replace("T", " ")  # → "2026-02-17 10:00"
            existing_obs[ts_str] = mslp

        # Trim to lookback window
        trimmed: list = []
        for ts_str, pres_mb in existing_obs.items():
            try:
                ts_norm = ts_str.replace(" ", "T") + "+00:00"
                dt = datetime.fromisoformat(ts_norm)
                if dt >= cutoff:
                    trimmed.append([ts_str, pres_mb])
            except Exception:
                continue
        trimmed.sort(key=lambda x: x[0])

        if trimmed:
            stations_data[sid] = {
                "lat":   meta.get("lat"),
                "lon":   meta.get("lon"),
                "name":  meta.get("name", sid),
                "state": meta.get("state", ""),
                "obs":   trimmed,
            }
            updated_count += 1

    cache_data = {
        "updated":       now.isoformat(),
        "lookback_hours": lookback_hours,
        "stations":      stations_data,
    }
    with open(ASOS_METAR_PRESSURE_FILE, "w") as f:
        json.dump(cache_data, f)

    # Invalidate in-memory cache
    _asos_metar_pressure_cache = None
    _asos_metar_pressure_mtime = None

    logger.info(f"METAR pressure sync complete: {updated_count} stations updated")
    return {
        "status": "success",
        "stations_updated": updated_count,
    }
