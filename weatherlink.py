"""
WeatherLink Observation Data Reader

Reads observation data from Davis WeatherLink CSV files stored at
~/Documents/Townhome_Weather/Weather_Data/

CSV Format:
- 5-minute resolution
- Monthly files: {year}/{MMYYYY}.csv
- Header on row 6, data starts on row 7
- Column 0: Date & Time (MM/DD/YY HH:MM) in LOCAL EASTERN TIME
- Column 4: Barometer (mb)
- Column 5: Temp (°F)

Note: WeatherLink stores times in local Eastern time, but model forecasts
are in UTC. This module handles the conversion.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Path to WeatherLink data
WEATHERLINK_PATH = Path.home() / "Documents" / "Townhome_Weather" / "Weather_Data"

# CSV column indices
COL_DATETIME = 0
COL_BAROMETER = 4
COL_TEMP = 5

# Timezone offset for Eastern Time
# Note: This is simplified - doesn't handle DST transitions perfectly
# EST = UTC-5, EDT = UTC-4
def get_eastern_offset(dt: datetime) -> timedelta:
    """
    Get the UTC offset for Eastern time on a given date.
    DST runs from 2nd Sunday in March to 1st Sunday in November.
    """
    year = dt.year

    # Find 2nd Sunday in March (DST starts at 2 AM)
    march_1 = datetime(year, 3, 1)
    days_until_sunday = (6 - march_1.weekday()) % 7
    dst_start = datetime(year, 3, 8 + days_until_sunday, 2, 0)

    # Find 1st Sunday in November (DST ends at 2 AM)
    nov_1 = datetime(year, 11, 1)
    days_until_sunday = (6 - nov_1.weekday()) % 7
    dst_end = datetime(year, 11, 1 + days_until_sunday, 2, 0)

    if dst_start <= dt.replace(tzinfo=None) < dst_end:
        return timedelta(hours=-4)  # EDT
    else:
        return timedelta(hours=-5)  # EST


def utc_to_eastern(utc_dt: datetime) -> datetime:
    """Convert a UTC datetime to Eastern local time."""
    offset = get_eastern_offset(utc_dt)
    return utc_dt + offset


def eastern_to_utc(eastern_dt: datetime) -> datetime:
    """Convert an Eastern local datetime to UTC."""
    offset = get_eastern_offset(eastern_dt)
    return eastern_dt - offset


def parse_datetime(date_str: str) -> datetime:
    """Parse WeatherLink datetime format (MM/DD/YY HH:MM). Returns local Eastern time."""
    return datetime.strptime(date_str.strip('"'), "%m/%d/%y %H:%M")


def get_csv_path(dt: datetime) -> Path:
    """Get the CSV file path for a given datetime."""
    month_str = f"{dt.month:02d}{dt.year}"
    return WEATHERLINK_PATH / str(dt.year) / f"{month_str}.csv"


def read_csv_data(start_date: datetime, end_date: datetime) -> list[dict]:
    """
    Read observation data from CSV files in date range.

    Args:
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of dicts with keys: time, temp, mslp
    """
    observations = []

    # Determine which months we need to read
    current = datetime(start_date.year, start_date.month, 1)
    end_month = datetime(end_date.year, end_date.month, 1)

    while current <= end_month:
        csv_path = get_csv_path(current)

        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='mac_roman') as f:
                    reader = csv.reader(f)

                    # Skip header rows (first 6 lines)
                    for _ in range(6):
                        next(reader, None)

                    for row in reader:
                        if not row or len(row) < 6:
                            continue

                        try:
                            obs_time = parse_datetime(row[COL_DATETIME])

                            # Filter to requested date range
                            if start_date <= obs_time <= end_date:
                                # Parse temp and pressure, handling '--' as None
                                temp_str = row[COL_TEMP].strip('"')
                                bar_str = row[COL_BAROMETER].strip('"')

                                temp = float(temp_str) if temp_str != '--' else None
                                mslp = float(bar_str) if bar_str != '--' else None

                                observations.append({
                                    'time': obs_time,
                                    'temp': temp,
                                    'mslp': mslp
                                })
                        except (ValueError, IndexError) as e:
                            # Skip malformed rows
                            continue

            except Exception as e:
                logger.warning(f"Error reading {csv_path}: {e}")

        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return observations


def find_nearest_observation(observations: list[dict], target_time: datetime, max_delta_minutes: int = 5) -> dict | None:
    """
    Find the observation nearest to target_time within max_delta_minutes.

    Args:
        observations: List of observation dicts (must be sorted by time)
        target_time: The time to match
        max_delta_minutes: Maximum allowed time difference

    Returns:
        Observation dict or None if no match within threshold
    """
    if not observations:
        return None

    max_delta = timedelta(minutes=max_delta_minutes)
    best_match = None
    best_delta = timedelta.max

    for obs in observations:
        delta = abs(obs['time'] - target_time)
        if delta < best_delta and delta <= max_delta:
            best_delta = delta
            best_match = obs
        # Early exit if we've passed the target time significantly
        elif obs['time'] > target_time + max_delta:
            break

    return best_match


def get_observations(times: list[datetime | None], times_are_utc: bool = True) -> dict:
    """
    Get observations matching forecast valid times.

    Args:
        times: List of forecast valid times (as datetime objects, may contain None)
        times_are_utc: If True, input times are UTC and will be converted to Eastern
                       for matching against local-time observations

    Returns:
        Dict with keys: temps, mslps, times (lists aligned with input times)
        Values are None where no observation is available
        The 'times' in the result are the matched observation times in ISO format (local Eastern)
    """
    if not times:
        return {"temps": [], "mslps": [], "times": []}

    # Convert UTC times to Eastern for matching against local-time observations
    # Handle None values in the list
    if times_are_utc:
        eastern_times = [utc_to_eastern(t) if t is not None else None for t in times]
    else:
        eastern_times = times

    # Filter out None values for determining date range
    valid_eastern_times = [t for t in eastern_times if t is not None]

    if not valid_eastern_times:
        return {"temps": [None] * len(times), "mslps": [None] * len(times), "times": [None] * len(times)}

    # Determine date range needed (in Eastern time, since that's what CSV uses)
    min_time = min(valid_eastern_times)
    max_time = max(valid_eastern_times)

    # Add buffer for matching
    start_date = min_time - timedelta(minutes=10)
    end_date = max_time + timedelta(minutes=10)

    # Read all observations in range (returns Eastern local times)
    observations = read_csv_data(start_date, end_date)

    # Sort observations by time for efficient searching
    observations.sort(key=lambda x: x['time'])

    # Match each forecast time to nearest observation
    temps = []
    mslps = []
    matched_times = []

    for target_time in eastern_times:
        if target_time is None:
            temps.append(None)
            mslps.append(None)
            matched_times.append(None)
            continue

        obs = find_nearest_observation(observations, target_time)

        if obs:
            temps.append(obs['temp'])
            mslps.append(obs['mslp'])
            # Return the matched time in ISO format (Eastern local time)
            matched_times.append(obs['time'].isoformat())
        else:
            temps.append(None)
            mslps.append(None)
            matched_times.append(None)

    return {
        "temps": temps,
        "mslps": mslps,
        "times": matched_times
    }


def get_observations_for_forecast_times(forecast_times: list[str]) -> dict:
    """
    Get observations for a list of forecast valid times (ISO format strings).

    This is the main entry point for app.py integration.

    Args:
        forecast_times: List of ISO format datetime strings (assumed to be UTC)

    Returns:
        Dict with keys: temps, mslps, times
        The 'times' are the matched observation times in Eastern local time
    """
    # Parse ISO format times (these are UTC from the model forecasts)
    times = []
    for time_str in forecast_times:
        try:
            # Handle both with and without timezone info
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            # Convert to naive datetime (still representing UTC)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            times.append(dt)
        except ValueError as e:
            logger.warning(f"Could not parse time {time_str}: {e}")
            times.append(None)

    # Filter out None values for the observation lookup
    valid_times = [t for t in times if t is not None]

    if not valid_times:
        return {"temps": [None] * len(times), "mslps": [None] * len(times), "times": [None] * len(times)}

    # Times from forecast are UTC, observations are in Eastern local time
    return get_observations(times, times_are_utc=True)


def fetch_missing_data(silent: bool = True) -> int:
    """
    Fetch any missing WeatherLink data by calling the fetch script.

    Returns:
        Number of new records fetched
    """
    import sys

    # Add the WeatherLink data path to sys.path temporarily
    fetch_script = WEATHERLINK_PATH / "fetch_weather_data.py"

    if not fetch_script.exists():
        logger.warning(f"fetch_weather_data.py not found at {fetch_script}")
        return 0

    try:
        # Import the fetch module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("fetch_weather_data", fetch_script)
        fetch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch_module)

        # Call the fetch_missing_data function
        return fetch_module.fetch_missing_data(silent=silent)
    except Exception as e:
        logger.warning(f"Error fetching missing WeatherLink data: {e}")
        return 0
