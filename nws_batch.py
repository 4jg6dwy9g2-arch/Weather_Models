#!/usr/bin/env python3
"""
Batch NWS forecast fetching for ASOS stations with rate limiting and caching.
Optimizes API usage by caching grid points and grouping stations by grid.
"""

import os
import json
import asyncio
import aiohttp
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Paths
NWS_GRID_CACHE_PATH = Path(__file__).resolve().parent / "data" / "nws_grid_cache.json"

# NWS API configuration
NWS_API = "https://api.weather.gov"
HEADERS = {
    "User-Agent": "(Weather Models App, contact@example.com)",
    "Accept": "application/geo+json"
}

# Rate limiting configuration
REQUESTS_PER_SECOND = 10  # Balanced rate limit (safe with User-Agent header)
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND


class GridPointCache:
    """Manages persistent cache of station -> NWS grid point mappings."""

    def __init__(self):
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load grid point cache from disk."""
        if NWS_GRID_CACHE_PATH.exists():
            try:
                with open(NWS_GRID_CACHE_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading NWS grid cache: {e}")
        return {}

    def _save_cache(self):
        """Save grid point cache to disk."""
        try:
            with open(NWS_GRID_CACHE_PATH, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving NWS grid cache: {e}")

    def get_grid_point(self, station_id: str) -> Optional[Dict]:
        """Get cached grid point for a station."""
        return self.cache.get(station_id)

    def set_grid_point(self, station_id: str, grid_id: str, grid_x: int, grid_y: int):
        """Cache grid point for a station."""
        self.cache[station_id] = {
            "grid_id": grid_id,
            "grid_x": grid_x,
            "grid_y": grid_y
        }
        self._save_cache()

    def get_all_cached_stations(self) -> List[str]:
        """Get list of all station IDs with cached grid points."""
        return list(self.cache.keys())


async def fetch_grid_point(session: aiohttp.ClientSession, lat: float, lon: float,
                          station_id: str, cache: GridPointCache) -> Optional[Tuple[str, int, int]]:
    """
    Fetch NWS grid point for a location.
    Returns (grid_id, grid_x, grid_y) or None if failed.
    """
    # Check cache first
    cached = cache.get_grid_point(station_id)
    if cached:
        return (cached["grid_id"], cached["grid_x"], cached["grid_y"])

    # Fetch from API
    url = f"{NWS_API}/points/{lat:.4f},{lon:.4f}"
    try:
        await asyncio.sleep(REQUEST_DELAY)  # Rate limiting
        async with session.get(url, headers=HEADERS, timeout=10) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch grid point for {station_id}: HTTP {response.status}")
                return None

            data = await response.json()
            props = data["properties"]
            grid_id = props["gridId"]
            grid_x = props["gridX"]
            grid_y = props["gridY"]

            # Cache it
            cache.set_grid_point(station_id, grid_id, grid_x, grid_y)

            return (grid_id, grid_x, grid_y)

    except Exception as e:
        logger.warning(f"Error fetching grid point for {station_id}: {e}")
        return None


async def fetch_hourly_forecast(session: aiohttp.ClientSession,
                                grid_id: str, grid_x: int, grid_y: int) -> Optional[List[Dict]]:
    """
    Fetch gridded forecast from NWS for a grid point.
    Uses gridded endpoint to get quantitative precipitation forecast (QPF).
    Returns list of hourly forecast periods or None if failed.
    """
    url = f"{NWS_API}/gridpoints/{grid_id}/{grid_x},{grid_y}"
    try:
        await asyncio.sleep(REQUEST_DELAY)  # Rate limiting
        async with session.get(url, headers=HEADERS, timeout=15) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch forecast for {grid_id}/{grid_x},{grid_y}: HTTP {response.status}")
                return None

            data = await response.json()
            props = data["properties"]

            # Extract temperature and QPF grids
            temp_values = props.get("temperature", {}).get("values", [])
            qpf_values = props.get("quantitativePrecipitation", {}).get("values", [])

            # Build union of hourly timestamps from temp + QPF
            temp_by_time = {}
            for temp_entry in temp_values:
                valid_time_str = temp_entry.get("validTime")
                if not valid_time_str:
                    continue
                if "/" in valid_time_str:
                    start_str, dur_str = valid_time_str.split("/")
                else:
                    start_str = valid_time_str
                    dur_str = "PT1H"
                try:
                    start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                except Exception:
                    continue
                temp_c = temp_entry.get("value")
                # Fill temperature for every hour within the period duration.
                # NWS uses variable-duration periods (PT2H, PT3H, PT4H, etc.) and
                # 6-hour forecast targets can fall inside these periods. Without this,
                # QPF expansion adds hourly timestamps with temp=None that become the
                # nearest match, causing gaps in the chart.
                dur_match = re.search(r'PT(\d+)H', dur_str)
                dur_hours = int(dur_match.group(1)) if dur_match else 1
                for h_offset in range(dur_hours):
                    temp_by_time[start_time + timedelta(hours=h_offset)] = temp_c

            # Parse QPF windows
            qpf_windows = []
            for qpf_entry in qpf_values:
                qpf_time_str = qpf_entry.get("validTime")
                if not qpf_time_str or "/" not in qpf_time_str:
                    continue
                qpf_start_str, qpf_duration_str = qpf_time_str.split("/")
                try:
                    qpf_start = datetime.fromisoformat(qpf_start_str.replace("Z", "+00:00"))
                except Exception:
                    continue

                import re
                duration_match = re.search(r'PT(\d+)H', qpf_duration_str)
                if not duration_match:
                    continue
                duration_hours = int(duration_match.group(1))
                qpf_end = qpf_start + timedelta(hours=duration_hours)
                qpf_windows.append((qpf_start, qpf_end, qpf_entry.get("value") or 0, duration_hours))

            # Collect all hourly timestamps from temp and QPF windows
            all_times = set(temp_by_time.keys())
            for start, end, total_mm, dur in qpf_windows:
                t = start
                while t < end:
                    all_times.add(t)
                    t += timedelta(hours=1)

            # Build hourly forecast from union of times
            forecast = []
            for start_time in sorted(all_times)[:168]:
                temp_c = temp_by_time.get(start_time)
                temp_f = (temp_c * 9/5) + 32 if temp_c is not None else None

                # Use None when no QPF window covers this hour so that hours
                # beyond the NWS QPF issuance window (typically ~3 days for some
                # offices) render as gaps in the chart rather than false zeros.
                precip_inches = None
                for qpf_start, qpf_end, total_mm, duration_hours in qpf_windows:
                    if qpf_start <= start_time < qpf_end:
                        precip_inches = round((total_mm / duration_hours) / 25.4, 3)
                        break

                forecast.append({
                    "datetime": start_time.isoformat(),
                    "temperature": round(temp_f) if temp_f is not None else None,
                    "precipitation": precip_inches  # None = no QPF issued; 0.0 = QPF says dry
                })

            return forecast[:168]  # Limit to 7 days

    except Exception as e:
        logger.warning(f"Error fetching forecast for {grid_id}/{grid_x},{grid_y}: {e}")
        return None


async def fetch_nws_forecasts_batch(stations: List[Dict]) -> Dict[str, Optional[List[Dict]]]:
    """
    Fetch NWS forecasts for multiple ASOS stations in batch.

    Args:
        stations: List of station dicts with keys: station_id, lat, lon

    Returns:
        Dict mapping station_id -> forecast data (list of hourly periods)

    Optimizations:
        - Caches grid points permanently to avoid redundant /points calls
        - Groups stations by grid point to fetch each unique forecast only once
        - Uses async requests with rate limiting
    """
    cache = GridPointCache()
    results = {}

    # Step 1: Get grid points for all stations (using cache when possible)
    logger.info(f"Fetching grid points for {len(stations)} ASOS stations...")

    async with aiohttp.ClientSession() as session:
        # Fetch grid points for stations not in cache
        grid_point_tasks = []
        station_lookup = {}  # station_id -> station dict

        for station in stations:
            station_id = station['station_id']
            station_lookup[station_id] = station

            cached = cache.get_grid_point(station_id)
            if cached:
                # Already cached, no need to fetch
                continue

            task = fetch_grid_point(session, station['lat'], station['lon'], station_id, cache)
            grid_point_tasks.append((station_id, task))

        if grid_point_tasks:
            logger.info(f"Fetching {len(grid_point_tasks)} uncached grid points from NWS API...")
            for station_id, task in grid_point_tasks:
                await task  # Execute with rate limiting

        # Step 2: Group stations by unique grid point
        grid_to_stations = defaultdict(list)

        for station in stations:
            station_id = station['station_id']
            grid_point = cache.get_grid_point(station_id)

            if not grid_point:
                logger.warning(f"No grid point available for {station_id}, skipping")
                results[station_id] = None
                continue

            # Validate grid coordinates are not None
            if grid_point['grid_x'] is None or grid_point['grid_y'] is None:
                logger.warning(f"Invalid grid coordinates for {station_id} (grid_x={grid_point['grid_x']}, grid_y={grid_point['grid_y']}), skipping")
                results[station_id] = None
                continue

            grid_key = f"{grid_point['grid_id']}/{grid_point['grid_x']},{grid_point['grid_y']}"
            grid_to_stations[grid_key].append(station_id)

        logger.info(f"Grouped {len(stations)} stations into {len(grid_to_stations)} unique grid points")

        # Step 3: Fetch forecasts for unique grid points with limited concurrency
        logger.info(f"Fetching {len(grid_to_stations)} unique NWS forecasts...")

        # Limit concurrent requests to avoid overwhelming the API
        semaphore = asyncio.Semaphore(20)  # Max 20 concurrent requests

        async def fetch_with_semaphore(grid_id, grid_x, grid_y):
            async with semaphore:
                return await fetch_hourly_forecast(session, grid_id, grid_x, grid_y)

        # Create all forecast tasks
        forecast_tasks = []
        grid_keys = []
        for grid_key, station_ids in grid_to_stations.items():
            # Parse grid key
            grid_id, coords = grid_key.split('/')
            grid_x, grid_y = map(int, coords.split(','))

            # Create task with semaphore
            task = fetch_with_semaphore(grid_id, grid_x, grid_y)
            forecast_tasks.append(task)
            grid_keys.append(grid_key)

        # Fetch all forecasts concurrently (but limited by semaphore)
        forecasts = await asyncio.gather(*forecast_tasks, return_exceptions=True)

        # Store results
        grid_forecasts = {}
        for grid_key, forecast in zip(grid_keys, forecasts):
            if isinstance(forecast, Exception):
                logger.warning(f"Failed to fetch forecast for {grid_key}: {forecast}")
                grid_forecasts[grid_key] = None
            else:
                grid_forecasts[grid_key] = forecast

        success_count = sum(1 for f in grid_forecasts.values() if f is not None)
        logger.info(f"Completed fetching NWS forecasts: {success_count}/{len(grid_forecasts)} successful")

        # Step 4: Distribute forecasts to all stations in each grid
        for grid_key, station_ids in grid_to_stations.items():
            forecast = grid_forecasts.get(grid_key)
            for station_id in station_ids:
                results[station_id] = forecast

    success_count = sum(1 for f in results.values() if f is not None)
    logger.info(f"Successfully fetched NWS forecasts for {success_count}/{len(stations)} stations")

    return results


def fetch_nws_forecasts_batch_sync(stations: List[Dict]) -> Dict[str, Optional[List[Dict]]]:
    """
    Synchronous wrapper for async batch forecast fetching.

    Args:
        stations: List of station dicts with keys: station_id, lat, lon

    Returns:
        Dict mapping station_id -> forecast data
    """
    return asyncio.run(fetch_nws_forecasts_batch(stations))


def transform_nws_to_asos_format(
    nws_forecasts: Dict[str, Optional[List[Dict]]],
    forecast_hours: List[int],
    init_time: datetime
) -> Dict[str, Dict[str, List]]:
    """
    Transform NWS forecast data to ASOS storage format.

    Args:
        nws_forecasts: Dict mapping station_id -> list of forecast periods
        forecast_hours: List of forecast hours (e.g., [6, 12, 24, 48, ...])
        init_time: Forecast initialization time

    Returns:
        Dict mapping station_id -> {'temps': [...], 'mslps': [...], 'precips': [...]}
        where lists are aligned with forecast_hours

    Note:
        - Temperature: Point value at the forecast hour
        - Precipitation: 6-hour accumulated total ending at the forecast hour
          (to match GFS/AIFS/IFS 6-hour accumulation format)
    """
    result = {}

    for station_id, forecast in nws_forecasts.items():
        # Initialize with None values
        temps = [None] * len(forecast_hours)
        mslps = [None] * len(forecast_hours)  # NWS doesn't provide MSLP
        precips = [None] * len(forecast_hours)

        if forecast:
            # Create lookup by valid time and hour offset
            forecast_by_time = {}
            forecast_by_hour = {}

            for period in forecast:
                try:
                    valid_time = datetime.fromisoformat(period['datetime'])
                    forecast_by_time[valid_time] = period

                    # Calculate hour offset from init time
                    hour_offset = int((valid_time - init_time).total_seconds() / 3600)
                    forecast_by_hour[hour_offset] = period
                except Exception:
                    continue

            # Match forecast hours to NWS periods
            for i, hour in enumerate(forecast_hours):
                target_time = init_time + timedelta(hours=hour)

                # Temperature: Find closest NWS forecast within 30 minutes
                closest_period = None
                min_delta = timedelta(hours=1)

                for valid_time, period in forecast_by_time.items():
                    delta = abs(valid_time - target_time)
                    if delta < min_delta:
                        min_delta = delta
                        closest_period = period

                # Only use if within 30 minutes
                if closest_period and min_delta < timedelta(minutes=30):
                    temps[i] = closest_period.get('temperature')

                # Precipitation: Accumulate 6-hour total ending at this forecast hour
                # For F006, sum F001-F006; for F012, sum F007-F012, etc.
                precip_6hr = 0.0
                found_any = False

                for h in range(hour - 5, hour + 1):  # 6 hours ending at 'hour'
                    if h > 0 and h in forecast_by_hour:  # Skip h=0 (init time has no precip)
                        hourly_precip = forecast_by_hour[h].get('precipitation')
                        if hourly_precip is None:
                            continue  # Hour not covered by any QPF window — skip
                        precip_6hr += hourly_precip
                        found_any = True

                # Only store a value when at least one hour in the window has QPF data.
                # Hours entirely outside the NWS QPF issuance window stay None so the
                # chart shows a gap rather than a misleading flat zero.
                if found_any:
                    precips[i] = precip_6hr

        result[station_id] = {
            'temps': temps,
            'mslps': mslps,
            'precips': precips
        }

    return result


if __name__ == "__main__":
    # Test with a few sample stations
    logging.basicConfig(level=logging.INFO)

    test_stations = [
        {"station_id": "KIAD", "lat": 38.9445, "lon": -77.4558},  # Dulles
        {"station_id": "KDCA", "lat": 38.8521, "lon": -77.0377},  # Reagan
        {"station_id": "KBWI", "lat": 39.1754, "lon": -76.6683},  # BWI
    ]

    print("Testing NWS batch forecast fetching...")
    results = fetch_nws_forecasts_batch_sync(test_stations)

    for station_id, forecast in results.items():
        if forecast:
            print(f"{station_id}: {len(forecast)} hours of forecast data")
        else:
            print(f"{station_id}: Failed to fetch forecast")
